"""
p5_lsi.py -- Phase 5 (Legal Statute Identification, multi-label over 100 statute IDs).

Reuses the Phase-4 machinery (token cache, section views, DocClassifier, chunk encoder) with a
multi-label head:
  * 100 sigmoid outputs, BCE with per-label pos_weight = sqrt(#neg / #pos)  (rare statutes count more)
  * optional KD: BCE between the student's and the teacher's tempered sigmoids (x T^2)
  * decision rules: one global threshold, or one threshold per statute (tuned on dev), always at
    least one label per case (argmax fallback)
  * metrics: macro-F1 over the 100 statutes (primary), micro-F1, subset accuracy
Data: data_p5/tok/{split}.npz (tokens) + data_p5/clean/{split}.json ({id, labels}); test labels never stored.
"""
from __future__ import annotations

import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import p2_peft as pp
import p4_cjpe as cj

NL = 100


class MSplit(cj.Split):
    """Tokens + multi-hot labels (labels = None for test)."""
    def __init__(self, npz, meta_json):
        z = np.load(npz)
        self.ids, self.off = z["ids"], z["offsets"]
        meta = json.load(open(meta_json))
        self.case_ids = [m["id"] for m in meta]
        self.n = len(meta)
        self.Y = None
        if meta[0]["labels"] is not None:
            self.Y = np.zeros((self.n, NL), dtype=np.float32)
            for i, m in enumerate(meta):
                self.Y[i, m["labels"]] = 1.0
        self.labels = self.Y                                         # (so cj helpers can be reused)


def build(ckpt, method="lora", bidirectional=True, partial_k=4, **kw):
    m = cj.build(ckpt, method, bidirectional, partial_k, **kw)      # DocClassifier with a 2-way head ...
    d = m.tagger.backbone.cfg.d_model
    head = nn.Linear(d, NL); nn.init.normal_(head.weight, 0.0, 0.02); nn.init.zeros_(head.bias)
    if isinstance(m.tagger.head, pp.DropPrefix):                     # ... replaced by a 100-way head
        m.tagger.head.head = head
    else:
        m.tagger.head = head
    return m


# ----------------------------------------------------------------------------- inference + metrics
@torch.no_grad()
def predict(model, split, mode="first", k=512, device="cuda", bs=32):
    """P(statute) for every case: (n, 100) float32."""
    model.eval(); out = np.zeros((split.n, NL), dtype=np.float32)
    order = np.argsort([min(k, split.off[i + 1] - split.off[i]) for i in range(split.n)])
    for b in range(0, split.n, bs):
        idx = order[b:b + bs]
        ids, mask = cj.collate([cj.section(split.arr(i), mode, k) for i in idx], device)
        with cj._autocast(device):
            out[idx] = torch.sigmoid(model(ids, mask).float()).cpu().numpy()
    return out


def decide(P, t):
    """Multi-hot decisions; t = scalar or (100,) thresholds; at least one label per case."""
    D = (P >= t).astype(np.int8)
    empty = D.sum(1) == 0
    D[empty, P[empty].argmax(1)] = 1
    return D


def metrics(Y, P, t=0.5):
    from sklearn.metrics import f1_score
    D = decide(P, t)
    return {"macro_f1": float(f1_score(Y, D, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(Y, D, average="micro", zero_division=0)),
            "subset_acc": float((D == Y).all(1).mean()), "labels_per_case": float(D.sum(1).mean())}


def tune_global(Y, P):
    ts = np.linspace(0.05, 0.9, 35)
    f = [metrics(Y, P, t)["macro_f1"] for t in ts]
    return float(ts[int(np.argmax(f))])


def tune_per_label(Y, P, grid=np.linspace(0.05, 0.9, 35)):
    """Best threshold for each statute separately (maximises that statute's F1 on dev)."""
    T = np.zeros(NL)
    for j in range(NL):
        y, p = Y[:, j], P[:, j]
        best, bt = -1, 0.5
        for t in grid:
            d = p >= t; tp = (d & (y == 1)).sum(); fp = (d & (y == 0)).sum(); fn = (~d & (y == 1)).sum()
            f = 2 * tp / max(1, 2 * tp + fp + fn)
            if f > best:
                best, bt = f, t
        T[j] = bt
    return T


def crossfit_per_label(Y, P, seed=0):
    """Honest estimate of per-label thresholding: tune on one half of dev, score on the other, swap."""
    idx = np.random.RandomState(seed).permutation(len(Y)); a, b = idx[: len(Y) // 2], idx[len(Y) // 2:]
    D = np.zeros_like(Y, dtype=np.int8)
    D[b] = decide(P[b], tune_per_label(Y[a], P[a])); D[a] = decide(P[a], tune_per_label(Y[b], P[b]))
    from sklearn.metrics import f1_score
    return float(f1_score(Y, D, average="macro", zero_division=0))


# ----------------------------------------------------------------------------- training
def train(model, tr, dev, mode="first", k=512, epochs=1, lr=5e-4, head_lr=1e-3, bs=8, device="cuda",
          seed=0, warmup=0.03, evals_per_epoch=2, soft=None, kd_T=2.0, kd_lam=0.5,
          layerwise_unfreeze=False, log=print):
    """BCE with pos_weight (+ optional KD on teacher probabilities `soft`), AdamW, warm-up + linear decay,
    fp16. Keeps the evaluation with the best dev macro-F1 at the tuned GLOBAL threshold."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model.to(device)
    pos = tr.Y.sum(0); pw = torch.tensor(np.sqrt((tr.n - pos) / np.maximum(pos, 1)), dtype=torch.float, device=device)
    layers = list(pp._layers(model.tagger))
    if layerwise_unfreeze:
        for p in model.tagger.backbone.parameters():
            p.requires_grad = False
    head_ids = {id(p) for p in model.head().parameters()}
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in head_ids
                                         and (p.requires_grad or layerwise_unfreeze)], "lr": lr},
                             {"params": list(model.head().parameters()), "lr": head_lr}], weight_decay=0.01)
    steps = epochs * math.ceil(tr.n / bs); warm = max(1, int(warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    eval_at = {int(steps * (j + 1) / (epochs * evals_per_epoch)) for j in range(epochs * evals_per_epoch)}
    best, best_state, hist, step, t0, run = -1.0, None, [], 0, time.time(), 0.0
    log(f"trainable {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.3f}M | {tr.n} cases | "
        f"{steps} steps | view {mode}-{k}" + (" | KD" if soft is not None else "")
        + (" | gradual unfreezing" if layerwise_unfreeze else ""))
    for ep in range(epochs):
        order = np.random.permutation(tr.n)
        for b in range(0, tr.n, bs):
            if layerwise_unfreeze:
                n_on = min(len(layers), 1 + int(step / max(1, steps) * len(layers)))
                for L in layers[len(layers) - n_on:]:
                    for p in L.parameters():
                        p.requires_grad = True
            model.train(); bi = order[b:b + bs]
            ids, mask = cj.collate([cj.section(tr.arr(i), mode, k) for i in bi], device)
            y = torch.from_numpy(tr.Y[bi]).to(device)
            with cj._autocast(device):
                z = model(ids, mask).float()
            loss = F.binary_cross_entropy_with_logits(z, y, pos_weight=pw)
            if soft is not None:                                      # KD on tempered sigmoids
                tp = torch.from_numpy(soft[bi]).to(device).float().clamp(1e-4, 1 - 1e-4)
                tz = torch.log(tp) - torch.log1p(-tp)                 # teacher logits
                kd = F.binary_cross_entropy_with_logits(z / kd_T, torch.sigmoid(tz / kd_T)) * kd_T * kd_T
                loss = (1 - kd_lam) * loss + kd_lam * kd
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1; run += loss.item()
            if step % 500 == 0:
                log(f"   step {step}/{steps} | loss {run/500:.4f} | {time.time()-t0:.0f}s"); run = 0.0
            if step in eval_at:
                P = predict(model, dev, mode, k, device); t = tune_global(dev.Y, P); m = metrics(dev.Y, P, t)
                hist.append({"step": step, "macro_f1": m["macro_f1"], "micro_f1": m["micro_f1"], "t": t})
                log(f"   EVAL step {step}: dev macro-F1 {m['macro_f1']:.4f} | micro {m['micro_f1']:.4f} @t={t:.2f} "
                    f"| {m['labels_per_case']:.1f} labels/case | {time.time()-t0:.0f}s")
                if m["macro_f1"] > best:
                    best = m["macro_f1"]
                    best_state = {n: q.detach().cpu().clone() for n, q in model.named_parameters()
                                  if q.requires_grad or layerwise_unfreeze}
    model.load_state_dict({n: v.to(device) for n, v in best_state.items()}, strict=False)
    return hist, best


class ChunkPoolerML(cj.ChunkPooler):
    """Hierarchical multi-label head: BiGRU + attention over chunk embeddings -> 100 logits."""
    def __init__(self, d=768, h=256, dropout=0.2):
        super().__init__(d, h, dropout)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * h, NL))
