"""
p4_cjpe.py -- Phase 4 (Court Judgment Prediction, binary) on top of ANY Phase-1 backbone.

Data comes from the token cache written by P4-1 (data_p4/tok/{split}.npz: all tokens of every case,
uint16, plus offsets + labels; test labels are stored as -1 and never used).

Input views of a long case (our A1-BPE tokens):
  section(arr, "last",  k)  last k tokens           (the decision itself was removed from the texts,
  section(arr, "first", k)  first k tokens           so the reasoning at the end is expected to matter most)
  section(arr, "middle",k)  k tokens around the centre
  section(arr, "headtail",k) first k/4 + last 3k/4
DocClassifier: backbone (bidirectional) -> masked mean pool -> Linear(d, 2).
Any adaptation method from p3_rr.apply_method_p3 / p2_peft.apply_method:
  full, partial (top-k layers), normgain (BitFit analogue), lora, rslora, lorma_plus, lorma_pi, ia3,
  adapter, prompt.
Class-weighted CE (train is 59/41, dev/test are 50/50). Decision threshold reported at 0.5 and tuned on dev.
"""
from __future__ import annotations

import contextlib
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import p2_model as pm
import p2_peft as pp
import p3_rr as rr

PAD_ID = 0


# ----------------------------------------------------------------------------- data
class Split:
    """All tokens of one split: arr(i) -> np.uint16 token ids of case i."""
    def __init__(self, path):
        z = np.load(path)
        self.ids, self.off, self.labels = z["ids"], z["offsets"], z["labels"]
        self.n = len(self.labels)

    def arr(self, i):
        return self.ids[self.off[i]:self.off[i + 1]]


def section(a, mode="last", k=512):
    n = len(a)
    if n <= k:
        return a
    if mode == "last":
        return a[n - k:]
    if mode == "first":
        return a[:k]
    if mode == "middle":
        s = (n - k) // 2
        return a[s:s + k]
    if mode == "headtail":
        h = k // 4
        return np.concatenate([a[:h], a[n - (k - h):]])
    raise ValueError(mode)


def collate(seqs, device):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.bool)
    for b, s in enumerate(seqs):
        ids[b, :len(s)] = torch.from_numpy(s.astype(np.int64)); mask[b, :len(s)] = True
    return ids.to(device), mask.to(device)


def _autocast(device):
    return (torch.autocast("cuda", dtype=torch.float16) if str(device).startswith("cuda")
            else contextlib.nullcontext())


# ----------------------------------------------------------------------------- model
class DocClassifier(nn.Module):
    def __init__(self, tagger, dropout=0.1):
        super().__init__()
        self.tagger = tagger
        self.drop = nn.Dropout(dropout)

    def embed(self, ids, mask):
        """Masked mean of the final hidden states, (B, d) in fp32."""
        tg = self.tagger
        ctx = pm.full_attention() if (tg.bidirectional and tg.arch != "encdec") else contextlib.nullcontext()
        with ctx:
            h = tg.hidden(ids)
        if isinstance(tg.head, pp.DropPrefix):                         # prompt tuning: drop soft prompt
            h = h[:, tg.head.n:]
        h = h.float(); m = mask.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1)

    def head(self):
        h = self.tagger.head
        return h.head if isinstance(h, pp.DropPrefix) else h

    def forward(self, ids, mask):
        return self.head()(self.drop(self.embed(ids, mask)))           # (B, 2)


def build(ckpt, method="lora", bidirectional=True, partial_k=4, **kw):
    tg = pm.build_tagger(ckpt, bidirectional=bidirectional)
    tg.head = nn.Linear(tg.backbone.cfg.d_model, 2)
    nn.init.normal_(tg.head.weight, 0.0, 0.02); nn.init.zeros_(tg.head.bias)
    if method == "partial":
        pp.apply_method(tg, "partial", top_k=partial_k)
    else:
        rr.apply_method_p3(tg, method, **kw)
    for p in tg.head.parameters():
        p.requires_grad = True
    return DocClassifier(tg)


# ----------------------------------------------------------------------------- inference + metrics
@torch.no_grad()
def predict(model, split, mode="last", k=512, device="cuda", bs=32):
    """P(label = 1) for every case of the split."""
    model.eval(); out = np.zeros(split.n, dtype=np.float32)
    order = np.argsort([min(k, split.off[i + 1] - split.off[i]) for i in range(split.n)])
    for b in range(0, split.n, bs):
        idx = order[b:b + bs]
        ids, mask = collate([section(split.arr(i), mode, k) for i in idx], device)
        with _autocast(device):
            p = torch.softmax(model(ids, mask).float(), -1)[:, 1]
        out[idx] = p.cpu().numpy()
    return out


def metrics(y, p, t=0.5):
    from sklearn.metrics import f1_score, precision_score, recall_score
    yh = (p >= t).astype(int)
    return {"macro_f1": float(f1_score(y, yh, average="macro")),
            "macro_p": float(precision_score(y, yh, average="macro", zero_division=0)),
            "macro_r": float(recall_score(y, yh, average="macro", zero_division=0)),
            "acc": float((y == yh).mean()), "pos_rate": float(yh.mean()), "t": float(t)}


def tune_threshold(y, p):
    ts = np.linspace(0.05, 0.95, 91)
    f = [metrics(y, p, t)["macro_f1"] for t in ts]
    return float(ts[int(np.argmax(f))])


# ----------------------------------------------------------------------------- training
def train(model, tr, dev, mode="last", k=512, epochs=1, lr=5e-4, head_lr=1e-3, bs=8, device="cuda",
          seed=0, warmup=0.03, evals_per_epoch=2, max_train=None, layerwise_unfreeze=False,
          soft=None, kd_T=2.0, kd_lam=0.5, log=print):
    """Class-weighted CE (+ optional KD on teacher probs `soft`), AdamW, warm-up + linear decay, fp16.
    Evaluates on dev `evals_per_epoch` times per epoch, keeps the best macro-F1 (threshold 0.5).
    layerwise_unfreeze: gradual unfreezing (top layer first, one more layer every 1/L of training)."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model.to(device)
    idx_all = np.arange(tr.n)
    if max_train:
        idx_all = np.random.RandomState(seed).permutation(tr.n)[:max_train]
    y = tr.labels
    cnt = np.bincount(y[idx_all], minlength=2).astype(float)
    w = torch.tensor(cnt.sum() / (2 * cnt), dtype=torch.float, device=device)   # balanced class weights
    layers = list(pp._layers(model.tagger))
    if layerwise_unfreeze:                                            # start: only head trainable
        for p in model.tagger.backbone.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad or layerwise_unfreeze]
    head_ids = {id(p) for p in model.head().parameters()}
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in head_ids
                                         and (p.requires_grad or layerwise_unfreeze)], "lr": lr},
                             {"params": list(model.head().parameters()), "lr": head_lr}], weight_decay=0.01)
    steps = epochs * math.ceil(len(idx_all) / bs); warm = max(1, int(warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    eval_at = {int(steps * (j + 1) / (epochs * evals_per_epoch)) for j in range(epochs * evals_per_epoch)}
    best, best_state, hist, step, t0, run_loss = -1.0, None, [], 0, time.time(), 0.0
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"trainable {n_tr/1e6:.3f}M | {len(idx_all)} cases | {steps} steps | view {mode}-{k}"
        + (" | gradual unfreezing" if layerwise_unfreeze else "") + (" | KD" if soft is not None else ""))
    for ep in range(epochs):
        order = np.random.permutation(idx_all)
        for b in range(0, len(order), bs):
            if layerwise_unfreeze:                                    # unfreeze one more top layer on schedule
                n_on = min(len(layers), 1 + int(step / max(1, steps) * len(layers)))
                for L in layers[len(layers) - n_on:]:
                    for p in L.parameters():
                        p.requires_grad = True
            model.train()
            bi = order[b:b + bs]
            ids, mask = collate([section(tr.arr(i), mode, k) for i in bi], device)
            lab = torch.tensor(y[bi], dtype=torch.long, device=device)
            with _autocast(device):
                logits = model(ids, mask).float()
            loss = F.cross_entropy(logits, lab, weight=w)
            if soft is not None:                                      # KD on teacher P(1)
                tp = torch.tensor(soft[bi], dtype=torch.float, device=device).clamp(1e-4, 1 - 1e-4)
                t_log = torch.stack([torch.log1p(-tp), torch.log(tp)], -1)
                kl = F.kl_div(F.log_softmax(logits / kd_T, -1), F.log_softmax(t_log / kd_T, -1),
                              log_target=True, reduction="batchmean") * kd_T * kd_T
                loss = (1 - kd_lam) * loss + kd_lam * kl
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1; run_loss += loss.item()
            if step % 500 == 0:
                log(f"   step {step}/{steps} | loss {run_loss/500:.4f} | {time.time()-t0:.0f}s"); run_loss = 0.0
            if step in eval_at:
                p = predict(model, dev, mode, k, device)
                m = metrics(dev.labels, p); m_t = metrics(dev.labels, p, tune_threshold(dev.labels, p))
                hist.append({"step": step, "f1@0.5": m["macro_f1"], "f1@tuned": m_t["macro_f1"], "t": m_t["t"],
                             "pos_rate": m["pos_rate"]})
                log(f"   EVAL step {step}: dev macro-F1 {m['macro_f1']:.4f} @0.5 (pos rate {m['pos_rate']:.2f}) | "
                    f"{m_t['macro_f1']:.4f} @t={m_t['t']:.2f} | {time.time()-t0:.0f}s")
                if m["macro_f1"] > best:
                    best = m["macro_f1"]
                    best_state = {n: q.detach().cpu().clone() for n, q in model.named_parameters()
                                  if q.requires_grad or layerwise_unfreeze}   # gradual unfreezing: keep all
    model.load_state_dict({k_: v.to(device) for k_, v in best_state.items()}, strict=False)
    return hist, best


# ----------------------------------------------------------------------------- hierarchical (chunk) view
@torch.no_grad()
def encode_chunks(model, split, chunk=512, device="cuda", bs=32, max_chunks=64):
    """Mean-pooled embedding of EVERY chunk of every case (the last `max_chunks` chunks for very long
    cases). Returns list of (n_chunks, d) float16 arrays."""
    model.eval()
    items = []                                                         # (case, chunk index, start)
    for i in range(split.n):
        n = split.off[i + 1] - split.off[i]
        starts = list(range(0, max(1, n), chunk))[-max_chunks:]
        items += [(i, j, s) for j, s in enumerate(starts)]
    out = [None] * split.n
    tmp = {}
    for b in range(0, len(items), bs):
        batch = items[b:b + bs]
        ids, mask = collate([split.arr(i)[s:s + chunk] for i, _, s in batch], device)
        with _autocast(device):
            e = model.embed(ids, mask).half().cpu().numpy()
        for (i, j, _), v in zip(batch, e):
            tmp.setdefault(i, []).append((j, v))
    for i in range(split.n):
        out[i] = np.stack([v for _, v in sorted(tmp[i], key=lambda x: x[0])])
    return out


class ChunkPooler(nn.Module):
    """Hierarchical head over a case's chunk embeddings: BiGRU + attention pooling -> 2 logits.
    The attention weights tell which parts of the judgment drive the prediction."""
    def __init__(self, d=768, h=256, dropout=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout))
        self.gru = nn.GRU(d, h, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * h, 1)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * h, 2))

    def forward(self, x, mask, return_att=False):                      # x (B, C, d), mask (B, C)
        z, _ = self.gru(self.inp(x))
        a = self.att(z).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1)
        o = self.out((a.unsqueeze(-1) * z).sum(1))
        return (o, a) if return_att else o
