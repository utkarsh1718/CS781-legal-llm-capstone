"""
p5_td.py -- Phase-5 final models retrained on TRAIN + CLEAN DEV (background, one process per GPU):

    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 setsid nohup python p5_td.py --plan plan.json >> log 2>&1 &
    python p5_td.py --smoke        # ~2 min check of every code path on a few cases (run first)

Jobs:
  {"type": "td", "name": X}        X's exact config (stats/p5_runs/X.json) trained on train + clean dev for
                                   X's full epoch count (no held-out data -> last step kept)  -> X_td
  {"type": "hier_td", "name": H}   hierarchical model H rebuilt on the chunk embeddings of <H's base>_td,
                                   trained on train + dev for H's best epoch, same seed      -> H_td
Outputs: p5_ckpt/<n>_td.pt, p5_prob/<n>_td.pkl ({"test"} only: these models have seen dev),
stats/p5_runs/<n>_td.json (written last = done marker). Test labels do not exist on disk.
"""
import argparse
import json
import math
import os
import pickle
import random
import re
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

WORK = "/kaggle/working"
sys.path.insert(0, f"{WORK}/src")
import p5_lsi as ls                                                   # noqa: E402
import p4_cjpe as cj                                                  # noqa: E402
import p2_peft as pp                                                  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--plan"); ap.add_argument("--smoke", action="store_true")
ARGS = ap.parse_args()
DEV = "cuda:0"
CK = {"m1": f"{WORK}/checkpoints/m1_dense_100m/best.pt", "m2": f"{WORK}/checkpoints/m2_mla_moe/best.pt",
      "m3": f"{WORK}/checkpoints/m3_encdec/best.pt", "m4": f"{WORK}/checkpoints/m4_copy/best.pt"}
OUT, PRB, CKD, CHK = (f"{WORK}/stats/p5_runs", f"{WORK}/p5_prob", f"{WORK}/p5_ckpt", f"{WORK}/p5_chunks")
for d in (OUT, PRB, CKD, CHK):
    os.makedirs(d, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Cat:
    """Several splits glued into one (tokens, offsets, multi-hot labels)."""
    def __init__(self, *parts):
        self.ids = np.concatenate([p.ids[:p.off[p.n]] for p in parts])
        offs, base = [np.zeros(1, dtype=np.int64)], 0
        for p in parts:
            offs.append(base + p.off[1:p.n + 1].astype(np.int64)); base += int(p.off[p.n])
        self.off = np.concatenate(offs)
        self.Y = np.vstack([p.Y[:p.n] for p in parts]) if all(p.Y is not None for p in parts) else None
        self.labels, self.n = self.Y, sum(p.n for p in parts)

    def arr(self, i):
        return self.ids[self.off[i]:self.off[i + 1]]


class Head:
    """First n cases of a split (smoke test only)."""
    def __init__(self, s, n):
        self.ids, self.off, self.n = s.ids, s.off, n
        self.Y = None if s.Y is None else s.Y[:n]

    def arr(self, i):
        return self.ids[self.off[i]:self.off[i + 1]]


TR, DV, TE = (ls.MSplit(f"{WORK}/data_p5/tok/{s}.npz", f"{WORK}/data_p5/clean/{s}.json")
              for s in ("train", "dev", "test"))
TD = Cat(TR, DV)
log(f"data: train {TR.n} + clean dev {DV.n} = {TD.n} cases | test {TE.n}")


def make_model(ckpt, method, student_layers=None, partial_k=4, **kw):
    m = ls.build(CK[ckpt], method, bidirectional=True, partial_k=partial_k, **kw)
    if student_layers:
        pp.shrink_dense(m.tagger, tuple(student_layers))
    return m


def split_cfg(cfg):
    """run config -> (model kwargs, training kwargs)"""
    c = {k: v for k, v in cfg.items() if v is not None}
    tr = {k: c.pop(k) for k in ("mode", "k", "epochs", "lr", "head_lr", "bs", "seed", "layerwise_unfreeze") if k in c}
    for k in ("teachers", "kd_T", "kd_lam"):
        c.pop(k, None)
    return c, tr


def train_td(model, tr, mode="first", k=1024, epochs=2, lr=3e-5, head_lr=1e-3, bs=8, seed=0, warmup=0.03,
             layerwise_unfreeze=False, max_steps=None):
    """Same optimiser / loss / schedule as p5_lsi.train, but no evaluation: the last step is kept."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model.to(DEV)
    pos = tr.Y.sum(0); pw = torch.tensor(np.sqrt((tr.n - pos) / np.maximum(pos, 1)), dtype=torch.float, device=DEV)
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
    scaler = torch.amp.GradScaler("cuda")
    step, t0, run = 0, time.time(), 0.0
    log(f"trainable {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.3f}M | {tr.n} cases | "
        f"{steps} steps | view {mode}-{k} | no dev (train+dev)")
    for ep in range(epochs):
        order = np.random.permutation(tr.n)
        for b in range(0, tr.n, bs):
            if layerwise_unfreeze:
                n_on = min(len(layers), 1 + int(step / max(1, steps) * len(layers)))
                for L in layers[len(layers) - n_on:]:
                    for p in L.parameters():
                        p.requires_grad = True
            model.train(); bi = order[b:b + bs]
            ids, mask = cj.collate([cj.section(tr.arr(i), mode, k) for i in bi], DEV)
            with cj._autocast(DEV):
                z = model(ids, mask).float()
            loss = F.binary_cross_entropy_with_logits(z, torch.from_numpy(tr.Y[bi]).to(DEV), pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1; run += loss.item()
            if step % 500 == 0:
                log(f"   step {step}/{steps} | loss {run/500:.4f} | {time.time()-t0:.0f}s"); run = 0.0
            if max_steps and step >= max_steps:
                return
    return


def td_job(base, smoke=False):
    t0, name = time.time(), f"{base}_td"
    R = json.load(open(f"{OUT}/{base}.json")); mk, tk = split_cfg(R["cfg"])
    method = mk["method"]
    model = make_model(mk.pop("ckpt"), mk.pop("method"), mk.pop("student_layers", None), mk.pop("partial_k", 4), **mk)
    log(f"=== {name}: {base}'s config on train + clean dev | {tk}")
    tr, te = (Cat(Head(TR, 12), Head(DV, 4)), Head(TE, 6)) if smoke else (TD, TE)
    train_td(model, tr, **tk, max_steps=3 if smoke else None)
    pt = ls.predict(model, te, tk.get("mode", "first"), tk.get("k", 1024), DEV)
    if smoke:
        log("td ok: test probs", pt.shape); return model
    torch.save({n: p.detach().half().cpu() for n, p in model.named_parameters()
                if p.requires_grad or tk.get("layerwise_unfreeze") or method in ("full", "partial")}, f"{CKD}/{name}.pt")
    pickle.dump({"test": pt.astype(np.float16)}, open(f"{PRB}/{name}.pkl", "wb"))
    json.dump({"cfg": R["cfg"], "base": base, "td": True, "min": round((time.time() - t0) / 60, 1),
               "test_labels_per_case@0.5": float(ls.decide(pt, 0.5).sum(1).mean())},
              open(f"{OUT}/{name}.json", "w"), indent=1)
    log(f"DONE {name} | {round((time.time() - t0) / 60, 1)} min")
    del model; torch.cuda.empty_cache()


def _load_td_base(base_td):
    R = json.load(open(f"{OUT}/{base_td[:-3]}.json")); mk, tk = split_cfg(R["cfg"])
    m = make_model(mk.pop("ckpt"), mk.pop("method"), mk.pop("student_layers", None), mk.pop("partial_k", 4), **mk)
    sd = {n: v.float() for n, v in torch.load(f"{CKD}/{base_td}.pt", map_location="cpu").items()}
    m.load_state_dict(sd, strict=False)
    return m.to(DEV)


def _pad(E, idx):
    C = max(len(E[i]) for i in idx)
    x = np.zeros((len(idx), C, E[idx[0]].shape[1]), dtype=np.float32); mk = np.zeros((len(idx), C), dtype=bool)
    for b, i in enumerate(idx):
        x[b, :len(E[i])] = E[i]; mk[b, :len(E[i])] = True
    return torch.from_numpy(x).to(DEV), torch.from_numpy(mk).to(DEV)


@torch.no_grad()
def _pool_predict(pool, E, bs=64):
    pool.eval(); p = np.zeros((len(E), ls.NL), dtype=np.float32)
    order = np.argsort([len(e) for e in E])
    for b in range(0, len(E), bs):
        idx = order[b:b + bs]; x, mk = _pad(E, idx)
        p[idx] = torch.sigmoid(pool(x, mk)).cpu().numpy()
    return p


def hier_td_job(hname, smoke_model=None):
    t0, name = time.time(), f"{hname}_td"
    if smoke_model is not None:                                       # smoke: 6 train+dev cases, 3 test cases
        tr, te, m = Cat(Head(TR, 4), Head(DV, 2)), Head(TE, 3), smoke_model
        E = {"td": cj.encode_chunks(m, tr, 1024, DEV, max_chunks=24), "test": cj.encode_chunks(m, te, 1024, DEV, max_chunks=24)}
        cfg, best_ep, seed, Y = {"lr": 1e-3}, 1, 0, tr.Y
    else:
        R = json.load(open(f"{OUT}/{hname}.json")); cfg = R["cfg"]
        best_ep = int(max(R["hist"], key=lambda h: h["macro_f1"])["epoch"])
        seed = int(re.search(r"_s(\d+)$", hname).group(1)) if re.search(r"_s(\d+)$", hname) else 0
        base_td = cfg["base"] + "_td"; cache = f"{CHK}/{base_td}.pkl"
        if os.path.exists(cache):
            E = pickle.load(open(cache, "rb")); log(f"   loaded chunk embeddings of {base_td}")
        else:
            m = _load_td_base(base_td); E = {}
            for s, sp in (("test", TE), ("td", TD)):
                ti = time.time(); E[s] = cj.encode_chunks(m, sp, cfg["chunk"], DEV, max_chunks=cfg["max_chunks"])
                log(f"   encoded {s}: {sum(len(e) for e in E[s])} chunks in {time.time()-ti:.0f}s")
            del m; torch.cuda.empty_cache(); pickle.dump(E, open(cache, "wb"))
        Y = TD.Y
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(E["td"]); pool = ls.ChunkPoolerML(E["td"][0].shape[1]).to(DEV)
    pos = Y.sum(0); pw = torch.tensor(np.sqrt((n - pos) / np.maximum(pos, 1)), dtype=torch.float, device=DEV)
    bs = 32; steps = best_ep * int(np.ceil(n / bs))
    opt = torch.optim.AdamW(pool.parameters(), lr=cfg["lr"], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)
    log(f"=== {name}: pooler on train+dev chunk embeddings | {best_ep} epochs | seed {seed}")
    for ep in range(best_ep):
        pool.train(); order = np.random.permutation(n)
        for b in range(0, n, bs):
            idx = order[b:b + bs]; x, mk = _pad(E["td"], idx)
            loss = F.binary_cross_entropy_with_logits(pool(x, mk), torch.from_numpy(Y[idx]).to(DEV), pos_weight=pw)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(pool.parameters(), 1.0); opt.step(); sched.step()
    pt = _pool_predict(pool, E["test"])
    if smoke_model is not None:
        log("hier_td ok: test probs", pt.shape); return
    torch.save(pool.state_dict(), f"{CKD}/{name}.pt")
    pickle.dump({"test": pt.astype(np.float16)}, open(f"{PRB}/{name}.pkl", "wb"))
    json.dump({"cfg": {**cfg, "type": "hier_td", "ref": hname}, "td": True, "epochs_trained": best_ep, "seed": seed,
               "min": round((time.time() - t0) / 60, 1)}, open(f"{OUT}/{name}.json", "w"), indent=1)
    log(f"DONE {name} | {round((time.time() - t0) / 60, 1)} min")


if ARGS.smoke:
    m = td_job("m1_partial4_first1024", smoke=True)
    hier_td_job("smoke", smoke_model=m)
    log("SMOKE TEST PASSED"); sys.exit(0)

for job in json.load(open(ARGS.plan)):
    name, typ = job["name"], job["type"]
    if os.path.exists(f"{OUT}/{name}_td.json"):
        log(f"skip {name}_td (already done)"); continue
    try:
        td_job(name) if typ == "td" else hier_td_job(name)
    except Exception:
        log(f"!!! {name}_td FAILED"); traceback.print_exc(); torch.cuda.empty_cache()
log("PLAN DONE")
