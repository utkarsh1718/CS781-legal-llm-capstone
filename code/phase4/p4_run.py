"""
p4_run.py -- background runner for Phase-4 (CJPE) experiments, one process per GPU:

    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 setsid nohup python p4_run.py --plan plan.json >> log 2>&1 &

plan.json = list of jobs. Job types:
  run  (default): {"name", "ckpt": "m1".."m4", "method", "mode": "last", "k": 512, "epochs", "lr",
                   "bs", "seed", "max_train", "layerwise_unfreeze", "partial_k", method kwargs...,
                   KD: "teachers": [run names], "student_layers": [...], "kd_T", "kd_lam"}
  prob:  {"type": "prob", "name": X, "train": true}  re-load run X and save P(1) for train/dev/test
  hier:  {"type": "hier", "name": N, "base": X, "epochs": 8}  hierarchical model: encode EVERY 512-token
         chunk of every case with fine-tuned run X (frozen), then train a BiGRU + attention pooler
         over the chunk sequence; also reports where in the judgment the attention goes
Outputs per run: stats/p4_runs/<name>.json (written last = done marker), p4_ckpt/<name>.pt (trained
params), p4_prob/<name>.pkl ({"dev": P(1), "test": P(1)[, "train"]}).
Model selection / thresholds use the CLEAN dev only (test cases removed); test labels are never read.
"""
import argparse
import json
import os
import pickle
import sys
import time
import traceback

import numpy as np
import torch

WORK = "/kaggle/working"
sys.path.insert(0, f"{WORK}/src")
import p4_cjpe as cj                                                  # noqa: E402
import p2_peft as pp                                                  # noqa: E402
import p3_rr as rr                                                    # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--plan", required=True)
PLAN = json.load(open(ap.parse_args().plan))
DEV = "cuda:0"
CK = {"m1": f"{WORK}/checkpoints/m1_dense_100m/best.pt", "m2": f"{WORK}/checkpoints/m2_mla_moe/best.pt",
      "m3": f"{WORK}/checkpoints/m3_encdec/best.pt", "m4": f"{WORK}/checkpoints/m4_copy/best.pt"}
OUT, PRB, CKD = f"{WORK}/stats/p4_runs", f"{WORK}/p4_prob", f"{WORK}/p4_ckpt"
for d in (OUT, PRB, CKD):
    os.makedirs(d, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


TR, DV, TE = (cj.Split(f"{WORK}/data_p4/tok/{s}.npz") for s in ("train", "dev", "test"))
log(f"data: train {TR.n} | clean dev {DV.n} | test {TE.n}")


def make_model(ckpt, method, student_layers=None, partial_k=4, **kw):
    m = cj.build(CK[ckpt], method, bidirectional=True, partial_k=partial_k, **kw)
    if student_layers:
        pp.shrink_dense(m.tagger, tuple(student_layers))
    return m


def run(name, ckpt="m1", method="lora", mode="last", k=512, epochs=1, lr=5e-4, bs=8, seed=0,
        max_train=None, layerwise_unfreeze=False, partial_k=4, teachers=None, student_layers=None,
        kd_T=2.0, kd_lam=0.5, **kw):
    t0 = time.time()
    model = make_model(ckpt, method, student_layers, partial_k, **kw)
    soft = None
    if teachers:                                                       # mean teacher P(1) on train cases
        soft = np.mean([pickle.load(open(f"{PRB}/{t}.pkl", "rb"))["train"] for t in teachers], 0)
    log(f"=== {name}: {ckpt} | {method} | {mode}-{k} | epochs {epochs} lr {lr}"
        + (f" | student layers {student_layers} | teachers {teachers}" if teachers or student_layers else ""))
    hist, best = cj.train(model, TR, DV, mode, k, epochs, lr, bs=bs, device=DEV, seed=seed,
                          max_train=max_train, layerwise_unfreeze=layerwise_unfreeze,
                          soft=soft, kd_T=kd_T, kd_lam=kd_lam, log=log)
    pd = cj.predict(model, DV, mode, k, DEV)
    torch.cuda.synchronize(); ti = time.time()
    pt = cj.predict(model, TE, mode, k, DEV)
    torch.cuda.synchronize(); infer = round(time.time() - ti, 1)
    t = cj.tune_threshold(DV.labels, pd)
    out = {"cfg": dict(ckpt=ckpt, method=method, mode=mode, k=k, epochs=epochs, lr=lr, bs=bs, seed=seed,
                       max_train=max_train, layerwise_unfreeze=layerwise_unfreeze, partial_k=partial_k,
                       teachers=teachers, student_layers=student_layers, kd_T=kd_T, kd_lam=kd_lam, **kw),
           "trainable_M": round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 3),
           "total_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
           "infer_test_s": infer, "hist": hist, "min": round((time.time() - t0) / 60, 1),
           "dev@0.5": cj.metrics(DV.labels, pd), "dev@tuned": cj.metrics(DV.labels, pd, t),
           "test_pos_rate@tuned": float((pt >= t).mean())}
    torch.save({n: p.detach().half().cpu() for n, p in model.named_parameters()
                if p.requires_grad or layerwise_unfreeze or method in ("full", "partial")}, f"{CKD}/{name}.pt")
    pickle.dump({"dev": pd, "test": pt}, open(f"{PRB}/{name}.pkl", "wb"))
    json.dump(out, open(f"{OUT}/{name}.json", "w"), indent=1)
    log(f"DONE {name}: dev macro-F1 {out['dev@0.5']['macro_f1']:.4f} @0.5 | {out['dev@tuned']['macro_f1']:.4f} "
        f"@t={t:.2f} | test pos-rate {out['test_pos_rate@tuned']:.2f} | {out['min']} min")
    del model; torch.cuda.empty_cache()


def prob_job(name, train=False):
    R = json.load(open(f"{OUT}/{name}.json")); c = {k: v for k, v in R["cfg"].items() if v is not None}
    mode, k = c.pop("mode"), c.pop("k")
    for key in ("epochs", "lr", "bs", "seed", "max_train", "layerwise_unfreeze", "teachers", "kd_T", "kd_lam"):
        c.pop(key, None)
    m = make_model(c.pop("ckpt"), c.pop("method"), c.pop("student_layers", None), c.pop("partial_k", 4), **c)
    sd = {n: v.float() for n, v in torch.load(f"{CKD}/{name}.pt", map_location="cpu").items()}
    m.load_state_dict(sd, strict=False); m.to(DEV)
    path = f"{PRB}/{name}.pkl"; out = pickle.load(open(path, "rb"))
    if train and "train" not in out:
        ti = time.time(); out["train"] = cj.predict(m, TR, mode, k, DEV)
        log(f"   {name}: train probs in {time.time()-ti:.0f}s")
    pickle.dump(out, open(path, "wb"))
    chk = cj.metrics(DV.labels, cj.predict(m, DV, mode, k, DEV))["macro_f1"]
    log(f"PROB {name}: dev re-check {chk:.4f} (should equal its @0.5 score)")
    del m; torch.cuda.empty_cache()


def _load_run(name):
    """Rebuild run `name` from its result file and load its trained weights."""
    R = json.load(open(f"{OUT}/{name}.json")); c = {k: v for k, v in R["cfg"].items() if v is not None}
    for key in ("mode", "k", "epochs", "lr", "bs", "seed", "max_train", "layerwise_unfreeze", "teachers",
                "kd_T", "kd_lam", "type", "base"):
        c.pop(key, None)
    m = make_model(c.pop("ckpt"), c.pop("method"), c.pop("student_layers", None), c.pop("partial_k", 4), **c)
    sd = {n: v.float() for n, v in torch.load(f"{CKD}/{name}.pt", map_location="cpu").items()}
    m.load_state_dict(sd, strict=False)
    return m.to(DEV)


def _pad(E, idx):
    C = max(len(E[i]) for i in idx)
    x = np.zeros((len(idx), C, E[idx[0]].shape[1]), dtype=np.float32); mk = np.zeros((len(idx), C), dtype=bool)
    for b, i in enumerate(idx):
        x[b, :len(E[i])] = E[i]; mk[b, :len(E[i])] = True
    return torch.from_numpy(x).to(DEV), torch.from_numpy(mk).to(DEV)


@torch.no_grad()
def _pool_predict(pool, E, bs=64, return_att=False):
    pool.eval(); n = len(E); p = np.zeros(n, dtype=np.float32); atts = [None] * n
    order = np.argsort([len(e) for e in E])
    for b in range(0, n, bs):
        idx = order[b:b + bs]; x, mk = _pad(E, idx)
        o, a = pool(x, mk, return_att=True)
        p[idx] = torch.softmax(o, -1)[:, 1].cpu().numpy()
        if return_att:
            for j, i in enumerate(idx):
                atts[i] = a[j, :len(E[i])].cpu().numpy()
    return (p, atts) if return_att else p


def hier_job(name, base, epochs=8, lr=1e-3, chunk=512, max_chunks=64, bs=32, seed=0):
    t0 = time.time(); torch.manual_seed(seed); np.random.seed(seed)
    cache = f"{WORK}/p4_chunks/{base}.pkl"; os.makedirs(f"{WORK}/p4_chunks", exist_ok=True)
    if os.path.exists(cache):
        E = pickle.load(open(cache, "rb")); log(f"   loaded chunk embeddings of {base}")
    else:
        m = _load_run(base); E = {}
        for s, sp in (("dev", DV), ("test", TE), ("train", TR)):
            ti = time.time(); E[s] = cj.encode_chunks(m, sp, chunk, DEV, max_chunks=max_chunks)
            log(f"   encoded {s}: {sum(len(e) for e in E[s])} chunks in {time.time()-ti:.0f}s")
        del m; torch.cuda.empty_cache(); pickle.dump(E, open(cache, "wb"))
    pool = cj.ChunkPooler(E["train"][0].shape[1]).to(DEV)
    y = TR.labels; cnt = np.bincount(y, minlength=2).astype(float)
    w = torch.tensor(cnt.sum() / (2 * cnt), dtype=torch.float, device=DEV)
    opt = torch.optim.AdamW(pool.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(TR.n / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    log(f"=== {name}: hierarchical BiGRU+attention over chunks of {base} | {sum(p.numel() for p in pool.parameters())/1e6:.2f}M params")
    best, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        pool.train(); order = np.random.permutation(TR.n); tot = 0.0
        for b in range(0, TR.n, bs):
            idx = order[b:b + bs]; x, mk = _pad(E["train"], idx)
            loss = torch.nn.functional.cross_entropy(pool(x, mk), torch.tensor(y[idx], device=DEV), weight=w)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(pool.parameters(), 1.0); opt.step(); sched.step(); tot += loss.item()
        pd = _pool_predict(pool, E["dev"]); m05 = cj.metrics(DV.labels, pd)
        mt = cj.metrics(DV.labels, pd, cj.tune_threshold(DV.labels, pd))
        hist.append({"epoch": ep + 1, "f1@0.5": m05["macro_f1"], "f1@tuned": mt["macro_f1"]})
        log(f"   epoch {ep+1}: loss {tot/np.ceil(TR.n/bs):.4f} | dev macro-F1 {m05['macro_f1']:.4f} @0.5 | {mt['macro_f1']:.4f} @t={mt['t']:.2f}")
        if m05["macro_f1"] > best:
            best = m05["macro_f1"]; best_state = {k: v.detach().clone() for k, v in pool.state_dict().items()}
    pool.load_state_dict(best_state)
    pd, att = _pool_predict(pool, E["dev"], return_att=True); pt = _pool_predict(pool, E["test"])
    t = cj.tune_threshold(DV.labels, pd)
    prof, n_docs = np.zeros(10), 0                                    # mean attention mass per position decile
    for a in att:
        if len(a) >= 3:                                               # (cases with >= 3 chunks)
            pos = np.minimum(9, (np.arange(len(a)) / len(a) * 10).astype(int))
            np.add.at(prof, pos, a); n_docs += 1
    out = {"cfg": {"type": "hier", "base": base, "epochs": epochs, "lr": lr, "chunk": chunk, "max_chunks": max_chunks},
           "trainable_M": round(sum(p.numel() for p in pool.parameters()) / 1e6, 3), "total_M": 0.0,
           "infer_test_s": 0.0, "hist": hist, "min": round((time.time() - t0) / 60, 1),
           "dev@0.5": cj.metrics(DV.labels, pd), "dev@tuned": cj.metrics(DV.labels, pd, t),
           "test_pos_rate@tuned": float((pt >= t).mean()),
           "attention_by_position_decile": [round(float(v), 4) for v in prof / max(1, n_docs)]}
    torch.save(pool.state_dict(), f"{CKD}/{name}.pt")
    pickle.dump({"dev": pd, "test": pt}, open(f"{PRB}/{name}.pkl", "wb"))
    json.dump(out, open(f"{OUT}/{name}.json", "w"), indent=1)
    log(f"DONE {name}: dev macro-F1 {out['dev@0.5']['macro_f1']:.4f} @0.5 | {out['dev@tuned']['macro_f1']:.4f} @t={t:.2f} "
        f"| attention by decile {out['attention_by_position_decile']} | {out['min']} min")


for job in PLAN:
    job = dict(job); name = job.pop("name"); typ = job.pop("type", "run")
    try:
        if typ == "prob":
            prob_job(name, job.get("train", False))
        elif typ == "hier":
            if os.path.exists(f"{OUT}/{name}.json"):
                log(f"skip {name} (already done)")
            else:
                hier_job(name, **job)
        elif os.path.exists(f"{OUT}/{name}.json"):
            log(f"skip {name} (already done)")
        else:
            run(name, **job)
    except Exception:
        log(f"!!! {name} FAILED"); traceback.print_exc(); torch.cuda.empty_cache()
log("PLAN DONE")
