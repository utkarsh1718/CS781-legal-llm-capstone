"""
p3_run.py -- run a list of Phase-3 experiments as a BACKGROUND process (survives browser reloads,
cell interrupts and kernel restarts). Launch one process per GPU:

    CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 setsid nohup python p3_run.py --plan plan.json > log 2>&1 &

plan.json = [{"name": "m1_rslora", "ckpt": "m1", "method": "rslora", "lr": 2e-4, "rs_r": 16}, ...]
Optional per-run keys: ctx ("doc"/"none"), W, use_pos, bidir, epochs, bs, seed + method kwargs.

Per finished run it writes
    stats/p3_runs/<name>.json   config, trainable params, per-epoch history, dev metrics at lambda 0/0.5/1
    p3_ckpt/<name>.pt           trained parameters only (PEFT: a few MB)
    p3_lp/<name>.pkl            dev + test log-probs per document (for ensembling / KD / submissions)
Runs whose result file already exists (or that are in the old stats/p3_results.json) are skipped.

Two extra job types:
  {"type": "lp", "name": X, "train": true}   re-load finished run X and (re)compute its log-probs
                                             (train split only if asked: needed for KD teachers)
  KD run: normal run + "teachers": [names], "student_layers": [0,2,4,6,8,10], "kd_T", "kd_lam"
          (teacher = mean of the teachers' TRAIN log-probs; student = M1 with only those layers)
"""
import argparse
import glob
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
import p3_rr as rr                                                    # noqa: E402
import p2_peft as pp                                                  # noqa: E402
from tokenizers import Tokenizer                                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
PLAN = json.load(open(ap.parse_args().plan))

DEV = "cuda:0"                                                       # = the GPU given by CUDA_VISIBLE_DEVICES
CK = {"m1": f"{WORK}/checkpoints/m1_dense_100m/best.pt", "m2": f"{WORK}/checkpoints/m2_mla_moe/best.pt",
      "m3": f"{WORK}/checkpoints/m3_encdec/best.pt", "m4": f"{WORK}/checkpoints/m4_copy/best.pt"}
OUT, LPD, CKD = f"{WORK}/stats/p3_runs", f"{WORK}/p3_lp", f"{WORK}/p3_ckpt"
for d in (OUT, LPD, CKD):
    os.makedirs(d, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---- data (same encoding, priors and class weights as the notebook) ----------------------------
def load_docs(path):
    obj = json.load(open(path))
    if isinstance(obj, dict):
        obj = [{"id": k, **v} if isinstance(v, dict) else {"id": k, "text": v} for k, v in obj.items()]
    return obj


tk = Tokenizer.from_file(f"{WORK}/tokenizer/hf_a1_bpe_v2_32k.json")
D = {os.path.basename(f)[:-5]: load_docs(f)
     for f in glob.glob(f"{WORK}/data_p3/Phase_3_RR/*/*.json") if not f.endswith("label_vocab.json")}
ENC, KEYS = {}, {"train": [], "dev": [], "test": []}
for dom in ["CL", "IT"]:
    for split in ["train", "dev", "test"]:
        e = rr.encode_docs(D[f"{dom}_{split}"], tk, dom)
        ENC.update(e); KEYS[split] += list(e)
ALLOWED, TRANS = rr.domain_stats(ENC, KEYS["train"])
cnt = np.bincount([l for k in KEYS["train"] for l in ENC[k]["labs"]], minlength=rr.NL).astype(float)
CW = torch.tensor(np.where(cnt > 0, np.sqrt(cnt.max() / np.maximum(cnt, 1)), 1.0), dtype=torch.float).clamp(max=10.0)
OLD = json.load(open(f"{WORK}/stats/p3_results.json")) if os.path.exists(f"{WORK}/stats/p3_results.json") else {}
log(f"data ready: {[len(KEYS[s]) for s in KEYS]} docs | plan: {[p['name'] for p in PLAN]}")


def run(name, ckpt="m1", method="lora", ctx="doc", W=512, use_pos=True, bidir=True,
        epochs=6, lr=5e-4, bs=None, seed=0, teachers=None, student_layers=None, kd_T=2.0, kd_lam=0.5, **kw):
    t0 = time.time(); bs = bs or (32 if ctx == "none" else 8)
    model = rr.build_rr(CK[ckpt], method, bidirectional=bidir, use_pos=use_pos, **kw)
    if student_layers:                                               # smaller student: keep a subset of layers
        pp.shrink_dense(model.tagger, tuple(student_layers))
    tr_ex = rr.make_examples(ENC, KEYS["train"], ctx, W)
    if teachers:                                                     # soft targets = mean teacher train log-probs
        T = [pickle.load(open(f"{LPD}/{t}.pkl", "rb"))["train"] for t in teachers]
        soft = {k: np.mean([t[k] for t in T], 0) for k in KEYS["train"]}
        for x in tr_ex:
            x["soft"] = [soft[x["key"]][i] for i in x["sidx"]]
    ev = lambda m: rr.evaluate(rr.predict(m, ENC, KEYS["dev"], ctx, W, DEV), ENC, KEYS["dev"], ALLOWED)
    log(f"=== {name}: {ckpt} | {method} | ctx={ctx} W={W} pos={use_pos} bidir={bidir} lr={lr} ===")
    hist, best = rr.train_rr(model, tr_ex, ev, epochs=epochs, lr=lr, bs=bs, class_weight=CW,
                             device=DEV, seed=seed, log=log, kd_T=kd_T, kd_lam=kd_lam)
    LPd = rr.predict(model, ENC, KEYS["dev"], ctx, W, DEV)
    torch.cuda.synchronize(); ti = time.time()
    LPt = rr.predict(model, ENC, KEYS["test"], ctx, W, DEV)
    torch.cuda.synchronize(); infer_s = round(time.time() - ti, 2)
    out = {"cfg": dict(ckpt=ckpt, method=method, ctx=ctx, W=W, use_pos=use_pos, bidir=bidir,
                       epochs=epochs, lr=lr, seed=seed, teachers=teachers, student_layers=student_layers,
                       kd_T=kd_T, kd_lam=kd_lam, **kw),
           "trainable_M": round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 3),
           "total_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2), "infer_test_s": infer_s,
           "hist": hist, "min": round((time.time() - t0) / 60, 1)}
    for lam in (0.0, 0.5, 1.0):
        r = rr.evaluate(LPd, ENC, KEYS["dev"], ALLOWED, TRANS, lam)
        out[f"lam{lam}"] = {k: {m: round(v, 4) for m, v in d.items()} for k, d in r.items()}
    rr.save_trainable(model, f"{CKD}/{name}.pt")
    pickle.dump({"dev": LPd, "test": LPt}, open(f"{LPD}/{name}.pkl", "wb"))
    json.dump(out, open(f"{OUT}/{name}.json", "w"), indent=1)                # written LAST = "done" marker
    log(f"DONE {name}: dev macro λ0 {out['lam0.0']['all']['macro_f1']:.4f} "
        f"(CL {out['lam0.0']['CL']['macro_f1']:.4f} IT {out['lam0.0']['IT']['macro_f1']:.4f}) "
        f"| λ0.5 {out['lam0.5']['all']['macro_f1']:.4f} | {out['min']} min")
    del model; torch.cuda.empty_cache()


def lp_job(name, train=False):
    """Re-load finished run `name` (config from its result file) and save its log-probs."""
    R = json.load(open(f"{OUT}/{name}.json")) if os.path.exists(f"{OUT}/{name}.json") else OLD[name]
    c = {k: v for k, v in R["cfg"].items() if v is not None}
    ctx, W, ckpt, method = c.pop("ctx"), c.pop("W"), c.pop("ckpt"), c.pop("method")
    bidir, use_pos, layers = c.pop("bidir"), c.pop("use_pos"), c.pop("student_layers", None)
    for k in ("epochs", "lr", "seed", "bs", "teachers", "kd_T", "kd_lam"):
        c.pop(k, None)
    m = rr.build_rr(CK[ckpt], method, bidirectional=bidir, use_pos=use_pos, **c)
    if layers:
        pp.shrink_dense(m.tagger, tuple(layers))
    rr.load_trainable(m, f"{CKD}/{name}.pt"); m.to(DEV)
    path = f"{LPD}/{name}.pkl"
    out = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
    for split in (["train"] if train else []) + ["dev", "test"]:
        if split not in out:
            ti = time.time(); out[split] = rr.predict(m, ENC, KEYS[split], ctx, W, DEV)
            log(f"   {name}: {split} log-probs in {time.time()-ti:.1f}s")
    pickle.dump(out, open(path, "wb"))
    chk = rr.evaluate(out["dev"], ENC, KEYS["dev"], ALLOWED)["all"]["macro_f1"]
    log(f"LP {name}: dev macro re-check {chk:.4f} (should equal its λ0 score)")
    del m; torch.cuda.empty_cache()


for cfg in PLAN:
    cfg = dict(cfg); name = cfg.pop("name")
    if cfg.pop("type", "run") == "lp":
        try:
            lp_job(name, cfg.get("train", False))
        except Exception:
            log(f"!!! LP {name} FAILED"); traceback.print_exc(); torch.cuda.empty_cache()
        continue
    if os.path.exists(f"{OUT}/{name}.json") or name in OLD:
        log(f"skip {name} (already done)"); continue
    try:
        run(name, **cfg)
    except Exception:
        log(f"!!! {name} FAILED"); traceback.print_exc(); torch.cuda.empty_cache()
log("PLAN DONE")
