"""
p6_run.py -- background runner for Phase-6 summarisation experiments (one process per GPU):

    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 setsid nohup python p6_run.py --plan plan.json >> log 2>&1 &
    python p6_run.py --smoke        # ~3-5 min: every model x method x code path on a few windows

Job (default type "run"):
  {"name", "ckpt": "m1".."m4", "method": full|lora|lorma_plus|lorma_pi|houlsby|pfeiffer|prompt|qlora,
   "lr", "epochs", "bs", "r", "alpha", "mode": "chunk" (window-level, default) | "trunc" (baseline),
   KD: "teacher": <finished run name>, "student_layers": [0,2,4,6,8,10], "kd_T", "kd_lam"}
  {"type": "gen", "name": X, "tag": T, "eos_bias": b, "no_repeat": n, "dedupe": bool}
        re-decode finished run X with other generation settings -> X__T (length / redundancy study)
  {"type": "cal", "name": X, "biases": [...], "n_docs": 20}
        LENGTH CALIBRATION: waits until run X is finished, decodes the first n_docs validation documents
        with every eos_bias in `biases`, keeps the one with the best (ROUGE-1 + ROUGE-L) / 2, then writes
        the full validation + test output with it -> X__cal (+ its zip)
Outputs per run: stats/p6_runs/<n>.json (written LAST = done marker), p6_ckpt/<n>.pt (trained params),
p6_gen/<n>.json ({"val": {id: summary}, "test": {id: summary}}), submissions_p6/p6_<n>/p6_<n>.zip.
Validation = the first 100 documents of our held-out split (never trained on).
"""
import argparse
import json
import os
import sys
import time
import traceback
import zipfile

import numpy as np
import torch

WORK = "/kaggle/working"
sys.path.insert(0, f"{WORK}/src")
import p6_sum as S                                                    # noqa: E402
from tokenizers import Tokenizer                                     # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--plan"); ap.add_argument("--smoke", action="store_true")
ARGS = ap.parse_args()
DEV = "cuda:0"
CK = {"m1": f"{WORK}/checkpoints/m1_dense_100m/best.pt", "m2": f"{WORK}/checkpoints/m2_mla_moe/best.pt",
      "m3": f"{WORK}/checkpoints/m3_encdec/best.pt", "m4": f"{WORK}/checkpoints/m4_copy/best.pt"}
OUT, CKD, GEN, SUB = (f"{WORK}/stats/p6_runs", f"{WORK}/p6_ckpt", f"{WORK}/p6_gen", f"{WORK}/submissions_p6")
for d in (OUT, CKD, GEN, SUB):
    os.makedirs(d, exist_ok=True)
N_VAL = 100


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


TK = Tokenizer.from_file(f"{WORK}/tokenizer/hf_a1_bpe_v2_32k.json")
decode = lambda ids: TK.decode(ids, skip_special_tokens=True)
IDS = {s: [m["id"] for m in json.load(open(f"{WORK}/data_p6/clean/{s}.json"))] for s in ("train", "val", "test")}
REF = {m["id"]: m["summary"] for m in json.load(open(f"{WORK}/data_p6/clean/val.json"))}
_DATA = {}


def data(mode, split):
    key = (mode, split)
    if key not in _DATA:
        _DATA[key] = (S.Chunks(f"{WORK}/data_p6/chunks/{split}.npz") if mode == "chunk" else
                      S.Trunc(f"{WORK}/data_p6/tok/{split}_doc.npz",
                              f"{WORK}/data_p6/tok/{split}_sum.npz" if split != "test" else None))
    return _DATA[key]


def build(ckpt, method, student_layers=None, r=16, alpha=32, **_):
    m = S.load_lm(CK[ckpt])
    if student_layers:
        S.shrink(m, tuple(student_layers))
    return S.apply_method(m, method, r=r, alpha=alpha)


def load_run(name):
    R = json.load(open(f"{OUT}/{name}.json")); c = R["cfg"]
    m = build(**c)
    sd = {k: v.float() for k, v in torch.load(f"{CKD}/{name}.pt", map_location="cpu").items()}
    m.load_state_dict(sd, strict=False)
    return m.to(DEV), c


def summarise(m, mode, split, docs, **gen_kw):
    d = data(mode, split); idx = d.windows_of(docs)
    t0 = time.time(); g = S.generate(m, d, idx, device=DEV, **{k: v for k, v in gen_kw.items() if k != "dedupe"})
    secs = time.time() - t0
    text = S.join_summaries(d, g, docs, decode, dedupe=gen_kw.get("dedupe", True))
    return {IDS[split][k]: v for k, v in text.items()}, secs, sum(len(v) for v in g.values())


def finish(name, cfg, m, mode, t0, extra, gen_kw):
    val, vs, vtok = summarise(m, mode, "val", list(range(N_VAL)), **gen_kw)
    test, ts, ttok = summarise(m, mode, "test", list(range(len(IDS["test"]))), **gen_kw)
    rs = S.rouge_scores([REF[i] for i in val], [val[i] for i in val])
    tr, tot, mem = S.n_params(m)
    json.dump({"val": val, "test": test}, open(f"{GEN}/{name}.json", "w"))
    folder = f"{SUB}/p6_{name}"; os.makedirs(folder, exist_ok=True)
    pred = [{"id": i, "summary": test[i]} for i in IDS["test"]]
    assert len(pred) == len(set(IDS["test"])) and all(p["summary"].strip() for p in pred)
    json.dump(pred, open(f"{folder}/prediction.json", "w"))
    with zipfile.ZipFile(f"{folder}/p6_{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{folder}/prediction.json", arcname="prediction.json")
    out = {"cfg": cfg, "gen": gen_kw, "trainable_M": round(tr / 1e6, 3), "total_M": round(tot / 1e6, 1),
           "model_MB": round(mem / 1e6, 1), "val_rouge": rs, "gen_tok_per_s": round((vtok + ttok) / max(1e-6, vs + ts), 1),
           "test_gen_min": round(ts / 60, 1), "test_words_median": int(np.median([len(v.split()) for v in test.values()])),
           "min": round((time.time() - t0) / 60, 1), **extra}
    json.dump(out, open(f"{OUT}/{name}.json", "w"), indent=1)                 # written LAST = done marker
    log(f"DONE {name}: val R1 {rs['rouge1']:.4f} R2 {rs['rouge2']:.4f} RL {rs['rougeL']:.4f} | length x{rs['len_ratio']:.2f} "
        f"| {out['gen_tok_per_s']} tok/s | {out['min']} min")


def run(name, ckpt="m1", method="full", lr=2e-4, epochs=1, bs=8, r=16, alpha=32, mode="chunk", teacher=None,
        student_layers=None, kd_T=2.0, kd_lam=0.5, max_steps=None, seed=0):
    t0 = time.time()
    cfg = dict(ckpt=ckpt, method=method, lr=lr, epochs=epochs, bs=bs, r=r, alpha=alpha, mode=mode,
               teacher=teacher, student_layers=student_layers, kd_T=kd_T, kd_lam=kd_lam, seed=seed)
    m = build(ckpt, method, student_layers, r, alpha)
    T = load_run(teacher)[0] if teacher else None
    tr_d, va_d = data(mode, "train"), data(mode, "val")
    log(f"=== {name}: {ckpt} | {method} | {mode} | lr {lr}" + (f" | KD from {teacher}" if teacher else ""))
    hist = S.train(m, tr_d, list(range(tr_d.n)), epochs=epochs, lr=lr, bs=bs, device=DEV, seed=seed, teacher=T,
                   kd_T=kd_T, kd_lam=kd_lam, max_steps=max_steps, log=log)
    del T; torch.cuda.empty_cache()
    vl = S.val_loss(m, va_d, va_d.windows_of(list(range(N_VAL))), device=DEV)
    log(f"   val target NLL {vl:.4f} | training {round((time.time()-t0)/60, 1)} min")
    torch.save({n: p.detach().half().cpu() for n, p in m.named_parameters() if p.requires_grad}, f"{CKD}/{name}.pt")
    finish(name, cfg, m, mode, t0, {"hist": hist, "val_nll": vl, "train_min": round((time.time() - t0) / 60, 1)},
           {"eos_bias": 0.0, "no_repeat": 4, "dedupe": True})
    del m; torch.cuda.empty_cache()


def gen_job(name, tag, **gen_kw):
    t0 = time.time(); m, c = load_run(name)
    log(f"=== {name}__{tag}: re-decode with {gen_kw}")
    finish(f"{name}__{tag}", {**c, "base": name}, m, c["mode"], t0, {}, {"eos_bias": 0.0, "no_repeat": 4, "dedupe": True, **gen_kw})
    del m; torch.cuda.empty_cache()


def cal_job(name, biases=(-1.0, -2.0, -3.0, -4.5), n_docs=20, wait_h=8.0):
    t_wait = time.time()
    while not os.path.exists(f"{OUT}/{name}.json"):                    # base run still training elsewhere
        if time.time() - t_wait > wait_h * 3600:
            raise TimeoutError(f"{name} did not finish")
        time.sleep(120)
    t0 = time.time(); m, c = load_run(name); res = {}
    log(f"=== {name}__cal: eos_bias sweep {list(biases)} on {n_docs} val documents")
    for b in biases:
        val, _, _ = summarise(m, c["mode"], "val", list(range(n_docs)), eos_bias=b, no_repeat=4, dedupe=True)
        rs = S.rouge_scores([REF[i] for i in val], [val[i] for i in val]); res[b] = rs
        log(f"   eos_bias {b:+.1f}: R1 {rs['rouge1']:.4f} RL {rs['rougeL']:.4f} | length x{rs['len_ratio']:.2f}")
    ok = [b for b in res if 0.8 <= res[b]["len_ratio"] <= 1.25]       # length close to the reference
    best = (max(ok, key=lambda b: res[b]["rouge1"] + res[b]["rougeL"]) if ok
            else min(res, key=lambda b: abs(res[b]["len_ratio"] - 1)))
    log(f"   -> eos_bias {best:+.1f}")
    finish(f"{name}__cal", {**c, "base": name}, m, c["mode"], t0, {"cal": {str(k): v for k, v in res.items()}},
           {"eos_bias": best, "no_repeat": 4, "dedupe": True})
    del m; torch.cuda.empty_cache()


def smoke():
    """Every model x method on a few windows: 2 train steps, val NLL, 3 windows decoded (6 tokens)."""
    tr, va = data("chunk", "train"), data("chunk", "val")
    few = [i for i in range(200) if tr.tg_off[i + 1] > tr.tg_off[i]][:4]
    combos = [("m1", "full"), ("m1", "lorma_plus"), ("m1", "lorma_pi"), ("m1", "qlora"), ("m1", "prompt"),
              ("m2", "houlsby"), ("m3", "full"), ("m3", "pfeiffer"), ("m3", "prompt"), ("m4", "lora")]
    for ck, meth in combos:
        m = build(ck, meth); h = S.train(m, tr, few, lr=1e-4, bs=2, device=DEV, max_steps=2, log=lambda *a: None)
        vl = S.val_loss(m, va, va.windows_of([0])[:2], device=DEV)
        g = S.generate(m, va, va.windows_of([0])[:3], max_new=6, device=DEV)
        print(f"OK {ck} {meth:10s} | trainable {S.n_params(m)[0]/1e6:.2f}M | loss {h and h[-1]['loss'] or 'n/a'} "
              f"| val NLL {vl:.3f} | sample: {decode(next(iter(g.values())))!r}")
        del m; torch.cuda.empty_cache()
    t = S.apply_method(S.load_lm(CK["m1"]), "full"); s = build("m1", "full", (0, 2, 4, 6, 8, 10))
    S.train(s, tr, few, bs=2, device=DEV, max_steps=2, teacher=t, log=lambda *a: None)
    print("OK KD student <- teacher | student layers", len(s.blocks))
    tc = data("trunc", "val"); m = build("m3", "full").to(DEV)
    print("OK trunc baseline: pairs", tc.n, "| NLL", round(S.val_loss(m, tc, [0, 1], device=DEV), 3))
    print("OK rouge:", S.rouge_scores(["the court held that the appeal fails"], ["the appeal fails the court held"]))
    print("SMOKE TEST PASSED")


if ARGS.smoke:
    smoke(); sys.exit(0)
for job in json.load(open(ARGS.plan)):
    job = dict(job); name = job.pop("name"); typ = job.pop("type", "run")
    target = f"{name}__{job['tag']}" if typ == "gen" else (f"{name}__cal" if typ == "cal" else name)
    if os.path.exists(f"{OUT}/{target}.json"):
        log(f"skip {target} (already done)"); continue
    try:
        if typ == "gen":
            gen_job(name, **job)
        elif typ == "cal":
            cal_job(name, **job)
        else:
            run(name, **job)
    except Exception:
        log(f"!!! {target} FAILED"); traceback.print_exc(); torch.cuda.empty_cache()
log("PLAN DONE")
