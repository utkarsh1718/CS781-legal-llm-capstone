"""
p3_extra.py -- Phase-3 upgrades, run as a BACKGROUND process (one per GPU), same pattern as p3_run.py:

    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 setsid nohup python p3_extra.py --plan plan.json >> log 2>&1 &
    python p3_extra.py --smoke          # 1-2 min end-to-end check of every code path (run this first)

Job types (plan.json = list of jobs):
  {"type": "hier", "name": N, "bases": [run, ...]}
        document-level model: per-sentence embeddings of the (frozen) fine-tuned runs `bases` for the
        WHOLE judgment -> 2-layer BiGRU over the sentence sequence -> 13 roles. Trained on train,
        best epoch chosen on dev. Embeddings are cached in p3_emb/<run>.pkl.
  {"type": "td", "name": X}
        retrain run X's exact config on TRAIN + DEV for X's best epoch count (no held-out data, so
        the last epoch is kept). Output X_td (test log-probs only).
  {"type": "hier_td", "name": N_td, "ref": N}
        hierarchical model N rebuilt on the train+dev versions of its bases (<base>_td), trained on
        train + dev for N's best epoch count.
  {"type": "seed", "name": X, "seed": s}
        rerun X's config with another seed (train only, dev selection) -> X_s<s> (ensemble diversity).
Outputs follow p3_run.py: stats/p3_runs/<name>.json (written LAST = done marker), p3_ckpt/<name>.pt,
p3_lp/<name>.pkl ({"dev", "test"} log-probs per document; td runs: {"test"} only).
"""
import argparse
import glob
import json
import os
import pickle
import random
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WORK = "/kaggle/working"
sys.path.insert(0, f"{WORK}/src")
import p3_rr as rr                                                    # noqa: E402
import p2_peft as pp                                                  # noqa: E402
from tokenizers import Tokenizer                                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--plan"); ap.add_argument("--smoke", action="store_true")
ARGS = ap.parse_args()

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
CK = {"m1": f"{WORK}/checkpoints/m1_dense_100m/best.pt", "m2": f"{WORK}/checkpoints/m2_mla_moe/best.pt",
      "m3": f"{WORK}/checkpoints/m3_encdec/best.pt", "m4": f"{WORK}/checkpoints/m4_copy/best.pt"}
OUT, LPD, CKD, EMB = f"{WORK}/stats/p3_runs", f"{WORK}/p3_lp", f"{WORK}/p3_ckpt", f"{WORK}/p3_emb"
for d in (OUT, LPD, CKD, EMB):
    os.makedirs(d, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---- data: identical encoding / priors / class weights to p3_run.py ------------------------------
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
KEYS["td"] = KEYS["train"] + KEYS["dev"]
ALLOWED, TRANS = rr.domain_stats(ENC, KEYS["train"])


def class_weights(keys):
    cnt = np.bincount([l for k in keys for l in ENC[k]["labs"]], minlength=rr.NL).astype(float)
    return torch.tensor(np.where(cnt > 0, np.sqrt(cnt.max() / np.maximum(cnt, 1)), 1.0),
                        dtype=torch.float).clamp(max=10.0)


CW, CW_TD = class_weights(KEYS["train"]), class_weights(KEYS["td"])
OLD = json.load(open(f"{WORK}/stats/p3_results.json")) if os.path.exists(f"{WORK}/stats/p3_results.json") else {}
log(f"data ready: train {len(KEYS['train'])} | dev {len(KEYS['dev'])} | test {len(KEYS['test'])} docs | device {DEV}")


# ---- helpers ------------------------------------------------------------------------------------
def result(name):
    """Result record of a finished run (p3_run.py json, or the old notebook results file)."""
    p = f"{OUT}/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else OLD[name]


def best_epoch(R):
    h = R.get("hist") or []
    if h and isinstance(h[0], dict) and "all" in h[0]:
        return int(max(h, key=lambda x: x["all"])["epoch"])
    return int((R.get("cfg") or {}).get("epochs") or 6)


def build_from(cfg):
    """Fresh model (pretrained backbone + method) from a run config; returns (model, ctx, W)."""
    c = {k: v for k, v in cfg.items() if v is not None}
    ctx, W, ckpt, method = c.pop("ctx"), c.pop("W"), c.pop("ckpt"), c.pop("method")
    bidir, use_pos, layers = c.pop("bidir"), c.pop("use_pos"), c.pop("student_layers", None)
    for k in ("epochs", "lr", "seed", "bs", "teachers", "kd_T", "kd_lam"):
        c.pop(k, None)
    m = rr.build_rr(CK[ckpt], method, bidirectional=bidir, use_pos=use_pos, **c)
    if layers:
        pp.shrink_dense(m.tagger, tuple(layers))
    return m, ctx, W


def lam_report(LPd):
    out = {}
    for lam in (0.0, 0.5, 1.0):
        r = rr.evaluate(LPd, ENC, KEYS["dev"], ALLOWED, TRANS, lam)
        out[f"lam{lam}"] = {k: {m: round(v, 4) for m, v in d.items()} for k, d in r.items()}
    return out


# ---- job 1: retrain on train + dev ---------------------------------------------------------------
def td_job(base):
    t0, name = time.time(), f"{base}_td"
    R = result(base); c = dict(R["cfg"])
    if c.get("teachers"):
        raise ValueError(f"{base} is a KD run; not retrained on train+dev")
    ep = best_epoch(R); lr = c.get("lr") or 5e-4; seed = c.get("seed") or 0
    m, ctx, W = build_from(c)
    bs = c.get("bs") or (32 if ctx == "none" else 8)
    ex = rr.make_examples(ENC, KEYS["td"], ctx, W)
    n = {"e": 0}

    def ev(_):                                                        # no held-out data: a counter makes
        n["e"] += 1; r = {"macro_f1": float(n["e"]), "acc": 0.0}      # train_rr keep the LAST epoch
        return {"all": r, "CL": r, "IT": r}
    log(f"=== {name}: {base}'s config on train+dev ({len(KEYS['td'])} docs) for {ep} epochs, lr {lr} "
        f"('dev macro-F1' in this log = epoch counter, not a score)")
    rr.train_rr(m, ex, ev, epochs=ep, lr=lr, bs=bs, class_weight=CW_TD, device=DEV, seed=seed, log=log)
    LPt = rr.predict(m, ENC, KEYS["test"], ctx, W, DEV)
    rr.save_trainable(m, f"{CKD}/{name}.pt")
    pickle.dump({"test": LPt}, open(f"{LPD}/{name}.pkl", "wb"))
    json.dump({"cfg": R["cfg"], "base": base, "td": True, "epochs_trained": ep,
               "min": round((time.time() - t0) / 60, 1)}, open(f"{OUT}/{name}.json", "w"), indent=1)
    log(f"DONE {name}: {ep} epochs on train+dev | {round((time.time() - t0) / 60, 1)} min")
    del m; torch.cuda.empty_cache()


# ---- job 2: another seed -----------------------------------------------------------------------
def seed_job(base, seed):
    t0, name = time.time(), f"{base}_s{seed}"
    R = result(base); c = dict(R["cfg"])
    ep = int(c.get("epochs") or 6); lr = c.get("lr") or 5e-4
    m, ctx, W = build_from(c)
    bs = c.get("bs") or (32 if ctx == "none" else 8)
    ex = rr.make_examples(ENC, KEYS["train"], ctx, W)
    ev = lambda mm: rr.evaluate(rr.predict(mm, ENC, KEYS["dev"], ctx, W, DEV), ENC, KEYS["dev"], ALLOWED)
    log(f"=== {name}: {base}'s config, seed {seed}, {ep} epochs")
    hist, _ = rr.train_rr(m, ex, ev, epochs=ep, lr=lr, bs=bs, class_weight=CW, device=DEV, seed=seed, log=log)
    LPd = rr.predict(m, ENC, KEYS["dev"], ctx, W, DEV); LPt = rr.predict(m, ENC, KEYS["test"], ctx, W, DEV)
    rr.save_trainable(m, f"{CKD}/{name}.pt")
    pickle.dump({"dev": LPd, "test": LPt}, open(f"{LPD}/{name}.pkl", "wb"))
    out = {"cfg": {**R["cfg"], "seed": seed}, "base": base, "hist": hist,
           "min": round((time.time() - t0) / 60, 1), **lam_report(LPd)}
    json.dump(out, open(f"{OUT}/{name}.json", "w"), indent=1)
    log(f"DONE {name}: dev macro λ0 {out['lam0.0']['all']['macro_f1']:.4f} | λ0.5 {out['lam0.5']['all']['macro_f1']:.4f} "
        f"| {out['min']} min")
    del m; torch.cuda.empty_cache()


# ---- job 3: document-level BiGRU over sentence embeddings ----------------------------------------
@torch.no_grad()
def sent_emb(m, keys, ctx, W, bs=16):
    """Per-sentence vector the role head sees (mean-pooled tokens + position), averaged over windows."""
    m.eval(); buf = {}
    hook = m.drop.register_forward_hook(lambda mod, inp, out: buf.__setitem__("x", inp[0].detach().float()))
    ex = sorted(rr.make_examples(ENC, keys, ctx, W), key=lambda x: len(x["ids"]))
    d = m.tagger.backbone.cfg.d_model
    acc = {k: np.zeros((len(ENC[k]["sents"]), d), np.float32) for k in keys}
    cnt = {k: np.zeros(len(ENC[k]["sents"])) for k in keys}
    for b in range(0, len(ex), bs):
        chunk = ex[b:b + bs]
        ids, seg, off, _, pos = rr.collate(chunk, DEV)
        with rr._autocast(DEV):
            m(ids, seg, off, pos)
        x, j = buf["x"].cpu().numpy(), 0
        for e in chunk:
            for i in e["sidx"]:
                acc[e["key"]][i] += x[j]; cnt[e["key"]][i] += 1; j += 1
    hook.remove()
    return {k: (acc[k] / cnt[k][:, None]).astype(np.float16) for k in keys}


def get_emb(run, splits=("train", "dev", "test")):
    path = f"{EMB}/{run}.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    src = run[:-3] if run.endswith("_td") else run
    m, ctx, W = build_from(result(src)["cfg"])
    rr.load_trainable(m, f"{CKD}/{run}.pt"); m.to(DEV)
    E = {}
    for s in splits:
        ti = time.time(); E[s] = sent_emb(m, KEYS[s], ctx, W)
        log(f"   embeddings {run} / {s}: {sum(len(v) for v in E[s].values())} sentences in {time.time()-ti:.0f}s")
    pickle.dump(E, open(path, "wb")); del m; torch.cuda.empty_cache()
    return E


class DocGRU(nn.Module):
    """Sentence embeddings of a whole judgment -> 2-layer BiGRU -> role logits for every sentence."""
    def __init__(self, d, h=256, p=0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d), nn.Dropout(p))
        self.dom = nn.Embedding(2, d)                                 # CL / IT
        self.gru = nn.GRU(d, h, num_layers=2, batch_first=True, bidirectional=True, dropout=p)
        self.out = nn.Sequential(nn.Dropout(p), nn.Linear(2 * h, rr.NL))

    def forward(self, x, dom):                                        # x (B, T, d), dom (B,)
        z, _ = self.gru(self.inp(x) + self.dom(dom)[:, None])
        return self.out(z)


def _feats(Es, k):
    split = "test" if k in TEST_SET else ("dev" if k in DEV_SET else "train")
    return np.concatenate([E[split][k] for E in Es], 1).astype(np.float32)


def _batch(Es, keys):
    X = [_feats(Es, k) for k in keys]; T = max(len(x) for x in X)
    x = np.zeros((len(keys), T, X[0].shape[1]), np.float32); y = np.full((len(keys), T), -100, np.int64)
    for b, (k, f) in enumerate(zip(keys, X)):
        x[b, :len(f)] = f; y[b, :len(f)] = ENC[k]["labs"][:len(f)] if ENC[k]["labs"][0] != -100 else -100
    dom = torch.tensor([0 if ENC[k]["dom"] == "CL" else 1 for k in keys], device=DEV)
    return torch.from_numpy(x).to(DEV), torch.from_numpy(y).to(DEV), dom


@torch.no_grad()
def _gru_lp(net, Es, keys):
    net.eval(); out = {}
    for b in range(0, len(keys), 8):
        ks = keys[b:b + 8]; x, _, dom = _batch(Es, ks)
        lp = net(x, dom).float().log_softmax(-1).cpu().numpy()
        for j, k in enumerate(ks):
            out[k] = lp[j, :len(ENC[k]["sents"])]
    return out


TEST_SET, DEV_SET = set(KEYS["test"]), set(KEYS["dev"])


def train_gru(Es, train_keys, epochs, cw, seed=0, select_on_dev=True, lr=1e-3, bs=4):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    net = DocGRU(_feats(Es, train_keys[0]).shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(train_keys) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    best, best_state, best_ep, hist = -1.0, None, epochs, []
    for ep in range(epochs):
        net.train(); order = list(train_keys); random.shuffle(order); tot = 0.0
        for b in range(0, len(order), bs):
            x, y, dom = _batch(Es, order[b:b + bs])
            loss = F.cross_entropy(net(x, dom).reshape(-1, rr.NL), y.reshape(-1), weight=cw.to(DEV), ignore_index=-100)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); tot += loss.item()
        if select_on_dev:
            LPd = _gru_lp(net, Es, KEYS["dev"])
            f = max(rr.evaluate(LPd, ENC, KEYS["dev"], ALLOWED, TRANS, lam)["all"]["macro_f1"] for lam in (0.0, 0.5))
            hist.append({"epoch": ep + 1, "loss": tot, "all": f})
            if f > best:
                best, best_ep = f, ep + 1
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, hist, best_ep


def hier_job(name, bases, epochs=30, seed=0):
    t0 = time.time(); Es = [get_emb(b) for b in bases]
    log(f"=== {name}: document BiGRU over sentence embeddings of {bases} ({Es[0]['dev'][KEYS['dev'][0]].shape[1] * len(Es)}-d)")
    net, hist, best_ep = train_gru(Es, KEYS["train"], epochs, CW, seed=seed)
    LPd, LPt = _gru_lp(net, Es, KEYS["dev"]), _gru_lp(net, Es, KEYS["test"])
    out = {"cfg": {"type": "hier", "bases": bases, "epochs": epochs, "seed": seed}, "best_epoch": best_ep,
           "hist": hist, "min": round((time.time() - t0) / 60, 1), **lam_report(LPd)}
    torch.save(net.state_dict(), f"{CKD}/{name}.pt")
    pickle.dump({"dev": LPd, "test": LPt}, open(f"{LPD}/{name}.pkl", "wb"))
    json.dump(out, open(f"{OUT}/{name}.json", "w"), indent=1)
    log(f"DONE {name}: dev macro λ0 {out['lam0.0']['all']['macro_f1']:.4f} | λ0.5 {out['lam0.5']['all']['macro_f1']:.4f} "
        f"(best epoch {best_ep}) | {out['min']} min")


def hier_td_job(name, ref):
    t0 = time.time(); R = result(ref); bases = [b + "_td" for b in R["cfg"]["bases"]]
    missing = [b for b in bases if not os.path.exists(f"{CKD}/{b}.pt")]
    if missing:
        raise FileNotFoundError(f"train+dev bases not finished yet: {missing}")
    Es = [get_emb(b) for b in bases]
    log(f"=== {name}: {ref}'s BiGRU on train+dev bases {bases} for {R['best_epoch']} epochs")
    net, _, _ = train_gru(Es, KEYS["td"], R["best_epoch"], CW_TD, seed=R["cfg"].get("seed", 0), select_on_dev=False)
    LPt = _gru_lp(net, Es, KEYS["test"])
    torch.save(net.state_dict(), f"{CKD}/{name}.pt")
    pickle.dump({"test": LPt}, open(f"{LPD}/{name}.pkl", "wb"))
    json.dump({"cfg": {"type": "hier_td", "ref": ref, "bases": bases}, "td": True, "epochs_trained": R["best_epoch"],
               "min": round((time.time() - t0) / 60, 1)}, open(f"{OUT}/{name}.json", "w"), indent=1)
    log(f"DONE {name} | {round((time.time() - t0) / 60, 1)} min")


# ---- smoke test: every code path on a few documents, nothing written to the real outputs -----------
if ARGS.smoke:
    base = "m1_rslora"
    m, ctx, W = build_from(result(base)["cfg"]); rr.load_trainable(m, f"{CKD}/{base}.pt"); m.to(DEV)
    ks = KEYS["train"][:3] + KEYS["dev"][:2] + KEYS["test"][:1]
    E = {s: sent_emb(m, [k for k in ks if (k in DEV_SET) == (s == "dev") and (k in TEST_SET) == (s == "test")], ctx, W)
         for s in ("train", "dev", "test")}
    log("embeddings ok:", {s: len(v) for s, v in E.items()}, "dim", next(iter(E["dev"].values())).shape)
    tr_keys = list(E["train"]); KEYS_DEV_BAK = KEYS["dev"]; KEYS["dev"] = list(E["dev"])
    net, hist, be = train_gru([E], tr_keys, 2, CW)
    log("BiGRU ok: hist", hist, "| test lp shape", next(iter(_gru_lp(net, [E], list(E["test"])).values())).shape)
    KEYS["dev"] = KEYS_DEV_BAK
    log("td-config ok: best epoch of", base, "=", best_epoch(result(base)), "| cfg", result(base)["cfg"])
    log("SMOKE TEST PASSED"); sys.exit(0)


PLAN = json.load(open(ARGS.plan))
for job in PLAN:
    job = dict(job); name = job.pop("name"); typ = job.pop("type")
    tgt = {"td": f"{name}_td", "seed": f"{name}_s{job.get('seed')}"}.get(typ, name)
    if os.path.exists(f"{OUT}/{tgt}.json"):
        log(f"skip {tgt} (already done)"); continue
    try:
        if typ == "td":
            td_job(name)
        elif typ == "seed":
            seed_job(name, job["seed"])
        elif typ == "hier":
            hier_job(name, **job)
        elif typ == "hier_td":
            hier_td_job(name, **job)
    except Exception:
        log(f"!!! {tgt} FAILED"); traceback.print_exc(); torch.cuda.empty_cache()
log("PLAN DONE")
