"""p7_pcr.py -- Phase 7 Prior Case Retrieval on the Phase-1 M1 backbone (bi-encoder over 512-token chunks).

Reuses p4_cjpe: build() (bidirectional M1 + adaptation method) and DocClassifier.embed() (masked mean pool).
Per run -> stats/p7_runs/<name>.json (written LAST = done marker), p7_ckpt/<name>.pt (trainable params),
p7_emb/<name>_{dev,test}.npz (chunk embeddings), submissions_p7/p7_<name>/p7_<name>.zip.
"""
import json, math, os, random, time, zipfile

import numpy as np
import torch
import torch.nn.functional as F

import p4_cjpe as cj

WORK = "/kaggle/working"
D7, R7 = f"{WORK}/data_p7", f"{WORK}/data_p7/Phase_7_PCR"
OUT, CKD, EMB, SUB = (f"{WORK}/stats/p7_runs", f"{WORK}/p7_ckpt", f"{WORK}/p7_emb", f"{WORK}/submissions_p7")
for d in (OUT, CKD, EMB, SUB):
    os.makedirs(d, exist_ok=True)
CKPT = f"{WORK}/checkpoints/m1_dense_100m/best.pt"
CHUNK = 512
# Learning rates per adaptation method (contrastive retrieval fine-tuning)
LR = {"full": 3e-5, "partial": 1e-4, "lora": 5e-4, "lorma_plus": 5e-4, "lorma_pi": 5e-4,
      "ia3": 3e-3, "prompt": 3e-3}


# ----------------------------------------------------------------------------- data
class Docs:
    """Token cache of every Phase-7 case (written by P7-2 cell 3): arr(case_id) -> uint16 BPE ids."""
    def __init__(self, path=f"{D7}/tok/docs.npz"):
        z = np.load(path)
        self.ids, self.off = z["ids"], z["offsets"]
        self.pos = {n: i for i, n in enumerate(z["names"].tolist())}

    def arr(self, d):
        i = self.pos[d]
        return self.ids[self.off[i]:self.off[i + 1]]


def load_split(split):
    """Queries {id: {...}} and the ordered candidate-id list of one split's own pool."""
    s = split.lower()
    q = json.load(open(f"{R7}/{split}/{s}.json"))
    c = json.load(open(f"{R7}/{split}/{s}_candidates.json"))
    return q, list(c)


def chunk_starts(n, max_chunks):
    """Start offsets of 512-token chunks; a tiny tail (< 64 tokens) is dropped; long cases keep
    `max_chunks` chunks spread evenly over the whole document (not just the beginning)."""
    st = list(range(0, max(1, n), CHUNK))
    if len(st) > 1 and n - st[-1] < 64:
        st = st[:-1]
    if len(st) > max_chunks:
        st = [st[i] for i in np.linspace(0, len(st) - 1, max_chunks).round().astype(int)]
    return st


# ----------------------------------------------------------------------------- model
def build(method, partial_k=4, r=16, alpha=32):
    """Bidirectional M1 + the adaptation method (Phase-4 builder; its 2-way head is unused here)."""
    kw = dict(r=r, alpha=alpha) if method in ("lora", "lorma_plus", "lorma_pi") else {}
    return cj.build(CKPT, method=method, bidirectional=True, partial_k=partial_k, **kw)


def embed_cases(model, docs, ids, k, device, train=True):
    """One unit vector per case = normalised mean of its normalised chunk vectors.
    train=True samples k random chunks (memory); train=False uses the first k evenly spread chunks."""
    seqs, owner = [], []
    for j, d in enumerate(ids):
        a = docs.arr(d)
        st = chunk_starts(len(a), 10 ** 9 if train else k)
        pick = random.sample(st, min(k, len(st))) if train else st
        for s in pick:
            seqs.append(a[s:s + CHUNK]); owner.append(j)
    x, mask = cj.collate(seqs, device)
    with cj._autocast(device):
        e = model.embed(x, mask)                                        # (chunks, d) fp32
    e = F.normalize(e, dim=-1)
    out = torch.zeros(len(ids), e.shape[1], device=device).index_add_(0, torch.tensor(owner, device=device), e)
    return F.normalize(out, dim=-1)


# ----------------------------------------------------------------------------- training
def train(model, method, epochs=3, bs=4, k=2, n_hard=2, tau=0.05, lr=None, device="cuda",
          log=print, seed=0, max_steps=None, teacher=None):
    """InfoNCE over [gold positive + BM25 hard negatives] of every query in the batch (in-batch negatives).
    Other gold cases of the same query and the query itself are masked out of the denominator."""
    random.seed(seed); torch.manual_seed(seed)
    q, _ = load_split("Train")
    bm = json.load(open(f"{D7}/bm25_top100_train.json"))
    items = [(qid, r["relevant_candidates"]) for qid, r in q.items() if r["relevant_candidates"]]
    docs = Docs()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr or LR[method], weight_decay=0.01)
    total = max_steps or epochs * math.ceil(len(items) / bs)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    model.train(); step, t0, hist, run_loss = 0, time.time(), [], 0.0
    for ep in range(max(1, epochs)):
        random.shuffle(items)
        for b in range(0, len(items), bs):
            batch = items[b:b + bs]
            qids, cand, tgt = [x[0] for x in batch], [], []
            for qid, gold in batch:
                g = set(gold)
                tgt.append(len(cand)); cand.append(random.choice(gold))
                hard = [c for c in bm[qid][:50] if c not in g and c != qid]
                cand += random.sample(hard, min(n_hard, len(hard)))
            qe = embed_cases(model, docs, qids, k, device)
            ce = embed_cases(model, docs, cand, k, device)
            S = qe @ ce.T / tau
            # Mask false negatives: other gold cases of query i, and query i itself, appearing as candidates
            bad = torch.zeros_like(S, dtype=torch.bool)
            for i, (qid, gold) in enumerate(batch):
                g = set(gold)
                for j, c in enumerate(cand):
                    if j != tgt[i] and (c in g or c == qid):
                        bad[i, j] = True
            loss = F.cross_entropy(S.masked_fill(bad, -1e4), torch.tensor(tgt, device=device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1; run_loss += loss.item()
            if step % 50 == 0 or step == total:
                hist.append({"step": step, "loss": run_loss / min(50, step)}); run_loss = 0.0
                log(f"   step {step}/{total} | loss {hist[-1]['loss']:.4f} | {time.time()-t0:.0f}s")
            if step >= total:
                return hist
    return hist


# ----------------------------------------------------------------------------- encoding + scoring
@torch.no_grad()
def encode(model, docs, ids, max_chunks=16, device="cuda", bs=32):
    """Chunk embeddings of every case: E (n, max_chunks, d) fp16 unit vectors, M (n, max_chunks) mask."""
    model.eval()
    items = [(j, s) for j, d in enumerate(ids) for s in chunk_starts(len(docs.arr(d)), max_chunks)]
    d_model = model.tagger.backbone.cfg.d_model
    E = np.zeros((len(ids), max_chunks, d_model), np.float16); M = np.zeros((len(ids), max_chunks), bool)
    cnt = np.zeros(len(ids), int)
    for b in range(0, len(items), bs):
        batch = items[b:b + bs]
        x, mask = cj.collate([docs.arr(ids[j])[s:s + CHUNK] for j, s in batch], device)
        with cj._autocast(device):
            e = F.normalize(model.embed(x, mask), dim=-1).half().cpu().numpy()
        for (j, _), v in zip(batch, e):
            E[j, cnt[j]] = v; M[j, cnt[j]] = True; cnt[j] += 1
    return E, M


@torch.no_grad()
def dense_scores(E, M, qrows, mode, device="cuda"):
    """(n_queries, n_pool) similarities. mean: cosine of mean-pooled case vectors.
    maxsim: for each query chunk, best-matching candidate chunk; averaged over query chunks."""
    Et = torch.from_numpy(E).to(device).float(); Mt = torch.from_numpy(M).to(device)
    if mode == "mean":
        v = F.normalize((Et * Mt[..., None]).sum(1), dim=-1)
        return (v[qrows] @ v.T).cpu().numpy()
    out = np.zeros((len(qrows), len(E)), np.float32)
    flat = Et.reshape(-1, Et.shape[-1])                                 # all candidate chunks
    for i, r in enumerate(qrows):
        qc = Et[r][Mt[r]]                                               # (q_chunks, d)
        s = (qc @ flat.T).reshape(len(qc), len(E), -1)                  # (q_chunks, n_pool, max_chunks)
        s = s.masked_fill(~Mt[None], -1e4).max(-1).values.mean(0)
        out[i] = s.cpu().numpy()
    return out


ALPHAS = [round(float(x), 2) for x in np.arange(0, 1.0001, 0.05)]


ALPHAS = [round(float(x), 2) for x in np.arange(0, 1.0001, 0.05)]


def znorm(S):
    """Per-query z-score over the pool, ignoring NaN (the query's own column)."""
    return (S - np.nanmean(S, 1, keepdims=True)) / (np.nanstd(S, 1, keepdims=True) + 1e-6)


def top7(S, qids, cids):
    """Rank each query's pool, excluding the query itself; best 7 ids."""
    pos = {c: j for j, c in enumerate(cids)}; S = S.copy()
    for i, qid in enumerate(qids):
        if qid in pos: S[i, pos[qid]] = -np.inf
    return {qid: [cids[j] for j in np.argsort(-S[i])[:7]] for i, qid in enumerate(qids)}


def evaluate(rank, queries):
    """Exact competition metric: micro P/R/F1 over the top 7 of every query."""
    tp = sum(len(set(rank[q][:7]) & set(r["relevant_candidates"])) for q, r in queries.items())
    ng = sum(len(r["relevant_candidates"]) for r in queries.values())
    P, Rc = tp / (7 * len(queries)), tp / ng
    return {"F1": 2 * P * Rc / (P + Rc + 1e-12), "P": P, "R": Rc}


def write_submission(rank, name, qids, cids):
    pool = set(cids); pred = {k: rank[k][:7] for k in qids}
    assert all(len(v) == 7 and len(set(v)) == 7 and set(v) <= pool and k not in v for k, v in pred.items())
    d = f"{SUB}/p7_{name}"; os.makedirs(d, exist_ok=True)
    json.dump(pred, open(f"{d}/prediction.json", "w"), indent=1)
    with zipfile.ZipFile(f"{d}/p7_{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{d}/prediction.json", "prediction.json")


def bm25_matrix(split, qids, cids):
    """Saved full BM25 score matrix of the split, aligned to (qids, cids)."""
    z = np.load(f"{D7}/bm25_scores_{split.lower()}.npz")
    assert list(z["qids"]) == qids and list(z["cids"]) == cids, "BM25 matrix order mismatch"
    return z["S"]


def score_split(name, E, M, split, log=print, device="cuda", fixed=None):
    """Dense (mean / maxsim) alone and fused with BM25 over alpha; the query's own column is NaN before
    z-scoring (so it cannot distort either distribution) and -inf at ranking. fixed=(mode, alpha) for Test."""
    q, cids = load_split(split); qids = list(q); pos = {c: j for j, c in enumerate(cids)}
    qrows = np.array([pos[x] for x in qids]); ar = np.arange(len(qids))
    Sb = bm25_matrix(split, qids, cids).astype(np.float32); Sb[ar, qrows] = np.nan; Sb = znorm(Sb)
    res = {}
    for mode in ([fixed[0]] if fixed else ("mean", "maxsim")):
        Sd = dense_scores(E, M, list(qrows), mode, device).astype(np.float32)
        Sd[ar, qrows] = np.nan; Sd = znorm(Sd)
        for a in ([fixed[1]] if fixed else ALPHAS):
            rank = top7(np.nan_to_num(a * Sd + (1 - a) * Sb, nan=-np.inf), qids, cids)
            res[(mode, a)] = (evaluate(rank, q) if split != "Test" else None, rank)
    return res, qids, cids


def rescore(name, log=print, device="cuda"):
    """Re-run Dev selection + Test submission for a finished run from its saved chunk embeddings."""
    rec = json.load(open(f"{OUT}/{name}.json"))
    emb = {}
    for s in ("Dev", "Test"):
        z = np.load(f"{EMB}/{name}_{s.lower()}.npz"); emb[s] = (z["E"], z["M"])
    dev, _, _ = score_split(name, *emb["Dev"], "Dev", log, device)
    best = max(dev, key=lambda k: dev[k][0]["F1"])
    test, qids, cids = score_split(name, *emb["Test"], "Test", log, device, fixed=best)
    write_submission(test[best][1], name, qids, cids)
    rec.update(dev_best={"mode": best[0], "alpha": best[1], **dev[best][0]},
               dev_dense_only={m: dev[(m, 1.0)][0] for m in ("mean", "maxsim")},
               dev_table={f"{m}|{a}": v[0] for (m, a), v in dev.items()},
               rescored="self column NaN in z-norm; alpha step 0.05")
    json.dump(rec, open(f"{OUT}/{name}.json", "w"), indent=1)
    return rec


# ----------------------------------------------------------------------------- one experiment
def run(name, method="full", epochs=3, bs=4, k=2, n_hard=2, tau=0.05, lr=None, partial_k=4, r=16, alpha=32,
        max_chunks=16, seed=0, log=print, device="cuda"):
    t0 = time.time()
    model = build(method, partial_k, r, alpha).to(device)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ntot = sum(p.numel() for p in model.parameters())
    log(f"=== {name}: {method} | trainable {ntr/1e6:.3f}M of {ntot/1e6:.1f}M | epochs {epochs}")
    torch.cuda.reset_peak_memory_stats()
    hist = train(model, method, epochs, bs, k, n_hard, tau, lr, device, log, seed) if epochs > 0 else []
    train_min = (time.time() - t0) / 60
    if epochs > 0:
        torch.save({n: p.detach().half().cpu() for n, p in model.named_parameters() if p.requires_grad},
                   f"{CKD}/{name}.pt")
    docs, emb, enc_s = Docs(), {}, {}
    for split in ("Dev", "Test"):
        _, cids = load_split(split); t = time.time()
        emb[split] = encode(model, docs, cids, max_chunks, device); enc_s[split] = time.time() - t
        np.savez(f"{EMB}/{name}_{split.lower()}.npz", E=emb[split][0], M=emb[split][1])
        log(f"   encoded {split}: {len(cids)} cases in {enc_s[split]:.0f}s")
    # Dev: every (pooling, alpha); pick the best -> apply unchanged to Test
    dev, _, _ = score_split(name, *emb["Dev"], "Dev", log, device)
    table = {f"{m}|{a}": v[0] for (m, a), v in dev.items()}
    best = max(dev, key=lambda key: dev[key][0]["F1"])
    for m in ("mean", "maxsim"):
        log(f"   Dev {m:6s}: dense-only F1 {dev[(m, 1.0)][0]['F1']:.4f} | "
            f"best fused {max(dev[(m, a)][0]['F1'] for a in ALPHAS):.4f}")
    test, qids, cids = score_split(name, *emb["Test"], "Test", log, device, fixed=best)
    write_submission(test[best][1], name, qids, cids)
    rec = {"name": name, "method": method,
           "cfg": dict(epochs=epochs, bs=bs, k=k, n_hard=n_hard, tau=tau, lr=lr or LR.get(method),
                       partial_k=partial_k, r=r, alpha=alpha, max_chunks=max_chunks, seed=seed),
           "trainable_M": round(ntr / 1e6, 3), "total_M": round(ntot / 1e6, 1),
           "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
           "train_min": round(train_min, 1), "encode_s": {s: round(v) for s, v in enc_s.items()},
           "dev_best": {"mode": best[0], "alpha": best[1], **dev[best][0]},
           "dev_dense_only": {m: dev[(m, 1.0)][0] for m in ("mean", "maxsim")},
           "dev_table": table, "hist": hist, "min": round((time.time() - t0) / 60, 1)}
    json.dump(rec, open(f"{OUT}/{name}.json", "w"), indent=1)          # written LAST = done marker
    log(f"DONE {name}: Dev F1 {dev[best][0]['F1']:.4f} (P {dev[best][0]['P']:.3f} R {dev[best][0]['R']:.3f}) "
        f"| {best[0]} alpha {best[1]} | {rec['min']} min")
    return rec


def smoke(log=print, device="cuda"):
    """Every method: build, 2 training steps, encode 6 cases -- catches wiring errors in ~2 minutes."""
    docs = Docs(); _, cids = load_split("Dev")
    for method in ("full", "partial", "lorma_plus", "ia3", "prompt"):
        m = build(method).to(device)
        ntr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        hist = train(m, method, bs=2, k=1, n_hard=1, device=device, log=lambda *a: None, max_steps=2)
        E, M = encode(m, docs, cids[:6], max_chunks=2, device=device)
        log(f"smoke {method:10s} trainable {ntr/1e6:8.3f}M | 2 steps ok | emb {E.shape} | "
            f"norms {np.linalg.norm(E[M].astype(np.float32), axis=-1).mean():.3f}")
        del m; torch.cuda.empty_cache()
    log("SMOKE TEST PASSED")


# ============================================================================= P7-4: QLoRA + knowledge distillation
import torch.nn as nn
import p6_sum as S6                                                     # Phase-6 NF4Linear / LoRA wrapping

LR["qlora"] = 5e-4
_build_base = build


def drop_layers(model, keep):
    """Keep only the given transformer blocks (student = layer subset of M1, initialised from M1)."""
    bb = model.tagger.backbone
    for name, mod in bb.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > max(keep):
            parent = bb.get_submodule(name.rsplit(".", 1)[0]) if "." in name else bb
            setattr(parent, name.rsplit(".", 1)[-1], nn.ModuleList([mod[i] for i in keep]))
            try:
                bb.cfg.n_layers = len(keep)
            except Exception:
                pass
            return name
    raise RuntimeError("no transformer block list found in the backbone")


def build(method, partial_k=4, r=16, alpha=32, student_layers=None):
    """Adds 'qlora' (NF4 frozen base + LoRA) and layer-subset students to the Phase-4 builder."""
    if method == "qlora":
        m = _build_base("full")
        for p in m.parameters():
            p.requires_grad = False
        bb = m.tagger.backbone
        if not hasattr(bb, "arch"):
            bb.arch = m.tagger.arch                                   # p6 helpers read m.arch ('dense' -> m.blocks)
        S6.quantize_nf4(bb)
        S6._wrap(bb, lambda l: S6.pp.LoRALinear(l, r, alpha), S6.LORA_TARGETS)
    else:
        m = _build_base(method, partial_k, r, alpha)
    if student_layers:
        drop_layers(m, student_layers)
    return m


def _sample(docs, ids, k):
    """Random k chunks per case (shared by teacher and student so both see identical inputs)."""
    seqs, owner = [], []
    for j, d in enumerate(ids):
        a = docs.arr(d); st = chunk_starts(len(a), 10 ** 9)
        for s in random.sample(st, min(k, len(st))):
            seqs.append(a[s:s + CHUNK]); owner.append(j)
    return seqs, owner


def _embed_seqs(model, seqs, owner, n, device):
    x, mask = cj.collate(seqs, device)
    with cj._autocast(device):
        e = F.normalize(model.embed(x, mask), dim=-1)
    out = torch.zeros(n, e.shape[1], device=device).index_add_(0, torch.tensor(owner, device=device), e)
    return F.normalize(out, dim=-1)


def load_trained(name, method="full", device="cuda"):
    """Rebuild a finished run's model: base M1 + its saved trainable parameters."""
    m = build(method)
    sd = {k: v.float() for k, v in torch.load(f"{CKD}/{name}.pt", map_location="cpu").items()}
    missing = m.load_state_dict(sd, strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys[:5]
    return m.to(device).eval()


def train_kd(student, teacher, epochs=3, bs=4, k=2, n_hard=2, tau=0.05, lr=1e-4, lam=0.5, T=2.0,
             device="cuda", log=print, seed=0, max_steps=None):
    random.seed(seed); torch.manual_seed(seed)
    q, _ = load_split("Train")
    bm = json.load(open(f"{D7}/bm25_top100_train.json"))
    items = [(qid, r["relevant_candidates"]) for qid, r in q.items() if r["relevant_candidates"]]
    docs = Docs(); params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    total = max_steps or epochs * math.ceil(len(items) / bs); warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler("cuda"); student.train(); teacher.eval()
    step, t0, hist, acc = 0, time.time(), [], [0.0, 0.0]
    for ep in range(max(1, epochs)):
        random.shuffle(items)
        for b in range(0, len(items), bs):
            batch = items[b:b + bs]; qids, cand, tgt = [x[0] for x in batch], [], []
            for qid, gold in batch:
                g = set(gold); tgt.append(len(cand)); cand.append(random.choice(gold))
                hard = [c for c in bm[qid][:50] if c not in g and c != qid]
                cand += random.sample(hard, min(n_hard, len(hard)))
            qs, qo = _sample(docs, qids, k); cs, co = _sample(docs, cand, k)
            with torch.no_grad():
                St = _embed_seqs(teacher, qs, qo, len(qids), device) @ _embed_seqs(teacher, cs, co, len(cand), device).T / tau
            Ss = _embed_seqs(student, qs, qo, len(qids), device) @ _embed_seqs(student, cs, co, len(cand), device).T / tau
            bad = torch.zeros_like(Ss, dtype=torch.bool)
            for i, (qid, gold) in enumerate(batch):
                g = set(gold)
                for j, c in enumerate(cand):
                    if j != tgt[i] and (c in g or c == qid):
                        bad[i, j] = True
            Ss, St = Ss.masked_fill(bad, -1e4), St.masked_fill(bad, -1e4)
            ce = F.cross_entropy(Ss, torch.tensor(tgt, device=device))
            kl = F.kl_div(F.log_softmax(Ss / T, -1), F.softmax(St / T, -1), reduction="batchmean") * T * T
            loss = (1 - lam) * ce + lam * kl
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1; acc[0] += ce.item(); acc[1] += kl.item()
            if step % 50 == 0 or step == total:
                n = min(50, step); hist.append({"step": step, "ce": acc[0] / n, "kl": acc[1] / n}); acc = [0.0, 0.0]
                log(f"   step {step}/{total} | ce {hist[-1]['ce']:.4f} kl {hist[-1]['kl']:.4f} | {time.time()-t0:.0f}s")
            if step >= total:
                return hist
    return hist


def run_kd(name, teacher="p7_full", student_layers=(0, 2, 4, 6, 8, 10), epochs=3, bs=4, k=2, n_hard=2,
           tau=0.05, lr=1e-4, lam=0.5, T=2.0, max_chunks=16, seed=0, log=print, device="cuda"):
    t0 = time.time()
    tm = load_trained(teacher, "full", device)
    st = build("full", student_layers=list(student_layers)).to(device)
    ntr = sum(p.numel() for p in st.parameters() if p.requires_grad); ntot = sum(p.numel() for p in st.parameters())
    log(f"=== {name}: KD {teacher} -> {len(student_layers)}-layer student | {ntot/1e6:.1f}M params | epochs {epochs}")
    torch.cuda.reset_peak_memory_stats()
    hist = train_kd(st, tm, epochs, bs, k, n_hard, tau, lr, lam, T, device, log, seed)
    del tm; torch.cuda.empty_cache(); train_min = (time.time() - t0) / 60
    torch.save({n: p.detach().half().cpu() for n, p in st.named_parameters()}, f"{CKD}/{name}.pt")
    docs, emb, enc_s = Docs(), {}, {}
    for split in ("Dev", "Test"):
        _, cids = load_split(split); t = time.time()
        emb[split] = encode(st, docs, cids, max_chunks, device); enc_s[split] = time.time() - t
        np.savez(f"{EMB}/{name}_{split.lower()}.npz", E=emb[split][0], M=emb[split][1])
        log(f"   encoded {split}: {len(cids)} cases in {enc_s[split]:.0f}s")
    dev, _, _ = score_split(name, *emb["Dev"], "Dev", log, device)
    best = max(dev, key=lambda key: dev[key][0]["F1"])
    test, qids, cids = score_split(name, *emb["Test"], "Test", log, device, fixed=best)
    write_submission(test[best][1], name, qids, cids)
    rec = {"name": name, "method": "kd", "teacher": teacher, "student_layers": list(student_layers),
           "cfg": dict(epochs=epochs, bs=bs, k=k, n_hard=n_hard, tau=tau, lr=lr, lam=lam, T=T, max_chunks=max_chunks),
           "trainable_M": round(ntr / 1e6, 3), "total_M": round(ntot / 1e6, 1),
           "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2), "train_min": round(train_min, 1),
           "encode_s": {s: round(v) for s, v in enc_s.items()},
           "dev_best": {"mode": best[0], "alpha": best[1], **dev[best][0]},
           "dev_dense_only": {m: dev[(m, 1.0)][0] for m in ("mean", "maxsim")},
           "dev_table": {f"{m}|{a}": v[0] for (m, a), v in dev.items()}, "hist": hist,
           "rescored": "fixed fusion", "min": round((time.time() - t0) / 60, 1)}
    json.dump(rec, open(f"{OUT}/{name}.json", "w"), indent=1)
    log(f"DONE {name}: Dev F1 {dev[best][0]['F1']:.4f} | {best[0]} alpha {best[1]} | {rec['min']} min")
    return rec


def smoke2(log=print, device="cuda"):
    docs = Docs(); _, cids = load_split("Dev")
    m = build("qlora").to(device); ntr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    train(m, "qlora", bs=2, k=1, n_hard=1, device=device, log=lambda *a: None, max_steps=2)
    E, M = encode(m, docs, cids[:4], max_chunks=2, device=device)
    log(f"smoke qlora   trainable {ntr/1e6:.3f}M | 2 steps ok | emb {E.shape}"); del m; torch.cuda.empty_cache()
    tm = load_trained("p7_full", "full", device)
    st = build("full", student_layers=[0, 2, 4, 6, 8, 10]).to(device)
    n = sum(p.numel() for p in st.parameters())
    train_kd(st, tm, bs=2, k=1, n_hard=1, device=device, log=lambda *a: None, max_steps=2)
    E, M = encode(st, docs, cids[:4], max_chunks=2, device=device)
    log(f"smoke kd      student {n/1e6:.1f}M params | 2 steps ok | emb {E.shape}")
    log("SMOKE2 PASSED")
