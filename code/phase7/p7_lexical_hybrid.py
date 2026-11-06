"""p7_lexical_hybrid.py -- Phase 7 lexical side + final hybrid fusion, exported from the Kaggle notebook.

These cells ran in ONE notebook kernel, in this order, after p7_pcr.py / p7_run.py experiments:
  P7-1   BM25 baseline (letters-only), evaluator, shortlists        -> Dev F1@7 0.394 (k1=20, b=1)
  P7-2c  saved BM25 score matrices + BPE token cache for the retriever
  P7-5B  BM25 with numbers + bigrams (0.440), citation matching (no signal: citations are scrubbed),
         precedent-popularity prior (weight 0 on Train), dense-ensemble fusion
  P7-5C  1-3-gram BM25 tuned on Train (0.4447), PRF (beta=0 on Train), every technique re-fused -> *_v2
Final: Dev F1@7 0.448 (hybrid_v2 = 1-3-gram BM25 + ensemble of p7_full / p7_kd_m1x6 / p7_qlora dense scores).

Notebook magics (`!cmd`) are kept as comments. The canonical, runnable record is phase7_notebook.ipynb.
"""


# ====================================================================================================
# P7-1: BM25 baseline, exact Micro-F1@7 evaluator, top-100 shortlists, submission writer   [notebook cell 331]
# ====================================================================================================
# ---- P7-1: BM25 baseline + exact Micro-F1@7 evaluator + top-100 shortlists + submission ----
import json, os, time, zipfile, numpy as np
from sklearn.feature_extraction.text import CountVectorizer

R, OUT = "/kaggle/working/data_p7/Phase_7_PCR", "/kaggle/working/data_p7"
SUB = "/kaggle/working/submissions_p7"; os.makedirs(SUB, exist_ok=True)

def load(split):
    s = split.lower()
    return json.load(open(f"{R}/{split}/{s}.json")), json.load(open(f"{R}/{split}/{s}_candidates.json"))

def evaluate(rank, queries):
    """Exact competition metric: micro P/R/F1 over the top 7 of every query (gold neither truncated nor padded)."""
    tp = sum(len(set(rank[q][:7]) & set(r["relevant_candidates"])) for q, r in queries.items())
    n_gold = sum(len(r["relevant_candidates"]) for r in queries.values())
    P, Rc = tp / (7 * len(queries)), tp / n_gold
    return {"F1": 2 * P * Rc / (P + Rc + 1e-12), "P": P, "R": Rc}

def bm25_weights(tf, k1, b):
    """Document-side BM25 weights: idf * tf(k1+1) / (tf + k1(1 - b + b*dl/avgdl)), on a sparse doc x term matrix."""
    tf = tf.tocsr().astype(np.float32, copy=True)
    dl = np.asarray(tf.sum(1)).ravel(); N = tf.shape[0]
    df = np.bincount(tf.indices, minlength=tf.shape[1])
    idf = np.log1p((N - df + 0.5) / (df + 0.5)).astype(np.float32)
    rows = np.repeat(np.arange(N), np.diff(tf.indptr))
    v = tf.data
    tf.data = idf[tf.indices] * v * (k1 + 1) / (v + k1 * (1 - b + b * dl[rows] / dl.mean()))
    return tf

# Word tokens (letters only, length >= 2), English stop words removed; <NAME> placeholders drop out
vec_kw = dict(lowercase=True, token_pattern=r"(?u)\b[a-z][a-z]+\b", stop_words="english", dtype=np.float32)

cache = {}
def prepare(split):
    # Count matrices for one split: vocabulary from its own pool; queries mapped into it
    q, c = load(split)
    cids, qids = list(c), list(q)
    cv = CountVectorizer(**vec_kw)
    C = cv.fit_transform(" ".join(c[i]["text"]) for i in cids)
    Q = cv.transform(" ".join(q[i]["text"]) for i in qids)
    cache[split] = (q, qids, cids, C, Q)

def bm25_rank(split, k1=1.2, b=0.75, qmode="log", topn=100):
    q, qids, cids, C, Q = cache[split]
    W = bm25_weights(C, k1, b)
    Qw = Q.copy(); Qw.data = np.log1p(Qw.data) if qmode == "log" else np.ones_like(Qw.data)
    S = (Qw @ W.T).toarray()                                   # queries x candidates
    pos = {cid: j for j, cid in enumerate(cids)}
    for i, qid in enumerate(qids):                             # exclude the query itself from its own pool
        if qid in pos: S[i, pos[qid]] = -np.inf
    top = np.argsort(-S, axis=1)[:, :topn]
    return {qid: [cids[j] for j in top[i]] for i, qid in enumerate(qids)}, S

t = time.time()
for split in ("Train", "Dev", "Test"): prepare(split)
print(f"tokenised all splits in {time.time()-t:.0f}s")

# Is the query ever its own gold? (sanity check for the self-exclusion)
q, *_ = cache["Train"]
print("queries listing themselves as gold:", sum(k in r["relevant_candidates"] for k, r in q.items()))

# Tune on Train (no training happens, so its labels are a large, clean validation set)
res = []
for qmode in ("log", "binary"):
    for k1 in (0.9, 1.2, 1.6, 2.0):
        for b in (0.5, 0.75, 0.9):
            rank, _ = bm25_rank("Train", k1, b, qmode, topn=7)
            res.append((evaluate(rank, cache["Train"][0])["F1"], qmode, k1, b))
res.sort(reverse=True)
print("\nTrain grid (top 5):", *[f"F1 {f:.4f} | q={m} k1={k} b={bb}" for f, m, k, bb in res[:5]], sep="\n  ")
_, qmode, k1, b = res[0]

# Confirm on Dev with the chosen setting; also recall of the top-100 shortlist (ceiling for re-ranking)
for split in ("Train", "Dev"):
    rank, _ = bm25_rank(split, k1, b, qmode, topn=100)
    m = evaluate(rank, cache[split][0])
    qs = cache[split][0]
    rec = {n: np.mean([len(set(rank[k][:n]) & set(r["relevant_candidates"])) / max(1, len(r["relevant_candidates"]))
                       for k, r in qs.items() if r["relevant_candidates"]]) for n in (7, 20, 50, 100)}
    print(f"{split}: F1@7 {m['F1']:.4f}  P@7 {m['P']:.4f}  R@7 {m['R']:.4f} | "
          f"gold recall in top-n: " + "  ".join(f"@{n} {v:.2f}" for n, v in rec.items()))

# Save top-100 shortlists for every split (hard negatives + re-ranking input)
for split in ("Train", "Dev", "Test"):
    rank, _ = bm25_rank(split, k1, b, qmode, topn=100)
    json.dump(rank, open(f"{OUT}/bm25_top100_{split.lower()}.json", "w"))
json.dump({"k1": k1, "b": b, "qmode": qmode}, open(f"{OUT}/bm25_cfg.json", "w"))

def write_submission(rank, name, split="Test"):
    """prediction.json: exactly 7 unique in-pool ids per query, best first; zipped with the file at the root."""
    q, qids, cids, *_ = cache[split]; pool = set(cids)
    pred = {k: rank[k][:7] for k in qids}
    assert all(len(v) == 7 and len(set(v)) == 7 and set(v) <= pool and k not in v for k, v in pred.items())
    d = f"{SUB}/p7_{name}"; os.makedirs(d, exist_ok=True)
    json.dump(pred, open(f"{d}/prediction.json", "w"), indent=1)
    with zipfile.ZipFile(f"{d}/p7_{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{d}/prediction.json", "prediction.json")
    print(f"submission: {d}/p7_{name}.zip ({len(pred)} queries)")

rank, _ = bm25_rank("Test", k1, b, qmode, topn=7)
write_submission(rank, "bm25")


# ====================================================================================================
# P7-2c: final BM25 (letters-only) score matrices + tokenisation of every case with the A1 BPE + smoke test   [notebook cell 339]
# ====================================================================================================
# ---- P7-2c: final BM25 (k1 sweep) + full score matrices, tokenise all cases with our BPE, smoke test ----
import json, numpy as np, subprocess, time
from tokenizers import Tokenizer

D7 = "/kaggle/working/data_p7"; R7 = f"{D7}/Phase_7_PCR"

# 1) BM25: k1 beyond 10 at b = 1.0, choose on Train, report Dev
res = sorted(((evaluate(bm25_rank("Train", k, 1.0, "log", topn=7)[0], cache["Train"][0])["F1"], k)
              for k in (10.0, 15.0, 20.0, 30.0, 50.0)), reverse=True)
print("Train:", "  ".join(f"k1={k}: {f:.4f}" for f, k in sorted(res, key=lambda x: x[1])))
k1 = res[0][1]
print(f"final BM25 k1={k1} b=1.0 -> Dev {evaluate(bm25_rank('Dev', k1, 1.0, 'log', topn=7)[0], cache['Dev'][0])}")

# Save shortlists, full score matrices (for fusion), config and the BM25 submission with the final setting
for split in ("Train", "Dev", "Test"):
    rank, S = bm25_rank(split, k1, 1.0, "log", topn=100)
    json.dump(rank, open(f"{D7}/bm25_top100_{split.lower()}.json", "w"))
    q, qids, cids, *_ = cache[split]
    np.savez(f"{D7}/bm25_scores_{split.lower()}.npz", S=np.where(np.isinf(S), -1e9, S).astype(np.float32),
             qids=np.array(qids), cids=np.array(cids))
json.dump({"k1": k1, "b": 1.0, "qmode": "log"}, open(f"{D7}/bm25_cfg.json", "w"))
write_submission(bm25_rank("Test", k1, 1.0, "log", topn=7)[0], "bm25")

# 2) Tokenise every Phase-7 case once with the A1 BPE (candidate pools contain all queries too)
TK = Tokenizer.from_file("/kaggle/working/tokenizer/hf_a1_bpe_v2_32k.json")
texts = {}
for split in ("Train", "Dev", "Test"):
    for d, r in json.load(open(f"{R7}/{split}/{split.lower()}_candidates.json")).items():
        texts.setdefault(d, " ".join(r["text"]))
names = sorted(texts); arrs = []; t = time.time()
for i in range(0, len(names), 200):                                    # batches keep RAM in check
    for e in TK.encode_batch([texts[n] for n in names[i:i + 200]]):
        arrs.append(np.array(e.ids[:131072], dtype=np.uint16))         # cap at 128k tokens (256 chunks)
lens = np.array([len(a) for a in arrs])
import os; os.makedirs(f"{D7}/tok", exist_ok=True)
np.savez(f"{D7}/tok/docs.npz", ids=np.concatenate(arrs), offsets=np.concatenate([[0], np.cumsum(lens)]),
         names=np.array(names))
print(f"\ntokenised {len(names)} cases in {time.time()-t:.0f}s | tokens median {int(np.median(lens)):,} "
      f"p90 {int(np.percentile(lens, 90)):,} | total {lens.sum()/1e6:.1f}M | "
      f"cases > 16 chunks: {100*np.mean(lens > 16*512):.0f}%")

# 3) Smoke test on GPU 0: every method builds, trains 2 steps, encodes (catches wiring errors before real runs)
print(subprocess.run("cd /kaggle/working/src && CUDA_VISIBLE_DEVICES=0 python3 p7_run.py --smoke 2>&1 | tail -n 25",
                     shell=True, capture_output=True, text=True).stdout)


# ====================================================================================================
# P7-5B: n-gram BM25 variants, citation-match and popularity-prior tests, dense ensemble -> hybrid   [notebook cell 353]
# ====================================================================================================
# ---- P7-5B: stronger lexical scoring + citation match + popularity prior + dense ensemble -> hybrid ----
import sys, json, re, glob, time, numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
sys.path.insert(0, "/kaggle/working/src"); import p7_pcr as P

S7 = "/kaggle/working/stats"
# All three splits: queries, ordered pool ids, pool texts (queries are inside their own pool)
SP = {}
for s in ("Train", "Dev", "Test"):
    q, cids = P.load_split(s)
    c = json.load(open(f"{P.R7}/{s}/{s.lower()}_candidates.json"))
    pos = {x: j for j, x in enumerate(cids)}
    SP[s] = dict(q=q, qids=list(q), cids=cids, qrows=np.array([pos[x] for x in q]),
                 text=[" ".join(c[x]["text"]) for x in cids])
    del c

def bm25W(tf, k1, b=1.0):
    """Document-side BM25 weights on a sparse doc x term matrix."""
    tf = tf.tocsr().astype(np.float32, copy=True)
    dl = np.asarray(tf.sum(1)).ravel(); N = tf.shape[0]
    df = np.bincount(tf.indices, minlength=tf.shape[1])
    idf = np.log1p((N - df + 0.5) / (df + 0.5)).astype(np.float32)
    rows = np.repeat(np.arange(N), np.diff(tf.indptr)); v = tf.data
    tf.data = idf[tf.indices] * v * (k1 + 1) / (v + k1 * (1 - b + b * dl[rows] / dl.mean()))
    return tf

def self_nan(S, s):
    """The query's own column -> NaN (excluded from z-scores and from ranking)."""
    S = S.astype(np.float32); S[np.arange(len(SP[s]["qids"])), SP[s]["qrows"]] = np.nan; return S

def f1(S, s):
    d = SP[s]; return P.evaluate(P.top7(np.nan_to_num(S, nan=-np.inf), d["qids"], d["cids"]), d["q"])

# ---------- 1) BM25 variants (queries are rows of the pool matrix, so each split is vectorised once)
VARIANTS = {
    "letters":       lambda: CountVectorizer(token_pattern=r"(?u)\b[a-z][a-z]+\b", stop_words="english"),
    "alnum":         lambda: HashingVectorizer(token_pattern=r"(?u)\b(?:[a-z]{2,}|\d+)\b", stop_words="english",
                                               n_features=2**22, alternate_sign=False, norm=None),
    "alnum+bigrams": lambda: HashingVectorizer(token_pattern=r"(?u)\b(?:[a-z]{2,}|\d+)\b", stop_words="english",
                                               ngram_range=(1, 2), n_features=2**23, alternate_sign=False, norm=None),
}
BM = {}
for vname, mk in VARIANTS.items():
    t = time.time(); X = {}
    for s in SP:
        v = mk(); X[s] = (v.fit_transform if hasattr(v, "fit_transform") and vname == "letters" else v.transform)(SP[s]["text"])
    best = None
    for k1 in (10.0, 20.0, 40.0):                                       # tune k1 on Train
        Q = X["Train"][SP["Train"]["qrows"]].astype(np.float32); Q.data = np.log1p(Q.data)
        St = self_nan((Q @ bm25W(X["Train"], k1).T).toarray(), "Train")
        m = f1(St, "Train")["F1"]
        if best is None or m > best[0]: best = (m, k1)
    k1 = best[1]; BM[vname] = {}
    for s in SP:
        Q = X[s][SP[s]["qrows"]].astype(np.float32); Q.data = np.log1p(Q.data)
        BM[vname][s] = self_nan((Q @ bm25W(X[s], k1).T).toarray(), s)
    print(f"BM25 {vname:<14} k1={k1:<5} Train F1 {best[0]:.4f} | Dev F1 {f1(BM[vname]['Dev'], 'Dev')['F1']:.4f}"
          f"  ({time.time()-t:.0f}s)")
    del X
lex_name = max(BM, key=lambda n: f1(BM[n]["Train"], "Train")["F1"])
print("-> lexical base:", lex_name)

# ---------- 2) Citation matching: candidate's own citation (from its header) found in the query text
CIT = re.compile(r"\(\d{4}\)\s*\d+\s*SCC\s*\d+|\d{4}\s*AIR\s*\d+|AIR\s*\d{4}\s*[A-Z][A-Za-z]*\s*\d+|"
                 r"\d{4}\s*SCR\s*\(\d+\)\s*\d+|\[\d{4}\]\s*\d+\s*SCR\s*\d+")
norm = lambda x: re.sub(r"\s+", "", x).upper()
def own_cites(text):
    m = re.search(r"CITATION:(.*?)(?:CITATOR|ACT:|HEADNOTE|$)", text[:1500])
    return {norm(x) for x in CIT.findall(m.group(1))} if m else set()
CITE = {}
for s in SP:
    d = SP[s]; own = [own_cites(t) for t in d["text"]]
    inq = [{norm(x) for x in CIT.findall(d["text"][r])} - own[r] for r in d["qrows"]]
    M = np.zeros((len(d["qids"]), len(d["cids"])), np.float32)
    for i, cs in enumerate(inq):
        if cs:
            for j, oc in enumerate(own):
                if oc & cs: M[i, j] = 1.0
    CITE[s] = M
    if s != "Test":
        gold = [(i, j) for i, qid in enumerate(d["qids"]) for j, c in enumerate(d["cids"])
                if c in set(d["q"][qid]["relevant_candidates"])]
        hits = sum(M[i, j] for i, j in gold); tot = M.sum()
        print(f"citations {s}: cands with own citation {100*np.mean([bool(o) for o in own]):.0f}% | "
              f"gold pairs matched {100*hits/len(gold):.1f}% | match precision {100*hits/max(tot,1):.1f}% ({int(tot)} matches)")

# ---------- 3) Popularity prior: how many TRAIN queries cite each candidate (leave-one-out on Train)
cnt = Counter(g for r in SP["Train"]["q"].values() for g in r["relevant_candidates"])
PRI = {}
for s in SP:
    d = SP[s]; base = np.array([cnt.get(c, 0) for c in d["cids"]], np.float32)
    M = np.tile(base, (len(d["qids"]), 1))
    if s == "Train":                                                    # remove each query's own labels
        pos = {c: j for j, c in enumerate(d["cids"])}
        for i, qid in enumerate(d["qids"]):
            for g in d["q"][qid]["relevant_candidates"]: M[i, pos[g]] -= 1
    PRI[s] = np.log1p(M)

# ---------- 4) Lexical combination, weights tuned on Train
Z = lambda S: P.znorm(S)
best = (0, 0, 0)
for wp in (0, 0.25, 0.5, 1, 2):
    for wc in (0, 1, 2, 4, 8):
        m = f1(Z(BM[lex_name]["Train"]) + wp * PRI["Train"] + wc * CITE["Train"], "Train")["F1"]
        if m > best[0]: best = (m, wp, wc)
_, wp, wc = best
LEX = {s: Z(BM[lex_name][s]) + wp * PRI[s] + wc * CITE[s] for s in SP}
print(f"lexical combo (Train-tuned): prior w={wp} cite w={wc} | Train {best[0]:.4f} | Dev {f1(LEX['Dev'], 'Dev')}")

# ---------- 5) Dense ensemble from saved embeddings (each run in its own Dev-best pooling), fused with LEX
runs = sorted(((json.load(open(f))["dev_best"]["F1"], json.load(open(f))) for f in glob.glob(f"{S7}/p7_runs/*.json")
               if "zeroshot" not in f), key=lambda x: -x[0])
DZ = {}
for _, r in runs:
    for s in ("Dev", "Test"):
        z = np.load(f"{P.EMB}/{r['name']}_{s.lower()}.npz")
        Sd = P.dense_scores(z["E"], z["M"], list(SP[s]["qrows"]), r["dev_best"]["mode"], "cpu")
        DZ.setdefault(r["name"], {})[s] = Z(self_nan(Sd, s))
results = []
for top in (1, 3, 5, len(runs)):                                       # ensemble of the best `top` models
    names = [r["name"] for _, r in runs[:top]]
    D = {s: np.nanmean([DZ[n][s] for n in names], 0) for s in ("Dev", "Test")}
    for a in P.ALPHAS:
        results.append((f1((1 - a) * LEX["Dev"] + a * Z(D["Dev"]), "Dev")["F1"], top, a))
results.sort(reverse=True)
print("\nbest hybrids on Dev:", *[f"F1 {m:.4f} | top-{t} models | α={a}" for m, t, a in results[:5]], sep="\n  ")

# ---------- 6) Write the hybrid test submission with the best Dev setting
m, top, a = results[0]; names = [r["name"] for _, r in runs[:top]]
Dt = np.nanmean([DZ[n]["Test"] for n in names], 0)
d = SP["Test"]; rank = P.top7(np.nan_to_num((1 - a) * LEX["Test"] + a * Z(Dt), nan=-np.inf), d["qids"], d["cids"])
P.write_submission(rank, "hybrid", d["qids"], d["cids"])
json.dump({"lexical": lex_name, "prior_w": wp, "cite_w": wc, "dense_models": names, "alpha": a,
           "dev": f1((1 - a) * LEX["Dev"] + a * Z(np.nanmean([DZ[n]["Dev"] for n in names], 0)), "Dev")},
          open(f"{S7}/p7_hybrid.json", "w"), indent=1)
print(f"\nwrote submissions_p7/p7_hybrid/  (Dev F1@7 {m:.4f})")


# ====================================================================================================
# P7-5C: tuned 1-3-gram BM25, pseudo-relevance feedback test, re-fusion of every technique (v2 submissions)   [notebook cell 355]
# ====================================================================================================
# ---- P7-5C: tune the n-gram BM25 + neighbour PRF on Train; re-fuse all techniques with it ----
import warnings, json, glob, time, numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
warnings.filterwarnings("ignore", message="Mean of empty slice")       # the query's own (NaN) column
S7 = "/kaggle/working/stats"; Z = P.znorm

def vectorise(ngram):
    v = HashingVectorizer(token_pattern=r"(?u)\b(?:[a-z]{2,}|\d+)\b", stop_words="english",
                          ngram_range=(1, ngram), n_features=2**23, alternate_sign=False, norm=None)
    return {s: v.transform(SP[s]["text"]).astype(np.float32).tocsr() for s in SP}

def qmat(X, rows):
    Q = X[rows].copy(); Q.data = np.log1p(Q.data); return Q

# ---------- 1) n-gram order, k1, b on Train
grid = []
for ng in (2, 3):
    t = time.time(); X = vectorise(ng)
    for k1 in (5.0, 7.0, 10.0):
        for b in (0.9, 1.0):
            St = self_nan((qmat(X["Train"], SP["Train"]["qrows"]) @ bm25W(X["Train"], k1, b).T).toarray(), "Train")
            grid.append((f1(St, "Train")["F1"], ng, k1, b))
    print(f"ngram 1-{ng} done in {time.time()-t:.0f}s"); del X
grid.sort(reverse=True)
print("Train top 4:", *[f"F1 {m:.4f} | 1-{n}gram k1={k} b={b}" for m, n, k, b in grid[:4]], sep="\n  ")
_, ng, k1, b = grid[0]

# Final lexical scores per split + the pool x pool similarity used for feedback
X = vectorise(ng); BASE, DD = {}, {}
for s in SP:
    W = bm25W(X[s], k1, b)
    BASE[s] = Z(self_nan((qmat(X[s], SP[s]["qrows"]) @ W.T).toarray(), s))
    dd = (qmat(X[s], np.arange(X[s].shape[0])) @ W.T).toarray().astype(np.float32)
    np.fill_diagonal(dd, np.nan); DD[s] = Z(dd)
del X
print(f"lexical 1-{ng}gram k1={k1} b={b}: Dev {f1(BASE['Dev'], 'Dev')['F1']:.4f}")

# ---------- 2) Neighbour smoothing (pseudo-relevance feedback), tuned on Train
def prf(s, beta, k):
    B = BASE[s]
    if beta == 0: return B
    top = np.argsort(-np.nan_to_num(B, nan=-np.inf), axis=1)[:, :k]           # each query's top-k candidates
    return B + beta * np.nanmean(DD[s][top], axis=1)                            # (nq, k, n) -> (nq, n)
best = max(((f1(prf("Train", be, k), "Train")["F1"], be, k) for be in (0, 0.1, 0.2, 0.3, 0.5, 0.8) for k in (3, 5, 10)))
_, beta, kprf = best
LEX2 = {s: prf(s, beta, kprf) for s in SP}
print(f"PRF beta={beta} k={kprf} (Train {best[0]:.4f}) -> Dev {f1(LEX2['Dev'], 'Dev')}")

# ---------- 3) Re-fuse every trained technique with the new lexical score (alpha on Dev); write *_v2 submissions
def dense_z(name, mode):
    if name not in DZ:                                                          # runs finished after P7-5B
        DZ[name] = {}
        for s in ("Dev", "Test"):
            z = np.load(f"{P.EMB}/{name}_{s.lower()}.npz")
            DZ[name][s] = Z(self_nan(P.dense_scores(z["E"], z["M"], list(SP[s]["qrows"]), mode, "cpu"), s))
    return DZ[name]

recs = {}
for f in sorted(glob.glob(f"{S7}/p7_runs/*.json")):
    r = json.load(open(f)); n = r["name"]; D = dense_z(n, r["dev_best"]["mode"])
    m, a = max((f1((1 - a) * LEX2["Dev"] + a * D["Dev"], "Dev")["F1"], a) for a in P.ALPHAS)
    d = SP["Test"]
    P.write_submission(P.top7(np.nan_to_num((1 - a) * LEX2["Test"] + a * D["Test"], nan=-np.inf), d["qids"], d["cids"]),
                       f"{n}_v2", d["qids"], d["cids"])
    recs[n] = {"dev_F1_v2": m, "alpha": a, "dev_F1_v1": r["dev_best"]["F1"]}
    print(f"{n:<16} v1 {r['dev_best']['F1']:.4f} -> v2 {m:.4f}  (α={a})")

# ---------- 4) Hybrid v2: ensemble of the best models' dense scores + new lexical
order = sorted(recs, key=lambda n: -recs[n]["dev_F1_v2"]); res = []
for top in (1, 3, 5):
    Dd = np.nanmean([DZ[n]["Dev"] for n in order[:top]], 0)
    res += [(f1((1 - a) * LEX2["Dev"] + a * Z(Dd), "Dev")["F1"], top, a) for a in P.ALPHAS]
m, top, a = max(res)
Dt = np.nanmean([DZ[n]["Test"] for n in order[:top]], 0); d = SP["Test"]
P.write_submission(P.top7(np.nan_to_num((1 - a) * LEX2["Test"] + a * Z(Dt), nan=-np.inf), d["qids"], d["cids"]),
                   "hybrid_v2", d["qids"], d["cids"])
json.dump({"lexical": {"ngram": ng, "k1": k1, "b": b, "prf_beta": beta, "prf_k": kprf},
           "per_run": recs, "hybrid": {"models": order[:top], "alpha": a, "dev_F1": m}},
          open(f"{S7}/p7_v2.json", "w"), indent=1)
print(f"\nhybrid_v2: top-{top} models α={a} -> Dev F1@7 {m:.4f}  (written to submissions_p7/p7_hybrid_v2/)")


# ====================================================================================================
# v2 re-fusion for runs that finished after P7-5C   [notebook cell 358]
# ====================================================================================================
# ---- v2 for p7_qlora_k3e5: re-fuse with the trigram lexical score, refresh hybrid_v2, back up ----
import json, glob, subprocess, numpy as np
W, S7 = "/kaggle/working", "/kaggle/working/stats"
n = "p7_qlora_k3e5"
r = json.load(open(f"{S7}/p7_runs/{n}.json"))                       # fails here if the run didn't finish
print(f"{n}: v1 Dev F1@7 {r['dev_best']['F1']:.4f} | dense-only mean {r['dev_dense_only']['mean']['F1']:.4f} "
      f"maxsim {r['dev_dense_only']['maxsim']['F1']:.4f} | train {r['train_min']} min")

# Re-fuse with the new lexical score, alpha on Dev, write its v2 test submission
D = dense_z(n, r["dev_best"]["mode"])
m, a = max((f1((1 - a) * LEX2["Dev"] + a * D["Dev"], "Dev")["F1"], a) for a in P.ALPHAS)
d = SP["Test"]
P.write_submission(P.top7(np.nan_to_num((1 - a) * LEX2["Test"] + a * D["Test"], nan=-np.inf), d["qids"], d["cids"]),
                   f"{n}_v2", d["qids"], d["cids"])
recs[n] = {"dev_F1_v2": m, "alpha": a, "dev_F1_v1": r["dev_best"]["F1"]}
print(f"{n}: v2 Dev F1@7 {m:.4f} (α={a})")

# Rebuild hybrid_v2 with this model included; keep whichever ensemble is best on Dev
order = sorted(recs, key=lambda x: -recs[x]["dev_F1_v2"]); res = []
for top in (1, 3, 5):
    Dd = np.nanmean([DZ[x]["Dev"] for x in order[:top]], 0)
    res += [(f1((1 - a) * LEX2["Dev"] + a * Z(Dd), "Dev")["F1"], top, a) for a in P.ALPHAS]
mh, top, ah = max(res)
Dt = np.nanmean([DZ[x]["Test"] for x in order[:top]], 0)
P.write_submission(P.top7(np.nan_to_num((1 - ah) * LEX2["Test"] + ah * Z(Dt), nan=-np.inf), d["qids"], d["cids"]),
                   "hybrid_v2", d["qids"], d["cids"])
v2 = json.load(open(f"{S7}/p7_v2.json"))
v2["per_run"] = recs; v2["hybrid"] = {"models": order[:top], "alpha": ah, "dev_F1": mh}
json.dump(v2, open(f"{S7}/p7_v2.json", "w"), indent=1)
print(f"hybrid_v2: top-{top} {order[:top]} α={ah} -> Dev F1@7 {mh:.4f}")

# Back up everything (v2 submissions, p7_v2.json, this run)
print(subprocess.run(f"python3 {W}/hf_backup.py {n} 2>&1 | grep -E 'backup ok|retry'",
                     shell=True, capture_output=True, text=True).stdout)
