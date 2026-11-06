"""
p2_ner.py -- Phase 2 (Legal NER) data pipeline: exact character offsets <-> model tokens.

Why this file exists
  Our Phase-1 models were trained on NORMALISED text (ASCII only, single spaces, no newlines),
  but L-NER is scored on character offsets of the ORIGINAL text, strictly (one wrong character
  = wrong entity). So we normalise each document while keeping, for every normalised character,
  the index of the original character it came from. Tokens are then mapped back exactly.

Pipeline
  normalise_with_map(text)          -> norm, idx   (idx[j] = original index of norm[j])
  encode_doc(text, tk)              -> token ids + each token's ORIGINAL char span (core, no
                                       leading space)
  bio_tags(spans, entities)         -> one BIO tag id per token (O, B-X, I-X; 25 tags)
  tags_to_entities(tags, spans)     -> [start, end, label] in original offsets
  make_windows(n, win, stride)      -> overlapping windows over long documents
  strict_f1(pred, gold)             -> micro P/R/F1 + per-label F1 (exact span + label)
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter, defaultdict

from common import PUNCT_MAP

LABELS = ['APP', 'RESP', 'A.COUNSEL', 'R.COUNSEL', 'JUDGE', 'WIT', 'AUTH', 'COURT',
          'STAT', 'PREC', 'DATE', 'CASENO']
TAGS = ["O"] + [f"{p}-{l}" for l in LABELS for p in ("B", "I")]     # 25 BIO tags
TAG2ID = {t: i for i, t in enumerate(TAGS)}


# ----------------------------------------------------------------------------- normalisation
def _norm_char(c: str) -> str:
    """Same rule as common.normalise, applied to ONE character (whitespace handled by caller)."""
    if c.isascii():
        return c
    s = unicodedata.normalize("NFKD", c.translate(PUNCT_MAP))
    return s.encode("ascii", "ignore").decode("ascii")


def normalise_with_map(text: str):
    """Return (norm, idx): norm == common.normalise(text) and idx[j] is the original index of
    norm[j]. A whitespace run becomes one space mapped to the run's first character."""
    out, idx, pending_space = [], [], None
    for i, ch in enumerate(text):
        if ch.isspace():
            if out and pending_space is None:
                pending_space = i                          # remember where the run started
            continue
        rep = _norm_char(ch)
        if not rep:
            continue                                      # dropped char (e.g. zero-width)
        if rep.isspace() or not rep.strip():
            if out and pending_space is None:
                pending_space = i
            continue
        if pending_space is not None:
            out.append(" "); idx.append(pending_space); pending_space = None
        for r in rep:
            if r.isspace():
                continue
            out.append(r); idx.append(i)
    return "".join(out), idx


# ----------------------------------------------------------------------------- tokens <-> chars
def encode_doc(text: str, tk):
    """Tokenise the normalised document. Returns (ids, spans) where spans[k] = (start, end) of
    token k's CORE (leading space stripped) in ORIGINAL offsets, or None for a pure-space token."""
    norm, idx = normalise_with_map(text)
    enc = tk.encode(norm, add_special_tokens=False)
    spans = []
    for s, e in enc.offsets:
        while s < e and norm[s] == " ":
            s += 1                                        # " court" -> core "court"
        spans.append((idx[s], idx[e - 1] + 1) if s < e else None)
    return enc.ids, spans


def bio_tags(spans, entities):
    """BIO tag id per token. A token is B-X if its core starts exactly at an entity start, I-X if
    it lies inside that entity, else O."""
    begin, inside = {}, {}
    for s, e, lab, *_ in entities:
        begin[s] = lab
        for c in range(s, e):
            inside[c] = (lab, s)
    tags = []
    for sp in spans:
        if sp is None:
            tags.append(0); continue
        s, _ = sp
        if s in begin:
            tags.append(TAG2ID[f"B-{begin[s]}"])
        elif s in inside:
            tags.append(TAG2ID[f"I-{inside[s][0]}"])
        else:
            tags.append(0)
    return tags


def tags_to_entities(tags, spans):
    """Decode BIO tags back to [start, end, label] (original offsets). An I-X that does not
    continue an X entity starts a new one (standard repair)."""
    ents, cur = [], None
    for t, sp in zip(tags, spans):
        if sp is None:
            continue                                      # spaces never break or extend a span
        tag = TAGS[t]
        if tag == "O":
            if cur: ents.append(cur); cur = None
            continue
        p, lab = tag.split("-", 1)
        if p == "B" or cur is None or cur[2] != lab:
            if cur: ents.append(cur)
            cur = [sp[0], sp[1], lab]
        else:
            cur[1] = sp[1]                                # extend the current entity
    if cur:
        ents.append(cur)
    return ents


def make_windows(n_tokens: int, win: int = 512, stride: int = 256):
    """Start offsets of overlapping windows covering [0, n_tokens)."""
    if n_tokens <= win:
        return [0]
    starts = list(range(0, n_tokens - win, stride))
    return starts + [n_tokens - win]


# ----------------------------------------------------------------------------- data + metric
def load_fold(text_path, gold_path=None, fold=""):
    T = json.load(open(text_path))
    G = json.load(open(gold_path)) if gold_path else {}
    return [{"id": str(k), "text": v, "ents": G.get(k, []), "fold": fold} for k, v in T.items()]


def dev_split(docs, n_dev_per_fold=5):
    """Deterministic held-out set: the n docs per fold with the smallest id-hash."""
    h = lambda d: hashlib.md5(d["id"].encode()).hexdigest()
    dev = set()
    for f in sorted({d["fold"] for d in docs}):
        dev |= {d["id"] for d in sorted((d for d in docs if d["fold"] == f), key=h)[:n_dev_per_fold]}
    return [d for d in docs if d["id"] not in dev], [d for d in docs if d["id"] in dev]


def strict_f1(pred: dict, gold: dict):
    """pred/gold: {doc_id: [[s, e, label, ...], ...]}. Exact span AND label must match."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for d in set(gold) | set(pred):
        G = {(e[0], e[1], e[2]) for e in gold.get(d, [])}
        P = {(e[0], e[1], e[2]) for e in pred.get(d, [])}
        for x in P & G: tp[x[2]] += 1
        for x in P - G: fp[x[2]] += 1
        for x in G - P: fn[x[2]] += 1
    def f(a, b, c):
        p = a / max(1, a + b); r = a / max(1, a + c)
        return p, r, 2 * p * r / max(1e-9, p + r)
    P_, R_, F_ = f(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    per = {l: round(f(tp[l], fp[l], fn[l])[2], 4) for l in LABELS}
    return {"precision": P_, "recall": R_, "f1": F_, "per_label_f1": per}


# ----------------------------------------------------------------------------- v2: split glued punctuation
def encode_doc(text: str, tk, split_punct: bool = True):
    """Tokenise the normalised document. Returns (ids, spans) where spans[k] = (start, end) of
    token k's CORE (leading space stripped) in ORIGINAL offsets, or None for a pure-space token.
    split_punct: our BPE glues punctuation runs into one token (e.g. '.[' or '].'), which can
    straddle an entity boundary ('... & Ors.' + '[1]'). Such tokens are re-encoded one character
    at a time so every punctuation mark gets its own token and boundary."""
    norm, idx = normalise_with_map(text)
    enc = tk.encode(norm, add_special_tokens=False)
    ids, spans = [], []
    for tid, (s, e) in zip(enc.ids, enc.offsets):
        c = s
        while c < e and norm[c] == " ":
            c += 1                                        # " court" -> core "court"
        core = norm[c:e]
        if split_punct and len(core) > 1 and not any(ch.isalnum() for ch in core):
            for j, ch in enumerate(core):                 # one token per punctuation mark
                piece = (" " if j == 0 and c > s else "") + ch
                for sub in tk.encode(piece, add_special_tokens=False).ids:
                    ids.append(sub)
                    is_space = tk.decode([sub]).strip() == ""
                    spans.append(None if is_space else (idx[c + j], idx[c + j] + 1))
            continue
        ids.append(tid)
        spans.append((idx[c], idx[e - 1] + 1) if c < e else None)
    return ids, spans


# ----------------------------------------------------------------------------- v3: constrained Viterbi decoding
def _allowed():
    """allowed[i, j]: tag j may follow tag i (BIO rule: I-X only after B-X or I-X)."""
    import numpy as np
    K = len(TAGS); A = np.ones((K, K), dtype=bool)
    for j, t in enumerate(TAGS):
        if t.startswith("I-"):
            lab = t[2:]
            A[:, j] = False
            A[TAG2ID[f"B-{lab}"], j] = A[TAG2ID[f"I-{lab}"], j] = True
    return A


def viterbi_bio(logp, spans):
    """Most probable VALID BIO sequence given per-token log-probs (T, K). Pure-space tokens
    (span None) are skipped, so they can neither break nor start an entity."""
    import numpy as np
    keep = [i for i, sp in enumerate(spans) if sp is not None]
    out = [0] * len(spans)
    if not keep:
        return out
    lp = np.asarray(logp)[keep]                                   # (n, K)
    A = _allowed(); NEG = -1e9
    trans = np.where(A, 0.0, NEG)                                 # (K, K)
    start = np.array([NEG if t.startswith("I-") else 0.0 for t in TAGS])
    n, K = lp.shape
    score = start + lp[0]; back = np.zeros((n, K), dtype=np.int32)
    for t in range(1, n):
        cand = score[:, None] + trans                             # prev x cur
        back[t] = cand.argmax(0)
        score = cand.max(0) + lp[t]
    best = [int(score.argmax())]
    for t in range(n - 1, 0, -1):
        best.append(int(back[t, best[-1]]))
    for i, tag in zip(keep, reversed(best)):
        out[i] = tag
    return out


# ----------------------------------------------------------------------------- v4: word-consistent Viterbi
def word_groups(spans, text):
    """Group sub-word tokens into words: token k continues the previous token's word if they touch
    (no space) and both sides are letters/digits (e.g. 'VENKAT' + 'ACH' + 'ALA')."""
    groups, prev = [], None
    for k, sp in enumerate(spans):
        if sp is None:
            continue
        if (prev is not None and sp[0] == prev[1] and text[prev[1] - 1].isalnum()
                and text[sp[0]].isalnum()):
            groups[-1].append(k)
        else:
            groups.append([k])
        prev = sp
    return groups


def _cont():
    """Tag that a word's continuation pieces must take: B-X/I-X -> I-X, O -> O."""
    return [0 if t == "O" else TAG2ID["I-" + t[2:]] for t in TAGS]


def viterbi_words(logp, spans, text):
    """Constrained Viterbi over WORDS: every sub-word piece of a word gets a consistent tag
    (first piece B-X or I-X, the rest I-X; or all O). A word's score for tag t is the exact joint
    log-prob of that consistent assignment, so a name can no longer be cut inside a word."""
    import numpy as np
    lp = np.asarray(logp)
    groups = word_groups(spans, text)
    out = [0] * len(spans)
    if not groups:
        return out
    cont = np.array(_cont())
    W = np.stack([lp[g[0]] + (lp[g[1:]][:, cont].sum(0) if len(g) > 1 else 0.0) for g in groups])
    A = _allowed(); NEG = -1e9
    trans = np.where(A, 0.0, NEG)
    start = np.array([NEG if t.startswith("I-") else 0.0 for t in TAGS])
    n = len(groups)
    score = start + W[0]; back = np.zeros((n, len(TAGS)), dtype=np.int32)
    for t in range(1, n):
        cand = score[:, None] + trans
        back[t] = cand.argmax(0); score = cand.max(0) + W[t]
    best = [int(score.argmax())]
    for t in range(n - 1, 0, -1):
        best.append(int(back[t, best[-1]]))
    for g, tag in zip(groups, reversed(best)):
        out[g[0]] = tag
        for k in g[1:]:
            out[k] = int(cont[tag])
    return out


# ----------------------------------------------------------------------------- v5: strict MACRO-F1 (official metric)
def strict_f1(pred: dict, gold: dict):
    """Exact span AND label. Returns micro P/R/F1, per-label F1, and MACRO-F1 (the official
    Phase-2 metric) over the labels that occur in the gold data (and over all 12, for reference)."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for d in set(gold) | set(pred):
        G = {(e[0], e[1], e[2]) for e in gold.get(d, [])}
        P = {(e[0], e[1], e[2]) for e in pred.get(d, [])}
        for x in P & G: tp[x[2]] += 1
        for x in P - G: fp[x[2]] += 1
        for x in G - P: fn[x[2]] += 1
    def f(a, b, c):
        p = a / max(1, a + b); r = a / max(1, a + c)
        return p, r, 2 * p * r / max(1e-9, p + r)
    P_, R_, F_ = f(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    per = {l: round(f(tp[l], fp[l], fn[l])[2], 4) for l in LABELS}
    present = [l for l in LABELS if tp[l] + fn[l] > 0]
    macro = sum(per[l] for l in present) / max(1, len(present))
    return {"precision": P_, "recall": R_, "f1": F_, "macro_f1": macro,
            "macro_f1_all12": sum(per.values()) / len(LABELS), "per_label_f1": per,
            "labels_in_gold": present}
