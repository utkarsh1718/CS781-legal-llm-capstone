"""
p2_model.py -- Phase 2 token tagger on top of ANY Phase-1 backbone (M1 dense, M2 MLA+MoE,
M3 encoder-decoder, M4 copy head) + windowed training / prediction.

  build_tagger(ckpt, bidirectional)  -> TokenTagger: backbone hidden states -> Linear -> 25 BIO tags
      * M1/M2/M4 decoders: tok_emb -> blocks -> norm_f (the LM head is not used)
      * M3: its bidirectional ENCODER only
      * bidirectional=True switches the decoders' attention from causal to full (every token sees
        its right context too, which matters for entity boundaries; cf. LLM2Vec, 2024)
  make_examples(docs, tk)            -> (ids, tags) windows for training
  predict_docs(model, docs, tk)      -> {doc_id: [[start, end, label], ...]} in original offsets
  train_tagger(...)                  -> fine-tunes, keeps the best dev-F1 weights
"""
from __future__ import annotations

import contextlib
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import p2_ner as ner

_SDPA = F.scaled_dot_product_attention


@contextlib.contextmanager
def full_attention():
    """Inside this block every scaled_dot_product_attention call ignores is_causal, so causal
    decoders attend in both directions. All our attention modules call F.scaled_dot_product_attention
    at run time, so swapping the function on the module is enough."""
    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
        return _SDPA(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False, **kw)
    F.scaled_dot_product_attention = sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = _SDPA


class TokenTagger(nn.Module):
    def __init__(self, backbone, arch: str, n_tags: int = len(ner.TAGS), dropout: float = 0.1,
                 bidirectional: bool = False):
        super().__init__()
        self.backbone, self.arch, self.bidirectional = backbone, arch, bidirectional
        d = backbone.cfg.d_model
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d, n_tags)
        nn.init.normal_(self.head.weight, 0.0, 0.02); nn.init.zeros_(self.head.bias)

    def hidden(self, ids):
        bb = self.backbone
        if self.arch == "encdec":
            return bb.encode(ids)                                   # M3: bidirectional encoder
        x = bb.tok_emb(ids)                                         # M1 / M2 / M4 decoders
        for blk in bb.blocks:
            x = blk(x, bb.rope_cos, bb.rope_sin)
        return bb.norm_f(x)

    def forward(self, ids):
        ctx = full_attention() if (self.bidirectional and self.arch != "encdec") else contextlib.nullcontext()
        with ctx:
            h = self.hidden(ids)
        return self.head(self.drop(h))                               # (B, T, n_tags)


def build_tagger(ckpt_path: str, bidirectional: bool = False, dropout: float = 0.1):
    """Load a Phase-1 best.pt and wrap it as a tagger (backbone weights in fp32)."""
    from train import build_model
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    bb = build_model(ck["arch"], ck["model_cfg"])
    bb.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    if ck["arch"] == "encdec":
        del bb.dec                                                   # decoder is not needed for tagging
    return TokenTagger(bb, ck["arch"], dropout=dropout, bidirectional=bidirectional)


# ----------------------------------------------------------------------------- data
def encode_all(docs, tk):
    """Tokenise every document once: {id: (ids, spans, tags)}."""
    out = {}
    for d in docs:
        ids, spans = ner.encode_doc(d["text"], tk)
        out[d["id"]] = (ids, spans, ner.bio_tags(spans, d["ents"]))
    return out


def make_examples(enc, doc_ids, win=512, stride=256):
    ex = []
    for i in doc_ids:
        ids, _, tags = enc[i]
        for s in ner.make_windows(len(ids), win, stride):
            ex.append((ids[s:s + win], tags[s:s + win]))
    return ex


def _batch(examples, device):
    L = max(len(x) for x, _ in examples)
    ids = torch.zeros(len(examples), L, dtype=torch.long)
    tags = torch.full((len(examples), L), -100, dtype=torch.long)     # -100 = padding, ignored
    for b, (x, t) in enumerate(examples):
        ids[b, :len(x)] = torch.tensor(x); tags[b, :len(t)] = torch.tensor(t)
    return ids.to(device), tags.to(device)


# ----------------------------------------------------------------------------- prediction
@torch.no_grad()
def predict_logits(model, ids, win=512, stride=256, device="cuda", bs=16):
    """Per-token log-probs for one document. Each token takes its prediction from the window in
    which it has the most context: most central for bidirectional models, latest position (most
    left context) for causal decoders."""
    model.eval()
    causal = model.arch != "encdec" and not model.bidirectional
    starts = ner.make_windows(len(ids), win, stride)
    out = torch.zeros(len(ids), model.head.out_features)
    best = torch.full((len(ids),), -1e9)
    for b in range(0, len(starts), bs):
        chunk = starts[b:b + bs]
        x, _ = _batch([(ids[s:s + win], ids[s:s + win]) for s in chunk], device)
        with torch.autocast("cuda", dtype=torch.float16):
            lp = model(x).float().log_softmax(-1).cpu()
        for r, s in enumerate(chunk):
            n = min(win, len(ids) - s)
            j = torch.arange(n, dtype=torch.float)
            score = j if causal else torch.minimum(j, n - 1 - j)   # how much context position j has
            take = score > best[s:s + n]
            out[s:s + n][take] = lp[r, :n][take]
            best[s:s + n] = torch.maximum(best[s:s + n], score)
    return out


def predict_docs(model, docs, enc, win=512, stride=256, device="cuda"):
    preds = {}
    for d in docs:
        ids, spans, _ = enc[d["id"]]
        tags = predict_logits(model, ids, win, stride, device).argmax(-1).tolist()
        preds[d["id"]] = ner.tags_to_entities(tags, spans)
    return preds


# ----------------------------------------------------------------------------- training
def train_tagger(model, train_ex, dev_docs, enc, epochs=5, lr=5e-5, head_lr=1e-3, bs=8,
                 weight_decay=0.01, warmup=0.06, class_weight=None, device="cuda", seed=0, log=print):
    """AdamW, linear warm-up + linear decay, fp16 autocast. Evaluates strict F1 on dev_docs after
    every epoch and restores the best epoch's trainable weights at the end."""
    torch.manual_seed(seed); random.seed(seed)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    head_ids = {id(p) for p in model.head.parameters()}
    groups = [{"params": [p for p in params if id(p) not in head_ids], "lr": lr},
              {"params": [p for p in params if id(p) in head_ids], "lr": head_lr}]
    opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
    steps = epochs * math.ceil(len(train_ex) / bs); warm = max(1, int(warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda")
    w = None if class_weight is None else class_weight.to(device)
    best_f1, best_state, hist, step = -1.0, None, [], 0
    n_train = sum(p.numel() for p in params)
    log(f"trainable params: {n_train/1e6:.2f}M | {len(train_ex)} windows | {steps} steps")
    for ep in range(epochs):
        model.train(); random.shuffle(train_ex); t0 = time.time(); tot = 0.0
        for b in range(0, len(train_ex), bs):
            ids, tags = _batch(train_ex[b:b + bs], device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(ids)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), tags.view(-1),
                                   weight=w, ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            tot += loss.item()
        r = ner.strict_f1(predict_docs(model, dev_docs, enc, device=device),
                          {d["id"]: d["ents"] for d in dev_docs})
        hist.append({"epoch": ep + 1, "loss": tot / math.ceil(len(train_ex) / bs), **r})
        log(f"epoch {ep+1}: loss {hist[-1]['loss']:.4f} | dev strict F1 {r['f1']:.4f} "
            f"(P {r['precision']:.3f} R {r['recall']:.3f}) | {time.time()-t0:.0f}s")
        if r["f1"] > best_f1:
            best_f1 = r["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return hist, best_f1


# ----------------------------------------------------------------------------- v2: Viterbi decoding + ignore space tokens
def encode_all(docs, tk):
    """Tokenise every document once: {id: (ids, spans, tags)}. Pure-space tokens get tag -100
    (ignored in the loss): they carry no characters, so they have no entity label."""
    out = {}
    for d in docs:
        ids, spans = ner.encode_doc(d["text"], tk)
        tags = [t if sp is not None else -100 for t, sp in zip(ner.bio_tags(spans, d["ents"]), spans)]
        out[d["id"]] = (ids, spans, tags)
    return out


DECODE = "viterbi"                      # "argmax" (independent per token) or "viterbi" (valid BIO only)


def predict_docs(model, docs, enc, win=512, stride=256, device="cuda", decode=None):
    decode = decode or DECODE
    preds = {}
    for d in docs:
        ids, spans, _ = enc[d["id"]]
        lp = predict_logits(model, ids, win, stride, device)
        tags = ner.viterbi_bio(lp.numpy(), spans) if decode == "viterbi" else lp.argmax(-1).tolist()
        preds[d["id"]] = ner.tags_to_entities(tags, spans)
    return preds


# ----------------------------------------------------------------------------- v3: document header as context
HEADER = 128                            # first HEADER tokens of the document are prepended to every window
SEP_ID = 3                              # <|sep|> (reserved special token from our A1-BPE tokenizer)


def _with_header(ids, s, win, header):
    """Window [s, s+win) with the document header in front (only when the window does not
    already start at the beginning). Returns (input ids, offset of the window inside them)."""
    if header and s > 0:
        return ids[:header] + [SEP_ID] + ids[s:s + win], header + 1
    return ids[s:s + win], 0


def make_examples(enc, doc_ids, win=512, stride=256, header=None):
    header = HEADER if header is None else header
    ex = []
    for i in doc_ids:
        ids, _, tags = enc[i]
        for s in ner.make_windows(len(ids), win, stride):
            x, off = _with_header(ids, s, win, header)
            ex.append((x, [-100] * off + tags[s:s + win]))        # header tokens are not trained on
    return ex


@torch.no_grad()
def predict_logits(model, ids, win=512, stride=256, device="cuda", bs=16, header=None):
    header = HEADER if header is None else header
    model.eval()
    causal = model.arch != "encdec" and not model.bidirectional
    starts = ner.make_windows(len(ids), win, stride)
    out = torch.zeros(len(ids), model.head.out_features)
    best = torch.full((len(ids),), -1e9)
    for b in range(0, len(starts), bs):
        chunk = starts[b:b + bs]
        packed = [_with_header(ids, s, win, header) for s in chunk]
        x, _ = _batch([(p, p) for p, _ in packed], device)
        with torch.autocast("cuda", dtype=torch.float16):
            lp = model(x).float().log_softmax(-1).cpu()
        for r, (s, (_, off)) in enumerate(zip(chunk, packed)):
            n = min(win, len(ids) - s)
            j = torch.arange(n, dtype=torch.float)
            score = j if causal else torch.minimum(j, n - 1 - j)
            take = score > best[s:s + n]
            out[s:s + n][take] = lp[r, off:off + n][take]
            best[s:s + n] = torch.maximum(best[s:s + n], score)
    return out


# ----------------------------------------------------------------------------- v4: word-level decoding
DECODE = "words"                        # "argmax" | "viterbi" (token level) | "words" (word-consistent)


def predict_docs(model, docs, enc, win=512, stride=256, device="cuda", decode=None):
    decode = decode or DECODE
    preds = {}
    for d in docs:
        ids, spans, _ = enc[d["id"]]
        lp = predict_logits(model, ids, win, stride, device).numpy()
        if decode == "words":
            tags = ner.viterbi_words(lp, spans, d["text"])
        elif decode == "viterbi":
            tags = ner.viterbi_bio(lp, spans)
        else:
            tags = lp.argmax(-1).tolist()
        preds[d["id"]] = ner.tags_to_entities(tags, spans)
    return preds
