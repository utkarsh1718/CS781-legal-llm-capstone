"""
valset.py -- validation data from held-out documents.
1) Generation set in the EXACT test format: 512-word prompt starting at a sentence/numbered-
   paragraph boundary + the next 128 words as reference -> val_prompts.json / val_refs.json.
2) Validation-loss shards val_<source>.bin for bits-per-word curves.
"""
from __future__ import annotations
import json, random, re
import numpy as np

PROMPT_WORDS, REF_WORDS = 512, 128

# Tokens ending in "." that do NOT end a sentence in Indian legal text
ABBREV = {"vs.", "v.", "no.", "nos.", "ors.", "anr.", "co.", "ltd.", "pvt.", "dr.", "mr.", "mrs.",
          "ms.", "sri.", "smt.", "shri.", "s.", "ss.", "art.", "arts.", "sec.", "secs.", "rs.",
          "viz.", "i.e.", "e.g.", "etc.", "cl.", "para.", "paras.", "p.", "pp.", "j.", "jj.",
          "st.", "r.", "o.", "govt.", "dept.", "addl.", "asstt.", "hon'ble", "u/s.", "w.e.f.",
          "cr.", "crl.", "civ.", "misc.", "spl.", "a.i.r.", "s.c.", "s.c.c.", "vol.", "ed.", "approx."}
_single_letter_dot = re.compile(r"^[A-Za-z]\.$")          # initials like "K."


def sentence_starts(words):
    """Word indices that plausibly start a sentence or numbered paragraph."""
    out = [0]
    for i in range(1, len(words)):
        prev, w = words[i - 1], words[i]
        if not prev.endswith((".", "?", "!", ":")):
            continue
        if prev.lower() in ABBREV or _single_letter_dot.match(prev):
            continue
        if w[0].isupper() or w[0].isdigit() or w[0] in "(\"'[":
            out.append(i)
    return out


def build_generation_set(val_jsonl_by_source, n_per_source, seed=1337):
    """One random sentence-boundary window per held-out doc, until n windows per source."""
    rng = random.Random(seed)
    prompts, refs, meta = {}, {}, {}
    for src, path in val_jsonl_by_source.items():
        docs = [json.loads(l) for l in open(path)]
        rng.shuffle(docs)
        k = 0
        for d in docs:
            if k >= n_per_source.get(src, 0):
                break
            w = d["text"].split()
            if len(w) < PROMPT_WORDS + REF_WORDS:
                continue
            cands = [i for i in sentence_starts(w) if i + PROMPT_WORDS + REF_WORDS <= len(w)]
            if not cands:
                continue
            i = rng.choice(cands)
            vid = f"{src[:2]}_{k:04d}"
            prompts[vid] = {"text": " ".join(w[i:i + PROMPT_WORDS])}
            refs[vid] = {"text": " ".join(w[i + PROMPT_WORDS:i + PROMPT_WORDS + REF_WORDS])}
            meta[vid] = {"source": src, "doc_i": d["doc_i"], "start_word": i, "doc_words": len(w)}
            k += 1
    return prompts, refs, meta


def build_val_loss_shards(val_jsonl_by_source, out_dir, encode_batch, max_tokens_by_source, eot_id=0):
    """Tokenize held-out docs into val_<src>.bin; returns token/word counts (for bits per word)."""
    stats = {}
    for src, path in val_jsonl_by_source.items():
        texts = [json.loads(l)["text"] for l in open(path)]
        cap = max_tokens_by_source.get(src, 1_000_000)
        n_tok = n_words = 0
        with open(f"{out_dir}/val_{src}.bin", "wb") as f:
            for i in range(0, len(texts), 256):
                batch = texts[i:i + 256]
                for t, ids in zip(batch, encode_batch(batch)):
                    ids = list(ids) + [eot_id]
                    f.write(np.asarray(ids, dtype=np.uint16).tobytes())
                    n_tok += len(ids); n_words += t.count(" ") + 1
                if n_tok >= cap:
                    break
        stats[src] = {"tokens": n_tok, "words": n_words, "tokens_per_word": n_tok / max(1, n_words)}
    return stats
