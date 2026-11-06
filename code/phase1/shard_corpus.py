"""
shard_corpus.py -- one pass per source: clean -> dedup -> decontaminate -> split -> sample ->
tokenize -> write uint16 token shards.
Outputs per source s: train_<s>.bin (uint16, docs end with <|endoftext|>=0), train_<s>_docs.npy
(doc start offsets), val_<s>.jsonl (normalised held-out docs), shard_stats_<s>.json.
"""
from __future__ import annotations
import json, os, time
import numpy as np
try:
    import orjson; _loads = orjson.loads
except ImportError:
    _loads = json.loads
import common

EOT_ID = 0          # <|endoftext|> is special id 0 in A1-BPE-v2


def _load_encoder(tok_hf_path, tok_py_path):
    """Rust tokenizer if available, else pure-Python A1-BPE-v2 (slow fallback)."""
    try:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(tok_hf_path)
        return lambda texts: [e.ids for e in tk.encode_batch(texts, add_special_tokens=False)]
    except Exception:
        from bpe_tokenizer import ByteBPE
        b = ByteBPE.load(tok_py_path)
        return lambda texts: [b.encode(t) for t in texts]


def process_source(source, local_path, out_dir, tok_hf_path, tok_py_path, cfg,
                   contaminated=frozenset(), batch_docs=256):
    t0 = t_last = time.time()
    encode = _load_encoder(tok_hf_path, tok_py_path)
    log = open(f"{out_dir}/shard_log_{source}.txt", "w", buffering=1)
    bin_f = open(f"{out_dir}/train_{source}.bin", "wb")
    val_f = open(f"{out_dir}/val_{source}.jsonl", "w")
    min_words, keep_frac = cfg["min_words"][source], cfg["keep_frac"][source]
    use_headnote = cfg.get("include_headnote", True)

    st = dict(docs_read=0, empty=0, too_short=0, duplicate=0, contaminated=0, val_docs=0,
              sampled_out=0, train_docs=0, headnote_docs=0, train_words=0, train_tokens=0)
    seen, doc_starts, offset, pending = set(), [], 0, []

    def flush():
        """Tokenize buffered docs in one Rust batch call and append them to the .bin."""
        nonlocal offset
        if not pending:
            return
        for ids in encode(pending):
            ids.append(EOT_ID)                                   # document separator
            arr = np.asarray(ids, dtype=np.uint16)
            bin_f.write(arr.tobytes())
            doc_starts.append(offset); offset += len(arr)
            st["train_tokens"] += len(arr)
        pending.clear()

    with open(local_path, "rb") as f:
        doc_i = -1
        for raw in f:
            s = raw.strip()
            if s in (b"[", b"]", b""):
                continue
            if s.endswith(b","):
                s = s[:-1]
            try:
                rec = _loads(s)
            except Exception:
                continue
            doc_i += 1                                           # same numbering as provenance.py
            st["docs_read"] += 1

            text = common.normalise(rec.get("text") or "")      # 1) test-format normalisation
            if not text:
                st["empty"] += 1; continue
            nw = text.count(" ") + 1                             # single-spaced -> word count
            if nw < min_words:                                   # 2) length filter
                st["too_short"] += 1; continue
            h = common.doc_hash(text)
            if h in seen:                                        # 3) exact dedup
                st["duplicate"] += 1; continue
            seen.add(h)
            if doc_i in contaminated:                            # 4) test-prompt source doc
                st["contaminated"] += 1; continue
            if common.split_of(text, source) == "val":           # 5) doc-level split
                st["val_docs"] += 1
                val_f.write(json.dumps({"doc_i": doc_i, "text": text}) + "\n")
                continue
            if keep_frac < 1.0 and ((h >> 20) % 10_000) >= keep_frac * 10_000:   # 6) sampling
                st["sampled_out"] += 1; continue

            st["train_docs"] += 1; st["train_words"] += nw
            pending.append(text)
            hn = rec.get("headnote") if use_headnote else None   # SC headnotes as extra docs
            if hn:
                hn = common.normalise(hn)
                if hn and hn.count(" ") + 1 >= 20:
                    st["headnote_docs"] += 1; st["train_words"] += hn.count(" ") + 1
                    pending.append(hn)
            if len(pending) >= batch_docs:                       # 7) tokenize + write
                flush()
            if time.time() - t_last > 30:
                t_last = time.time()
                log.write(f"{source}: read {st['docs_read']:,} | train docs {st['train_docs']:,} | "
                          f"tokens {st['train_tokens']/1e6:.1f}M | {(t_last-t0)/60:.1f} min\n")
    flush()
    bin_f.close(); val_f.close()
    np.save(f"{out_dir}/train_{source}_docs.npy", np.asarray(doc_starts, dtype=np.int64))
    st["elapsed_min"] = round((time.time() - t0) / 60, 2)
    st["tokens_per_word"] = st["train_tokens"] / max(1, st["train_words"])
    st["bin_bytes"] = os.path.getsize(f"{out_dir}/train_{source}.bin")
    json.dump(st, open(f"{out_dir}/shard_stats_{source}.json", "w"), indent=1)
    log.write(f"DONE {source}: {st}\n"); log.close()
    return source, st
