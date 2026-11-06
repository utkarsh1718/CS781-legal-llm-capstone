"""Single pass over one Phase-1 source file (JSON array, one object per line).
Outputs in out_dir: summary_<src>.json, wordlens_<src>.npy, sample_<src>.jsonl, log_<src>.txt"""
from __future__ import annotations
import hashlib, json, random, time, zlib
from collections import Counter
import numpy as np
try:
    import orjson; _loads = orjson.loads
except ImportError:
    _loads = json.loads
try:
    import ahocorasick
except ImportError:
    ahocorasick = None

PCTS = [0, 1, 5, 25, 50, 75, 95, 99, 100]
MIN_VAL_WORDS = 640  # 512-word prompt + 128-word reference


def profile_source(source, local_path, out_dir, needles=(), reservoir_k=5000,
                   seed=1337, log_every=5_000, max_parse_errors=100):
    log = open(f"{out_dir}/log_{source}.txt", "w", buffering=1)
    rng = random.Random(seed + zlib.crc32(source.encode()))
    automaton = None
    if needles and ahocorasick is not None:
        automaton = ahocorasick.Automaton()
        for needle, payload in needles:
            automaton.add_word(needle, payload)
        automaton.make_automaton()

    word_lens, char_lens = [], []
    n = errors = bytes_read = 0
    nonascii_docs = nonascii_extra_bytes = newlines = 0
    seen, dup_docs, dup_words = set(), 0, 0
    headnote_docs = headnote_words = 0
    keys, years, groups = Counter(), Counter(), Counter()
    reservoir, matches = [], []
    t0 = t_last_log = time.time()

    with open(local_path, "rb") as f:
        for raw in f:
            bytes_read += len(raw)
            s = raw.strip()
            if s in (b"[", b"]", b""):
                continue
            if s.endswith(b","):
                s = s[:-1]
            try:
                rec = _loads(s)
            except Exception as e:
                errors += 1
                log.write(f"PARSE ERROR #{errors} at doc~{n}: {e!r} | {s[:120]!r}\n")
                if errors > max_parse_errors:
                    raise RuntimeError(f"{source}: too many parse errors") from e
                continue

            text = rec.get("text") or ""
            if n < 2000:
                keys.update(rec.keys())
            toks = text.split()
            nw = len(toks)
            word_lens.append(nw); char_lens.append(len(text))
            newlines += text.count("\n")
            if not text.isascii():
                nonascii_docs += 1
                nonascii_extra_bytes += len(text.encode("utf-8")) - len(text)

            norm = " ".join(toks)
            h = hashlib.blake2b(norm.encode("utf-8"), digest_size=8).digest()
            if h in seen:
                dup_docs += 1; dup_words += nw
            else:
                seen.add(h)

            hn = rec.get("headnote")
            if hn:
                headnote_docs += 1; headnote_words += len(hn.split())
            years[str(rec.get("year") or (rec.get("date") or "")[:4])] += 1
            groups[rec.get("court") or rec.get("tribunal") or rec.get("category") or "?"] += 1

            if automaton is not None and norm:
                for _end, (tid, pos) in automaton.iter(norm):
                    if len(matches) < 20_000:
                        matches.append((tid, pos, n))

            item = {"i": n, "text": text,
                    "meta": {k: rec.get(k) for k in ("title", "court", "tribunal", "category", "year", "date")
                             if rec.get(k) is not None}}
            if len(reservoir) < reservoir_k:
                reservoir.append(item)
            else:
                j = rng.randrange(n + 1)
                if j < reservoir_k:
                    reservoir[j] = item

            n += 1
            if n % log_every == 0 or time.time() - t_last_log > 30:
                t_last_log = time.time(); dt = t_last_log - t0
                log.write(f"{source}: {n:,} docs | {bytes_read/1e9:.2f} GB | "
                          f"{bytes_read/1e6/dt:.1f} MB/s | {dt/60:.1f} min\n")

    wl = np.asarray(word_lens, dtype=np.int64)
    np.save(f"{out_dir}/wordlens_{source}.npy", wl.astype(np.int32))
    with open(f"{out_dir}/sample_{source}.jsonl", "w", encoding="utf-8") as g:
        for item in reservoir:
            g.write(json.dumps(item, ensure_ascii=False) + "\n")
    total_words = int(wl.sum()); long_mask = wl >= MIN_VAL_WORDS
    summary = {
        "source": source, "bytes": bytes_read, "docs": n, "parse_errors": errors,
        "total_words": total_words, "total_chars": int(np.sum(char_lens)),
        "words_mean": float(wl.mean()) if n else 0.0,
        "words_percentiles": dict(zip(map(str, PCTS), np.percentile(wl, PCTS).astype(int).tolist() if n else [])),
        "docs_ge_640_words": int(long_mask.sum()),
        "words_in_docs_ge_640": int(wl[long_mask].sum()),
        "frac_docs_nonascii": nonascii_docs / max(1, n),
        "nonascii_extra_bytes_per_1k_chars": 1000 * nonascii_extra_bytes / max(1, int(np.sum(char_lens))),
        "newlines_per_1k_words": 1000 * newlines / max(1, total_words),
        "exact_dup_docs": dup_docs, "exact_dup_frac_docs": dup_docs / max(1, n),
        "exact_dup_frac_words": dup_words / max(1, total_words),
        "headnote_docs": headnote_docs, "headnote_words": headnote_words,
        "record_keys": dict(keys.most_common()), "top_groups": groups.most_common(25),
        "n_groups": len(groups), "years": dict(sorted(years.items())),
        "test_fingerprint_hits": len(matches),
        "test_ids_matched": sorted({m[0] for m in matches}, key=lambda x: int(x) if str(x).isdigit() else str(x)),
        "test_matches_sample": matches[:200], "leak_check_ran": automaton is not None,
        "elapsed_min": (time.time() - t0) / 60,
    }
    with open(f"{out_dir}/summary_{source}.json", "w") as g:
        json.dump(summary, g, indent=1)
    log.write(f"DONE {source}: {n:,} docs in {summary['elapsed_min']:.1f} min\n"); log.close()
    return summary


def build_needles(test_dict, width=20, positions=(0, 246, 492)):
    """Three 20-word fingerprints per test prompt (start / middle / end)."""
    out = []
    for tid, v in test_dict.items():
        w = v["text"].split()
        for p in positions:
            if p + width <= len(w):
                out.append((" ".join(w[p:p + width]), (tid, p)))
    return out
