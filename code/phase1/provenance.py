"""Where do the official test prompts come from? Checks whole-prompt containment in training docs
under candidate normalisations. Does NOT record the text following a prompt (diagnostic only)."""
from __future__ import annotations
import json, time, unicodedata
try:
    import orjson; _loads = orjson.loads
except ImportError:
    _loads = json.loads
import ahocorasick

PUNCT_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"', "\u2032": "'",
    "\u2013": "-", "\u2014": "-", "\u2012": "-", "\u2212": "-", "\u00ad": "",
    "\u2026": "...", "\u00a0": " ", "\u2022": "*", "\u00b7": ".",
})

def _collapse(s): return " ".join(s.split())

NORMALISERS = {
    "ws_only":        lambda t: _collapse(t),
    "drop_nonascii":  lambda t: _collapse(t.encode("ascii", "ignore").decode()),
    "nfkd_ascii":     lambda t: _collapse(unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()),
    "punctmap_ascii": lambda t: _collapse(unicodedata.normalize("NFKD", t.translate(PUNCT_MAP))
                                          .encode("ascii", "ignore").decode()),
}

def scan_source(source, local_path, out_dir, test, needles):
    A = ahocorasick.Automaton()
    for needle, payload in needles:
        A.add_word(needle, payload)
    A.make_automaton()
    log = open(f"{out_dir}/prov_log_{source}.txt", "w", buffering=1)
    out = open(f"{out_dir}/provenance_{source}.jsonl", "w")
    n = n_hit_docs = 0
    t0 = t_last = time.time()
    with open(local_path, "rb") as f:
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
            text = rec.get("text") or ""
            n += 1
            ws = _collapse(text)
            if not ws:
                continue
            hits = {}
            for _e, (tid, pos) in A.iter(ws):
                hits.setdefault(tid, set()).add(pos)
            if not text.isascii():
                for _e, (tid, pos) in A.iter(NORMALISERS["punctmap_ascii"](text)):
                    hits.setdefault(tid, set()).add(pos)
            if not hits:
                continue
            n_hit_docs += 1
            cache = {}
            for tid, positions in hits.items():
                prompt = test[tid]["text"]
                row = {"tid": tid, "source": source, "doc_i": n - 1,
                       "fingerprints_hit": sorted(positions), "full_match": None,
                       "doc_words": len(ws.split())}
                for name, fn in NORMALISERS.items():
                    if name not in cache:
                        cache[name] = fn(text)
                    idx = cache[name].find(prompt)
                    if idx >= 0:
                        before, after = cache[name][:idx], cache[name][idx + len(prompt):]
                        row.update(full_match=name, words_before=len(before.split()),
                                   words_after=len(after.split()),
                                   starts_at_word_boundary=(idx == 0 or before.endswith(" ")),
                                   doc_nonascii=not text.isascii())
                        break
                out.write(json.dumps(row) + "\n")
            if time.time() - t_last > 30:
                t_last = time.time()
                log.write(f"{source}: {n:,} docs | {n_hit_docs:,} hit docs | {(t_last-t0)/60:.1f} min\n")
    out.close()
    log.write(f"DONE {source}: {n:,} docs, {n_hit_docs:,} hit docs in {(time.time()-t0)/60:.1f} min\n")
    log.close()
    return source, n, n_hit_docs
