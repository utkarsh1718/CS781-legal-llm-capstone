"""
bpe_tokenizer.py -- A1 BPE v2: byte-level BPE for the Phase-1 legal LM.
Core learner from Assignment-1 (Q5) + lossless regex pre-tokeniser + digit grouping +
reserved specials + export to HF `tokenizers` (Rust) for fast bulk encoding.
"""
from __future__ import annotations
import heapq, json, re, time
from collections import Counter, defaultdict

# 16 reserved ids: doc separator, padding, masking/sep/cls for later phases, spares
SPECIALS = ["<|endoftext|>", "<|pad|>", "<|mask|>", "<|sep|>", "<|cls|>", "<|bos|>"] + \
           [f"<|extra_{i}|>" for i in range(10)]


def make_pattern(digit_group: int = 3) -> str:
    """Pre-tokenisation regex (text is already ASCII-normalised). A chunk is one of:
    contraction | optional space + letters | optional space + <=k digits |
    optional space + punctuation run | whitespace. Every character lands in exactly one chunk."""
    d = f"[0-9]{{1,{digit_group}}}" if digit_group > 1 else "[0-9]"
    return (r"'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?" + d +
            r"| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


def _bytes_to_unicode():
    """GPT-2's reversible byte -> printable-unicode map (HF ByteLevel BPE stores tokens this way)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + \
         list(range(ord("\xae"), ord("\xff") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


class ByteBPE:
    def __init__(self, digit_group: int = 3, specials=SPECIALS):
        self.digit_group = digit_group
        self.pattern = make_pattern(digit_group)
        self._pat = re.compile(self.pattern)
        self.specials = list(specials)
        self.n_sp = len(self.specials)
        self.id_to_bytes = {self.n_sp + b: bytes([b]) for b in range(256)}   # byte b -> id n_sp+b
        self.merges: list[tuple[int, int, int]] = []    # (left, right, new) in rank order
        self._rank, self._cache, self.train_info = {}, {}, {}

    # ---------------------------------------------------------------- training (A1 algorithm)
    def train(self, texts, vocab_size: int, min_freq: int = 2, verbose: bool = True, log_every: int = 2000):
        t0 = time.time()
        counts, n_chars = Counter(), 0
        for t in texts:                                   # 1) count unique chunks
            n_chars += len(t)
            counts.update(self._pat.findall(t))
        types = [(w, f) for w, f in counts.items() if f >= min_freq]   # hapaxes: little signal, lots of cost
        words = [[self.n_sp + b for b in w.encode("utf-8")] for w, _ in types]
        freqs = [f for _, f in types]
        if verbose:
            print(f"[bpe] {n_chars/1e6:.1f}M chars | {len(counts):,} unique chunks | "
                  f"{len(types):,} kept (freq>={min_freq}) | {time.time()-t0:.0f}s", flush=True)

        pair_freq, pair_words = defaultdict(int), defaultdict(set)   # 2) pair stats + index
        for wi, w in enumerate(words):
            for p in zip(w, w[1:]):
                pair_freq[p] += freqs[wi]
                pair_words[p].add(wi)
        heap = [(-f, p) for p, f in pair_freq.items()]
        heapq.heapify(heap)

        bytes_to_id = {v: k for k, v in self.id_to_bytes.items()}
        next_id = self.n_sp + 256
        while next_id < vocab_size and heap:              # 3) greedy merges
            negf, p = heapq.heappop(heap)
            if pair_freq.get(p, 0) != -negf:              # lazy deletion: stale heap entry
                continue
            if -negf < min_freq:
                break
            a, b = p
            nb = self.id_to_bytes[a] + self.id_to_bytes[b]
            nid = bytes_to_id.get(nb)
            if nid is None:                               # same byte string can arise from 2 merges
                nid, next_id = next_id, next_id + 1
                self.id_to_bytes[nid] = nb
                bytes_to_id[nb] = nid
            self.merges.append((a, b, nid))

            for wi in list(pair_words.pop(p, ())):        # only chunks containing the pair
                w = words[wi]
                nw, i, changed = [], 0, False
                while i < len(w):                         # merge left-to-right, non-overlapping
                    if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                        nw.append(nid); i += 2; changed = True
                    else:
                        nw.append(w[i]); i += 1
                if not changed:
                    continue
                f = freqs[wi]
                old_c, new_c = Counter(zip(w, w[1:])), Counter(zip(nw, nw[1:]))
                for q in old_c.keys() | new_c.keys():     # incremental pair-count update
                    d = new_c[q] - old_c[q]
                    if d:
                        pair_freq[q] += d * f
                        if d > 0:
                            pair_words[q].add(wi)
                        if pair_freq[q] > 0:
                            heapq.heappush(heap, (-pair_freq[q], q))
                        else:
                            pair_freq.pop(q, None)
                words[wi] = nw
            pair_freq.pop(p, None)
            if verbose and len(self.merges) % log_every == 0:
                print(f"[bpe] merges {len(self.merges):,} | vocab {next_id:,}/{vocab_size:,} | "
                      f"last freq {-negf:,} | {time.time()-t0:.0f}s", flush=True)

        self._build_rank()
        self.train_info = {"vocab_size": self.vocab_size, "merges": len(self.merges), "chars": n_chars,
                           "unique_chunks": len(counts), "kept_chunks": len(types), "min_freq": min_freq,
                           "digit_group": self.digit_group, "seconds": round(time.time() - t0, 1)}
        if verbose:
            print(f"[bpe] done: {self.train_info}", flush=True)
        return self

    def _build_rank(self):
        self._rank = {(a, b): (r, n) for r, (a, b, n) in enumerate(self.merges)}
        self._cache = {}

    @property
    def vocab_size(self):
        return self.n_sp + len(self.id_to_bytes)

    # ---------------------------------------------------------------- encode / decode (reference)
    def _encode_chunk(self, chunk: str):
        """Apply merges by rank: repeatedly merge the lowest-rank adjacent pair (GPT-2 style)."""
        ids = self._cache.get(chunk)
        if ids is not None:
            return ids
        w = [self.n_sp + b for b in chunk.encode("utf-8")]
        while len(w) > 1:
            best = None
            for i in range(len(w) - 1):
                r = self._rank.get((w[i], w[i + 1]))
                if r is not None and (best is None or r[0] < best[0]):
                    best = r
            if best is None:
                break
            r, nid = best
            a, b = self.merges[r][0], self.merges[r][1]
            nw, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    nw.append(nid); i += 2
                else:
                    nw.append(w[i]); i += 1
            w = nw
        if len(self._cache) < 1_000_000:
            self._cache[chunk] = w
        return w

    def encode(self, text: str):
        out = []
        for chunk in self._pat.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids):
        return b"".join(self.specials[i].encode() if i < self.n_sp else self.id_to_bytes[i]
                        for i in ids).decode("utf-8", errors="replace")

    # ---------------------------------------------------------------- persistence
    def save(self, path):
        json.dump({"format": "A1-BPE-v2", "digit_group": self.digit_group, "pattern": self.pattern,
                   "specials": self.specials,
                   "tokens": {str(i): b.hex() for i, b in self.id_to_bytes.items()},
                   "merges": self.merges, "train_info": self.train_info}, open(path, "w"))

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        obj = cls(digit_group=d["digit_group"], specials=d["specials"])
        obj.id_to_bytes = {int(i): bytes.fromhex(h) for i, h in d["tokens"].items()}
        obj.merges = [tuple(m) for m in d["merges"]]
        obj.train_info = d.get("train_info", {})
        obj._build_rank()
        return obj

    # ---------------------------------------------------------------- fast Rust encoder
    def to_hf(self):
        """Same vocab + merges + pre-tokeniser, applied by HF `tokenizers` (Rust)."""
        from tokenizers import Tokenizer, Regex, models, pre_tokenizers, decoders, AddedToken
        b2u = _bytes_to_unicode()
        tok_str = lambda bs: "".join(b2u[x] for x in bs)
        vocab = {s: i for i, s in enumerate(self.specials)}
        vocab.update({tok_str(bs): i for i, bs in self.id_to_bytes.items()})
        merges = [(tok_str(self.id_to_bytes[a]), tok_str(self.id_to_bytes[b])) for a, b, _ in self.merges]
        tk = Tokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=False))
        tk.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(self.pattern), behavior="isolated"),   # our regex chunks
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        tk.decoder = decoders.ByteLevel()
        tk.add_special_tokens([AddedToken(s, special=True, normalized=False) for s in self.specials])
        return tk


def fertility(tok, texts, encode=None):
    """Tokens per whitespace word and bytes per token."""
    enc = encode or tok.encode
    n_tok = n_words = n_bytes = 0
    for t in texts:
        n_tok += len(enc(t)); n_words += len(t.split()); n_bytes += len(t.encode())
    return {"tokens_per_word": n_tok / max(1, n_words), "bytes_per_token": n_bytes / max(1, n_tok),
            "tokens": n_tok, "words": n_words}


def verify_hf_equivalence(bpe, hf_tok, texts, max_report=3):
    """Count docs where the Python and Rust encoders disagree (or Rust fails to round-trip)."""
    bad = 0
    for t in texts:
        a = bpe.encode(t)
        b = hf_tok.encode(t, add_special_tokens=False).ids
        if a != b or hf_tok.decode(b) != t:
            bad += 1
            if bad <= max_report:
                k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
                print(f"MISMATCH at token {k}: ours {a[k:k+5]} vs rust {b[k:k+5]} | {t[:80]!r}")
    return bad
