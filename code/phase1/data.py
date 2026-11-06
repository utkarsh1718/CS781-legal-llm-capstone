"""
data.py -- batches from per-source uint16 shards with a (possibly time-varying) mixture.
* memory-mapped shards (no RAM load); each sample reads seq_len+1 tokens
* sampling = pure function of (seed, step, rank, micro) -> exact resume, no loader state,
  DDP ranks never draw the same windows
* recipes: natural | sc_heavy | curriculum (broad mix -> SC-heavy anneal during LR decay)
"""
from __future__ import annotations
import numpy as np
import torch

SOURCES = ["acts", "supreme_court", "tribunals", "high_courts"]

MIXTURES = {
    "sc_heavy":  {"supreme_court": 0.50, "high_courts": 0.20, "tribunals": 0.15, "acts": 0.15},
    "broad":     {"supreme_court": 0.30, "high_courts": 0.30, "tribunals": 0.25, "acts": 0.15},
    "sc_anneal": {"supreme_court": 0.75, "acts": 0.10, "high_courts": 0.10, "tribunals": 0.05},
}


def mixture_weights(recipe: str, frac: float, n_tokens: dict, switch_at: float = 0.8):
    """Source weights at training progress `frac` in [0, 1]."""
    if recipe == "natural":
        tot = sum(n_tokens.values())
        return {s: n_tokens[s] / tot for s in SOURCES}
    if recipe == "curriculum":
        return MIXTURES["broad"] if frac < switch_at else MIXTURES["sc_anneal"]
    return MIXTURES[recipe]


class ShardSampler:
    def __init__(self, data_dir: str, split: str = "train", seq_len: int = 1024):
        self.seq_len = seq_len
        self.arrs = {s: np.memmap(f"{data_dir}/{split}_{s}.bin", dtype=np.uint16, mode="r")
                     for s in SOURCES}
        self.n_tokens = {s: len(a) for s, a in self.arrs.items()}

    def batch(self, weights: dict, B: int, seed: int, step: int, rank: int, micro: int):
        rng = np.random.default_rng([seed, step, rank, micro])       # deterministic stream
        names = list(weights)
        p = np.array([weights[s] for s in names], dtype=np.float64); p /= p.sum()
        srcs = rng.choice(len(names), size=B, p=p)                    # source per sequence
        T = self.seq_len
        buf = np.empty((B, T + 1), dtype=np.int64)
        for i, si in enumerate(srcs):
            a = self.arrs[names[si]]
            off = int(rng.integers(0, len(a) - T - 1))                # random window
            buf[i] = a[off:off + T + 1]
        t = torch.from_numpy(buf)
        return t[:, :-1], t[:, 1:], np.bincount(srcs, minlength=len(names))


def fixed_val_windows(data_dir: str, seq_len: int, n_windows: int):
    """Deterministic, spread-out validation windows per source: {src: LongTensor (N, T+1)}."""
    out = {}
    for s in SOURCES:
        a = np.memmap(f"{data_dir}/val_{s}.bin", dtype=np.uint16, mode="r")
        n = min(n_windows, (len(a) - 1) // (seq_len + 1))
        idx = np.linspace(0, len(a) - seq_len - 2, n).astype(np.int64)
        out[s] = torch.from_numpy(np.stack([a[i:i + seq_len + 1].astype(np.int64) for i in idx]))
    return out
