"""
common.py -- rules shared by every Phase-1 stage (tokenizer, sharding, validation, generation).

normalise(text): make training text look like the official test prompts -- ASCII only, single
    spaces, no newlines (0/1000 test prompts contain a newline, double space or non-ASCII).
split_of(norm_text, source): deterministic DOCUMENT-level train/val assignment from a hash of
    the full normalised text -> exact duplicates always land in the SAME split.
"""
from __future__ import annotations
import hashlib, unicodedata

# Typographic characters -> ASCII equivalents (applied before stripping accents)
PUNCT_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"',
    "\u2013": "-", "\u2014": "-", "\u2012": "-", "\u2212": "-", "\u2010": "-", "\u2011": "-",
    "\u2026": "...", "\u00a0": " ", "\u00ad": "", "\u2022": "*", "\u00b7": ".", "\u200b": "",
    "\ufeff": "",
})

SOURCES = ["acts", "supreme_court", "tribunals", "high_courts"]

# Share of documents held out for validation. SC is the test domain -> largest share (~1,100 docs).
VAL_FRAC = {"supreme_court": 0.02, "acts": 0.02, "tribunals": 0.002, "high_courts": 0.002}


def normalise(text: str) -> str:
    """ASCII-fold (map punctuation, strip accents, drop the rest) and collapse all whitespace."""
    if not text:
        return ""
    if not text.isascii():                                   # fast path: most docs are ASCII
        text = unicodedata.normalize("NFKD", text.translate(PUNCT_MAP))
        text = text.encode("ascii", "ignore").decode("ascii")
    return " ".join(text.split())                            # newlines/tabs/double spaces -> 1 space


def doc_hash(norm_text: str) -> int:
    """Stable 64-bit hash (blake2b; Python's hash() is randomised per process)."""
    return int.from_bytes(hashlib.blake2b(norm_text.encode(), digest_size=8).digest(), "big")


def split_of(norm_text: str, source: str) -> str:
    """'val' or 'train' -- identical on every run and machine."""
    return "val" if (doc_hash(norm_text) % 1_000_000) < VAL_FRAC[source] * 1_000_000 else "train"
