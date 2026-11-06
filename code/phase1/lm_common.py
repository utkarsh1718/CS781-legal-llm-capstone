"""
lm_common.py -- building blocks shared by all Phase-1 architectures + KV-cached generation.
RMSNorm (fp32 stats) | RoPE | QK-norm (fp16 stability) | SwiGLU | SDPA attention (memory-
efficient kernel on T4) | KV cache: prompt processed once, then one token per step.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()                                              # fp32 statistics
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(x.dtype) * self.weight.to(x.dtype)


def rope_cache(seq_len: int, head_dim: int, theta: float = 10000.0, device=None):
    """cos/sin tables of shape (seq_len, head_dim/2), fp32."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    ang = torch.outer(torch.arange(seq_len, device=device).float(), inv)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, T, hd); cos/sin: (T, hd/2). Rotate-half formulation, computed in fp32."""
    h = x.shape[-1] // 2
    xf = x.float()
    x1, x2 = xf[..., :h], xf[..., h:]
    c, s = cos[None, None], sin[None, None]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).to(x.dtype)


@dataclass
class KVCache:
    k: torch.Tensor          # (B, n_kv_heads, max_T, head_dim)
    v: torch.Tensor


class Attention(nn.Module):
    """Multi-head attention with grouped-query K/V (n_kv_heads <= n_heads), QK-norm, RoPE."""

    def __init__(self, d_model, n_heads, n_kv_heads, qk_norm=True):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.h, self.kv, self.hd = n_heads, n_kv_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, n_heads * self.hd, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.hd, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(n_heads * self.hd, d_model, bias=False)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm, self.k_norm = RMSNorm(self.hd), RMSNorm(self.hd)

    def forward(self, x, cos, sin, cache: KVCache | None = None, pos: int = 0):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rope(q, cos[pos:pos + T], sin[pos:pos + T])
        k = apply_rope(k, cos[pos:pos + T], sin[pos:pos + T])
        if cache is not None:                                        # write new K/V, read all
            cache.k[:, :, pos:pos + T] = k
            cache.v[:, :, pos:pos + T] = v
            k, v = cache.k[:, :, :pos + T], cache.v[:, :, :pos + T]
        if self.kv != self.h:                                        # GQA: share K/V across groups
            rep = self.h // self.kv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        # causal mask needed only when several queries are processed at once (training/prefill
        # at pos=0); decoding uses T=1, which may attend to everything cached.
        assert T == 1 or pos == 0, "chunked prefill (T>1 at pos>0) is not supported"
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        return self.wo(y.transpose(1, 2).reshape(B, T, self.h * self.hd))


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)   # gate
        self.w3 = nn.Linear(d_model, hidden, bias=False)   # up
        self.w2 = nn.Linear(hidden, d_model, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def count_params(model: nn.Module) -> int:
    """Unique parameters (tied embedding counted once)."""
    return sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())


# --------------------------------------------------------------------------- generation
def _apply_sampling(logits, temperature, top_k, top_p):
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)                       # greedy
    logits = logits / temperature
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:                                         # nucleus sampling
        sl, si = torch.sort(logits, descending=True, dim=-1)
        cp = sl.softmax(-1).cumsum(-1)
        remove = cp - sl.softmax(-1) > top_p                          # always keep the top token
        sl = sl.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, si, sl)
    return torch.multinomial(logits.softmax(-1), 1)


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.0, top_k=0, top_p=1.0,
             repetition_penalty=1.0, banned_ids=(0,), no_repeat_ngram=0):
    """KV-cached generation for a batch of EQUAL-length prompts. idx: (B, T) -> (B, max_new_tokens).
    banned_ids: <|endoftext|> by default so the model cannot stop before 128 words."""
    model.eval()
    B, T = idx.shape
    max_T = T + max_new_tokens
    # cache dtype must match the K/V activations: fp16 under autocast, else the weights' dtype
    cdt = torch.float16 if torch.is_autocast_enabled() else next(model.parameters()).dtype
    caches = model.make_caches(B, max_T, idx.device, dtype=cdt)
    logits = model(idx, caches=caches, pos=0)[:, -1].float()         # prefill the prompt once
    out, seen = [], idx.clone()
    for i in range(max_new_tokens):
        if repetition_penalty != 1.0:                                 # CTRL-style penalty
            g = logits.gather(1, seen)
            logits.scatter_(1, seen, torch.where(g > 0, g / repetition_penalty, g * repetition_penalty))
        if banned_ids:
            logits[:, list(banned_ids)] = float("-inf")
        if no_repeat_ngram and seen.size(1) >= no_repeat_ngram:       # block repeated n-grams
            n = no_repeat_ngram
            for b in range(B):
                s = seen[b].tolist(); prefix = tuple(s[-(n - 1):])
                bad = [s[j + n - 1] for j in range(len(s) - n + 1) if tuple(s[j:j + n - 1]) == prefix]
                if bad:
                    logits[b, bad] = float("-inf")
        nxt = _apply_sampling(logits, temperature, top_k, top_p)      # (B, 1)
        out.append(nxt)
        seen = torch.cat([seen, nxt], 1)
        if i + 1 < max_new_tokens:                                    # one cached step
            logits = model(nxt, caches=caches, pos=T + i)[:, -1].float()
    return torch.cat(out, 1)
