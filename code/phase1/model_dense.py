"""
model_dense.py -- M1: dense decoder-only Transformer (the controlled baseline).
Pre-norm: x = x + Attn(RMSNorm(x)); x = x + SwiGLU(RMSNorm(x)). GQA + QK-norm + RoPE,
tied input/output embeddings.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from lm_common import RMSNorm, Attention, SwiGLU, KVCache, rope_cache


@dataclass
class DenseConfig:
    vocab_size: int = 32000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_hidden: int = 2048          # ~8/3 * d_model, multiple of 64 (tensor cores)
    max_seq_len: int = 2048         # RoPE table length (training uses 1024)
    rope_theta: float = 10000.0
    qk_norm: bool = True
    tie_embeddings: bool = True
    init_std: float = 0.02


class Block(nn.Module):
    def __init__(self, c: DenseConfig):
        super().__init__()
        self.norm1 = RMSNorm(c.d_model)
        self.attn = Attention(c.d_model, c.n_heads, c.n_kv_heads, c.qk_norm)
        self.norm2 = RMSNorm(c.d_model)
        self.mlp = SwiGLU(c.d_model, c.ffn_hidden)

    def forward(self, x, cos, sin, cache=None, pos=0):
        x = x + self.attn(self.norm1(x), cos, sin, cache, pos)
        return x + self.mlp(self.norm2(x))


class DenseGPT(nn.Module):
    def __init__(self, c: DenseConfig):
        super().__init__()
        self.cfg = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.norm_f = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight                  # weight tying
        cos, sin = rope_cache(c.max_seq_len, c.d_model // c.n_heads, c.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        # GPT-2 trick: shrink residual-output projections by 1/sqrt(2 * n_layers)
        for n, p in self.named_parameters():
            if n.endswith("wo.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, 0.0, c.init_std / math.sqrt(2 * c.n_layers))

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, self.cfg.init_std)

    def make_caches(self, B, max_T, device, dtype=torch.float16):
        hd, kv = self.cfg.d_model // self.cfg.n_heads, self.cfg.n_kv_heads
        return [KVCache(torch.zeros(B, kv, max_T, hd, device=device, dtype=dtype),
                        torch.zeros(B, kv, max_T, hd, device=device, dtype=dtype))
                for _ in self.blocks]

    def forward(self, idx, targets=None, caches=None, pos: int = 0):
        x = self.tok_emb(idx)
        for i, blk in enumerate(self.blocks):
            x = blk(x, self.rope_cos, self.rope_sin, None if caches is None else caches[i], pos)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits                                              # (B, T, V)
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1),
                               ignore_index=-1)
