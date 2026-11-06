# model_encdec.py -- M3: encoder-decoder Transformer trained on "prompt -> continuation".
# Each 1,024-token window: first enc_len=640 tokens -> BIDIRECTIONAL encoder; decoder (causal
# self-attn + cross-attn) predicts the next 384. Generation: encode prompt once, cache each
# decoder layer's cross-attention K/V once, normal KV cache for decoder self-attention.
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from lm_common import RMSNorm, SwiGLU, rope_cache, apply_rope


@dataclass
class EncDecConfig:
    vocab_size: int = 32000
    d_model: int = 768
    n_enc_layers: int = 6
    n_dec_layers: int = 6
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_hidden: int = 2048
    enc_len: int = 640             # encoder part of each training window (prompt side)
    bos_id: int = 5                # <|bos|> starts every decoder sequence
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    init_std: float = 0.02


@dataclass
class DecCache:
    k: torch.Tensor                # decoder self-attention keys   (B, kv, max_T, hd)
    v: torch.Tensor
    ck: torch.Tensor = None        # cross-attention keys/values from the encoder (B, kv, Te, hd)
    cv: torch.Tensor = None


class SelfAttn(nn.Module):
    # GQA self-attention with QK-norm + RoPE; causal (decoder) or bidirectional (encoder)
    def __init__(self, c: EncDecConfig, causal: bool):
        super().__init__()
        self.h, self.kv, self.hd, self.causal = c.n_heads, c.n_kv_heads, c.d_model // c.n_heads, causal
        self.wq = nn.Linear(c.d_model, self.h * self.hd, bias=False)
        self.wk = nn.Linear(c.d_model, self.kv * self.hd, bias=False)
        self.wv = nn.Linear(c.d_model, self.kv * self.hd, bias=False)
        self.wo = nn.Linear(self.h * self.hd, c.d_model, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.hd), RMSNorm(self.hd)

    def forward(self, x, cos, sin, cache=None, pos=0):
        B, T, _ = x.shape
        q = self.q_norm(self.wq(x).view(B, T, self.h, self.hd).transpose(1, 2))
        k = self.k_norm(self.wk(x).view(B, T, self.kv, self.hd).transpose(1, 2))
        v = self.wv(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos[pos:pos + T], sin[pos:pos + T]), apply_rope(k, cos[pos:pos + T], sin[pos:pos + T])
        if cache is not None:                                        # decoder generation
            cache.k[:, :, pos:pos + T], cache.v[:, :, pos:pos + T] = k, v
            k, v = cache.k[:, :, :pos + T], cache.v[:, :, :pos + T]
        rep = self.h // self.kv
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        assert T == 1 or pos == 0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal and T > 1)
        return self.wo(y.transpose(1, 2).reshape(B, T, self.h * self.hd))


class CrossAttn(nn.Module):
    # decoder queries attend to encoder states (no mask, no RoPE); K/V computable once per prompt
    def __init__(self, c: EncDecConfig):
        super().__init__()
        self.h, self.kv, self.hd = c.n_heads, c.n_kv_heads, c.d_model // c.n_heads
        self.wq = nn.Linear(c.d_model, self.h * self.hd, bias=False)
        self.wk = nn.Linear(c.d_model, self.kv * self.hd, bias=False)
        self.wv = nn.Linear(c.d_model, self.kv * self.hd, bias=False)
        self.wo = nn.Linear(self.h * self.hd, c.d_model, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.hd), RMSNorm(self.hd)

    def mem_kv(self, mem):
        B, S, _ = mem.shape
        k = self.k_norm(self.wk(mem).view(B, S, self.kv, self.hd).transpose(1, 2))
        v = self.wv(mem).view(B, S, self.kv, self.hd).transpose(1, 2)
        return k, v

    def forward(self, x, k, v):
        B, T, _ = x.shape
        q = self.q_norm(self.wq(x).view(B, T, self.h, self.hd).transpose(1, 2))
        rep = self.h // self.kv
        y = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1))
        return self.wo(y.transpose(1, 2).reshape(B, T, self.h * self.hd))


class EncLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = RMSNorm(c.d_model), RMSNorm(c.d_model)
        self.attn, self.mlp = SelfAttn(c, causal=False), SwiGLU(c.d_model, c.ffn_hidden)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)                      # bidirectional
        return x + self.mlp(self.n2(x))


class DecLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2, self.n3 = RMSNorm(c.d_model), RMSNorm(c.d_model), RMSNorm(c.d_model)
        self.self_attn, self.cross = SelfAttn(c, causal=True), CrossAttn(c)
        self.mlp = SwiGLU(c.d_model, c.ffn_hidden)

    def forward(self, x, cos, sin, ck, cv, cache=None, pos=0):
        x = x + self.self_attn(self.n1(x), cos, sin, cache, pos)    # causal over the continuation
        x = x + self.cross(self.n2(x), ck, cv)                        # look at the encoded prompt
        return x + self.mlp(self.n3(x))


class EncDecLM(nn.Module):
    def __init__(self, c: EncDecConfig):
        super().__init__()
        self.cfg = c
        self.max_prompt_tokens = c.enc_len                     # inference keeps the last 640 tokens
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)  # shared by encoder, decoder, output
        self.enc = nn.ModuleList([EncLayer(c) for _ in range(c.n_enc_layers)])
        self.dec = nn.ModuleList([DecLayer(c) for _ in range(c.n_dec_layers)])
        self.enc_norm, self.dec_norm = RMSNorm(c.d_model), RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        cos, sin = rope_cache(c.max_seq_len, c.d_model // c.n_heads, c.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        L = c.n_enc_layers + c.n_dec_layers
        for n, p in self.named_parameters():                   # scaled residual-output init
            if n.endswith(("wo.weight", "w2.weight")):
                nn.init.normal_(p, 0.0, c.init_std / math.sqrt(2 * L))
        self._prompt_len = 0

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, self.cfg.init_std)

    def encode(self, src):
        x = self.tok_emb(src)
        for layer in self.enc:
            x = layer(x, self.rope_cos, self.rope_sin)
        return self.enc_norm(x)

    def decode_tf(self, mem, dec_in):
        # teacher-forced decoder pass over the whole target sequence -> logits (B, Td, V)
        x = self.tok_emb(dec_in)
        for layer in self.dec:
            ck, cv = layer.cross.mem_kv(mem)
            x = layer(x, self.rope_cos, self.rope_sin, ck, cv)
        return self.lm_head(self.dec_norm(x))

    def make_caches(self, B, max_T, device, dtype=torch.float16):
        kv, hd = self.cfg.n_kv_heads, self.cfg.d_model // self.cfg.n_heads
        return [DecCache(torch.zeros(B, kv, max_T, hd, device=device, dtype=dtype),
                         torch.zeros(B, kv, max_T, hd, device=device, dtype=dtype)) for _ in self.dec]

    def _gen_step(self, idx, caches, pos):
        # pos == 0: idx is the PROMPT -> encode once, cache cross K/V, decode <|bos|>
        # pos  > 0: idx is the last generated token (generation.py passes pos = prompt_len + i)
        if pos == 0:
            mem = self.encode(idx)
            for layer, c in zip(self.dec, caches):
                c.ck, c.cv = layer.cross.mem_kv(mem)
            self._prompt_len = idx.size(1)
            tok, dpos = torch.full((idx.size(0), 1), self.cfg.bos_id, device=idx.device), 0
        else:
            tok, dpos = idx, pos - self._prompt_len + 1
        x = self.tok_emb(tok)
        for layer, c in zip(self.dec, caches):
            x = layer(x, self.rope_cos, self.rope_sin, c.ck, c.cv, c, dpos)
        return self.lm_head(self.dec_norm(x))

    def forward(self, idx, targets=None, caches=None, pos: int = 0):
        if caches is not None:
            return self._gen_step(idx, caches, pos)
        # training / eval: rebuild the (T+1)-token window, split it into prompt | continuation
        full = torch.cat([idx[:, :1], targets], dim=1)
        T, s = idx.size(1), self.cfg.enc_len
        src, tgt = full[:, :s], full[:, s:T]                                   # 640 | 384 tokens
        bos = torch.full((idx.size(0), 1), self.cfg.bos_id, device=idx.device, dtype=idx.dtype)
        logits = self.decode_tf(self.encode(src), torch.cat([bos, tgt[:, :-1]], 1))
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-1)
