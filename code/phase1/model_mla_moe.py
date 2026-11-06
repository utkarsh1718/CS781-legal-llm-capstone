"""
model_mla_moe.py -- M2: DeepSeek-V2/V3-style decoder: Multi-head Latent Attention (MLA) +
DeepSeekMoE (fine-grained routed experts + shared expert) with auxiliary-loss-free balancing.
MLA:  q = W_q x -> [q_nope | q_rope]; c_kv = RMSNorm(W_dkv x); k_rope = RoPE(W_kr x) (shared);
      k_nope = W_uk c_kv; v = W_uv c_kv.  Decode (absorbed): score = (W_uk^T q_nope).c_kv + q_rope.k_rope
MoE:  s_i = sigmoid(u.e_i); top-k by (s_i + b_i); gate = s / sum(s_topk); y = Shared(x) + sum g_i E_i(x)
      after each optimizer step: b_i += gamma * sign(mean_load - load_i)   (no auxiliary loss)
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from lm_common import RMSNorm, SwiGLU, rope_cache, apply_rope


@dataclass
class MLAMoEConfig:
    vocab_size: int = 32000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    qk_nope_dim: int = 32          # per-head key/query part without position (from the latent)
    qk_rope_dim: int = 32          # per-head decoupled RoPE part (key shared across heads)
    v_head_dim: int = 64           # nope + rope == v dim -> SDPA's fused kernel applies
    kv_latent_dim: int = 256       # d_c: the only K/V state that is cached
    n_routed_experts: int = 16
    n_shared_experts: int = 1
    top_k: int = 3                 # shared(1) + routed(3) = 4 x 512 = 2048 active hidden (= M1)
    expert_hidden: int = 512       # fine-grained experts (M1's 2048-wide FFN split 4 ways)
    first_dense_layers: int = 1    # layer 0 keeps a dense FFN (stabilises early routing)
    dense_ffn_hidden: int = 2048
    capacity_factor: float = 1.25  # training-time slots per expert = 1.25 x average load
    bias_update_rate: float = 1e-3 # gamma of the aux-loss-free balancing
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    init_std: float = 0.02


@dataclass
class MLACache:
    c: torch.Tensor                # (B, max_T, d_c)        compressed KV latent
    kr: torch.Tensor               # (B, 1, max_T, rope)    shared RoPE key


class MLAttention(nn.Module):
    def __init__(self, c: MLAMoEConfig):
        super().__init__()
        self.h, self.nope, self.rope, self.vd, self.dc = (c.n_heads, c.qk_nope_dim, c.qk_rope_dim,
                                                          c.v_head_dim, c.kv_latent_dim)
        qk = self.nope + self.rope
        assert qk == self.vd, "keep qk head dim == v head dim so SDPA can use its fused kernel"
        self.wq = nn.Linear(c.d_model, self.h * qk, bias=False)
        self.w_dkv = nn.Linear(c.d_model, self.dc, bias=False)          # down-projection to latent
        self.w_kr = nn.Linear(c.d_model, self.rope, bias=False)         # decoupled RoPE key
        self.w_uk = nn.Linear(self.dc, self.h * self.nope, bias=False)  # latent -> keys
        self.w_uv = nn.Linear(self.dc, self.h * self.vd, bias=False)    # latent -> values
        self.wo = nn.Linear(self.h * self.vd, c.d_model, bias=False)
        self.q_norm, self.kv_norm, self.kr_norm = RMSNorm(qk), RMSNorm(self.dc), RMSNorm(self.rope)
        self.scale = qk ** -0.5

    def forward(self, x, cos, sin, cache: MLACache | None = None, pos: int = 0):
        B, T, _ = x.shape
        q = self.q_norm(self.wq(x).view(B, T, self.h, self.nope + self.rope).transpose(1, 2))
        q_nope, q_rope = q.split([self.nope, self.rope], dim=-1)
        q_rope = apply_rope(q_rope, cos[pos:pos + T], sin[pos:pos + T])
        c_kv = self.kv_norm(self.w_dkv(x))                                      # (B, T, d_c)
        k_rope = apply_rope(self.kr_norm(self.w_kr(x)).unsqueeze(1),
                            cos[pos:pos + T], sin[pos:pos + T])                 # (B, 1, T, rope)
        if cache is not None:                                                   # cache latents only
            cache.c[:, pos:pos + T] = c_kv
            cache.kr[:, :, pos:pos + T] = k_rope

        if T > 1 or cache is None:       # training / prefill: materialise K,V once, fused SDPA
            assert pos == 0, "chunked prefill is not supported"
            k_nope = self.w_uk(c_kv).view(B, T, self.h, self.nope).transpose(1, 2)
            v = self.w_uv(c_kv).view(B, T, self.h, self.vd).transpose(1, 2)
            k = torch.cat([k_nope, k_rope.expand(B, self.h, T, self.rope)], dim=-1)
            y = F.scaled_dot_product_attention(torch.cat([q_nope, q_rope], -1), k, v, is_causal=T > 1)
        else:                            # decoding: ABSORBED form, attends straight to the latents
            C = cache.c[:, :pos + 1]                                            # (B, S, d_c)
            Kr = cache.kr[:, 0, :pos + 1]                                       # (B, S, rope)
            w_uk = self.w_uk.weight.view(self.h, self.nope, self.dc)
            w_uv = self.w_uv.weight.view(self.h, self.vd, self.dc)


class Block(nn.Module):
    def __init__(self, c: MLAMoEConfig, dense: bool):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(c.d_model), RMSNorm(c.d_model)
        self.attn = MLAttention(c)
        self.ffn = SwiGLU(c.d_model, c.dense_ffn_hidden) if dense else DeepSeekMoE(c)

    def forward(self, x, cos, sin, cache=None, pos=0):
        x = x + self.attn(self.norm1(x), cos, sin, cache, pos)
        return x + self.ffn(self.norm2(x))


class MLAMoEGPT(nn.Module):
    def __init__(self, c: MLAMoEConfig):
        super().__init__()
        self.cfg = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList([Block(c, dense=i < c.first_dense_layers) for i in range(c.n_layers)])
        self.norm_f = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = rope_cache(c.max_seq_len, c.qk_rope_dim, c.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.moe_layers = [b.ffn for b in self.blocks if isinstance(b.ffn, DeepSeekMoE)]
        self.apply(self._init)
        for n, p in self.named_parameters():               # expert weights + scaled residual outputs
            if n.endswith((".w1", ".w3")):
                nn.init.normal_(p, 0.0, c.init_std)
            if n.endswith((".w2", "wo.weight", "w2.weight")):
                nn.init.normal_(p, 0.0, c.init_std / math.sqrt(2 * c.n_layers))

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, self.cfg.init_std)

    def active_params(self):
        # parameters used per token: shared + top-k routed experts (not all 16)
        c = self.cfg
        total = sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())
        inactive = len(self.moe_layers) * (c.n_routed_experts - c.top_k) * 3 * c.d_model * c.expert_hidden
        return total - inactive

    def make_caches(self, B, max_T, device, dtype=torch.float16):
        c = self.cfg
        return [MLACache(torch.zeros(B, max_T, c.kv_latent_dim, device=device, dtype=dtype),
                         torch.zeros(B, 1, max_T, c.qk_rope_dim, device=device, dtype=dtype))
                for _ in self.blocks]

    @torch.no_grad()
    def after_step(self):
        # Aux-loss-free balancing: loads summed over DDP ranks -> identical bias update everywhere
        if not self.moe_layers:
            return {}
        loads = torch.stack([m.load_accum for m in self.moe_layers])          # (L, E)
        drops = torch.stack([m.drop_accum for m in self.moe_layers]).sum(0)   # (dropped, total)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loads); dist.all_reduce(drops)
        mean = loads.mean(1, keepdim=True)
        for m, l, mu in zip(self.moe_layers, loads, mean):
            m.router_bias += self.cfg.bias_update_rate * torch.sign(mu - l)
            m.load_accum.zero_(); m.drop_accum.zero_()
        max_over_mean = (loads.max(1).values / mean.squeeze(1).clamp_min(1)).mean()
        return {"moe_max/mean": round(max_over_mean.item(), 3),
                "drop%": round(100 * (drops[0] / drops[1].clamp_min(1)).item(), 2)}

    def forward(self, idx, targets=None, caches=None, pos: int = 0):
        x = self.tok_emb(idx)
        for i, blk in enumerate(self.blocks):
            x = blk(x, self.rope_cos, self.rope_sin, None if caches is None else caches[i], pos)
        logits = self.lm_head(self.norm_f(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)


class DeepSeekMoE(nn.Module):
    # 1 shared expert + E fine-grained routed experts, top-k routing with aux-loss-free bias
    def __init__(self, c: MLAMoEConfig):
        super().__init__()
        E, d, h = c.n_routed_experts, c.d_model, c.expert_hidden
        self.E, self.k, self.cap = E, c.top_k, c.capacity_factor
        self.shared = SwiGLU(d, h * c.n_shared_experts)
        self.router = nn.Linear(d, E, bias=False)
        self.w1 = nn.Parameter(torch.empty(E, d, h))     # gate  (all experts stacked)
        self.w3 = nn.Parameter(torch.empty(E, d, h))     # up
        self.w2 = nn.Parameter(torch.empty(E, h, d))     # down
        self.register_buffer("router_bias", torch.zeros(E))                   # saved, not trained
        self.register_buffer("load_accum", torch.zeros(E), persistent=False)  # tokens per expert
        self.register_buffer("drop_accum", torch.zeros(2), persistent=False)  # (dropped, total)

    def forward(self, x):
        B, T, d = x.shape
        N, E, k = B * T, self.E, self.k
        xf = x.reshape(N, d)
        with torch.autocast("cuda", enabled=False):                          # routing in fp32
            aff = torch.sigmoid(xf.float() @ self.router.weight.float().t())  # (N, E)
        _, idx = torch.topk(aff + self.router_bias, k, dim=-1)                # bias only SELECTS
        gate = aff.gather(1, idx)
        gate = gate / gate.sum(-1, keepdim=True)                              # (N, k)

        flat_e = idx.reshape(-1)                                              # (N*k,)
        flat_tok = torch.arange(N, device=x.device).repeat_interleave(k)
        C = N if not self.training else int(math.ceil(self.cap * N * k / E))  # no drops at eval
        counts = torch.zeros(E, device=x.device).scatter_add_(0, flat_e, torch.ones_like(flat_e, dtype=torch.float))
        order = torch.argsort(flat_e, stable=True)                            # group by expert
        se, tok = flat_e[order], flat_tok[order]
        starts = torch.cumsum(counts, 0) - counts
        slot = torch.arange(N * k, device=x.device) - starts[se].long()       # position in expert
        keep = slot < C
        dest = torch.where(keep, se * C + slot, torch.full_like(se, E * C))   # overflow -> dummy row

        buf = xf.new_zeros(E * C + 1, d)
        buf[dest] = xf[tok]                                                   # dispatch
        h = buf[:E * C].view(E, C, d)
        h = F.silu(torch.bmm(h, self.w1)) * torch.bmm(h, self.w3)              # all experts, 1 bmm
        o = torch.bmm(h, self.w2).reshape(E * C, d)
        o = torch.cat([o, o.new_zeros(1, d)], 0)
        wts = (gate.reshape(-1)[order] * keep).to(o.dtype).unsqueeze(1)
        y = torch.zeros(N, d, device=x.device, dtype=o.dtype).index_add_(0, tok, o[dest] * wts)  # combine

        if self.training:                                                     # stats for balancing
            self.load_accum += counts
            self.drop_accum += torch.stack([(~keep).sum().float(), xf.new_tensor(float(N * k))])
        return self.shared(x) + y.view(B, T, d).to(x.dtype)


class MLAttention(nn.Module):
    # complete definition (replaces the truncated one above when the module loads)
    def __init__(self, c: MLAMoEConfig):
        super().__init__()
        self.h, self.nope, self.rope, self.vd, self.dc = (c.n_heads, c.qk_nope_dim, c.qk_rope_dim,
                                                          c.v_head_dim, c.kv_latent_dim)
        qk = self.nope + self.rope
        assert qk == self.vd, "keep qk head dim == v head dim so SDPA can use its fused kernel"
        self.wq = nn.Linear(c.d_model, self.h * qk, bias=False)
        self.w_dkv = nn.Linear(c.d_model, self.dc, bias=False)          # down-projection to latent
        self.w_kr = nn.Linear(c.d_model, self.rope, bias=False)         # decoupled RoPE key
        self.w_uk = nn.Linear(self.dc, self.h * self.nope, bias=False)  # latent -> keys
        self.w_uv = nn.Linear(self.dc, self.h * self.vd, bias=False)    # latent -> values
        self.wo = nn.Linear(self.h * self.vd, c.d_model, bias=False)
        self.q_norm, self.kv_norm, self.kr_norm = RMSNorm(qk), RMSNorm(self.dc), RMSNorm(self.rope)
        self.scale = qk ** -0.5

    def forward(self, x, cos, sin, cache=None, pos=0):
        B, T, _ = x.shape
        q = self.q_norm(self.wq(x).view(B, T, self.h, self.nope + self.rope).transpose(1, 2))
        q_nope, q_rope = q.split([self.nope, self.rope], dim=-1)
        q_rope = apply_rope(q_rope, cos[pos:pos + T], sin[pos:pos + T])
        c_kv = self.kv_norm(self.w_dkv(x))                                      # (B, T, d_c)
        k_rope = apply_rope(self.kr_norm(self.w_kr(x)).unsqueeze(1),
                            cos[pos:pos + T], sin[pos:pos + T])                 # (B, 1, T, rope)
        if cache is not None:                                                   # cache latents only
            cache.c[:, pos:pos + T] = c_kv
            cache.kr[:, :, pos:pos + T] = k_rope

        if T > 1 or cache is None:       # training / prefill: materialise K,V once, fused SDPA
            assert pos == 0, "chunked prefill is not supported"
            k_nope = self.w_uk(c_kv).view(B, T, self.h, self.nope).transpose(1, 2)
            v = self.w_uv(c_kv).view(B, T, self.h, self.vd).transpose(1, 2)
            k = torch.cat([k_nope, k_rope.expand(B, self.h, T, self.rope)], dim=-1)
            y = F.scaled_dot_product_attention(torch.cat([q_nope, q_rope], -1), k, v, is_causal=T > 1)
        else:                            # decoding: ABSORBED form, attends straight to the latents
            Cc = cache.c[:, :pos + 1]                                           # (B, S, d_c)
            Kr = cache.kr[:, 0, :pos + 1]                                       # (B, S, rope)
            w_uk = self.w_uk.weight.view(self.h, self.nope, self.dc)
            w_uv = self.w_uv.weight.view(self.h, self.vd, self.dc)
            q_lat = torch.einsum("bhtn,hnc->bhtc", q_nope, w_uk.to(q_nope.dtype))   # W_uk^T q
            s = (torch.einsum("bhtc,bsc->bhts", q_lat, Cc.to(q_lat.dtype)) +
                 torch.einsum("bhtr,bsr->bhts", q_rope, Kr.to(q_rope.dtype)))
            p = torch.softmax(s.float() * self.scale, dim=-1).to(q_lat.dtype)
            ctx = torch.einsum("bhts,bsc->bhtc", p, Cc.to(p.dtype))            # (B, H, 1, d_c)
            y = torch.einsum("bhtc,hvc->bhtv", ctx, w_uv.to(ctx.dtype))         # W_uv . ctx
        return self.wo(y.transpose(1, 2).reshape(B, T, self.h * self.vd))
