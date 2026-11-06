# model_copy.py -- M4: dense decoder (M1 backbone) + a learned LEGAL COPY HEAD.
# Legal continuations reuse the prompt heavily (party names, section numbers, quoted phrases);
# the decoding study showed a fixed copy bonus helps up to beta~1 and then hurts. M4 LEARNS it.
# Output = mixture of two distributions, mixed by a per-token gate g_t = sigmoid(w . h_t + b):
#   p(y_t) = (1 - g_t) * softmax(W h_t)[y_t]            (generate from the vocabulary)
#          +      g_t  * sum_{i<t, y_i = y_t} a_{t,i}     (copy: point at an earlier position)
# a_{t,i} = softmax_i(q_t . k_i / sqrt(d_c)) over i < t, q_t = Wq h_t, k_i = Wk h_i.
# Induction-style pointer: key i is the state that PRECEDED token y_i = x_{i+1}, so the head
# learns "the context now looks like the context before that earlier token -> copy it".
# Strictly i < t, so the target y_t is never visible. Extra parameters: 2*768*128 + 769 ~ 0.2M.
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_dense import DenseGPT, DenseConfig

NEG = -1e4                                    # finite "minus infinity" (keeps gradients NaN-free)


@dataclass
class CopyConfig(DenseConfig):
    copy_dim: int = 128                       # query/key size of the pointer attention
    gate_bias_init: float = -2.0              # sigmoid(-2) ~ 0.12: start mostly generating


@dataclass
class CopyCache:
    k: torch.Tensor                           # pointer keys  (B, max_T, copy_dim)
    ids: torch.Tensor                         # input token ids (B, max_T); value of key i = ids[i+1]


class CopyGPT(DenseGPT):
    def __init__(self, c: CopyConfig):
        super().__init__(c)                                          # M1 backbone + its init
        self.copy_q = nn.Linear(c.d_model, c.copy_dim, bias=False)
        self.copy_k = nn.Linear(c.d_model, c.copy_dim, bias=False)
        self.gate = nn.Linear(c.d_model, 1)
        for m in (self.copy_q, self.copy_k, self.gate):
            nn.init.normal_(m.weight, 0.0, c.init_std)
        nn.init.constant_(self.gate.bias, c.gate_bias_init)

    def _hidden(self, idx, caches=None, pos=0):
        x = self.tok_emb(idx)
        for i, blk in enumerate(self.blocks):
            x = blk(x, self.rope_cos, self.rope_sin, None if caches is None else caches[i], pos)
        return self.norm_f(x)

    def make_caches(self, B, max_T, device, dtype=torch.float16):
        # M1's per-layer KV caches + one pointer cache (keys + token ids) at the end
        return super().make_caches(B, max_T, device, dtype) + [
            CopyCache(torch.zeros(B, max_T, self.cfg.copy_dim, device=device, dtype=dtype),
                      torch.zeros(B, max_T, dtype=torch.long, device=device))]

    def copy_loss(self, h, targets):
        """Mean NLL of the gated mixture (a proper probability -> comparable to M1's loss)."""
        B, T, _ = h.shape
        z = self.lm_head(h)
        lp_vocab = -F.cross_entropy(z.float().view(-1, z.size(-1)), targets.reshape(-1),
                                    reduction="none").view(B, T)           # log p_vocab(y_t)
        with torch.autocast("cuda", enabled=False):                        # pointer math in fp32
            hf = h.float()
            allowed = torch.ones(T, T, dtype=torch.bool, device=h.device).tril(-1)  # i < t
            g = self.gate(hf).squeeze(-1).masked_fill(~allowed.any(-1), NEG)  # t=0: nothing to copy
            s = self.copy_q(hf) @ self.copy_k(hf).transpose(1, 2) / math.sqrt(self.cfg.copy_dim)
            la = torch.log_softmax(s.masked_fill(~allowed, NEG), -1)       # log a_{t,i}
            match = targets[:, None, :] == targets[:, :, None]              # [b,t,i]: y_i == y_t
            lp_copy = torch.logsumexp(la.masked_fill(~(match & allowed), NEG), -1)   # (B, T)
            lp = torch.logaddexp(F.logsigmoid(-g) + lp_vocab, F.logsigmoid(g) + lp_copy)
        return -lp.mean()

    def forward(self, idx, targets=None, caches=None, pos: int = 0):
        if caches is None:
            h = self._hidden(idx)
            if targets is None:
                return self.lm_head(h)                                      # vocab-only logits
            return self.copy_loss(h, targets)
        # ---- generation: returns LOG-PROBS of the mixture for the last position, (B, 1, V) ----
        h = self._hidden(idx, caches, pos)
        cc, T = caches[-1], idx.size(1)
        cc.k[:, pos:pos + T] = self.copy_k(h).to(cc.k.dtype)
        cc.ids[:, pos:pos + T] = idx
        t = pos + T - 1                                                     # current position
        if pos == 0:
            self._prompt_len = T                                            # remember where the prompt ends
        hl = h[:, -1:]
        z = self.lm_head(hl).float()
        with torch.autocast("cuda", enabled=False):
            out = torch.log_softmax(z, -1)                                  # t=0: nothing to copy
            if t >= 1:
                hf = hl.float()
                g = self.gate(hf) + getattr(self, "copy_gate_bias", 0.0)   # decode-time gate shift
                out = out + F.logsigmoid(-g)
                # copy_prompt_only: point only at prompt tokens (no copying of the model's own output)
                n = min(t, self._prompt_len - 1) if getattr(self, "copy_prompt_only", False) else t
                if n >= 1:
                    a = torch.softmax(self.copy_q(hf) @ cc.k[:, :n].float().transpose(1, 2)
                                      / math.sqrt(self.cfg.copy_dim), -1)   # (B, 1, n), keys i < n
                    pc = torch.zeros_like(out).scatter_add_(2, cc.ids[:, None, 1:n + 1], a)
                    out = torch.logaddexp(out, F.logsigmoid(g) + torch.log(pc))
            out = out + torch.logsumexp(z, -1, keepdim=True)            # logit scale, as M1
        return out
