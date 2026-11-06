# generation.py -- KV-cached decoding with copy bonus, n-sample shared-prefix caching, penalties
from __future__ import annotations
import dataclasses
import torch


def _expand_cache(c, n):
    # repeat every batch row of a KV cache n times (shared-prompt caching for MBR sampling)
    return type(c)(**{f.name: getattr(c, f.name).repeat_interleave(n, 0) for f in dataclasses.fields(c)})


def _apply_sampling(logits, temperature, top_k, top_p):
    # greedy if temperature <= 0, else temperature + optional top-k / nucleus sampling
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:
        sl, si = torch.sort(logits, descending=True, dim=-1)
        cp = sl.softmax(-1).cumsum(-1)
        remove = cp - sl.softmax(-1) > top_p                          # always keep the top token
        sl = sl.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, si, sl)
    return torch.multinomial(logits.softmax(-1), 1)


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.0, top_k=0, top_p=1.0,
             repetition_penalty=1.0, banned_ids=(0,), no_repeat_ngram=0,
             first_allowed=None, ngram_window=384, prompt_bonus=0.0, num_return=1):
    # idx (B,T) equal-length prompts -> (B*num_return, max_new_tokens), rows grouped per prompt.
    # repetition_penalty: generated tokens only | prompt_bonus: +beta on prompt tokens (copy prior)
    # num_return: prompt prefilled ONCE, KV cache copied n times (shared-prefix caching)
    model.eval()
    B, T = idx.shape
    cdt = torch.float16 if torch.is_autocast_enabled() else next(model.parameters()).dtype
    caches = model.make_caches(B, T + max_new_tokens, idx.device, dtype=cdt)
    logits = model(idx, caches=caches, pos=0)[:, -1].float()          # prefill once per prompt
    in_prompt = torch.zeros(B, logits.size(-1), dtype=torch.bool, device=idx.device).scatter_(1, idx, True)
    in_prompt[:, :16] = False                                         # never boost special tokens
    if num_return > 1:
        caches = [_expand_cache(c, num_return) for c in caches]
        logits = logits.repeat_interleave(num_return, 0)
        in_prompt = in_prompt.repeat_interleave(num_return, 0)
        idx = idx.repeat_interleave(num_return, 0)
        B = B * num_return
    out, seen = [], idx.clone()
    for i in range(max_new_tokens):
        if repetition_penalty != 1.0 and i > 0:
            gen = seen[:, T:]
            g = logits.gather(1, gen)
            logits.scatter_(1, gen, torch.where(g > 0, g / repetition_penalty, g * repetition_penalty))
        if prompt_bonus:
            logits = logits + prompt_bonus * in_prompt
        if banned_ids:
            logits[:, list(banned_ids)] = float("-inf")
        if i == 0 and first_allowed is not None:                      # start a new word
            logits[:, ~first_allowed] = float("-inf")
        if no_repeat_ngram and seen.size(1) >= no_repeat_ngram:
            n = no_repeat_ngram
            for b in range(B):
                s = seen[b, -ngram_window:].tolist(); prefix = tuple(s[-(n - 1):])
                bad = [s[j + n - 1] for j in range(len(s) - n + 1) if tuple(s[j:j + n - 1]) == prefix]
                if bad:
                    logits[b, bad] = float("-inf")
        nxt = _apply_sampling(logits, temperature, top_k, top_p)
        out.append(nxt)
        seen = torch.cat([seen, nxt], 1)
        if i + 1 < max_new_tokens:
            logits = model(nxt, caches=caches, pos=T + i)[:, -1].float()
    return torch.cat(out, 1)
