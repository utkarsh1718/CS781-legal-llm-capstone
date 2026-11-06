"""
p6_sum.py -- Phase 6 (abstractive legal summarisation) on ANY of the four Phase-1 models.

Strategy ("window-level summarisation"): every judgment is cut into ~520-word windows (<= 640 tokens);
each window is trained to produce ITS part of the reference summary (the summary sentences that
overlap it most, built in P6-1; empty for windows the summary skips). At test time every window is
summarised and the pieces are joined in document order -> output length scales with input length.

  decoder-only  M1 / M2 / M4 : [window] <sep> [target] <eos>         loss on target + <eos> only
  encoder-dec   M3           : encoder <- window, decoder <- <bos> target, predicts target <eos>
  M4 (copy head): exact mixture NLL (vocabulary vs pointer into the window), masked to the target

Adaptation methods (apply_method): full | lora | lorma_plus | lorma_pi | houlsby (adapter after the
attention AND the FFN sub-layer) | pfeiffer (adapter after the FFN only) | prompt (soft prompt,
M1/M2/M3) | qlora (frozen weights quantised to 4-bit NF4, double-packed in uint8, + LoRA).
Knowledge distillation: student (M1 with 6 of 12 layers) <- teacher logits (word-level KD).
Everything is plain PyTorch; no PEFT / transformers library.
"""
from __future__ import annotations

import contextlib
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import p2_peft as pp

EOS, PAD, SEP, BOS = 0, 1, 3, 5
IN_MAX, TG_MAX = 640, 384
DEC_ONLY = ("dense", "mla_moe", "copy")


# ----------------------------------------------------------------------------- data
class Chunks:
    """Window-level pairs from P6-1 (data_p6/chunks/<split>.npz)."""
    def __init__(self, path):
        z = np.load(path)
        self.in_ids, self.in_off = z["in_ids"], z["in_off"]
        self.has_tg = "tg_ids" in z
        if self.has_tg:
            self.tg_ids, self.tg_off = z["tg_ids"], z["tg_off"]
        self.doc, self.pos = z["doc"], z["pos"]
        self.n = len(self.doc)

    def inp(self, i):
        return self.in_ids[self.in_off[i]:self.in_off[i + 1]].astype(np.int64)

    def tgt(self, i):
        return self.tg_ids[self.tg_off[i]:self.tg_off[i + 1]].astype(np.int64) if self.has_tg else np.zeros(0, np.int64)

    def windows_of(self, docs):
        s = set(docs)
        return [i for i in range(self.n) if self.doc[i] in s]


class Trunc(Chunks):
    """Long-document BASELINE: one pair per document = first 640 document tokens -> first 384
    summary tokens (data_p6/tok/<split>_{doc,sum}.npz)."""
    def __init__(self, doc_npz, sum_npz=None):
        d = np.load(doc_npz); ids, off = d["ids"], d["offsets"]; n = len(off) - 1
        I = [ids[off[i]:off[i + 1]][:IN_MAX] for i in range(n)]
        self.in_ids = np.concatenate(I); self.in_off = np.concatenate([[0], np.cumsum([len(a) for a in I])])
        self.has_tg = sum_npz is not None
        if self.has_tg:
            s = np.load(sum_npz); sids, soff = s["ids"], s["offsets"]
            T = [sids[soff[i]:soff[i + 1]][:TG_MAX] for i in range(n)]
            self.tg_ids = np.concatenate(T); self.tg_off = np.concatenate([[0], np.cumsum([len(a) for a in T])])
        self.doc, self.pos, self.n = np.arange(n), np.zeros((n, 2), dtype=np.int64), n


# ----------------------------------------------------------------------------- models
def load_lm(ckpt):
    from train import build_model
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = build_model(ck["arch"], ck["model_cfg"])
    m.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    m.arch = ck["arch"]; m.n_soft = 0
    return m


def layers(m):
    return list(m.enc) + list(m.dec) if m.arch == "encdec" else list(m.blocks)


def shrink(m, keep=(0, 2, 4, 6, 8, 10)):
    """KD student: keep every other layer of a decoder (DistilBERT-style initialisation)."""
    m.blocks = nn.ModuleList([m.blocks[i] for i in keep])
    return m


# ----------------------------------------------------------------------------- adaptation methods
LORA_TARGETS = ("wq", "wv", "w_uv", "wo")


def _wrap(m, make, targets):
    n = 0
    for L in layers(m):
        for mod in list(L.modules()):
            for name in targets:
                lin = getattr(mod, name, None)
                if lin is not None and hasattr(lin, "in_features") and not isinstance(lin, (pp.LoRALinear,)):
                    setattr(mod, name, make(lin)); n += 1
    return n


class Adapter(nn.Module):
    """Bottleneck adapter on a sub-layer's output: y + U relu(D y), U = 0 at start."""
    def __init__(self, sub, d, bottleneck=64):
        super().__init__()
        self.sub = sub
        self.down, self.up = nn.Linear(d, bottleneck), nn.Linear(bottleneck, d)
        nn.init.normal_(self.down.weight, 0.0, 0.02); nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, *a, **kw):
        y = self.sub(*a, **kw)
        return y + self.up(F.relu(self.down(y)))


def add_adapters(m, kind, bottleneck=64):
    """houlsby: after attention AND FFN sub-layers (Houlsby et al. 2019); pfeiffer: after the FFN
    only (Pfeiffer et al. 2021)."""
    d, n = m.cfg.d_model, 0
    attn_names = ("attn", "self_attn") if kind == "houlsby" else ()
    for L in layers(m):
        for name in attn_names + ("mlp", "ffn"):
            sub = getattr(L, name, None)
            if sub is not None:
                setattr(L, name, Adapter(sub, d, bottleneck)); n += 1
    return n


class SoftPrompt(nn.Module):
    """Prompt tuning (Lester et al. 2021): n trainable vectors prepended to the INPUT side only.
    Active only while `on` is set (prefill / encoder), never for later decoding steps."""
    def __init__(self, emb, n=20):
        super().__init__()
        self.emb, self.n, self.on = emb, n, False
        idx = torch.randint(1000, 30000, (n,))
        self.prompt = nn.Parameter(emb.weight[idx].detach().clone())

    @property
    def weight(self):                                                  # keeps lm_head tying intact
        return self.emb.weight

    def forward(self, ids):
        x = self.emb(ids)
        if not self.on:
            return x
        return torch.cat([self.prompt.unsqueeze(0).expand(x.size(0), -1, -1).to(x.dtype), x], 1)


@contextlib.contextmanager
def soft_on(m):
    sp = m.tok_emb if isinstance(m.tok_emb, SoftPrompt) else None
    if sp is not None:
        sp.on = True
    try:
        yield
    finally:
        if sp is not None:
            sp.on = False


NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])


class NF4Linear(nn.Module):
    """QLoRA base layer (Dettmers et al. 2023): frozen weight stored as 4-bit NormalFloat codes,
    two codes per uint8, one absmax scale per block of 64 weights; de-quantised on the fly."""
    def __init__(self, lin: nn.Linear, block=64):
        super().__init__()
        W = lin.weight.detach().float()
        self.out_features, self.in_features, self.block = W.shape[0], W.shape[1], block
        flat = W.reshape(-1); pad = (-flat.numel()) % block
        flat = torch.cat([flat, flat.new_zeros(pad)]).view(-1, block)
        scale = flat.abs().amax(1, keepdim=True).clamp(min=1e-8)
        q = flat / scale
        codes = torch.cat([(q[i:i + 4096, :, None] - NF4.to(q.device)).abs().argmin(-1)
                           for i in range(0, q.size(0), 4096)]).to(torch.uint8).view(-1)
        if codes.numel() % 2:
            codes = torch.cat([codes, codes.new_zeros(1)])
        self.register_buffer("packed", (codes[0::2] << 4) | codes[1::2])
        self.register_buffer("scale", scale.half())
        self.register_buffer("lut", NF4.clone())
        self.n = W.numel()

    def weight_fp(self, dtype):
        c = torch.stack([self.packed >> 4, self.packed & 15], 1).view(-1).long()
        w = (self.lut[c].view(-1, self.block) * self.scale.float()).view(-1)[:self.n]
        return w.view(self.out_features, self.in_features).to(dtype)

    def forward(self, x):
        return F.linear(x, self.weight_fp(x.dtype))


def quantize_nf4(m):
    n = 0
    for L in layers(m):
        for mod in list(L.modules()):
            for name, child in list(mod.named_children()):
                if isinstance(child, nn.Linear) and name != "router":
                    setattr(mod, name, NF4Linear(child)); n += 1
    return n


class IA3Out(nn.Module):
    """IA^3: y = (W x) * l  (rescales keys / values)."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.l = nn.Parameter(torch.ones(base.out_features))

    def forward(self, x):
        return self.base(x) * self.l


class IA3In(nn.Module):
    """IA^3: y = W (x * l)  (rescales the FFN intermediate activation before the down-projection)."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.l = nn.Parameter(torch.ones(base.in_features))

    def forward(self, x):
        return self.base(x * self.l)


def apply_method(m, method, r=16, alpha=32, bottleneck=64, n_prompt=20):
    """Freeze the Phase-1 model and add the method's trainable parameters."""
    if method == "full":
        for p in m.parameters():
            p.requires_grad = True
        return m
    for p in m.parameters():
        p.requires_grad = False
    if method == "lora":
        _wrap(m, lambda l: pp.LoRALinear(l, r, alpha), LORA_TARGETS)
    elif method == "lorma_plus":
        _wrap(m, lambda l: pp.LoRMAPlusLinear(l, r, alpha), LORA_TARGETS)
    elif method == "lorma_pi":
        _wrap(m, lambda l: pp.LoRMAPiLinear(l, r), LORA_TARGETS)
    elif method in ("houlsby", "pfeiffer"):
        add_adapters(m, method, bottleneck)
    elif method == "prompt":
        assert m.arch != "copy", "prompt tuning is not wired into M4's pointer cache"
        m.tok_emb = SoftPrompt(m.tok_emb, n_prompt); m.n_soft = n_prompt
        if m.arch == "encdec":                                         # M3: soft prompt on the ENCODER only
            enc = m.encode

            def encode_with_prompt(src):
                with soft_on(m):
                    return enc(src)
            m.encode = encode_with_prompt
    elif method == "qlora":
        quantize_nf4(m)
        _wrap(m, lambda l: pp.LoRALinear(l, r, alpha), LORA_TARGETS)
    elif method == "ia3":                                             # IA^3 (Liu et al., 2022)
        for L in layers(m):
            for mod in list(L.modules()):
                for name in ("wk", "wv", "w_uk", "w_uv"):              # keys / values (dense + MLA)
                    lin = getattr(mod, name, None)
                    if isinstance(lin, nn.Linear):
                        setattr(mod, name, IA3Out(lin))
                if type(mod).__name__ == "SwiGLU" and isinstance(getattr(mod, "w2", None), nn.Linear):
                    mod.w2 = IA3In(mod.w2)                             # FFN intermediate activation
    else:
        raise ValueError(method)
    if m.arch == "copy":                                               # M4's small copy head always learns
        for mod in (m.copy_q, m.copy_k, m.gate):
            for p in mod.parameters():
                p.requires_grad = True
    return m


def n_params(m):
    seen = {id(p): p for p in m.parameters()}
    tot = sum(p.numel() for p in seen.values())
    tr = sum(p.numel() for p in seen.values() if p.requires_grad)
    mem = sum(p.numel() * p.element_size() for p in seen.values()) + \
        sum(b.numel() * b.element_size() for n, b in m.named_buffers() if "rope" not in n)
    return tr, tot, mem


# ----------------------------------------------------------------------------- batching + loss
def make_batch(arch, pairs, device):
    """pairs: [(input ids, target ids)]."""
    if arch == "encdec":
        L = min(len(x) for x, _ in pairs)                              # no encoder padding: trim to the
        src = torch.tensor(np.stack([x[:L] for x, _ in pairs]))       # shortest window of the batch
        Td = max(len(y) for _, y in pairs) + 1
        din = torch.full((len(pairs), Td), PAD, dtype=torch.long); y = torch.full((len(pairs), Td), -1, dtype=torch.long)
        for b, (_, t) in enumerate(pairs):
            seq = list(t) + [EOS]
            din[b, :len(seq)] = torch.tensor([BOS] + list(t)); y[b, :len(seq)] = torch.tensor(seq)
        return {"src": src.to(device), "din": din.to(device), "y": y.to(device)}
    seqs = [np.concatenate([x, [SEP], t, [EOS]]) for x, t in pairs]
    T = max(len(s) for s in seqs) - 1
    x = torch.full((len(pairs), T), PAD, dtype=torch.long)
    y = torch.full((len(pairs), T), -1, dtype=torch.long)              # loss targets (-1 = ignored)
    yf = torch.full((len(pairs), T), -2, dtype=torch.long)             # next token everywhere (copy keys)
    for b, (s, (inp, _)) in enumerate(zip(seqs, pairs)):
        s = torch.tensor(s); n = len(s) - 1
        x[b, :n] = s[:-1]; yf[b, :n] = s[1:]
        y[b, len(inp):n] = s[1:][len(inp):]                            # from <sep> onwards: target + <eos>
    return {"x": x.to(device), "y": y.to(device), "yf": yf.to(device)}


def _copy_nll(m, h, yf, y):
    """M4 mixture NLL with a loss mask (Phase-1 copy_loss assumes every position is a target)."""
    B, T, _ = h.shape
    z = m.lm_head(h)
    lp_vocab = -F.cross_entropy(z.float().view(-1, z.size(-1)), yf.clamp(min=0).view(-1), reduction="none").view(B, T)
    with torch.autocast("cuda", enabled=False):
        hf = h.float()
        allowed = torch.ones(T, T, dtype=torch.bool, device=h.device).tril(-1)
        g = m.gate(hf).squeeze(-1).masked_fill(~allowed.any(-1), -1e4)
        s = m.copy_q(hf) @ m.copy_k(hf).transpose(1, 2) / math.sqrt(m.cfg.copy_dim)
        la = torch.log_softmax(s.masked_fill(~allowed, -1e4), -1)
        match = (yf[:, None, :] == yf[:, :, None]) & (yf[:, None, :] >= 0)
        lp_copy = torch.logsumexp(la.masked_fill(~(match & allowed), -1e4), -1)
        lp = torch.logaddexp(F.logsigmoid(-g) + lp_vocab, F.logsigmoid(g) + lp_copy)
    mask = y >= 0
    return -(lp * mask).sum() / mask.sum().clamp(min=1)


def logits_and_targets(m, batch):
    """(logits at loss positions (N, V), targets (N,)) for CE / KD; M4 handled by loss()."""
    if m.arch == "encdec":
        mem = m.encode(batch["src"])                                   # (soft prompt, if any, added inside)
        z = m.decode_tf(mem, batch["din"])
        y = batch["y"]
    else:
        with soft_on(m):
            z = m(batch["x"])
        z = z[:, m.n_soft:]
        y = batch["y"]
    mask = y >= 0
    return z[mask], y[mask]


def loss(m, batch):
    if m.arch == "copy":
        return _copy_nll(m, m._hidden(batch["x"]), batch["yf"], batch["y"])
    z, y = logits_and_targets(m, batch)
    return F.cross_entropy(z.float(), y)


# ----------------------------------------------------------------------------- training
def batches(data, idx, bs, seed, drop_empty=0.5):
    """Length-bucketed batches; drop a share of the empty-target windows (they are 25% of all)."""
    rng = random.Random(seed)
    keep = [i for i in idx if data.tg_off[i + 1] > data.tg_off[i] or rng.random() >= drop_empty]
    rng.shuffle(keep)
    out = []
    for c in range(0, len(keep), bs * 50):                             # sort inside blocks of 50 batches
        blk = sorted(keep[c:c + bs * 50], key=lambda i: data.in_off[i + 1] - data.in_off[i] +
                     data.tg_off[i + 1] - data.tg_off[i])
        out += [blk[b:b + bs] for b in range(0, len(blk), bs)]
    rng.shuffle(out)
    return out


def train(m, data, idx, epochs=1, lr=2e-4, bs=8, device="cuda", seed=0, warmup=0.03, teacher=None,
          kd_T=2.0, kd_lam=0.5, max_steps=None, log=print):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    m.to(device); params = [p for p in m.parameters() if p.requires_grad]
    if teacher is not None:
        teacher.to(device).eval()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    plan = [b for ep in range(epochs) for b in batches(data, idx, bs, seed + ep)]
    if max_steps:
        plan = plan[:max_steps]
    steps = len(plan); warm = max(1, int(warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    tr, tot, _ = n_params(m)
    log(f"trainable {tr/1e6:.3f}M of {tot/1e6:.1f}M | {steps} steps" + (" | KD" if teacher is not None else ""))
    t0, run, hist = time.time(), 0.0, []
    for step, b in enumerate(plan):
        m.train()
        batch = make_batch(m.arch, [(data.inp(i), data.tgt(i)) for i in b], device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
            if teacher is None:
                L = loss(m, batch)
            else:
                zs, y = logits_and_targets(m, batch)
                with torch.no_grad():
                    zt, _ = logits_and_targets(teacher, batch)
                ce = F.cross_entropy(zs.float(), y)
                kl = F.kl_div(F.log_softmax(zs.float() / kd_T, -1), F.log_softmax(zt.float() / kd_T, -1),
                              log_target=True, reduction="batchmean") * kd_T * kd_T
                L = (1 - kd_lam) * ce + kd_lam * kl
        opt.zero_grad(set_to_none=True)
        scaler.scale(L).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); sched.step(); run += L.item()
        if hasattr(m, "after_step"):
            m.after_step()                                             # M2: aux-loss-free expert balancing
        if (step + 1) % 250 == 0:
            hist.append({"step": step + 1, "loss": run / 250})
            log(f"   step {step+1}/{steps} | loss {run/250:.4f} | {time.time()-t0:.0f}s"); run = 0.0
    return hist


@torch.no_grad()
def val_loss(m, data, idx, bs=8, device="cuda"):
    """Mean target NLL on held-out windows (teacher forcing) -> a cheap, length-free quality signal."""
    m.eval(); tot, n = 0.0, 0
    for c in range(0, len(idx), bs):
        b = idx[c:c + bs]
        batch = make_batch(m.arch, [(data.inp(i), data.tgt(i)) for i in b], device)
        with torch.autocast("cuda", dtype=torch.float16):
            tot += loss(m, batch).item() * len(b); n += len(b)
    return tot / max(1, n)


# ----------------------------------------------------------------------------- generation
@torch.no_grad()
def generate(m, data, idx, max_new=TG_MAX, bs=16, eos_bias=0.0, no_repeat=4, device="cuda", log=None):
    """Greedy KV-cached decoding of every window in idx; stops a batch when all rows emitted <eos>.
    eos_bias > 0 -> shorter outputs, < 0 -> longer. no_repeat: block repeated n-grams.
    Returns {window index: generated token list}."""
    m.eval(); out = {}
    order = sorted(idx, key=lambda i: data.in_off[i + 1] - data.in_off[i])
    for c in range(0, len(order), bs):
        b = order[c:c + bs]; B = len(b)
        L = min(int(data.in_off[i + 1] - data.in_off[i]) for i in b)
        src = torch.tensor(np.stack([data.inp(i)[:L] for i in b]), device=device)
        prompt = src if m.arch == "encdec" else torch.cat([src, torch.full((B, 1), SEP, device=device)], 1)
        T = prompt.size(1); ns = 0 if m.arch == "encdec" else m.n_soft
        with torch.autocast("cuda", dtype=torch.float16):
            caches = m.make_caches(B, T + ns + max_new + 2, device, dtype=torch.float16)
            with (soft_on(m) if m.arch != "encdec" else contextlib.nullcontext()):   # M3: encode() adds it
                z = m(prompt, caches=caches, pos=0)[:, -1].float()
            gen = torch.zeros(B, 0, dtype=torch.long, device=device)
            done = torch.zeros(B, dtype=torch.bool, device=device)
            for t in range(max_new):
                z[:, 1:16] = float("-inf")                                 # specials other than <eos>
                z[:, EOS] += eos_bias
                if no_repeat and gen.size(1) >= no_repeat - 1:
                    for r in range(B):
                        s = gen[r].tolist(); pre = tuple(s[-(no_repeat - 1):])
                        bad = [s[j + no_repeat - 1] for j in range(len(s) - no_repeat + 1) if tuple(s[j:j + no_repeat - 1]) == pre]
                        if bad:
                            z[r, bad] = float("-inf")
                nxt = z.argmax(-1)
                nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
                gen = torch.cat([gen, nxt[:, None]], 1); done |= nxt == EOS
                if bool(done.all()):
                    break
                z = m(nxt[:, None], caches=caches, pos=T + ns + t)[:, -1].float()
        for r, i in enumerate(b):
            s = gen[r].tolist()
            out[i] = s[:s.index(EOS)] if EOS in s else s
    return out


def join_summaries(data, gen, docs, decode, dedupe=True):
    """Per document: the windows' outputs in document order, joined; optionally drop sentences
    already produced by an earlier window (redundancy reduction)."""
    import re
    res = {}
    for d in docs:
        wins = sorted([i for i in gen if data.doc[i] == d], key=lambda i: data.pos[i][0])
        sents, seen = [], set()
        for i in wins:
            for s in re.split(r"(?<=[.;:?!])\s+", decode(gen[i]).strip()):
                k = " ".join(s.lower().split())
                if s and (not dedupe or k not in seen):
                    sents.append(s); seen.add(k)
        res[d] = " ".join(sents).strip() or "."
    return res


def rouge_scores(refs, hyps):
    """ROUGE-1/2/L F1 (rouge-score package, stemming) + generated/reference length ratio."""
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    R = [sc.score(r, h) for r, h in zip(refs, hyps)]
    out = {k: float(np.mean([x[k].fmeasure for x in R])) for k in ("rouge1", "rouge2", "rougeL")}
    out["len_ratio"] = float(np.mean([len(h.split()) / max(1, len(r.split())) for r, h in zip(refs, hyps)]))
    return out
