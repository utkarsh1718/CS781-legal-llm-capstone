"""
p2_peft.py -- adaptation methods for the Phase-2 tagger (all four backbones) + knowledge distillation.

  apply_method(tagger, method, **kw) freezes the backbone and adds the method's trainable parts
    "full"      : everything trainable (reference)
    "partial"   : top-k transformer layers + head
    "normgain"  : BitFit-style. Our models have no bias vectors, so the analogous tiny parameter
                  set is the RMSNorm gains (~19k params) + head
    "lora"      : W x + (a/r) B A x                     on query + value projections (Hu et al. 2022)
    "lorma_plus": (I + (a/r) B A) W x  = y + (a/r) B (A y),  B=0, A random   (LoRMA+, Bihany et al. 2025)
    "lorma_pi"  : I_pi(B A) W x, rows of BA cyclically rotated (row i by i); B[:,0]=1, A[0,0]=1, so
                  I_pi(BA) = I at start (LoRMA-pi)
    "adapter"   : bottleneck adapter after every layer, y + U relu(D y), U=0 (Houlsby/Pfeiffer)
    "prompt"    : P trainable soft-prompt vectors prepended to the input (Lester et al. 2021)
  train_kd(student, teacher, ...)   : CE + T^2 KL(teacher || student) on soft tag distributions
  shrink_dense(tagger, keep)        : student = M1/M4 with only the given layers kept (DistilBERT init)
"""
from __future__ import annotations

import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import p2_ner as ner
import p2_model as pm

TARGETS = ("wq", "wv", "w_uv")          # query + value projections (w_uv = MLA's value up-projection)


def _layers(tagger):
    bb = tagger.backbone
    return bb.enc if tagger.arch == "encdec" else bb.blocks


def count(tagger):
    tot = sum(p.numel() for p in {id(p): p for p in tagger.parameters()}.values())
    tr = sum(p.numel() for p in tagger.parameters() if p.requires_grad)
    return tr, tot


# ----------------------------------------------------------------------------- low-rank layers
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=8):
        super().__init__()
        self.base, self.s = base, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + self.s * F.linear(F.linear(x, self.A), self.B)


class LoRMAPlusLinear(nn.Module):
    """h = (I + (a/r) BA) W0 x  -- computed as y + s * B (A y) with y = W0 x (cost like LoRA)."""
    def __init__(self, base: nn.Linear, r=8, alpha=8):
        super().__init__()
        d = base.out_features
        self.base, self.s = base, alpha / r
        self.A = nn.Parameter(torch.empty(r, d)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(d, r))                       # B = 0 -> identity at start

    def forward(self, x):
        y = self.base(x)
        return y + self.s * F.linear(F.linear(y, self.A), self.B)


class LoRMAPiLinear(nn.Module):
    """h = I_pi(BA) W0 x, I_pi rotates row i of BA cyclically by i (rank inflation)."""
    def __init__(self, base: nn.Linear, r=8):
        super().__init__()
        d = base.out_features
        self.base = base
        B = torch.randn(d, r) * 0.02; B[:, 0] = 1.0                      # first column of B = ones
        A = torch.zeros(r, d); A[0, 0] = 1.0                             # A[0,0] = 1, rest 0
        self.B, self.A = nn.Parameter(B), nn.Parameter(A)
        i = torch.arange(d)
        self.register_buffer("rot", (i[None, :] - i[:, None]) % d, persistent=False)  # new[i,j]=old[i,j-i]

    def forward(self, x):
        M = (self.B @ self.A).gather(1, self.rot)                        # I_pi(BA), = I at init
        return F.linear(self.base(x), M.to(x.dtype))                    # (M y) for every token


def _wrap_linears(tagger, make):
    n = 0
    for layer in _layers(tagger):
        for mod in list(layer.modules()):              # snapshot: we replace children
            for name in TARGETS:
                lin = getattr(mod, name, None)
                if isinstance(lin, nn.Linear):
                    setattr(mod, name, make(lin)); n += 1
    return n


# ----------------------------------------------------------------------------- adapters + prompts
class AdapterLayer(nn.Module):
    """Wraps a transformer layer: out = layer(x, ...) ; out + U relu(D out) (U = 0 at start)."""
    def __init__(self, layer, d, bottleneck=64):
        super().__init__()
        self.layer = layer
        self.down, self.up = nn.Linear(d, bottleneck), nn.Linear(bottleneck, d)
        nn.init.normal_(self.down.weight, 0.0, 0.02); nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, x, *args, **kw):
        y = self.layer(x, *args, **kw)
        return y + self.up(F.relu(self.down(y)))


class PromptEmbedding(nn.Module):
    def __init__(self, emb: nn.Embedding, n_prompt=20):
        super().__init__()
        self.emb, self.n = emb, n_prompt
        idx = torch.randint(1000, 30000, (n_prompt,))                   # init from real token embeddings
        self.prompt = nn.Parameter(emb.weight[idx].detach().clone())

    def forward(self, ids):
        x = self.emb(ids)
        return torch.cat([self.prompt.unsqueeze(0).expand(x.size(0), -1, -1).to(x.dtype), x], 1)


class DropPrefix(nn.Module):
    """Head wrapper: drops the soft-prompt positions before tagging."""
    def __init__(self, head, n):
        super().__init__()
        self.head, self.n, self.out_features = head, n, head.out_features

    def forward(self, h):
        return self.head(h[:, self.n:])


# ----------------------------------------------------------------------------- dispatcher
def apply_method(tagger, method, r=8, alpha=8, top_k=4, bottleneck=64, n_prompt=20):
    if method == "full":
        return tagger
    for p in tagger.backbone.parameters():
        p.requires_grad = False
    if method == "partial":
        for layer in list(_layers(tagger))[-top_k:]:
            for p in layer.parameters():
                p.requires_grad = True
    elif method == "normgain":
        for n_, p in tagger.backbone.named_parameters():
            if "norm" in n_:
                p.requires_grad = True
    elif method == "lora":
        _wrap_linears(tagger, lambda l: LoRALinear(l, r, alpha))
    elif method == "lorma_plus":
        _wrap_linears(tagger, lambda l: LoRMAPlusLinear(l, r, alpha))
    elif method == "lorma_pi":
        _wrap_linears(tagger, lambda l: LoRMAPiLinear(l, r))
    elif method == "adapter":
        L = _layers(tagger); d = tagger.backbone.cfg.d_model
        for i in range(len(L)):
            L[i] = AdapterLayer(L[i], d, bottleneck)
    elif method == "prompt":
        bb = tagger.backbone
        bb.tok_emb = PromptEmbedding(bb.tok_emb, n_prompt)
        tagger.head = DropPrefix(tagger.head, n_prompt)
    else:
        raise ValueError(method)
    for p in tagger.head.parameters():
        p.requires_grad = True
    return tagger


# ----------------------------------------------------------------------------- distillation
def shrink_dense(tagger, keep=(0, 2, 4, 6, 8, 10)):
    """Student from a decoder tagger: keep only the listed layers (every other layer, as in
    DistilBERT's initialisation)."""
    bb = tagger.backbone
    bb.blocks = nn.ModuleList([bb.blocks[i] for i in keep])
    return tagger


def train_kd(student, teacher, train_ex, dev_docs, enc, epochs=5, lr=1e-4, head_lr=1e-3, bs=8,
             T=2.0, lam=0.5, device="cuda", seed=0, log=print):
    """loss = (1-lam) * CE(student, gold) + lam * T^2 * KL(teacher_T || student_T), padding ignored."""
    torch.manual_seed(seed); random.seed(seed)
    student.to(device); teacher.to(device).eval()
    params = [p for p in student.parameters() if p.requires_grad]
    head_ids = {id(p) for p in student.head.parameters()}
    opt = torch.optim.AdamW([{"params": [p for p in params if id(p) not in head_ids], "lr": lr},
                             {"params": [p for p in params if id(p) in head_ids], "lr": head_lr}],
                            weight_decay=0.01)
    steps = epochs * math.ceil(len(train_ex) / bs); warm = max(1, int(0.06 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda")
    best_f1, best_state, hist = -1.0, None, []
    log(f"student trainable {sum(p.numel() for p in params)/1e6:.2f}M | {len(train_ex)} windows")
    for ep in range(epochs):
        student.train(); random.shuffle(train_ex); t0 = time.time(); tot = 0.0
        for b in range(0, len(train_ex), bs):
            ids, tags = pm._batch(train_ex[b:b + bs], device)
            mask = tags != -100
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                t_logits = teacher(ids).float()
            with torch.autocast("cuda", dtype=torch.float16):
                s_logits = student(ids).float()
            ce = F.cross_entropy(s_logits[mask], tags[mask])
            kl = F.kl_div(F.log_softmax(s_logits[mask] / T, -1), F.log_softmax(t_logits[mask] / T, -1),
                          log_target=True, reduction="batchmean") * T * T
            loss = (1 - lam) * ce + lam * kl
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
        r = ner.strict_f1(pm.predict_docs(student, dev_docs, enc, device=device),
                          {d["id"]: d["ents"] for d in dev_docs})
        hist.append({"epoch": ep + 1, "loss": tot / math.ceil(len(train_ex) / bs), **r})
        log(f"epoch {ep+1}: loss {hist[-1]['loss']:.4f} | dev strict F1 {r['f1']:.4f} "
            f"(P {r['precision']:.3f} R {r['recall']:.3f}) | {time.time()-t0:.0f}s")
        if r["f1"] > best_f1:
            best_f1 = r["f1"]; best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    student.load_state_dict(best_state)
    return hist, best_f1
