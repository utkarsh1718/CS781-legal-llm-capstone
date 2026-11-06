"""
p3_rr.py -- Phase 3 (Rhetorical Role prediction) on top of ANY Phase-1 backbone.

Idea: a document is a sequence of sentences. We feed WINDOWS of consecutive sentences to the
backbone as one token stream  [SEP] s_1 [SEP] s_2 ... , pool the hidden states of each sentence
(its [SEP] marker + its tokens, mean) and classify the pooled vector into one of 13 roles.
  ctx="none" : every window holds ONE sentence           -> sentence-only model
  ctx="doc"  : windows of ~W tokens of neighbouring sentences, 50% overlap -> document context
               (a sentence's prediction = mean log-prob over the windows that contain it)
Optional extras:
  * relative-position embedding (10 buckets of i / n_sentences), zero-initialised
  * per-domain label mask (never predict a role that does not occur in that domain's training data)
  * label-transition Viterbi (roles come in runs: P(same as previous) = 0.94 CL / 0.75 IT)

PEFT methods: everything in p2_peft.apply_method (lora, lorma_plus, lorma_pi, prompt, full, ...)
plus two new ones for Phase 3:
  "rslora" : LoRA with rank-stabilised scaling alpha / sqrt(r) (Kalajdzievski, 2023)
  "ia3"    : IA^3 (Liu et al., 2022): learned vectors rescale keys, values and the FFN
             intermediate activation (input of the down-projection); all initialised to 1
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

import p2_model as pm
import p2_peft as pp
from common import normalise
from lm_common import SwiGLU

LABELS = ['Fact', 'Issue', 'ArgumentPetitioner', 'ArgumentRespondent', 'Statute', 'Dissent',
          'PrecedentReliedUpon', 'PrecedentNotReliedUpon', 'PrecedentOverruled', 'RulingByLowerCourt',
          'RatioOfTheDecision', 'RulingByPresentCourt', 'None']
L2I = {l: i for i, l in enumerate(LABELS)}
NL = len(LABELS)
SEP_ID, PAD_ID = 3, 0            # <|sep|> marks the start of every sentence; padding is masked out
MAX_SENT = 128                   # truncate very long sentences (p95 is ~90-107 tokens)
N_POS = 10                       # relative-position buckets


def _autocast(device):
    dev = str(device)
    return (torch.autocast("cuda", dtype=torch.float16) if dev.startswith("cuda")
            else contextlib.nullcontext())


# ----------------------------------------------------------------------------- data
def encode_docs(docs, tk, domain):
    """{key: {sents: [token ids], labs: [label ids or -100], dom, orig_id}}; key = 'DOM::id'."""
    out = {}
    for d in docs:
        sents = [tk.encode(normalise(s), add_special_tokens=False).ids[:MAX_SENT] for s in d["text"]]
        labs = [L2I[l] for l in d["labels"]] if "labels" in d else [-100] * len(sents)
        out[f"{domain}::{d['id']}"] = {"sents": sents, "labs": labs, "dom": domain, "orig_id": d["id"]}
    return out


def sent_windows(sents, W=512, ctx="doc"):
    """Sentence ranges [a, b). ctx='none': one sentence per window. ctx='doc': greedy windows of at
    most W tokens; the next window starts at the sentence where the previous one reached half its
    tokens (~50% overlap), so every sentence is seen with left AND right context."""
    n = len(sents)
    if ctx == "none":
        return [(i, i + 1) for i in range(n)]
    L = [len(s) + 1 for s in sents]                                  # +1 for the [SEP] marker
    wins, a = [], 0
    while a < n:
        b, tot = a, 0
        while b < n and (tot + L[b] <= W or b == a):
            tot += L[b]; b += 1
        wins.append((a, b))
        if b >= n:
            break
        a2, t = a, 0
        while a2 < b and t + L[a2] <= tot / 2:
            t += L[a2]; a2 += 1
        while a2 < b and sum(L[a2:b]) + L[b] > W:                     # next window must add new sentences
            a2 += 1
        a = max(a2, a + 1)
    return wins


def make_examples(enc, keys, ctx="doc", W=512):
    ex = []
    for k in keys:
        e = enc[k]; n = len(e["sents"])
        for a, b in sent_windows(e["sents"], W, ctx):
            ids, seg = [], []
            for j, i in enumerate(range(a, b)):
                s = [SEP_ID] + e["sents"][i]
                ids += s; seg += [j] * len(s)
            ex.append({"ids": ids, "seg": seg, "labs": e["labs"][a:b], "key": k, "sidx": list(range(a, b)),
                       "pos": [min(N_POS - 1, int(N_POS * i / max(1, n))) for i in range(a, b)]})
    return ex


def collate(batch, device):
    B, L = len(batch), max(len(x["ids"]) for x in batch)
    ids = torch.full((B, L), PAD_ID, dtype=torch.long)
    seg = torch.full((B, L), -1, dtype=torch.long)
    labs, pos, off, o = [], [], [], 0
    for b, x in enumerate(batch):
        n = len(x["ids"])
        ids[b, :n] = torch.tensor(x["ids"]); seg[b, :n] = torch.tensor(x["seg"])
        labs += x["labs"]; pos += x["pos"]; off.append(o); o += len(x["labs"])
    return (ids.to(device), seg.to(device), torch.tensor(off, device=device),
            torch.tensor(labs, device=device), torch.tensor(pos, device=device))


# ----------------------------------------------------------------------------- model
class RRModel(nn.Module):
    """Backbone (TokenTagger from Phase 2) -> mean-pool per sentence (+ position) -> role logits."""
    def __init__(self, tagger, use_pos=True, dropout=0.1):
        super().__init__()
        self.tagger, self.use_pos = tagger, use_pos
        d = tagger.backbone.cfg.d_model
        self.drop = nn.Dropout(dropout)
        self.pos = nn.Embedding(N_POS, d)
        nn.init.zeros_(self.pos.weight)                               # starts as "no position info"

    def forward(self, ids, seg, off, pos):
        tg = self.tagger
        ctx = pm.full_attention() if (tg.bidirectional and tg.arch != "encdec") else contextlib.nullcontext()
        with ctx:
            h = tg.hidden(ids)
        head = tg.head
        if isinstance(head, pp.DropPrefix):                           # prompt tuning: drop soft prompt
            h, head = h[:, head.n:], head.head
        h = h.float()
        valid = seg >= 0
        gid = (seg + off[:, None])[valid]                             # global sentence index per token
        n = pos.numel()
        pooled = torch.zeros(n, h.size(-1), device=h.device).index_add_(0, gid, h[valid])
        cnt = torch.zeros(n, device=h.device).index_add_(0, gid, torch.ones_like(gid, dtype=torch.float))
        pooled = pooled / cnt.clamp(min=1)[:, None]
        if self.use_pos:
            pooled = pooled + self.pos(pos)
        return head(self.drop(pooled))                                # (n_sentences, NL)


class RSLoRALinear(pp.LoRALinear):
    """LoRA with rank-stabilised scaling alpha / sqrt(r) instead of alpha / r."""
    def __init__(self, base, r=16, alpha=16):
        super().__init__(base, r, alpha)
        self.s = alpha / math.sqrt(r)


class IA3Out(nn.Module):
    """y = (W x) * l  (rescales keys / values)."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.l = nn.Parameter(torch.ones(base.out_features))

    def forward(self, x):
        return self.base(x) * self.l


class IA3In(nn.Module):
    """y = W (x * l)  (rescales the FFN intermediate activation before the down-projection)."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.l = nn.Parameter(torch.ones(base.in_features))

    def forward(self, x):
        return self.base(x * self.l)


def apply_method_p3(tagger, method, r=8, alpha=8, **kw):
    if method == "rslora":
        for p in tagger.backbone.parameters():
            p.requires_grad = False
        rr = kw.pop("rs_r", 16)
        pp._wrap_linears(tagger, lambda l: RSLoRALinear(l, rr, rr))
    elif method == "ia3":
        for p in tagger.backbone.parameters():
            p.requires_grad = False
        for layer in pp._layers(tagger):
            for mod in list(layer.modules()):
                for name in ("wk", "wv", "w_uk", "w_uv"):            # keys / values (dense + MLA)
                    lin = getattr(mod, name, None)
                    if isinstance(lin, nn.Linear):
                        setattr(mod, name, IA3Out(lin))
                if isinstance(mod, SwiGLU) and isinstance(mod.w2, nn.Linear):
                    mod.w2 = IA3In(mod.w2)                            # FFN intermediate activation
    else:
        pp.apply_method(tagger, method, r=r, alpha=alpha, **kw)
    for p in tagger.head.parameters():
        p.requires_grad = True
    return tagger


def build_rr(ckpt, method="lora", bidirectional=True, use_pos=True, **kw):
    tg = pm.build_tagger(ckpt, bidirectional=bidirectional)
    tg.head = nn.Linear(tg.backbone.cfg.d_model, NL)
    nn.init.normal_(tg.head.weight, 0.0, 0.02); nn.init.zeros_(tg.head.bias)
    apply_method_p3(tg, method, **kw)
    return RRModel(tg, use_pos=use_pos)


def save_trainable(model, path):
    """Only the parameters that were trained (PEFT runs stay a few MB)."""
    torch.save({n: p.detach().half().cpu() for n, p in model.named_parameters() if p.requires_grad}, path)


def load_trainable(model, path):
    sd = {k: v.float() for k, v in torch.load(path, map_location="cpu").items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    return model


# ----------------------------------------------------------------------------- domain priors
def domain_stats(enc, keys):
    """Per domain: allowed-label mask (seen in training) and log transition matrix (add-1)."""
    allowed, trans = {}, {}
    for dom in sorted({enc[k]["dom"] for k in keys}):
        C = np.ones((NL, NL)); seen = np.zeros(NL, dtype=bool)
        for k in keys:
            if enc[k]["dom"] != dom:
                continue
            labs = enc[k]["labs"]
            for l in labs:
                seen[l] = True
            for a, b in zip(labs[:-1], labs[1:]):
                C[a, b] += 1
        allowed[dom], trans[dom] = seen, np.log(C / C.sum(1, keepdims=True))
    return allowed, trans


def viterbi(score, trans):
    n, K = score.shape
    dp, back = score[0].copy(), np.zeros((n, K), dtype=np.int64)
    for t in range(1, n):
        cand = dp[:, None] + trans
        back[t] = cand.argmax(0); dp = cand.max(0) + score[t]
    path = [int(dp.argmax())]
    for t in range(n - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


def decode(lp, allowed=None, trans=None, lam=0.0):
    s = lp + (np.where(allowed, 0.0, -1e9) if allowed is not None else 0.0)
    if lam and trans is not None and len(s) > 1:
        return viterbi(s, lam * trans)
    return s.argmax(1).tolist()


# ----------------------------------------------------------------------------- inference + metrics
@torch.no_grad()
def predict(model, enc, keys, ctx="doc", W=512, device="cuda", bs=16):
    """{key: (n_sentences, NL) mean log-probs over all windows containing each sentence}."""
    model.eval()
    ex = sorted(make_examples(enc, keys, ctx, W), key=lambda x: len(x["ids"]))
    acc = {k: np.zeros((len(enc[k]["sents"]), NL)) for k in keys}
    cnt = {k: np.zeros(len(enc[k]["sents"])) for k in keys}
    for b in range(0, len(ex), bs):
        chunk = ex[b:b + bs]
        ids, seg, off, _, pos = collate(chunk, device)
        with _autocast(device):
            lp = model(ids, seg, off, pos).float().log_softmax(-1).cpu().numpy()
        j = 0
        for x in chunk:
            for i in x["sidx"]:
                acc[x["key"]][i] += lp[j]; cnt[x["key"]][i] += 1; j += 1
    return {k: acc[k] / cnt[k][:, None] for k in keys}


def scores(gold, pred):
    from sklearn.metrics import f1_score, precision_score, recall_score
    g, p = np.asarray(gold), np.asarray(pred)
    return {"macro_f1": f1_score(g, p, average="macro", zero_division=0),
            "macro_p": precision_score(g, p, average="macro", zero_division=0),
            "macro_r": recall_score(g, p, average="macro", zero_division=0),
            "acc": float((g == p).mean())}


def evaluate(LP, enc, keys, allowed=None, trans=None, lam=0.0):
    """Macro-F1 (sklearn: labels = gold U pred) for all dev sentences and per domain."""
    G, P = {"all": [], "CL": [], "IT": []}, {"all": [], "CL": [], "IT": []}
    for k in keys:
        dom = enc[k]["dom"]
        pr = decode(LP[k], None if allowed is None else allowed[dom],
                    None if trans is None else trans[dom], lam)
        for name in ("all", dom):
            G[name] += enc[k]["labs"]; P[name] += pr
    return {n: scores(G[n], P[n]) for n in G if G[n]}


# ----------------------------------------------------------------------------- training
def train_rr(model, train_ex, eval_fn, epochs=8, lr=5e-4, head_lr=1e-3, bs=8, class_weight=None,
             device="cuda", seed=0, warmup=0.06, log=print):
    """AdamW, warm-up + linear decay, fp16. Length-bucketed batches (little padding). After every
    epoch eval_fn(model) -> metrics dict; keeps the epoch with the best dev macro-F1 ('all')."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    fast = {id(p) for p in model.tagger.head.parameters()} | {id(p) for p in model.pos.parameters()}
    opt = torch.optim.AdamW([{"params": [p for p in params if id(p) not in fast], "lr": lr},
                             {"params": [p for p in params if id(p) in fast], "lr": head_lr}],
                            weight_decay=0.01)
    steps = epochs * math.ceil(len(train_ex) / bs); warm = max(1, int(warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    w = None if class_weight is None else class_weight.to(device)
    best, best_state, hist = -1.0, None, []
    log(f"trainable {sum(p.numel() for p in params)/1e6:.3f}M | {len(train_ex)} windows | {steps} steps")
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0.0
        order = sorted(range(len(train_ex)), key=lambda i: len(train_ex[i]["ids"]) + random.random() * 16)
        batches = [order[i:i + bs] for i in range(0, len(order), bs)]
        random.shuffle(batches)
        for bidx in batches:
            ids, seg, off, labs, pos = collate([train_ex[i] for i in bidx], device)
            with _autocast(device):
                logits = model(ids, seg, off, pos)
            loss = F.cross_entropy(logits.float(), labs, weight=w, ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
        r = eval_fn(model)
        hist.append({"epoch": ep + 1, "loss": tot / len(batches), **{k: v["macro_f1"] for k, v in r.items()}})
        log(f"epoch {ep+1}: loss {hist[-1]['loss']:.4f} | dev macro-F1 all {r['all']['macro_f1']:.4f} "
            f"(CL {r['CL']['macro_f1']:.4f} IT {r['IT']['macro_f1']:.4f}) acc {r['all']['acc']:.3f} | {time.time()-t0:.0f}s")
        if r["all"]["macro_f1"] > best:
            best = r["all"]["macro_f1"]
            best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=False)
    return hist, best


# ----------------------------------------------------------------------------- v2: training with optional KD
def train_rr(model, train_ex, eval_fn, epochs=8, lr=5e-4, head_lr=1e-3, bs=8, class_weight=None,
             device="cuda", seed=0, warmup=0.06, log=print, kd_T=2.0, kd_lam=0.5):
    """As v1, plus knowledge distillation: if the examples carry 'soft' (teacher log-probs per
    sentence), loss = (1 - kd_lam) * CE(gold) + kd_lam * T^2 * KL(teacher_T || student_T)."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    fast = {id(p) for p in model.tagger.head.parameters()} | {id(p) for p in model.pos.parameters()}
    opt = torch.optim.AdamW([{"params": [p for p in params if id(p) not in fast], "lr": lr},
                             {"params": [p for p in params if id(p) in fast], "lr": head_lr}],
                            weight_decay=0.01)
    steps = epochs * math.ceil(len(train_ex) / bs); warm = max(1, int(warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    w = None if class_weight is None else class_weight.to(device)
    kd = "soft" in train_ex[0]
    best, best_state, hist = -1.0, None, []
    log(f"trainable {sum(p.numel() for p in params)/1e6:.3f}M | {len(train_ex)} windows | {steps} steps"
        + (f" | KD T={kd_T} lam={kd_lam}" if kd else ""))
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0.0
        order = sorted(range(len(train_ex)), key=lambda i: len(train_ex[i]["ids"]) + random.random() * 16)
        batches = [order[i:i + bs] for i in range(0, len(order), bs)]
        random.shuffle(batches)
        for bidx in batches:
            ids, seg, off, labs, pos = collate([train_ex[i] for i in bidx], device)
            with _autocast(device):
                logits = model(ids, seg, off, pos)
            loss = F.cross_entropy(logits.float(), labs, weight=w, ignore_index=-100)
            if kd:
                t = torch.tensor(np.concatenate([np.asarray(train_ex[i]["soft"]) for i in bidx]),
                                 dtype=torch.float, device=device)
                kl = F.kl_div(F.log_softmax(logits.float() / kd_T, -1), F.log_softmax(t / kd_T, -1),
                              log_target=True, reduction="batchmean") * kd_T * kd_T
                loss = (1 - kd_lam) * loss + kd_lam * kl
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
        r = eval_fn(model)
        hist.append({"epoch": ep + 1, "loss": tot / len(batches), **{k: v["macro_f1"] for k, v in r.items()}})
        log(f"epoch {ep+1}: loss {hist[-1]['loss']:.4f} | dev macro-F1 all {r['all']['macro_f1']:.4f} "
            f"(CL {r['CL']['macro_f1']:.4f} IT {r['IT']['macro_f1']:.4f}) acc {r['all']['acc']:.3f} | {time.time()-t0:.0f}s")
        if r["all"]["macro_f1"] > best:
            best = r["all"]["macro_f1"]
            best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=False)
    return hist, best
