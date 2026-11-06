# infer.py -- checkpoint -> 128-word continuations (optionally n candidates + MBR) -> submission
from __future__ import annotations
import json, os, sys, zipfile
from collections import Counter
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                   # noqa: E402
from generation import generate                 # noqa: E402

N_WORDS = 128


def load_model(ckpt_path, device="cuda"):
    # best.pt (fp16 weights + arch/model_cfg) -> fp32 module; inference runs under fp16 autocast
    from train import build_model
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ck["arch"], ck["model_cfg"])
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    return model.to(device).eval(), ck


def load_tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def space_start_mask(tk, vocab_size, device):
    # bool (V,): token text starts with a space -> allowed as the first generated token
    m = torch.zeros(vocab_size, dtype=torch.bool)
    for i in range(16, vocab_size):
        m[i] = tk.decode([i]).startswith(" ")
    return m.to(device)


def _ensure_128(model, tk, prompt_ids, text, max_prompt, max_new_tokens, device, gen_kw, rounds=3):
    # extend a short continuation by generating more; last resort: pad from the prompt's tail
    for _ in range(rounds):
        if len(text.split()) >= N_WORDS:
            return text
        ctx = (prompt_ids + tk.encode(" " + text, add_special_tokens=False).ids)[-max_prompt:]
        with torch.autocast("cuda", dtype=torch.float16):
            y = generate(model, torch.tensor([ctx], device=device), max_new_tokens, **gen_kw)
        text = common.normalise(text + " " + tk.decode(y[0].tolist()))
    w = text.split()
    if len(w) < N_WORDS:
        w = w + tk.decode(prompt_ids).split()[-(N_WORDS - len(w)):]
    return " ".join(w)


@torch.no_grad()
def continue_prompts(model, tk, prompts, device="cuda", batch_size=32, max_ctx=1024,
                     max_new_tokens=200, first_mask=None, num_return=1, **gen_kw):
    # {id: text} -> {id: text} (num_return=1) or {id: [n candidates]}; every text = 128 words
    ids = {k: tk.encode(v, add_special_tokens=False).ids for k, v in prompts.items()}
    max_prompt = min(max_ctx - max_new_tokens, getattr(model, 'max_prompt_tokens', 10**9))
    ids = {k: v[-max_prompt:] for k, v in ids.items()}
    order = sorted(ids, key=lambda k: len(ids[k]))              # similar lengths share a batch
    bs = max(1, batch_size // num_return)
    out, n_trunc = {}, 0
    for b in range(0, len(order), bs):
        keys = order[b:b + bs]
        L = min(len(ids[k]) for k in keys)                       # cut batch to shortest (left)
        n_trunc += sum(len(ids[k]) > L for k in keys)
        x = torch.tensor([ids[k][-L:] for k in keys], device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            y = generate(model, x, max_new_tokens, first_allowed=first_mask, num_return=num_return, **gen_kw)
        rows = [common.normalise(tk.decode(r)) for r in y.tolist()]
        for j, k in enumerate(keys):
            out[k] = rows[j * num_return:(j + 1) * num_return]
    n_short = 0
    for k in out:
        for j, t in enumerate(out[k]):
            if len(t.split()) < N_WORDS:
                n_short += 1
                out[k][j] = _ensure_128(model, tk, ids[k], t, max_prompt, max_new_tokens, device, gen_kw)
    final = {k: [" ".join(t.split()[:N_WORDS]) for t in v] for k, v in out.items()}
    stats = {"n": len(final), "left_truncated_in_batch": n_trunc, "short_rows_extended": n_short}
    return ({k: v[0] for k, v in final.items()} if num_return == 1 else final), stats


def ngram_f1(a, b, orders=(1, 2)):
    # symmetric unigram+bigram F1: cheap, model-free MBR utility
    wa, wb, f = a.split(), b.split(), 0.0
    for n in orders:
        A, B = Counter(zip(*[wa[i:] for i in range(n)])), Counter(zip(*[wb[i:] for i in range(n)]))
        ov = sum((A & B).values())
        p, r = ov / max(1, sum(A.values())), ov / max(1, sum(B.values()))
        f += 2 * p * r / max(1e-9, p + r)
    return f / len(orders)


def mbr_select(cands):
    # Minimum Bayes Risk: candidate with the highest mean utility against all the others
    if len(cands) == 1:
        return cands[0], 0
    sc = [sum(ngram_f1(c, o) for j, o in enumerate(cands) if j != i) / (len(cands) - 1)
          for i, c in enumerate(cands)]
    best = max(range(len(cands)), key=sc.__getitem__)
    return cands[best], best


def write_submission(preds, test_ids, out_dir, name):
    # strict checks, then prediction.json at the ROOT of <name>.zip
    test_ids = set(map(str, test_ids))
    assert set(preds) == test_ids, f"ID mismatch: missing {len(test_ids - set(preds))}, extra {len(set(preds) - test_ids)}"
    bad = [k for k, v in preds.items() if len(v.split()) != N_WORDS]
    assert not bad, f"{len(bad)} predictions are not exactly {N_WORDS} words"
    os.makedirs(out_dir, exist_ok=True)
    js = {k: {"text": preds[k]} for k in sorted(preds, key=lambda s: int(s) if s.isdigit() else s)}
    pj = f"{out_dir}/prediction.json"
    json.dump(js, open(pj, "w"), ensure_ascii=True)
    zp = f"{out_dir}/{name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"]
    return zp
