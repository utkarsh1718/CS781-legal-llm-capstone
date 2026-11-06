"""
eval_local.py -- local proxy for the Phase-1 metrics on our validation set:
BERTScore F1 (roberta-large, library default), ROUGE-L F1 (stemmed), BLEU-4 (sacrebleu corpus).
Official scorer settings are unknown -> use for RELATIVE comparisons on identical prompts.
"""
from __future__ import annotations
import numpy as np


def _txt(v):
    return v["text"] if isinstance(v, dict) else v


def score(preds: dict, refs: dict, device="cuda", bert_model="roberta-large", batch_size=32,
          by_group: dict | None = None):
    keys = sorted(preds)
    cands = [_txt(preds[k]) for k in keys]
    gold = [_txt(refs[k]) for k in keys]
    from bert_score import score as bert_score
    _, _, F = bert_score(cands, gold, model_type=bert_model, lang="en", batch_size=batch_size,
                         device=device, verbose=False)
    F = F.numpy()
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl = np.array([rs.score(g, c)["rougeL"].fmeasure for c, g in zip(cands, gold)])
    import sacrebleu
    out = {"n": len(keys), "bertscore_f1": float(F.mean()), "rougeL_f1": float(rl.mean()),
           "bleu4": float(sacrebleu.corpus_bleu(cands, [gold]).score)}
    if by_group:                                          # e.g. per-source BERTScore
        for g in sorted(set(by_group[k] for k in keys)):
            idx = [i for i, k in enumerate(keys) if by_group[k] == g]
            out[f"bertscore_f1[{g}]"] = float(F[idx].mean())
    return out


def repetition_rate(texts, n=3):
    """Mean fraction of repeated word 3-grams per text (degeneration check)."""
    r = []
    for t in texts:
        w = _txt(t).split()
        grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
        r.append(1 - len(set(grams)) / max(1, len(grams)))
    return float(np.mean(r))
