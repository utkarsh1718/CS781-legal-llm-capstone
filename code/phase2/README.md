# Phase 2: Legal Named Entity Recognition (L-NER)

> **Official test result (Codabench leaderboard):** strict macro-F1 **0.6407** (other columns 0.6534 / 0.6424), **rank 10** (submitted 2026-10-07).

**Task.** For each judgment, return every legal entity as `[start_char, end_char, label]` on the **original** text.
12 labels: APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO.
**Metric: strict macro-F1.** An entity counts only if both boundaries and the label match exactly.
**Data:** 70 labelled documents (fold1 + fold2) and 35 test documents (fold3). We held out 10 documents (5 per fold, chosen by hash) as dev.

**Depends on:** Phase 1 models and tokenizer (`code/phase1`).

---

## 1. Pipeline: exact offsets ↔ model tokens (`p2_ner.py`)

Our models were trained on normalised text (ASCII, single spaces, no newlines), but scoring uses original character offsets.
1. `normalise_with_map(text)` returns the Phase 1 normalised text **and** the original index of every normalised character.
2. Tokenise with A1-BPE, then map each token's core (leading space removed) back to an original span.
3. **BIO tags** (25 classes). No spans overlap, so BIO is enough.
4. Decode tags back to original-offset spans; sliding windows of 512 tokens with stride 256.
5. Strict micro- and macro-F1 plus per-label F1, exactly as the official evaluator computes them.

An **oracle check** (gold → BIO → spans) measures the ceiling for any token tagger: entities whose boundary falls inside a token can't be recovered.

## 2. Tagger on every backbone (`p2_model.py`)

Backbone hidden states → dropout → linear layer → 25 tags. M1, M2 and M4 use their decoder stack, M3 its encoder.
**Causal vs bidirectional:** removing the causal mask during fine-tuning (as in LLM2Vec) lets a decoder see the right-hand context ("State of Haryana" vs "State of Haryana's counsel"). Each token takes its prediction from the window where it has the most context.

## 3. Decoding improvements (no retraining)

| Step (M1 full fine-tuning) | Dev strict micro-F1 |
|---|---|
| Causal attention | 0.451 |
| Bidirectional attention | 0.480 |
| + **constrained Viterbi** (only valid BIO sequences; CRF-style inference on the existing probabilities) | ~0.580 |
| + **word-consistent Viterbi** (an entity can't start or end inside a word, e.g. `VENKATACH`) | 0.585 |
| + document header prepended to every window | +0.003–0.009, within noise, 33% more compute → **dropped** |

Error analysis sorted every error into wrong label, boundary, boundary + label, spurious or missed, and reported lenient F1 alongside strict F1. Boundary errors and APP / RESP confusion dominated.

## 4. Adaptation methods (`p2_peft.py`) on all four backbones

All methods are implemented in plain PyTorch (no PEFT library). A sanity check confirms each one reproduces the pretrained hidden states exactly at initialisation.

| Run | Backbone | Method | Trainable | Dev strict micro-F1 |
|---|---|---|---|---|
| baseline | M1 | full fine-tuning | 100.1M | 0.585 |
| 1 | M1 | LoRA (r = 8, q / v) | 0.26M | **0.619** |
| 2 | M1 | **LoRMA+** (multiplicative, identity inflation) | 0.22M | 0.582 |
| 3 | M1 | **LoRMAπ** (permutation inflation) | 0.22M | 0.560 |
| 4 | M3 encoder | full fine-tuning | 62.4M | 0.359 |
| 5 | M2 | LoRMA+ | 0.31M | 0.578 |
| 6 | M4 | bottleneck adapters (64) | 1.2M | 0.617 |
| 7 | **KD student** (6 of 12 M1 layers) | distilled from the M1 + LoRA teacher | 62.4M | 0.615 |
| control | same student | gold labels only | 62.4M | 0.612 |

Runs 1–3 use identical settings, which gives a clean **additive (LoRA) vs multiplicative (LoRMA+ / LoRMAπ)** comparison: on 55 training documents, additive LoRA adapted best. M3's encoder lagged far behind the decoders turned bidirectional.

**Knowledge distillation:** loss = 0.5·CE + 0.5·T²·KL(teacher_T ‖ student_T) with T = 2, the student initialised from every other M1 layer (as in DistilBERT). The student keeps the teacher's quality at **62% of its size, with inference 1.3× faster** (4.4 s vs 5.6 s on dev). KD's gain over the gold-only control is small (+0.003) because the dataset is small.

## 5. Optimising for the official metric (macro-F1)

Once we switched to selecting models by macro-F1:
- **Class-weighted loss** (inverse √frequency) improved LoRA's macro-F1 by +0.05.
- **Technique ranking by macro-F1** (M1, our data only): full 0.505 > LoRA 0.495 > prompt tuning 0.439 > norm-gain (BitFit analogue) 0.405. The more trainable parameters, the better; the two smallest methods over-predict.

## 6. External data: InLegalNER (allowed by the TAs; MIT licence)

About 9.4k judgment sentences and about 1.5k preambles from the same domain.
- **Label mapping:** PETITIONER → APP, RESPONDENT → RESP, and so on. LAWYER / ORG / PROVISION spans are ignored (loss −100), because they don't map cleanly onto our labels.
- **Leakage control:** any record whose word 8-grams overlap the **test fold** is dropped, and records overlapping 30% or more with our dev documents are dropped too.
- **Two-stage training:** InLegalNER (Stage A), then our 55 documents (Stage B), each compared against a same-code control.

| Recipe | Without external data | With InLegalNER |
|---|---|---|
| LoRA (3 seeds) | 0.495 | +0.038 on average, positive on every seed |
| **Full fine-tuning + bidirectional** | 0.505 | **0.601** (micro 0.608 → 0.679) |

The gain from external data **grows with the number of trainable parameters**.

## 7. Final system

- Full fine-tuning + bidirectional + InLegalNER over 3 seeds: **0.603 ± 0.021** macro-F1.
- Seed ensemble (mean token log-probs, then word-Viterbi): 0.620. Adding the LoRA-ext model: **0.635** (micro 0.707); JUDGE gains the most (+0.10).
- **Document-level consistency rules did not help** (−0.02 micro). The gold annotations don't label every repeated party mention, so the APP / RESP errors come from *which mentions* get selected, not from inconsistent labels.
- Final test models were **retrained on all 70 documents** (3 seeds, fixed 4 epochs) and ensembled with the LoRA-ext model.

## Files

| File | Purpose |
|---|---|
| `p2_ner.py` | Offset mapping, BIO encoding / decoding, windows, constrained and word-consistent Viterbi, strict metrics |
| `p2_model.py` | Token tagger over M1–M4, causal / bidirectional, windowed training and prediction |
| `p2_peft.py` | Full, partial, norm-gain, LoRA, LoRMA+ / LoRMAπ, adapters, prompt tuning, KD (reused by Phases 3–7) |
| `phase2_notebook.ipynb` | All Phase 2 cells with outputs (InLegalNER conversion, ensembles, submissions) |

Results: `results/stats/p2_results.json`, `p2_ext_results.json`, `p2_ensemble.json`, `p2_ablation_downstream.json`.
