# Phase 4: Court Judgment Prediction (CJPE)

> **Official test result (Codabench leaderboard):** macro-F1 **0.8094** (other columns 0.799 / 0.8294), **rank 14** (submitted 2026-10-17).

**Task.** Predict the binary outcome (0/1) of a case from its judgment text, with the decision itself removed.
**Metric:** macro-F1 over both classes. **Output:** `[{"id", "pred"}]`.

**Depends on:** `code/phase1`, `code/phase2/p2_model.py` + `p2_peft.py`, `code/phase3/p3_rr.py` (IA³, rsLoRA).
**Reused by:** Phase 5 (`p5_lsi.py`) and Phase 7 (`p7_pcr.py`), through `DocClassifier`, `build()` and `encode_chunks()`.

---

## 1. Data findings and integrity

- 37,387 training cases (about 141M tokens; median about 2.7k tokens, so **only 8% fit in 512 tokens**).
- **Prior shift:** train is 59/41, dev and test are 50/50. So the loss uses balanced class weights and the threshold is tuned on dev.
- 5,710 duplicate texts in train; duplicate groups with conflicting labels were dropped, leaving **31,677 unique cases**.
- **Leak found and reported to the TAs:** all 1,517 test cases also appear in dev (identical texts), and `test.json` carries labels. We **never used test labels** and **removed those cases from dev**. Selection and calibration use the remaining **1,988 clean dev cases** only.
- Every case is tokenised **once** into a uint16 cache, so every input view (first / middle / last k tokens, all chunks) is cheap to build later.

## 2. Model (`p4_cjpe.py`)

Phase 1 backbone (bidirectional) → masked mean pooling → `Linear(d, 2)`, with class-weighted CE. Macro-F1 is reported at threshold 0.5 and at a threshold tuned on clean dev.

## 3. Which part of a judgment predicts the outcome? (M1 + LoRA, 1 epoch, 512 tokens)

| View | Dev macro-F1 (tuned threshold) |
|---|---|
| **Last 512 tokens** | **0.696** |
| Head + tail (128 + 384) | 0.672 |
| Middle 512 | 0.586 |
| First 512 | 0.579 |

The end of the judgment (the court's reasoning just before the removed decision) carries the outcome; the opening facts carry little.

## 4. Techniques × architectures (last-512 view, 1 epoch, budget-matched; `p4_run.py`)

| Run | Technique | Dev macro-F1 |
|---|---|---|
| m1_full_last512 | **full fine-tuning** | **0.738** |
| m1_lora_last1024 | LoRA on the last **1,024** tokens | 0.731 |
| m1_unfreeze_last512 | gradual layer unfreezing | 0.730 |
| m1_partial4_last512 | partial (top 4 layers) | 0.728 |
| m4_adapter_last512 | adapters (M4) | 0.702 |
| m1_adapter_last512 | adapters | 0.698 |
| m1_lorma_plus_last512 | **LoRMA+** | 0.698 |
| m1_lora_last512 | LoRA | 0.696 |
| m2_lorma_plus_last512 | LoRMA+ (M2) | 0.685 |
| m1_normgain_last512 | norm-gain (BitFit analogue; our layers have no biases) | 0.634 |
| m3_full_last512 | full fine-tuning (M3 encoder) | 0.623 |

With 31.7k training cases, **trainable capacity matters** (the opposite of Phase 3's 80 documents). Doubling the closing context from 512 to 1,024 tokens adds +0.035.

## 5. Final and hierarchical models

| System | Dev macro-F1 |
|---|---|
| M1 full fine-tuning, last 1,024 tokens, 2 epochs | 0.776 |
| **Hierarchical:** frozen fine-tuned encoder over every 512-token chunk (up to the last 64) → BiGRU + attention pooling | 0.791 |
| Hierarchical over the last-1,024 model (1,024-token chunks) | **0.806** |
| Greedy ensemble (2 hierarchical + partial) | 0.819 in-sample / **0.806 honest** |
| Greedy ensemble with extra pooler seeds | 0.823 in-sample / **0.815 honest** |

**Where the evidence is:** in the hierarchical model, 73% of the attention mass falls on the **last 40%** of the judgment, peaking at decile 8 (0.30), just before the removed decision. The opening deciles get little attention, apart from the case header.

## 6. Knowledge distillation

**Teacher:** M1 full fine-tuning, last 512 (100M). **Student:** 6 of the 12 layers (62M), 0.5·CE + 0.5·T²·KL, T = 2, 1 epoch.
**Result:** student 0.723 vs **0.733 without KD** (control). **KD did not help here.** With 31.7k labelled cases, the gold labels are already plentiful, and the teacher's soft targets on its own training cases are over-confident. The student keeps about 98% of the teacher's score at 62% of the size.

## 7. Analysis in the notebook (P4-7)

False positives vs false negatives, macro-F1 by case length, calibration, and the attention-by-decile profile.

## Files

| File | Purpose |
|---|---|
| `p4_cjpe.py` | Token-cache split, section views, `DocClassifier`, `build()`, `encode_chunks()`, hierarchical `ChunkPooler`, metrics and thresholds |
| `p4_run.py` | Background runner (techniques, hierarchical models, KD) |
| `phase4_notebook.ipynb` | Data cleaning, section study, ensembles, error analysis, submissions |

Results: `results/stats/p4_runs/`, `p4_submissions.json`, `p4_seed_ensembles.json`. The Phase 4 seed upgrades are in the `PX` section of `code/phase5/phase5_notebook.ipynb`.
