# Phase 5: Legal Statute Identification (LSI)

> **Official test result (Codabench leaderboard):** macro-F1 **0.4399** (micro-F1 0.5309 · subset accuracy 0.0628), **rank 13** (submitted 2026-10-27).

**Task.** Multi-label classification: predict the set of relevant statutes (out of 100 statute IDs) for each case.
**Metric:** macro-F1 over statute IDs (micro-F1 and subset accuracy secondary), so rare statutes count as much as common ones. **Output:** `[{"id", "pred": [statute ids]}]`.

**Depends on:** `code/phase4/p4_cjpe.py` (token cache, section views, `DocClassifier`, chunk encoder, `ChunkPooler`), `code/phase2`–`3` (adaptation methods), `code/phase1`.

---

## 1. Model (`p5_lsi.py`, `p5_run.py`)

- The Phase 4 document classifier with a **100-way sigmoid head**.
- **Loss:** BCE with per-statute `pos_weight = √(neg/pos)`, for class imbalance.
- **Decision rules:** one global threshold; one threshold per statute (cross-fitted on two halves of dev for an honest estimate); and per-statute thresholds **shrunk halfway** towards the global one, against over-fitting. Every case gets at least one statute.
- **KD on tempered sigmoids**, with a same-size control without the teacher.
- Splits were cleaned and checked for overlap (Phase 4's test set had been inside dev). Everything is plain PyTorch.

## 2. Section study (M1 + LoRA, 1 epoch)

| View | Dev macro-F1 |
|---|---|
| First 512 | 0.310 |
| Last 512 | 0.308 |
| Head + tail 512 | 0.310 |
| Middle 512 | 0.306 |
| **First 1,024** | **0.325** |

Unlike Phase 4, **no single section stands out**: statutes are cited throughout the judgment. **Length is what helps.**

## 3. Techniques × architectures (first-512, 1 epoch; honest dev macro-F1, global threshold)

| Run | Technique | Trainable | Dev macro-F1 |
|---|---|---|---|
| m1_full_first512 | full fine-tuning | 100.2M | **0.355** |
| m1_partial4_first512 | partial (top 4 layers) | 25.2M | 0.340 |
| m1_unfreeze_first512 | gradual unfreezing | 75.6M | 0.337 |
| m4_adapter_first512 | adapters (M4) | 1.27M | 0.323 |
| m1_adapter_first512 | adapters | 1.27M | 0.322 |
| m3_full_first512 | full fine-tuning (M3) | 62.4M | 0.316 |
| m1_rslora_first512 | rsLoRA | 0.57M | 0.314 |
| m2_lorma_plus_first512 | LoRMA+ (M2) | 0.37M | 0.313 |
| m1_lora_first512 | LoRA | 0.32M | 0.309 |
| m1_lorma_plus_first512 | **LoRMA+** | 0.27M | 0.309 |
| m1_ia3_first512 | IA³ | 0.11M | 0.303 |
| m1_normgain_first512 | norm-gain | 0.10M | 0.294 |
| m1_prompt_first512 | prompt tuning | 0.09M | 0.279 |

Performance rises with trainable capacity, as in Phase 4 (large training set).

## 4. Final models: longer input and hierarchical models

| System | Dev macro-F1 (honest) |
|---|---|
| M1 full fine-tuning, first 1,024, 2 epochs | 0.396 |
| M1 partial, first 1,024 | 0.379 |
| **Hierarchical** (frozen first-1,024 full FT → BiGRU + attention over every 1,024-token chunk), 3 pooler seeds | 0.408–0.410 |
| Hierarchical over partial fine-tuning, 3 seeds | 0.401–0.403 |
| Ensemble of all hierarchical models, shrunk per-statute thresholds | 0.430 |
| **+ label-correlation stacker** (per-statute logistic regression over all 100 scores; 5-fold out-of-fold) | **0.434** |

Final submissions also include versions retrained on **train + clean dev (+24% data)**, with thresholds taken from their dev-validated twins. A **rate-matched** variant guards against calibration shift: it matches the predicted labels per case using test predictions only, never labels.

## 5. Knowledge distillation

6-layer M1 student (62.4M) on first-1,024: **0.378 with KD vs 0.365 without** (+0.013), against the 100M teacher's 0.396 (95% of the teacher at 62% of the size).

## 6. Analysis (best ensemble, clean dev; `p5_analysis.json`)

- **Rare statutes are the weak ones:** F1 by training-frequency quartile is 0.32 → 0.42 → 0.44 → 0.56 (Spearman ρ = 0.56 between frequency and F1).
- **Labels per case:** 3.91 true vs 4.37 predicted (slight over-prediction).
- **Cases with more statutes are easier:** per-case F1 is 0.36 for cases with 1 true statute and 0.55 for cases with 7 or more.
- **Longer judgments are easier:** per-case F1 is 0.41 for ≤ 512 tokens and 0.62 for > 4,096 tokens. Short cases simply mention fewer statutes.
- **Co-occurrence errors:** for the pair (12, 13), the partner is wrongly added 32–51% of the time when only one is true. That motivated the label-correlation stacker.
- **Attention:** the first-1,024 hierarchical models put about 85% of their attention on the **first** chunk (the statement of facts and charges), the opposite of Phase 4.

## Files

| File | Purpose |
|---|---|
| `p5_lsi.py` | Multi-label head, pos_weight BCE, thresholds (global / per-label / shrunk, cross-fitted), KD |
| `p5_run.py` | Background runner (section study, techniques, hierarchical models, KD) |
| `p5_td.py` | Train + clean-dev retraining of the final models |
| `phase5_notebook.ipynb` | All Phase 5 cells, plus the **PX** section (Phase 4 + 5 final upgrades: train + dev, stacker, Phase 4 pooler seeds) |

Results: `results/stats/p5_runs/`, `p5_systems.json`, `p5_lc.json`, `p5_analysis.json`.
