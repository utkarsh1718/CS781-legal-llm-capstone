# Phase 3: Rhetorical Role Prediction (RR)

**Task.** Assign one of 13 rhetorical roles (Fact, Issue, ArgumentPetitioner, …, RulingByPresentCourt, None) to **every sentence** of a judgment, in two domains: **Corporate Law (CL)** and **IT**.
**Metric:** macro-F1. **Output:** `{doc_id: [role per sentence]}`.

**Depends on:** `code/phase1` (models), `code/phase2/p2_model.py` and `p2_peft.py` (backbone wrapper, adaptation methods).

---

## 1. Model (`p3_rr.py`)

- **Input:** windows of consecutive sentences `[SEP] s1 [SEP] s2 …`, up to 512 tokens with about 50% overlap. `ctx="none"` gives the sentence-only baseline.
- **Sentence vector:** the mean of its hidden states ([SEP] marker plus tokens), plus a zero-initialised **relative-position embedding** (10 buckets).
- **Prediction:** a linear head over 13 roles. A sentence's log-probs are averaged over every window that contains it.
- **Domain priors:** a per-domain **label mask** (roles never seen in that domain are never predicted) and an optional **transition Viterbi**, λ·log P(role_t | role_t−1).
- **Loss:** class-weighted CE (inverse √frequency, capped at 10). Joint CL + IT training; epochs selected by dev macro-F1.

## 2. Does context help? (M1 + LoRA, dev macro-F1)

| Input | Dev macro-F1 |
|---|---|
| Sentence only | 0.377 (+0.07 with transition Viterbi) |
| + document context (sentence windows) | 0.532 (+0.155; CL +0.17) |
| + relative position | **0.544** |

Transition Viterbi adds nothing on top of document context: the windowed model has already learned the role transitions.

## 3. Techniques × architectures (`p3_run.py`, background runner)

Fixed recipe: document-context windows, position embedding, class weights, label masks, 6 epochs. The learning rate is tuned per technique (allowed by the spec).

| Run | Model | Technique | Dev macro | CL | IT |
|---|---|---|---|---|---|
| m1_rslora | M1 | **rsLoRA** (r = 16, α/√r) | **0.555** | 0.519 | 0.558 |
| m1_lora_doc_pos | M1 | LoRA (r = 8) | 0.552 | 0.444 | 0.572 |
| m1_lorma_plus | M1 | **LoRMA+** | 0.550 | 0.493 | 0.557 |
| m4_lora | M4 | LoRA | 0.550 | 0.484 | 0.560 |
| m1_full | M1 | full fine-tuning | 0.547 | 0.462 | 0.578 |
| kd_student_m1x6 | 6-layer M1 | **KD** | 0.546 | 0.455 | 0.589 |
| m2_lorma_plus | M2 | LoRMA+ | 0.530 | 0.410 | 0.536 |
| m1_ia3 | M1 | IA³ | 0.500 | 0.436 | 0.547 |
| nokd_student_m1x6 | 6-layer M1 | no KD (control) | 0.492 | 0.422 | 0.505 |
| m1_prompt | M1 | prompt tuning (20 vectors) | 0.445 | 0.385 | 0.460 |
| m3_lora | M3 encoder | LoRA | 0.380 | 0.364 | 0.437 |

With only about 80 training documents, low-rank methods match or beat full fine-tuning (the opposite of Phase 4, which has 31.7k cases). IA³ and prompt tuning don't have enough capacity.

## 4. Knowledge distillation

**Teacher:** the mean posterior of the 4 best M1 models. **Student:** M1 with layers 0, 2, 4, 6, 8, 10 (62.4M parameters); loss = 0.5·CE + 0.5·T²·KL with T = 2.
**Result: 0.546 vs 0.492 without KD (+0.054 over the control; +0.087 in the first run).** The student is within 0.01 of the best 100M teacher, with 38% fewer parameters and **about 2× faster test inference (2.2 s vs 4.5 s)**.
*Caveat:* the teachers are over-confident on their own training sentences; T = 2 softens this.

## 5. Ensembles, upgrades and submissions (`p3_extra.py`)

| System | Dev macro-F1 |
|---|---|
| Ensemble of the top 7 (mean log-probs) | 0.583 (test: **0.518**) |
| **Document BiGRU over frozen rsLoRA sentence vectors** (`h_rslora`) | **0.602** (best single system) |
| Ensemble of the top 7 + document BiGRUs | 0.584 |
| + extra seeds | 0.575 |

- **Document BiGRU:** a 2-layer BiGRU reads the whole judgment's sentence sequence, so every sentence sees the entire document, not just its 512-token window (the idea that worked in Phases 4–5).
- **Train + dev retraining (`*_td`):** test came out about 0.06 below dev, a sign of too little data. The top configurations were retrained on train + dev for the epoch count that peaked on dev.

**Error analysis** (P3-5 in the notebook): per-role F1 for CL and IT, and the most frequent confusions between closely related roles.

## Files

| File | Purpose |
|---|---|
| `p3_rr.py` | Sentence-window model, position embedding, label masks, transition Viterbi, rsLoRA / IA³ |
| `p3_run.py` | Background runner (one process per GPU; skip-if-done) |
| `p3_extra.py` | Document BiGRU, train + dev retraining, extra seeds |
| `phase3_notebook.ipynb` | All Phase 3 cells with outputs |

Results: `results/stats/p3_runs/`, `p3_results.json`, `p3_submissions.json`, `p3_upgrade.json`.
