# Phase 6: Abstractive Legal Summarisation

**Task.** Legal judgment → abstractive summary. **Metric:** BERTScore F1 (ROUGE-1 / ROUGE-L secondary). **Output:** `[{"id", "summary"}]`.
**Rules:** start from the Phase 1 models and use all four; at least 5 adaptation techniques (LoRMA compulsory) plus KD; handle long documents explicitly.

**Depends on:** `code/phase1` (models, KV-cached generation), `code/phase2/p2_peft.py` (LoRA / LoRMA layers). **Reused by:** Phase 7 (NF4 quantisation, IA³ branch).

---

## 1. Strategy: window-level summarisation (`p6_sum.py`)

Judgments are far longer than our 1,024-token context. Instead of truncating:
1. Each document is cut into consecutive **520-word windows** (input ≤ 640 tokens).
2. Each reference-summary sentence is **assigned to the window with the largest word-bigram overlap** (unigram overlap breaks ties). A window's target is its sentences, in summary order (≤ 384 tokens).
3. At test time every window is summarised; outputs are joined in document order and **duplicate sentences are removed**.

Input + target ≤ 1,024 tokens, exactly the pretraining context. M3 was pretrained with exactly this 640/384 split.

**Model-specific training:** decoder-only models (M1, M2, M4) use `[window] <sep> [target] <eos>` with the loss on the target only. M3 uses encoder ← window, decoder ← target. M4 uses the exact copy-head mixture loss, pointing into the window.

**Methods (plain PyTorch):** full, LoRA, LoRMA+ / LoRMAπ, Houlsby adapters (after attention and FFN), Pfeiffer adapters (after FFN only), prompt tuning, IA³, and **QLoRA with our own NF4 4-bit quantisation** (two codes per byte) + LoRA. **KD:** a 6-layer M1 student learns from the full M1 teacher's token distributions (T = 2).

**Generation:** greedy KV-cached decoding, n-gram repeat blocking, early stop at `<eos>`, and an **`eos_bias`** added to the end-of-sequence logit for length control.

## 2. Length calibration (the single biggest gain)

Uncalibrated models stop far too early: their summaries are 0.06–0.66× the reference length. For each run we sweep `eos_bias` on 20 validation documents, keep the setting with the best ROUGE-1 inside a length band of 0.8–1.25×, and regenerate validation and test (`<run>__cal`).

| Run | ROUGE-1 before → after | Length before → after |
|---|---|---|
| m1_full | 0.422 → **0.577** | 0.49× → 1.19× |
| m1_qlora | 0.274 → 0.524 | 0.27× → 1.10× |
| m1_ia3 | 0.058 → 0.415 | 0.06× → 0.92× |

## 3. Results (100 held-out validation documents, calibrated)

| Run | Model | Technique | Trainable | R-1 | R-2 | R-L | Length |
|---|---|---|---|---|---|---|---|
| m1_full__cal | M1 | full fine-tuning (KD teacher) | 100.1M | **0.577** | 0.323 | 0.305 | 1.19 |
| m4_full__cal | M4 | full fine-tuning | 100.3M | 0.575 | **0.332** | **0.318** | 1.22 |
| m2_full__cal | M2 | full fine-tuning | 270.4M | 0.563 | 0.323 | 0.298 | 1.49 |
| kd_m1x6__cal | 6-layer M1 | **KD** | 62.3M | 0.555 | 0.306 | 0.288 | 1.36 |
| m4_lora__cal | M4 | LoRA | 0.98M | 0.552 | 0.299 | 0.285 | 1.18 |
| m2_houlsby__cal | M2 | Houlsby adapters | 2.38M | 0.534 | 0.283 | 0.270 | 1.30 |
| m1_lorma__cal | M1 | **LoRMA+** | 0.69M | 0.533 | 0.275 | 0.273 | 1.10 |
| m1_houlsby__cal | M1 | Houlsby adapters | 2.38M | 0.527 | 0.273 | 0.273 | 1.14 |
| m1_qlora__cal | M1 | **QLoRA** (NF4) | 0.79M (base 25.4M stored) | 0.524 | 0.270 | 0.269 | 1.10 |
| m1_lora__cal | M1 | LoRA | 0.79M | 0.521 | 0.271 | 0.271 | 1.07 |
| m3_full__cal | M3 | full fine-tuning | 109.5M | 0.467 | 0.200 | 0.231 | 1.55 |
| m3_pfeiffer__cal | M3 | Pfeiffer adapters | 1.19M | 0.417 | 0.158 | 0.219 | 1.04 |
| m1_ia3__cal | M1 | IA³ | 0.03M | 0.415 | 0.182 | 0.212 | 0.92 |
| m3_trunc__cal | M3 | full fine-tuning, **truncated input** (baseline) | 109.5M | 0.361 | 0.134 | 0.202 | 0.53 |
| lead baseline | – | first sentences of the document | – | 0.499 | 0.230 | 0.249 | – |

## 4. Findings

- **Full fine-tuning beats every PEFT method on all four backbones.** Generating long abstractive text needs more capacity than classification did.
- **PEFT methods on M1 land within 0.012 of each other** (LoRMA+ 0.533, Houlsby 0.527, QLoRA 0.524, LoRA 0.521). For this task, which low-rank or adapter method you choose barely matters. **IA³ falls far behind (0.415):** rescaling vectors alone can't learn to generate summaries.
- **Long documents:** on the same M3 model, window-level summarisation beats truncation by **+0.106 ROUGE-1** (0.467 vs 0.361). Truncation can't produce full-length summaries at all: it reaches only 0.53× the reference length even with calibration.
- **Knowledge distillation:** the 6-layer student keeps **96% of the teacher's ROUGE-1** with 38% fewer parameters and generates **4.1× faster** (415 vs 101 tokens/s).
- **The copy head helps:** M4 full fine-tuning has the **best R-2 and R-L** of every run. Copying exact legal phrasing from the window pays off.
- **ROUGE rewards over-length outputs:** re-decoding kd_m1x6 and m2_houlsby at `eos_bias −1.0` (length ≈ 1.0×) *lowered* ROUGE-1 (0.555 → 0.542 and 0.534 → 0.507). Recall-heavy ROUGE favours longer summaries; this is reported, not exploited further.
- **M2 MoE + adapters:** Houlsby on M2 (0.534) beats Houlsby on M1 (0.527) with the same trainable budget.

## 5. Engineering notes

- Background runner with skip-if-done, a "done" record written last, and an automatic backup after each run. Two kernel crashes cost at most one job each.
- `m2_full`'s training finished, but the crash came during its uncalibrated decode. Its base record was reconstructed from the identical config (`note` field in `p6_runs/m2_full.json`), and its calibrated result is a normal run.
- IA³ wasn't originally wired into the Phase 6 dispatcher (`ValueError: ia3`). It was added by reusing the Phase 3 IA³ modules.

## Files

| File | Purpose |
|---|---|
| `p6_sum.py` | Window pairs, model-specific losses, all adaptation methods (incl. NF4 QLoRA and IA³), KD, generation with `eos_bias`, ROUGE |
| `p6_run.py` | Background runner: training, validation and test generation, calibration jobs, submission zips |
| `phase6_notebook.ipynb` | Data profiling, window alignment, launches, calibration, monitoring |

Results: `results/stats/p6_runs/` (`<run>.json` uncalibrated, `<run>__cal.json` calibrated, `__cal_b100` re-decodes at −1.0).
