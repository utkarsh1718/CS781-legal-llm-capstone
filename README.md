# CS781 Capstone: Legal Language Models from Scratch (Phases 1–7)

**Utkarsh Singh · IIT Kanpur · CS781 (Large Language Models), Prof. Ashutosh Modi · 2026**

This repository holds the code, experiment records and documentation for the CS781 capstone. The capstone is a seven-phase competition on **Indian legal text**. In Phase 1 we pretrain our own language models from scratch. In Phases 2–7 every task is solved by **adapting those same models**, with no external pretrained model, LLM API or tokenizer.

| Phase | Task | Metric | Our approach (best system) | Best dev | **Test (leaderboard)** | Rank |
|---|---|---|---|---|---|---|
| 1 | Legal text continuation (512-word prompt → next 128 words) | BERTScore F1 | 4 LMs (~100M each) pretrained from scratch on 700M tokens with our own 32k BPE; MBR-9 + copy-aware decoding | 0.8499 (M2, local) | **0.8445** | 9 |
| 2 | Legal NER (12 labels, character spans) | strict macro-F1 | Bidirectional M1 + full fine-tuning + InLegalNER intermediate training + word-consistent Viterbi; 4-model ensemble | 0.635 | **0.6407** | 10 |
| 3 | Rhetorical role prediction (13 roles, CL + IT) | macro-F1 | Sentence windows with document context + relative position; document-level BiGRU over frozen rsLoRA sentence vectors; ensembles | 0.602 | **0.5296** | 38 |
| 4 | Court judgment prediction (binary) | macro-F1 | M1 full fine-tuning on the closing reasoning + hierarchical BiGRU-attention over all chunks; seed ensemble | 0.815 (honest) | **0.8094** | 14 |
| 5 | Legal statute identification (100 labels) | macro-F1 | Full / partial fine-tuning on first-1024 + hierarchical models + shrunk per-statute thresholds + label-correlation stacker | 0.434 (honest) | **0.4399** | 13 |
| 6 | Abstractive summarisation | BERTScore (ROUGE secondary) | Window-level summarisation (summary sentences aligned to 520-word windows) + per-model length calibration | ROUGE-1 0.577 | **0.8494** | 25 |
| 7 | Prior case retrieval (top-7) | Micro-F1@7 | 1–3-gram BM25 fused with M1 bi-encoders (chunk MaxSim), contrastive training with BM25 hard negatives | 0.448 | **0.4383** | 43 |

"Honest" means the members, thresholds and fusion weights were chosen on one half of dev and scored on the other half (cross-fitted), so the number isn't inflated by selection on the same data. Each phase's README lists every experiment.

---

## Official leaderboard results (Codabench, best submission per phase)

| Phase | Task | Primary metric | **Test score** | Secondary columns | Rank | Date |
|---|---|---|---|---|---|---|
| 1 | Text continuation | BERTScore F1 | **0.8445** | ROUGE-L 0.2084 · BLEU-4 0.0703 | 9 | 2026-10-01 |
| 2 | Legal NER | strict macro-F1 | **0.6407** | other columns 0.6534 / 0.6424 | 10 | 2026-10-07 |
| 3 | Rhetorical roles | macro-F1 | **0.5296** | other columns 0.5388 / 0.5273 | 38 | 2026-10-14 |
| 4 | Judgment prediction | macro-F1 | **0.8094** | other columns 0.799 / 0.8294 | 14 | 2026-10-17 |
| 5 | Statute identification | macro-F1 | **0.4399** | micro-F1 0.5309 · subset accuracy 0.0628 | 13 | 2026-10-27 |
| 6 | Summarisation | BERTScore F1 | **0.8494** | ROUGE-1 0.5522 · ROUGE-L 0.2848 | 25 | 2026-10-30 |
| 7 | Prior case retrieval | Micro-F1@7 | **0.4383** | Micro-P@7 0.4473 · Micro-R@7 0.4296 | 43 | 2026-11-05 |

Test scores track the dev scores closely in Phases 1, 2, 4, 5 and 7. Phase 3 dropped about 0.05–0.07 from dev to test (about 80 training documents), which motivated the train+dev retraining.

## Ground rules we worked under

- **Compute:** Kaggle, 2× NVIDIA T4 (16 GB, **no bf16**, so all training is fp16 + GradScaler), 30 GPU-hours per week per account, 12-hour sessions.
- **Models:** all four backbones are trained from scratch in Phase 1 (50–300M parameters). Every later phase must start from them.
- **Tokenizer:** our own byte-level BPE, grown out of Assignment 1 (`code/phase1/bpe_tokenizer.py`, saved in `tokenizer/`).
- **No external pretrained models, corpora or APIs.** The one exception is InLegalNER in Phase 2, which the TAs allowed. It is documented, and its leakage against the test fold was checked.
- **Phases 2–7:** at least 6 substantially different experiments each, including at least 5 adaptation techniques (LoRMA compulsory) and 1 knowledge-distillation experiment. Every phase meets this; see the per-phase tables.

## The four Phase 1 backbones

| | Architecture | Params (active) | Key ideas |
|---|---|---|---|
| **M1** | Dense decoder | 100.1M | Pre-norm, RMSNorm, SwiGLU, RoPE, GQA (12 query / 4 KV heads), QK-norm for fp16 stability, tied embeddings |
| **M2** | DeepSeek-style decoder | 270.4M (101.7M) | Multi-head Latent Attention (288 cached values per token per layer vs 512 for GQA) + DeepSeekMoE (1 shared + 16 routed experts, top-3), auxiliary-loss-free balancing, static-shape expert dispatch |
| **M3** | Encoder–decoder | 109.5M | 640-token bidirectional encoder → 384-token decoder; mirrors "read the prompt, write the continuation" |
| **M4** | Dense decoder + copy head | 100.3M | Learned pointer–generator gate (+0.2% parameters); exact mixture NLL, so its loss is directly comparable to M1's |

All four use the same recipe: 700M tokens, 256 × 1,024-token batches, WSD learning-rate schedule, and a curriculum that switches to Supreme-Court-heavy data when the LR decays. That makes the architecture the only variable.

## Repository layout

```
code/
  phase1/   tokenizer, corpus pipeline, M1–M4 model code, training engine, generation, local eval
  phase2/   L-NER: char-offset alignment, token tagger, PEFT methods + KD
  phase3/   rhetorical roles: sentence-window model, background runner, document-BiGRU upgrades
  phase4/   judgment prediction: section views, document classifier, hierarchical chunk model
  phase5/   statute identification: multi-label head, thresholds, train+dev retraining
  phase6/   summarisation: window-level pairs, 8 adaptation methods on all 4 models, calibration
  phase7/   retrieval: chunk bi-encoder, contrastive training, QLoRA, KD, lexical hybrid
  phaseN/phaseN_notebook.ipynb   that phase's section of the Kaggle notebook (with outputs)
results/stats/   every run's result JSON, plans and logs (the source of every number in the READMEs)
tokenizer/       A1-BPE-v2 32k (our Python format + Hugging Face `tokenizers` export)
report_tables/   LaTeX result tables generated from results/stats
```

Code that later phases reuse lives in earlier phases. Phase 2's `p2_model.py` / `p2_peft.py` build every adapted backbone. Phase 4's `p4_cjpe.py` provides the document encoder used in Phases 5 and 7. Phase 6's `p6_sum.py` provides NF4 quantisation and IA³ for Phase 7. Each phase's README names its dependencies.

## Execution instructions

### Environment
- **Kaggle notebook, GPU T4 × 2**, Python 3.12, internet on. Extra packages: `pip install -r requirements.txt`.
- Every path in the code assumes `/kaggle/working` as the working directory.
- **Setup:** copy every `code/phase*/*.py` into `/kaggle/working/src/` (the modules import each other by name) and `tokenizer/*.json` into `/kaggle/working/tokenizer/`.
- **Data:** the course dataset `Exploration-Lab/CS-781-Capstone` on Hugging Face (subsets `phase_1` … `phase_7`).

### Running each phase
Each phase can be rerun end to end from its notebook section, `code/phaseN/phaseN_notebook.ipynb`; the cells run in order. The heavy steps are scripts:

| Phase | Data preparation | Training / experiments | Inference / submission |
|---|---|---|---|
| 1 | `profile_corpus.py` → `shard_corpus.py` (4 sources in parallel) → `valset.py` | `torchrun --standalone --nproc_per_node=2 src/train.py --config <run.json>` (same command resumes) | `infer.py` (`continue_prompts`, MBR + copy, `write_submission`); `eval_local.py` for local BERTScore / ROUGE / BLEU |
| 2 | Phase 2 notebook cells (`p2_ner.py`: offset mapping, BIO); InLegalNER conversion cells | notebook cells calling `p2_model.train_tagger` with methods from `p2_peft.py` | word-Viterbi decoding + ensemble cells → `prediction.json` |
| 3 | notebook cell P3-1 (sentence windows) | `python src/p3_run.py --plan <plan.json>`; document-BiGRU and train+dev: `p3_extra.py` | ensemble / submission cells (P3-5, P3-X) |
| 4 | notebook cell P4-1 (dedup, test-in-dev cleaning, token cache) | `python src/p4_run.py --plan <plan.json>` | threshold tuning, ensembles, submission cells (P4-4, P4-6) |
| 5 | notebook cell P5-1 (multi-label cache, clean splits) | `python src/p5_run.py --plan <plan.json>`; train+dev: `python src/p5_td.py --plan <plan.json>` | shrunk thresholds, label-correlation stacker, submission cells (P5-8, P5-B3, PX) |
| 6 | notebook cell P6-1 (window–summary alignment, token cache) | `python src/p6_run.py --plan <plan.json>` (`--smoke` first; `"type": "cal"` jobs run the `eos_bias` calibration) | written automatically: `submissions_p6/p6_<run>/p6_<run>.zip` |
| 7 | `p7_lexical_hybrid.py` cells P7-1, P7-2c (BM25 matrices, BPE cache) | `python src/p7_run.py --plan <plan.json>` (`--smoke`, `--smoke2` first) | per run: `submissions_p7/p7_<run>/`; final hybrid: cells P7-5B/C in `p7_lexical_hybrid.py` |

A **plan** is a JSON list of runs, for example `[{"name": "p7_lorma_plus", "method": "lorma_plus", "epochs": 3}]`. Launch one runner per GPU in the background:
```bash
cd /kaggle/working/src && CUDA_VISIBLE_DEVICES=0 nohup python p7_run.py --plan ../stats/plan_gpu0.json > ../stats/gpu0.log 2>&1 &
```
A run is done when its result JSON (`stats/pN_runs/<name>.json`) exists. Finished runs are skipped on relaunch, so a crashed session loses at most one run.

Checkpoints, data caches and embeddings are not in this repository because of size (they were kept in a private Hugging Face repo during the project).

## Submissions per phase

Each submission is described, with its validation result, in that phase's README and in `results/stats`:

| Phase | Submitted systems (best first) | Record |
|---|---|---|
| 1 | copy baseline; M1 greedy (rep 1.2); M1 / M2 / M3 MBR-9 + copy; M4 prompt-only pointer (δ = +1) + MBR-9 | `decoding_m*_v2.json` |
| 2 | single models per method (LoRA, LoRMA+/π, adapters, KD, M3 / M2 variants); full fine-tuning + bidirectional + InLegalNER; seed ensemble + LoRA-ext (final) | `p2_results.json`, `p2_ensemble.json` |
| 3 | 11 technique × backbone runs; top-7 ensemble; document BiGRU (`h_*`); train+dev retrains | `p3_submissions.json`, `p3_upgrade.json` |
| 4 | technique runs (last 512 / 1,024); hierarchical BiGRU; greedy and seed ensembles | `p4_submissions.json`, `p4_seed_ensembles.json` |
| 5 | technique runs; first-1,024 full / partial; hierarchical ensembles (shrunk thresholds); label-correlation stacker; train+dev and rate-matched variants | `p5_systems.json`, `p5_lc.json` |
| 6 | 14 calibrated runs `<run>__cal` (all 4 backbones; full, LoRA, LoRMA+, QLoRA, Houlsby, Pfeiffer, IA³, KD, truncation baseline) | `p6_runs/*__cal.json` |
| 7 | 10 retrieval runs (zero-shot, full, partial, LoRMA+, QLoRA, IA³, prompt, KD, 2 longer-context runs), each re-fused with 1–3-gram BM25 (`*_v2`); `hybrid_v2` (final) | `p7_runs/*.json`, `p7_v2.json` |

## Engineering practices that mattered

- **Resumable everything:** deterministic data sampling, so pretraining resumes exactly across sessions; result JSONs written last as "done" markers; automatic backup after every run.
- **Honest evaluation:** test labels were never used. Phase 4's test set was found inside dev (identical texts), reported to the TAs, and removed from dev. Thresholds and ensembles are cross-fitted.
- **Smoke tests before every launch:** every model × method code path runs for 2 steps before any real GPU time is spent.
- **Documented failures:** the repetition-penalty scale bug (Phase 1, M4), the self-column z-normalisation bug in retrieval fusion (Phase 7), and negative results such as the QK-norm ablation, document-consistency rules, citation matching and pseudo-relevance feedback are all recorded in the phase READMEs, with the fix or the reason.
