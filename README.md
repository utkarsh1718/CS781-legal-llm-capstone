# CS781 Capstone: Legal Language Models from Scratch (Phases 1–7)

**Utkarsh Singh · IIT Kanpur · CS781 (Large Language Models), Prof. Ashutosh Modi · 2026**

This repository holds the code, experiment records and documentation for the CS781 capstone. The capstone is a seven-phase competition on **Indian legal text**. In Phase 1 we pretrain our own language models from scratch. In Phases 2–7 every task is solved by **adapting those same models**, with no external pretrained model, LLM API or tokenizer.

| Phase | Task | Metric | Our approach (best system) | Best dev |
|---|---|---|---|---|
| 1 | Legal text continuation (512-word prompt → next 128 words) | BERTScore F1 | 4 LMs (~100M each) pretrained from scratch on 700M tokens with our own 32k BPE; MBR-9 + copy-aware decoding | 0.8499 (M2, local) |
| 2 | Legal NER (12 labels, character spans) | strict macro-F1 | Bidirectional M1 + full fine-tuning + InLegalNER intermediate training + word-consistent Viterbi; 4-model ensemble | 0.635 |
| 3 | Rhetorical role prediction (13 roles, CL + IT) | macro-F1 | Sentence windows with document context + relative position; document-level BiGRU over frozen rsLoRA sentence vectors; ensembles | 0.602 |
| 4 | Court judgment prediction (binary) | macro-F1 | M1 full fine-tuning on the closing reasoning + hierarchical BiGRU-attention over all chunks; seed ensemble | 0.815 (honest) |
| 5 | Legal statute identification (100 labels) | macro-F1 | Full / partial fine-tuning on first-1024 + hierarchical models + shrunk per-statute thresholds + label-correlation stacker | 0.434 (honest) |
| 6 | Abstractive summarisation | BERTScore (ROUGE secondary) | Window-level summarisation (summary sentences aligned to 520-word windows) + per-model length calibration | ROUGE-1 0.577 |
| 7 | Prior case retrieval (top-7) | Micro-F1@7 | 1–3-gram BM25 fused with M1 bi-encoders (chunk MaxSim), contrastive training with BM25 hard negatives | 0.448 |

"Honest" means the members, thresholds and fusion weights were chosen on one half of dev and scored on the other half (cross-fitted), so the number isn't inflated by selection on the same data. Each phase's README lists every experiment.

---

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

## Reproducing

Everything ran on Kaggle with the code in `/kaggle/working/src/` and data under `/kaggle/working/`. To rerun a phase:

1. Copy every `code/phase*/*.py` into one `src/` folder (the modules import each other by name).
2. Download the phase's data from the course dataset `Exploration-Lab/CS-781-Capstone` on Hugging Face.
3. Run the cells of `code/phaseN/phaseN_notebook.ipynb` in order.

Long experiments run as background processes (`p*_run.py --plan plan.json`), one per GPU. A run is marked done when its result JSON exists, and finished runs are skipped on relaunch, so a crashed session loses at most one run.

Checkpoints, data caches and embeddings are not in this repository because of size (they were kept in a private Hugging Face repo during the project).

## Engineering practices that mattered

- **Resumable everything:** deterministic data sampling, so pretraining resumes exactly across sessions; result JSONs written last as "done" markers; automatic backup after every run.
- **Honest evaluation:** test labels were never used. Phase 4's test set was found inside dev (identical texts), reported to the TAs, and removed from dev. Thresholds and ensembles are cross-fitted.
- **Smoke tests before every launch:** every model × method code path runs for 2 steps before any real GPU time is spent.
- **Documented failures:** the repetition-penalty scale bug (Phase 1, M4), the self-column z-normalisation bug in retrieval fusion (Phase 7), and negative results such as the QK-norm ablation, document-consistency rules, citation matching and pseudo-relevance feedback are all recorded in the phase READMEs, with the fix or the reason.
