# Phase 1: Legal language models from scratch

> **Official test result (Codabench leaderboard):** BERTScore F1 **0.8445** (ROUGE-L 0.2084 · BLEU-4 0.0703), **rank 9** (submitted 2026-10-01).

**Task.** Given a 512-word prompt from an Indian legal document, generate the next 128 words.
**Metric:** BERTScore F1 (ROUGE-L and BLEU-4 secondary). **Rules:** 50–300M parameters, our own tokenizer, no pretrained weights.

This phase builds everything the later phases reuse: the tokenizer, the corpus pipeline, four model architectures, the training engine and KV-cached generation.

---

## 1. Data understanding (`profile_corpus.py`, `provenance.py`)

One streaming pass per source, in constant memory: document counts, length distribution, exact-duplicate rate, non-ASCII and newline rates, year and court distributions, a 5,000-document reservoir sample, and an Aho–Corasick leakage check against the test prompts.

| Source | Documents | Words | Exact-duplicate words |
|---|---|---|---|
| Supreme Court | 55,549 | 210M | 0.3% |
| High Courts | 2,536,738 | 1,595M | **25.7%** |
| Tribunals | 869,132 | 976M | 0.4% |
| Acts | 18,826 | 120M | 0.3% |
| **Total** | | **2.90B** | |

**Findings that shaped the design:**
- The 1,000 test prompts are **ASCII-normalised, with single spaces and no newlines**, so all training text is normalised the same way (`common.normalise`).
- About 90% of test prompts are Supreme-Court-style, so the corpus mix leans towards SC.
- 426 of 1,000 test prompts share a 20-word span with training data. A precise check found **9 SC documents that contain a full test prompt**; they were removed (decontamination).

## 2. Tokenizer: A1-BPE-v2, 32k (`bpe_tokenizer.py`, `common.py`)

A byte-level BPE grown out of my Assignment 1 tokenizer.
- **Kept from A1:** the merge learner, which counts unique pre-tokenised chunks and keeps a pair → chunks index and a lazy max-heap, so each merge only touches the chunks containing that pair.
- **Lossless pre-tokeniser:** a GPT-2-style regex where each word keeps its leading space, so `decode(encode(x)) == x`. A1's `nltk` + `</w>` scheme dropped whitespace.
- **Digit grouping:** numbers are split into chunks of at most 3 digits, so citations such as "AIR 1984 SC 1020" don't each use up a vocabulary entry.
- **16 reserved special tokens** (document separator, pad, mask, sep, cls, spares), used by the task heads in Phases 2–7.
- **Fast encoding:** the learned merges are exported to Hugging Face's Rust `tokenizers` engine. An assertion checks that both encoders give identical ids.

## 3. Corpus construction (`shard_corpus.py`, `valset.py`)

`normalise → length filter → exact dedup → decontaminate → hash-based document split → sample → tokenise → uint16 shards`, run as 4 parallel processes.
- Train/validation split by document hash, so duplicates always land in the same split. Validation is 2% for SC and Acts, 0.2% for HC and Tribunals.
- 40% hash-sampling of HC and Tribunals to fit the compute budget; SC and Acts kept in full; SC headnotes added as extra documents.
- The source mixture is chosen **at training time** by sampling weights, so changing the mix never requires re-tokenising.
- **Validation set in the exact test format:** 512-word prompts starting at a sentence boundary plus the next 128 words (1,000 SC + 150 per other source), along with per-source loss shards for **bits per word**. Bits per word is comparable across tokenizers and architectures; perplexity per token is not.

## 4. Models (`lm_common.py`, `model_dense.py`, `model_mla_moe.py`, `model_encdec.py`, `model_copy.py`)

| | Design | Params (active) | Why |
|---|---|---|---|
| **M1** | Dense decoder: pre-norm, RMSNorm (fp32 statistics), SwiGLU, RoPE, **GQA 12/4**, **QK-norm**, tied embeddings | 100.1M | Controlled baseline. QK-norm bounds attention logits in fp16 (the T4 has no bf16). GQA gives a 3× smaller KV cache. |
| **M2** | **MLA** (latent d_c = 256 + shared 32-dim RoPE key; absorbed up-projections at decode time) + **DeepSeekMoE** (1 shared + 16 routed experts, top-3, aux-loss-free bias balancing, static-shape batched dispatch) | 270.4M (101.7M) | Compute-matched dense vs sparse comparison. MLA caches 288 values per token per layer, vs 512 for GQA, which makes multi-sample MBR cheap. |
| **M3** | Encoder–decoder, 6 + 6 layers: 640-token bidirectional encoder, 384-token decoder with cross-attention | 109.5M | Mirrors the task. The encoder is reused for Phases 2–3. |
| **M4** | M1 + learned **pointer–generator copy head**: $p = (1-g)\,p_{vocab} + g\,p_{copy}$, an induction-style pointer, exact mixture NLL | 100.3M (+0.2%) | Learns *when* to copy, instead of a fixed decoding-time bonus. |

Attention uses `scaled_dot_product_attention`, which on the T4 is the memory-efficient kernel (FlashAttention-2 needs sm_80+). All models have KV-cached generation; M3 caches the cross-attention K/V once per prompt.

## 5. Training engine (`train.py`, `data.py`)

- fp16 autocast + GradScaler, **DDP on 2× T4** via `torchrun`, gradient accumulation, and `torch.compile` (a measured **1.65× speed-up**).
- **WSD schedule:** 3% warmup, a constant 6e-4, then cosine decay to 6e-5 over the last 20%.
- **Curriculum:** SC 30 / HC 30 / Tribunals 25 / Acts 15 for the first 80% of training, then an **SC-heavy anneal (SC 75%)** while the LR decays.
- **Same budget for every model:** 700M tokens, 256 × 1,024 tokens per step (2,670 steps).
- Batches are a pure function of (seed, step, rank), so **resume is exact** across Kaggle sessions. Checkpoints are written atomically (`latest.pt`, plus `best.pt` by SC validation loss).

| Run | Params (active) | Train hours | tok/s (2×T4) | SC bits/word | HC | Tribunals | Acts |
|---|---|---|---|---|---|---|---|
| M1 dense | 100.1M | 4.92 | 39.1k | 4.154 | 3.497 | 4.280 | 3.914 |
| M2 MLA + MoE | 270.4M (101.7M) | 6.20 | 31.5k | **4.066** | **3.368** | **4.135** | **3.806** |

SC validation loss: M1 2.338, **M4 2.301**, **M2 2.288**. At equal active compute, the sparse model and the copy head both beat the dense baseline.

## 6. Generation and decoding study (`generation.py`, `infer.py`, `eval_local.py`)

Generation is KV-cached, keeps the last 824 prompt tokens, bans `<|endoftext|>`, forces the first token to start a new word, and cuts output to exactly 128 words. Local metrics: BERTScore (roberta-large), ROUGE-L and BLEU-4 on 350 validation prompts. These are only used to compare settings with each other.

| System (350 val prompts) | BERTScore F1 | SC subset | rep3 |
|---|---|---|---|
| Copy the last 128 prompt words (baseline) | 0.8430 | 0.8412 | 0 |
| M1 greedy, repetition penalty 1.2 | 0.8438 | 0.8405 | 0.052 |
| M1 + prompt-copy bonus β = 1 | 0.8453 | 0.8432 | 0.085 |
| M1 **MBR-9** + copy β = 1 | 0.8495 | 0.8462 | 0.086 |
| M2 MBR-9 + copy β = 1 | **0.8499** | 0.8460 | 0.081 |
| M3 MBR-9 + copy β = 1 | 0.8377 | 0.8380 | 0.094 |
| M4 prompt-only pointer, gate δ = +1, MBR-9 | 0.8493 | 0.8459 | 0.054 |

- **Copy-aware decoding.** Officially, the trivial copy baseline (0.8412) beat plain M1 (0.8364). Legal continuations reuse the prompt heavily, so we added a prompt-copy logit bonus.
- **MBR (Minimum Bayes Risk).** Sample 9 candidates and pick the most "central" one by n-gram F1. A shared-prefix KV cache means the prompt is prefilled only once.
- **M4's copy head matches hand-tuned decoding without a hand-tuned bonus**, with about 35% less repetition.

**Failure we diagnosed (M4).** Greedy M4 first looped (rep3 = 0.54). The CTRL repetition penalty divides positive logits, but M4 returned **log-probabilities (≤ 0)**, so the penalty did almost nothing. The fix was to return the mixture on the logit scale ($\log p_{mix} + \mathrm{logsumexp}(z)$), plus a **prompt-only pointer** and a calibrated **gate shift δ**. The lesson: decoding heuristics that depend on the scale of the outputs break silently when the output layer changes.

## 7. Ablations of M1's own design choices (100M-token budget each)

| Variant | SC val loss | L-NER macro-F1 (Phase 2, 3 seeds) |
|---|---|---|
| Base (QK-norm on, tied embeddings) | 3.427 | 0.412 ± 0.010 |
| **QK-norm off** | **3.270** | **0.440 ± 0.003** |
| Untied embeddings (+24.5M params) | 3.293 | 0.421 ± 0.010 |

QK-norm *hurt* at this small budget. RMS-normalised q and k, scaled by 1/√64, start attention very diffuse, and 381 steps aren't enough for the norm gains to grow. The gap narrows by 30–40% in the second half of training, so the full-budget M1 keeps QK-norm for fp16 stability (its effect at full budget is untested).

## Files

| File | Purpose |
|---|---|
| `profile_corpus.py`, `provenance.py` | Corpus statistics, leakage and test-prompt provenance |
| `common.py`, `bpe_tokenizer.py` | Normalisation, deterministic split, A1-BPE-v2 tokenizer |
| `shard_corpus.py`, `valset.py`, `data.py` | Corpus pipeline, validation sets, mixture sampler |
| `lm_common.py`, `model_*.py` | Shared blocks and M1–M4 |
| `train.py` | DDP / fp16 / compile / WSD / curriculum training engine |
| `generation.py`, `infer.py`, `eval_local.py` | KV-cached decoding, MBR + copy, local metrics, submission writing |
| `backup_hf.py` | Private Hugging Face backup of checkpoints and results |
| `phase1_notebook.ipynb` | The full Phase 1 notebook, with outputs |
