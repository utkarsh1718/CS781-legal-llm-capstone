# Phase 7: Prior Case Retrieval (PCR)

> **Official test result (Codabench leaderboard):** Micro-F1@7 **0.4383** (Micro-P@7 0.4473 · Micro-R@7 0.4296), **rank 43** (submitted 2026-11-05).

**Task.** For each query case, rank its split's candidate pool and return **exactly 7** unique candidate IDs, best first.
**Metric: Micro-F1@7** (Micro-P@7 and Micro-R@7 secondary). Gold sets are neither truncated nor padded.
**Rules:** start from the Phase 1 model; at least 5 adaptation techniques (**LoRMA compulsory**) plus a **mandatory KD** experiment; no external models or data.

**Depends on:** `code/phase4/p4_cjpe.py` (bidirectional M1 builder, masked-mean `embed`), `code/phase2/p2_peft.py` and `code/phase3/p3_rr.py` (adaptation methods), `code/phase6/p6_sum.py` (NF4 quantisation, LoRA wrapping), `code/phase1` (model, tokenizer).

---

## 1. Data (`Phase_7_PCR`, from the course Hugging Face dataset)

| Split | Queries | Pool | Gold per query (mean / median / max) | Oracle Micro-F1@7 |
|---|---|---|---|---|
| Train | 827 | 4,320 | 6.42 / 4 / 52 (some 0) | 0.639 |
| Dev | 118 | 1,023 | 8.25 / 5 / 92 | 0.618 |
| Test | 237 | 1,727 | – | – |

- **Every query is inside its own pool.** It must be excluded from its own ranking, or it wastes 1 of the 7 slots.
- **Long cases:** median about 4.5k words (4.5k BPE tokens), p90 about 15k; **26% of cases exceed 16 × 512 tokens**.
- No queries are shared across splits, but some candidates are, so embeddings are cached per case.
- Because exactly 7 predictions are forced, the best achievable score is about 0.62–0.64.

## 2. Lexical baseline and evaluator (`p7_lexical_hybrid.py`, P7-1)

My own sparse BM25 (`scikit-learn` count / hashing matrices, vectorised scoring), with the self-match excluded and k1 and b tuned on **Train**. There's no training, so all 827 labelled queries can be used for tuning.

| Lexical variant | Train F1@7 | **Dev F1@7** |
|---|---|---|
| BM25, letters only, first grid (k1 = 2, b = 0.9) | 0.196 | 0.240 |
| BM25, letters only, tuned (k1 = 20, b = 1.0) | 0.294 | 0.394 |
| + numbers | 0.300 | 0.400 |
| + **bigrams** | 0.334 | 0.440 |
| + **trigrams** (1–3-gram, k1 = 10, b = 1.0) | **0.344** | **0.445** |

Strong saturation and full length normalisation matter; **legal phrases** (bigrams and trigrams such as "code of criminal procedure", "section 302", "natural justice") add +0.05.

## 3. Neural retriever (`p7_pcr.py`, `p7_run.py`)

- **Encoder:** bidirectional M1 (Phase 1) with the adaptation method under test; each 512-token chunk becomes a masked-mean vector.
- **Long documents:** up to **16 chunks per case, spread evenly over the whole document**; a case vector is the mean of its normalised chunk vectors.
- **Scoring:** mean-pool cosine vs **MaxSim** (each query chunk matched to its best candidate chunk, averaged), over each split's **full pool**. BM25's top 100 holds only 59–67% of the gold, so re-ranking a shortlist would cap recall.
- **Training:** InfoNCE (τ = 0.05). Per query: 1 gold positive, 2 **BM25 hard negatives** (from the top 50, excluding gold) and **in-batch negatives**. Other gold cases of the same query in the batch are **masked** (no false negatives). 2 random chunks per case during training (memory); 3 epochs.
- **Fusion:** per-query z-scores of dense and lexical scores, $s = \alpha\,z_{dense} + (1-\alpha)\,z_{lex}$, with α chosen on Dev.
- **QLoRA:** M1's block linears quantised to NF4 (my Phase 6 implementation) + LoRA r = 16.
- **KD (mandatory):** teacher `p7_full` (12 layers); student = M1 layers 0, 2, 4, 6, 8, 10. Loss = 0.5·InfoNCE + 0.5·T²·KL(teacher ‖ student) over in-batch candidate similarities (T = 2). Teacher and student see identical sampled chunks.

## 4. Results (Dev, 118 queries)

| Run | Technique | Trainable | Dense mean | Dense MaxSim | Fused (letters BM25) | **Fused (1–3-gram BM25), v2** | Peak mem | Train min |
|---|---|---|---|---|---|---|---|---|
| p7_zeroshot | none (Phase 1 M1 as-is) | 0 | 0.295 | 0.295 | 0.397 | 0.447 | 0.9 GB | 0 |
| p7_prompt | prompt tuning | 0.017M | 0.305 | 0.320 | 0.396 | 0.445 | 6.9 GB | 9.2 |
| p7_ia3 | IA³ | 0.032M | 0.316 | 0.331 | 0.398 | 0.446 | 7.7 GB | 9.6 |
| p7_lorma_plus | **LoRMA+** | 0.395M | 0.341 | 0.359 | **0.409** | 0.446 | 7.0 GB | 9.0 |
| p7_qlora | **QLoRA** (NF4 base) | 0.786M | 0.358 | **0.369** | 0.407 | **0.448** | 7.0 GB | 10.9 |
| p7_partial | partial (top 4 layers) | 25.2M | 0.349 | 0.347 | 0.401 | 0.446 | 4.1 GB | 5.7 |
| p7_full | full fine-tuning (teacher) | 100.1M | 0.351 | 0.351 | 0.402 | **0.448** | 11.8 GB | 10.7 |
| p7_kd_m1x6 | **KD student** (6 layers) | 62.3M | 0.329 | 0.342 | 0.402 | **0.448** | 6.6 GB | 8.4 |
| p7_lorma_k3e5 | LoRMA+, 3 chunks / case, 5 epochs | 0.395M | 0.341 | 0.367 | 0.411 | 0.447 | 10.2 GB | 21.1 |
| p7_qlora_k3e5 | QLoRA, 3 chunks / case, 5 epochs | 0.786M | 0.358 | 0.358 | 0.411 | **0.448** | 10.2 GB | 26.6 |
| **hybrid_v2** | 1–3-gram BM25 + dense ensemble (full, KD, QLoRA), α = 0.65 | – | | | | **0.448** | | |

## 5. Findings

- **Adapting M1 clearly improves its own retrieval:** dense F1 rises from 0.295 (zero-shot) to 0.36–0.37 (QLoRA, LoRMA+). Phase 1 representations transfer, but they need contrastive adaptation.
- **Capacity ordering:** prompt < IA³ < partial ≈ full < LoRMA+ ≈ QLoRA. **The low-rank methods beat full fine-tuning** with only 827 training queries; full fine-tuning over-adapts.
- **MaxSim beats mean pooling** in 6 of the 10 runs (3 ties, 1 worse). Legal relevance is often a match between specific passages, not whole documents.
- **More context per case in training** (3 chunks instead of 2) improves LoRMA+'s MaxSim (0.359 → 0.367).
- **KD:** the 6-layer student matches the teacher's fused score (0.448 vs 0.448) and keeps 97% of its dense MaxSim, with **38% fewer parameters, 2× faster encoding (49 s vs 100 s for the Dev pool) and 44% less peak memory**.
- **Fusion:** the models add a consistent but small gain over a strong lexical baseline. With the trigram BM25, every technique lands at 0.445–0.448. On 118 queries, 0.001 F1 is about one more correct case, so **those differences are within noise**. The dense-only columns are the fair comparison between techniques.

**Negative results (Section P7-5B/C):**
- **Citation matching:** a candidate's own citation (from its header) almost never appears in a query (1 match in all of Train). Citations are scrubbed from the texts.
- **Precedent popularity prior** (how often a candidate is gold for training queries): weight 0 on Train, so dropped.
- **Pseudo-relevance feedback** (boosting candidates similar to a query's top hits): β = 0 on Train. Gold precedents aren't simply near-duplicates of the top lexical hits.

**Bug found and fixed:** the BM25 matrix stored each query's own column as −1e9 (to exclude it), and that value entered the per-query z-score. It flattened every BM25 score, so every fused run silently fell back to "BM25 only" (α = 0). The fix marks the self column NaN and uses NaN-aware z-scores; all runs were rescored from saved embeddings with no retraining (`rescore()`).

## Files

| File | Purpose |
|---|---|
| `p7_pcr.py` | Token cache reader, chunking, encoder build (incl. QLoRA, layer-subset students), InfoNCE training, KD training, encoding, mean / MaxSim scoring, fusion, exact Micro-F1@7, submission writer, `rescore()` |
| `p7_run.py` | Background runner (`--plan`, `--smoke`, `--smoke2`), skip-if-done |
| `p7_lexical_hybrid.py` | Notebook cells for BM25 tuning, n-gram BM25, citation / prior / PRF tests, and the v2 re-fusion |
| `phase7_notebook.ipynb` | The complete Phase 7 notebook, with outputs |

Results: `results/stats/p7_runs/*.json` (one per technique), `p7_v2.json` (trigram re-fusion and hybrid).
