# PatentLens — Model Training Results

Produced by training on the full 100,000-patent corpus (`data/raw/patents_g06n3_wide_100k.csv`).
Re-run to regenerate every number and chart here — see Reproducing below.

## Dataset

- 100,000 US patents under CPC subclass **G06N3** (neural network architectures), title + abstract text, real citation graph. Pulled from Google Patents' public BigQuery dataset.
- Citation ground truth: 44,985 of 100,000 patents (45.0%) cite at least one other patent that also falls inside the sample — 183,973 usable citation pairs total. (Compare: an earlier 3,000-patent pilot sample had only 127 query patents / 147 pairs, 4.2% coverage — the larger corpus makes every metric below far less noisy.)
- Evaluated on 10,000 of those 44,985 query patents (randomly sampled, fixed seed) — see Limitations.
- Scaling further, or fetching directly via BigQuery yourself, is wired up in [`src/patentlens/data_fetch.py`](src/patentlens/data_fetch.py).

## Methodology

**Models compared:**

| Model | Type |
|---|---|
| TF-IDF | Lexical, unigrams+bigrams, 5000-term vocabulary |
| BM25 | Lexical, probabilistic ranking (custom sparse-matrix scorer — see Implementation note) |
| LSA | TF-IDF + Truncated SVD (100 components, 22.0% variance explained) |
| MiniLM | General-purpose sentence embeddings (`all-MiniLM-L6-v2`), GPU-encoded, FAISS-indexed |

PatentSBERTa (a patent-claims-tuned embedding model) and a BM25+PatentSBERTa hybrid were part of an earlier pilot run on the 3,000-patent sample but are not included in this 100k-patent run — PatentSBERTa's GPU encoding time (~55 min on a 4GB laptop GPU) didn't fit the available time budget for this pass. On the smaller sample it underperformed general-purpose MiniLM, so it's a lower-priority addition; re-adding it is a straightforward re-run (`retrieval.EmbeddingRetriever(model_name="AI-Growth-Lab/PatentSBERTa")`) whenever there's GPU time to spare.

**Evaluation:** for every sampled query patent, rank all other patents by similarity/score and check whether the true cited patent(s) land near the top. This is a proxy metric — patent citations are a legal/prior-art decision, not a pure text-similarity judgment — but it's the only ground truth available without manual labeling.

**Metrics:** Recall@k, Precision@k, NDCG@k (k=5,10,20), and MRR, each with a 95% bootstrap confidence interval (2,000 resamples). NDCG additionally rewards ranking a true citation *higher* within the top k, not just including it.

**Significance:** a paired bootstrap test (5,000 resamples, same queries for both models each time) checks whether the top model by MRR is *actually* better than each alternative, not just higher on a bar chart.

**Implementation note:** the `rank_bm25` library's scorer loops over every document with Python-level dict lookups per query term — fine at a few hundred evaluation queries, but ~0.35s/query at 100k documents (10,000 queries would take most of an hour). `src/patentlens/retrieval.py`'s `Bm25Retriever` reimplements standard BM25 (Robertson-Sparck Jones idf, k1/b saturation) as a precomputed sparse matrix, so ranking becomes one sparse matrix-vector multiply — the same trick that makes TF-IDF fast. Benchmarked at ~17ms/query at 100k documents, a ~20x speedup, with output verified against hand-checked examples.

## Results

| Model | Recall@10 | Precision@10 | NDCG@10 | MRR |
|---|---|---|---|---|
| **MiniLM** | 0.1034 | 0.0308 | 0.0814 | **0.1126** |
| BM25 | 0.1020 | 0.0308 | 0.0809 | 0.1106 |
| TF-IDF | 0.0790 | 0.0238 | 0.0660 | 0.0953 |
| LSA | 0.0629 | 0.0192 | 0.0553 | 0.0822 |

Full table (all k values): [`models/metrics_pivot.csv`](models/metrics_pivot.csv) (gitignored — regenerate by re-running, see below).

![Recall@k comparison](outputs/recall_comparison.png)
![NDCG@k comparison](outputs/ndcg_comparison.png)
![MRR comparison](outputs/mrr_comparison.png)

### Is MiniLM actually the best model, or just highest on the chart?

Paired bootstrap test, MiniLM vs. each alternative, on MRR:

| Compared to | Mean MRR gap | 95% CI | p-value | Significant at 0.05? |
|---|---|---|---|---|
| LSA | +0.0303 | [0.0271, 0.0335] | <0.001 | **Yes** |
| TF-IDF | +0.0173 | [0.0142, 0.0203] | <0.001 | **Yes** |
| BM25 | +0.0020 | [-0.0011, 0.0050] | 0.21 | No |

**Takeaway: MiniLM is confidently better than TF-IDF and LSA, but statistically indistinguishable from BM25.** Even with 10,000 evaluation queries (vs. ~150 citation pairs in the original pilot), a mean-MRR gap of 0.002 isn't distinguishable from noise. Report "MiniLM and BM25 are statistically tied for best" — not "MiniLM wins."

## Key findings

1. **Scores dropped substantially vs. the 3,000-patent pilot** (e.g., MiniLM MRR 0.113 here vs. ~0.17 there). **This is expected, not a regression.** At 100,000 patents there are far more plausible-looking distractors for any retriever to get confused by than at 3,000 — recall on a fixed-size top-k naturally falls as the haystack grows. The pilot's higher numbers were partly an artifact of a smaller, easier search space.
2. **Semantic (MiniLM) and lexical (BM25) retrieval are statistically tied at this scale.** Neither approach is a clear winner — that held true in the 3,000-patent pilot too, so it's a consistent finding, not noise from one run.
3. **LSA remains the weakest model** — compressing to 100 components keeps only ~22% of the original TF-IDF variance at this corpus size, losing more signal than it saves in compute.
4. **Absolute scores are low across the board** (best MRR ≈ 0.11, meaning the true citation lands around rank 9 on average when it's found at all). Patent citations reflect legal/prior-art judgment as much as topical similarity, and the search space here is large — no text-only retrieval model should be expected to get anywhere close to perfect recall.
5. **Citation ground-truth coverage improved dramatically with scale** — 45.0% of patents now have an in-sample citation (up from 4.2% in the pilot), which is why every confidence interval above is much tighter than the pilot's despite evaluating on a 10k-query sample rather than the full 44,985.

## Limitations

- **Evaluated on a 10,000-query sample, not the full 44,985 ground-truth queries** — a pragmatic tradeoff for training-run turnaround time, not a data limitation. 10,000 is already ~65x the pilot's entire ground truth (147 pairs), so this is a substantial statistical-power upgrade, not a step down.
- **Citation-based evaluation is a proxy.** A model that's "wrong" about a citation may still be topically on point — citations are filed for legal reasons (prior art disclosure requirements), not because two patents read as similar.
- **PatentSBERTa and the Hybrid model are not included in this run** (see Methodology) — on the smaller pilot sample PatentSBERTa underperformed MiniLM, so its omission here is a time-budget tradeoff, not expected to change the headline conclusion, but it hasn't been verified at this corpus size.
- **TF-IDF and LSA evaluation is still slower than ideal** (~15-17 min each for 10k queries, vs. BM25's ~4 min after the sparse-matrix rewrite) — their `rank()` implementations still do a full `argsort` per query rather than the partial-sort/argpartition optimization applied to BM25. Not a correctness issue, just a known remaining efficiency gap for future work.

## Reproducing

```bash
# from the PatentLens repo root, with venv activated
python scripts/train.py          # checkpointed: safe to re-run, skips any already-completed step
streamlit run app.py
```

`scripts/train.py` saves each fitted model and each model's evaluation results to `models/` as soon as they're computed (not just at the end), so an interrupted run can be resumed by simply running it again — already-completed steps are loaded from disk instead of recomputed. Delete files under `models/` to force specific steps to redo.
