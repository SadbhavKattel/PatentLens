# PatentLens — Evaluation of the Patent Similarity Engine

**Model:** TF-IDF + cosine-similarity retrieval engine
**Corpus:** 3,000 neural-network (G06N3) patents · 20,000-term vocabulary (1–2 grams)
**Goal:** given a patent (or a draft idea), surface the most similar existing patents and flag potential overlap.

## Methodology

- **Data processing:** combine `title` + `abstract`, lowercase, strip punctuation/whitespace; 83 near-duplicate filings detected in EDA.
- **Vectorisation:** `TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words="english", min_df=2, max_df=0.95)`.
- **Similarity:** cosine similarity between TF-IDF vectors; results ranked descending.
- **Evaluation ground truth (constructed — the model is unsupervised):**
  - **CPC codes** (dense, ~4 per patent) → ranking metrics.
  - **Citations** (sparse — only 0.2% point inside the corpus) → threshold audit + a "truly related" check.

---

## 1. Retrieval quality (vs CPC ground truth)

| Metric | Score | Read |
|---|---|---|
| Random baseline (chance two patents share a CPC) | 0.595 | reference point |
| Precision@1 | 0.731 | top hit shares a CPC 73% of the time |
| Precision@5 | 0.703 | |
| Precision@10 | 0.691 | |
| **Mean Reciprocal Rank** | **0.822** | first relevant hit is near the top |
| **NDCG@10** (graded by # shared CPC codes) | **0.761** | strong ranking — the honest headline |

*Note:* Precision@k looks high, but the 0.60 baseline is also high (narrow single-class corpus), so **NDCG and MRR** are the meaningful numbers — they reward ranking the *most* related patents highest, not just any CPC match.

Secondary check on the sparse citation signal (127 patents with in-corpus citations): Recall@10 = 0.232, MRR = 0.164.

## 2. Is the similarity score meaningful?

| Metric | Score |
|---|---|
| **ROC-AUC** — similarity separating true citation links from random pairs | **0.829** |

(0.5 = coin flip, 1.0 = perfect. 0.83 on a hard, heavily-imbalanced task means the score genuinely tracks relatedness — not noise.)

## 3. Novelty-threshold audit (key finding)

The engine flags "potential overlap" at a **hard-coded 0.30**. Audited against real citation links:

| Threshold | Precision | Recall | Verdict |
|---|---|---|---|
| **0.30 (current)** | **0.955** | **0.143** | almost never false-alarms, but **misses ~86% of truly related patents** |
| ~0.10 (data-optimal, best F1 = 0.454) | — | — | far better precision/recall balance |

**Recommendation:** lower the threshold to ≈0.10, or replace the yes/no verdict with a ranked list + tunable slider that exposes the precision/recall trade-off.

## 4. Bias & failure-mode audit

| Audit | Result | Verdict |
|---|---|---|
| Length bias | corr(abstract length, similarity) = 0.01 | ✅ none (TF-IDF is L2-normalized) |
| Temporal bias | mean top-1 similarity flat, 0.30–0.37 across 2015–2025 | ✅ minimal |
| Hubness | Gini = 0.38; one patent retrieved 133× | ⚠️ monitor |
| Duplicate leakage | 83 near-duplicate filings score ~1.00 vs 0.31 for the rest | ⚠️ de-duplicate before deploying |

---

## Limitations

- **Bag-of-words:** TF-IDF has no semantics — synonyms and paraphrases are missed; generic ML vocabulary ("neural network", "training data") inflates similarity between otherwise-unrelated patents.
- **Weak ground truth:** citation coverage is 0.2% and the CPC base rate is ~60%, so all metrics are **lower bounds**, not guarantees.
- **Duplicates:** near-identical filings score ~1.00 and crowd out genuinely different prior art until removed.
- **Scale:** the evaluation builds a full 3,000×3,000 similarity matrix — fine here, but would need approximate nearest-neighbour search for millions of patents.

## Bottom line

The engine is a **strong ranked screen** (ROC-AUC 0.83, NDCG@10 0.76) whose one tunable decision — the 0.30 threshold — was mis-set. Framed responsibly it's a **first-pass prior-art screen, never a legal novelty clearance.** Every claim above is backed by a metric, a baseline, and an audit.

*Figures for each section are in `figures/` (1 = retrieval precision, 2 = ROC, 3 = threshold audit, 4 = hubness). Full walkthrough with code: `notebooks/patent_similarity_engine.ipynb`.*
