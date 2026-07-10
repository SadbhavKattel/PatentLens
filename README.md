# PatentLens

A TF-IDF patent similarity engine for AI/ML patents, with a full evaluation of how well it actually works. *AI4ALL Ignite project.*

Given a patent — or a brand-new draft idea — PatentLens finds the most similar existing patents and flags potential overlap, acting as a **first-pass prior-art screen**.

## Data

- **`patents_g06n3_wide.csv`** — 3,000 patents in CPC class **G06N3** (neural networks / machine learning).
- Fields: `publication_number`, `filing_date`, `publication_date`, `title`, `abstract`, `cpc_codes`, `cited_patents`.

## Approach

1. **Data processing** — combine `title` + `abstract`, lowercase, strip punctuation and extra whitespace. EDA surfaced 83 near-duplicate filings.
2. **Vectorisation** — `TfidfVectorizer` with a 20,000-term vocabulary, unigrams + bigrams (so phrases like *"neural network"* count), English stop-words removed, `min_df=2`, `max_df=0.95`.
3. **Similarity** — cosine similarity between TF-IDF vectors; neighbours ranked descending.
4. **Two entry points** — find patents similar to one already in the corpus, or check a new title/abstract for overlap against a tunable novelty threshold.

## Evaluation

The engine is **unsupervised**, so there is no label to score against. Ground truth is *constructed* from the data:

- **CPC codes** (dense) → ranking metrics.
- **Citations** (sparse — only 0.2% land inside the corpus) → threshold audit and a "truly related" check.

| Aspect | Result |
|---|---|
| Ranking quality | **NDCG@10 = 0.76**, **MRR = 0.82** (vs 0.60 random baseline) |
| Score meaningfulness | **ROC-AUC = 0.83** separating true citation links |
| Novelty threshold (0.30) | precision 0.96 but **recall 0.14** — misses ~86% of related patents → lower to ≈0.10 |
| Bias audit | no length/temporal bias; mild hubness (Gini 0.38); 83 duplicates score ~1.00 |

Full metrics and figures: [`outputs/week8/metrics.md`](outputs/week8/metrics.md) and [`outputs/week8/figures/`](outputs/week8/figures/).

## Limitations

- **Bag-of-words:** TF-IDF has no semantics — synonyms/paraphrases are missed, and generic ML vocabulary inflates similarity between unrelated patents.
- **Weak ground truth:** low citation coverage and a high CPC base rate mean metrics are lower bounds.
- **Duplicates** must be removed before deployment; **scale** would require approximate nearest-neighbour search beyond a few thousand patents.
- It is a screening aid, **not** a legal novelty clearance.

## Repo layout

```
notebooks/
  01_eda.ipynb                     # exploratory data analysis
  patent_similarity_engine.ipynb   # the engine + full evaluation
outputs/
  eda_overview.png
  week8/
    metrics.md                     # evaluation write-up
    figures/                       # evaluation charts
patents_g06n3_wide.csv             # dataset
requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
```

Open `notebooks/patent_similarity_engine.ipynb` and run top to bottom. (Sections 1–4 build the engine; 5–8 evaluate and audit it.)
