# PatentLens

PatentLens is a patent similarity search engine that uses **Natural Language Processing (NLP)** to identify existing patents most similar to a user's invention description. It focuses on **AI and Machine Learning patents (CPC G06N3)** and demonstrates how text retrieval can serve as a lightweight **prior-art screen**.

Describe an idea in free text and PatentLens ranks the most relevant existing patents, scoring each with a calibrated confidence badge. Four retrieval models are implemented and evaluated head-to-head against **real patent citations** as ground truth.

> **Scope:** a first-pass prior-art *screen*, never a legal novelty clearance.

---

## Results at a glance

Evaluated on a 100,000-patent corpus, 10,000 citation-linked query patents:

| Model | Recall@10 | NDCG@10 | MRR | Search latency |
|---|---|---|---|---|
| **MiniLM** | 0.1034 | 0.0814 | **0.1126** | 29 ms |
| BM25 | 0.1020 | 0.0809 | 0.1106 | 26 ms |
| TF-IDF | 0.0790 | 0.0660 | 0.0953 | 119 ms |
| LSA | 0.0629 | 0.0553 | 0.0822 | 118 ms |

MiniLM and BM25 are **statistically tied** on ranking quality (paired bootstrap, p = 0.21) — but only MiniLM holds up on citation pairs with little shared vocabulary (AUC 0.696 vs. BM25's 0.517, i.e. chance).

Full methodology, significance tests, bias audits, and every figure: **[RESULTS.md](RESULTS.md)**.

---

## Repository structure

```text
PatentLens/
├── app.py                  # Streamlit UI — free-text search over the corpus
├── pyproject.toml          # package metadata; makes `pip install -e .` work
├── requirements.txt        # direct dependencies
├── README.md
├── RESULTS.md              # evaluation findings, figures, limitations
│
├── data/
│   └── patents_g06n3_wide.csv   # 3,000-patent pilot corpus (committed)
│
├── src/patentlens/         # the library — all reusable logic lives here
│   ├── artifacts.py        # project paths + artifact loading, shared by every entry point
│   ├── cleaning.py         # text normalization, stopwords, feature derivation
│   ├── retrieval.py        # TF-IDF, BM25, LSA, embedding, and hybrid retrievers
│   ├── evaluation.py       # Recall/Precision/NDCG/MRR, bootstrap CIs, significance tests
│   └── data_fetch.py       # optional: pull a larger corpus from Google Patents BigQuery
│
├── scripts/                # entry points — thin orchestration over src/patentlens
│   ├── fetch_corpus.py     # optional: pull a larger corpus from BigQuery (costs quota)
│   ├── train.py            # fit + evaluate all four models, write models/
│   ├── citation_signal_test.py  # do cited pairs outscore random pairs?
│   └── product_metrics.py  # latency, thresholds, diversity, footprint (+ charts)
│
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
│
└── outputs/                # committed figures + metric snapshots used by RESULTS.md
```

Two directories are **generated, not committed**: `models/` (fitted models and evaluation caches, rebuilt by `scripts/train.py`) and `data/raw/` (large corpora fetched via `data_fetch.py`).

**Nothing here ships pre-trained.** A fresh clone has no `models/` directory, and the app refuses to start without one — you build it locally in the Quick start below.

---

## Setup

```bash
git clone https://github.com/SadbhavKattel/PatentLens.git
cd PatentLens

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optionally `pip install -e .` to import `patentlens` from anywhere. It isn't required — `app.py` and every script add `src/` to the path themselves, so a plain clone runs as-is.

---

## Quick start

No Google Cloud account needed. This runs on the 3,000-patent pilot corpus committed at
`data/patents_g06n3_wide.csv`.

```bash
python scripts/train.py            # fit + evaluate all four models
python scripts/product_metrics.py  # calibrate the result badges
streamlit run app.py
```

**Both scripts are required before the app is worth looking at:**

| Step | Writes to `models/` | What breaks without it |
|---|---|---|
| `train.py` | `patents.parquet`, `tfidf.joblib`, `bm25.joblib`, `lsa.joblib`, `minilm/` | The app shows "No trained artifacts found" and stops. |
| `product_metrics.py` | `product_threshold.csv`, `product_threshold_curve.csv` | The app runs, but **every badge reads `N/A`** — the confidence scoring is inert. |

`train.py` is **checkpointed**: each step is saved the moment it completes and skipped on
re-run, so an interrupted run resumes where it stopped. Delete a file under `models/` to
force that step to redo.

This path trains quickly, but it will **not** reproduce the numbers in RESULTS.md — those
come from a 100,000-patent corpus that isn't in this repo. See below.

### Also available

```bash
python scripts/citation_signal_test.py           # cited-vs-random pair separation (AUC)
python scripts/product_metrics.py --charts-only  # redraw figures without re-benchmarking
```

Each writes its CSVs to `models/` and its figures to `outputs/`.

---

## Reproducing the 100k-patent results

Everything in [RESULTS.md](RESULTS.md) was produced on 100,000 patents pulled from Google
Patents' public BigQuery dataset. That corpus is too large to commit and lands in
`data/raw/`, which is gitignored — so rebuilding it is on you, and it is **not** a single
command.

**1. Prerequisites**

- A Google Cloud project **with billing enabled**. Querying `patents-public-data` carries
  no license fee, but running a BigQuery job still consumes quota.
- Authenticate: `gcloud auth application-default login` (in Colab: `from google.colab import auth; auth.authenticate_user()`)
- Install the two optional dependencies — they are commented out in `requirements.txt` by
  default, since nothing else in the project needs them:

  ```bash
  pip install google-cloud-bigquery db-dtypes
  ```

**2. Check the cost before you spend it.** The query filters on country + CPC prefix, but
BigQuery still scans the title/abstract/cpc/citation columns of every US patent to
evaluate that filter — on the order of **a few hundred GB**, against a 1 TiB/month free
tier. `--row-cap` limits rows returned, not bytes scanned, so it won't reduce that.

```bash
python scripts/fetch_corpus.py --project-id your-gcp-project --estimate-only
```

**3. Fetch the corpus.**

```bash
python scripts/fetch_corpus.py --project-id your-gcp-project
```

Prints the same estimate, waits for confirmation, then writes
`data/raw/patents_g06n3_wide_100k.csv` — the exact filename
[`artifacts.py`](src/patentlens/artifacts.py) looks for, so `train.py` picks it up with
no further configuration. Pass `--yes` to skip the prompt, `--force` to overwrite an
existing file, or `--cpc-prefix` / `--country` / `--row-cap` to pull a different slice.

**4. Delete `models/` before retraining.** `train.py` automatically prefers the 100k CSV
over the pilot — but checkpointing keys off file existence, not corpus identity, so a
leftover `models/` from the pilot run gets reused silently and you'll get the old corpus
back. `fetch_corpus.py` warns you if it finds one. Clear it, then re-run the Quick start
commands.

Budget hours, not minutes: encoding 100k abstracts, plus an evaluation sweep that takes
~15-17 minutes per model for TF-IDF and LSA alone.

---

## How it works

**Retrieval** — four models behind one interface (`fit` / `rank` / `rank_text` / `score_pair` / `save` / `load`), so they are directly comparable and interchangeable in the app:

- **TF-IDF** — unigrams + bigrams, cosine similarity
- **BM25** — probabilistic ranking, reimplemented as a precomputed sparse matrix (~20× faster than the `rank_bm25` library at 100k documents)
- **LSA** — TF-IDF compressed via Truncated SVD
- **MiniLM** — sentence embeddings (`all-MiniLM-L6-v2`) indexed with FAISS

Similarity is always computed for **one query against the corpus**, never as a full n×n matrix — that is what lets the same code scale from 3,000 to 100,000+ patents.

**Evaluation** — real patent citations as ground truth. Every metric carries a 95% bootstrap confidence interval, and model-vs-model claims go through a paired bootstrap test rather than a bar-chart eyeball.

---

## Limitations

- Citation-based ground truth is a **proxy** — patents are cited for legal reasons, not purely textual similarity.
- Absolute scores are low (best MRR ≈ 0.11): the search space is large and no text-only model should be expected to approach perfect recall.
- The corpus is a single CPC subclass (G06N3) and US-only.

Discussed in full in [RESULTS.md](RESULTS.md).

---

## Project

Developed as part of the **AI4ALL Ignite Responsible AI Project**, demonstrating how NLP-based information retrieval can support transparent and accessible patent prior-art search.
