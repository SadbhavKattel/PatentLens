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

## Usage

### 1. Train the models

```bash
python scripts/train.py
```

Fits TF-IDF, BM25, LSA, and MiniLM, evaluates each against citation ground truth, and writes everything to `models/`. Uses `data/patents_g06n3_wide.csv` unless a larger corpus is present at `data/raw/patents_g06n3_wide_100k.csv`.

The pipeline is **checkpointed** — every step is saved as soon as it completes and skipped on re-run, so an interrupted run resumes where it stopped. Delete a file under `models/` to force that step to redo.

### 2. Launch the app

```bash
streamlit run app.py
```

Requires `models/` from step 1. Pick a model in the sidebar, describe an idea, and get ranked patents with **STRONG / MODERATE / WEAK** badges. Those thresholds are calibrated per-model from the precision-recall sweep in `scripts/product_metrics.py` — run it to activate them, or badges show `N/A`.

### 3. Reproduce the evaluation

```bash
python scripts/citation_signal_test.py         # cited-vs-random pair separation (AUC)
python scripts/product_metrics.py              # latency, thresholds, diversity, footprint
python scripts/product_metrics.py --charts-only  # redraw figures without re-benchmarking
```

Each writes its CSVs to `models/` and its figures to `outputs/`.

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
