# PatentLens

**Patent similarity search for AI/ML patents, evaluated against real citation data.**

Describe an invention in plain English and PatentLens ranks the most similar existing
patents, scoring each with a calibrated confidence badge. Four retrieval models —
TF-IDF, BM25, LSA, and MiniLM embeddings — are implemented behind one interface and
evaluated head-to-head against **real patent citations** as ground truth, over a corpus
of 100,000 US patents in CPC subclass **G06N3** (neural network architectures).

> **Scope:** a first-pass prior-art *screen*, never a legal novelty clearance.

📹 **[Watch the demo](https://drive.google.com/file/d/13LsTh2QYd1Lc040RNTuhv92l28DCclQo/view?usp=sharing)** · 📊 **[Full evaluation report](docs/RESULTS.md)** · ⚙️ **[Setup and running](docs/SETUP.md)**

---

## Results at a glance

Evaluated on a 100,000-patent corpus, 10,000 citation-linked query patents:

| Model | Recall@10 | NDCG@10 | MRR | Search latency |
|---|---|---|---|---|
| **MiniLM** | 0.1034 | 0.0814 | **0.1126** | 29 ms |
| BM25 | 0.1020 | 0.0809 | 0.1106 | 26 ms |
| TF-IDF | 0.0790 | 0.0660 | 0.0953 | 119 ms |
| LSA | 0.0629 | 0.0553 | 0.0822 | 118 ms |

**MiniLM and BM25 are statistically tied** on ranking quality (paired bootstrap,
p = 0.21) — the headline metric can't separate them. But on citation pairs with little
shared vocabulary, BM25 collapses to chance (AUC 0.517) while MiniLM holds at 0.696.
Semantic retrieval earns its keep exactly where keyword matching structurally cannot: when
two related patents don't reuse each other's words.

Methodology, significance tests, product metrics, and every figure:
**[docs/RESULTS.md](docs/RESULTS.md)**.

---

## Quick start

```bash
git clone https://github.com/SadbhavKattel/PatentLens.git
cd PatentLens
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/train.py            # fit + evaluate all four models
python scripts/product_metrics.py  # calibrate the result badges
streamlit run app.py
```

Runs on the 3,000-patent pilot corpus committed to the repo — no cloud account needed,
a couple of minutes end to end. **Nothing here ships pre-trained**: both scripts are
required before the app works, and results on the pilot corpus won't match the 100k
numbers above.

Full instructions, the BigQuery procedure for rebuilding the 100k corpus, and
troubleshooting: **[docs/SETUP.md](docs/SETUP.md)**.

---

## How it works

**Retrieval** — four models behind one interface (`fit` / `rank` / `rank_text` /
`score_pair` / `save` / `load`), so they're directly comparable and interchangeable in
the app:

| Model | Approach |
|---|---|
| **TF-IDF** | Unigrams + bigrams, cosine similarity |
| **BM25** | Probabilistic ranking, reimplemented as a precomputed sparse matrix (~20× faster than `rank_bm25` at 100k documents) |
| **LSA** | TF-IDF compressed via Truncated SVD |
| **MiniLM** | Sentence embeddings (`all-MiniLM-L6-v2`) indexed with FAISS |

Similarity is always computed for **one query against the corpus**, never as a full n×n
matrix — that's what lets the same code scale from 3,000 to 100,000+ patents.

**Evaluation** — real patent citations as ground truth. Every metric carries a 95%
bootstrap confidence interval, and model-vs-model claims go through a paired bootstrap
test rather than a bar-chart eyeball.

**Confidence badges** — the STRONG / MODERATE / WEAK label on each result comes from a
per-model precision-recall sweep against real citations, not an arbitrary cutoff. A given
score means the same thing every time, regardless of what else the search returned.

---

## Repository structure

```text
PatentLens/
├── app.py                  # Streamlit UI — free-text search over the corpus
├── requirements.txt        # direct dependencies (lower bounds, not pins)
├── pyproject.toml          # package metadata; makes `pip install -e .` work
│
├── docs/
│   ├── SETUP.md            # install, running, BigQuery corpus, troubleshooting
│   └── RESULTS.md          # evaluation findings, figures, limitations
│
├── src/patentlens/         # the library — all reusable logic lives here
│   ├── artifacts.py        # project paths + artifact loading, shared by every entry point
│   ├── cleaning.py         # text normalization, stopwords, feature derivation
│   ├── retrieval.py        # TF-IDF, BM25, LSA, embedding, and hybrid retrievers
│   ├── evaluation.py       # Recall/Precision/NDCG/MRR, bootstrap CIs, significance tests
│   └── data_fetch.py       # the Google Patents BigQuery query
│
├── scripts/                # entry points — thin orchestration over src/patentlens
│   ├── fetch_corpus.py     # optional: pull a larger corpus from BigQuery (costs quota)
│   ├── train.py            # fit + evaluate all four models, write models/
│   ├── citation_signal_test.py  # do cited pairs outscore random pairs?
│   └── product_metrics.py  # latency, thresholds, diversity, footprint (+ charts)
│
├── data/
│   └── patents_g06n3_wide.csv   # 3,000-patent pilot corpus (committed)
│
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
│
└── outputs/                # committed 100k-run figures, embedded by docs/RESULTS.md
```

Two directories are **generated, not committed**: `models/` (fitted models, evaluation
caches, and your own runs' figures under `models/figures/`) and `data/raw/` (large
corpora fetched from BigQuery).

`outputs/` is a curated snapshot of the 100k run, not a scratch directory — scripts write
figures to `models/figures/` and only touch `outputs/` when passed `--publish-figures`.

---

## Limitations

- Citation-based ground truth is a **proxy** — patents are cited for legal reasons, not
  purely textual similarity.
- Absolute scores are low (best MRR ≈ 0.11): the search space is large, and no text-only
  model should be expected to approach perfect recall.
- The corpus is a single CPC subclass (G06N3) and US-only.
- PatentSBERTa and a BM25+PatentSBERTa hybrid were evaluated on the pilot corpus but not
  at 100k scale — a time-budget tradeoff, not a finding.

Discussed in full in [docs/RESULTS.md](docs/RESULTS.md).

---

## Participants

Built by a three-person team for the AI4ALL Ignite program:

- **Mutawakil Rabiu** — [@Muta4ever](https://github.com/Muta4ever)
- **Sadbhav Kattel** — [@SadbhavKattel](https://github.com/SadbhavKattel)
- **Seemya Momin** — [@seemyamomin](https://github.com/seemyamomin)

---

## Project

Developed as part of the **AI4ALL Ignite Responsible AI Project**, demonstrating how
NLP-based information retrieval can support transparent and accessible patent prior-art
search.

<!-- TODO: confirm licensing with AI4ALL before adding a LICENSE file -->
**License:** not yet determined — please contact the contributors before reusing this
code.
