# Setup and running

Everything needed to get PatentLens running locally, plus the full procedure for
rebuilding the 100,000-patent corpus behind [RESULTS.md](RESULTS.md).

**Nothing in this repository ships pre-trained.** A fresh clone has no `models/`
directory, and the app refuses to start without one — you build it in [Quick
start](#quick-start) below. This is deliberate: fitted models and FAISS indexes are
large binary artifacts that don't belong in git.

---

## Install

```bash
git clone https://github.com/SadbhavKattel/PatentLens.git
cd PatentLens

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer. The install pulls PyTorch via `sentence-transformers`, so expect a
few minutes and a few hundred MB.

`pip install -e .` is optional — it lets you `import patentlens` from anywhere, but
`app.py` and every script add `src/` to the path themselves, so a plain clone runs as-is.

Dependencies are specified as **lower bounds**, not exact pins. The floors are the
versions that produced RESULTS.md; the pipeline has also been verified end-to-end on a
clean clone with the newer set pip resolves today. Pin exactly if you need a
byte-reproducible environment.

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

On the pilot corpus the whole sequence takes a couple of minutes. It will **not**
reproduce the numbers in RESULTS.md — those come from the 100k corpus, which isn't in
this repo. See [Reproducing the 100k-patent results](#reproducing-the-100k-patent-results).

### Also available

```bash
python scripts/citation_signal_test.py           # cited-vs-random pair separation (AUC)
python scripts/product_metrics.py --charts-only  # redraw figures without re-benchmarking
```

---

## How the pipeline behaves

Three things are worth knowing before you run anything twice.

**`train.py` is checkpointed.** Each step is saved the moment it completes and skipped on
re-run, so an interrupted run resumes where it stopped rather than starting over. Delete a
specific file under `models/` to force just that step to redo — e.g. `rm models/bm25.joblib`
to refit only BM25.

**Checkpointing keys off file existence, not corpus identity.** If you swap in a different
corpus without clearing `models/` first, the pipeline happily reuses artifacts built from
the *previous* dataset and reports results for the wrong data. Whenever the corpus changes,
`rm -rf models/` first. `fetch_corpus.py` warns you if it finds a stale one.

**Figures default to `models/figures/`, which is gitignored.** `outputs/` is a committed
snapshot of the 100k run that RESULTS.md embeds — not a scratch directory. Writing there
is opt-in:

```bash
python scripts/train.py --publish-figures   # also accepted by the other two scripts
```

Only use it after a full 100k-corpus run. Publishing pilot-corpus charts into `outputs/`
leaves RESULTS.md describing 100k results above figures showing 3k ones.

---

## Reproducing the 100k-patent results

Everything in [RESULTS.md](RESULTS.md) was produced on 100,000 patents pulled from Google
Patents' public BigQuery dataset. That corpus is too large to commit and lands in
`data/raw/`, which is gitignored — so rebuilding it is on you.

### 1. Prerequisites

- A Google Cloud project **with billing enabled**. Querying `patents-public-data` carries
  no license fee, but running a BigQuery job still consumes quota.
- Authenticate:

  ```bash
  gcloud auth application-default login
  # in Colab: from google.colab import auth; auth.authenticate_user()
  ```

- Install the two optional dependencies. They're commented out of `requirements.txt` by
  default, since nothing else in the project needs them:

  ```bash
  pip install google-cloud-bigquery db-dtypes
  ```

### 2. Check the cost before you spend it

The query filters on country + CPC prefix, but BigQuery still scans the
title/abstract/cpc/citation columns of **every US patent** to evaluate that filter — on
the order of a few hundred GB, against a 1 TiB/month free tier. `--row-cap` limits rows
returned, not bytes scanned, so it won't reduce that.

```bash
python scripts/fetch_corpus.py --project-id your-gcp-project --estimate-only
```

### 3. Fetch the corpus

```bash
python scripts/fetch_corpus.py --project-id your-gcp-project
```

Prints the same estimate, waits for confirmation, then writes
`data/raw/patents_g06n3_wide_100k.csv` — the exact filename
[`artifacts.py`](../src/patentlens/artifacts.py) looks for, so `train.py` picks it up with
no further configuration.

| Flag | Effect |
|---|---|
| `--estimate-only` | Print the cost estimate and exit without querying |
| `--yes` | Skip the confirmation prompt (non-interactive runs) |
| `--force` | Overwrite an existing destination CSV |
| `--cpc-prefix` / `--country` / `--row-cap` | Pull a different slice (defaults: `G06N3`, `US`, 100000) |

### 4. Retrain from scratch

```bash
rm -rf models/          # required — see "How the pipeline behaves" above
python scripts/train.py --publish-figures
python scripts/product_metrics.py --publish-figures
python scripts/citation_signal_test.py --publish-figures
```

Budget hours, not minutes: encoding 100k abstracts, plus an evaluation sweep that runs
~15-17 minutes per model for TF-IDF and LSA alone.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `No trained artifacts found in .../models` | `train.py` hasn't been run yet. |
| Every result badge reads `N/A` | `product_metrics.py` hasn't been run — it writes the threshold CSVs the badges are calibrated from. |
| Results don't match RESULTS.md | You're on the 3,000-patent pilot corpus. Expected — see [Reproducing](#reproducing-the-100k-patent-results). |
| Retrained on a new corpus, results unchanged | Stale `models/`. Run `rm -rf models/` and retrain. |
| `google-cloud-bigquery is not installed` | Expected — it's commented out of `requirements.txt`. `fetch_corpus.py` prints the install command. |
| Hugging Face rate-limit warning during training | Harmless. Set `HF_TOKEN` for faster downloads if it bothers you. |
