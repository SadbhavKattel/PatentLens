# PatentLens

A TF-IDF patent similarity engine for AI/ML patents, with a full evaluation of how well it actually works. *AI4ALL Ignite project.*

Given a patent — or a brand-new draft idea — PatentLens finds the most similar existing patents and flags potential overlap, acting as a **first-pass prior-art screen**.

## Data

- **`patents_g06n3_wide.csv`** — 3,000 patents in CPC class **G06N3** (neural networks / machine learning).
- Fields: `publication_number`, `filing_date`, `publication_date`, `title`, `abstract`, `cpc_codes`, `cited_patents`.

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
