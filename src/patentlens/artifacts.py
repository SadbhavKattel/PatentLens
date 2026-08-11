"""Project paths and artifact loading, shared by the scripts and the Streamlit app.

`scripts/train.py` fits every model and writes it under `models/`; the other scripts
and `app.py` read those artifacts back. Centralizing the paths and the load block here
keeps each entry point from carrying its own copy of the same constants and the same
four-model loading sequence.

Layout:
    data/      corpus CSVs (the 3k pilot is committed; larger fetched corpora go in
               data/raw/ and are gitignored)
    models/    fitted models + evaluation caches, written by scripts/train.py (gitignored)
    outputs/   figures and metric CSVs used by RESULTS.md (committed)

`PROJECT_ROOT` resolves from this file's location, which holds for a source checkout
and for an editable install (`pip install -e .`).
"""

import json
import time
from pathlib import Path

import pandas as pd

from . import retrieval

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVAL_CACHE_DIR = MODELS_DIR / "_eval_cache"
FIGURES_DIR = MODELS_DIR / "figures"

# Prefer a larger fetched corpus when one is present; otherwise use the 3,000-patent
# pilot CSV committed to the repo.
RAW_CSV_CANDIDATES = [
    DATA_DIR / "raw" / "patents_g06n3_wide_100k.csv",
    DATA_DIR / "patents_g06n3_wide.csv",
]


def log(msg):
    """Timestamped progress line. The scripts are long-running, so they stream output."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_raw_csv():
    """First existing entry in RAW_CSV_CANDIDATES, or None."""
    return next((p for p in RAW_CSV_CANDIDATES if p.exists()), None)


def figure_path(filename, publish=False):
    """Where a script should write a generated figure.

    `outputs/` is a committed snapshot of the 100,000-patent run, and RESULTS.md embeds
    those exact files. Writing there by default meant any local run -- which on a fresh
    clone is the 3,000-patent pilot corpus -- silently replaced published figures with
    charts from a different dataset, leaving RESULTS.md's prose describing 100k results
    above charts showing 3k ones.

    So figures default to the gitignored `models/figures/`, and overwriting the committed
    set is opt-in via each script's `--publish-figures` flag.
    """
    directory = OUTPUTS_DIR if publish else FIGURES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def load_corpus():
    """The cleaned corpus written by scripts/train.py."""
    return pd.read_parquet(MODELS_DIR / "patents.parquet")


def load_ground_truth():
    """Citation ground truth cached by scripts/train.py, as {query_idx: [cited_idx, ...]}."""
    with open(EVAL_CACHE_DIR / "ground_truth.json") as f:
        return {int(k): v for k, v in json.load(f).items()}


def load_retrievers():
    """The four models scripts/train.py fits, loaded from models/."""
    tfidf = retrieval.TfidfRetriever.load(MODELS_DIR / "tfidf.joblib")
    bm25 = retrieval.Bm25Retriever.load(MODELS_DIR / "bm25.joblib")
    lsa = retrieval.LsaRetriever.load(MODELS_DIR / "lsa.joblib", tfidf)
    minilm = retrieval.EmbeddingRetriever.load(MODELS_DIR / "minilm")
    return {"TF-IDF": tfidf, "BM25": bm25, "LSA": lsa, "MiniLM": minilm}
