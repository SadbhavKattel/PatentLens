"""PatentLens: free-text patent similarity search.

Type an idea, pick a model in the sidebar, get back the most similar existing patents
as a notification-style list with a color-coded similarity badge.

Run with:
    streamlit run app.py

Expects artifacts produced by scripts/train.py in ./models/.
"""

import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from patentlens import artifacts, cleaning, retrieval  # noqa: E402
from patentlens.artifacts import MODELS_DIR  # noqa: E402

st.set_page_config(page_title="PatentLens", layout="centered")

st.markdown(
    """
    <style>
    div[data-testid="stAppViewContainer"], .stMarkdown, .stTextArea textarea,
    .result-title, .result-meta, .result-snippet {
        font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    h1 { font-weight: 600 !important; letter-spacing: -0.01em; }
    .result-card {
        display: flex;
        align-items: stretch;
        gap: 14px;
        background: #ffffff;
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 4px;
        padding: 13px 15px;
        margin-bottom: 10px;
    }
    .result-body { flex: 1; min-width: 0; }
    .result-title {
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 3px;
        line-height: 1.35;
        color: #14171f;
    }
    .result-title a { text-decoration: none; color: #14171f; }
    .result-title a:hover { text-decoration: underline; }
    .result-meta {
        font-family: ui-monospace, "SF Mono", "Cascadia Mono", Consolas, monospace;
        font-size: 0.72rem;
        color: #5b6472;
        margin-bottom: 6px;
    }
    .result-snippet { font-size: 0.83rem; color: #3a3f4b; line-height: 1.42; }
    .score-badge {
        flex-shrink: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-width: 64px;
        border-radius: 3px;
        padding: 6px 8px;
        font-family: ui-monospace, "SF Mono", "Cascadia Mono", Consolas, monospace;
    }
    .score-badge .score-val { font-size: 0.98rem; font-weight: 700; line-height: 1.1; }
    .score-badge .score-tag { font-size: 0.6rem; font-weight: 600; letter-spacing: 0.05em; margin-top: 3px; }
    .badge-green { background: #1a7f37; color: #ffffff; }
    .badge-yellow { background: #9a6700; color: #ffffff; }
    .badge-red { background: #b42318; color: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading models and patent corpus...")
def load_everything():
    df = artifacts.load_corpus()
    retrievers = artifacts.load_retrievers()

    # PatentSBERTa/Hybrid are optional -- not every training run includes them (the slow
    # step on large corpora), so only wire them in when their artifacts actually exist.
    patentsberta_dir = MODELS_DIR / "patentsberta"
    if patentsberta_dir.exists():
        patentsberta = retrieval.EmbeddingRetriever.load(patentsberta_dir)
        hybrid = retrieval.HybridRetriever(
            [retrievers["BM25"], patentsberta], weights=[1.0, 1.0],
            name="Hybrid (BM25 + PatentSBERTa)",
        )
        retrievers["PatentSBERTa"] = patentsberta
        retrievers[hybrid.name] = hybrid

    stop_words = cleaning.get_stopwords()
    thresholds = load_score_thresholds()
    return df, retrievers, stop_words, thresholds


def load_score_thresholds():
    """Per-model score cutoffs for the STRONG/MODERATE/WEAK badge, calibrated from
    scripts/product_metrics.py's precision-recall sweep against real patent citations --
    not an arbitrary cutoff, and not relative to whatever else is in the current result
    set. A given score means the same thing for a given model every time, regardless of
    what else the search returned (min-max-normalizing within one result set, the
    previous approach, could paint two nearly-identical scores as STRONG and WEAK just
    because they happened to be the top and bottom of a tightly-clustered batch).

    MODERATE starts at the model's best-F1 operating point (the threshold that best
    balances precision/recall for flagging "related"). STRONG starts where precision
    against real citations reaches ~90% -- i.e. results this model itself has shown are
    trustworthy, not a guess.
    """
    best_path = MODELS_DIR / "product_threshold.csv"
    curve_path = MODELS_DIR / "product_threshold_curve.csv"
    if not (best_path.exists() and curve_path.exists()):
        return {}

    best_df = pd.read_csv(best_path)
    curve_df = pd.read_csv(curve_path)
    thresholds = {}
    for _, row in best_df.iterrows():
        model = row["model"]
        moderate = float(row["best_threshold"])
        sub = curve_df[curve_df["model"] == model]
        high_precision = sub[sub["precision"] >= 0.9]
        strong = float(high_precision["threshold"].min()) if len(high_precision) else float(sub["threshold"].max())
        thresholds[model] = (moderate, strong)
    return thresholds


def google_patents_url(publication_number: str) -> str:
    return f"https://patents.google.com/patent/{publication_number.replace('-', '')}"


def _authors_label(row) -> str:
    inventors = row["inventors"] if isinstance(row["inventors"], list) else []
    if not inventors:
        return None
    label = ", ".join(inventors[:3])
    if len(inventors) > 3:
        label += f" +{len(inventors) - 3} more"
    return label


def _badge_class(score: float, calibration) -> tuple:
    if calibration is None:
        # No calibration data for this model (e.g. product_metrics.py hasn't been run,
        # or this is an optional model like PatentSBERTa it doesn't cover) -- show a
        # neutral badge rather than a color that would imply a confidence we can't back.
        return "badge-yellow", "N/A"
    moderate, strong = calibration
    if score >= strong:
        return "badge-green", "STRONG"
    if score >= moderate:
        return "badge-yellow", "MODERATE"
    return "badge-red", "WEAK"


def render_results(df, idxs, scores, calibration):
    for idx, score in zip(idxs, scores):
        row = df.loc[idx]
        badge_class, badge_tag = _badge_class(score, calibration)

        title = html.escape(row["title"])
        url = google_patents_url(row["publication_number"])
        authors = _authors_label(row)
        meta_parts = [html.escape(row["publication_number"]), f"filed {row['filing_date']}"]
        if authors:
            meta_parts.append(html.escape(authors))
        meta = " · ".join(meta_parts)

        abstract = row["abstract"]
        snippet = html.escape(abstract[:220] + ("..." if len(abstract) > 220 else ""))

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-body">
                    <div class="result-title"><a href="{url}" target="_blank">{title}</a></div>
                    <div class="result-meta">{meta}</div>
                    <div class="result-snippet">{snippet}</div>
                </div>
                <div class="score-badge {badge_class}">
                    <span class="score-val">{score:.3f}</span>
                    <span class="score-tag">{badge_tag}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    st.title("PatentLens")

    if not (MODELS_DIR / "patents.parquet").exists():
        st.error(
            f"No trained artifacts found in {MODELS_DIR}. Run "
            "scripts/train.py first -- it saves everything this app needs."
        )
        return

    df, retrievers, stop_words, thresholds = load_everything()

    with st.sidebar:
        st.subheader("Settings")
        # Default to MiniLM rather than whichever model happens to come first in the
        # loading order. It ties BM25 on ranking metrics but is the only model that
        # still separates real citations from random pairs when the two patents share
        # little vocabulary (AUC 0.696 vs BM25's 0.517) -- see docs/RESULTS.md. Falling
        # back to index 0 keeps this working if MiniLM isn't in a given models/ build.
        model_names = list(retrievers.keys())
        default_model = model_names.index("MiniLM") if "MiniLM" in model_names else 0
        model_name = st.selectbox("Model", model_names, index=default_model)
        top_k = st.slider("Number of results", min_value=3, max_value=20, value=8)
        st.caption(f"Corpus: {len(df):,} patents")

    idea = st.text_area(
        "Write your idea here",
        placeholder="e.g. a neural network that learns to control robotic arms using reinforcement learning...",
        height=130,
        label_visibility="visible",
    )
    search = st.button("Search", type="primary")

    if not search:
        return
    if not idea.strip():
        st.warning("Describe an idea first.")
        return

    retriever = retrievers[model_name]
    cleaned = cleaning.clean_text(idea, stop_words)
    idxs, scores = retriever.rank_text(cleaned, top_k=top_k)

    if not idxs:
        st.info("No similar patents found.")
        return

    render_results(df, idxs, scores, thresholds.get(model_name))


if __name__ == "__main__":
    main()
