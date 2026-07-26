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

PROJECT_ROOT = Path(__file__).parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from patentlens import cleaning, retrieval  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"

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
    df = pd.read_parquet(MODELS_DIR / "patents.parquet")

    tfidf = retrieval.TfidfRetriever.load(MODELS_DIR / "tfidf.joblib")
    bm25 = retrieval.Bm25Retriever.load(MODELS_DIR / "bm25.joblib")
    lsa = retrieval.LsaRetriever.load(MODELS_DIR / "lsa.joblib", tfidf)
    minilm = retrieval.EmbeddingRetriever.load(MODELS_DIR / "minilm")

    retrievers = {
        "TF-IDF": tfidf,
        "BM25": bm25,
        "LSA": lsa,
        "MiniLM": minilm,
    }

    # PatentSBERTa/Hybrid are optional -- not every training run includes them (the slow
    # step on large corpora), so only wire them in when their artifacts actually exist.
    patentsberta_dir = MODELS_DIR / "patentsberta"
    if patentsberta_dir.exists():
        patentsberta = retrieval.EmbeddingRetriever.load(patentsberta_dir)
        hybrid = retrieval.HybridRetriever(
            [bm25, patentsberta], weights=[1.0, 1.0], name="Hybrid (BM25 + PatentSBERTa)"
        )
        retrievers["PatentSBERTa"] = patentsberta
        retrievers[hybrid.name] = hybrid

    stop_words = cleaning.get_stopwords()
    return df, retrievers, stop_words


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


def _badge_class(normalized_score: float) -> tuple:
    if normalized_score >= 0.66:
        return "badge-green", "STRONG"
    if normalized_score >= 0.33:
        return "badge-yellow", "MODERATE"
    return "badge-red", "WEAK"


def _normalize_scores(scores):
    # Similarity scales aren't comparable across models (BM25 is unbounded, cosine models
    # are ~0-1), so color-coding is relative to THIS result set, not an absolute cutoff --
    # it answers "which of these top-k are the strongest matches," not "is 0.4 good."
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def render_results(df, idxs, scores):
    normalized = _normalize_scores(scores)
    for idx, score, norm in zip(idxs, scores, normalized):
        row = df.loc[idx]
        badge_class, badge_tag = _badge_class(norm)

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

    df, retrievers, stop_words = load_everything()

    with st.sidebar:
        st.subheader("Settings")
        model_name = st.selectbox("Model", list(retrievers.keys()))
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

    render_results(df, idxs, scores)


if __name__ == "__main__":
    main()
