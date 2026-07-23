"""PatentLens demo: patent similarity search across TF-IDF, BM25, LSA, MiniLM,
PatentSBERTa, and a BM25+PatentSBERTa hybrid, plus a model-comparison dashboard.

Run with:
    streamlit run app.py

Expects artifacts produced by notebooks/02_model_training.ipynb in ./models/.
"""

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from patentlens import cleaning, retrieval  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(page_title="PatentLens", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="Loading models and patent corpus...")
def load_everything():
    df = pd.read_parquet(MODELS_DIR / "patents.parquet")

    tfidf = retrieval.TfidfRetriever.load(MODELS_DIR / "tfidf.joblib")
    bm25 = retrieval.Bm25Retriever.load(MODELS_DIR / "bm25.joblib")
    lsa = retrieval.LsaRetriever.load(MODELS_DIR / "lsa.joblib", tfidf)
    minilm = retrieval.EmbeddingRetriever.load(MODELS_DIR / "minilm")
    patentsberta = retrieval.EmbeddingRetriever.load(MODELS_DIR / "patentsberta")
    hybrid = retrieval.HybridRetriever(
        [bm25, patentsberta], weights=[1.0, 1.0], name="Hybrid (BM25 + PatentSBERTa)"
    )

    retrievers = {
        "TF-IDF": tfidf,
        "BM25": bm25,
        "LSA": lsa,
        "MiniLM": minilm,
        "PatentSBERTa": patentsberta,
        hybrid.name: hybrid,
    }

    metrics_summary = pd.read_csv(MODELS_DIR / "metrics_summary.csv")
    metrics_pivot = pd.read_csv(MODELS_DIR / "metrics_pivot.csv", index_col=0)
    stop_words = cleaning.get_stopwords()

    return df, retrievers, metrics_summary, metrics_pivot, stop_words


def google_patents_url(publication_number: str) -> str:
    return f"https://patents.google.com/patent/{publication_number.replace('-', '')}"


def render_result_card(row, score):
    with st.container(border=True):
        cols = st.columns([5, 1])
        with cols[0]:
            st.markdown(f"**{row['title']}**")
            st.caption(f"{row['publication_number']} · filed {row['filing_date']}")
        with cols[1]:
            st.metric("Score", f"{score:.3f}")
        abstract = row["abstract"]
        st.write(abstract[:400] + ("..." if len(abstract) > 400 else ""))
        cpc = row["cpc_codes"]
        if isinstance(cpc, (list,)) and cpc:
            st.caption("CPC: " + ", ".join(cpc[:6]))
        st.markdown(f"[View on Google Patents]({google_patents_url(row['publication_number'])})")


def search_tab(df, retrievers, stop_words):
    left, right = st.columns([1, 2])
    with left:
        model_name = st.selectbox("Model", list(retrievers.keys()), index=len(retrievers) - 1)
        top_k = st.slider("Number of results", min_value=3, max_value=20, value=5)
        query_mode = st.radio("Query type", ["Pick an existing patent", "Free-text description"])

    retriever = retrievers[model_name]

    with left:
        if query_mode == "Pick an existing patent":
            search_text = st.text_input("Filter by title keyword")
            options = df
            if search_text:
                options = df[df["title"].str.contains(search_text, case=False, na=False)]
            options = options.head(200)
            if options.empty:
                st.warning("No patents match that filter.")
                return
            labels = options.apply(
                lambda r: f"{r['publication_number']} — {r['title'][:70]}", axis=1
            )
            choice = st.selectbox("Patent", labels)
            query_idx = options.index[list(labels).index(choice)]
            run = st.button("Find similar patents", type="primary")
        else:
            free_text = st.text_area(
                "Describe an invention",
                placeholder="e.g. a neural network that learns to control robotic arms using reinforcement learning...",
                height=140,
            )
            run = st.button("Search", type="primary")

    with right:
        if not run:
            st.info("Configure a query on the left, then run the search.")
            return

        if query_mode == "Pick an existing patent":
            st.subheader("Query patent")
            q_row = df.loc[query_idx]
            st.markdown(f"**{q_row['title']}**")
            st.caption(q_row["publication_number"])
            st.write(q_row["abstract"][:400])
            st.divider()
            idxs, scores = retriever.rank(query_idx, top_k=top_k)
        else:
            if not free_text.strip():
                st.warning("Enter a description first.")
                return
            cleaned = cleaning.clean_text(free_text, stop_words)
            idxs, scores = retriever.rank_text(cleaned, top_k=top_k)

        st.subheader(f"Top {len(idxs)} similar patents — {model_name}")
        for idx, score in zip(idxs, scores):
            render_result_card(df.loc[idx], score)


def comparison_tab(metrics_summary, metrics_pivot):
    st.subheader("Retrieval quality vs. real patent citations")
    st.caption(
        "Ground truth: for each patent, whichever of its cited patents also fall inside "
        "the sampled corpus. Recall@k = fraction of true citations found in the top k "
        "results. Precision@k = fraction of the top k that are true citations. MRR = how "
        "close the first true citation lands to rank 1. NDCG@k additionally rewards ranking "
        "true citations higher, not just including them."
    )

    st.dataframe(metrics_pivot, use_container_width=True)

    metric_family = st.selectbox("Metric", ["Recall", "Precision", "NDCG", "MRR"])
    if metric_family == "MRR":
        chart_df = metrics_summary[metrics_summary["metric"] == "MRR"]
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("model:N", sort="-y", title=None),
                y=alt.Y("mean:Q", title="MRR"),
                tooltip=["model", "mean", "ci_low", "ci_high"],
                color=alt.Color("model:N", legend=None),
            )
        )
        error_bars = (
            alt.Chart(chart_df)
            .mark_errorbar()
            .encode(x="model:N", y="ci_low:Q", y2="ci_high:Q")
        )
        st.altair_chart(chart + error_bars, use_container_width=True)
    else:
        chart_df = metrics_summary[metrics_summary["metric"].str.startswith(metric_family + "@")]
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("metric:N", title=None, sort=None),
                y=alt.Y("mean:Q", title=f"{metric_family}@k"),
                color=alt.Color("model:N"),
                xOffset="model:N",
                tooltip=["model", "metric", "mean", "ci_low", "ci_high"],
            )
        )
        st.altair_chart(chart, use_container_width=True)

    st.caption(
        "Error bars/tooltips show 95% bootstrap confidence intervals — with a few hundred "
        "citation pairs, point estimates alone can be misleading."
    )


def main():
    st.title("🔍 PatentLens")
    st.caption("Patent similarity search across lexical, latent, and semantic retrieval models.")

    if not (MODELS_DIR / "patents.parquet").exists():
        st.error(
            f"No trained artifacts found in {MODELS_DIR}. Run "
            "notebooks/02_model_training.ipynb first — its last cell saves everything this "
            "app needs."
        )
        return

    df, retrievers, metrics_summary, metrics_pivot, stop_words = load_everything()
    st.caption(f"Corpus: {len(df):,} patents")

    tab1, tab2 = st.tabs(["Search", "Model comparison"])
    with tab1:
        search_tab(df, retrievers, stop_words)
    with tab2:
        comparison_tab(metrics_summary, metrics_pivot)


if __name__ == "__main__":
    main()
