"""Checkpointed training pipeline for PatentLens.

Fits TF-IDF, BM25, LSA, and MiniLM on the patent corpus, evaluates each against real
citation ground truth (Recall/Precision/NDCG@k, MRR, bootstrap CIs, significance tests),
and saves everything to models/ for the Streamlit app.

Every step -- cleaning, each model fit, each model's evaluation -- is saved to disk as
soon as it's computed and skipped on the next run if already present. Safe to interrupt
and re-run: it picks up wherever it left off instead of starting over. Delete specific
files under models/ to force those steps to redo (e.g. `rm models/bm25.joblib` to refit
just BM25).

Figures go to the gitignored models/figures/ by default. Pass --publish-figures to write
them to outputs/ instead, which is the committed set RESULTS.md embeds -- only correct on
a full 100k-corpus run.

Run from the repo root: `python scripts/train.py`
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from patentlens import artifacts, cleaning, evaluation, retrieval  # noqa: E402
from patentlens.artifacts import EVAL_CACHE_DIR, MODELS_DIR, log  # noqa: E402

sns.set_style("whitegrid")

MAX_EVAL_QUERIES = 10000  # see RESULTS.md "Limitations" for why this is a sample, not all queries


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--publish-figures", action="store_true",
        help="write figures to the committed outputs/ instead of models/figures/. "
             "Only use this on a full 100k-corpus run -- outputs/ is what RESULTS.md embeds.",
    )
    args = parser.parse_args()
    publish = args.publish_figures

    MODELS_DIR.mkdir(exist_ok=True)
    EVAL_CACHE_DIR.mkdir(exist_ok=True)

    # ---- Step 1: data + cleaning ----
    cleaned_path = MODELS_DIR / "patents.parquet"
    if cleaned_path.exists():
        log("Loading already-cleaned patents.parquet...")
        df = pd.read_parquet(cleaned_path)
    else:
        raw_csv_path = artifacts.find_raw_csv()
        if raw_csv_path is None:
            raise FileNotFoundError(
                f"No raw patent CSV found. Expected one of: {artifacts.RAW_CSV_CANDIDATES}"
            )
        log(f"Loading {raw_csv_path.name}...")
        df = pd.read_csv(raw_csv_path)
        log(f"Loaded shape: {df.shape}")
        log("Cleaning text...")
        df = cleaning.prepare_dataframe(df)
        df.to_parquet(cleaned_path)
        log("Cleaning done, saved patents.parquet")

    # ---- Step 2: citation ground truth ----
    gt_path = EVAL_CACHE_DIR / "ground_truth.json"
    if gt_path.exists():
        log("Loading cached ground truth...")
        ground_truth = artifacts.load_ground_truth()
    else:
        log("Building citation ground truth...")
        ground_truth, _ = evaluation.build_ground_truth(df)
        with open(gt_path, "w") as f:
            json.dump({str(k): v for k, v in ground_truth.items()}, f)

    n_queries = len(ground_truth)
    n_pairs = sum(len(v) for v in ground_truth.values())
    log(f"Query patents with >=1 in-sample citation: {n_queries} / {len(df)} ({n_queries/len(df):.1%})")
    log(f"Total usable citation pairs: {n_pairs}")

    # ---- Step 3: fit retrievers ----
    tfidf_path = MODELS_DIR / "tfidf.joblib"
    if tfidf_path.exists():
        log("Loading cached TF-IDF...")
        tfidf = retrieval.TfidfRetriever.load(tfidf_path)
    else:
        log("Fitting TF-IDF...")
        tfidf = retrieval.TfidfRetriever().fit(df['clean_text'])
        tfidf.save(tfidf_path)
        log(f"TF-IDF matrix shape: {tfidf.matrix.shape} (saved)")

    bm25_path = MODELS_DIR / "bm25.joblib"
    if bm25_path.exists():
        log("Loading cached BM25...")
        bm25 = retrieval.Bm25Retriever.load(bm25_path)
    else:
        log("Fitting BM25...")
        bm25 = retrieval.Bm25Retriever().fit(df['clean_text'])
        bm25.save(bm25_path)
        log(f"BM25 fit on {bm25.term_counts.shape[0]} documents (saved)")

    lsa_path = MODELS_DIR / "lsa.joblib"
    if lsa_path.exists():
        log("Loading cached LSA...")
        lsa = retrieval.LsaRetriever.load(lsa_path, tfidf)
    else:
        log("Fitting LSA...")
        lsa = retrieval.LsaRetriever(tfidf, n_components=100).fit()
        lsa.save(lsa_path)
        log(f"LSA matrix shape: {lsa.lsa_matrix.shape}, "
            f"variance explained: {lsa.svd.explained_variance_ratio_.sum():.4f} (saved)")

    minilm_path = MODELS_DIR / "minilm"
    if minilm_path.exists():
        log("Loading cached MiniLM...")
        minilm = retrieval.EmbeddingRetriever.load(minilm_path)
    else:
        log("Fitting MiniLM embeddings...")
        minilm = retrieval.EmbeddingRetriever(model_name="all-MiniLM-L6-v2", name="MiniLM").fit(
            df['clean_text'], batch_size=128
        )
        minilm.save(minilm_path)
        log(f"MiniLM embeddings shape: {minilm.embeddings.shape} (saved)")

    retrievers = {"TF-IDF": tfidf, "BM25": bm25, "LSA": lsa, "MiniLM": minilm}

    # ---- Step 4: evaluate each model (cached per-model) ----
    k_values = (5, 10, 20)

    eval_queries_path = EVAL_CACHE_DIR / "eval_query_ids.json"
    if eval_queries_path.exists():
        with open(eval_queries_path) as f:
            eval_ids = set(json.load(f))
        ground_truth_eval = {k: v for k, v in ground_truth.items() if k in eval_ids}
        log(f"Loaded cached eval query sample: {len(ground_truth_eval)} queries")
    else:
        ground_truth_eval = evaluation.sample_ground_truth(ground_truth, max_queries=MAX_EVAL_QUERIES)
        with open(eval_queries_path, "w") as f:
            json.dump([int(k) for k in ground_truth_eval.keys()], f)
        log(f"Evaluating on {len(ground_truth_eval)} of {len(ground_truth)} ground-truth queries (sampled + cached)")

    all_summaries, all_per_query = {}, {}
    for name, retriever in retrievers.items():
        cache_path = EVAL_CACHE_DIR / f"{name.replace(' ', '_')}.joblib"
        if cache_path.exists():
            log(f"Loading cached evaluation for {name}...")
            cached = joblib.load(cache_path)
            all_summaries[name] = cached['summary']
            all_per_query[name] = cached['per_query']
        else:
            log(f"Evaluating {name}...")
            rank_fn = lambda q, k, r=retriever: r.rank(q, top_k=k)[0]
            summary, per_query = evaluation.evaluate_retriever(rank_fn, ground_truth_eval, k_values=k_values)
            all_summaries[name] = summary
            all_per_query[name] = per_query
            joblib.dump({'summary': summary, 'per_query': per_query}, cache_path)

        s = all_summaries[name]
        log(f"{name}: MRR={s['MRR']['mean']:.4f}  Recall@10={s['Recall@10']['mean']:.4f}  NDCG@10={s['NDCG@10']['mean']:.4f}")

    # ---- Step 5: summary table, plots, significance tests ----
    summary_df = evaluation.summary_to_dataframe(all_summaries)
    pivot = summary_df.pivot(index='model', columns='metric', values='mean')
    metric_order = [f'{m}@{k}' for m in ['Recall', 'Precision', 'NDCG'] for k in k_values] + ['MRR']
    pivot = pivot[[c for c in metric_order if c in pivot.columns]].round(4)
    log("\n" + str(pivot))

    _plot_metric_with_ci(summary_df, 'Recall', k_values, "Recall@k with 95% bootstrap CI")
    plt.savefig(artifacts.figure_path("recall_comparison.png", publish), dpi=150)

    _plot_metric_with_ci(summary_df, 'NDCG', k_values, "NDCG@k with 95% bootstrap CI (ranking quality)")
    plt.savefig(artifacts.figure_path("ndcg_comparison.png", publish), dpi=150)

    mrr_df = summary_df[summary_df['metric'] == 'MRR'].sort_values('mean', ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = sns.color_palette("Set2", len(mrr_df))
    ax.bar(mrr_df['model'], mrr_df['mean'],
           yerr=[mrr_df['mean'] - mrr_df['ci_low'], mrr_df['ci_high'] - mrr_df['mean']],
           capsize=4, color=colors)
    ax.set_ylabel("Mean Reciprocal Rank")
    ax.set_title("MRR by model (higher = true citation ranked closer to #1)", fontweight='bold')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(artifacts.figure_path("mrr_comparison.png", publish), dpi=150)

    log("Running significance tests...")
    best_model_name = mrr_df.iloc[0]['model']
    best_per_query_mrr = all_per_query[best_model_name]['MRR']

    sig_rows = []
    for name, per_query in all_per_query.items():
        if name == best_model_name:
            continue
        result = evaluation.paired_bootstrap_test(per_query['MRR'], best_per_query_mrr)
        sig_rows.append({
            'best_model': best_model_name,
            'compared_to': name,
            'mrr_mean_diff': round(result['mean_diff'], 4),
            'ci_low': round(result['ci_low'], 4),
            'ci_high': round(result['ci_high'], 4),
            'p_value': round(result['p_value'], 4),
            'significant_at_0.05': result['significant_at_0.05'],
        })
    significance_df = pd.DataFrame(sig_rows).sort_values('p_value')
    log(f"Best model by mean MRR: {best_model_name}")
    log("\n" + str(significance_df))

    summary_df.to_csv(MODELS_DIR / "metrics_summary.csv", index=False)
    pivot.to_csv(MODELS_DIR / "metrics_pivot.csv")
    significance_df.to_csv(MODELS_DIR / "significance_tests.csv", index=False)

    figures_dest = artifacts.OUTPUTS_DIR if publish else artifacts.FIGURES_DIR
    log(f"Saved figures to {figures_dest.relative_to(artifacts.PROJECT_ROOT)}/")
    log(f"Saved final artifacts to {MODELS_DIR}")
    for p in sorted(MODELS_DIR.rglob("*")):
        if p.is_file():
            print(" ", p.relative_to(MODELS_DIR))
    log("DONE")


def _plot_metric_with_ci(summary_df, metric_prefix, k_values, title):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    models = list(summary_df['model'].unique())
    colors = sns.color_palette("Set2", len(models))
    x = np.arange(len(k_values))
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        means, los, his = [], [], []
        for k in k_values:
            row = summary_df[(summary_df['model'] == model) & (summary_df['metric'] == f'{metric_prefix}@{k}')].iloc[0]
            means.append(row['mean'])
            los.append(row['mean'] - row['ci_low'])
            his.append(row['ci_high'] - row['mean'])
        ax.bar(x + i * width, means, width, label=model, color=colors[i], yerr=[los, his], capsize=3)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.set_ylabel(f"{metric_prefix}@k")
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    main()
