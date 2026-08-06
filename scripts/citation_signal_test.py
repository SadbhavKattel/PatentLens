"""Do truly-cited patents score higher than random pairs -- even when their text barely
overlaps? Runs evaluation.citation_signal_test() for every model against the cached
artifacts from scripts/train.py, on a sample of citation pairs (not the full ground
truth -- see MAX_PAIRS below).

Run from the repo root, after scripts/train.py has produced models/: `python scripts/citation_signal_test.py`
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from patentlens import evaluation, retrieval  # noqa: E402

sns.set_style("whitegrid")

MODELS_DIR = PROJECT_ROOT / "models"
EVAL_DIR = MODELS_DIR / "_eval_cache"
MAX_PAIRS = 15000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("Loading cached corpus + ground truth...")
    df = pd.read_parquet(MODELS_DIR / "patents.parquet")
    with open(EVAL_DIR / "ground_truth.json") as f:
        ground_truth = {int(k): v for k, v in json.load(f).items()}
    log(f"{len(ground_truth)} query patents, {sum(len(v) for v in ground_truth.values())} citation pairs available")

    log("Loading cached models...")
    tfidf = retrieval.TfidfRetriever.load(MODELS_DIR / "tfidf.joblib")
    bm25 = retrieval.Bm25Retriever.load(MODELS_DIR / "bm25.joblib")
    lsa = retrieval.LsaRetriever.load(MODELS_DIR / "lsa.joblib", tfidf)
    minilm = retrieval.EmbeddingRetriever.load(MODELS_DIR / "minilm")

    retrievers = {"TF-IDF": tfidf, "BM25": bm25, "LSA": lsa, "MiniLM": minilm}

    log("Building token sets for text-overlap control...")
    token_sets = df['clean_text'].str.split().apply(set).tolist()

    rows = []
    chart_data = {}
    for name, retriever in retrievers.items():
        log(f"Running citation-signal test for {name} ({MAX_PAIRS} sampled pairs)...")
        result = evaluation.citation_signal_test(
            retriever.score_pair, ground_truth, len(df),
            token_sets=token_sets, max_pairs=MAX_PAIRS,
        )
        chart_data[name] = result
        rows.append({
            'model': name,
            'n_pairs': result['n_pairs'],
            'cited_pair_mean_score': round(result['pos_mean'], 4),
            'random_pair_mean_score': round(result['neg_mean'], 4),
            'lift_pct': round(result['lift_pct'], 1),
            'auc': round(result['auc'], 4),
            'low_overlap_n_pairs': result['low_overlap_n_pairs'],
            'low_overlap_threshold_jaccard': round(result['low_overlap_threshold'], 4),
            'low_overlap_cited_mean_score': round(result['low_overlap_pos_mean'], 4),
            'low_overlap_random_mean_score': round(result['low_overlap_neg_mean'], 4),
            'low_overlap_lift_pct': round(result['low_overlap_lift_pct'], 1),
            'low_overlap_auc': round(result['low_overlap_auc'], 4) if result['low_overlap_auc'] else None,
        })
        log(f"{name}: AUC={result['auc']:.4f}  lift={result['lift_pct']:.1f}%  "
            f"low-text-overlap AUC={result['low_overlap_auc']:.4f}  low-overlap lift={result['low_overlap_lift_pct']:.1f}%")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(MODELS_DIR / "citation_signal_test.csv", index=False)
    log("\n" + str(results_df.to_string(index=False)))

    _plot_score_distributions(chart_data)
    plt.savefig(PROJECT_ROOT / "outputs" / "citation_signal_distributions.png", dpi=150)

    _plot_auc_bars(results_df)
    plt.savefig(PROJECT_ROOT / "outputs" / "citation_signal_auc.png", dpi=150)

    log("Saved models/citation_signal_test.csv and outputs/citation_signal_*.png")
    log("DONE")


def _plot_score_distributions(chart_data):
    models = list(chart_data.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4), sharey=False)
    if len(models) == 1:
        axes = [axes]
    for ax, name in zip(axes, models):
        r = chart_data[name]
        data = pd.DataFrame({
            'score': list(r['pos_scores']) + list(r['neg_scores']),
            'pair type': ['cited patent'] * len(r['pos_scores']) + ['random patent'] * len(r['neg_scores']),
        })
        sns.boxplot(data=data, x='pair type', y='score', ax=ax, hue='pair type',
                    palette={'cited patent': '#2a9d8f', 'random patent': '#e76f51'}, legend=False)
        ax.set_title(f"{name}\nAUC={r['auc']:.3f}", fontweight='bold')
        ax.set_xlabel('')
    fig.suptitle("Similarity score: truly-cited pairs vs. random pairs", fontweight='bold', y=1.03)
    plt.tight_layout()
    return fig


def _plot_auc_bars(results_df):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(results_df))
    ax.bar([i - 0.2 for i in x], results_df['auc'], width=0.4, label='All citation pairs', color='#2a9d8f')
    ax.bar([i + 0.2 for i in x], results_df['low_overlap_auc'].fillna(0), width=0.4,
           label='Low text-overlap pairs only', color='#e76f51')
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Chance (0.5)')
    ax.set_xticks(list(x))
    ax.set_xticklabels(results_df['model'])
    ax.set_ylabel("AUC (ability to separate cited from random pairs)")
    ax.set_title("Does the model catch citations even with little shared vocabulary?", fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    main()
