"""Product-relevant metrics beyond ranking quality: the questions someone deciding
whether to actually USE this tool would ask.

  - Latency: how long does one search take, per model? (real-time vs. batch feasibility)
  - Threshold precision/recall: at what similarity score should "potential overlap" be
    flagged, and what's the tradeoff at that cutoff?
  - Result diversity: are the top-10 results actually 10 different patents, or near-
    duplicates of each other? (redundant results look impressive but aren't useful)
  - Footprint: how much disk/memory does each model need? (deployment cost)

Writes five CSVs to models/ and the four figures RESULTS.md embeds to outputs/.
The app also reads models/product_threshold{,_curve}.csv from here to calibrate its
STRONG/MODERATE/WEAK badges.

Run from the repo root, after scripts/train.py:

    python scripts/product_metrics.py                 # compute metrics, then draw charts
    python scripts/product_metrics.py --charts-only   # redraw charts from existing CSVs
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from patentlens import artifacts, evaluation  # noqa: E402
from patentlens.artifacts import MODELS_DIR, OUTPUTS_DIR, log  # noqa: E402

sns.set_style("whitegrid")
PALETTE = {"TF-IDF": "#e76f51", "BM25": "#f4a261", "LSA": "#e9c46a", "MiniLM": "#2a9d8f"}


def model_colors(models):
    return [PALETTE.get(m, "#457b9d") for m in models]


# --------------------------------------------------------------------------- metrics

def benchmark_latency(retrievers, sample_texts, n_queries=100):
    rows = []
    for name, retriever in retrievers.items():
        texts = sample_texts[:n_queries]
        # warm-up call (first call can include lazy model loading)
        retriever.rank_text(texts[0], top_k=10)

        times = []
        for text in texts:
            t0 = time.perf_counter()
            retriever.rank_text(text, top_k=10)
            times.append((time.perf_counter() - t0) * 1000)
        times = np.array(times)
        rows.append({
            'model': name,
            'mean_ms': round(times.mean(), 2),
            'median_ms': round(np.median(times), 2),
            'p95_ms': round(np.percentile(times, 95), 2),
            'n_queries': len(times),
        })
        log(f"{name}: mean {times.mean():.1f}ms, median {np.median(times):.1f}ms, p95 {np.percentile(times,95):.1f}ms")
    return pd.DataFrame(rows)


def threshold_sweep(retrievers, ground_truth, n_docs, max_pairs=8000, n_thresholds=60):
    """Reuses the citation-signal-test pos/neg pair sampling to build a precision-recall
    curve per model, then reports the best-F1 operating point.
    """
    rows = []
    curve_rows = []
    for name, retriever in retrievers.items():
        log(f"Threshold sweep for {name}...")
        result = evaluation.citation_signal_test(
            retriever.score_pair, ground_truth, n_docs, max_pairs=max_pairs
        )
        pos, neg = result['pos_scores'], result['neg_scores']
        all_scores = np.concatenate([pos, neg])
        thresholds = np.percentile(all_scores, np.linspace(1, 99, n_thresholds))

        best_f1, best_t, best_p, best_r = -1, None, None, None
        for t in thresholds:
            tp = (pos >= t).sum()
            fp = (neg >= t).sum()
            fn = (pos < t).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            curve_rows.append({'model': name, 'threshold': t, 'precision': precision, 'recall': recall, 'f1': f1})
            if f1 > best_f1:
                best_f1, best_t, best_p, best_r = f1, t, precision, recall

        rows.append({
            'model': name,
            'best_threshold': round(float(best_t), 4),
            'precision_at_best': round(float(best_p), 4),
            'recall_at_best': round(float(best_r), 4),
            'f1_at_best': round(float(best_f1), 4),
        })
        log(f"{name}: best threshold={best_t:.3f}  precision={best_p:.3f}  recall={best_r:.3f}  F1={best_f1:.3f}")

    return pd.DataFrame(rows), pd.DataFrame(curve_rows)


def result_diversity(retrievers, df, sample_size=200, top_k=10, seed=42):
    rng = np.random.default_rng(seed)
    sample_idxs = rng.choice(len(df), size=sample_size, replace=False)

    rows = []
    for name, retriever in retrievers.items():
        log(f"Diversity check for {name}...")
        avg_pairwise_sims = []
        for q in sample_idxs:
            idxs, _ = retriever.rank(int(q), top_k=top_k)
            if len(idxs) < 2:
                continue
            pairwise = [
                retriever.score_pair(idxs[i], idxs[j])
                for i in range(len(idxs)) for j in range(i + 1, len(idxs))
            ]
            avg_pairwise_sims.append(np.mean(pairwise))
        avg_pairwise_sims = np.array(avg_pairwise_sims)
        rows.append({
            'model': name,
            'avg_pairwise_similarity_in_top10': round(float(avg_pairwise_sims.mean()), 4),
        })
        log(f"{name}: avg pairwise similarity among top-{top_k} results = {avg_pairwise_sims.mean():.4f}")
    return pd.DataFrame(rows)


def model_footprint():
    rows = []
    specs = [
        ("TF-IDF", MODELS_DIR / "tfidf.joblib", None),
        ("BM25", MODELS_DIR / "bm25.joblib", None),
        ("LSA", MODELS_DIR / "lsa.joblib", 100),
        ("MiniLM", MODELS_DIR / "minilm", 384),
    ]
    for name, path, dims in specs:
        if path.is_dir():
            size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
        elif path.exists():
            size_mb = path.stat().st_size / 1e6
        else:
            continue
        rows.append({'model': name, 'artifact_size_mb': round(size_mb, 1), 'embedding_dims': dims})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- charts

def plot_latency():
    df = pd.read_csv(MODELS_DIR / "product_latency.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(df['model'], df['mean_ms'], color=model_colors(df['model']),
                   yerr=df['p95_ms'] - df['mean_ms'], capsize=5)
    for b, v in zip(bars, df['mean_ms']):
        ax.annotate(f'{v:.0f}ms', (b.get_x() + b.get_width() / 2, b.get_height() + 2),
                    ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel("Mean time per search (ms)\n(error bar = p95)")
    ax.set_title("How long does one search take?", fontweight='bold', fontsize=13)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "product_latency.png", dpi=150)
    plt.close(fig)


def plot_threshold_curves():
    curve_df = pd.read_csv(MODELS_DIR / "product_threshold_curve.csv")
    best_df = pd.read_csv(MODELS_DIR / "product_threshold.csv")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for model in curve_df['model'].unique():
        sub = curve_df[curve_df['model'] == model].sort_values('recall')
        ax.plot(sub['recall'], sub['precision'], label=model, color=PALETTE.get(model, "#457b9d"), linewidth=2)
        best = best_df[best_df['model'] == model].iloc[0]
        ax.scatter([best['recall_at_best']], [best['precision_at_best']],
                   color=PALETTE.get(model, "#457b9d"), s=70, zorder=5, edgecolor='white', linewidth=1.2)

    ax.set_xlabel("Recall (share of truly related patents caught)")
    ax.set_ylabel("Precision (share of flagged patents that are truly related)")
    ax.set_title("Precision vs. recall for flagging \"potential overlap\"\n(dot = best-F1 operating point)",
                 fontweight='bold', fontsize=13)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc='lower left', fontsize=9)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "product_threshold_curve.png", dpi=150)
    plt.close(fig)


def plot_diversity():
    df = pd.read_csv(MODELS_DIR / "product_diversity.csv")
    cosine_models = df[df['model'] != 'BM25']
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(cosine_models['model'], cosine_models['avg_pairwise_similarity_in_top10'],
                   color=model_colors(cosine_models['model']))
    for b, v in zip(bars, cosine_models['avg_pairwise_similarity_in_top10']):
        ax.annotate(f'{v:.2f}', (b.get_x() + b.get_width() / 2, b.get_height() + 0.015),
                    ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel("Avg. similarity between results\nwithin the same top-10 list")
    ax.set_title("Are the top-10 results actually different patents,\nor near-duplicates of each other?",
                 fontweight='bold', fontsize=12.5)
    ax.set_ylim(0, 1.0)
    ax.text(0.98, 0.03, "BM25 omitted: its raw score isn't on a\n0-1 scale, so it's not directly comparable here.",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8, style='italic', color='#555')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "product_diversity.png", dpi=150)
    plt.close(fig)


def plot_footprint():
    df = pd.read_csv(MODELS_DIR / "product_footprint.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(df['model'], df['artifact_size_mb'], color=model_colors(df['model']))
    for b, v in zip(bars, df['artifact_size_mb']):
        ax.annotate(f'{v:.0f} MB', (b.get_x() + b.get_width() / 2, b.get_height() + 4),
                    ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel("Disk footprint (MB)")
    ax.set_title("Storage cost per model\n(100,000-patent corpus)", fontweight='bold', fontsize=13)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "product_footprint.png", dpi=150)
    plt.close(fig)


def draw_all_charts():
    plot_latency()
    plot_threshold_curves()
    plot_diversity()
    plot_footprint()
    log("Saved product_latency.png, product_threshold_curve.png, product_diversity.png, "
        "product_footprint.png to outputs/")


# ------------------------------------------------------------------------------ main

def compute_all_metrics():
    log("Loading cached corpus, ground truth, and models...")
    df = artifacts.load_corpus()
    ground_truth = artifacts.load_ground_truth()
    retrievers = artifacts.load_retrievers()

    rng = np.random.default_rng(7)
    sample_texts = df['clean_text'].iloc[rng.choice(len(df), size=150, replace=False)].tolist()

    log("--- Latency ---")
    latency_df = benchmark_latency(retrievers, sample_texts)
    latency_df.to_csv(MODELS_DIR / "product_latency.csv", index=False)

    log("--- Threshold precision/recall ---")
    threshold_df, curve_df = threshold_sweep(retrievers, ground_truth, len(df))
    threshold_df.to_csv(MODELS_DIR / "product_threshold.csv", index=False)
    curve_df.to_csv(MODELS_DIR / "product_threshold_curve.csv", index=False)

    log("--- Result diversity ---")
    diversity_df = result_diversity(retrievers, df)
    diversity_df.to_csv(MODELS_DIR / "product_diversity.csv", index=False)

    log("--- Footprint ---")
    footprint_df = model_footprint()
    footprint_df.to_csv(MODELS_DIR / "product_footprint.csv", index=False)
    log("\n" + str(footprint_df))

    log("Saved product_latency.csv, product_threshold.csv, product_threshold_curve.csv, "
        "product_diversity.csv, product_footprint.csv to models/")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--charts-only", action="store_true",
        help="skip the benchmarks and redraw the figures from the CSVs already in models/",
    )
    args = parser.parse_args()

    if not args.charts_only:
        compute_all_metrics()

    log("--- Charts ---")
    draw_all_charts()
    log("DONE")


if __name__ == "__main__":
    main()
