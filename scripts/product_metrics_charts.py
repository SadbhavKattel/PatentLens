"""Charts for the product metrics computed by scripts/product_metrics.py.
Run after that script. Saves to outputs/.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

sns.set_style("whitegrid")
PALETTE = {"TF-IDF": "#e76f51", "BM25": "#f4a261", "LSA": "#e9c46a", "MiniLM": "#2a9d8f"}


def model_colors(models):
    return [PALETTE.get(m, "#457b9d") for m in models]


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


if __name__ == "__main__":
    plot_latency()
    plot_threshold_curves()
    plot_diversity()
    plot_footprint()
    print("Saved product_latency.png, product_threshold_curve.png, product_diversity.png, product_footprint.png to outputs/")
