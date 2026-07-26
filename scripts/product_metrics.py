"""Product-relevant metrics beyond ranking quality: the questions someone deciding
whether to actually USE this tool would ask.

  - Latency: how long does one search take, per model? (real-time vs. batch feasibility)
  - Threshold precision/recall: at what similarity score should "potential overlap" be
    flagged, and what's the tradeoff at that cutoff? (same audit your collaborator ran
    for TF-IDF, extended to all 4 models)
  - Result diversity: are the top-10 results actually 10 different patents, or near-
    duplicates of each other? (redundant results look impressive but aren't useful)
  - Footprint: how much disk/memory does each model need? (deployment cost)

Run from the repo root, after scripts/train.py: `python scripts/product_metrics.py`
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from patentlens import evaluation, retrieval  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
EVAL_DIR = MODELS_DIR / "_eval_cache"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    curve per model, then reports the best-F1 operating point -- same audit style as the
    collaborator's TF-IDF threshold analysis, extended to every model here.
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


def main():
    log("Loading cached corpus, ground truth, and models...")
    df = pd.read_parquet(MODELS_DIR / "patents.parquet")
    with open(EVAL_DIR / "ground_truth.json") as f:
        ground_truth = {int(k): v for k, v in json.load(f).items()}

    tfidf = retrieval.TfidfRetriever.load(MODELS_DIR / "tfidf.joblib")
    bm25 = retrieval.Bm25Retriever.load(MODELS_DIR / "bm25.joblib")
    lsa = retrieval.LsaRetriever.load(MODELS_DIR / "lsa.joblib", tfidf)
    minilm = retrieval.EmbeddingRetriever.load(MODELS_DIR / "minilm")
    retrievers = {"TF-IDF": tfidf, "BM25": bm25, "LSA": lsa, "MiniLM": minilm}

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
    log("DONE")


if __name__ == "__main__":
    main()
