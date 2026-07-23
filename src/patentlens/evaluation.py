"""Retrieval-quality evaluation against real patent citations.

Upgrades over the original notebook's Recall@k-only approach:
  - Precision@k, MRR, and NDCG@k in addition to Recall@k (ranking quality, not just coverage)
  - Bootstrap confidence intervals on every metric, since the citation ground truth is a
    small sample (a few hundred query patents even in a large corpus) and point estimates
    alone are misleading.
"""

import numpy as np
import pandas as pd


def build_ground_truth(df: pd.DataFrame):
    """Map each patent's row index to the row indices of patents it cites *within the sample*.

    Only citations that land inside the sampled corpus are usable as ground truth, so
    coverage naturally improves as the corpus grows.
    """
    pub_to_idx = {pub: idx for idx, pub in enumerate(df['publication_number'])}
    sample_ids = set(pub_to_idx)

    ground_truth = {}
    for idx, row in df.iterrows():
        cited_in_sample = [
            pub_to_idx[c] for c in row['cited_patents']
            if c in sample_ids and pub_to_idx[c] != idx
        ]
        if cited_in_sample:
            ground_truth[idx] = cited_in_sample
    return ground_truth, pub_to_idx


def recall_at_k(ranked, true_set, k):
    top_k = set(ranked[:k])
    return len(top_k & true_set) / len(true_set)


def precision_at_k(ranked, true_set, k):
    top_k = set(ranked[:k])
    return len(top_k & true_set) / k


def reciprocal_rank(ranked, true_set):
    for pos, idx in enumerate(ranked, start=1):
        if idx in true_set:
            return 1.0 / pos
    return 0.0


def ndcg_at_k(ranked, true_set, k):
    dcg = sum(
        1.0 / np.log2(pos + 2) for pos, idx in enumerate(ranked[:k]) if idx in true_set
    )
    ideal_hits = min(len(true_set), k)
    idcg = sum(1.0 / np.log2(pos + 2) for pos in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=42):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        boot_means[b] = rng.choice(values, size=n, replace=True).mean()
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boot_means, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)


def evaluate_retriever(rank_fn, ground_truth, k_values=(5, 10, 20), max_rank_pool=50):
    """Evaluate a retriever against citation ground truth.

    rank_fn(query_idx, top_k) -> list[int] of document indices ranked best-first,
    excluding the query itself. Every retriever class in `retrieval.py` exposes
    `.rank(query_idx, top_k=...)` returning `(indices, scores)`, so pass e.g.
    `lambda q, k: retriever.rank(q, top_k=k)[0]`.

    Only pulls the top `max_rank_pool` per query (largest k_value used), never a full
    n x n matrix, so this stays cheap even as the corpus grows into the tens of thousands.
    """
    max_k = max(max(k_values), max_rank_pool)
    per_query = {f'Recall@{k}': [] for k in k_values}
    per_query.update({f'Precision@{k}': [] for k in k_values})
    per_query.update({f'NDCG@{k}': [] for k in k_values})
    per_query['MRR'] = []

    for query_idx, true_list in ground_truth.items():
        true_set = set(true_list)
        ranked = rank_fn(query_idx, max_k)

        per_query['MRR'].append(reciprocal_rank(ranked, true_set))
        for k in k_values:
            per_query[f'Recall@{k}'].append(recall_at_k(ranked, true_set, k))
            per_query[f'Precision@{k}'].append(precision_at_k(ranked, true_set, k))
            per_query[f'NDCG@{k}'].append(ndcg_at_k(ranked, true_set, k))

    summary = {}
    for metric, values in per_query.items():
        lo, hi = bootstrap_ci(values)
        summary[metric] = {
            'mean': float(np.mean(values)) if values else 0.0,
            'ci_low': lo,
            'ci_high': hi,
            'n_queries': len(values),
        }
    return summary, per_query


def paired_bootstrap_test(values_a, values_b, n_boot=5000, seed=42):
    """Paired bootstrap significance test for "does model B actually beat model A?"

    values_a/values_b must be the SAME metric evaluated on the SAME queries in the same
    order (e.g. two entries of the per_query dict returned by evaluate_retriever for the
    same metric key) -- that pairing is what makes this more powerful than comparing two
    independent confidence intervals by eye.

    Resamples queries (not points) with replacement, recomputes the mean difference each
    time, and reports a two-sided p-value for H0: mean(B) == mean(A). With citation ground
    truth this small, "looks better" isn't the same as "is significantly better" -- this
    is the check for the latter.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired arrays must come from the same queries, in the same order")

    n = len(a)
    diffs = b - a
    observed_diff = diffs.mean()

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        boot_diffs[i] = diffs[sample_idx].mean()

    if observed_diff >= 0:
        p_value = min(1.0, 2 * np.mean(boot_diffs <= 0))
    else:
        p_value = min(1.0, 2 * np.mean(boot_diffs >= 0))

    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    return {
        'mean_diff': float(observed_diff),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'p_value': float(p_value),
        'significant_at_0.05': bool(p_value < 0.05),
        'n_queries': n,
    }


def summary_to_dataframe(all_summaries: dict):
    """all_summaries: {model_name: summary_dict_from_evaluate_retriever}."""
    rows = []
    for model_name, summary in all_summaries.items():
        for metric, stats in summary.items():
            rows.append({
                'model': model_name,
                'metric': metric,
                'mean': stats['mean'],
                'ci_low': stats['ci_low'],
                'ci_high': stats['ci_high'],
                'n_queries': stats['n_queries'],
            })
    return pd.DataFrame(rows)
