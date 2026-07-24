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


def sample_ground_truth(ground_truth: dict, max_queries=3000, seed=42):
    """Subsample ground-truth queries when there are more than max_queries.

    TF-IDF/BM25 have no sublinear index -- ranking one query means scanning the whole
    corpus, so evaluating one at a time in a Python loop is O(queries x corpus_size).
    That's fine at a few hundred queries but becomes impractical in the tens of thousands
    (e.g. ~45k ground-truth queries on a 100k-patent corpus took ~50 minutes for TF-IDF
    alone). A few thousand queries already gives far tighter confidence intervals than
    the original 127-query evaluation ever had, so capping here trades a small amount of
    additional precision for evaluation that finishes in minutes instead of hours.
    """
    if len(ground_truth) <= max_queries:
        return ground_truth
    rng = np.random.default_rng(seed)
    keys = np.array(list(ground_truth.keys()))
    sampled_keys = rng.choice(keys, size=max_queries, replace=False)
    return {k: ground_truth[k] for k in sampled_keys}


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


def citation_signal_test(score_fn, ground_truth, n_docs, token_sets=None,
                          max_pairs=15000, low_overlap_percentile=25, seed=42):
    """Do truly-cited patents score higher than random pairs -- even when their text
    barely overlaps?

    Recall/MRR/NDCG (evaluate_retriever) test *ranking*: does the true citation land near
    the top of a full-corpus search. This tests something more basic and arguably more
    convincing: for a real (citing, cited) pair, is score_fn(citing, cited) systematically
    higher than score_fn(citing, random_other_patent)? An AUC of 0.5 would mean the score
    is no better than chance at telling a true citation from a random pair; 1.0 would mean
    perfect separation.

    score_fn(idx_a, idx_b) -> float. Every retriever in retrieval.py exposes
    `.score_pair(idx_a, idx_b)` for this.

    token_sets: optional list/dict of idx -> set(tokens), e.g. from
    `df['clean_text'].str.split().apply(set)`. When given, also computes Jaccard word
    overlap for every pair and repeats the analysis restricted to the pairs with the
    LEAST text overlap (bottom `low_overlap_percentile`) -- i.e. citation pairs a pure
    keyword-matching model would have no way to find. If the model still separates those
    from random pairs, that's evidence it's capturing real relatedness beyond shared words.

    max_pairs caps how many (citing, cited) pairs get sampled -- this is an O(pairs) test,
    not O(queries x corpus) like evaluate_retriever, so it stays fast even sampled generously.
    """
    rng = np.random.default_rng(seed)

    all_pairs = [(q, c) for q, cited in ground_truth.items() for c in cited]
    if len(all_pairs) > max_pairs:
        idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        all_pairs = [all_pairs[i] for i in idx]

    cited_by_query = {q: set(cited) | {q} for q, cited in ground_truth.items()}

    pos_scores, neg_scores, jaccards = [], [], []
    for query_idx, cited_idx in all_pairs:
        pos_scores.append(score_fn(query_idx, cited_idx))

        exclude = cited_by_query[query_idx]
        while True:
            neg_idx = int(rng.integers(0, n_docs))
            if neg_idx not in exclude:
                break
        neg_scores.append(score_fn(query_idx, neg_idx))

        if token_sets is not None:
            a, b = token_sets[query_idx], token_sets[cited_idx]
            union = len(a | b)
            jaccards.append(len(a & b) / union if union else 0.0)

    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)

    result = {
        'n_pairs': len(pos_scores),
        'pos_mean': float(pos_scores.mean()),
        'neg_mean': float(neg_scores.mean()),
        'pos_median': float(np.median(pos_scores)),
        'neg_median': float(np.median(neg_scores)),
        'lift_pct': float((pos_scores.mean() - neg_scores.mean()) / (abs(neg_scores.mean()) + 1e-9) * 100),
        'auc': _roc_auc(pos_scores, neg_scores),
        'pos_scores': pos_scores,
        'neg_scores': neg_scores,
    }

    if token_sets is not None:
        jaccards = np.array(jaccards)
        threshold = np.percentile(jaccards, low_overlap_percentile)
        low_mask = jaccards <= threshold
        low_pos, low_neg = pos_scores[low_mask], neg_scores[low_mask]
        result['low_overlap_threshold'] = float(threshold)
        result['low_overlap_n_pairs'] = int(low_mask.sum())
        result['low_overlap_pos_mean'] = float(low_pos.mean()) if low_mask.sum() else 0.0
        result['low_overlap_neg_mean'] = float(low_neg.mean()) if low_mask.sum() else 0.0
        result['low_overlap_lift_pct'] = (
            float((low_pos.mean() - low_neg.mean()) / (abs(low_neg.mean()) + 1e-9) * 100)
            if low_mask.sum() else 0.0
        )
        result['low_overlap_auc'] = _roc_auc(low_pos, low_neg) if low_mask.sum() >= 10 else None

    return result


def _roc_auc(pos_scores, neg_scores):
    """AUC via the Mann-Whitney U statistic -- avoids an sklearn dependency for one metric.
    Equivalent to sklearn.metrics.roc_auc_score(y_true, y_score) for this pos/neg setup.
    """
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None
    combined = np.concatenate([pos_scores, neg_scores])
    ranks = pd.Series(combined).rank().values
    pos_rank_sum = ranks[:len(pos_scores)].sum()
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


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
