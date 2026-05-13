"""
Ranking evaluation metrics.

NDCG (Normalized Discounted Cumulative Gain) is the primary metric for this system.
It measures whether known-good places appear near the top of the ranked output.
Position matters: rank 1 is worth more than rank 5, which is worth more than rank 10.
"""
import math


def dcg_at_k(ranked_names: list[str], relevant: set[str], k: int) -> float:
    """Discounted Cumulative Gain at k.

    Args:
        ranked_names: Ordered list of place names returned by the pipeline.
        relevant: Set of place names known to be correct answers (ground truth).
        k: Cutoff rank.

    Returns:
        DCG score. Higher is better; maximum is IDCG (see ndcg_at_k).
    """
    gain = 0.0
    for i, name in enumerate(ranked_names[:k], start=1):
        if name in relevant:
            gain += 1.0 / math.log2(i + 1)
    return gain


def ndcg_at_k(ranked_names: list[str], relevant: set[str], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Normalizes DCG by the ideal DCG (IDCG) — the score achieved if all relevant
    items appear at the very top of the ranking. Result is in [0, 1].

    A score of 1.0 means every relevant item appears in the top k positions.
    A score of 0.0 means no relevant item appears in the top k positions.

    Args:
        ranked_names: Ordered list of place names returned by the pipeline.
        relevant: Set of place names known to be correct answers (ground truth).
        k: Cutoff rank. Use 5 for this system (sla primary_eval_metric: ndcg@5).

    Returns:
        NDCG@k in [0.0, 1.0].
    """
    actual_dcg = dcg_at_k(ranked_names, relevant, k)

    # Ideal DCG: pretend all relevant items are in positions 1..min(|relevant|, k)
    n_relevant_in_top_k = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_relevant_in_top_k + 1))

    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg
