"""Ranking metrics for offline recommender evaluation.

Every metric takes a single user's *ranked* candidate list (best first) together
with the set of relevant (ground-truth positive) item ids, and returns a scalar.
`evaluate_rankings` aggregates them (macro-average over users) at several cut-offs.

A "ranking" here is a list of item_ids ordered by descending predicted score.
Relevance is binary: an item is relevant if it is in `relevant`.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked[:k]
    if not top:
        return 0.0
    hits = sum(1 for item in top if item in relevant)
    return hits / float(k)


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = ranked[:k]
    hits = sum(1 for item in top if item in relevant)
    return hits / float(len(relevant))


def hit_rate_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if at least one relevant item appears in the top-k, else 0.0."""
    return 1.0 if any(item in relevant for item in ranked[:k]) else 0.0


def average_precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """AP@k — precision averaged over the ranks of relevant items in the top-k.

    Averaging over min(|relevant|, k) matches the standard MAP@k convention.
    """
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0
    for idx, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / float(idx)
    denom = min(len(relevant), k)
    return score / float(denom) if denom else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant item (0 if none present)."""
    for idx, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / idx
    return 0.0


def dcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for idx, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(idx + 1)
    return dcg


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg_at_k(ranked, relevant, k) / idcg


def evaluate_user(ranked: Sequence[str], relevant: set[str], k_list: Iterable[int]) -> dict[str, float]:
    """All metrics for one user's ranked list at each cut-off in k_list."""
    out: dict[str, float] = {"mrr": reciprocal_rank(ranked, relevant)}
    for k in k_list:
        out[f"precision@{k}"] = precision_at_k(ranked, relevant, k)
        out[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        out[f"hit@{k}"] = hit_rate_at_k(ranked, relevant, k)
        out[f"map@{k}"] = average_precision_at_k(ranked, relevant, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant, k)
    return out


def evaluate_rankings(
    rankings: dict[str, list[str]],
    relevants: dict[str, set[str]],
    k_list: Iterable[int],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Macro-average metrics over all users.

    Args:
        rankings: user_id -> item_ids ordered best-first.
        relevants: user_id -> set of relevant item_ids.
        k_list: ranking cut-offs.

    Returns:
        (aggregate_metrics, per_user_metrics). Only users with >=1 relevant item
        are scored (a user with no ground truth is not evaluable).
    """
    k_list = list(k_list)
    per_user: dict[str, dict[str, float]] = {}
    for user_id, ranked in rankings.items():
        relevant = relevants.get(user_id, set())
        if not relevant:
            continue
        per_user[user_id] = evaluate_user(ranked, relevant, k_list)

    if not per_user:
        return {}, {}

    metric_names = next(iter(per_user.values())).keys()
    aggregate = {
        name: sum(u[name] for u in per_user.values()) / len(per_user)
        for name in metric_names
    }
    aggregate["n_users_evaluated"] = float(len(per_user))
    return aggregate, per_user
