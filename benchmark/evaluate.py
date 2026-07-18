"""Shared evaluation helpers: turn (user,item)->score maps into per-user
rankings and compute aggregate ranking metrics. Used by every ranker so the
comparison is apples-to-apples."""

from __future__ import annotations

import hashlib

import pandas as pd

from benchmark.metrics import evaluate_rankings


def _tiebreak(user_id: str, item_id: str) -> float:
    """Deterministic pseudo-random tiebreaker in [0,1) to avoid position bias on ties."""
    h = hashlib.md5(f"{user_id}\x00{item_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def scores_to_rankings(
    pool: pd.DataFrame,
    scores: dict[tuple[str, str], float],
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> dict[str, list[str]]:
    """Turn a flat (user,item)->score map into per-user item rankings (best first)."""
    rankings: dict[str, list[str]] = {}
    for user_id, grp in pool.groupby(user_col):
        items = grp[item_col].astype(str).tolist()
        items.sort(key=lambda it: (-scores.get((user_id, it), 0.0), -_tiebreak(user_id, it)))
        rankings[user_id] = items
    return rankings


def evaluate_scores(
    pool: pd.DataFrame,
    scores: dict[tuple[str, str], float],
    relevants: dict[str, set[str]],
    k_list: list[int],
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Rank the pool by score and compute aggregate metrics against ground truth."""
    rankings = scores_to_rankings(pool, scores, user_col, item_col)
    rel = {u: relevants[u] for u in rankings if u in relevants}
    aggregate, _ = evaluate_rankings(rankings, rel, k_list)
    return aggregate, rankings
