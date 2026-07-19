"""Build labeled (user, item, label) training examples for the feature rankers.

Positives are strong-positive training interactions; negatives are sampled items
the user never interacted with in the training window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark.data_prep import STRONG_POSITIVE_ACTIONS


def build_training_pairs(
    train_interactions: pd.DataFrame,
    items: pd.DataFrame,
    n_neg_per_pos: int = 4,
    seed: int = 42,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> pd.DataFrame:
    """Return a DataFrame with columns [user_id, item_id, label] (1=pos, 0=neg)."""
    rng = np.random.default_rng(seed)
    all_items = items[item_col].astype(str).unique()

    df = train_interactions.copy()
    df[user_col] = df[user_col].astype(str)
    df[item_col] = df[item_col].astype(str)

    pos = (
        df[df["action"].isin(STRONG_POSITIVE_ACTIONS)][[user_col, item_col]]
        .drop_duplicates()
    )
    seen = df.groupby(user_col)[item_col].agg(set).to_dict()

    rows = []
    for user_id, grp in pos.groupby(user_col):
        user_pos = list(grp[item_col])
        forbidden = seen.get(user_id, set())
        candidates = np.array([i for i in all_items if i not in forbidden])
        for it in user_pos:
            rows.append({user_col: user_id, item_col: it, "label": 1})
        n_neg = min(len(user_pos) * n_neg_per_pos, len(candidates))
        if n_neg > 0:
            negs = rng.choice(candidates, size=n_neg, replace=False)
            for it in negs:
                rows.append({user_col: user_id, item_col: str(it), "label": 0})

    return pd.DataFrame(rows)
