"""Classic (non-LLM) baseline recommenders.

All baselines are trained only on the sampled users' *training* interactions and
expose ``score_pairs`` so the orchestrator can score the exact same candidate pool
that the LLM engine scores. Three baselines are provided:

* ``PopularityRecommender`` — user-agnostic global popularity (weak reference).
* ``ItemKNNRecommender`` — item-based collaborative filtering (primary baseline).
* ``RandomRecommender`` — seeded random scores (sanity floor).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from benchmark.data_prep import ACTION_WEIGHTS


def _positive_weight(action: str) -> float:
    """Non-negative interaction weight (negatives contribute no positive signal)."""
    return max(0.0, ACTION_WEIGHTS.get(str(action).strip().lower(), 0.0))


class BaseRecommender(ABC):
    name: str = "base"

    def __init__(self, user_col: str = "user_id", item_col: str = "item_id"):
        self.user_col = user_col
        self.item_col = item_col

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> "BaseRecommender":
        ...

    @abstractmethod
    def score_pairs(self, pairs: list[tuple[str, str]]) -> dict[tuple[str, str], float]:
        ...


class PopularityRecommender(BaseRecommender):
    """Score(item) = summed positive action weight across the training set."""

    name = "popularity"

    def fit(self, train: pd.DataFrame) -> "PopularityRecommender":
        w = train["action"].map(_positive_weight)
        self.item_pop = (
            pd.Series(w.to_numpy(), index=train[self.item_col].astype(str))
            .groupby(level=0).sum()
        )
        self._max = float(self.item_pop.max()) if len(self.item_pop) else 1.0
        return self

    def score_pairs(self, pairs: list[tuple[str, str]]) -> dict[tuple[str, str], float]:
        denom = self._max or 1.0
        return {
            (u, i): float(self.item_pop.get(str(i), 0.0)) / denom
            for (u, i) in pairs
        }


class ItemKNNRecommender(BaseRecommender):
    """Item-based collaborative filtering with cosine item-item similarity.

    Builds a (users x items) weighted interaction matrix from the training data,
    computes item-item cosine similarity, and scores a (user, item) pair as the
    sum of similarities between the candidate item and the user's training items,
    weighted by the user's interaction strength. Falls back to popularity for
    users/items absent from the training matrix.
    """

    name = "item_knn"

    def __init__(self, user_col: str = "user_id", item_col: str = "item_id", topk_neighbors: int = 50):
        super().__init__(user_col, item_col)
        self.topk_neighbors = topk_neighbors

    def fit(self, train: pd.DataFrame) -> "ItemKNNRecommender":
        df = train.copy()
        df[self.user_col] = df[self.user_col].astype(str)
        df[self.item_col] = df[self.item_col].astype(str)
        df["w"] = df["action"].map(_positive_weight)
        # Collapse repeat (user,item) signals to their max weight.
        df = df.groupby([self.user_col, self.item_col], as_index=False)["w"].max()

        self.users_index = {u: k for k, u in enumerate(df[self.user_col].unique())}
        self.items_index = {i: k for k, i in enumerate(df[self.item_col].unique())}
        self.items_inv = {k: i for i, k in self.items_index.items()}

        rows = df[self.user_col].map(self.users_index).to_numpy()
        cols = df[self.item_col].map(self.items_index).to_numpy()
        vals = df["w"].to_numpy(dtype=float)
        n_users, n_items = len(self.users_index), len(self.items_index)
        self.ui = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

        # Item-item cosine similarity (dense: n_items is small, ~500).
        self.sim = cosine_similarity(self.ui.T)
        np.fill_diagonal(self.sim, 0.0)  # exclude self-similarity
        if self.topk_neighbors and self.topk_neighbors < n_items:
            # Keep only the top-k neighbours per item for a cleaner, faster signal.
            for j in range(n_items):
                row = self.sim[j]
                if row.size > self.topk_neighbors:
                    cutoff_idx = np.argpartition(row, -self.topk_neighbors)[: -self.topk_neighbors]
                    row[cutoff_idx] = 0.0

        # Popularity fallback for cold users/items.
        self._pop = PopularityRecommender(self.user_col, self.item_col).fit(train)
        return self

    def _user_vector(self, user_id: str) -> np.ndarray | None:
        idx = self.users_index.get(str(user_id))
        if idx is None:
            return None
        return self.ui.getrow(idx).toarray().ravel()

    def score_pairs(self, pairs: list[tuple[str, str]]) -> dict[tuple[str, str], float]:
        scores: dict[tuple[str, str], float] = {}
        # Cache the item->user-affinity contribution per user.
        user_profiles: dict[str, np.ndarray | None] = {}
        for (u, i) in pairs:
            u, i = str(u), str(i)
            item_idx = self.items_index.get(i)
            if item_idx is None:
                scores[(u, i)] = self._pop.score_pairs([(u, i)])[(u, i)] * 0.0
                continue
            if u not in user_profiles:
                user_profiles[u] = self._user_vector(u)
            uvec = user_profiles[u]
            if uvec is None:
                scores[(u, i)] = 0.0
                continue
            scores[(u, i)] = float(self.sim[item_idx].dot(uvec))
        # Normalise to [0,1] for readability (rank-preserving).
        if scores:
            mx = max(scores.values())
            if mx > 0:
                scores = {k: v / mx for k, v in scores.items()}
        return scores


class RandomRecommender(BaseRecommender):
    """Seeded random scores — a sanity floor for the metrics."""

    name = "random"

    def __init__(self, user_col: str = "user_id", item_col: str = "item_id", seed: int = 42):
        super().__init__(user_col, item_col)
        self.rng = np.random.default_rng(seed)

    def fit(self, train: pd.DataFrame) -> "RandomRecommender":
        return self

    def score_pairs(self, pairs: list[tuple[str, str]]) -> dict[tuple[str, str], float]:
        return {(u, i): float(self.rng.random()) for (u, i) in pairs}


BASELINE_REGISTRY = {
    PopularityRecommender.name: PopularityRecommender,
    ItemKNNRecommender.name: ItemKNNRecommender,
    RandomRecommender.name: RandomRecommender,
}
