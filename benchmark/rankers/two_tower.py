"""Two-tower neural ranker over the engineered features.

A user tower and an item tower each map their feature vector to a shared
embedding space; the score of a (user, item) pair is the dot product of the two
embeddings. Trained pointwise with BCE on positive vs sampled-negative pairs.

This is the classic retrieval/ranking factorization: only user features go into
the user tower and only item features into the item tower (cross/pair features
would break the factorization, so they are intentionally not used here — the
GBDT ranker uses them instead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark.rankers.features import FeatureBuilder
from benchmark.rankers.labels import build_training_pairs


class TwoTowerRanker:
    name = "two_tower"

    def __init__(self, items, users, train_interactions, engine_name="benchmark_two_tower",
                 emb_dim=32, hidden=64, epochs=15, lr=1e-3, batch_size=256,
                 n_neg_per_pos=4, seed=42, user_col="user_id", item_col="item_id"):
        self.items, self.users, self.train = items, users, train_interactions
        self.engine_name = engine_name
        self.emb_dim, self.hidden, self.epochs = emb_dim, hidden, epochs
        self.lr, self.batch_size, self.n_neg_per_pos = lr, batch_size, n_neg_per_pos
        self.seed, self.user_col, self.item_col = seed, user_col, item_col
        self.fb: FeatureBuilder | None = None

    def _build_towers(self, torch, in_dim: int):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(in_dim, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.emb_dim),
        )

    def fit(self) -> "TwoTowerRanker":
        import torch

        torch.manual_seed(self.seed)
        self.fb = FeatureBuilder(self.engine_name, self.items, self.users, self.train,
                                 self.user_col, self.item_col).build()

        pairs_df = build_training_pairs(self.train, self.items, self.n_neg_per_pos,
                                        self.seed, self.user_col, self.item_col)
        if pairs_df.empty:
            raise RuntimeError("No training pairs available for the two-tower ranker.")

        users = pairs_df[self.user_col].tolist()
        items = pairs_df[self.item_col].tolist()
        y = pairs_df["label"].to_numpy(dtype=np.float32)

        Xu = self.fb.user_matrix(users)
        Xi = self.fb.item_matrix(items)
        # Standardize (fit on training rows; reused at scoring time).
        self._u_mean, self._u_std = Xu.mean(0), Xu.std(0) + 1e-6
        self._i_mean, self._i_std = Xi.mean(0), Xi.std(0) + 1e-6
        Xu = (Xu - self._u_mean) / self._u_std
        Xi = (Xi - self._i_mean) / self._i_std

        self.user_tower = self._build_towers(torch, Xu.shape[1])
        self.item_tower = self._build_towers(torch, Xi.shape[1])
        params = list(self.user_tower.parameters()) + list(self.item_tower.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        Xu_t = torch.tensor(Xu); Xi_t = torch.tensor(Xi); y_t = torch.tensor(y)
        n = len(y)
        rng = np.random.default_rng(self.seed)
        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            total = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                ub = self.user_tower(Xu_t[idx]); ib = self.item_tower(Xi_t[idx])
                logits = (ub * ib).sum(dim=1)
                loss = loss_fn(logits, y_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss) * len(idx)
            # (silent; caller logs summary)
        self._torch = torch
        return self

    def score_pairs(self, pairs: list[tuple[str, str]]) -> dict[tuple[str, str], float]:
        torch = self._torch
        users = [str(u) for u, _ in pairs]
        items = [str(i) for _, i in pairs]
        Xu = (self.fb.user_matrix(users) - self._u_mean) / self._u_std
        Xi = (self.fb.item_matrix(items) - self._i_mean) / self._i_std
        with torch.no_grad():
            ub = self.user_tower(torch.tensor(Xu.astype(np.float32)))
            ib = self.item_tower(torch.tensor(Xi.astype(np.float32)))
            logits = (ub * ib).sum(dim=1).numpy()
        return {(u, i): float(s) for (u, i), s in zip(pairs, logits)}
