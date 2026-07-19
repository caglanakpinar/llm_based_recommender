"""Gradient-boosted decision-tree ranker over the engineered features.

Trains a binary classifier (relevant vs not) on positive vs sampled-negative
(user, item) pairs, using the full user + item + pair feature table (including
cross/pair features, which is where a GBDT shines over a two-tower model).
Predicted probability of the positive class is the ranking score.

Backends:
* ``catboost``   — CatBoostClassifier with native categorical handling (default).
* ``hist_gbdt``  — sklearn HistGradientBoostingClassifier (ordinal-encoded
                   categoricals); runs without installing catboost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark.rankers.features import FeatureBuilder
from benchmark.rankers.labels import build_training_pairs


class GBDTRanker:
    name = "gbdt"

    def __init__(self, items, users, train_interactions, engine_name="benchmark_gbdt",
                 backend="catboost", n_neg_per_pos=4, seed=42,
                 user_col="user_id", item_col="item_id", **model_params):
        self.items, self.users, self.train = items, users, train_interactions
        self.engine_name = engine_name
        self.backend = backend
        self.n_neg_per_pos = n_neg_per_pos
        self.seed, self.user_col, self.item_col = seed, user_col, item_col
        self.model_params = model_params
        self.fb: FeatureBuilder | None = None

    def fit(self) -> "GBDTRanker":
        self.fb = FeatureBuilder(self.engine_name, self.items, self.users, self.train,
                                 self.user_col, self.item_col).build()
        pairs_df = build_training_pairs(self.train, self.items, self.n_neg_per_pos,
                                        self.seed, self.user_col, self.item_col)
        if pairs_df.empty:
            raise RuntimeError("No training pairs available for the GBDT ranker.")

        pairs = list(pairs_df[[self.user_col, self.item_col]].itertuples(index=False, name=None))
        X = self.fb.pair_frame(pairs)
        y = pairs_df["label"].to_numpy(dtype=int)
        self.cat_cols = self.fb.categorical_columns()

        if self.backend == "catboost":
            self._fit_catboost(X, y)
        elif self.backend == "hist_gbdt":
            self._fit_hist_gbdt(X, y)
        else:
            raise ValueError(f"Unknown GBDT backend '{self.backend}' (use catboost|hist_gbdt)")
        return self

    # -- catboost -------------------------------------------------------------
    def _fit_catboost(self, X: pd.DataFrame, y: np.ndarray) -> None:
        try:
            from catboost import CatBoostClassifier, Pool
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "catboost is not installed. Install it with `poetry add catboost` "
                "(or run the config with backend='hist_gbdt' for the sklearn variant)."
            ) from e
        params = {"iterations": 400, "depth": 6, "learning_rate": 0.05,
                  "loss_function": "Logloss", "verbose": False,
                  "random_seed": self.seed}
        params.update({k: v for k, v in self.model_params.items()
                       if k in {"iterations", "depth", "learning_rate", "l2_leaf_reg"}})
        self._cat_feature_names = self.cat_cols
        pool = Pool(X, y, cat_features=self.cat_cols)
        self.model = CatBoostClassifier(**params)
        self.model.fit(pool)

    # -- sklearn HistGradientBoosting -----------------------------------------
    def _fit_hist_gbdt(self, X: pd.DataFrame, y: np.ndarray) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import OrdinalEncoder

        self._encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xc = X.copy()
        if self.cat_cols:
            Xc[self.cat_cols] = self._encoder.fit_transform(Xc[self.cat_cols].astype(str))
        self._feature_cols = list(Xc.columns)
        cat_mask = [c in self.cat_cols for c in self._feature_cols]
        self.model = HistGradientBoostingClassifier(
            random_state=self.seed, categorical_features=cat_mask,
            max_iter=self.model_params.get("iterations", 300),
        )
        self.model.fit(Xc.to_numpy(dtype=float) if not self.cat_cols else Xc, y)

    # -- scoring --------------------------------------------------------------
    def score_pairs(self, pairs: list[tuple[str, str]]) -> dict[tuple[str, str], float]:
        X = self.fb.pair_frame(pairs)
        if self.backend == "catboost":
            proba = self.model.predict_proba(X)[:, 1]
        else:
            Xc = X.copy()
            if self.cat_cols:
                Xc[self.cat_cols] = self._encoder.transform(Xc[self.cat_cols].astype(str))
            proba = self.model.predict_proba(Xc)[:, 1]
        return {(u, i): float(p) for (u, i), p in zip(pairs, proba)}
