"""Feature builder for the non-LLM rankers.

Reuses the *already-engineered* features from the prompt-engineering stage
(``core2.features.FEATURES`` computed by the ``core2.prompting`` classes) rather
than recomputing anything. It exposes:

* ``user_matrix`` / ``item_matrix`` — dense numeric matrices for the two towers.
* ``pair_frame`` — a per-(user,item) table (numeric + categorical) for GBDT.

The three prompt contexts are built exactly as in the LLM pipeline, so the two
paths consume identical feature definitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core2.datasets import DataSets
from core2.prompting import ItemPrompt, UserItemPrompt, UserPrompt
from core2.features import FEATURES

# Feature keys that are categorical/text rather than numeric.
USER_CATEGORICAL = {"segment", "top_categories", "top_tags", "lifecycle_stage"}
ITEM_CATEGORICAL = {"category"}
PAIR_CATEGORICAL = {"last_action_user_item", "exclusion_reason_code"}


def _to_numeric_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Coerce the given columns to float, filling non-numeric/missing with 0."""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else 0.0
    return out.fillna(0.0)


class FeatureBuilder:
    def __init__(self, engine_name: str, items: pd.DataFrame, users: pd.DataFrame,
                 train_interactions: pd.DataFrame, user_col: str = "user_id", item_col: str = "item_id"):
        self.engine_name = engine_name
        self.user_col = user_col
        self.item_col = item_col
        self._items = items.reset_index(drop=True)
        self._users = users.reset_index(drop=True)
        self._train = train_interactions.reset_index(drop=True)
        self._built = False

    # -- build the engineered feature contexts --------------------------------
    def build(self) -> "FeatureBuilder":
        datasets = DataSets(self.engine_name)
        datasets.item, datasets.user, datasets.item_user = self._items, self._users, self._train

        up = UserPrompt(self.engine_name, datasets); up.build_user_feature_dataset()
        ip = ItemPrompt(self.engine_name, datasets); ip.build_item_feature_dataset()
        uip = UserItemPrompt(self.engine_name, datasets); uip.build_user_item_feature_dataset()

        user_keys = list(FEATURES["user_features"].keys())
        item_keys = list(FEATURES["item_features"].keys())
        pair_keys = list(FEATURES["user_item_pair_features"].keys())

        self.user_num_cols = [k for k in user_keys if k not in USER_CATEGORICAL]
        self.item_num_cols = [k for k in item_keys if k not in ITEM_CATEGORICAL] + ["price"]
        self.pair_num_cols = [k for k in pair_keys if k not in PAIR_CATEGORICAL]
        self.user_cat_cols = [c for c in USER_CATEGORICAL if c in up.context.columns]
        self.item_cat_cols = [c for c in ITEM_CATEGORICAL if c in ip.context.columns]
        self.pair_cat_cols = [c for c in PAIR_CATEGORICAL if c in uip.context.columns]

        # Numeric frames indexed by id(s).
        self._user_num = _to_numeric_df(up.context, self.user_num_cols).set_axis(
            up.context[self.user_col].astype(str), axis=0)
        self._item_num = _to_numeric_df(ip.context, self.item_num_cols).set_axis(
            ip.context[self.item_col].astype(str), axis=0)
        pair_idx = pd.MultiIndex.from_frame(
            uip.context[[self.user_col, self.item_col]].astype(str))
        self._pair_num = _to_numeric_df(uip.context, self.pair_num_cols).set_index(pair_idx)

        # Categorical frames (strings) for GBDT.
        self._user_cat = up.context.set_index(up.context[self.user_col].astype(str))[self.user_cat_cols].astype(str) if self.user_cat_cols else pd.DataFrame(index=self._user_num.index)
        self._item_cat = ip.context.set_index(ip.context[self.item_col].astype(str))[self.item_cat_cols].astype(str) if self.item_cat_cols else pd.DataFrame(index=self._item_num.index)
        self._pair_cat = uip.context.set_index(pair_idx)[self.pair_cat_cols].astype(str) if self.pair_cat_cols else pd.DataFrame(index=self._pair_num.index)

        # One-hot maps for towers (segment, category give useful signal).
        self._user_onehot_col = "segment" if "segment" in up.context.columns else None
        self._item_onehot_col = "category" if "category" in ip.context.columns else None
        self._user_onehot_vals = sorted(up.context[self._user_onehot_col].astype(str).unique()) if self._user_onehot_col else []
        self._item_onehot_vals = sorted(ip.context[self._item_onehot_col].astype(str).unique()) if self._item_onehot_col else []
        self._user_seg = up.context.set_index(up.context[self.user_col].astype(str))[self._user_onehot_col].astype(str) if self._user_onehot_col else None
        self._item_cat_series = ip.context.set_index(ip.context[self.item_col].astype(str))[self._item_onehot_col].astype(str) if self._item_onehot_col else None

        self.user_tower_dim = len(self.user_num_cols) + len(self._user_onehot_vals)
        self.item_tower_dim = len(self.item_num_cols) + len(self._item_onehot_vals)
        self._built = True
        return self

    # -- tower feature matrices ----------------------------------------------
    def user_matrix(self, user_ids: list[str]) -> np.ndarray:
        num = self._user_num.reindex([str(u) for u in user_ids]).fillna(0.0).to_numpy(dtype=np.float32)
        if self._user_onehot_vals:
            oh = self._onehot(self._user_seg, user_ids, self._user_onehot_vals)
            return np.hstack([num, oh])
        return num

    def item_matrix(self, item_ids: list[str]) -> np.ndarray:
        num = self._item_num.reindex([str(i) for i in item_ids]).fillna(0.0).to_numpy(dtype=np.float32)
        if self._item_onehot_vals:
            oh = self._onehot(self._item_cat_series, item_ids, self._item_onehot_vals)
            return np.hstack([num, oh])
        return num

    @staticmethod
    def _onehot(series: pd.Series, ids: list[str], vocab: list[str]) -> np.ndarray:
        idx = {v: k for k, v in enumerate(vocab)}
        out = np.zeros((len(ids), len(vocab)), dtype=np.float32)
        vals = series.reindex([str(i) for i in ids])
        for r, v in enumerate(vals):
            j = idx.get(str(v))
            if j is not None:
                out[r, j] = 1.0
        return out

    # -- GBDT pair table ------------------------------------------------------
    def pair_frame(self, pairs: list[tuple[str, str]]) -> pd.DataFrame:
        """One row per (user,item): user + item + pair features (numeric + categorical)."""
        users = [str(u) for u, _ in pairs]
        items = [str(i) for _, i in pairs]
        mi = pd.MultiIndex.from_arrays([users, items], names=[self.user_col, self.item_col])

        u_num = self._user_num.reindex(users).fillna(0.0).reset_index(drop=True)
        i_num = self._item_num.reindex(items).fillna(0.0).reset_index(drop=True)
        p_num = self._pair_num.reindex(mi).fillna(0.0).reset_index(drop=True)
        u_cat = self._user_cat.reindex(users).fillna("none").reset_index(drop=True)
        i_cat = self._item_cat.reindex(items).fillna("none").reset_index(drop=True)
        p_cat = self._pair_cat.reindex(mi).fillna("none").reset_index(drop=True)

        frame = pd.concat([u_num, i_num, p_num, u_cat, i_cat, p_cat], axis=1)
        frame.columns = self.pair_feature_columns()
        return frame

    def pair_feature_columns(self) -> list[str]:
        # Source prefixes keep names unique (e.g. eligible_flag appears in both
        # the item and the pair feature sets).
        return (
            [f"u_{c}" for c in self.user_num_cols]
            + [f"i_{c}" for c in self.item_num_cols]
            + [f"p_{c}" for c in self.pair_num_cols]
            + [f"u_{c}" for c in self.user_cat_cols]
            + [f"i_{c}" for c in self.item_cat_cols]
            + [f"p_{c}" for c in self.pair_cat_cols]
        )

    def categorical_columns(self) -> list[str]:
        return (
            [f"u_{c}" for c in self.user_cat_cols]
            + [f"i_{c}" for c in self.item_cat_cols]
            + [f"p_{c}" for c in self.pair_cat_cols]
        )
