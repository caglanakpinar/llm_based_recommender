from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import warnings


warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
NEGATIVE_ACTIONS = {"dislike", "remove"}
POSITIVE_ACTIONS = {"view", "click", "like", "add_to_cart", "purchase", "rate", "share"}


def _user_id_col(kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("user_id", "user_id"))


def _item_id_col(kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("item_id", "item_id"))


def _group_user(kwargs: dict[str, Any]) -> list[str]:
    return [_user_id_col(kwargs)]


def _group_item(kwargs: dict[str, Any]) -> list[str]:
    return [_item_id_col(kwargs)]


def _group_pair(kwargs: dict[str, Any]) -> list[str]:
    return [_user_id_col(kwargs), _item_id_col(kwargs)]


def _normalize_actions(df: pd.DataFrame) -> pd.Series:
    if "action" not in df.columns:
        return pd.Series(index=df.index, dtype=str)
    return df["action"].astype(str).str.strip().str.lower()


def _timestamps(df: pd.DataFrame) -> pd.Series:
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if "created_at" in df.columns:
        return pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    return pd.Series(index=df.index, dtype="datetime64[ns, UTC]")


def _to_numeric(s: pd.Series | Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _series_or_empty(df: pd.DataFrame, col: str, dtype: str = "object") -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(index=df.index, dtype=dtype)


def _mode_as_str(s: pd.Series) -> str:
    if s.empty:
        return ""
    mode = s.dropna().astype(str).mode()
    if mode.empty:
        return ""
    return str(mode.iloc[0])


def _mean_or_zero(s: pd.Series) -> float:
    n = _to_numeric(s).dropna()
    if n.empty:
        return 0.0
    return float(n.mean())


def _std_or_zero(s: pd.Series) -> float:
    n = _to_numeric(s).dropna()
    if n.empty:
        return 0.0
    return float(n.std(ddof=0))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _days_since_latest(ts: pd.Series) -> float:
    valid = pd.to_datetime(ts, errors="coerce", utc=True).dropna()
    if valid.empty:
        return 0.0
    delta = pd.Timestamp.now(tz="UTC") - valid.max()
    return float(max(delta.total_seconds() / 86400.0, 0.0))


def _recency_weight(ts: pd.Series, half_life_days: float = 30.0) -> float:
    valid = pd.to_datetime(ts, errors="coerce", utc=True).dropna()
    if valid.empty:
        return 0.0
    age_days = (pd.Timestamp.now(tz="UTC") - valid).dt.total_seconds() / 86400.0
    lam = np.log(2.0) / max(half_life_days, 1.0)
    weights = np.exp(-lam * age_days)
    return float(np.mean(weights)) if len(weights) else 0.0


def _split_tags(v: Any) -> set[str]:
    if isinstance(v, list):
        return {str(x).strip().lower() for x in v if str(x).strip()}
    if isinstance(v, str):
        return {x.strip().lower() for x in v.split(",") if x.strip()}
    return set()


def _action_weight(action: str) -> float:
    mapping = {
        "purchase": 1.0,
        "add_to_cart": 0.85,
        "like": 0.75,
        "rate": 0.70,
        "share": 0.65,
        "click": 0.50,
        "view": 0.30,
        "remove": -0.5,
        "dislike": -1.0,
    }
    return float(mapping.get(str(action).strip().lower(), 0.0))


def _ensure_group_cols(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in group_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


def _group_feature(
    data: pd.DataFrame,
    feature_name: str,
    group_cols: list[str],
    calc: Callable[[pd.DataFrame], Any],
) -> pd.DataFrame:
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()
    df = _ensure_group_cols(df, group_cols)

    if df.empty:
        return pd.DataFrame(columns=[*group_cols, feature_name])

    rows: list[dict[str, Any]] = []
    grouped = df.groupby(group_cols, dropna=False, as_index=False)
    for keys, grp in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row[feature_name] = calc(grp)
        rows.append(row)
    return pd.DataFrame(rows)


# -------------------------
# Item features (group: item_id)
# -------------------------

def num_unique_users_interacted(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    item_col = _item_id_col(kwargs)
    user_col = _user_id_col(kwargs)
    return _group_feature(
        data,
        "num_unique_users_interacted",
        [item_col],
        lambda g: float(g[user_col].nunique(dropna=True)) if user_col in g.columns else 0.0,
    )


def num_purchases_item(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    item_col = _item_id_col(kwargs)
    return _group_feature(
        data,
        "num_purchases_item",
        [item_col],
        lambda g: float((_normalize_actions(g) == "purchase").sum()),
    )


def purchase_rate_item(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    item_col = _item_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        actions = _normalize_actions(g)
        return _safe_ratio(float((actions == "purchase").sum()), float(len(g)))

    return _group_feature(data, "purchase_rate_item", [item_col], _calc)


def num_common_users_item_target(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    item_col = _item_id_col(kwargs)
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        if user_col not in g.columns:
            return 0.0
        if "is_target_user" in g.columns:
            target_users = set(g.loc[_to_numeric(g["is_target_user"]).fillna(0) > 0, user_col].dropna().astype(str))
        else:
            target_users = set(g.get("target_user_id", pd.Series(dtype=str)).dropna().astype(str))
        item_users = set(g[user_col].dropna().astype(str))
        return float(len(target_users & item_users))

    return _group_feature(data, "num_common_users_item_target", [item_col], _calc)


def num_similar_users_purchased_item(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    item_col = _item_id_col(kwargs)
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        actions = _normalize_actions(g)
        purchased = actions == "purchase"
        if "is_similar_user" in g.columns:
            similar = _to_numeric(g["is_similar_user"]).fillna(0) > 0
        else:
            similar = pd.Series(True, index=g.index)
        if user_col in g.columns:
            return float(g.loc[purchased & similar, user_col].nunique(dropna=True))
        return float((purchased & similar).sum())

    return _group_feature(data, "num_similar_users_purchased_item", [item_col], _calc)

def num_similar_users_purchased_item_per_user(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    item_col = _item_id_col(kwargs)
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        actions = _normalize_actions(g)
        purchased = actions == "purchase"
        if "is_similar_user" in g.columns:
            similar = _to_numeric(g["is_similar_user"]).fillna(0) > 0
        else:
            similar = pd.Series(True, index=g.index)
        if user_col in g.columns:
            return float(g.loc[purchased & similar, user_col].nunique(dropna=True))
        return float((purchased & similar).sum())

    return _group_feature(data, "num_similar_users_purchased_item_per_user", [user_col, item_col], _calc)


def avg_similarity_users_item(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    item_col = _item_id_col(kwargs)
    return _group_feature(
        data,
        "avg_similarity_users_item",
        [item_col],
        lambda g: _mean_or_zero(_series_or_empty(g, "similarity", "float")),
    )


def category_match_score(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    item_col = _item_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        if "category_match" in g.columns:
            return _mean_or_zero(g["category_match"])
        if "target_category" in g.columns and "category" in g.columns:
            match = g["target_category"].astype(str).str.lower() == g["category"].astype(str).str.lower()
            return _mean_or_zero(match.astype(float))
        return 0.0

    return _group_feature(data, "category_match_score", [item_col], _calc)


def tag_overlap_score(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    item_col = _item_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        if "tag_overlap" in g.columns:
            return _mean_or_zero(g["tag_overlap"])
        if "target_tags" not in g.columns or "tags" not in g.columns:
            return 0.0
        vals = []
        for _, row in g.iterrows():
            a = _split_tags(row.get("target_tags"))
            b = _split_tags(row.get("tags"))
            denom = max(len(a | b), 1)
            vals.append(len(a & b) / denom)
        return float(np.mean(vals)) if vals else 0.0

    return _group_feature(data, "tag_overlap_score", [item_col], _calc)


def price_fit_score(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    item_col = _item_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        if "price_fit" in g.columns:
            return _mean_or_zero(g["price_fit"])
        price = _to_numeric(_series_or_empty(g, "price", "float")).dropna()
        if price.empty:
            return 0.0
        target = _to_numeric(_series_or_empty(g, "target_avg_price", "float")).dropna()
        center = float(target.mean()) if not target.empty else float(price.mean())
        score = 1.0 - np.minimum(np.abs(price - center) / max(center, 1.0), 1.0)
        return float(score.mean()) if len(score) else 0.0

    return _group_feature(data, "price_fit_score", [item_col], _calc)


def days_since_last_item_interaction(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    item_col = _item_id_col(kwargs)
    return _group_feature(data, "days_since_last_item_interaction", [item_col], lambda g: _days_since_latest(_timestamps(g)))


def lookback_window_days(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Return the lookback window days feature per item from pre-calculated data or parameter."""
    item_col = _item_id_col(kwargs)
    
    def _calc(g: pd.DataFrame) -> float:
        if "lookback_days" in g.columns:
            return _mean_or_zero(_to_numeric(g["lookback_days"]))
        lookback_window = int(kwargs.get("lookback_window_days", 30))
        return float(lookback_window)
    
    return _group_feature(data, "lookback_window_days", [item_col], _calc)


def recent_trend_score(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    item_col = _item_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        ts = _timestamps(g).dropna()
        if ts.empty:
            return 0.0
        
        # Get lookback_days from data if available, otherwise from kwargs
        lookback_days_val = g["lookback_window_days"]
        now = pd.Timestamp.now(tz="UTC")
        recent = ((now - ts).dt.days <= 7).sum()
        medium = (((now - ts).dt.days > 7) & ((now - ts).dt.days <= lookback_days_val)).sum()
        return _safe_ratio(float(recent + 0.5 * medium), float(len(ts)))

    return _group_feature(data, "recent_trend_score", [item_col], _calc)


def item_recency_weight(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    item_col = _item_id_col(kwargs)
    half_life_days = float(kwargs.get("half_life_days", 30.0))
    return _group_feature(
        data,
        "item_recency_weight",
        [item_col],
        lambda g: _recency_weight(_timestamps(g), half_life_days=half_life_days),
    )


def already_purchased_flag(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    # Pair-related eligibility check grouped by user-item.
    group_cols = _group_pair(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        if "already_purchased" in g.columns:
            return float((_to_numeric(g["already_purchased"]).fillna(0) > 0).any())
        return float((_normalize_actions(g) == "purchase").any())

    return _group_feature(data, "already_purchased_flag", group_cols, _calc)


def negative_feedback_flag(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    group_cols = _group_pair(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        if "negative_feedback" in g.columns:
            return float((_to_numeric(g["negative_feedback"]).fillna(0) > 0).any())
        return float(_normalize_actions(g).isin(NEGATIVE_ACTIONS).any())

    return _group_feature(data, "negative_feedback_flag", group_cols, _calc)


def eligible_flag(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    # default for pair scoring
    group_cols = _group_pair(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        purchased = bool((already_purchased_flag(g, **kwargs)["already_purchased_flag"] > 0).any())
        negative = bool((negative_feedback_flag(g, **kwargs)["negative_feedback_flag"] > 0).any())
        return float(not (purchased or negative))

    return _group_feature(data, "eligible_flag", group_cols, _calc)


def relevance_score(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    # Pair-level relevance scoring. If label exists, fit a simple linear model.
    group_cols = _group_pair(kwargs)
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()
    df = _ensure_group_cols(df, group_cols)
    if df.empty:
        return pd.DataFrame(columns=[*group_cols, "relevance_score"])

    # If explicit relevance labels already exist, preserve grouped values.
    if "relevance_score" in df.columns:
        return _group_feature(df, "relevance_score", group_cols, lambda g: _mean_or_zero(g["relevance_score"]))

    pair_df = df[group_cols].drop_duplicates().reset_index(drop=True)

    model_feature_funcs = [
        num_user_item_interactions,
        days_since_last_user_item_interaction,
        num_views_user_item,
        num_clicks_user_item,
        purchased_user_item_flag,
        negative_user_item_flag,
        weighted_user_item_signal,
        recency_weighted_user_item_signal,
        pair_confidence_score,
        item_popularity_percentile,
        item_conversion_rate,
        pair_novelty_score,
        num_similar_users_interacted_item,
        num_similar_users_purchased_item,
        peer_agreement_score,
        next_item_probability,
        session_cooccurrence_score,
        item_transition_score,
        eligible_flag,
    ]

    for fn in model_feature_funcs:
        try:
            fdf = fn(df, **kwargs)
            feature_cols = [c for c in fdf.columns if c not in group_cols]
            if len(feature_cols) != 1:
                continue
            # Check that all merge keys exist in the feature dataframe
            missing_cols = [col for col in group_cols if col not in fdf.columns]
            if missing_cols:
                continue
            pair_df = pair_df.merge(fdf, on=group_cols, how="left")
        except Exception:
            # Skip features that fail to compute
            continue

    numeric_cols = [
        c for c in pair_df.columns
        if c not in group_cols and pd.api.types.is_numeric_dtype(pair_df[c])
    ]
    if not numeric_cols:
        pair_df["relevance_score"] = 0.0
        return pair_df[[*group_cols, "relevance_score"]]

    x = pair_df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    label_col = str(kwargs.get("label_col", "label"))
    reg_lambda = float(kwargs.get("reg_lambda", 1e-3))
    y_df = None
    if label_col in df.columns:
        y_df = _group_feature(df, label_col, group_cols, lambda g: _mean_or_zero(g[label_col]))
        y_df = y_df.rename(columns={label_col: "_y"})

    if y_df is not None:
        train_df = pair_df.merge(y_df, on=group_cols, how="left")
        y = pd.to_numeric(train_df["_y"], errors="coerce").fillna(np.nan).to_numpy(dtype=float)
        valid = np.isfinite(y)
    else:
        train_df = pair_df.copy()
        y = np.array([], dtype=float)
        valid = np.array([], dtype=bool)

    x_train = x[valid] if valid.size else np.empty((0, x.shape[1]))
    y_train = y[valid] if valid.size else np.empty((0,))

    if x_train.shape[0] >= max(5, x_train.shape[1]) and np.nanstd(y_train) > 0:
        x_mean = x_train.mean(axis=0)
        x_std = x_train.std(axis=0)
        x_std[x_std == 0] = 1.0

        xz = (x_train - x_mean) / x_std
        xa = np.hstack([np.ones((xz.shape[0], 1)), xz])
        identity = np.eye(xa.shape[1])
        identity[0, 0] = 0.0
        beta = np.linalg.pinv(xa.T @ xa + reg_lambda * identity) @ xa.T @ y_train

        xz_full = (x - x_mean) / x_std
        xa_full = np.hstack([np.ones((xz_full.shape[0], 1)), xz_full])
        preds = xa_full @ beta
        scores = np.clip(preds, 0.0, 1.0)
    else:
        # Fallback deterministic weighted score when labels are not available.
        conf = pair_df.get("pair_confidence_score", pd.Series(0.0, index=pair_df.index)).astype(float)
        rec = pair_df.get("recency_weighted_user_item_signal", pd.Series(0.0, index=pair_df.index)).astype(float)
        peer = pair_df.get("peer_agreement_score", pd.Series(0.0, index=pair_df.index)).astype(float)
        conv = pair_df.get("item_conversion_rate", pd.Series(0.0, index=pair_df.index)).astype(float)
        nxt = pair_df.get("next_item_probability", pd.Series(0.0, index=pair_df.index)).astype(float)
        elig = pair_df.get("eligible_flag", pd.Series(1.0, index=pair_df.index)).astype(float)
        neg = pair_df.get("negative_user_item_flag", pd.Series(0.0, index=pair_df.index)).astype(float)

        scores = (
            0.35 * conf
            + 0.20 * np.clip(rec, 0.0, 1.0)
            + 0.15 * peer
            + 0.10 * conv
            + 0.10 * nxt
            + 0.10 * np.clip(elig, 0.0, 1.0)
            - 0.25 * np.clip(neg, 0.0, 1.0)
        )
        scores = np.clip(scores.to_numpy(dtype=float), 0.0, 1.0)

    pair_df["relevance_score"] = scores
    return pair_df[[*group_cols, "relevance_score"]]


# -------------------------
# User features (group: user_id)
# -------------------------

def segment(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> str:
        # If segment already provided in data, use it
        if "segment" in g.columns:
            existing = _mode_as_str(_series_or_empty(g, "segment"))
            if existing:
                return existing

        # RFM-based segmentation
        now = pd.Timestamp.now(tz="UTC")
        ts = _timestamps(g).dropna()

        # Recency: days since last interaction
        recency = float((now - ts.max()).total_seconds() / 86400.0) if not ts.empty else 999.0

        # Frequency: number of interactions
        frequency = float(len(g))

        # Monetary: average purchase price or purchase count as proxy
        actions = _normalize_actions(g)
        if "price" in g.columns:
            monetary = _mean_or_zero(_to_numeric(g.loc[actions == "purchase", "price"]))
        else:
            monetary = float((actions == "purchase").sum())

        # Score each dimension (1=low, 3=high)
        r_score = 3 if recency <= 7 else (2 if recency <= 30 else 1)
        f_score = 3 if frequency >= 10 else (2 if frequency >= 3 else 1)
        m_score = 3 if monetary >= 100 else (2 if monetary >= 10 else 1)
        rfm = r_score + f_score + m_score

        if rfm >= 8:
            return "champions"
        elif rfm >= 6:
            return "loyal"
        elif r_score >= 2 and f_score >= 2:
            return "potential_loyalist"
        elif r_score == 3 and f_score == 1:
            return "new_customer"
        elif r_score <= 1 and f_score >= 2:
            return "at_risk"
        elif r_score == 1 and f_score == 1:
            return "lost"
        else:
            return "occasional"

    return _group_feature(data, "segment", [user_col], _calc)


def profile_completeness_score(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        cols = [c for c in [user_col, "segment", "notes"] if c in g.columns]
        if not cols:
            return 0.0
        return float(g[cols].notna().mean(axis=1).mean())

    return _group_feature(data, "profile_completeness_score", [user_col], _calc)


def profile_age_days(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        if "profile_created_at" in g.columns:
            return _days_since_latest(g["profile_created_at"])
        return _days_since_latest(_timestamps(g))

    return _group_feature(data, "profile_age_days", [user_col], _calc)


def num_total_interactions(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)
    return _group_feature(data, "num_total_interactions", [user_col], lambda g: float(len(g)))


def num_active_days(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        ts = _timestamps(g).dropna()
        if ts.empty:
            return 0.0
        return float(ts.dt.date.nunique())

    return _group_feature(data, "num_active_days", [user_col], _calc)


def days_since_last_interaction(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)
    return _group_feature(data, "days_since_last_interaction", [user_col], lambda g: _days_since_latest(_timestamps(g)))


def top_categories(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> str:
        if "category" not in g.columns:
            return ""
        top = g["category"].dropna().astype(str).value_counts().head(3).index.tolist()
        return ",".join(top)

    return _group_feature(data, "top_categories", [user_col], _calc)


def category_concentration_score(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        if "category" not in g.columns:
            return 0.0
        vc = g["category"].dropna().astype(str).value_counts(normalize=True)
        if vc.empty:
            return 0.0
        return float(vc.iloc[0])

    return _group_feature(data, "category_concentration_score", [user_col], _calc)


def top_tags(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> str:
        if "tags" not in g.columns:
            return ""
        tags: list[str] = []
        for v in g["tags"].dropna():
            tags.extend(list(_split_tags(v)))
        if not tags:
            return ""
        top = pd.Series(tags).value_counts().head(5).index.tolist()
        return ",".join(top)

    return _group_feature(data, "top_tags", [user_col], _calc)


def tag_diversity_score(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        if "tags" not in g.columns:
            return 0.0
        all_tags: set[str] = set()
        total = 0
        for v in g["tags"].dropna():
            split = _split_tags(v)
            all_tags |= split
            total += len(split)
        if total == 0:
            return 0.0
        return float(len(all_tags) / total)

    return _group_feature(data, "tag_diversity_score", [user_col], _calc)


def num_purchases_user(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)
    return _group_feature(data, "num_purchases_user", [user_col], lambda g: float((_normalize_actions(g) == "purchase").sum()))


def num_likes_user(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)
    return _group_feature(data, "num_likes_user", [user_col], lambda g: float((_normalize_actions(g) == "like").sum()))


def ctr_user(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        actions = _normalize_actions(g)
        clicks = float((actions == "click").sum())
        views = float((actions == "view").sum())
        return _safe_ratio(clicks, views)

    return _group_feature(data, "ctr_user", [user_col], _calc)


def purchase_conversion_rate_user(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        actions = _normalize_actions(g)
        purchases = float((actions == "purchase").sum())
        intents = float(actions.isin(POSITIVE_ACTIONS).sum())
        return _safe_ratio(purchases, intents)

    return _group_feature(data, "purchase_conversion_rate_user", [user_col], _calc)


def avg_view_price_user(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        if "price" not in g.columns:
            return 0.0
        actions = _normalize_actions(g)
        return _mean_or_zero(_to_numeric(g.loc[actions == "view", "price"]))

    return _group_feature(data, "avg_view_price_user", [user_col], _calc)


def avg_purchase_price_user(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        if "price" not in g.columns:
            return 0.0
        actions = _normalize_actions(g)
        return _mean_or_zero(_to_numeric(g.loc[actions == "purchase", "price"]))

    return _group_feature(data, "avg_purchase_price_user", [user_col], _calc)


def price_sensitivity_score(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        if "price" not in g.columns:
            return 0.0
        std = _std_or_zero(_to_numeric(g["price"]))
        return float(1.0 / (1.0 + std))

    return _group_feature(data, "price_sensitivity_score", [user_col], _calc)


def num_similar_users(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        if "is_similar_user" in g.columns and user_col in g.columns:
            mask = _to_numeric(g["is_similar_user"]).fillna(0) > 0
            return float(g.loc[mask, user_col].nunique(dropna=True))
        if user_col in g.columns:
            return float(g[user_col].nunique(dropna=True))
        return 0.0

    return _group_feature(data, "num_similar_users", [user_col], _calc)


def avg_similarity_to_neighbors(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        if "similarity" not in g.columns:
            return 0.0
        if "is_neighbor" in g.columns:
            mask = _to_numeric(g["is_neighbor"]).fillna(0) > 0
            return _mean_or_zero(g.loc[mask, "similarity"])
        return _mean_or_zero(g["similarity"])

    return _group_feature(data, "avg_similarity_to_neighbors", [user_col], _calc)


def peer_influence_score(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        if "peer_influence" in g.columns:
            return _mean_or_zero(g["peer_influence"])
        sim = float(avg_similarity_to_neighbors(g, **kwargs)["avg_similarity_to_neighbors"].mean())
        similar = float(num_similar_users(g, **kwargs)["num_similar_users"].mean())
        return float(np.clip(0.6 * sim + 0.4 * _safe_ratio(similar, max(len(g), 1.0)), 0.0, 1.0))

    return _group_feature(data, "peer_influence_score", [user_col], _calc)


def churn_risk_score(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        recency = float(days_since_last_interaction(g, **kwargs)["days_since_last_interaction"].mean())
        active_days = float(num_active_days(g, **kwargs)["num_active_days"].mean())
        risk = 1.0 - np.exp(-recency / 30.0)
        activity_factor = 1.0 / (1.0 + active_days)
        return float(np.clip(0.7 * risk + 0.3 * activity_factor, 0.0, 1.0))

    return _group_feature(data, "churn_risk_score", [user_col], _calc)


def reactivation_likelihood(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)
    return _group_feature(
        data,
        "reactivation_likelihood",
        [user_col],
        lambda g: float(np.clip(1.0 - float(churn_risk_score(g, **kwargs)["churn_risk_score"].mean()), 0.0, 1.0)),
    )


def lifecycle_stage(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> str:
        total = float(len(g))
        if total == 0:
            return "new"
        if total < 5:
            return "early"
        risk = float(churn_risk_score(g, **kwargs)["churn_risk_score"].mean())
        if risk > 0.7:
            return "churn_risk"
        return "active"

    return _group_feature(data, "lifecycle_stage", [user_col], _calc)


# -------------------------
# User-item pair features (group: user_id, item_id)
# -------------------------

def num_user_item_interactions(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    return _group_feature(data, "num_user_item_interactions", _group_pair(kwargs), lambda g: float(len(g)))


def last_action_user_item(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    group_cols = _group_pair(kwargs)

    def _calc(g: pd.DataFrame) -> str:
        if "action" not in g.columns:
            return ""
        if "timestamp" not in g.columns and "created_at" not in g.columns:
            return _mode_as_str(g["action"])
        ordered = g.copy()
        ordered["__ts"] = _timestamps(ordered)
        ordered = ordered.sort_values("__ts")
        if ordered.empty:
            return ""
        return str(ordered["action"].iloc[-1])

    return _group_feature(data, "last_action_user_item", group_cols, _calc)


def days_since_last_user_item_interaction(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    return _group_feature(
        data,
        "days_since_last_user_item_interaction",
        _group_pair(kwargs),
        lambda g: _days_since_latest(_timestamps(g)),
    )


def num_views_user_item(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    return _group_feature(data, "num_views_user_item", _group_pair(kwargs), lambda g: float((_normalize_actions(g) == "view").sum()))


def num_clicks_user_item(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    return _group_feature(data, "num_clicks_user_item", _group_pair(kwargs), lambda g: float((_normalize_actions(g) == "click").sum()))


def purchased_user_item_flag(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    return _group_feature(data, "purchased_user_item_flag", _group_pair(kwargs), lambda g: float((_normalize_actions(g) == "purchase").any()))


def negative_user_item_flag(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    return _group_feature(data, "negative_user_item_flag", _group_pair(kwargs), lambda g: float(_normalize_actions(g).isin(NEGATIVE_ACTIONS).any()))


def weighted_user_item_signal(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    return _group_feature(
        data,
        "weighted_user_item_signal",
        _group_pair(kwargs),
        lambda g: float(np.mean([_action_weight(a) for a in _normalize_actions(g)])) if len(g) else 0.0,
    )


def recency_weighted_user_item_signal(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    group_cols = _group_pair(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        actions = _normalize_actions(g)
        ts = _timestamps(g)
        if actions.empty:
            return 0.0
        weights = actions.map(_action_weight).to_numpy(dtype=float)
        rec = _recency_weight(ts, half_life_days=float(kwargs.get("half_life_days", 30.0)))
        if rec == 0:
            return float(np.mean(weights)) if len(weights) else 0.0
        # Use scalar recency average to scale interaction score.
        return float(np.mean(weights) * rec)

    return _group_feature(data, "recency_weighted_user_item_signal", group_cols, _calc)


def pair_confidence_score(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    group_cols = _group_pair(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        interactions = float(len(g))
        signal = float(recency_weighted_user_item_signal(g, **kwargs)["recency_weighted_user_item_signal"].mean())
        return float(np.clip((1.0 - np.exp(-interactions / 5.0)) * (0.5 + 0.5 * max(signal, 0.0)), 0.0, 1.0))

    return _group_feature(data, "pair_confidence_score", group_cols, _calc)


def item_popularity_percentile(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    group_cols = _group_pair(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        if "item_popularity_percentile" in g.columns:
            return _mean_or_zero(g["item_popularity_percentile"])
        if "item_interaction_count" not in g.columns:
            return 0.0
        counts = _to_numeric(g["item_interaction_count"]).dropna()
        if counts.empty:
            return 0.0
        return float(counts.rank(pct=True).mean())

    return _group_feature(data, "item_popularity_percentile", group_cols, _calc)


def item_conversion_rate(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    group_cols = _group_pair(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        actions = _normalize_actions(g)
        purchases = float((actions == "purchase").sum())
        total = float(len(g))
        return _safe_ratio(purchases, total)

    return _group_feature(data, "item_conversion_rate", group_cols, _calc)


def pair_novelty_score(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    return _group_feature(data, "pair_novelty_score", _group_pair(kwargs), lambda g: float(np.clip(1.0 / (1.0 + len(g)), 0.0, 1.0)))


def num_similar_users_interacted_item(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    group_cols = _group_pair(kwargs)
    user_col = _user_id_col(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        if "is_similar_user" in g.columns and user_col in g.columns:
            mask = _to_numeric(g["is_similar_user"]).fillna(0) > 0
            return float(g.loc[mask, user_col].nunique(dropna=True))
        if user_col in g.columns:
            return float(g[user_col].nunique(dropna=True))
        return 0.0

    return _group_feature(data, "num_similar_users_interacted_item", group_cols, _calc)


def peer_agreement_score(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    group_cols = _group_pair(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        interacted = float(num_similar_users_interacted_item(g, **kwargs)["num_similar_users_interacted_item"].mean())
        purchased = float(num_similar_users_purchased_item(g, **kwargs)["num_similar_users_purchased_item"].mean())
        return _safe_ratio(purchased, interacted)

    return _group_feature(data, "peer_agreement_score", group_cols, _calc)


def next_item_probability(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    group_cols = _group_pair(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        if "next_item_probability" in g.columns:
            return _mean_or_zero(g["next_item_probability"])
        conf = float(pair_confidence_score(g, **kwargs)["pair_confidence_score"].mean())
        peer = float(peer_agreement_score(g, **kwargs)["peer_agreement_score"].mean())
        return float(np.clip(0.6 * conf + 0.4 * peer, 0.0, 1.0))

    return _group_feature(data, "next_item_probability", group_cols, _calc)


def session_cooccurrence_score(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    group_cols = _group_pair(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        if "session_cooccurrence_score" in g.columns:
            return _mean_or_zero(g["session_cooccurrence_score"])
        if "session_id" not in g.columns:
            return 0.0
        sessions = g["session_id"].dropna().astype(str)
        if sessions.empty:
            return 0.0
        return float(sessions.value_counts(normalize=True).iloc[0])

    return _group_feature(data, "session_cooccurrence_score", group_cols, _calc)


def item_transition_score(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    group_cols = _group_pair(kwargs)

    def _calc(g: pd.DataFrame) -> float:
        if "item_transition_score" in g.columns:
            return _mean_or_zero(g["item_transition_score"])
        sess = float(session_cooccurrence_score(g, **kwargs)["session_cooccurrence_score"].mean())
        nxt = float(next_item_probability(g, **kwargs)["next_item_probability"].mean())
        return float(np.clip(0.5 * sess + 0.5 * nxt, 0.0, 1.0))

    return _group_feature(data, "item_transition_score", group_cols, _calc)


def exclusion_reason_code(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    group_cols = _group_pair(kwargs)

    def _calc(g: pd.DataFrame) -> str:
        purchased = float(already_purchased_flag(g, **kwargs)["already_purchased_flag"].mean())
        negative = float(negative_feedback_flag(g, **kwargs)["negative_feedback_flag"].mean())
        if purchased > 0:
            return "already_purchased"
        if negative > 0:
            return "negative_feedback"
        return "none"

    return _group_feature(data, "exclusion_reason_code", group_cols, _calc)


FEATURES = {
    "item_features": {
        "num_unique_users_interacted": num_unique_users_interacted,
        "num_purchases_item": num_purchases_item,
        "purchase_rate_item": purchase_rate_item,
        "num_common_users_item_target": num_common_users_item_target,
        "num_similar_users_purchased_item": num_similar_users_purchased_item,
        "avg_similarity_users_item": avg_similarity_users_item,
        "category_match_score": category_match_score,
        "tag_overlap_score": tag_overlap_score,
        "price_fit_score": price_fit_score,
        "days_since_last_item_interaction": days_since_last_item_interaction,
        "lookback_window_days": lookback_window_days,
        "recent_trend_score": recent_trend_score,
        "item_recency_weight": item_recency_weight,
        "already_purchased_flag": already_purchased_flag,
        "negative_feedback_flag": negative_feedback_flag,
        "eligible_flag": eligible_flag
    },
    "user_features": {
        "profile_completeness_score": profile_completeness_score,
        "profile_age_days": profile_age_days,
        "num_total_interactions": num_total_interactions,
        "num_active_days": num_active_days,
        "days_since_last_interaction": days_since_last_interaction,
        "top_categories": top_categories,
        "category_concentration_score": category_concentration_score,
        "top_tags": top_tags,
        "tag_diversity_score": tag_diversity_score,
        "num_purchases_user": num_purchases_user,
        "num_likes_user": num_likes_user,
        "ctr_user": ctr_user,
        "purchase_conversion_rate_user": purchase_conversion_rate_user,
        "avg_view_price_user": avg_view_price_user,
        "avg_purchase_price_user": avg_purchase_price_user,
        "price_sensitivity_score": price_sensitivity_score,
        "num_similar_users": num_similar_users,
        "avg_similarity_to_neighbors": avg_similarity_to_neighbors,
        "peer_influence_score": peer_influence_score,
        "churn_risk_score": churn_risk_score,
        "reactivation_likelihood": reactivation_likelihood,
        "lifecycle_stage": lifecycle_stage,
    },
    "user_item_pair_features": {
        "num_user_item_interactions": num_user_item_interactions,
        "last_action_user_item": last_action_user_item,
        "days_since_last_user_item_interaction": days_since_last_user_item_interaction,
        "num_views_user_item": num_views_user_item,
        "num_clicks_user_item": num_clicks_user_item,
        "purchased_user_item_flag": purchased_user_item_flag,
        "negative_user_item_flag": negative_user_item_flag,
        "weighted_user_item_signal": weighted_user_item_signal,
        "recency_weighted_user_item_signal": recency_weighted_user_item_signal,
        "pair_confidence_score": pair_confidence_score,
        "item_popularity_percentile": item_popularity_percentile,
        "item_conversion_rate": item_conversion_rate,
        "pair_novelty_score": pair_novelty_score,
        "num_similar_users_interacted_item": num_similar_users_interacted_item,
        "num_similar_users_purchased_item_per_user": num_similar_users_purchased_item_per_user,
        "peer_agreement_score": peer_agreement_score,
        "next_item_probability": next_item_probability,
        "session_cooccurrence_score": session_cooccurrence_score,
        "item_transition_score": item_transition_score,
        "eligible_flag": eligible_flag,
        "exclusion_reason_code": exclusion_reason_code,
        "relevance_score": relevance_score,
    },
}

