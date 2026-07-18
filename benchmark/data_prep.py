"""Data preparation for the offline benchmark.

Loads the item/user catalogs and interaction logs under ``data/``, samples a set
of evaluation users, applies a per-user temporal train/test split, and builds a
fair, bounded candidate pool with binary ground-truth labels.

Evaluation protocol (standard "sampled" re-ranking):
  * For each user, interactions are sorted by time and split into train (early)
    and test (late) portions.
  * Ground-truth relevant items = distinct items the user had a strong-positive
    action on (purchase / add_to_cart / like) during the *test* window.
  * The candidate pool = relevant items + popularity-sampled negatives (items the
    user never interacted with). Both engines score this identical pool, so the
    comparison isolates ranking quality from candidate generation.

Both the baseline and the LLM engine are trained on the *same* data: the training
interactions of the sampled users. Nothing from the test window leaks into training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root = parent of this file's directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# Strong-positive actions define ground-truth relevance. Weak-implicit actions
# (view/click) and negatives (dislike/remove) are intentionally excluded.
STRONG_POSITIVE_ACTIONS = ("purchase", "add_to_cart", "like")

# Graded action weights (mirrors core2.features._action_weight) used to weight
# the interaction matrix for the collaborative-filtering baseline.
ACTION_WEIGHTS: dict[str, float] = {
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


@dataclass
class BenchmarkData:
    """Everything the engines and evaluator need for one benchmark run."""

    items: pd.DataFrame                 # full item catalog
    users: pd.DataFrame                 # sampled users' profiles
    train_interactions: pd.DataFrame    # sampled users' training interactions (LLM engine)
    full_train_interactions: pd.DataFrame  # ALL users' training interactions (CF baseline)
    pool: pd.DataFrame                  # columns: user_id, item_id, label (0/1)
    relevants: dict[str, set[str]]      # user_id -> relevant item ids (label==1)
    user_ids: list[str] = field(default_factory=list)
    user_col: str = "user_id"
    item_col: str = "item_id"

    def pairs(self) -> list[tuple[str, str]]:
        """All (user_id, item_id) candidate pairs to be scored."""
        return list(
            self.pool[[self.user_col, self.item_col]].itertuples(index=False, name=None)
        )

    # -- persistence: share one sampled dataset across every ranker config -----
    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.items.to_parquet(out_dir / "items.parquet", index=False)
        self.users.to_parquet(out_dir / "users.parquet", index=False)
        self.train_interactions.to_parquet(out_dir / "train_interactions.parquet", index=False)
        self.full_train_interactions.to_parquet(out_dir / "full_train_interactions.parquet", index=False)
        self.pool.to_parquet(out_dir / "pool.parquet", index=False)
        (out_dir / "relevants.json").write_text(
            json.dumps({u: sorted(v) for u, v in self.relevants.items()})
        )
        (out_dir / "meta.json").write_text(
            json.dumps({"user_ids": self.user_ids, "user_col": self.user_col, "item_col": self.item_col})
        )

    @classmethod
    def load(cls, in_dir: Path) -> "BenchmarkData":
        in_dir = Path(in_dir)
        meta = json.loads((in_dir / "meta.json").read_text())
        relevants = {u: set(v) for u, v in json.loads((in_dir / "relevants.json").read_text()).items()}
        return cls(
            items=pd.read_parquet(in_dir / "items.parquet"),
            users=pd.read_parquet(in_dir / "users.parquet"),
            train_interactions=pd.read_parquet(in_dir / "train_interactions.parquet"),
            full_train_interactions=pd.read_parquet(in_dir / "full_train_interactions.parquet"),
            pool=pd.read_parquet(in_dir / "pool.parquet"),
            relevants=relevants,
            user_ids=meta["user_ids"],
            user_col=meta.get("user_col", "user_id"),
            item_col=meta.get("item_col", "item_id"),
        )


def load_or_build_benchmark_data(cache_dir: Path, *, rebuild: bool = False, **build_kwargs) -> BenchmarkData:
    """Load a cached shared dataset, or build+cache one so all configs align.

    The build parameters are recorded; if they change, the cache is rebuilt.
    """
    cache_dir = Path(cache_dir)
    signature = {k: (str(v) if isinstance(v, Path) else v) for k, v in build_kwargs.items()}
    sig_path = cache_dir / "build_signature.json"
    if not rebuild and (cache_dir / "meta.json").exists() and sig_path.exists():
        if json.loads(sig_path.read_text()) == signature:
            return BenchmarkData.load(cache_dir)
    data = build_benchmark_data(**build_kwargs)
    data.save(cache_dir)
    sig_path.write_text(json.dumps(signature))
    return data


def load_catalogs(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the full item and user catalogs from parquet."""
    items = pd.concat(
        [pd.read_parquet(p) for p in sorted((data_dir / "sample_items").glob("*.parquet"))],
        ignore_index=True,
    ).drop_duplicates(subset="item_id").reset_index(drop=True)
    users = pd.concat(
        [pd.read_parquet(p) for p in sorted((data_dir / "sample_users").glob("*.parquet"))],
        ignore_index=True,
    ).drop_duplicates(subset="user_id").reset_index(drop=True)
    return items, users


def load_interactions(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the interaction logs (only the id-consistent ``interactions_0*`` files).

    The tiny ``interactions.parquet`` (9 rows, ``user_001``/``item_001`` toy ids
    that do not match the catalogs) is deliberately excluded.
    """
    files = sorted((data_dir / "sample_interactions").glob("interactions_0*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No interactions_0*.parquet files found under {data_dir/'sample_interactions'}"
        )
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    df["action"] = df["action"].astype(str).str.strip().str.lower()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    return df


def temporal_split(user_df: pd.DataFrame, test_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one user's interactions into (train, test) by time.

    The most recent ``test_frac`` fraction (at least 1 row) becomes the test set.
    """
    ordered = user_df.sort_values("timestamp", kind="stable")
    n = len(ordered)
    n_test = max(1, int(round(n * test_frac)))
    n_test = min(n_test, n - 1) if n > 1 else n
    train = ordered.iloc[: n - n_test]
    test = ordered.iloc[n - n_test :]
    return train, test


def population_train_mask(interactions: pd.DataFrame, test_frac: float) -> pd.Series:
    """Vectorized per-user temporal split → boolean mask of *training* rows.

    Consistent with ``temporal_split``: each user's most-recent
    ``max(1, round(n*test_frac))`` interactions (clamped to leave >=1 train row)
    are marked as test; everything earlier is training.
    """
    df = interactions.sort_values(["user_id", "timestamp"], kind="stable")
    rev_rank = df.groupby("user_id").cumcount(ascending=False)  # 0 == most recent
    size = df.groupby("user_id")["item_id"].transform("size")
    n_test = np.maximum(1, np.round(size * test_frac)).astype(int)
    n_test = np.where(size > 1, np.minimum(n_test, size - 1), n_test)
    is_test = rev_rank.to_numpy() < n_test
    return pd.Series(~is_test, index=df.index).reindex(interactions.index)


def _popularity_weights(interactions: pd.DataFrame, item_col: str) -> pd.Series:
    """Item sampling weights ~ number of strong-positive interactions (smoothed)."""
    pos = interactions[interactions["action"].isin(STRONG_POSITIVE_ACTIONS)]
    counts = pos.groupby(item_col).size()
    return counts.add(1.0, fill_value=0.0)  # +1 smoothing so every item is reachable


def build_benchmark_data(
    n_users: int = 100,
    test_frac: float = 0.3,
    n_negatives: int = 20,
    min_train_interactions: int = 5,
    min_relevant: int = 1,
    seed: int = 42,
    negative_sampling: str = "uniform",
    data_dir: Path = DATA_DIR,
) -> BenchmarkData:
    """Sample users and assemble the full evaluation dataset.

    Args:
        n_users: number of evaluation users to sample.
        test_frac: fraction of each user's most-recent interactions held out.
        n_negatives: negatives per user in the candidate pool.
        min_train_interactions: minimum training interactions to keep a user.
        min_relevant: minimum strong-positive test items to keep a user.
        seed: RNG seed for reproducibility.
        negative_sampling: "uniform" (each unseen item equally likely) or
            "popularity" (harder negatives, weighted by item popularity).
    """
    rng = np.random.default_rng(seed)
    items, users = load_catalogs(data_dir)
    interactions = load_interactions(data_dir)
    all_item_ids = items["item_id"].astype(str).unique()

    # Full-population training split (used to fit the CF/popularity baseline so
    # item-item co-occurrence has population-scale signal). No test rows leak in.
    full_train_interactions = interactions[
        population_train_mask(interactions, test_frac).fillna(False).to_numpy()
    ].reset_index(drop=True)

    # Candidate users: enough history and known profile.
    known_users = set(users["user_id"].astype(str))
    counts = interactions.groupby("user_id").size()
    eligible = [
        u for u, c in counts.items()
        if c >= (min_train_interactions + 1) and str(u) in known_users
    ]
    rng.shuffle(eligible)

    pop_weights = _popularity_weights(interactions, "item_id")

    sampled_users: list[str] = []
    train_frames: list[pd.DataFrame] = []
    pool_rows: list[dict] = []
    relevants: dict[str, set[str]] = {}

    for user_id in eligible:
        if len(sampled_users) >= n_users:
            break
        user_df = interactions[interactions["user_id"] == user_id]
        train, test = temporal_split(user_df, test_frac)
        if len(train) < min_train_interactions:
            continue

        rel_items = set(
            test.loc[test["action"].isin(STRONG_POSITIVE_ACTIONS), "item_id"].astype(str)
        )
        # A relevant item the user already engaged with in training is not a
        # meaningful "future" recommendation target — drop those.
        train_items = set(train["item_id"].astype(str))
        rel_items -= train_items
        if len(rel_items) < min_relevant:
            continue

        # Popularity-sampled negatives: items the user never touched at all.
        user_all_items = set(user_df["item_id"].astype(str))
        forbidden = user_all_items | rel_items
        neg_candidates = np.array([i for i in all_item_ids if i not in forbidden])
        if len(neg_candidates) == 0:
            continue
        n_neg = min(n_negatives, len(neg_candidates))
        if negative_sampling == "popularity":
            w = pop_weights.reindex(neg_candidates).fillna(1.0).to_numpy(dtype=float)
            w = w / w.sum()
            negatives = rng.choice(neg_candidates, size=n_neg, replace=False, p=w)
        else:  # uniform
            negatives = rng.choice(neg_candidates, size=n_neg, replace=False)

        sampled_users.append(str(user_id))
        train_frames.append(train)
        relevants[str(user_id)] = rel_items
        for it in rel_items:
            pool_rows.append({"user_id": str(user_id), "item_id": it, "label": 1})
        for it in negatives:
            pool_rows.append({"user_id": str(user_id), "item_id": str(it), "label": 0})

    if not sampled_users:
        raise RuntimeError(
            "No users met the sampling criteria; loosen min_train_interactions/min_relevant."
        )

    train_interactions = pd.concat(train_frames, ignore_index=True)
    pool = pd.DataFrame(pool_rows)
    sampled_user_profiles = (
        users[users["user_id"].astype(str).isin(sampled_users)].reset_index(drop=True)
    )

    return BenchmarkData(
        items=items,
        users=sampled_user_profiles,
        train_interactions=train_interactions,
        full_train_interactions=full_train_interactions,
        pool=pool,
        relevants=relevants,
        user_ids=sampled_users,
    )
