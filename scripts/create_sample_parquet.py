"""Create sample parquet datasets for testing (interactions, users, items)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INTERACTIONS_OUT = ROOT / "data" / "sample_interactions"
USERS_OUT = ROOT / "data" / "sample_users"
ITEMS_OUT = ROOT / "data" / "sample_items"

NUM_USERS = 200
NUM_ITEMS = 100
# Interactions are generated per user (see below) so every user has candidates;
# these bounds give ~3k total interactions across the sample.
MIN_INTERACTIONS_PER_USER = 8
MAX_INTERACTIONS_PER_USER = 22
ROWS_PER_FILE = 50_000
SEED = 42

CATEGORIES = ["electronics", "books", "travel", "home", "sports", "fashion"]
SEGMENTS = ["new_customer", "returning_customer", "vip", "churn_risk"]
ACTIONS = ["view", "click", "add_to_cart", "purchase", "like", "dislike"]
ACTION_WEIGHTS = np.array([0.35, 0.25, 0.15, 0.10, 0.10, 0.05])

rng = np.random.default_rng(SEED)

user_ids = [f"user_{i:03d}" for i in range(1, NUM_USERS + 1)]
item_ids = [f"item_{i:03d}" for i in range(1, NUM_ITEMS + 1)]

user_segments = rng.choice(SEGMENTS, size=NUM_USERS)
user_notes = [
    f"Sample notes for {uid} in segment {seg}."
    for uid, seg in zip(user_ids, user_segments)
]

item_titles = [f"Product {iid}" for iid in item_ids]
item_categories = rng.choice(CATEGORIES, size=NUM_ITEMS)
item_tags = [
    json.dumps([cat, "sample", f"tag{i % 7}"])
    for i, cat in enumerate(item_categories)
]
item_prices = np.round(rng.uniform(5, 250, size=NUM_ITEMS), 2)
item_descriptions = [
    f"A {cat} product for testing recommendations."
    for cat in item_categories
]

# Generate interactions per user so every user_id has interactions (and therefore
# retrieval candidates). Each user gets a random count of distinct items.
inter_user_ids: list[str] = []
inter_item_ids: list[str] = []
for uid in user_ids:
    n = int(rng.integers(MIN_INTERACTIONS_PER_USER, MAX_INTERACTIONS_PER_USER + 1))
    n = min(n, NUM_ITEMS)
    chosen = rng.choice(item_ids, size=n, replace=False)
    inter_user_ids.extend([uid] * n)
    inter_item_ids.extend(chosen.tolist())

NUM_INTERACTIONS = len(inter_user_ids)
actions = rng.choice(ACTIONS, size=NUM_INTERACTIONS, p=ACTION_WEIGHTS)

base_ts = np.datetime64("2026-01-01T00:00:00")
offsets = rng.integers(0, 180 * 24 * 3600, size=NUM_INTERACTIONS)
timestamps = base_ts + offsets.astype("timedelta64[s]")
timestamp_str = pd.to_datetime(timestamps).strftime("%Y-%m-%dT%H:%M:%SZ")

interactions_df = pd.DataFrame(
    {
        "user_id": inter_user_ids,
        "item_id": inter_item_ids,
        "action": actions,
        "timestamp": timestamp_str,
    }
)
interactions_df["value"] = np.where(
    rng.random(NUM_INTERACTIONS) < 0.08, rng.integers(1, 6, NUM_INTERACTIONS), None
)
interactions_df["session_id"] = np.where(
    rng.random(NUM_INTERACTIONS) < 0.6,
    [f"sess_{i // 3}" for i in range(NUM_INTERACTIONS)],
    None,
)
interactions_df["context"] = np.where(
    rng.random(NUM_INTERACTIONS) < 0.4,
    rng.choice(["search", "homepage", "email", "recommendations"], NUM_INTERACTIONS),
    None,
)

users_df = pd.DataFrame(
    {
        "user_id": user_ids,
        "segment": user_segments,
        "notes": user_notes,
    }
)

items_df = pd.DataFrame(
    {
        "item_id": item_ids,
        "title": item_titles,
        "category": item_categories,
        "tags": item_tags,
        "price": item_prices,
        "description": item_descriptions,
    }
)


def _write_chunks(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    # Remove any previously generated files (old chunks, stray one-off parquet)
    # so the folder only ever holds the current, format-consistent sample.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for file_idx, start in enumerate(range(0, len(df), ROWS_PER_FILE)):
        chunk = df.iloc[start : start + ROWS_PER_FILE]
        path = out_dir / f"{prefix}_{file_idx + 1:02d}.parquet"
        chunk.to_parquet(path, index=False)
        print(f"Wrote {path} ({len(chunk):,} rows)")


_write_chunks(interactions_df, INTERACTIONS_OUT, "interactions")
_write_chunks(users_df, USERS_OUT, "users")
_write_chunks(items_df, ITEMS_OUT, "items")

print(
    f"\nTotal: {NUM_INTERACTIONS:,} interactions · "
    f"{NUM_USERS:,} users · {NUM_ITEMS:,} items\n"
    f"  interactions -> {INTERACTIONS_OUT}\n"
    f"  users        -> {USERS_OUT}\n"
    f"  items        -> {ITEMS_OUT}"
)
