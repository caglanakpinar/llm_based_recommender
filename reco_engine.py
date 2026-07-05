"""Named reco-engine registry, parquet loading, and build helpers."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from core.configs import Configs
from core.rag import LangChainRAG
from embedding_store import (
    EmbeddingStore,
    collect_user_records,
    load_default_item_catalog,
    load_default_llminput,
    load_default_user_profile,
    render_prompt,
)

REQUIRED_INTERACTION_COLUMNS = frozenset({"user_id", "item_id", "action", "timestamp"})
OPTIONAL_INTERACTION_COLUMNS = frozenset({"value", "session_id", "context"})
USER_PROFILE_OPTIONAL_FIELDS = frozenset({"segment", "notes"})
ITEM_CATALOG_OPTIONAL_FIELDS = frozenset({"tags", "price", "description"})
PARQUET_DATASET_KINDS = ("interactions", "users", "items")
ENGINE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
DRAFT_UPLOADS_DIR = Configs.current_dir / "store" / "uploads" / "draft"


def validate_engine_name(name: str) -> str:
    name = name.strip()
    if not name or not ENGINE_NAME_PATTERN.match(name):
        raise ValueError(
            "Engine name must be 1–64 chars: letters, numbers, underscore, hyphen."
        )
    return name


def engine_path(name: str) -> Path:
    return Configs.current_dir / Configs.project_name_for(validate_engine_name(name))


def list_engines() -> list[str]:
    return Configs.list_available_engines()


def engine_exists(name: str) -> bool:
    return (engine_path(name) / "docs" / "configs.yaml").is_file()


def normalize_llminput(llminput: dict[str, Any]) -> dict[str, Any]:
    """Migrate legacy llminput keys to current schema."""
    data = dict(llminput)
    if "target_user_profile" in data and "user_profile" not in data:
        data["user_profile"] = data.pop("target_user_profile")
    if "target_user_id" in data:
        uid = data.pop("target_user_id")
        profile = dict(data.get("user_profile") or {})
        profile.setdefault("user_id", uid)
        data["user_profile"] = profile
    return data


def get_engine_configs(name: str) -> Configs:
    if not engine_exists(name):
        raise FileNotFoundError(f"Reco engine not found: {name}")
    return Configs.from_engine_name(validate_engine_name(name))


def load_engine_meta(name: str) -> dict[str, Any]:
    configs = get_engine_configs(name)
    return {
        "name": configs.reco_engine_name,
        "created_at": configs.created_at,
        "interactions_parquet_folder": configs.interactions_parquet_folder,
        "users_parquet_folder": configs.users_parquet_folder,
        "items_parquet_folder": configs.items_parquet_folder,
        "parquet_folder": configs.interactions_parquet_folder,
        "top_k": configs.top_k,
        "user_id": configs.user_id,
        "num_items": configs.num_items,
        "num_users": configs.num_users,
    }


def load_engine_llminput(name: str) -> dict[str, Any]:
    configs = get_engine_configs(name)
    with configs.llminput_path.open(encoding="utf-8") as f:
        return normalize_llminput(json.load(f))


def load_rendered_prompt(name: str) -> str:
    configs = get_engine_configs(name)
    if configs.rendered_prompt_path.exists():
        return configs.rendered_prompt_path.read_text(encoding="utf-8")
    return render_prompt(load_engine_llminput(name), configs)


def delete_engine(name: str) -> None:
    path = engine_path(name)
    if path.exists():
        shutil.rmtree(path)


def _normalize_interaction_row(row: dict[str, Any]) -> dict[str, Any]:
    event: dict[str, Any] = {
        "timestamp": str(row["timestamp"]),
        "item_id": str(row["item_id"]),
        "action": str(row["action"]),
    }
    for col in OPTIONAL_INTERACTION_COLUMNS:
        if col in row and pd.notna(row[col]):
            value = row[col]
            if col == "value":
                try:
                    value = int(value) if float(value).is_integer() else float(value)
                except (TypeError, ValueError):
                    value = value
            event[col] = value
    return event


def _parse_tags(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in text.split(",") if part.strip()]


def _require_columns(
    df: pd.DataFrame,
    columns: dict[str, str],
    label: str,
    *,
    optional_fields: frozenset[str] | None = None,
) -> None:
    skip = optional_fields or frozenset()
    missing = [
        col
        for field, col in columns.items()
        if col and field not in skip and col not in df.columns
    ]
    if missing:
        raise ValueError(f"Parquet data missing {label} columns: {sorted(missing)}")


def extract_user_profiles(
    df: pd.DataFrame,
    columns: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Build user_id -> profile map from user parquet column mapping."""
    _require_columns(df, columns, "user profile", optional_fields=USER_PROFILE_OPTIONAL_FIELDS)
    uid_col = columns["user_id"]
    seg_col = columns.get("segment") or ""
    notes_col = columns.get("notes") or ""

    profiles: dict[str, dict[str, Any]] = {}
    for _, row in df.drop_duplicates(subset=[uid_col]).iterrows():
        uid = str(row[uid_col])
        profile: dict[str, Any] = {"user_id": uid}
        if seg_col and seg_col in df.columns and pd.notna(row.get(seg_col)):
            profile["segment"] = str(row[seg_col])
        if notes_col and notes_col in df.columns and pd.notna(row.get(notes_col)):
            profile["notes"] = str(row[notes_col])
        profiles[uid] = profile
    return profiles


def extract_item_catalog(
    df: pd.DataFrame,
    columns: dict[str, str],
) -> list[dict[str, Any]]:
    """Build item catalog from item parquet column mapping."""
    _require_columns(
        df, columns, "item catalog", optional_fields=ITEM_CATALOG_OPTIONAL_FIELDS
    )
    item_id_col = columns["item_id"]
    parquet_cols = [item_id_col] + [
        col for field, col in columns.items()
        if field != "item_id" and col and col in df.columns
    ]
    subset = df[parquet_cols].drop_duplicates(subset=[item_id_col])

    items: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        item: dict[str, Any] = {"item_id": str(row[item_id_col])}
        for field, col in columns.items():
            if field == "item_id" or not col or col not in df.columns:
                continue
            if pd.isna(row.get(col)):
                continue
            value = row[col]
            if field == "tags":
                item[field] = _parse_tags(value)
            elif field == "price":
                try:
                    item[field] = int(value) if float(value).is_integer() else float(value)
                except (TypeError, ValueError):
                    item[field] = value
            else:
                item[field] = str(value)
        items.append(item)
    return items


def _load_parquet_folder(folder_path: str, label: str) -> pd.DataFrame:
    folder = Path(folder_path).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    parquet_files = sorted(folder.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No .parquet files found in {folder} ({label})")

    frames = [pd.read_parquet(file) for file in parquet_files]
    return pd.concat(frames, ignore_index=True)


def load_parquet_interactions(folder_path: str) -> pd.DataFrame:
    """Load and validate interaction parquet files from a folder."""
    df = _load_parquet_folder(folder_path, "interactions")
    missing = REQUIRED_INTERACTION_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Interaction parquet missing required columns: {sorted(missing)}. "
            f"Required: {sorted(REQUIRED_INTERACTION_COLUMNS)}"
        )
    return df


def load_parquet_users(folder_path: str) -> pd.DataFrame:
    """Load user profile parquet files from a folder."""
    return _load_parquet_folder(folder_path, "users")


def load_parquet_items(folder_path: str) -> pd.DataFrame:
    """Load item catalog parquet files from a folder."""
    return _load_parquet_folder(folder_path, "items")


def list_parquet_user_ids(users_folder_path: str) -> list[str]:
    df = load_parquet_users(users_folder_path)
    uid_col = "user_id"
    if uid_col not in df.columns:
        raise ValueError(f"User parquet missing required column: {uid_col}")
    return sorted(df[uid_col].astype(str).unique().tolist())


def save_uploaded_parquet_files(
    uploaded_files: list[Any],
    subdir: str,
    dataset_kind: str = "interactions",
) -> Path:
    """Save uploaded parquet files under draft or engine uploads folder."""
    if dataset_kind not in PARQUET_DATASET_KINDS:
        raise ValueError(
            f"dataset_kind must be one of {PARQUET_DATASET_KINDS}, got {dataset_kind!r}"
        )
    if not uploaded_files:
        raise ValueError("No parquet files uploaded.")

    if subdir == "draft":
        dest = DRAFT_UPLOADS_DIR / dataset_kind
    else:
        engine_name = validate_engine_name(subdir)
        if engine_exists(engine_name):
            dest = get_engine_configs(engine_name).uploads_path / dataset_kind
        else:
            dest = engine_path(engine_name) / Configs.DEFAULT_UPLOADS_DIR / dataset_kind

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    for uploaded in uploaded_files:
        name = Path(uploaded.name).name
        if not name.lower().endswith(".parquet"):
            raise ValueError(f"Only .parquet files are allowed: {name}")
        (dest / name).write_bytes(uploaded.getvalue())

    if dataset_kind == "interactions":
        load_parquet_interactions(str(dest))
    elif dataset_kind == "users":
        load_parquet_users(str(dest))
    else:
        load_parquet_items(str(dest))
    return dest


def resolve_parquet_folder(
    *,
    upload_dir: str | None,
    manual_path: str | None,
    fallback_path: str | None = None,
) -> str | None:
    """Prefer browsed upload, then manual path, then existing engine path."""
    if upload_dir and Path(upload_dir).is_dir():
        return upload_dir
    if manual_path and manual_path.strip():
        return manual_path.strip()
    if fallback_path and fallback_path.strip():
        return fallback_path.strip()
    return None


def collect_all_user_records(
    profiles: dict[str, dict[str, Any]],
    interactions_folder: str,
) -> list[dict[str, Any]]:
    """Build one embedding record per user from profiles + interactions parquet."""
    df = load_parquet_interactions(interactions_folder)
    df["user_id"] = df["user_id"].astype(str)
    records: list[dict[str, Any]] = []
    for i, uid in enumerate(sorted(profiles.keys())):
        profile = profiles[uid]
        user_df = df[df["user_id"] == uid].sort_values("timestamp")
        events = [
            _normalize_interaction_row(row)
            for row in user_df.to_dict(orient="records")
        ]
        records.append(
            {
                "user_id": uid,
                "role": "target" if i == 0 else "peer",
                "profile": profile,
                "interactions": events,
            }
        )
    return records


def interactions_from_parquet(
    folder_path: str,
    target_user_id: str,
    user_profiles: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split parquet interactions into target user events and peer users."""
    df = load_parquet_interactions(folder_path)
    df["user_id"] = df["user_id"].astype(str)
    target_user_id = str(target_user_id)

    target_df = df[df["user_id"] == target_user_id].sort_values("timestamp")
    target_interactions = [
        _normalize_interaction_row(row)
        for row in target_df.to_dict(orient="records")
    ]

    peer_df = df[df["user_id"] != target_user_id]
    other_users: list[dict[str, Any]] = []
    for user_id, group in peer_df.groupby("user_id"):
        events = [
            _normalize_interaction_row(row)
            for row in group.sort_values("timestamp").to_dict(orient="records")
        ]
        peer: dict[str, Any] = {"user_id": str(user_id), "interactions": events}
        if user_profiles and str(user_id) in user_profiles:
            profile = user_profiles[str(user_id)]
            peer["segment"] = profile.get("segment")
            if profile.get("notes"):
                peer["notes"] = profile["notes"]
        other_users.append(peer)

    return target_interactions, other_users


def build_llminput_from_form(
    *,
    top_k: int,
    constraints: str,
    user_profile_columns: dict[str, str],
    item_catalog_columns: dict[str, str],
    interactions_parquet_folder: str,
    users_parquet_folder: str,
    items_parquet_folder: str,
    llm_chat: str = "",
    target_user_id: str | None = None,
) -> dict[str, Any]:
    """Assemble llminput from UI fields and separate parquet sources."""
    repo_defaults = Configs.from_engine_name("default")
    defaults = load_default_llminput(repo_defaults)

    users_df = load_parquet_users(users_parquet_folder)
    items_df = load_parquet_items(items_parquet_folder)
    profiles = extract_user_profiles(users_df, user_profile_columns)
    if not profiles:
        raise ValueError("User parquet contains no profiles.")

    item_catalog = extract_item_catalog(items_df, item_catalog_columns)

    if target_user_id is None:
        target_user_id = sorted(profiles.keys())[0]
    target_user_id = str(target_user_id)
    if target_user_id not in profiles:
        raise ValueError(
            f"Target user '{target_user_id}' not found in user parquet "
            f"(column {user_profile_columns['user_id']})."
        )

    profile = profiles[target_user_id]
    target_ix, other_ix = interactions_from_parquet(
        interactions_parquet_folder, target_user_id, profiles
    )

    return {
        "top_k": top_k,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "allowed_actions": defaults.get("allowed_actions", ""),
        "action_weights": defaults.get("action_weights", ""),
        "constraints": constraints,
        "user_profile": profile,
        "user_profile_columns": user_profile_columns,
        "item_catalog_columns": item_catalog_columns,
        "target_user_interactions": target_ix,
        "other_users_interactions": other_ix,
        "item_catalog": item_catalog,
        "llm_chat": llm_chat.strip(),
    }


def llminput_for_target_user(
    *,
    target_user_id: str,
    base_llminput: dict[str, Any],
    interactions_parquet_folder: str,
    users_parquet_folder: str,
) -> dict[str, Any]:
    """Rebuild user-specific llminput fields for a chosen target user at runtime."""
    users_df = load_parquet_users(users_parquet_folder)
    user_profile_columns = base_llminput.get(
        "user_profile_columns", Configs.DEFAULT_USER_PROFILE_COLUMNS
    )
    profiles = extract_user_profiles(users_df, user_profile_columns)
    target_user_id = str(target_user_id)
    if target_user_id not in profiles:
        raise ValueError(
            f"Target user '{target_user_id}' not found in user parquet "
            f"(column {user_profile_columns['user_id']})."
        )

    target_ix, other_ix = interactions_from_parquet(
        interactions_parquet_folder, target_user_id, profiles
    )
    merged = dict(base_llminput)
    merged["user_profile"] = profiles[target_user_id]
    merged["target_user_interactions"] = target_ix
    merged["other_users_interactions"] = other_ix
    merged["generated_at"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return merged


def copy_parquet_datasets_to_engine(
    engine_root: Path,
    interactions_parquet_folder: str,
    users_parquet_folder: str,
    items_parquet_folder: str,
) -> None:
    """Copy parquet dataset files from source folders to engine's data folder."""
    data_dir = engine_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for each dataset type
    interactions_dest = data_dir / "interactions"
    users_dest = data_dir / "users"
    items_dest = data_dir / "items"

    for dest_dir in [interactions_dest, users_dest, items_dest]:
        dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy parquet files from source folders
    def copy_parquet_files(src_folder: str, dest_folder: Path) -> None:
        src_path = Path(src_folder)
        if src_path.exists() and src_path.is_dir():
            for parquet_file in src_path.glob("*.parquet"):
                dest_file = dest_folder / parquet_file.name
                shutil.copy2(parquet_file, dest_file)
                print(f"Copied {parquet_file.name} to {dest_folder.relative_to(engine_root)}/")

    copy_parquet_files(interactions_parquet_folder, interactions_dest)
    copy_parquet_files(users_parquet_folder, users_dest)
    copy_parquet_files(items_parquet_folder, items_dest)

    print(f"✓ Parquet datasets copied to {data_dir.relative_to(engine_root)}/")


def build_engine(
    name: str,
    llminput: dict[str, Any],
    *,
    interactions_parquet_folder: str,
    users_parquet_folder: str,
    items_parquet_folder: str,
) -> EmbeddingStore:
    """Build and persist a named reco engine under {name}_reco_engine/."""
    name = validate_engine_name(name)
    profile = llminput.get("user_profile") or {}

    configs = Configs.create(
        name,
        top_k=llminput.get("top_k", 3),
        constraints=llminput.get("constraints", ""),
        llm_chat=llminput.get("llm_chat", ""),
        user_profile_columns=llminput.get(
            "user_profile_columns", Configs.DEFAULT_USER_PROFILE_COLUMNS
        ),
        item_catalog_columns=llminput.get(
            "item_catalog_columns", Configs.DEFAULT_ITEM_CATALOG_COLUMNS
        ),
        interactions_parquet_folder=str(Path(interactions_parquet_folder).resolve()),
        users_parquet_folder=str(Path(users_parquet_folder).resolve()),
        items_parquet_folder=str(Path(items_parquet_folder).resolve()),
        user_id=profile.get("user_id"),
    )

    # Copy parquet datasets to engine's data folder
    copy_parquet_datasets_to_engine(
        configs.engine_root,
        interactions_parquet_folder,
        users_parquet_folder,
        items_parquet_folder,
    )

    store = EmbeddingStore(configs)
    items = llminput["item_catalog"]
    users_df = load_parquet_users(users_parquet_folder)
    profiles = extract_user_profiles(
        users_df, llminput.get("user_profile_columns", Configs.DEFAULT_USER_PROFILE_COLUMNS)
    )
    all_user_records = collect_all_user_records(profiles, interactions_parquet_folder)

    item_vectors = store.build_item_embeddings(items)
    user_vectors = store.build_user_embeddings_from_records(all_user_records)
    default_records, default_vectors = store.build_default_prompt_embeddings(llminput)
    rendered = render_prompt(llminput, configs)

    store.save(
        items,
        item_vectors,
        all_user_records,
        user_vectors,
        default_records,
        default_vectors,
        llminput,
        rendered,
    )

    with configs.llminput_path.open("w", encoding="utf-8") as f:
        json.dump(llminput, f, indent=2, ensure_ascii=False)

    configs.num_items = len(items)
    configs.num_users = len(all_user_records)
    configs.user_id = profile.get("user_id")
    configs.interactions_parquet_folder = str(Path(interactions_parquet_folder).resolve())
    configs.users_parquet_folder = str(Path(users_parquet_folder).resolve())
    configs.items_parquet_folder = str(Path(items_parquet_folder).resolve())
    configs.save()

    # Save rendered prompt to engine folder
    engine_prompt_path = configs.engine_root / "rendered_prompt.md"
    engine_prompt_path.write_text(rendered, encoding="utf-8")
    print(f"✓ Saved rendered prompt → {engine_prompt_path}")

    return store


def get_engine_store(name: str) -> EmbeddingStore:
    return EmbeddingStore(get_engine_configs(name))


def default_item_catalog_json() -> str:
    return json.dumps(load_default_item_catalog(), indent=2, ensure_ascii=False)


def default_user_profile_json() -> str:
    return json.dumps(load_default_user_profile(), indent=2, ensure_ascii=False)


class RAGRecommender(LangChainRAG):
    """
    Recommender engine that uses RAG (Retrieval-Augmented Generation) to provide recommendations.
    Uses embedding vectors from store/embeddings/ to find and rank recommendations
    without needing an external LLM API.
    """
    def __init__(self, model: str = "scratch-model", engine_name: str = "default"):
        super().__init__(api_key=None, model=model)
        self.engine_name = engine_name
        self._embedding_store = None
        self.prompt_template = self.load_prompt_template()

    def load_prompt_template(self) -> str:
        configs = Configs.from_engine_name(self.engine_name)
        return configs.prompt_path_resolved.read_text(encoding="utf-8")

    def generate_llm_prompt(self, result: dict[str, Any], llminput: dict[str, Any] | None = None) -> str:
        import json

        configs = Configs.from_engine_name(self.engine_name)
        payload = dict(load_default_llminput(configs))
        if llminput:
            payload.update(llminput)
        payload["llm_chat"] = result_to_text(result)

        template = self.prompt_template
        skip_keys = frozenset({"user_profile_columns", "item_catalog_columns"})
        for key, value in payload.items():
            if key in skip_keys:
                continue
            if isinstance(value, (dict, list)):
                replacement = json.dumps(value, indent=2, ensure_ascii=False)
            else:
                replacement = str(value) if value is not None else ""
            template = template.replace("{" + key + "}", replacement)
        return template
    
    def get_provider(self):
        """Determine the LLM provider based on available API keys."""
        hf_token = self.api_key or self.get_hf_token()
        if hf_token:
            return "huggingface"
        return "ollama"

    def _get_llm_caller(self) -> BaseLLM:
        if self.api_key:
            return GPTCaller(api_key=self.api_key, model=self.model)
        return FreeLLMCaller(model=self.model, provider="huggingface") # self.get_provider())

    def call_llm(self, prompt: str) -> str:
        caller = self._get_llm_caller()
        return caller.call(prompt)

    @property
    def embedding_store(self):
        """Lazy load EmbeddingStore."""

