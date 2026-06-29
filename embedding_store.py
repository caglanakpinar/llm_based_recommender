"""
Embed item catalog + default llminput prompt placeholders from recommender_prompt.md.
Persists vectors and default prompt results under store/.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from core.configs import Configs

ROOT = Path(__file__).resolve().parent
HF_CACHE_DIR = ROOT / ".cache" / "huggingface"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Backward-compatible module-level aliases (defaults from Configs).
DATA_DIR = Configs.current_dir / Configs.DEFAULT_DATA_DIR
STORE_DIR = Configs.current_dir / Configs.DEFAULT_STORE_DIR
ENGINES_DIR = Configs.current_dir / Configs.DEFAULT_ENGINES_DIR
PROMPT_PATH = Configs.current_dir / Configs.DEFAULT_PROMPT_PATH
MODEL_NAME = Configs.DEFAULT_MODEL_NAME
PROMPT_PLACEHOLDER_SECTIONS = Configs.DEFAULT_PROMPT_PLACEHOLDER_SECTIONS
DEFAULT_LLMINPUT_KEYS = Configs.DEFAULT_LLMINPUT_KEYS

_tokenizer: AutoTokenizer | None = None
_model: AutoModel | None = None


def _load_model() -> tuple[AutoTokenizer, AutoModel]:
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        cache_dir = str(HF_CACHE_DIR / "models")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
        _model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
        _model.eval()
    return _tokenizer, _model


def get_embedding(text_list: str | list[str]) -> np.ndarray:
    """Process text(s) into L2-normalized embeddings."""
    if isinstance(text_list, str):
        text_list = [text_list]

    tokenizer, model = _load_model()
    encoded_input = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )
    with torch.no_grad():
        model_output = model(**encoded_input)

    token_embeddings = model_output[0]
    attention_mask = encoded_input["attention_mask"]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sentence_embeddings = torch.sum(token_embeddings * mask, 1) / torch.clamp(
        mask.sum(1), min=1e-9
    )
    return F.normalize(sentence_embeddings, p=2, dim=1).cpu().numpy()


def item_to_text(item: dict[str, Any]) -> str:
    """Build embeddable text from recommender_prompt.md item schema."""
    tags = ", ".join(item.get("tags") or [])
    return (
        f"Title: {item.get('title', '')}. "
        f"Category: {item.get('category', '')}. "
        f"Tags: {tags}. "
        f"Price: {item.get('price', '')}. "
        f"Description: {item.get('description', '')}"
    )


def user_to_text(user: dict[str, Any]) -> str:
    """Build embeddable text from user profile JSON + interactions."""
    profile = user.get("profile") or {}
    events = "; ".join(_interaction_to_text(e) for e in user.get("interactions") or [])

    parts = [
        f"User {user.get('user_id')} ({user.get('role', 'user')})",
        f"profile {json.dumps(profile, ensure_ascii=False)}",
    ]
    if events:
        parts.append(f"interactions: {events}")

    return ". ".join(parts) + "."


def result_to_text(result: dict[str, Any]) -> str:
    """Build a text prompt from a recommendation result for embedding."""
    lines = [
        f"Method: {result.get('method', '')}",
        f"Engine: {result.get('engine', '')}",
        f"Item search results: {result.get('item_search_results', 0)}",
        f"Query similarity results: {result.get('query_similarity', 0)}",
        "Recommendations:",
    ]

    for rec in result.get("recommendations", []):
        lines.append(
            f"Rank {rec.get('rank')} item {rec.get('item_id')} score {rec.get('score'):.4f}"
            f" reason {rec.get('reason', '')}"
        )

    if result.get("user_matches"):
        lines.append("User matches:")
        for user in result["user_matches"]:
            lines.append(
                f"User {user.get('user_id')} score {user.get('score'):.4f}"
            )

    return "\n".join(lines)


USER_OVERRIDE_KEYS = frozenset(
    {
        "user_profile",
        "target_user_interactions",
        "other_users_interactions",
    }
)


def collect_user_records(llminput: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build user records from llminput (target user + peers).
    Matches recommender_prompt.md sections 3–5.
    """
    users: list[dict[str, Any]] = []

    profile = dict(llminput.get("user_profile") or {})
    target_id = profile.get("user_id")
    if target_id:
        profile.setdefault("user_id", target_id)

    users.append(
        {
            "user_id": profile.get("user_id", "unknown"),
            "role": "target",
            "profile": profile,
            "interactions": list(llminput.get("target_user_interactions") or []),
        }
    )

    for peer in llminput.get("other_users_interactions") or []:
        peer_profile: dict[str, Any] = {"user_id": peer["user_id"]}
        if peer.get("segment"):
            peer_profile["segment"] = peer["segment"]
        if peer.get("notes"):
            peer_profile["notes"] = peer["notes"]
        users.append(
            {
                "user_id": peer["user_id"],
                "role": "peer",
                "profile": peer_profile,
                "interactions": list(peer.get("interactions") or []),
            }
        )

    return users


def _user_record_affected(role: str, override_keys: set[str]) -> bool:
    if role == "target":
        return bool(override_keys & USER_OVERRIDE_KEYS - {"other_users_interactions"})
    if role == "peer":
        return "other_users_interactions" in override_keys
    return False


def build_merged_user_records(
    merged_llminput: dict[str, Any],
    default_user_records: list[dict[str, Any]],
    override_keys: set[str],
) -> list[dict[str, Any]]:
    """
    Hybrid user embeddings:
    - overridden / new users -> freshly embedded
    - rest -> reuse default vectors from user_records.json
    """
    fresh_users = collect_user_records(merged_llminput)
    default_by_id = {r["user_id"]: r for r in default_user_records}

    merged: list[dict[str, Any]] = []
    embed_texts: list[str] = []
    embed_positions: list[int] = []

    for user in fresh_users:
        uid = user["user_id"]
        role = user["role"]
        row: dict[str, Any] = {
            "user_id": uid,
            "role": role,
            "embedding_text": user_to_text(user),
            "user": user,
        }

        affected = _user_record_affected(role, override_keys) or uid not in default_by_id
        if affected or not default_by_id:
            row["vector_source"] = "generated"
            embed_positions.append(len(merged))
            embed_texts.append(row["embedding_text"])
        else:
            default_row = default_by_id[uid]
            row["vector_source"] = "default"
            row["vector"] = default_row["vector"]
            row["embedding_text"] = default_row["embedding_text"]
            row["user"] = default_row["user"]

        merged.append(row)

    if embed_texts:
        fresh_vectors = get_embedding(embed_texts)
        for pos, vector in zip(embed_positions, fresh_vectors):
            merged[pos]["vector"] = vector.tolist()

    for i, row in enumerate(merged):
        row["index"] = i

    return merged


def extract_default_prompt_header(template_path: Path = PROMPT_PATH) -> str:
    """
    Static header from recommender_prompt.md (lines 1–16):
    title, intent, engine description, and goals — before {action_weights}.
    """
    text = template_path.read_text(encoding="utf-8")
    match = re.search(
        r"(# LLM Recommender System[\s\S]*?)(?=\n\{action_weights\})",
        text,
    )
    if match:
        return match.group(1).strip()
    return "# LLM Recommender System — Prompt Template"


def _interaction_to_text(event: dict[str, Any]) -> str:
    parts = [
        f"at {event.get('timestamp', '')}",
        f"user action {event.get('action', '')}",
        f"on item {event.get('item_id', '')}",
    ]
    if event.get("value") is not None:
        parts.append(f"value {event['value']}")
    if event.get("context"):
        parts.append(f"context {event['context']}")
    return ", ".join(parts)


def llminput_value_to_text(
    key: str,
    value: Any,
    items: list[dict[str, Any]] | None = None,
    *,
    configs: Configs | None = None,
) -> str:
    """Convert a single llminput placeholder value into embeddable text."""
    sections = (configs or Configs.from_engine_name("default")).prompt_placeholder_sections
    section = sections.get(key, "llminput")
    prefix = f"[{section}] placeholder {{{key}}}:"

    if key == "user_profile" and isinstance(value, dict):
        return f"{prefix} {json.dumps(value, ensure_ascii=False)}"

    if key == "target_user_interactions" and isinstance(value, list):
        events = "; ".join(_interaction_to_text(e) for e in value)
        return f"{prefix} Target user history: {events}"

    if key == "other_users_interactions" and isinstance(value, list):
        chunks = []
        for peer in value:
            uid = peer.get("user_id", "")
            events = "; ".join(_interaction_to_text(e) for e in peer.get("interactions", []))
            chunks.append(f"user {uid}: {events}")
        return f"{prefix} Other users: {' | '.join(chunks)}"

    if key == "item_catalog" and isinstance(value, list):
        titles = ", ".join(item.get("title", item.get("item_id", "")) for item in value)
        return f"{prefix} Catalog summary ({len(value)} items): {titles}"

    if key == "top_k":
        return (
            f"{prefix} Generate {value} personalized recommendations for the Target User."
        )

    if key == "llm_chat":
        return f"{prefix} Additional instructions: {value}"

    if isinstance(value, (dict, list)):
        return f"{prefix} {json.dumps(value, ensure_ascii=False)}"

    return f"{prefix} {value}"


def merge_llminput(
    base: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], set[str]]:
    """Merge user overrides onto defaults. Returns (merged llminput, override keys)."""
    merged = dict(base)
    keys: set[str] = set()
    if not overrides:
        return merged, keys

    for key, value in overrides.items():
        merged[key] = value
        keys.add(key)

    return merged, keys


def parse_llminput_text(text: str) -> dict[str, Any]:
    """
    Parse partial llminput from natural text, e.g.:
      "top_k is 5, user_profile is {\"user_id\": \"user_999\", \"segment\": \"vip\"}"
      "placeholder top_k is 5"
    """
    overrides: dict[str, Any] = {}
    pattern = re.compile(
        r"(?:placeholder\s+)?(?P<key>[a-z_]+)\s+is\s+(?P<value>.+?)"
        r"(?=(?:\s*,\s*(?:placeholder\s+)?[a-z_]+\s+is\s+)|$)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text.strip()):
        key = match.group("key").lower()
        raw = match.group("value").strip().strip('"').strip("'")
        overrides[key] = _coerce_llminput_value(key, raw)
    return overrides


def _coerce_llminput_value(key: str, raw: str) -> Any:
    if key in {"top_k", "rank", "price", "price_min", "price_max", "value"}:
        if raw.isdigit():
            return int(raw)
        try:
            return float(raw)
        except ValueError:
            pass
    if raw.startswith("{") or raw.startswith("["):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw


def _record_affected_by_overrides(record: dict[str, Any], override_keys: set[str]) -> bool:
    if not override_keys:
        return False
    if record.get("type") == "static":
        return (
            record.get("record_id") == "default:recommendation_intent"
            and "user_profile" in override_keys
        )
    placeholder = record.get("placeholder")
    if placeholder == "item_catalog":
        return "item_catalog" in override_keys
    return bool(placeholder and placeholder in override_keys)


def _rebuild_record_from_llminput(
    record: dict[str, Any],
    merged_llminput: dict[str, Any],
    configs: Configs,
) -> dict[str, Any]:
    """Rebuild one prompt chunk using merged llminput values."""
    items = merged_llminput.get("item_catalog") or []
    updated = dict(record)

    if record.get("record_id") == "default:recommendation_intent":
        profile = merged_llminput.get("user_profile") or {}
        uid = profile.get("user_id", "")
        updated["embedding_text"] = (
            "LLM Recommender System. Return personalized recommendations for the "
            f"Target User {uid} using action history, "
            "peer interactions, item catalog, and engine rules."
        )
        updated["value"] = updated["embedding_text"]
        return updated

    placeholder = record.get("placeholder")
    if not placeholder:
        return updated

    if placeholder == "item_catalog":
        value = items
        updated["value"] = {
            "num_items": len(items),
            "item_ids": [i["item_id"] for i in items],
        }
    else:
        value = merged_llminput[placeholder]
        updated["value"] = value

    updated["embedding_text"] = llminput_value_to_text(
        placeholder, value, items, configs=configs
    )
    return updated


def build_merged_prompt_records(
    merged_llminput: dict[str, Any],
    default_records: list[dict[str, Any]],
    override_keys: set[str],
    configs: Configs,
) -> list[dict[str, Any]]:
    """
    Build prompt embedding chunks:
    - placeholders present in override_keys -> freshly embedded
    - everything else -> reuse default vectors from store
    """
    merged: list[dict[str, Any]] = []
    embed_texts: list[str] = []
    embed_positions: list[int] = []

    for record in default_records:
        row = dict(record)
        if _record_affected_by_overrides(record, override_keys):
            row = _rebuild_record_from_llminput(record, merged_llminput, configs)
            row["vector_source"] = "generated"
            embed_positions.append(len(merged))
            embed_texts.append(row["embedding_text"])
        else:
            row["vector_source"] = "default"
            if "vector" not in row:
                raise ValueError(
                    f"Default vector missing for record {row.get('record_id')}. "
                    "Run: poetry run python embedding_store.py"
                )

        merged.append(row)

    if embed_texts:
        fresh_vectors = get_embedding(embed_texts)
        for pos, vector in zip(embed_positions, fresh_vectors):
            merged[pos]["vector"] = vector.tolist()

    for i, row in enumerate(merged):
        row["index"] = i

    return merged


def build_default_prompt_records(
    llminput: dict[str, Any],
    configs: Configs,
) -> list[dict[str, Any]]:
    """
    Build default embedding records from recommender_prompt.md + llminput placeholders.
    """
    items = llminput.get("item_catalog") or []
    records: list[dict[str, Any]] = []

    header_text = extract_default_prompt_header(configs.prompt_path_resolved)
    records.append(
        {
            "record_id": "default:prompt_header",
            "type": "static",
            "section": "1. Engine Description",
            "placeholder": None,
            "value": header_text,
            "embedding_text": header_text,
        }
    )

    records.append(
        {
            "record_id": "default:recommendation_intent",
            "type": "static",
            "section": "6. Request",
            "placeholder": None,
            "value": (
                "The model returns personalized recommendations for the Target User "
                "based on llminput placeholder values."
            ),
            "embedding_text": (
                "LLM Recommender System. Return personalized recommendations for the "
                f"Target User {(llminput.get('user_profile') or {}).get('user_id', '')} "
                "using action history, peer interactions, item catalog, and engine rules."
            ),
        }
    )

    for key in configs.default_llminput_keys:
        if key not in llminput:
            continue
        value = llminput[key]
        records.append(
            {
                "record_id": f"llminput:{key}",
                "type": "llminput",
                "section": configs.prompt_placeholder_sections.get(key, "llminput"),
                "placeholder": key,
                "value": value,
                "embedding_text": llminput_value_to_text(key, value, items, configs=configs),
            }
        )

    if "item_catalog" in llminput:
        records.append(
            {
                "record_id": "llminput:item_catalog_summary",
                "type": "llminput",
                "section": configs.prompt_placeholder_sections["item_catalog"],
                "placeholder": "item_catalog",
                "value": {"num_items": len(items), "item_ids": [i["item_id"] for i in items]},
                "embedding_text": llminput_value_to_text(
                    "item_catalog", items, items, configs=configs
                ),
            }
        )

    for i, record in enumerate(records):
        record["index"] = i

    return records


def load_default_item_catalog(configs: Configs | None = None) -> list[dict[str, Any]]:
    cfg = configs or Configs.from_engine_name("default")
    path = cfg.data_path / "default_item_catalog.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_default_user_profile(configs: Configs | None = None) -> dict[str, Any]:
    cfg = configs or Configs.from_engine_name("default")
    path = cfg.data_path / "default_user_profile.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_default_target_user_profile(configs: Configs | None = None) -> dict[str, Any]:
    """Backward-compatible alias."""
    return load_default_user_profile(configs)


def load_default_user_profiles(configs: Configs | None = None) -> list[dict[str, Any]]:
    cfg = configs or Configs.from_engine_name("default")
    path = cfg.data_path / "default_user_profiles.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_default_llminput(configs: Configs | None = None) -> dict[str, Any]:
    cfg = configs or Configs.from_engine_name("default")
    path = cfg.data_path / "default_llminput.json"
    with path.open(encoding="utf-8") as f:
        llminput = json.load(f)
    llminput["item_catalog"] = load_default_item_catalog(cfg)
    llminput["user_profile"] = load_default_user_profile(cfg)
    return llminput


def render_prompt(llminput: dict[str, Any], configs: Configs) -> str:
    """Fill {placeholder} tokens in recommender_prompt.md."""
    template = configs.prompt_path_resolved.read_text(encoding="utf-8")
    payload = dict(llminput)
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


def _write_faiss_index(vectors: np.ndarray, index_path: Path) -> None:
    import faiss

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    faiss.write_index(index, str(index_path))


class EmbeddingStore:
    """FAISS-backed store for item catalog + default llminput prompt embeddings."""

    def __init__(self, configs: Configs) -> None:
        self.configs = configs
        self.store_dir = configs.store_dir_path
        self.embeddings_dir = configs.embeddings_dir
        self.default_dir = configs.default_dir
        self.item_vectors_path = configs.item_vectors_path
        self.item_index_path = configs.item_index_path
        self.item_records_path = configs.item_records_path
        self.user_vectors_path = configs.user_vectors_path
        self.user_index_path = configs.user_index_path
        self.user_records_path = configs.user_records_path
        self.default_vectors_path = configs.default_vectors_path
        self.default_index_path = configs.default_index_path
        self.default_records_path = configs.default_records_path
        self.default_results_path = configs.default_results_path
        self.rendered_prompt_path = configs.rendered_prompt_path
        self.session_dir = configs.session_dir

    def load_default_prompt_records(self) -> list[dict[str, Any]]:
        if not self.default_records_path.exists():
            raise FileNotFoundError(
                f"No default prompt records at {self.default_records_path}. "
                "Run: poetry run python embedding_store.py"
            )
        with self.default_records_path.open(encoding="utf-8") as f:
            return json.load(f)

    def load_default_user_records(self) -> list[dict[str, Any]]:
        if not self.user_records_path.exists():
            raise FileNotFoundError(
                f"No user records at {self.user_records_path}. "
                "Run: poetry run python embedding_store.py"
            )
        with self.user_records_path.open(encoding="utf-8") as f:
            return json.load(f)

    def generate_prompt_embeddings(
        self,
        llminput_overrides: dict[str, Any] | None = None,
        llminput_text: str | None = None,
        *,
        base_llminput: dict[str, Any] | None = None,
        save_session: bool = True,
    ) -> dict[str, Any]:
        """
        Hybrid prompt embeddings for recommender_prompt.md:
        - chunks whose {placeholder} you fill -> newly generated vectors
        - all other chunks -> reused default vectors from store/embeddings/default/
        """
        if llminput_text:
            text_overrides = parse_llminput_text(llminput_text)
            llminput_overrides = {**(llminput_overrides or {}), **text_overrides}

        if base_llminput is None:
            results_path = self.default_results_path
            if results_path.exists():
                with results_path.open(encoding="utf-8") as f:
                    base_llminput = json.load(f).get("llminput") or load_default_llminput(
                        self.configs
                    )
            else:
                base_llminput = load_default_llminput(self.configs)

        merged_llminput, override_keys = merge_llminput(base_llminput, llminput_overrides)
        default_records = self.load_default_prompt_records()

        if override_keys:
            print(
                f"Generating embeddings for overrides: {', '.join(sorted(override_keys))}; "
                "reusing default vectors for the rest."
            )
        else:
            print("No overrides provided — using all default prompt vectors.")

        records = build_merged_prompt_records(
            merged_llminput, default_records, override_keys, self.configs
        )
        vectors = np.array([r["vector"] for r in records], dtype=np.float32)
        rendered_prompt = render_prompt(merged_llminput, self.configs)

        default_user_records = self.load_default_user_records()
        user_records = build_merged_user_records(
            merged_llminput, default_user_records, override_keys
        )
        user_vectors = np.array([r["vector"] for r in user_records], dtype=np.float32)

        user_override_keys = sorted(override_keys & set(USER_OVERRIDE_KEYS))
        if user_override_keys:
            print(
                f"Generating user embeddings for overrides: {', '.join(user_override_keys)}; "
                "reusing default user vectors for the rest."
            )

        result = {
            "llminput": merged_llminput,
            "override_keys": sorted(override_keys),
            "records": records,
            "vectors": vectors,
            "user_records": user_records,
            "user_vectors": user_vectors,
            "rendered_prompt": rendered_prompt,
            "generated_count": sum(1 for r in records if r["vector_source"] == "generated"),
            "default_count": sum(1 for r in records if r["vector_source"] == "default"),
            "user_generated_count": sum(
                1 for r in user_records if r["vector_source"] == "generated"
            ),
            "user_default_count": sum(
                1 for r in user_records if r["vector_source"] == "default"
            ),
        }

        if save_session:
            self._save_session_embeddings(result)

        return result

    def _save_session_embeddings(self, result: dict[str, Any]) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)

        vectors_path = self.session_dir / "prompt_vectors.npy"
        records_path = self.session_dir / "prompt_records.json"
        rendered_path = self.session_dir / "rendered_prompt.md"
        meta_path = self.session_dir / "session_meta.json"

        np.save(vectors_path, result["vectors"])
        with records_path.open("w", encoding="utf-8") as f:
            json.dump(result["records"], f, indent=2, ensure_ascii=False)
        rendered_path.write_text(result["rendered_prompt"], encoding="utf-8")

        user_vectors_path = self.session_dir / "user_vectors.npy"
        user_records_path = self.session_dir / "user_records.json"
        np.save(user_vectors_path, result["user_vectors"])
        with user_records_path.open("w", encoding="utf-8") as f:
            json.dump(result["user_records"], f, indent=2, ensure_ascii=False)
        _write_faiss_index(result["user_vectors"], self.session_dir / "user_faiss.index")

        meta = {
            "override_keys": result["override_keys"],
            "generated_count": result["generated_count"],
            "default_count": result["default_count"],
            "user_generated_count": result["user_generated_count"],
            "user_default_count": result["user_default_count"],
            "vectors_file": str(vectors_path.relative_to(ROOT)),
            "records_file": str(records_path.relative_to(ROOT)),
            "rendered_prompt_file": str(rendered_path.relative_to(ROOT)),
            "user_vectors_file": str(user_vectors_path.relative_to(ROOT)),
            "user_records_file": str(user_records_path.relative_to(ROOT)),
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        _write_faiss_index(result["vectors"], self.session_dir / "prompt_faiss.index")

        print(f"Saved session prompt vectors -> {vectors_path}")
        print(f"Saved session prompt records -> {records_path}")
        print(f"Saved session rendered prompt -> {rendered_path}")
        print(f"Saved session user vectors -> {user_vectors_path}")
        print(f"Saved session user records -> {user_records_path}")

    def build_item_embeddings(self, items: list[dict[str, Any]]) -> np.ndarray:
        texts = [item_to_text(item) for item in items]
        print(f"Embedding {len(texts)} catalog items...")
        return get_embedding(texts)

    def build_user_embeddings(self, llminput: dict[str, Any]) -> np.ndarray:
        users = collect_user_records(llminput)
        return self.build_user_embeddings_from_records(users)

    def build_user_embeddings_from_records(
        self, users: list[dict[str, Any]]
    ) -> np.ndarray:
        texts = [user_to_text(user) for user in users]
        print(f"Embedding {len(texts)} users...")
        return get_embedding(texts)

    def build_default_prompt_embeddings(
        self, llminput: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        records = build_default_prompt_records(llminput, self.configs)
        texts = [r["embedding_text"] for r in records]
        print(f"Embedding {len(texts)} default prompt records (llminput + header)...")
        vectors = get_embedding(texts)
        for record, vector in zip(records, vectors):
            record["vector"] = vector.tolist()
        return records, vectors

    def save(
        self,
        items: list[dict[str, Any]],
        item_vectors: np.ndarray,
        users: list[dict[str, Any]],
        user_vectors: np.ndarray,
        default_records: list[dict[str, Any]],
        default_vectors: np.ndarray,
        llminput: dict[str, Any],
        rendered_prompt: str,
    ) -> None:
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.default_dir.mkdir(parents=True, exist_ok=True)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        np.save(self.item_vectors_path, item_vectors)
        _write_faiss_index(item_vectors, self.item_index_path)

        item_records = [
            {
                "index": i,
                "item_id": item["item_id"],
                "embedding_text": item_to_text(item),
                "item": item,
                "vector": item_vectors[i].tolist(),
            }
            for i, item in enumerate(items)
        ]
        with self.item_records_path.open("w", encoding="utf-8") as f:
            json.dump(item_records, f, indent=2, ensure_ascii=False)

        np.save(self.user_vectors_path, user_vectors)
        _write_faiss_index(user_vectors, self.user_index_path)

        user_records = [
            {
                "index": i,
                "user_id": user["user_id"],
                "role": user["role"],
                "embedding_text": user_to_text(user),
                "user": user,
                "vector": user_vectors[i].tolist(),
            }
            for i, user in enumerate(users)
        ]
        with self.user_records_path.open("w", encoding="utf-8") as f:
            json.dump(user_records, f, indent=2, ensure_ascii=False)

        np.save(self.default_vectors_path, default_vectors)
        _write_faiss_index(default_vectors, self.default_index_path)
        with self.default_records_path.open("w", encoding="utf-8") as f:
            json.dump(default_records, f, indent=2, ensure_ascii=False)

        dimension = int(item_vectors.shape[1])
        repo_root = self.configs.current_dir
        model = self.configs.model_name
        default_results = {
            "llminput": llminput,
            "item_catalog": items,
            "item_embeddings": {
                "model": model,
                "dimension": dimension,
                "num_items": len(items),
                "vectors_file": str(self.item_vectors_path.relative_to(repo_root)),
                "faiss_index_file": str(self.item_index_path.relative_to(repo_root)),
                "records_file": str(self.item_records_path.relative_to(repo_root)),
            },
            "user_embeddings": {
                "model": model,
                "dimension": dimension,
                "num_users": len(users),
                "vectors_file": str(self.user_vectors_path.relative_to(repo_root)),
                "faiss_index_file": str(self.user_index_path.relative_to(repo_root)),
                "records_file": str(self.user_records_path.relative_to(repo_root)),
            },
            "default_prompt_embeddings": {
                "model": model,
                "dimension": dimension,
                "num_records": len(default_records),
                "vectors_file": str(self.default_vectors_path.relative_to(repo_root)),
                "faiss_index_file": str(self.default_index_path.relative_to(repo_root)),
                "records_file": str(self.default_records_path.relative_to(repo_root)),
                "placeholders": [
                    r["placeholder"] for r in default_records if r.get("placeholder")
                ],
            },
            "rendered_prompt": rendered_prompt,
        }
        with self.default_results_path.open("w", encoding="utf-8") as f:
            json.dump(default_results, f, indent=2, ensure_ascii=False)

        self.rendered_prompt_path.write_text(rendered_prompt, encoding="utf-8")

        print(f"Saved item vectors -> {self.item_vectors_path}")
        print(f"Saved item FAISS index -> {self.item_index_path}")
        print(f"Saved item records -> {self.item_records_path}")
        print(f"Saved user vectors -> {self.user_vectors_path}")
        print(f"Saved user FAISS index -> {self.user_index_path}")
        print(f"Saved user records -> {self.user_records_path}")
        print(f"Saved default prompt vectors -> {self.default_vectors_path}")
        print(f"Saved default prompt FAISS index -> {self.default_index_path}")
        print(f"Saved default prompt records -> {self.default_records_path}")
        print(f"Saved default prompt results -> {self.default_results_path}")
        print(f"Saved rendered prompt -> {self.rendered_prompt_path}")

    def _search_index(
        self,
        query: str,
        index_path: Path,
        records_path: Path,
        top_k: int,
        result_key: str,
    ) -> list[dict[str, Any]]:
        import faiss

        if not index_path.exists():
            raise FileNotFoundError(f"No index at {index_path}. Run: python embedding_store.py")

        index = faiss.read_index(str(index_path))
        with records_path.open(encoding="utf-8") as f:
            records = json.load(f)

        query_vector = get_embedding(query)
        scores, indices = index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            record = records[idx]
            hit = {
                "score": float(score),
                "index": int(idx),
                "embedding_text": record.get("embedding_text", ""),
            }
            if result_key == "item":
                hit["item_id"] = record["item_id"]
                hit["item"] = record["item"]
            elif result_key == "user":
                hit["user_id"] = record["user_id"]
                hit["role"] = record.get("role")
                hit["user"] = record["user"]
            else:
                hit["record_id"] = record.get("record_id")
                hit["placeholder"] = record.get("placeholder")
                hit["section"] = record.get("section")
                hit["type"] = record.get("type")
            results.append(hit)
        return results

    def search_items(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return self._search_index(
            query, self.item_index_path, self.item_records_path, top_k, "item"
        )

    def search_users(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return self._search_index(
            query, self.user_index_path, self.user_records_path, top_k, "user"
        )

    def search_default_prompt(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return self._search_index(
            query,
            self.default_index_path,
            self.default_records_path,
            top_k,
            "default",
        )

    def update_prompt_embeddings(
        self,
        new_llminput: dict[str, Any],
        save_to_store: bool = True,
    ) -> dict[str, Any]:
        """
        Dynamically update embeddings when new prompts arrive.
        Generates embeddings for the new llminput and updates store/embeddings/
        """
        print(f"📥 Updating embeddings with new prompt...")

        # Load existing records
        try:
            existing_records = self.load_default_prompt_records()
            existing_vectors = np.load(self.default_vectors_path)
        except FileNotFoundError:
            print("⚠️ No existing embeddings found. Starting fresh.")
            existing_records = []
            existing_vectors = None

        # Generate new embedding records from the new llminput
        new_records = build_default_prompt_records(new_llminput, self.configs)
        new_texts = [r["embedding_text"] for r in new_records]
        new_vectors = get_embedding(new_texts)

        # Merge: keep existing records + add new ones (deduplication by record_id)
        existing_by_id = {r["record_id"]: r for r in existing_records}
        
        for i, new_record in enumerate(new_records):
            record_id = new_record["record_id"]
            if record_id not in existing_by_id:
                # New record - add it
                new_record["index"] = len(existing_by_id)
                existing_by_id[record_id] = new_record
            else:
                # Update existing record with new vector and text
                existing_by_id[record_id]["embedding_text"] = new_record["embedding_text"]
                existing_by_id[record_id]["vector"] = new_vectors[i].tolist()

        # Rebuild vectors array from all records
        merged_records = sorted(existing_by_id.values(), key=lambda r: r.get("index", 0))
        merged_vectors = np.array([r["vector"] for r in merged_records], dtype=np.float32)

        # Update indexes
        for i, record in enumerate(merged_records):
            record["index"] = i

        # Save updated embeddings
        if save_to_store:
            self.embeddings_dir.mkdir(parents=True, exist_ok=True)
            self.default_dir.mkdir(parents=True, exist_ok=True)

            np.save(self.default_vectors_path, merged_vectors)
            _write_faiss_index(merged_vectors, self.default_index_path)

            with self.default_records_path.open("w", encoding="utf-8") as f:
                json.dump(merged_records, f, indent=2, ensure_ascii=False)

            print(f"✓ Updated embeddings: {len(merged_records)} total records")
            print(f"✓ Saved to: {self.default_vectors_path}")
            print(f"✓ FAISS index updated: {self.default_index_path}")

        return {
            "records": merged_records,
            "vectors": merged_vectors,
            "num_records": len(merged_records),
            "embedding_texts": [r["embedding_text"] for r in merged_records],
        }

    # Backward-compatible alias
    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return self.search_items(query, top_k)


def build_and_store(engine_name: str = "default") -> EmbeddingStore:
    configs = Configs.create(engine_name)
    store = EmbeddingStore(configs)
    llminput = load_default_llminput(configs)
    items = llminput["item_catalog"]
    users = collect_user_records(llminput)

    item_vectors = store.build_item_embeddings(items)
    user_vectors = store.build_user_embeddings(llminput)
    default_records, default_vectors = store.build_default_prompt_embeddings(llminput)
    rendered = render_prompt(llminput, configs)

    store.save(
        items,
        item_vectors,
        users,
        user_vectors,
        default_records,
        default_vectors,
        llminput,
        rendered,
    )
    configs.num_items = len(items)
    configs.num_users = len(users)
    configs.save()
    return store


# Backward-compatible alias
ItemEmbeddingStore = EmbeddingStore


if __name__ == "__main__":
    store = build_and_store()

    print("\n--- Item search: sci-fi books ---")
    for hit in store.search_items("sci-fi fiction books", top_k=3):
        print(f"[{hit['score']:.4f}] {hit['item_id']} -> {hit['item']['title']}")

    print("\n--- User search: sci-fi and electronics ---")
    for hit in store.search_users("likes sci-fi books and electronics", top_k=3):
        print(f"[{hit['score']:.4f}] {hit['user_id']} ({hit['role']})")

    print("\n--- Default prompt search: target user preferences ---")
    for hit in store.search_default_prompt("target user likes sci-fi and electronics", top_k=3):
        label = hit.get("placeholder") or hit.get("record_id")
        print(f"[{hit['score']:.4f}] {label} ({hit.get('section')})")

    print("\n--- Hybrid prompt embeddings (partial llminput override) ---")
    session = store.generate_prompt_embeddings(
        llminput_text='top_k is 5, user_profile is {"user_id": "user_999", "segment": "vip"}'
    )
    print(
        f"Generated {session['generated_count']} prompt chunk(s), "
        f"reused {session['default_count']} default prompt chunk(s)."
    )
    print(
        f"Generated {session['user_generated_count']} user(s), "
        f"reused {session['user_default_count']} default user(s)."
    )
    for row in session["records"]:
        if row["vector_source"] == "generated":
            label = row.get("placeholder") or row.get("record_id")
            print(f"  [generated] {label}")
