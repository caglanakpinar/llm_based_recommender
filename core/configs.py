"""Reco engine configuration loaded from and persisted to docs/configs.yaml."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from core.logger import logger


@dataclass
class DataSetConfigs:
    file: str | Path
    _from: str  # csv | excel | json | parquet | pickle | sql
    arguments: dict[str, str] = field(default_factory=dict)


class Configs:
    """Configuration for one reco engine project folder (e.g. test_reco_engine/)."""

    current_dir = Path(os.path.abspath(__file__)).parent.parent
    ENGINE_SUFFIX = "_reco_engine"

    # Repo-level defaults (embedding_store.py lines 27–30).
    DEFAULT_DATA_DIR = "data"
    DEFAULT_STORE_DIR = "store"
    DEFAULT_ENGINES_DIR = "store/engines"
    DEFAULT_PROMPT_PATH = "recommender_prompt.md"
    DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

    DEFAULT_PROMPT_PLACEHOLDER_SECTIONS: dict[str, str] = {
        "action_weights": "1. Engine Description",
        "allowed_actions": "4. Target User — Item Interactions",
        "user_profile": "3. User Profile",
        "target_user_interactions": "4. Target User — Item Interactions",
        "other_users_interactions": "5. Other Users — Item Interactions",
        "item_catalog": "2. Item Catalog",
        "top_k": "6. Request",
        "generated_at": "7. Expected LLM Response Format",
        "constraints": "6. Request",
        "llm_chat": "9. Additional Instructions",
    }

    DEFAULT_LLMINPUT_KEYS: tuple[str, ...] = (
        "action_weights",
        "allowed_actions",
        "user_profile",
        "target_user_interactions",
        "other_users_interactions",
        "constraints",
        "top_k",
        "generated_at",
        "llm_chat",
    )

    DEFAULT_USER_PROFILE_COLUMNS: dict[str, str] = {
        "user_id": "user_id",
        "segment": "segment",
        "notes": "notes",
    }

    DEFAULT_ITEM_CATALOG_COLUMNS: dict[str, str] = {
        "item_id": "item_id",
        "title": "title",
        "category": "category",
        "tags": "tags",
        "price": "price",
        "description": "description",
    }

    # Embedding artifact paths relative to the engine project root.
    DEFAULT_EMBEDDINGS_DIR = "embeddings"
    DEFAULT_EMBEDDINGS_DEFAULT_DIR = "embeddings/default"
    DEFAULT_ITEM_VECTORS_FILE = "embeddings/item_vectors.npy"
    DEFAULT_ITEM_FAISS_INDEX_FILE = "embeddings/item_faiss.index"
    DEFAULT_ITEM_RECORDS_FILE = "embeddings/item_records.json"
    DEFAULT_USER_VECTORS_FILE = "embeddings/user_vectors.npy"
    DEFAULT_USER_FAISS_INDEX_FILE = "embeddings/user_faiss.index"
    DEFAULT_USER_RECORDS_FILE = "embeddings/user_records.json"
    DEFAULT_DEFAULT_PROMPT_VECTORS_FILE = "embeddings/default/default_prompt_vectors.npy"
    DEFAULT_DEFAULT_PROMPT_FAISS_INDEX_FILE = "embeddings/default/default_prompt_faiss.index"
    DEFAULT_DEFAULT_PROMPT_RECORDS_FILE = "embeddings/default/default_prompt_records.json"
    DEFAULT_DEFAULT_PROMPT_RESULTS_FILE = "default_prompt_results.json"
    DEFAULT_RENDERED_PROMPT_FILE = "rendered_prompt.md"
    DEFAULT_SESSION_DIR = "embeddings/session"
    DEFAULT_LLMINPUT_FILE = "llminput.json"
    DEFAULT_UPLOADS_DIR = "uploads"

    _YAML_KEYS = frozenset(
        {
            "reco_engine_name",
            "data_dir",
            "store_dir",
            "engines_dir",
            "prompt_path",
            "model_name",
            "embeddings_dir",
            "embeddings_default_dir",
            "item_vectors_file",
            "item_faiss_index_file",
            "item_records_file",
            "user_vectors_file",
            "user_faiss_index_file",
            "user_records_file",
            "default_prompt_vectors_file",
            "default_prompt_faiss_index_file",
            "default_prompt_records_file",
            "default_prompt_results_file",
            "rendered_prompt_file",
            "session_dir",
            "llminput_file",
            "uploads_dir",
            "prompt_placeholder_sections",
            "default_llminput_keys",
            "user_profile_columns",
            "item_catalog_columns",
            "top_k",
            "constraints",
            "parquet_folder",
            "interactions_parquet_folder",
            "users_parquet_folder",
            "items_parquet_folder",
            "llm_chat",
            "user_id",
            "num_items",
            "num_users",
            "created_at",
            "datasets",
        }
    )

    def __init__(self, project_name: str) -> None:
        self.project_name = project_name
        self.reco_engine_name = self._short_name(project_name)
        self._paths()
        if self.config_file_path.is_file():
            self._read_from_config_yaml()
        else:
            self._apply_defaults()
        self._setup_embedding_paths()

    @classmethod
    def from_engine_name(cls, engine_name: str) -> Configs:
        return cls(cls.project_name_for(engine_name))

    @classmethod
    def project_name_for(cls, engine_name: str) -> str:
        return f"{engine_name.strip()}{cls.ENGINE_SUFFIX}"

    @classmethod
    def short_name_from_project(cls, project_name: str) -> str:
        if project_name.endswith(cls.ENGINE_SUFFIX):
            return project_name[: -len(cls.ENGINE_SUFFIX)]
        return project_name

    @classmethod
    def list_available_engines(cls, root: Path | None = None) -> list[str]:
        """Return engine short names for folders containing docs/configs.yaml."""
        root = root or cls.current_dir
        names: list[str] = []
        for path in sorted(root.iterdir()):
            if not path.is_dir() or not path.name.endswith(cls.ENGINE_SUFFIX):
                continue
            if (path / "docs" / "configs.yaml").is_file():
                names.append(cls.short_name_from_project(path.name))
        return names

    @classmethod
    def create(cls, reco_engine_name: str, **overrides: Any) -> Configs:
        """Create a new engine folder with docs/configs.yaml."""
        project_name = cls.project_name_for(reco_engine_name)
        dest = cls.current_dir / project_name
        if dest.exists():
            shutil.rmtree(dest)

        configs = cls.__new__(cls)
        configs.project_name = project_name
        configs.reco_engine_name = reco_engine_name.strip()
        configs._paths()
        configs._apply_defaults()
        for key, value in overrides.items():
            if key in cls._YAML_KEYS or key == "datasets":
                setattr(configs, key, value)
        configs.created_at = datetime.now(timezone.utc).isoformat()
        configs.save()
        configs._setup_embedding_paths()
        dest.mkdir(parents=True, exist_ok=True)
        configs.docs_path.mkdir(parents=True, exist_ok=True)
        (dest / "data").mkdir(parents=True, exist_ok=True)
        configs.embeddings_dir.mkdir(parents=True, exist_ok=True)
        configs.default_dir.mkdir(parents=True, exist_ok=True)
        configs.uploads_path.mkdir(parents=True, exist_ok=True)
        return configs

    @staticmethod
    def _short_name(project_name: str) -> str:
        return Configs.short_name_from_project(project_name)

    def _paths(self) -> None:
        self.engine_root = self.current_dir / self.project_name
        self.docs_path = self.engine_root / "docs"
        self.config_file_path = self.docs_path / "configs.yaml"
        self.resource_path = self.current_dir / "resource"
        self.model_artifacts_path = self.engine_root / "model_artifacts"
        self.evaluate_path = self.engine_root / "evaluates"

    def _apply_defaults(self) -> None:
        self.reco_engine_name = getattr(self, "reco_engine_name", self._short_name(self.project_name))
        self.data_dir = self.DEFAULT_DATA_DIR
        self.store_dir = self.DEFAULT_STORE_DIR
        self.engines_dir = self.DEFAULT_ENGINES_DIR
        self.prompt_path = self.DEFAULT_PROMPT_PATH
        self.model_name = self.DEFAULT_MODEL_NAME
        self.embeddings_dir_rel = self.DEFAULT_EMBEDDINGS_DIR
        self.embeddings_default_dir_rel = self.DEFAULT_EMBEDDINGS_DEFAULT_DIR
        self.item_vectors_file = self.DEFAULT_ITEM_VECTORS_FILE
        self.item_faiss_index_file = self.DEFAULT_ITEM_FAISS_INDEX_FILE
        self.item_records_file = self.DEFAULT_ITEM_RECORDS_FILE
        self.user_vectors_file = self.DEFAULT_USER_VECTORS_FILE
        self.user_faiss_index_file = self.DEFAULT_USER_FAISS_INDEX_FILE
        self.user_records_file = self.DEFAULT_USER_RECORDS_FILE
        self.default_prompt_vectors_file = self.DEFAULT_DEFAULT_PROMPT_VECTORS_FILE
        self.default_prompt_faiss_index_file = self.DEFAULT_DEFAULT_PROMPT_FAISS_INDEX_FILE
        self.default_prompt_records_file = self.DEFAULT_DEFAULT_PROMPT_RECORDS_FILE
        self.default_prompt_results_file = self.DEFAULT_DEFAULT_PROMPT_RESULTS_FILE
        self.rendered_prompt_file = self.DEFAULT_RENDERED_PROMPT_FILE
        self.session_dir_rel = self.DEFAULT_SESSION_DIR
        self.llminput_file = self.DEFAULT_LLMINPUT_FILE
        self.uploads_dir_rel = self.DEFAULT_UPLOADS_DIR
        self.prompt_placeholder_sections = dict(self.DEFAULT_PROMPT_PLACEHOLDER_SECTIONS)
        self.default_llminput_keys = list(self.DEFAULT_LLMINPUT_KEYS)
        self.user_profile_columns = dict(self.DEFAULT_USER_PROFILE_COLUMNS)
        self.item_catalog_columns = dict(self.DEFAULT_ITEM_CATALOG_COLUMNS)
        self.top_k = 3
        self.constraints = (
            "- Exclude items already purchased or disliked by the Target User.\n"
            "- Prefer items in categories the user engaged with recently.\n"
            "- Balance exploitation with one exploratory item."
        )
        self.parquet_folder = None
        self.interactions_parquet_folder = None
        self.users_parquet_folder = None
        self.items_parquet_folder = None
        self.llm_chat = ""
        self.user_id = None
        self.num_items = 0
        self.num_users = 0
        self.created_at = None
        self.datasets: dict[str, DataSetConfigs] = {}

    def _setup_embedding_paths(self) -> None:
        self.data_path = self.resolve_repo_path(self.data_dir)
        self.prompt_path_resolved = self.resolve_repo_path(self.prompt_path)
        self.store_path = self.resolve_repo_path(self.store_dir)
        self.engines_path = self.resolve_repo_path(self.engines_dir)

        self.store_dir_path = self.engine_root
        self.embeddings_dir = self.engine_root / self.embeddings_dir_rel
        self.default_dir = self.engine_root / self.embeddings_default_dir_rel
        self.item_vectors_path = self.engine_root / self.item_vectors_file
        self.item_index_path = self.engine_root / self.item_faiss_index_file
        self.item_records_path = self.engine_root / self.item_records_file
        self.user_vectors_path = self.engine_root / self.user_vectors_file
        self.user_index_path = self.engine_root / self.user_faiss_index_file
        self.user_records_path = self.engine_root / self.user_records_file
        self.default_vectors_path = self.engine_root / self.default_prompt_vectors_file
        self.default_index_path = self.engine_root / self.default_prompt_faiss_index_file
        self.default_records_path = self.engine_root / self.default_prompt_records_file
        self.default_results_path = self.engine_root / self.default_prompt_results_file
        self.rendered_prompt_path = self.engine_root / self.rendered_prompt_file
        self.session_dir = self.engine_root / self.session_dir_rel
        self.llminput_path = self.engine_root / self.llminput_file
        self.uploads_path = self.engine_root / self.uploads_dir_rel

    def resolve_repo_path(self, relative: str | Path) -> Path:
        path = Path(relative)
        if path.is_absolute():
            return path
        return self.current_dir / path

    def _read_from_config_yaml(self) -> None:
        if not self.config_file_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file_path}")

        try:
            with self.config_file_path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Error parsing YAML file '{self.config_file_path}': {e}"
            ) from e

        if not isinstance(config, dict):
            raise TypeError(
                f"Expected root of YAML file '{self.config_file_path}' to be a dictionary, "
                f"got {type(config).__name__}"
            )

        self._apply_defaults()
        for key, value in config.items():
            if key == "datasets":
                self.datasets = {}
                for name, args in (value or {}).items():
                    if not isinstance(args, dict):
                        logger.warning(
                            "Skipping invalid dataset config for '%s'.", name
                        )
                        continue
                    file_path = args.get("file")
                    from_type = args.get("_from")
                    arguments = args.get("arguments", {})
                    if file_path is None or from_type is None:
                        raise ValueError(
                            f"Dataset '{name}' requires 'file' and '_from' keys."
                        )
                    self.datasets[name] = DataSetConfigs(
                        file=Path(file_path),
                        _from=from_type,
                        arguments=arguments if isinstance(arguments, dict) else {},
                    )
            elif key in self._YAML_KEYS:
                setattr(self, key, value)
            else:
                setattr(self, key, value)

        if self.interactions_parquet_folder is None and self.parquet_folder:
            self.interactions_parquet_folder = self.parquet_folder

        if "embeddings_dir" in config:
            self.embeddings_dir_rel = config["embeddings_dir"]
        if "embeddings_default_dir" in config:
            self.embeddings_default_dir_rel = config["embeddings_default_dir"]
        if "session_dir" in config:
            self.session_dir_rel = config["session_dir"]
        if "uploads_dir" in config:
            self.uploads_dir_rel = config["uploads_dir"]

    def to_yaml_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "reco_engine_name": self.reco_engine_name,
            "data_dir": self.data_dir,
            "store_dir": self.store_dir,
            "engines_dir": self.engines_dir,
            "prompt_path": self.prompt_path,
            "model_name": self.model_name,
            "embeddings_dir": self.embeddings_dir_rel,
            "embeddings_default_dir": self.embeddings_default_dir_rel,
            "item_vectors_file": self.item_vectors_file,
            "item_faiss_index_file": self.item_faiss_index_file,
            "item_records_file": self.item_records_file,
            "user_vectors_file": self.user_vectors_file,
            "user_faiss_index_file": self.user_faiss_index_file,
            "user_records_file": self.user_records_file,
            "default_prompt_vectors_file": self.default_prompt_vectors_file,
            "default_prompt_faiss_index_file": self.default_prompt_faiss_index_file,
            "default_prompt_records_file": self.default_prompt_records_file,
            "default_prompt_results_file": self.default_prompt_results_file,
            "rendered_prompt_file": self.rendered_prompt_file,
            "session_dir": self.session_dir_rel,
            "llminput_file": self.llminput_file,
            "uploads_dir": self.uploads_dir_rel,
            "prompt_placeholder_sections": self.prompt_placeholder_sections,
            "default_llminput_keys": self.default_llminput_keys,
            "user_profile_columns": self.user_profile_columns,
            "item_catalog_columns": self.item_catalog_columns,
            "top_k": self.top_k,
            "constraints": self.constraints,
            "parquet_folder": self.interactions_parquet_folder or self.parquet_folder,
            "interactions_parquet_folder": self.interactions_parquet_folder,
            "users_parquet_folder": self.users_parquet_folder,
            "items_parquet_folder": self.items_parquet_folder,
            "llm_chat": self.llm_chat,
            "user_id": self.user_id,
            "num_items": self.num_items,
            "num_users": self.num_users,
            "created_at": self.created_at,
        }
        if self.datasets:
            data["datasets"] = {
                name: {
                    "file": str(cfg.file),
                    "_from": cfg._from,
                    "arguments": cfg.arguments,
                }
                for name, cfg in self.datasets.items()
            }
        return data

    def save(self) -> None:
        self.docs_path.mkdir(parents=True, exist_ok=True)
        with self.config_file_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.to_yaml_dict(),
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    def update_from_build(
        self,
        *,
        llminput: dict[str, Any],
        parquet_folder: str | None = None,
    ) -> None:
        profile = llminput.get("user_profile") or {}
        self.top_k = llminput.get("top_k", self.top_k)
        self.constraints = llminput.get("constraints", self.constraints)
        self.llm_chat = llminput.get("llm_chat", self.llm_chat)
        self.user_profile_columns = llminput.get(
            "user_profile_columns", self.user_profile_columns
        )
        self.item_catalog_columns = llminput.get(
            "item_catalog_columns", self.item_catalog_columns
        )
        self.user_id = profile.get("user_id")
        self.interactions_parquet_folder = (
            str(Path(parquet_folder).resolve()) if parquet_folder else self.interactions_parquet_folder
        )
        if self.interactions_parquet_folder is None and self.parquet_folder:
            self.interactions_parquet_folder = self.parquet_folder
        self.parquet_folder = self.interactions_parquet_folder
        if self.prompt_placeholder_sections != llminput.get("prompt_placeholder_sections"):
            pass  # sections come from yaml defaults unless explicitly overridden
        self.save()
