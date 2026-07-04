import pandas as pd
from pathlib import Path
from typing import Any

from core2.configs import Configs


class DataSets(Configs):
    def __init__(self, engine_name: str = "default"):
        super().__init__(self.project_name_for(engine_name))
        self.engine_name = engine_name
        self._embedding_store = None
        self.prompt_template = self.load_prompt_template()
        self.item: list[dict[str, Any]] = []
        self.user: list[dict[str, Any]] = []
        self.item_user: list[dict[str, Any]] = []

    def load_prompt_template(self) -> str:
        """Load prompt template text if the configured prompt file exists."""
        prompt_path = self.resolve_repo_path(self.prompt_path)
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return ""

    def _resolve_parquet_dir(self, folder_value: str | Path | None, fallback_dir: Path) -> Path:
        """Resolve configured parquet folder or fall back to uploads/<kind>."""
        if folder_value:
            configured = Path(folder_value)
            if configured.exists():
                return configured
            try:
                resolved = self.resolve_repo_path(configured)
                if resolved.exists():
                    return resolved
            except Exception:
                pass
        return fallback_dir

    def read_parquet_file(self, parquet_path: str | Path) -> list[dict[str, Any]]:
        """Read a single parquet file and return records."""
        path = Path(parquet_path)
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")

    def _read_parquet_folder(self, folder: str | Path | None) -> list[dict[str, Any]]:
        """Read all parquet files in a folder and combine records."""
        if folder is None:
            return []
        folder_path = Path(folder)
        if not folder_path.exists() or not folder_path.is_dir():
            return []

        files = sorted(folder_path.glob("*.parquet"))
        if not files:
            return []

        rows: list[dict[str, Any]] = []
        for parquet_file in files:
            rows.extend(self.read_parquet_file(parquet_file))
        return rows

    def get_data(self) -> dict[str, list[dict[str, Any]]]:
        """Read engine parquet datasets and assign item/user/item_user attributes."""
        uploads_root = self.uploads_path
        items_fallback = uploads_root / "items"
        users_fallback = uploads_root / "users"
        interactions_fallback = uploads_root / "interactions"

        items_dir = self._resolve_parquet_dir(self.items_parquet_folder, items_fallback)
        users_dir = self._resolve_parquet_dir(self.users_parquet_folder, users_fallback)
        interactions_dir = self._resolve_parquet_dir(
            self.interactions_parquet_folder,
            interactions_fallback,
        )

        self.item = self._read_parquet_folder(items_dir)
        self.user = self._read_parquet_folder(users_dir)
        self.item_user = self._read_parquet_folder(interactions_dir)

        return {
            "item": self.item,
            "user": self.user,
            "item_user": self.item_user,
        }
