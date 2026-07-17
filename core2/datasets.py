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
        self.item: pd.DataFrame = pd.DataFrame()
        self.user: pd.DataFrame = pd.DataFrame()
        self.item_user: pd.DataFrame = pd.DataFrame()

    def load_prompt_template(self) -> str:
        """Load prompt template text if the configured prompt file exists."""
        if self.prompt_path_resolved.exists():
            return self.prompt_path_resolved.read_text(encoding="utf-8")
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

    def _read_parquet_folder(self, folder: str | Path | None) -> pd.DataFrame:
        """Read all parquet files in a folder and combine records into a DataFrame."""
        if folder is None:
            return pd.DataFrame()
        folder_path = Path(folder)
        if not folder_path.exists() or not folder_path.is_dir():
            return pd.DataFrame()

        files = sorted(folder_path.glob("*.parquet"))
        if not files:
            return pd.DataFrame()

        dfs = []
        for parquet_file in files:
            df = pd.read_parquet(parquet_file)
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True).iloc[:1000] if dfs else pd.DataFrame()

    def get_data(self) -> dict[str, pd.DataFrame]:
        """Read engine parquet datasets and assign item/user/item_user attributes."""
        uploads_root = self.uploads_path
        
        # Try multiple fallback locations for parquet files
        items_fallback = uploads_root / "items"
        users_fallback = uploads_root / "users"
        interactions_fallback = uploads_root / "interactions"
        
        # Also check the data/ folder in the repo
        data_root = self.resolve_repo_path("data")
        data_items_fallback = data_root / "sample_items"
        data_users_fallback = data_root / "sample_users"
        data_interactions_fallback = data_root / "sample_interactions"

        items_dir = self._resolve_parquet_dir(self.items_parquet_folder, items_fallback)
        if not items_dir.exists():
            items_dir = self._resolve_parquet_dir(None, data_items_fallback)
        
        users_dir = self._resolve_parquet_dir(self.users_parquet_folder, users_fallback)
        if not users_dir.exists():
            users_dir = self._resolve_parquet_dir(None, data_users_fallback)
        
        interactions_dir = self._resolve_parquet_dir(
            self.interactions_parquet_folder,
            interactions_fallback,
        )
        if not interactions_dir.exists():
            interactions_dir = self._resolve_parquet_dir(None, data_interactions_fallback)

        self.item = self._read_parquet_folder(items_dir)
        self.user = self._read_parquet_folder(users_dir)
        self.item_user = self._read_parquet_folder(interactions_dir)

        return {
            "item": self.item,
            "user": self.user,
            "item_user": self.item_user,
        }
