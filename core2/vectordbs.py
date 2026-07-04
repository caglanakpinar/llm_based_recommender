from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json

import faiss
import numpy as np
import pandas as pd

from core2.configs import Configs
from core2.datasets import DataSets
from core2.features import FEATURES
from core2.embeddings import create_embedder


class BaseVectorDB(Configs):
    """Base class for vector database operations."""

    def __init__(self, engine_name: str, datasets: DataSets, dimension: int = 128, metric: str = "L2"):
        super().__init__(project_name=engine_name)
        self.prompt: str | None = None
        self.items = datasets.item
        self.users = datasets.user
        self.items_users = datasets.item_user
        self.dimension = int(dimension)
        self.metric = str(metric).upper()
        self.index = None
        self.embedder = create_embedder(engine_name, dimension=self.DEFAULT_EMBEDDING_DIMENSION, normalize=True)
        self._initialize_index()

    def _initialize_index(self):
        """Initialize FAISS index based on configuration."""
        if self.metric == "IP":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
    
    @abstractmethod
    def write(self, vectors: np.ndarray, ids: Optional[List[int]] = None) -> None:
        """Write vectors to the database."""
        pass
    
    @abstractmethod
    def read(self, query_vectors: np.ndarray, k: int = 10) -> tuple:
        """Read/search vectors from the database."""
        pass
    
    def add_vectors(self, vectors: np.ndarray, ids: Optional[List[int]] = None) -> None:
        """Add vectors to the FAISS index."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        self.index.add(vectors)
    
    def search_vectors(self, query_vectors: np.ndarray, k: int = 10) -> tuple:
        """Search for nearest vectors in the FAISS index."""
        query_vectors = np.asarray(query_vectors, dtype=np.float32)
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        distances, indices = self.index.search(query_vectors, k)
        return distances, indices


class ItemDB(BaseVectorDB):
    """Vector database for items."""

    def __init__(self, datasets: DataSets):
        super().__init__(datasets)
        self.items = datasets.item
        self.prompt = ""
    
    def write(self, vectors: np.ndarray, ids: Optional[List[int]] = None) -> None:
        """Write item vectors to the database."""
        self.add_vectors(vectors, ids)
    
    def read(self, query_vectors: np.ndarray, k: int = 10) -> tuple:
        """Search for similar items."""
        return self.search_vectors(query_vectors, k)


class UserDB(BaseVectorDB):
    """Vector database for users."""

    def __init__(self, engine_name, datasets: DataSets):
        super().__init__(engine_name, datasets)
        self.users = datasets.user
    
    def write(self, vectors: np.ndarray, ids: Optional[List[int]] = None) -> None:
        """Write user vectors to the database."""
        self.add_vectors(vectors, ids)
    
    def read(self, query_vectors: np.ndarray, k: int = 10) -> tuple:
        """Search for similar users."""
        return self.search_vectors(query_vectors, k)

    def _prompt_from_row(self, row: pd.Series) -> str:
        user_id = row.get("user_id", "")
        segment = row.get("segment", "")
        notes = row.get("notes", "")

        feature_parts = []
        for name in FEATURES["user_features"].keys():
            feature_parts.append(f"{name}={row.get(name, '')}")

        return (
            f"user_id: {user_id}; segment: {segment}; notes: {notes}; "
            f"features: {'; '.join(feature_parts)}"
        )

    def build_user_feature_dataset(self) -> pd.DataFrame:
        """Build feature rows for unique users from FEATURES['user_features']."""
        unique_users = self.users.groupby("user_id").first().reset_index()
        kwargs = {'user_id': self.user_id}
        user_feature_funcs = FEATURES.get("user_features", {})
        for feature_name, feature_fn in user_feature_funcs.items():
            _feature_df = feature_fn(self.users, **kwargs)
            unique_users = unique_users.merge(_feature_df, on=self.user_id, how="left").fillna(0) 
            
        unique_users["generated_prompt"] = unique_users.apply(self._prompt_from_row, axis=1)
        return unique_users

    def generate_and_store_user_vectors(self) -> pd.DataFrame:
        """Generate prompts for users, encode vectors, and store in FAISS."""
        dataset = self.build_user_feature_dataset()
        if dataset.empty:
            return dataset

        # Keep all metadata in a sidecar records file; FAISS stores vectors only.
        vectors = self.embedder.text_to_vector(dataset["generated_prompt"].fillna("").astype(str).tolist())
        dataset["generated_prompt_emb_vectors"] = vectors.tolist()
        self.write(vectors, ids=dataset["user_id"].tolist() if "user_id" in dataset.columns else None)

        # Persist complete dataset rows (user_id, segment, notes, prompt, vectors, and features).
        self.user_records_path.parent.mkdir(parents=True, exist_ok=True)
        with self.user_records_path.open("w", encoding="utf-8") as f:
            json.dump(dataset.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

        return dataset
    

class ItemUserDB(BaseVectorDB):
    """Vector database for item-user interactions."""
    def __init__(self, datasets: DataSets):
        super().__init__(datasets)
        self.interactions = datasets.item_user

    def default_prompt(self) -> str:
        """Return the default prompt template."""
        return self.configs.prompt_template or "Default prompt template not found."
    
    def write(self, vectors: np.ndarray, ids: Optional[List[int]] = None) -> None:
        """Write item-user interaction vectors to the database."""
        self.add_vectors(vectors, ids)
    
    def read(self, query_vectors: np.ndarray, k: int = 10) -> tuple:
        """Search for similar item-user interactions."""
        return self.search_vectors(query_vectors, k)
