from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json

import faiss
import numpy as np
import pandas as pd
import chromadb

from core2.configs import Configs
from core2.datasets import DataSets
from core2.features import FEATURES
from core2.embeddings import create_embedder
from core2.prompting import BasePrompt


class BaseFaissDB(Configs):
    """Base class for vector database operations."""

    def __init__(self, engine_name: str, dimension: int = 128, metric: str = "L2", prompt: BasePrompt | None = None, embedding_model_name: str = Configs.DEFAULT_EMBEDDING_MODEL_NAME):
        super().__init__(project_name=engine_name)
        self.prompt: BasePrompt | None = prompt 
        self.dimension = int(dimension)
        self.metric = str(metric).upper()
        self.index = None
        self.index_path = self.resolve_repo_path(self.DEFAULT_CONTEXT_FAISS_NAME)
        self.embedder = create_embedder(embedding_model_name, engine_name, dimension=self.DEFAULT_EMBEDDING_DIMENSION, normalize=True)
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
        faiss.write_index(self.index, self.index_path.as_posix())
    
    def search_vectors(self, query_vectors: np.ndarray, k: int = 10) -> tuple:
        """Search for nearest vectors in the FAISS index."""
        query_vectors = np.asarray(query_vectors, dtype=np.float32)
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        distances, indices = self.index.search(query_vectors, k)
        return distances, indices


class BaseChromaDB(Configs):
    """Base class for Chroma vector database operations."""

    def __init__(
        self,
        engine_name: str,
        dimension: int = 128,
        collection_name: str = Configs.DEFAULT_CONTEXT_CHROMODB_NAME,
        prompt: BasePrompt | None = None,
    ):
        super().__init__(project_name=engine_name)
        project_name = engine_name
        super().__init__(project_name=project_name)
        self.prompt: BasePrompt | None = prompt
        self.dimension = int(dimension)
        self.collection_name = str(collection_name)
        self._chroma_client = None
        self._collection = None
        self.persist_directory = self.resolve_repo_path(Configs.DEFAULT_CONTEXT_CHROMODB_PATH)

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        self._chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        self._collection = self._chroma_client.get_or_create_collection(name=self.collection_name)
        return self._collection

    def write(
        self,
        documents: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Write text documents to the Chroma collection."""
        if not documents:
            return
        collection = self._get_collection()
        docs = [str(d) for d in documents]
        doc_ids = ids or [f"doc_{i}" for i in range(len(docs))]
        collection.upsert(ids=doc_ids, documents=docs, metadatas=metadatas)
    
    def read(self, query_texts: List[str], k: int = 10) -> Dict[str, Any]:
        """Read/search text documents from the Chroma collection."""
        collection = self._get_collection()
        texts = [str(t) for t in query_texts]
        return collection.query(query_texts=texts, n_results=int(k))

    def update_text(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Update existing Chroma documents by id."""
        if not ids or not documents:
            return
        collection = self._get_collection()
        collection.update(ids=[str(i) for i in ids], documents=[str(d) for d in documents], metadatas=metadatas)


class ContextDB(BaseChromaDB):
    """Context database for storing and retrieving context documents."""

    def __init__(
        self,
        engine_name: str,
        dimension: int = 128,
        collection_name: str = Configs.DEFAULT_CONTEXT_CHROMODB_NAME,
        prompt: BasePrompt | None = None,
    ):
        super().__init__(
            engine_name=engine_name,
            dimension=dimension,
            collection_name=collection_name,
            prompt=prompt,
        )
        self.context = prompt.context

    def write_context(self):
        self.context['id'] = self.context.apply(
            lambda row: f"{row[self.user_id]}_{row[self.item_id]}", axis=1
        )
        self.write(
            ids=self.context['id'].astype(str).tolist(),
            documents=self.context["generated_prompt"].astype(str).tolist(),
            metadatas=self.context.to_dict(orient="records"),
        )


class ContextVectorDB(BaseFaissDB):
    """Context vector database for storing and retrieving context vectors."""

    def __init__(
        self,
        engine_name: str,
        dimension: int = 128,
        metric: str = "L2",
        prompt: BasePrompt | None = None,
    ):
        super().__init__(
            engine_name=engine_name,
            dimension=dimension,
            metric=metric,
        )
        self.context = prompt.context
        self.embedder = create_embedder(engine_name, dimension=self.dimension, normalize=True)

    def write_context_vectors(self):
        self.context['id'] = self.context.apply(
            lambda row: f"{row[self.user_id]}_{row[self.item_id]}", axis=1
        )
        self.context['embedding'] = self.context.apply(
            lambda row: self.embedder.encode(row['generated_prompt']), axis=1
        )
        vectors = np.vstack(self.context["embedding"].values).astype(np.float32)
        self.add_vectors(vectors=vectors, ids=self.context['id'].astype(str).tolist())
