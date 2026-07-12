from __future__ import annotations

import os
import hashlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from core2.configs import Configs


class BaseEmbedder(Configs, ABC):
	"""Base class for embedders used by vector databases.

	Responsibilities:
	- load environment/config context from `Configs`
	- expose a stable embedding contract (`text_to_vector`)
	- provide optional vector normalization for similarity search

	Notes:
	- subclasses should return `float32` vectors with shape (N, D)
	- `dimension` must reflect the actual vector width used by FAISS
	"""

	def __init__(
		self,
		engine_name: str = "default",
		model_name: str | None = None,
		normalize: bool = True,
		**kwargs: Any
	) -> None:
		super().__init__(self.project_name_for(engine_name))
		Configs.configure_hf_environment()
		self.engine_name = engine_name
		self.model_name = model_name or self.model_name
		self.normalize = normalize

	@property
	@abstractmethod
	def dimension(self) -> int:
		"""Vector dimension for this embedder."""

	@abstractmethod
	def text_to_vector(self, texts: list[str]) -> np.ndarray:
		"""Encode a list of strings into vectors (shape: N x D)."""

	def embed_one(self, text: str) -> np.ndarray:
		vectors = self.text_to_vector([text])
		return vectors[0]

	def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
		if not self.normalize:
			return vectors
		norms = np.linalg.norm(vectors, axis=1, keepdims=True)
		norms[norms == 0] = 1.0
		return vectors / norms


class SentenceTransformerEmbedder(BaseEmbedder):
	"""Sentence-Transformers embedder (general-purpose semantic encoder).

	Best for:
	- production-ready semantic retrieval with minimal custom code
	- broad text similarity use cases (queries, prompts, metadata)

	Trade-offs:
	- requires model download and runtime memory
	- inference is slower than hashing but substantially higher quality
	"""

	def __init__(
		self,
		engine_name: str = "default",
		model_name: str = "BAAI/bge-small-en-v1.5",
		normalize: bool = True,
		**kwargs: Any
	) -> None:
		super().__init__(engine_name, model_name=model_name, normalize=normalize)
		from sentence_transformers import SentenceTransformer

		cache_dir = str(Configs.HF_CACHE_DIR / "models")
		self._model = SentenceTransformer("BAAI/bge-small-en-v1.5", cache_folder=cache_dir, token=os.getenv("HF_TOKEN"))
		self._dimension = int(self._model.get_sentence_embedding_dimension())

	@property
	def dimension(self) -> int:
		return self._dimension

	def text_to_vector(self, texts: list[str]) -> np.ndarray:
		if not texts:
			return np.empty((0, self.dimension), dtype=np.float32)
		vectors = self._model.encode(
			texts,
			convert_to_numpy=True,
			show_progress_bar=False,
			normalize_embeddings=False,
		).astype(np.float32)
		return self._normalize_vectors(vectors)

class BGEEmbedder(SentenceTransformerEmbedder):
	"""BGE preset (`BAAI/bge-small-en-v1.5`).

	Best for:
	- strong retrieval quality with relatively small model footprint
	- recommendation/search pipelines that need good semantic matching
	"""

	def __init__(self, engine_name: str = "default", normalize: bool = True, model_name: str = "BAAI/bge-small-en-v1.5", **kwargs: Any) -> None:
		super().__init__(engine_name, model_name="BAAI/bge-small-en-v1.5", normalize=normalize)


class E5Embedder(SentenceTransformerEmbedder):
	"""E5 preset (`intfloat/e5-small-v2`).

	Best for:
	- retrieval tasks where query/document style alignment matters
	- setups that may later adopt explicit `query:` / `passage:` formatting
	"""

	def __init__(self, engine_name: str = "default", model_name: str = "intfloat/e5-small-v2", normalize: bool = True, **kwargs: Any) -> None:
		super().__init__(engine_name, model_name="intfloat/e5-small-v2", normalize=normalize)


class GTEEmbedder(SentenceTransformerEmbedder):
	"""GTE preset (`thenlper/gte-small`).

	Best for:
	- lightweight semantic search and clustering
	- balanced latency/quality when BGE or E5 are not preferred
	"""

	def __init__(self, engine_name: str = "default", model_name: str = "thenlper/gte-small", normalize: bool = True, **kwargs: Any) -> None:
		super().__init__(engine_name, model_name="thenlper/gte-small", normalize=normalize)


class HuggingFaceMeanPoolEmbedder(BaseEmbedder):
	"""Raw Transformers embedder with attention-aware mean pooling.

	Best for:
	- custom experimentation with encoder backbones not wrapped by
	  sentence-transformers
	- advanced users who want tighter control over tokenization/pooling

	Trade-offs:
	- more boilerplate and model-specific tuning
	- can be less optimal than sentence-transformers checkpoints without
	  task-specific fine-tuning
	"""

	def __init__(
		self,
		engine_name: str = "default",
		model_name: str = "BAAI/bge-small-en-v1.5",
		normalize: bool = True,
		**kwargs: Any,
	) -> None:
		super().__init__(engine_name, model_name=model_name, normalize=normalize)
		import torch
		from transformers import AutoModel, AutoTokenizer

		self._torch = torch
		self._max_length = kwargs.get("max_length", 512)
		cache_dir = str(Configs.HF_CACHE_DIR / "models")
		self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)
		self._model = AutoModel.from_pretrained(self.model_name, cache_dir=cache_dir)
		self._model.eval()
		self._dimension = int(getattr(self._model.config, "hidden_size", 768))

	@property
	def dimension(self) -> int:
		return self._dimension

	def text_to_vector(self, texts: list[str]) -> np.ndarray:
		if not texts:
			return np.empty((0, self.dimension), dtype=np.float32)

		with self._torch.no_grad():
			encoded = self._tokenizer(
				texts,
				padding=True,
				truncation=True,
				max_length=self._max_length,
				return_tensors="pt",
			)
			outputs = self._model(**encoded)
			token_embeddings = outputs.last_hidden_state
			attention_mask = encoded["attention_mask"]

			mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
			summed = self._torch.sum(token_embeddings * mask, dim=1)
			counts = self._torch.clamp(mask.sum(dim=1), min=1e-9)
			vectors = (summed / counts).cpu().numpy().astype(np.float32)

		return self._normalize_vectors(vectors)


class HashingEmbedder(BaseEmbedder):
	"""Dependency-free deterministic hashing embedder.

	Best for:
	- fast local tests and smoke checks
	- environments where ML model downloads are unavailable

	Trade-offs:
	- lexical hashing, not semantic understanding
	- lower retrieval quality than neural embedders
	"""

	def __init__(
		self,
		engine_name: str = "default",
		normalize: bool = True,
		**kwargs: Any
	) -> None:
		super().__init__(engine_name, model_name="hashing-embedder", normalize=normalize)
		self._dimension = int(kwargs.get("dimension", 384))

	@property
	def dimension(self) -> int:
		return self._dimension

	def _token_index(self, token: str) -> int:
		digest = hashlib.md5(token.encode("utf-8")).hexdigest()
		return int(digest, 16) % self._dimension

	def text_to_vector(self, texts: list[str]) -> np.ndarray:
		if not texts:
			return np.empty((0, self.dimension), dtype=np.float32)

		vectors = np.zeros((len(texts), self._dimension), dtype=np.float32)
		for i, text in enumerate(texts):
			for token in str(text).lower().split():
				vectors[i, self._token_index(token)] += 1.0
		return self._normalize_vectors(vectors)


EMBEDDER_REGISTRY: dict[str, type[BaseEmbedder]] = {
	"sentence_transformer": SentenceTransformerEmbedder,
	"bge": BGEEmbedder,
	"e5": E5Embedder,
	"gte": GTEEmbedder,
	"hf_mean_pool": HuggingFaceMeanPoolEmbedder,
	"hashing": HashingEmbedder,
}


EMBEDDER_DESCRIPTIONS: dict[str, str] = {
	"sentence_transformer": "General-purpose semantic encoder via sentence-transformers.",
	"bge": "Strong retrieval default (BAAI/bge-small-en-v1.5).",
	"e5": "E5 family encoder; useful for query-document retrieval patterns.",
	"gte": "Lightweight GTE encoder for balanced speed/quality.",
	"hf_mean_pool": "Raw Hugging Face encoder + mean pooling for custom experiments.",
	"hashing": "Deterministic non-neural fallback for tests and constrained environments.",
}


def create_embedder(name: str, engine_name: str = "default", **kwargs: Any) -> BaseEmbedder:
	key = str(name).strip().lower()
	if key not in EMBEDDER_REGISTRY:
		available = ", ".join(sorted(EMBEDDER_REGISTRY.keys()))
		raise ValueError(f"Unknown embedder '{name}'. Available: {available}")
	return EMBEDDER_REGISTRY[key](engine_name=engine_name, **kwargs)


def describe_embedders() -> dict[str, str]:
	"""Return short human-readable descriptions for available embedders."""
	return dict(EMBEDDER_DESCRIPTIONS)
