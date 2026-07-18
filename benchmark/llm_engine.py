"""Wrapper that builds the core2 LLM recommendation engine for the benchmark.

This mirrors ``test_api.py`` end-to-end (item/user/user-item prompts -> relevance
context -> FAISS vector DB -> retrieval -> LLM ranker) but seeds the pipeline with
the benchmark's training interactions instead of reading the whole ``data/`` folder,
and it lets the ranker score an arbitrary (user, item) candidate pool.

The engine is trained ONLY on the sampled users' training interactions (no test
leakage). LLM scores are cached to disk so runs are resumable/cheap to repeat.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

from core2.configs import Configs
from core2.datasets import DataSets
from core2.prompting import (
    ItemPrompt,
    RelevanceScorePrompt,
    UserItemPrompt,
    UserPrompt,
)
from core2.dbs import ContextDB, ContextVectorDB
from core2.embeddings import create_embedder
from core2.retrieval import Retrieval
from core2.ranking import LLMRanker
from core2.llms import create_llm


class ConfigurableContextVectorDB(ContextVectorDB):
    """ContextVectorDB variant that uses a caller-supplied embedder.

    core2's ``ContextVectorDB`` hardcodes the ``sentence_transformer`` embedder;
    this subclass lets the benchmark swap in any embedder from the registry
    (e.g. ``hf_mean_pool``) and sizes the FAISS index to that embedder's
    dimension.
    """

    def __init__(self, engine_name: str, embedder, prompt, metric: str = "L2"):
        Configs.__init__(self, project_name=engine_name)
        self.prompt = prompt
        self.dimension = int(embedder.dimension)
        self.metric = str(metric).upper()
        self.index = None
        self.index_path = self.resolve_repo_path(self.DEFAULT_CONTEXT_FAISS_NAME)
        self.embedding_model_name = getattr(embedder, "model_name", "custom")
        self.embedder = embedder
        self._initialize_index()
        self.context = prompt.context_wt_relevance_score


def build_llm_ranker(
    engine_name: str,
    items: pd.DataFrame,
    users: pd.DataFrame,
    train_interactions: pd.DataFrame,
    *,
    embedder_name: str = "sentence_transformer",
    llm_provider: str = "ollama",
    llm_model: str | None = "llama3.2:latest",
    temperature: float = 0.2,
    max_new_tokens: int = 16,
    log=print,
) -> LLMRanker:
    """Construct an LLMRanker over the given (train-only) data.

    The DataSets container is populated by direct attribute injection rather than
    ``get_data()`` so we control exactly which rows the engine sees. The embedder
    and LLM (provider/model/temperature) are parameterized so each benchmark
    config produces a distinct engine.
    """
    log(f"[llm] building engine '{engine_name}': "
        f"{len(items)} items, {len(users)} users, {len(train_interactions)} train interactions "
        f"| embedder={embedder_name}")

    datasets = DataSets(engine_name)
    datasets.item = items.reset_index(drop=True)
    datasets.user = users.reset_index(drop=True)
    datasets.item_user = train_interactions.reset_index(drop=True)

    item_prompts = ItemPrompt(engine_name, datasets)
    item_prompts.build_item_feature_dataset()
    log("[llm]   item prompts built")

    user_prompts = UserPrompt(engine_name, datasets)
    user_prompts.build_user_feature_dataset()
    log("[llm]   user prompts built")

    user_item_prompts = UserItemPrompt(engine_name, datasets)
    user_item_prompts.build_user_item_feature_dataset()
    log("[llm]   user-item prompts built")

    context_prompts = RelevanceScorePrompt(
        engine_name, datasets, user_prompts, item_prompts, user_item_prompts
    )
    context_prompts.generate_rag_retrieval_context()
    log(f"[llm]   relevance context built ({len(context_prompts.context)} rows)")

    embedder = create_embedder(embedder_name, engine_name, normalize=True)
    dimension = int(embedder.dimension)
    context_vector_db = ConfigurableContextVectorDB(
        engine_name=engine_name, embedder=embedder, prompt=context_prompts
    )
    context_vector_db.write_context_vectors()
    log(f"[llm]   FAISS context vectors written (dim={dimension})")

    context_db = ContextDB(engine_name=engine_name, dimension=dimension, prompt=context_prompts)
    retrieval = Retrieval(
        engine_name=engine_name,
        datasets=datasets,
        context_prompts=context_prompts,
        context_vector_db=context_vector_db,
        context_db=context_db,
    )
    log("[llm]   retrieval ready")

    ranker = LLMRanker(
        engine_name=engine_name,
        datasets=datasets,
        retrieval=retrieval,
        context_prompts=context_prompts,
        llm_model_name=llm_provider,
    )
    # Override with a fully-configured client (model, token budget, temperature).
    llm_kwargs: dict = {"temperature": temperature, "max_new_tokens": max_new_tokens}
    if llm_model:
        llm_kwargs["model"] = llm_model
    client = create_llm(llm_provider, engine_name, **llm_kwargs)
    # Some providers (e.g. Google) pin model/temperature inside initialize_model;
    # set them explicitly so the config's values actually take effect.
    if llm_model:
        client.model = llm_model
    client.temperature = float(temperature)
    ranker.llm_model_name = client
    log(f"[llm]   ranker ready (provider={llm_provider}, model={llm_model}, temp={temperature})")
    return ranker


class CachedLLMScorer:
    """Scores (user, item) pairs via the LLM ranker with a JSON disk cache."""

    def __init__(self, ranker: LLMRanker, cache_path: Path, log=print):
        self.ranker = ranker
        self.cache_path = Path(cache_path)
        self.log = log
        self.cache: dict[str, float] = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text())
                self.log(f"[llm] loaded {len(self.cache)} cached scores from {self.cache_path.name}")
            except Exception:
                self.cache = {}

    @staticmethod
    def _key(user_id: str, item_id: str) -> str:
        return f"{user_id}\t{item_id}"

    def _flush(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache))

    def score_pairs(
        self, pairs: list[tuple[str, str]], flush_every: int = 25
    ) -> dict[tuple[str, str], float]:
        results: dict[tuple[str, str], float] = {}
        todo = [(u, i) for (u, i) in pairs if self._key(u, i) not in self.cache]
        self.log(f"[llm] scoring {len(pairs)} pairs ({len(pairs)-len(todo)} cached, {len(todo)} to compute)")

        done = 0
        t0 = time.time()
        for (u, i) in pairs:
            key = self._key(u, i)
            if key in self.cache:
                results[(u, i)] = self.cache[key]
                continue
            try:
                out = self.ranker.generate_scores(u, i)
                score = float(out.get("relevance_score", 0.0)) if isinstance(out, dict) else 0.0
            except Exception as exc:  # keep the run going on a single bad pair
                self.log(f"[llm] scoring failed for ({u},{i}): {exc}")
                score = 0.0
            self.cache[key] = score
            results[(u, i)] = score
            done += 1
            if done % flush_every == 0:
                self._flush()
                rate = done / max(1e-9, time.time() - t0)
                self.log(f"[llm]   {done}/{len(todo)} computed ({rate:.2f} pairs/s)")
        self._flush()
        return results
