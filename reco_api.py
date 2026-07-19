"""Standalone KServe-compatible recommendation API.

Kept in its own module (not ``app.py``) on purpose: Streamlit 1.58 serves its UI
through uvicorn/ASGI, and any FastAPI app defined in the module Streamlit runs
gets picked up by Streamlit's own server. Isolating the API here lets it run on
its own port without colliding with the Streamlit process.
"""

from __future__ import annotations

import threading
from typing import Any

from fastapi import Body, FastAPI

from core2.configs import Configs
from core2.datasets import DataSets
from core2.dbs import ContextDB, ContextVectorDB
from core2.logger import logger
from core2.prompting import (
    ItemPrompt,
    RelevanceScorePrompt,
    UserItemPrompt,
    UserPrompt,
)
from core2.ranking import LLMRanker
from core2.reco_engine import BuildRecoEngine
from core2.retrieval import Retrieval
from reco_engine import engine_exists

KSERVE_API_PORT = 8080
KSERVE_API_URL = f"http://localhost:{KSERVE_API_PORT}/v1/models/reco:predict"

# Predictors built by the API server, keyed by engine name.
_SERVER_PREDICTORS: dict[str, Any] = {}
_SERVER_PREDICTOR_LOCK = threading.Lock()


def build_engine_prompts(engine_name: str):
    """Rebuild datasets and the relevance-context prompt for an engine.

    Every prompt stage reads its parquet cache when present, so this is cheap
    for an already-built engine (no LLM/embedding regeneration).
    """
    datasets = DataSets(engine_name)
    datasets.get_data()

    item_prompts = ItemPrompt(engine_name, datasets)
    item_prompts.build_item_feature_dataset()

    user_prompts = UserPrompt(engine_name, datasets)
    user_prompts.build_user_feature_dataset()

    user_item_prompts = UserItemPrompt(engine_name, datasets)
    user_item_prompts.build_user_item_feature_dataset()

    context_prompts = RelevanceScorePrompt(
        engine_name=engine_name,
        datasets=datasets,
        item_prompts=item_prompts,
        user_prompts=user_prompts,
        user_item_prompts=user_item_prompts,
    )
    context_prompts.generate_rag_retrieval_context()
    return datasets, context_prompts


def assemble_predictor(
    engine_name: str,
    datasets,
    context_prompts,
    *,
    embedding_model_name: str,
    llm_platform: str,
    llm_model: str | None,
    write: bool,
):
    """Build DBs -> retrieval -> ranker -> KServe predictor from built prompts.

    When ``write`` is True the context vectors/documents are (re)persisted (used
    on first build). Otherwise the persisted FAISS index is loaded, so the
    predictor can be reconstructed without re-embedding the corpus.
    """
    context_vector_db = ContextVectorDB(
        engine_name=engine_name,
        prompt=context_prompts,
        embedding_model_name=embedding_model_name,
    )
    if write:
        context_vector_db.write_context_vectors()
    else:
        try:
            context_vector_db.load_context_vectors()
        except FileNotFoundError:
            context_vector_db.write_context_vectors()

    context_db = ContextDB(engine_name=engine_name, prompt=context_prompts)
    if write:
        context_db.write_context()

    retrieve = Retrieval(
        engine_name=engine_name,
        datasets=datasets,
        context_prompts=context_prompts,
        context_vector_db=context_vector_db,
        context_db=context_db,
    )
    ranker = LLMRanker(
        engine_name=engine_name,
        datasets=datasets,
        retrieval=retrieve,
        context_prompts=context_prompts,
        llm_model_name=llm_platform,
        llm_model=(llm_model or None),
    )
    eng = BuildRecoEngine(
        engine_name=engine_name,
        datasets=datasets,
        retrieval=retrieve,
        ranker=ranker,
        context_prompts=context_prompts,
    )
    return eng.initialize_kserve_api()


def build_predictor_for_engine(engine_name: str):
    """Reconstruct a predictor for ``engine_name`` from persisted artifacts.

    Reads the embedding model / LLM platform / model name saved in the engine's
    ``docs/configs.yaml`` and rebuilds the prompt/DB/ranker pipeline. These drive
    which embedder and LLM the served engine uses. No caching here — callers cache.
    """
    configs = Configs.from_engine_name(engine_name)
    embedding_model_name = (
        getattr(configs, "embedding_model", None) or Configs.DEFAULT_EMBEDDING_MODEL_NAME
    )
    llm_platform = getattr(configs, "llm_platform", None) or Configs.DEFAULT_LLM_MODEL_NAME
    llm_model = (getattr(configs, "llm_model", None) or "").strip() or None

    datasets, context_prompts = build_engine_prompts(engine_name)
    # The FAISS index / Chroma store live at repo root and are shared across
    # engines, so re-embed this engine's (small) context on start to guarantee
    # the served vectors match the selected engine rather than the last built.
    return assemble_predictor(
        engine_name,
        datasets,
        context_prompts,
        embedding_model_name=embedding_model_name,
        llm_platform=llm_platform,
        llm_model=llm_model,
        write=True,
    )


def server_predict(engine_name: str, api_request: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a predict request to the (lazily built) predictor for an engine."""
    if not engine_name:
        return {"error": "engine_name is required"}
    if not engine_exists(engine_name):
        return {"error": f"unknown engine '{engine_name}'"}
    with _SERVER_PREDICTOR_LOCK:
        predictor = _SERVER_PREDICTORS.get(engine_name)
        if predictor is None:
            logger.info("KServe API building predictor for engine '%s'", engine_name)
            predictor = build_predictor_for_engine(engine_name)
            _SERVER_PREDICTORS[engine_name] = predictor
    return predictor.predict(api_request)


reco_api_app = FastAPI(title="Reco Engine API")


@reco_api_app.get("/v1/models/{model_name}")
async def _model_ready(model_name: str) -> dict[str, Any]:
    return {"name": model_name, "ready": True}


@reco_api_app.post("/v1/models/{model_name}:predict")
async def _predict(model_name: str, payload: dict = Body(...)) -> dict[str, Any]:
    engine_name = payload.get("engine_name") or model_name
    return server_predict(engine_name, payload)


def start_api(port: int = KSERVE_API_PORT) -> dict[str, Any]:
    """Start the recommendation API in a background daemon thread.

    Serves ``POST /v1/models/{name}:predict``; the target engine is taken from
    the request body (``engine_name``), falling back to the URL model name.
    Signal handlers are disabled so uvicorn can run off the main thread.
    """
    import uvicorn

    config = uvicorn.Config(
        reco_api_app, host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=server.run, daemon=True, name="reco-kserve-api")
    thread.start()
    logger.info("Reco KServe API server starting on port %s", port)
    return {"port": port, "thread": thread}
