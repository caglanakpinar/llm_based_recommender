from typing import Dict, List, Any
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from kserve import Model, ModelServer
from core2.configs import Configs
from core2.logger import logger
from core2.ranking import BaseRelevanceRanking, LLMRanker

# Per-candidate relevance scoring makes one LLM call each; those calls are
# network-bound, so scoring candidates concurrently avoids the request timing
# out on users with many candidates. Override with RECO_SCORING_WORKERS.
MAX_SCORING_WORKERS = max(1, int(os.getenv("RECO_SCORING_WORKERS", "8")))


def _mask_key(value: Any) -> str:
    """Render an API key for logs without leaking it: show only length and the
    last 4 characters, e.g. ``set(len=39, …abcd)``. Never logs the full secret."""
    text = str(value or "")
    if not text:
        return "MISSING/EMPTY"
    tail = text[-4:] if len(text) >= 4 else ""
    return f"set(len={len(text)}, …{tail})"


class RecoEnginePredictor(Model):
    """KServe predictor for recommendation engine."""

    def __init__(self, name: str, ranker: LLMRanker):
        super().__init__(name)
        self.name = name
        self.ranker = ranker
        self.retrieval = getattr(ranker, 'retrieval', None)
        self.ready = True
        # Log the LLM wiring once at construction so every served engine records
        # which provider/model/key it will use before any prediction runs.
        self._log_llm_config()

    def _log_llm_config(self) -> None:
        """Log the resolved LLM provider, model, and a masked API-key indicator."""
        llm = getattr(self.ranker, "llm_model_name", None)
        if llm is None:
            logger.warning("[predictor:%s] ranker has no llm_model_name attached", self.name)
            return
        logger.info(
            "[predictor:%s] LLM ready | provider=%s | model=%s | engine=%s | api_key=%s",
            self.name,
            type(llm).__name__,
            getattr(llm, "model", "?"),
            getattr(llm, "engine_name", "?"),
            _mask_key(getattr(llm, "api_key", None)),
        )

    def load(self) -> bool:
        """Load the model."""
        self.ready = True
        return self.ready

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict recommendations for a user.

        Process:
        1. Retrieve candidate items from retrieval
        2. Score each candidate with ranking model
        3. Return ranked recommendations with scores

        Args:
            request: Dict with keys:
                - user_id: target user
                - top_k: number of recommendations (optional)

        Returns:
            Dict with recommendations and scores
        """
        user_id = request.get("user_id")
        top_k = request.get("top_k", 10)
        logger.info(
            "[predictor:%s] predict() called | user_id=%s | top_k=%s | request_keys=%s",
            self.name, user_id, top_k, list(request.keys()),
        )
        # Re-log LLM config at call time so each request's logs are self-contained
        # (this is the answer to "where is the api key" during an API call).
        self._log_llm_config()

        if not user_id:
            logger.error("[predictor:%s] missing user_id in request", self.name)
            return {"error": "user_id is required"}

        # 1. Retrieve candidate items from retrieval
        if self.retrieval is None:
            logger.error("[predictor:%s] no retrieval attached to ranker", self.name)
            return {"error": "retrieval unavailable", "user_id": user_id}

        candidates = self.retrieval.retrieve_candidates(user_id, top_k=top_k * 2)
        logger.info(
            "[predictor:%s] step 1 retrieval | user_id=%s | candidates=%s | ids=%s",
            self.name,
            user_id,
            len(candidates) if candidates else 0,
            [c.get("item_id") for c in (candidates or [])][:20],
        )

        if not candidates or len(candidates) == 0:
            logger.warning(
                "[predictor:%s] no candidates for user_id=%s — returning empty recommendations "
                "(user unknown to retrieval or no interactions/similar items)",
                self.name, user_id,
            )
            return {"recommendations": [], "user_id": user_id}

        # 2. Score each candidate with the ranking model. Each score is an
        # independent, network-bound LLM call, so run them concurrently to keep
        # the total well under the client request timeout.
        valid_candidates = [c for c in candidates if c.get("item_id")]
        skipped = len(candidates) - len(valid_candidates)
        if skipped:
            logger.warning(
                "[predictor:%s] step 2 skipped %d candidate(s) with no item_id",
                self.name, skipped,
            )

        def _score_candidate(candidate: Dict[str, Any]) -> Dict[str, Any] | None:
            item_id = candidate.get("item_id")
            try:
                score_result = self.ranker.generate_scores(user_id, item_id)
            except Exception as exc:
                logger.exception(
                    "[predictor:%s] step 2 scoring FAILED | user_id=%s | item_id=%s | error=%s",
                    self.name, user_id, item_id, exc,
                )
                return None
            score = score_result.get("relevance_score", 0.0) if isinstance(score_result, dict) else 0.0
            logger.info("[predictor:%s] step 2 scored | item_id=%s | score=%s", self.name, item_id, score)
            return {
                "item_id": item_id,
                "title": candidate.get("title", ""),
                "category": candidate.get("category", ""),
                "score": float(score),
                "signals": candidate.get("signals", []),
            }

        workers = max(1, min(MAX_SCORING_WORKERS, len(valid_candidates)))
        t0 = time.perf_counter()
        scored_items = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(_score_candidate, valid_candidates):
                if result is not None:
                    scored_items.append(result)
        logger.info(
            "[predictor:%s] step 2 scored %d/%d candidates in %.1fs (workers=%d)",
            self.name, len(scored_items), len(valid_candidates), time.perf_counter() - t0, workers,
        )

        # 3. Sort by score and return top_k
        ranked = sorted(scored_items, key=lambda x: x["score"], reverse=True)[:top_k]
        logger.info(
            "[predictor:%s] step 3 done | scored=%s | returned=%s | top_scores=%s",
            self.name,
            len(scored_items),
            len(ranked),
            [round(r["score"], 4) for r in ranked],
        )

        return {
            "user_id": user_id,
            "recommendations": ranked,
            "total_scored": len(scored_items),
            "returned": len(ranked)
        }


class BuildRecoEngine(Configs):
    """Class for building the recommendation engine."""
    def __init__(self, engine_name: str, datasets, retrieval, ranker: LLMRanker, context_prompts):
        super().__init__(project_name=engine_name)
        self.datasets = datasets
        self.retrieval = retrieval
        self.ranker = ranker
        self.context_prompts = context_prompts
        self.predictor = None

    def initialize_kserve_api(self):
        """Initialize KServe API for serving recommendations."""
        try:
            self.predictor = RecoEnginePredictor(
                name=self.project_name,
                ranker=self.ranker
            )
            logger.info("[build:%s] KServe predictor initialized successfully", self.project_name)
            print(f"[DEBUG] KServe predictor initialized successfully for '{self.project_name}'")
            return self.predictor
        except Exception as e:
            logger.exception("[build:%s] Failed to initialize KServe predictor", self.project_name)
            print(f"[ERROR] Failed to initialize KServe predictor: {str(e)}")
            raise

    def reco_engine_serve(self):
        """Initialize the recommendation engine (for use in background thread)."""
        try:
            # Initialize the predictor
            self.initialize_kserve_api()
            print(f"[DEBUG] Recommendation engine ready for '{self.project_name}'")
            # Note: KServe server would normally be started here, but we run it via HTTP requests instead
        except Exception as e:
            print(f"[ERROR] Failed to initialize recommendation engine: {str(e)}")
        