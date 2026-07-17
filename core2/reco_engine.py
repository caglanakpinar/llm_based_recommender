from typing import Dict, List, Any
import json
import pandas as pd
from kserve import Model, ModelServer
from core2.configs import Configs
from core2.ranking import BaseRelevanceRanking, LLMRanker


class RecoEnginePredictor(Model):
    """KServe predictor for recommendation engine."""
    
    def __init__(self, name: str, ranker: LLMRanker):
        super().__init__(name)
        self.name = name
        self.ranker = ranker
        self.retrieval = getattr(ranker, 'retrieval', None)
        self.ready = True
    
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
        
        if not user_id:
            return {"error": "user_id is required"}
        
        # 1. Retrieve candidate items from retrieval
        candidates = self.retrieval.retrieve_candidates(user_id, top_k=top_k * 2)
        
        if not candidates or len(candidates) == 0:
            return {"recommendations": [], "user_id": user_id}
        
        # 2. Score each candidate with ranking model
        scored_items = []
        for candidate in candidates:
            item_id = candidate.get("item_id")
            if not item_id:
                continue
            
            # Generate relevance score for user-item pair
            score_result = self.ranker.generate_scores(user_id, item_id)
            score = score_result.get("relevance_score", 0.0) if isinstance(score_result, dict) else 0.0
            
            scored_items.append({
                "item_id": item_id,
                "title": candidate.get("title", ""),
                "category": candidate.get("category", ""),
                "score": float(score),
                "signals": candidate.get("signals", [])
            })
        
        # 3. Sort by score and return top_k
        ranked = sorted(scored_items, key=lambda x: x["score"], reverse=True)[:top_k]
        
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
            print(f"[DEBUG] KServe predictor initialized successfully for '{self.project_name}'")
            return self.predictor
        except Exception as e:
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
        