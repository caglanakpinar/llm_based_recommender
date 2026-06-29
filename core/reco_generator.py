from core.base_llm import BaseLLM
from core.configs import Configs
from core.llms import FreeLLMCaller, GPTCaller
from embedding_store import (
    get_embedding,
    result_to_text,
    load_default_llminput,
)
from typing import Any


class RecoGenerator(BaseLLM):
    """
    Vector-based LLM using FAISS embeddings for similarity search.
    Uses embedding vectors from store/embeddings/ to find and rank recommendations
    without needing an external LLM API.
    """
    def __init__(self, model: str = "scratch-model", engine_name: str = "default"):
        super().__init__(api_key=None, model=model)
        self.engine_name = engine_name
        self._embedding_store = None
        self.prompt_template = self.load_prompt_template()

    def load_prompt_template(self) -> str:
        configs = Configs.from_engine_name(self.engine_name)
        return configs.prompt_path_resolved.read_text(encoding="utf-8")

    def generate_llm_prompt(self, result: dict[str, Any], llminput: dict[str, Any] | None = None) -> str:
        import json

        configs = Configs.from_engine_name(self.engine_name)
        payload = dict(load_default_llminput(configs))
        if llminput:
            payload.update(llminput)
        payload["llm_chat"] = result_to_text(result)

        template = self.prompt_template
        skip_keys = frozenset({"user_profile_columns", "item_catalog_columns"})
        for key, value in payload.items():
            if key in skip_keys:
                continue
            if isinstance(value, (dict, list)):
                replacement = json.dumps(value, indent=2, ensure_ascii=False)
            else:
                replacement = str(value) if value is not None else ""
            template = template.replace("{" + key + "}", replacement)
        return template

    def _get_llm_caller(self) -> BaseLLM:
        if self.api_key:
            return GPTCaller(api_key=self.api_key, model=self.model)
        return FreeLLMCaller(model=self.model, provider="ollama")

    def call_llm(self, prompt: str) -> str:
        caller = self._get_llm_caller()
        return caller.call(prompt)

    @property
    def embedding_store(self):
        """Lazy load EmbeddingStore."""
        if self._embedding_store is None:
            from embedding_store import EmbeddingStore
            from core.configs import Configs
            configs = Configs.from_engine_name(self.engine_name)
            self._embedding_store = EmbeddingStore(configs)
        return self._embedding_store

    def call(self, prompt: str) -> str:
        """
        Generate recommendations using FAISS vector similarity.
        Returns JSON recommendations based on embedding similarity.
        """
        import json
        # RETRIEVAL CALLER
        try:
            # Search in item embeddings
            item_results = self.embedding_store.search_items(prompt, top_k=5)
            user_results = self.embedding_store.search_users(prompt, top_k=3)
            prompt_results = self.embedding_store.search_default_prompt(prompt, top_k=3)

            # Build recommendations from similarity search results
            recommendations = []
            for i, hit in enumerate(item_results[:3], 1):
                recommendations.append({
                    "rank": i,
                    "item_id": hit["item_id"],
                    "title": hit["item"]["title"],
                    "category": hit["item"]["category"],
                    "score": float(hit["score"]),
                    "reason": f"Vector similarity: {hit['score']:.2%} match with query",
                    "signals": [
                        f"embedding_similarity: {hit['score']:.4f}",
                        f"category: {hit['item']['category']}",
                        f"tags: {', '.join(hit['item'].get('tags', []))}"
                    ]
                })

            result = {
                "method": "vector_similarity_scratch",
                "engine": self.engine_name,
                "recommendations": recommendations,
                "item_search_results": len(item_results),
                "user_matches": [{"user_id": u["user_id"], "score": float(u["score"])} for u in user_results],
                "query_similarity": len(prompt_results),
            }

            result_prompt = result_to_text(result)
            result_embedding = get_embedding(result_prompt)
            rendered_prompt = self.generate_llm_prompt(result)

            result["result_prompt"] = result_prompt
            result["result_embedding"] = result_embedding[0].tolist()
            result["rendered_prompt"] = rendered_prompt

            try:
                llm_response = self.call_llm(rendered_prompt)
                result["llm_response"] = llm_response
                try:
                    result["llm_response_json"] = json.loads(llm_response)
                except json.JSONDecodeError:
                    result["llm_response_json"] = None
            except Exception as llm_exc:
                result["llm_response_error"] = str(llm_exc)

            return json.dumps(result, indent=2)

        except Exception as e:
            error_result = {
                "method": "vector_similarity_scratch",
                "error": str(e),
                "message": "Failed to generate recommendations from FAISS vectors",
            }
            return json.dumps(error_result, indent=2)

    def search_similar_items(self, query: str, top_k: int = 5) -> list[dict]:
        """Direct access to FAISS item similarity search."""
        return self.embedding_store.search_items(query, top_k)

    def search_similar_users(self, query: str, top_k: int = 3) -> list[dict]:
        """Direct access to FAISS user similarity search."""
        return self.embedding_store.search_users(query, top_k)

    def search_similar_prompts(self, query: str, top_k: int = 3) -> list[dict]:
        """Direct access to FAISS prompt similarity search."""
        return self.embedding_store.search_default_prompt(query, top_k)
        