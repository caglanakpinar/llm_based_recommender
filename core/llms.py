from core.configs import Configs


class BaseLLM(Configs):
    def __init__(self, api_key: str | None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model

    def call(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement the call method.")
    

class GPTCaller(BaseLLM):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        super().__init__(api_key, model)

    def call(self, prompt: str) -> str:
        import openai

        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content or ""

class FreeLLMCaller(BaseLLM):
    """
    Free LLM Caller supporting multiple free LLM providers:
    - HuggingFace Inference API (requires HF_TOKEN)
    - Ollama (local, no API key needed)
    - Replicate (requires REPLICATE_API_TOKEN)
    """
    def __init__(self, api_key: str = None, model: str = "mistral-7b-instruct", provider: str = "huggingface"):
        super().__init__(api_key, model)
        self.provider = provider.lower()

    def call(self, prompt: str) -> str:
        """Call free LLM with support for multiple providers."""
        if self.provider == "huggingface":
            return self._call_huggingface(prompt)
        elif self.provider == "ollama":
            return self._call_ollama(prompt)
        elif self.provider == "replicate":
            return self._call_replicate(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Choose: huggingface, ollama, replicate")

    def _call_huggingface(self, prompt: str) -> str:
        """Call HuggingFace Inference API (free tier available)."""
        import os
        from huggingface_hub import InferenceClient

        api_key = self.api_key or os.getenv("HF_TOKEN")
        if not api_key:
            raise ValueError("HF_TOKEN env var required or pass api_key")
        client = InferenceClient(api_key=api_key)
        response = client.text_generation(prompt=prompt, model=self.model, max_new_tokens=512, temperature=0.7)
        return response or ""

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama LLM (completely free, runs locally)."""
        import requests
        ollama_url = "http://localhost:11434/api/generate"
        try:
            response = requests.post(ollama_url, json={"model": self.model, "prompt": prompt, "stream": False, "temperature": 0.7}, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Ollama not running. Start with: ollama serve")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {str(e)}")

    def _call_replicate(self, prompt: str) -> str:
        """Call Replicate API (free credits available)."""
        import os
        import replicate
        api_token = self.api_key or os.getenv("REPLICATE_API_TOKEN")
        if not api_token:
            raise ValueError("REPLICATE_API_TOKEN required")
        replicate.api.token = api_token
        output = replicate.run(self.model, input={"prompt": prompt, "temperature": 0.7})
        return "".join(output) if isinstance(output, list) else str(output)


class BuiltOnScratchLLM(BaseLLM):
    """
    Vector-based LLM using FAISS embeddings for similarity search.
    Uses embedding vectors from store/embeddings/ to find and rank recommendations
    without needing an external LLM API.
    """
    def __init__(self, model: str = "scratch-model", engine_name: str = "default"):
        super().__init__(api_key=None, model=model)
        self.engine_name = engine_name
        self._embedding_store = None

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
        import numpy as np

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

        