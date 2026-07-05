from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from core.configs import Configs
from core.llms import FreeLLMCaller
from core.logger import logger
from embedding_store import EmbeddingStore, get_embedding

# TODO: train retrieve embedding model
# TODO: train llms with reco prompt answers
@dataclass
class IndexSource:
    """Describes a FAISS index plus its accompanying records file."""

    name: str
    index_path: Path
    records_path: Path
    vector_path: Path | None = None



class LangChainRAG(Configs):
    """Simple FAISS-based RAG pipeline that uses free Hugging Face LLMs.

    The pipeline discovers FAISS indexes under a folder, embeds the user query,
    retrieves the nearest records, and builds a prompt that includes chat history
    and retrieved context. It prefers LangChain when the optional packages are
    installed, and otherwise falls back to the repository's free LLM wrapper.
    """

    def __init__(
        self,
        index_dir: str | os.PathLike[str] | None = None,
        *,
        llm_model: str = "microsoft/Phi-3-mini-4k-instruct",
        api_key: str | None = None,
    ) -> None:
        self.index_dir = Path(index_dir or Configs.current_dir)
        self.llm_model = self._normalize_model_name(llm_model)
        self.api_key = api_key
        self.chat_history: list[dict[str, str]] = []
        self.sources = discover_index_sources(self.index_dir)
        logger.info(f"Discovered {len(self.sources)} FAISS index sources under {self.index_dir}")
        self._llm = self._build_llm()
        logger.info(
            "LangChainRAG initialized (_llm_available=%s, _llm_type=%s)",
            self._llm is not None,
            type(self._llm).__name__ if self._llm is not None else "None",
        )

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        """Map OpenAI-style names to HuggingFace model IDs for free providers."""
        mapped = {
            "gpt-4o-mini": "meta-llama/Llama-2-7b-chat-hf",
            "gpt-4o": "meta-llama/Llama-2-13b-chat-hf",
            "gpt-4-turbo": "mistralai/Mistral-7B-Instruct-v0.1",
            "gpt-3.5-turbo": "mistralai/Mistral-7B-Instruct-v0.1",
        }
        return mapped.get(model_name, model_name)

    @staticmethod
    def _is_item_source(source_name: str) -> bool:
        name = str(source_name or "").strip().lower()
        return name == "item" or name.startswith("item_") or name.endswith("_item")

    def _call_embedding_model(self, text: str) -> np.ndarray:
        """Call the embedding model to get a vector representation of the text."""
        return get_embedding(text)

    def _build_llm(self) -> Any | None:
        """Create a LangChain chat model when the optional dependency exists."""
        logger.info(f"Building LangChain RAG with model: {self.llm_model}")
        try:
            from langchain_huggingface import ChatHuggingFace
            from langchain_huggingface.llms import HuggingFaceEndpoint
        except Exception:
            logger.exception("Failed to import LangChain dependencies")
            return None

        try:
            endpoint_llm = HuggingFaceEndpoint(
                repo_id=self.llm_model,
                huggingfacehub_api_token=(self.api_key or "").strip() or None,
                temperature=0.2,
                do_sample=False,
                max_new_tokens=512,
            )
            llm = ChatHuggingFace(llm=endpoint_llm)
            logger.info(f"Using LangChain ChatHuggingFace model: {self.llm_model}")
            return llm
        except Exception:
            logger.exception("Failed to create LangChain ChatHuggingFace model")
            return None

    def add_chat_history(self, role: str, content: str) -> None:
        self.chat_history.append({"role": role, "content": content})

    def clear_chat_history(self) -> None:
        self.chat_history.clear()

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve relevant hits from all discovered FAISS indexes."""
        if not self.sources:
            raise FileNotFoundError(
                f"No FAISS indexes found under {self.index_dir}."
            )

        query_vector = get_embedding(query)
        combined: list[dict[str, Any]] = []
        for source in self.sources:
            index = faiss.read_index(str(source.index_path))
            with source.records_path.open(encoding="utf-8") as handle:
                records = json.load(handle)

            scores, indices = index.search(query_vector, top_k)
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                record = records[int(idx)]
                combined.append(
                    {
                        "source": source.name,
                        "score": float(score),
                        "record": record,
                        "text": record.get("embedding_text") or record.get("content") or "",
                    }
                )

        combined.sort(key=lambda item: item["score"], reverse=True)

        # Keep results relevance-first, but guarantee item candidates are present
        # for recommendation fallback rendering.
        selected = list(combined[:top_k])
        min_item_hits = 1 if top_k <= 3 else min(3, max(2, top_k // 3))
        current_item_hits = sum(1 for hit in selected if self._is_item_source(hit.get("source", "")))
        if current_item_hits < min_item_hits:
            extra_item_hits = [
                hit
                for hit in combined
                if self._is_item_source(hit.get("source", "")) and hit not in selected
            ]
            needed = min_item_hits - current_item_hits
            selected.extend(extra_item_hits[:needed])

            item_selected = [hit for hit in selected if self._is_item_source(hit.get("source", ""))]
            non_item_selected = [
                hit for hit in selected if not self._is_item_source(hit.get("source", ""))
            ]
            non_item_selected.sort(key=lambda item: item["score"], reverse=True)

            keep_items = item_selected[:min(min_item_hits, len(item_selected))]
            remaining_slots = max(0, top_k - len(keep_items))
            selected = keep_items + non_item_selected[:remaining_slots]

        selected.sort(key=lambda item: item["score"], reverse=True)
        return selected[:top_k]

    def _normalize_chat_history(self, chat_history: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
        history = chat_history or self.chat_history
        normalized: list[dict[str, str]] = []
        for entry in history:
            if not isinstance(entry, dict):
                continue
            role = str(entry.get("role", "user")).strip().lower()
            content = str(entry.get("content", "")).strip()
            if role and content:
                normalized.append({"role": role, "content": content})
        return normalized

    def _build_prompt(
        self,
        query: str,
        context_items: list[dict[str, Any]],
        top_k: int,
    ) -> str:
        history = self._normalize_chat_history(self.chat_history)
        history_text = ""
        if history:
            history_text = "\n".join(
                f"{entry['role'].capitalize()}: {entry['content']}" for entry in history
            )

        context_text = "\n\n".join(
            f"[{item['source']}] score={item['score']:.3f}\n{item['text']}"
            for item in context_items
        )

        return (
            "You are a helpful recommender assistant. Use the retrieved context and the chat history "
            "to return recommendation candidates for the Target User.\n\n"
            f"Chat history:\n{history_text or 'None'}\n\n"
            f"Question:\n{query}\n\n"
            f"Retrieved context:\n{context_text or 'No relevant context found.'}\n\n"
            "Return ONLY valid JSON with this exact shape (no markdown):\n"
            "{\n"
            f"  \"top_k\": {top_k},\n"
            "  \"recommendations\": [\n"
            "    {\n"
            "      \"rank\": 1,\n"
            "      \"item_id\": \"...\",\n"
            "      \"title\": \"...\",\n"
            "      \"category\": \"...\",\n"
            "      \"tags\": [\"...\"],\n"
            "      \"price\": \"...\",\n"
            "      \"description\": \"...\",\n"
            "      \"score\": 0.0,\n"
            "      \"reason\": \"...\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            
            "Rules: recommend exactly top_k items, use unique item_id values, and prefer source=item records.\n"
            "Based on the informations above, return relevance scores between 0-1:"
        )

    def _extract_json_payload(self, text: str) -> dict[str, Any] | None:
        if not text or not isinstance(text, str):
            return None

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        for match in re.finditer(r"\{[\s\S]*\}", text):
            candidate = match.group(0)
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        return None

    def _extract_item_record(self, context_item: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(context_item, dict):
            return None
        record = context_item.get("record")
        if not isinstance(record, dict):
            return None

        if isinstance(record.get("item"), dict):
            item = dict(record["item"])
            item.setdefault("item_id", record.get("item_id"))
            return item if item.get("item_id") else None

        candidate = {
            "item_id": record.get("item_id"),
            "title": record.get("title"),
            "category": record.get("category"),
            "tags": record.get("tags"),
            "price": record.get("price"),
            "description": record.get("description"),
        }
        return candidate if candidate.get("item_id") else None

    def _build_fallback_recommendations(
        self,
        context_items: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        recommendations: list[dict[str, Any]] = []
        seen_item_ids: set[str] = set()

        for hit in context_items:
            if not isinstance(hit, dict):
                continue
            source = str(hit.get("source", ""))
            item = self._extract_item_record(hit)
            if not item:
                continue

            item_id = str(item.get("item_id", "")).strip()
            if not item_id or item_id in seen_item_ids:
                continue
            seen_item_ids.add(item_id)

            tags = item.get("tags")
            if not isinstance(tags, list):
                tags = [] if tags is None else [str(tags)]

            recommendations.append(
                {
                    "rank": len(recommendations) + 1,
                    "item_id": item_id,
                    "title": item.get("title") or item_id,
                    "category": item.get("category") or "",
                    "tags": tags,
                    "price": item.get("price"),
                    "description": item.get("description") or "",
                    "score": float(hit.get("score", 0.0)),
                    "reason": (
                        f"Retrieved from {source} context with similarity score "
                        f"{float(hit.get('score', 0.0)):.3f}."
                    ),
                }
            )

            if len(recommendations) >= top_k:
                break

        return recommendations

    def _load_item_records_from_sources(self, limit: int = 200) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for source in self.sources:
            if not self._is_item_source(source.name):
                continue
            try:
                with source.records_path.open(encoding="utf-8") as handle:
                    source_records = json.load(handle)
            except Exception:
                logger.exception("Failed loading item records from %s", source.records_path)
                continue

            if not isinstance(source_records, list):
                continue

            for record in source_records:
                item: dict[str, Any] | None = None
                if isinstance(record, dict):
                    if isinstance(record.get("item"), dict):
                        item = dict(record["item"])
                        item.setdefault("item_id", record.get("item_id"))
                    else:
                        item = {
                            "item_id": record.get("item_id"),
                            "title": record.get("title"),
                            "category": record.get("category"),
                            "tags": record.get("tags"),
                            "price": record.get("price"),
                            "description": record.get("description"),
                        }

                if not isinstance(item, dict):
                    continue
                item_id = str(item.get("item_id", "")).strip()
                if not item_id or item_id in seen_ids:
                    continue

                seen_ids.add(item_id)
                records.append(item)
                if len(records) >= limit:
                    return records

        return records

    def _build_catalog_fallback_recommendations(self, top_k: int) -> list[dict[str, Any]]:
        records = self._load_item_records_from_sources(limit=max(50, top_k * 10))
        recommendations: list[dict[str, Any]] = []

        for item in records[:top_k]:
            tags = item.get("tags")
            if not isinstance(tags, list):
                tags = [] if tags is None else [str(tags)]

            recommendations.append(
                {
                    "rank": len(recommendations) + 1,
                    "item_id": str(item.get("item_id", "")),
                    "title": item.get("title") or str(item.get("item_id", "")),
                    "category": item.get("category") or "",
                    "tags": tags,
                    "price": item.get("price"),
                    "description": item.get("description") or "",
                    "score": 0.2,
                    "reason": "Catalog fallback recommendation when LLM output is unstructured.",
                }
            )

        return recommendations

    def _call_llm(self, prompt: str) -> str:
        logger.info(
            "Entering _call_llm (_llm_available=%s, _llm_type=%s)",
            self._llm is not None,
            type(self._llm).__name__ if self._llm is not None else "None",
        )
        if self._llm is not None:
            try:
                from langchain_core.messages import HumanMessage
                response = self._llm.invoke([HumanMessage(content=prompt)])
                return str(getattr(response, "content", response))
            except Exception as e:
                err = str(e)
                if "401" in err or "unauthorized" in err.lower():
                    logger.warning("LangChain HuggingFace call unauthorized (401). Falling back to FreeLLMCaller.")
                else:
                    logger.exception("LangChain LLM call failed; falling back to FreeLLMCaller")

        logger.warning(
            "LangChain chat model unavailable; falling back to FreeLLMCaller (model=%s, provider=huggingface)",
            self.llm_model,
        )

        try:
            caller = FreeLLMCaller(
                api_key=(self.api_key or "").strip() or None,
                model=self.llm_model,
                provider="huggingface",
                embedding_store=None,
                top_k=5,
            )
            text = caller.call(prompt)
            return text if isinstance(text, str) else str(text)
        except Exception:
            logger.exception("FreeLLMCaller fallback failed")
            return ""


    def answer(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Run retrieval and generate an answer using the discovered FAISS indexes."""
        retrieval_top_k = max(top_k * 4, 12)
        logger.info(f"Answering query with retrieval_top_k={retrieval_top_k}, top_k={top_k}")
        context_items = self.retrieve(query, top_k=retrieval_top_k)
        logger.info(f"Retrieved {len(context_items)} context items for query: {query}")
        prompt = self._build_prompt(query, context_items, top_k=top_k)
        logger.info(f"Generated prompt for LLM:\n{prompt}")
        answer_text = self._call_llm(prompt)
        if not isinstance(answer_text, str):
            answer_text = str(answer_text or "")
        prompt_with_answer = f"{prompt}\n\nLLM Answer:\n{answer_text}"
        logger.info(f"Prompt with LLM answer:\n{prompt_with_answer}")
        parsed = self._extract_json_payload(answer_text)

        fallback_recommendations = self._build_fallback_recommendations(context_items, top_k=top_k)
        if not fallback_recommendations:
            logger.warning("No item-shaped retrieval hits found; using catalog fallback recommendations")
            fallback_recommendations = self._build_catalog_fallback_recommendations(top_k=top_k)

        recommendations = fallback_recommendations
        print(f"Fallback recommendations: {json.dumps(fallback_recommendations, indent=2)}")
        if isinstance(parsed, dict):
            parsed_recs = parsed.get("recommendations")
            if isinstance(parsed_recs, list) and parsed_recs:
                recs: list[dict[str, Any]] = []
                for i, rec in enumerate(parsed_recs[:top_k], start=1):
                    if not isinstance(rec, dict):
                        continue
                    rec_copy = dict(rec)
                    rec_copy.setdefault("rank", i)
                    rec_copy.setdefault("tags", [])
                    rec_copy.setdefault("description", "")
                    rec_copy.setdefault("price", None)
                    rec_copy.setdefault("title", rec_copy.get("item_id", f"Item {i}"))
                    recs.append(rec_copy)
                if recs:
                    recommendations = recs

        return {
            "answer": answer_text,
            "context": context_items,
            "prompt": prompt,
            "top_k": int(top_k),
            "recommendations": recommendations,
            "chat_history": self._normalize_chat_history(self.chat_history),
        }


def discover_index_sources(base_dir: str | os.PathLike[str] | None = None) -> list[IndexSource]:
    """Discover FAISS index + records pairs under a folder.

    This is intentionally permissive so it can work for the repository's
    default store layout as well as custom engine folders.
    """

    search_root = Path(base_dir or Configs.current_dir)
    if not search_root.exists():
        return []

    discovered: list[IndexSource] = []
    seen: set[tuple[str, str]] = set()
    logger.info(f"Discovering FAISS index sources under {search_root}")

    for index_path in sorted(search_root.rglob("*_faiss.index")):
        records_path = index_path.with_name(index_path.name.replace("_faiss.index", "_records.json"))
        if not records_path.exists():
            continue

        name = index_path.stem.replace("_faiss", "")
        if name == "item":
            source_name = "item"
        elif name == "user":
            source_name = "user"
        elif name == "default_prompt":
            source_name = "default_prompt"
        else:
            source_name = name

        key = (str(index_path), str(records_path))
        if key in seen:
            continue
        seen.add(key)
        discovered.append(
            IndexSource(
                name=source_name,
                index_path=index_path,
                records_path=records_path,
                vector_path=index_path.with_name(index_path.name.replace("_faiss.index", "_vectors.npy")),
            )
        )
    logger.info(f"Discovered {len(discovered)} FAISS index sources under {search_root}")

    return discovered
