from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from core.configs import Configs
from core.llms import FreeLLMCaller
from embedding_store import get_embedding


@dataclass
class IndexSource:
    """Describes a FAISS index plus its accompanying records file."""

    name: str
    index_path: Path
    records_path: Path
    vector_path: Path | None = None


class LangChainRAG:
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
        chat_history: list[dict[str, str]] | None = None,
    ) -> None:
        self.index_dir = Path(index_dir or Configs.current_dir)
        self.llm_model = llm_model
        self.api_key = api_key
        self.chat_history: list[dict[str, str]] = chat_history or []
        self.sources = discover_index_sources(self.index_dir)
        self._llm = self._build_llm()

    def _build_llm(self) -> Any | None:
        """Create a LangChain chat model when the optional dependency exists."""
        try:
            from langchain_huggingface import ChatHuggingFace
            from langchain_core.messages import AIMessage, HumanMessage
        except Exception:
            return None

        try:
            llm = ChatHuggingFace(
                model_id=self.llm_model,
                huggingfacehub_api_token=self.api_key,
                temperature=0.2,
            )
            return llm
        except Exception:
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
        return combined[:top_k]

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
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        history = self._normalize_chat_history(chat_history)
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
            "to answer the user. If the answer is not present in the context, be honest and say so.\n\n"
            f"Chat history:\n{history_text or 'None'}\n\n"
            f"Question:\n{query}\n\n"
            f"Retrieved context:\n{context_text or 'No relevant context found.'}\n\n"
            "Answer briefly and clearly."
        )

    def _call_llm(self, prompt: str) -> str:
        if self._llm is not None:
            try:
                from langchain_core.messages import HumanMessage

                response = self._llm.invoke([HumanMessage(content=prompt)])
                return str(getattr(response, "content", response))
            except Exception:
                pass

        caller = FreeLLMCaller(api_key=self.api_key, model=self.llm_model, provider="huggingface")
        return caller.call(prompt)

    def answer(
        self,
        query: str,
        top_k: int = 5,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Run retrieval and generate an answer using the discovered FAISS indexes."""
        context_items = self.retrieve(query, top_k=top_k)
        prompt = self._build_prompt(query, context_items, chat_history=chat_history)
        answer_text = self._call_llm(prompt)
        return {
            "answer": answer_text,
            "context": context_items,
            "prompt": prompt,
            "chat_history": self._normalize_chat_history(chat_history),
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

    return discovered
