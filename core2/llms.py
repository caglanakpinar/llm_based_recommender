from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib import request

import anthropic
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import InferenceClient
from google import genai
from google.genai import types as genai_types

from core2.configs import Configs


class BaseLLM(Configs):
	"""Base LLM interface.

	Each subclass should:
	1. Initialize its client/model in ``initialize_model``
	2. Execute generation in ``call``
	"""

	def __init__(
		self,
		engine_name: str = "default",
		*,
		model: str | None = None,
		api_key: str | None = None,
		temperature: float = 0.2,
		max_new_tokens: int = 256,
	) -> None:
		project_name = self.project_name_for(engine_name)
		super().__init__(project_name=project_name)
		self.engine_name = engine_name
		self.model = model or self.model_name
		self.api_key = api_key
		self.temperature = float(temperature)
		self.max_new_tokens = int(max_new_tokens)
		self._model_client: Any | None = None
		self.initialize_model()

	@abstractmethod
	def initialize_model(self) -> None:
		"""Initialize provider-specific model/client state."""

	@abstractmethod
	def call(self, prompt: str, **kwargs: Any) -> str:
		"""Run one generation call and return plain text."""


class HuggingFaceInferenceLLM(BaseLLM):
	"""Free-tier Hugging Face Inference API caller."""

	def initialize_model(self) -> None:

		self.model = self.DEFAULT_LLM_HUGGING_FACE_MODEL_NAME
		self.api_key = self.api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
		# meta-llama/Llama-3.2-1B-Instruct is only live on featherless-ai; auto provider
		# selection skips providers not explicitly enabled on the account, so pin it.
		self._model_client = InferenceClient(api_key=self.api_key, provider="featherless-ai")

	def call(self, prompt: str, **kwargs: Any) -> str:
		response = self._model_client.chat_completion(
			messages=[{"role": "user", "content": str(prompt)}],
			model=self.model,
			max_tokens=int(kwargs.get("max_new_tokens", self.max_new_tokens)),
			temperature=float(kwargs.get("temperature", self.temperature)),
		)
		print(f"HF Inference API response: {response}")
		return str(response.choices[0].message.content or "").strip()


class OllamaLLM(BaseLLM):
	"""Local Ollama caller (no paid API key required)."""

	def __init__(
		self,
		engine_name: str = "default",
		*,
		model: str = "llama3.1:8b",
		base_url: str = "http://localhost:11434",
		**kwargs: Any,
	) -> None:
		self.base_url = base_url.rstrip("/")
		super().__init__(engine_name=engine_name, model=model, **kwargs)

	def initialize_model(self) -> None:
		# Ollama is contacted over HTTP; no warmup required here.
		self._model_client = {"base_url": self.base_url}

	def call(self, prompt: str, **kwargs: Any) -> str:
		payload = {
			"model": self.model,
			"prompt": str(prompt),
			"stream": False,
			"options": {
				"temperature": float(kwargs.get("temperature", self.temperature)),
				"num_predict": int(kwargs.get("max_new_tokens", self.max_new_tokens)),
			},
		}

		req = request.Request(
			url=f"{self.base_url}/api/generate",
			data=json.dumps(payload).encode("utf-8"),
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		with request.urlopen(req, timeout=int(kwargs.get("timeout", 120))) as resp:
			body = json.loads(resp.read().decode("utf-8"))
		return str(body.get("response", "")).strip()


class GoogleGeminiLLM(BaseLLM):
	"""Google Gemini caller via the google-genai SDK."""

	def initialize_model(self) -> None:
		self.model = self.DEFAULT_LLM_GOOGLE_MODEL_NAME
		self.api_key = self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
		if not self.api_key:
			raise ValueError("GoogleGeminiLLM requires GOOGLE_API_KEY (or GEMINI_API_KEY) to be set")
		self._model_client = genai.Client(api_key=self.api_key)
		self.temperature = float(self.DEFAULT_GEMINI_TEMPERATURE) if self.temperature is None else float(self.temperature)
		self.model = self.model or self.DEFAULT_LLM_GOOGLE_MODEL_NAME
		self.max_output_tokens = int(self.max_new_tokens) if self.max_new_tokens is not None else int(self.DEFAULT_GEMINI_MAX_TOKENS)

	def call(self, prompt: str, **kwargs: Any) -> str:
		response = self._model_client.models.generate_content(
			model=self.model,
			contents=str(prompt),
			config=genai_types.GenerateContentConfig(
				temperature=self.temperature,
				max_output_tokens=self.max_output_tokens,
			),
		)
		return str(response.text or "").strip()


class ClaudeLLM(BaseLLM):
	"""Anthropic Claude caller via the official ``anthropic`` SDK.

	Reads the API key from ``CLAUDE_KEY`` (falling back to ``ANTHROPIC_API_KEY``)
	and defaults to the model named in ``DEFAULT_LLM_CLAUDE_MODEL_NAME``.
	"""

	def initialize_model(self) -> None:
		# BaseLLM seeds self.model with the embedding default; swap in the Claude default.
		if not self.model or self.model == self.DEFAULT_MODEL_NAME:
			self.model = self.DEFAULT_LLM_CLAUDE_MODEL_NAME
		self.api_key = self.api_key or os.getenv("CLAUDE_KEY") or os.getenv("ANTHROPIC_API_KEY")
		if not self.api_key:
			raise ValueError("ClaudeLLM requires CLAUDE_KEY (or ANTHROPIC_API_KEY) to be set")
		self._model_client = anthropic.Anthropic(api_key=self.api_key)
		self.temperature = float(self.DEFAULT_LLM_CLAUDE_TEMPERATURE) if self.temperature is None else float(self.temperature)
		self.max_output_tokens = int(self.max_new_tokens) if self.max_new_tokens is not None else int(self.DEFAULT_LLM_CLAUDE_MAX_TOKENS)
		self.model = self.model or self.DEFAULT_LLM_CLAUDE_MODEL_NAME

	def call(self, prompt: str, **kwargs: Any) -> str:
		response = self._model_client.messages.create(
			model=self.model,
			max_tokens=self.max_output_tokens,
			temperature=self.temperature,
			messages=[{"role": "user", "content": str(prompt)}],
		)
		return "".join(
			block.text for block in response.content if block.type == "text"
		).strip()


class TransformersLocalLLM(BaseLLM):
	"""Local free text-generation via Hugging Face Transformers pipeline."""

	def __init__(
		self,
		engine_name: str = "default",
		*,
		model: str = "distilgpt2",
		device: int = -1,
		**kwargs: Any,
	) -> None:
		self.device = int(device)
		super().__init__(engine_name=engine_name, model=model, **kwargs)

	def initialize_model(self) -> None:
		Configs.configure_hf_environment()
		cache_dir = str(Path(self.HF_CACHE_DIR) / "models")
		self._model_client = pipeline(
			task="text-generation",
			model=self.model,
			tokenizer=self.model,
			device=self.device,
			model_kwargs={"cache_dir": cache_dir},
		)

	def call(self, prompt: str, **kwargs: Any) -> str:
		outputs = self._model_client(
			str(prompt),
			max_new_tokens=int(kwargs.get("max_new_tokens", self.max_new_tokens)),
			temperature=float(kwargs.get("temperature", self.temperature)),
			do_sample=bool(kwargs.get("do_sample", True)),
			return_full_text=bool(kwargs.get("return_full_text", False)),
		)
		if not outputs:
			return ""
		return str(outputs[0].get("generated_text", "")).strip()


class GPT2CausalLocalLLM(BaseLLM):
	"""Prompt-only local GPT-2 generation (no retrieval/database dependency)."""

	def __init__(
		self,
		engine_name: str = "default",
		*,
		model: str = "gpt2",
		device: str | None = None,
		**kwargs: Any,
	) -> None:
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.gen_tokenizer: Any | None = None
		self.gen_model: Any | None = None
		super().__init__(engine_name=engine_name, model=model, **kwargs)

	def initialize_model(self) -> None:
		Configs.configure_hf_environment()
		cache_dir = str(Path(self.HF_CACHE_DIR) / "models")
		self.gen_tokenizer = AutoTokenizer.from_pretrained(self.model, cache_dir=cache_dir)
		self.gen_model = AutoModelForCausalLM.from_pretrained(self.model, cache_dir=cache_dir)
		# GPT-2 has no pad token by default; set EOS as pad for generation stability.
		if self.gen_tokenizer.pad_token is None:
			self.gen_tokenizer.pad_token = self.gen_tokenizer.eos_token
		self.gen_model = self.gen_model.to(self.device)
		self.gen_model.eval()
		self._model_client = self.gen_model

	def call(self, prompt: str, **kwargs: Any) -> str:
		if self.gen_tokenizer is None or self.gen_model is None:
			raise RuntimeError("GPT2CausalLocalLLM is not initialized")

		inputs = self.gen_tokenizer(str(prompt), return_tensors="pt")
		inputs = {k: v.to(self.device) for k, v in inputs.items()}

		with torch.no_grad():
			generation_output = self.gen_model.generate(
				**inputs,
				max_new_tokens=int(kwargs.get("max_new_tokens", self.max_new_tokens)),
				do_sample=bool(kwargs.get("do_sample", False)),
				temperature=float(kwargs.get("temperature", self.temperature)),
			)

		return str(
			self.gen_tokenizer.decode(generation_output[0], skip_special_tokens=True)
		).strip()


class DistilGPT2LocalLLM(GPT2CausalLocalLLM):
	"""Prompt-only local DistilGPT-2 preset."""

	def __init__(self, engine_name: str = "default", **kwargs: Any) -> None:
		super().__init__(engine_name=engine_name, model="distilgpt2", **kwargs)


class GPT2MediumLocalLLM(GPT2CausalLocalLLM):
	"""Prompt-only local GPT-2 Medium preset."""

	def __init__(self, engine_name: str = "default", **kwargs: Any) -> None:
		super().__init__(engine_name=engine_name, model="gpt2-medium", **kwargs)


class GPT2LargeLocalLLM(GPT2CausalLocalLLM):
	"""Prompt-only local GPT-2 Large preset."""

	def __init__(self, engine_name: str = "default", **kwargs: Any) -> None:
		super().__init__(engine_name=engine_name, model="gpt2-large", **kwargs)


class GPT2XLLocalLLM(GPT2CausalLocalLLM):
	"""Prompt-only local GPT-2 XL preset."""

	def __init__(self, engine_name: str = "default", **kwargs: Any) -> None:
		super().__init__(engine_name=engine_name, model="gpt2-xl", **kwargs)


class TinyGPT2LocalLLM(GPT2CausalLocalLLM):
	"""Prompt-only tiny GPT-2 preset for quick CPU tests."""

	def __init__(self, engine_name: str = "default", **kwargs: Any) -> None:
		super().__init__(engine_name=engine_name, model="sshleifer/tiny-gpt2", **kwargs)


class TinyStories33MLocalLLM(GPT2CausalLocalLLM):
	"""Very small TinyStories 33M preset for low-resource local runs."""

	def __init__(self, engine_name: str = "default", **kwargs: Any) -> None:
		super().__init__(engine_name=engine_name, model="roneneldan/TinyStories-33M", **kwargs)


class TinyStories1MLocalLLM(GPT2CausalLocalLLM):
	"""Ultra-small TinyStories 1M preset for minimal local footprint."""

	def __init__(self, engine_name: str = "default", **kwargs: Any) -> None:
		super().__init__(engine_name=engine_name, model="roneneldan/TinyStories-1M", **kwargs)


class SmolLM135MLocalLLM(GPT2CausalLocalLLM):
	"""Small instruction model that can run locally on CPU/GPU."""

	def __init__(self, engine_name: str = "default", **kwargs: Any) -> None:
		super().__init__(
			engine_name=engine_name,
			model="HuggingFaceTB/SmolLM2-135M-Instruct",
			**kwargs,
		)


class Qwen25HalfBLocalLLM(GPT2CausalLocalLLM):
	"""Compact Qwen 2.5 0.5B instruct model preset."""

	def __init__(self, engine_name: str = "default", **kwargs: Any) -> None:
		super().__init__(engine_name=engine_name, model="Qwen/Qwen2.5-0.5B-Instruct", **kwargs)


FREE_LLM_REGISTRY: dict[str, type[BaseLLM]] = {
	"google": GoogleGeminiLLM,
	"claude": ClaudeLLM,
	"anthropic": ClaudeLLM,
	"huggingface": HuggingFaceInferenceLLM,
	"ollama": OllamaLLM,
	"transformers_local": TransformersLocalLLM,
	"gpt2_local": GPT2CausalLocalLLM,
	"distilgpt2_local": DistilGPT2LocalLLM,
	"gpt2_medium_local": GPT2MediumLocalLLM,
	"gpt2_large_local": GPT2LargeLocalLLM,
	"gpt2_xl_local": GPT2XLLocalLLM,
	"tiny_gpt2_local": TinyGPT2LocalLLM,
	"tinystories_33m_local": TinyStories33MLocalLLM,
	"tinystories_1m_local": TinyStories1MLocalLLM,
	"smollm_135m_local": SmolLM135MLocalLLM,
	"qwen25_05b_local": Qwen25HalfBLocalLLM,
}


def create_llm(provider: str, engine_name: str = "default", **kwargs: Any) -> BaseLLM:
	key = str(provider).strip().lower()
	if key not in FREE_LLM_REGISTRY:
		available = ", ".join(sorted(FREE_LLM_REGISTRY.keys()))
		raise ValueError(f"Unknown free LLM provider '{provider}'. Available: {available}")
	return FREE_LLM_REGISTRY[key](engine_name=engine_name, **kwargs)

