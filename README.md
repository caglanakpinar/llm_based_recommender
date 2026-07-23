# LLM-Based Recommender

A **Retrieval-Augmented Generation (RAG) recommender**: instead of training a ranking
model, user/item/interaction data is turned into text prompts, embedded into a vector
space, and indexed. At request time, relevant context is retrieved for a given
user/item pair and handed to an LLM, which returns a relevance score used to rank
candidates. Swapping the embedding model or the LLM provider does not require
retraining anything — both are pluggable via small factory functions.

The implementation lives entirely under [`core2/`](core2/) and is served through a
Streamlit UI ([`app.py`](app.py)).

> **Status:** this branch (`dev_core2`) is mid-migration from an older `core/` package
> to `core2/`. A few root-level modules (`app.py`, `reco_engine.py`, `recommender.py`,
> `embedding_store.py`, `schema_defaults.py`) still import `from core...`, which no
> longer exists — those need to be repointed at `core2` before `run_app.sh` will work
> again. Everything under `core2/` is unaffected and works standalone.

## Architecture (RAG pipeline)

```
DataSets (parquet: items, users, interactions)                         core2/datasets.py
        │
        ▼
ItemPrompt / UserPrompt / UserItemPrompt                               core2/prompting.py
        │  feature engineering (core2/features.py) + per-row text      core2/features.py
        │  rendered from item.md / user.md / user_item_pair.md
        ▼
RelevanceScorePrompt                                                   core2/prompting.py
        │  merges item/user/interaction prompts into one corpus,
        │  computes a relevance_score label, and renders one
        │  "generated_prompt" per (user, item) row — this corpus
        │  is the RAG knowledge base
        ▼
   ┌────────────────────────┬─────────────────────────┐
   ▼                        ▼
ContextVectorDB (FAISS)   ContextDB (Chroma)                           core2/dbs.py
   embeds every               stores the same rows as
   generated_prompt via       text + metadata for
   an Embedder (below)        text-based lookup
   and indexes the vectors
        │
        ▼
Retrieval                                                               core2/retrieval.py
        │  per user: candidates = last-interacted items ∪ FAISS
        │  nearest-neighbor "similar" items (self-match excluded)
        │  retrieve_context(query) embeds a query and does the
        │  actual RAG similarity search against ContextVectorDB
        ▼
LLMRanker.generate_scores(user_id, item_id)                             core2/ranking.py
        │  builds a query prompt, retrieves RAG context, assembles
        │  "Context: ... Question: ... Answer:" and calls the LLM
        ▼
BaseLLM subclass (below)                                                core2/llms.py
        ▼
RecoEnginePredictor.predict({user_id, top_k})                           core2/reco_engine.py
        a kserve.Model: retrieves 2×top_k candidates, scores each
        via the LLM, sorts by score, returns the top_k items
```

### Embeddings (`core2/embeddings.py`)

All embedders implement `BaseEmbedder.text_to_vector(texts) -> np.ndarray` and return
L2-normalized `float32` vectors by default, so FAISS similarity search is consistent
regardless of which one is active. Selected via `create_embedder(name, engine_name)`
through the `EMBEDDER_REGISTRY`:

| Registry key | Class | Backing model | Notes |
|---|---|---|---|
| `sentence_transformer` (default) | `SentenceTransformerEmbedder` | `BAAI/bge-small-en-v1.5` | General-purpose semantic encoder, used by both `ContextVectorDB` and `ContextDB`. |
| `bge` | `BGEEmbedder` | `BAAI/bge-small-en-v1.5` | Named preset of the same model; strong retrieval default. |
| `e5` | `E5Embedder` | `intfloat/e5-small-v2` | Query/document-style retrieval encoder. |
| `gte` | `GTEEmbedder` | `thenlper/gte-small` | Lightweight alternative for balanced speed/quality. |
| `hf_mean_pool` | `HuggingFaceMeanPoolEmbedder` | any `AutoModel` (default `BAAI/bge-small-en-v1.5`) | Raw Transformers forward pass + attention-masked mean pooling, for backbones not wrapped by `sentence-transformers`. |
| `hashing` | `HashingEmbedder` | none (MD5 token hashing) | Dependency-free, deterministic, lexical (not semantic) fallback for tests/offline environments. |

Model downloads are cached under `.cache/huggingface` (`Configs.configure_hf_environment()`).

### Vector storage & retrieval (`core2/dbs.py`, `core2/retrieval.py`)

Two databases are built from the same `generated_prompt` corpus, in parallel:

- **`ContextVectorDB`** (`BaseFaissDB`) — a FAISS `IndexFlatL2` (or `IndexFlatIP`)
  index. `write_context_vectors()` embeds every row's `generated_prompt` and adds it
  to the index, keyed by `f"{user_id}_{item_id}"`; `search_vectors(query_vector, k)`
  does the nearest-neighbor search.
- **`ContextDB`** (`BaseChromaDB`) — a persistent Chroma collection storing the same
  rows as raw text plus full-row metadata, for text-native queries.

`Retrieval` sits on top of both and serves two distinct needs:

- `users_last_interactions()` (run once at build time) precomputes, per user, their
  historical items plus FAISS "similar" items — searching outward from `k+1` and
  expanding until it finds a genuinely different item, since the query prompt is
  itself indexed and would otherwise just match itself. The union becomes each user's
  `candidates` set, served by `retrieve_candidates(user_id, top_k)`.
- `retrieve_context(query_text, k)` embeds an arbitrary query at request time and
  searches `ContextVectorDB` for the `k` most relevant context rows — this is the
  "R" in RAG, called from `LLMRanker.generate_scores()` for every candidate scored.

### LLMs (`core2/llms.py`)

`BaseLLM` is a small ABC: subclasses implement `initialize_model()` (set up a
client) and `call(prompt, **kwargs) -> str` (run one generation and return text).
Selected via `create_llm(provider, engine_name)` through the `FREE_LLM_REGISTRY`:

| Registry key | Class | Backend | API key |
|---|---|---|---|
| `claude` / `anthropic` (default) | `ClaudeLLM` | `claude-sonnet-4-5` via the `anthropic` SDK | `CLAUDE_KEY` or `ANTHROPIC_API_KEY` |
| `google` | `GoogleGeminiLLM` | `gemini-3.5-flash` via the `google-genai` SDK | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| `huggingface` | `HuggingFaceInferenceLLM` | `meta-llama/Llama-3.2-1B-Instruct` via HF Inference API (pinned to the `featherless-ai` provider) | `HF_TOKEN` or `HUGGINGFACE_API_TOKEN` |
| `ollama` | `OllamaLLM` | any locally-running Ollama model (default `llama3.1:8b`) over HTTP | none |
| `transformers_local` | `TransformersLocalLLM` | any local HF `text-generation` pipeline (default `distilgpt2`) | none |
| `gpt2_local`, `distilgpt2_local`, `gpt2_medium_local`, `gpt2_large_local`, `gpt2_xl_local`, `tiny_gpt2_local`, `tinystories_33m_local`, `tinystories_1m_local`, `smollm_135m_local`, `qwen25_05b_local` | `GPT2CausalLocalLLM` presets | local `AutoModelForCausalLM` checkpoints, CPU/GPU auto-detected | none |

The default provider is `Configs.DEFAULT_LLM_MODEL_NAME` (`claude`); pass a
registry key to `create_llm(provider, engine_name)` to pick another.

`LLMRanker` calls whichever provider is configured with the assembled RAG prompt and
parses the first float out of the response as the relevance score (`_parse_score`).
Because every provider implements the same two-method contract, adding a new LLM is
just a new `BaseLLM` subclass plus a registry entry.

> **Thinking models and token budgets** — Gemini 3.x spends reasoning tokens from
> the same `max_output_tokens` budget as the answer, so a short-answer call
> (like the ranker's "reply with a score") can return an **empty string** that
> `_parse_score` turns into `0.0`. `GoogleGeminiLLM` therefore defaults to
> `thinking_budget=0` (`Configs.DEFAULT_GEMINI_THINKING_BUDGET`) and logs a
> warning whenever a reply comes back empty. Pass `thinking_budget=None` to let
> the model think, and give it ≥256 output tokens if you do.

## Package layout (`core2/`)

| File | Purpose |
|---|---|
| `configs.py` | `Configs` — per-engine settings persisted to `<engine>/docs/configs.yaml`; default paths, model names, HF cache setup. |
| `datasets.py` | `DataSets` — loads item/user/interaction parquet (with sample-data fallback) into DataFrames. |
| `features.py` | Pandas feature functions (`item_features`, `user_features`, `user_item_pair_features`) and the `relevance_score(...)` label used for prompting. |
| `prompting.py` | `BasePrompt`/`UserPrompt`/`ItemPrompt`/`UserItemPrompt`/`RelevanceScorePrompt` — render `.md` templates into per-row text prompts and build the RAG context corpus. Each stage caches to `<engine_root>/cache/*.parquet` — see the caching note below. |
| `embeddings.py` | Embedder registry — see [Embeddings](#embeddings-core2embeddingspy) above. |
| `dbs.py` | `ContextVectorDB` (FAISS) and `ContextDB` (Chroma) — see [Vector storage & retrieval](#vector-storage--retrieval-core2dbspy-core2retrievalpy) above. |
| `retrieval.py` | `Retrieval` — candidate generation and RAG context lookups — see above. |
| `llms.py` | LLM provider registry — see [LLMs](#llms-core2llmspy) above. |
| `ranking.py` | `BaseRelevanceRanking` (feature-based scoring) and `LLMRanker` (RAG+LLM scorer: `generate_scores(user_id, item_id)`). |
| `reco_engine.py` | `RecoEnginePredictor` (a `kserve.Model` implementing `predict()`) and `BuildRecoEngine` — wires datasets/retrieval/ranker together and serves predictions. |
| `logger.py` | Shared module logger. |
| `item.md`, `user.md`, `user_item_pair.md`, `relevance_score_generator.md` | Prompt template fragments consumed by `prompting.py`. |

> **Prompt cache is keyed by engine name only.** `build_*_feature_dataset()` and
> `generate_rag_retrieval_context()` persist to `<engine_root>/cache/*.parquet` and
> reload them whenever the file exists — regardless of whether the underlying data
> changed. Rebuilding an engine **under the same name with different data** will
> silently reuse the old prompts and RAG context. Delete `<engine_root>/cache/` or
> pass `use_cache=False` when the data changes (the benchmark always does).

## UI

The UI is a **Streamlit** app (`app.py`, `streamlit run app.py`) with two pages:

- **Reco Engine Generator** (builder page) — upload or point to parquet
  items/users/interactions, map columns, set `top_k` and free-text ranking
  constraints, then build a named engine. This persists `<name>/docs/configs.yaml`,
  runs the full pipeline above to build the vector/text indexes, and starts a
  background thread serving a `RecoEnginePredictor` (KServe model) for that engine.
- **Reco Generator** — pick a built engine and a target user, and get back ranked
  item recommendations (score + signals) rendered as a table.

## Setup

Requires Python 3.11–3.12 and [Poetry](https://python-poetry.org/).

```bash
poetry install
```

### Environment variables

| Variable | Used for | Required? |
|---|---|---|
| `CLAUDE_KEY` or `ANTHROPIC_API_KEY` | `ClaudeLLM` (`claude-sonnet-4-5`) — the **default** LLM provider | Yes, unless you switch providers |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `GoogleGeminiLLM` (`gemini-3.5-flash`) — the provider the benchmark defaults to | Only if using the `google` provider |
| `HF_TOKEN` / `HUGGINGFACE_API_TOKEN` | Hugging Face model downloads (embeddings) and `HuggingFaceInferenceLLM` | Only if using HF-hosted inference or gated models |

Local providers (`ollama`, local GPT-2/TinyStories/SmolLM/Qwen models) need no API key
— see the LLMs table above, and pass the registry key to `create_llm(provider, engine_name)`
to select one.

## Usage

### Streamlit app

```bash
./run_app.sh
# or: poetry run streamlit run app.py
```

(See the migration note above — root-level imports need to be fixed on this branch
before this works.)

### Programmatic

Once an engine's datasets and prompts are built, the pipeline can be driven directly:

```python
from core2.datasets import DataSets
from core2.prompting import ItemPrompt, UserPrompt, UserItemPrompt, RelevanceScorePrompt
from core2.dbs import ContextVectorDB, ContextDB
from core2.retrieval import Retrieval
from core2.ranking import LLMRanker
from core2.reco_engine import BuildRecoEngine

engine_name = "my_engine"
datasets = DataSets(engine_name).get_data()
# build ItemPrompt / UserPrompt / UserItemPrompt, then RelevanceScorePrompt
# to get a `context_prompts` object with a generated RAG corpus ...

context_vector_db = ContextVectorDB(engine_name, dimension=384, prompt=context_prompts)
context_vector_db.write_context_vectors()
context_db = ContextDB(engine_name, dimension=384, prompt=context_prompts)
context_db.write_context()

retrieval = Retrieval(engine_name, datasets, context_prompts, context_vector_db, context_db)
ranker = LLMRanker(engine_name, datasets, retrieval, context_prompts)  # defaults to Configs.DEFAULT_LLM_MODEL_NAME
eng = BuildRecoEngine(engine_name, datasets, retrieval, ranker, context_prompts)

predictor = eng.initialize_kserve_api()
response = predictor.predict({"user_id": "user_00004", "top_k": 5})
```

## Benchmark

An offline ranking benchmark lives under [`benchmark/`](benchmark/) — see its
[README](benchmark/README.md) for the full protocol. In short: a temporal split
(most-recent 30% of each user's interactions held out), strong-positive actions
(`purchase`/`add_to_cart`/`like`) as ground truth, and every engine ranking the
**same** candidate pool per user (relevant items + 20 sampled negatives), scored
with NDCG / MAP / MRR / Precision / Recall / HitRate `@k`.

### Results (2026-07-22 run)

Setup: 100 sampled users, candidate pool of 2,179 pairs (179 positives), `seed=42`.
Baselines fit on the full-population training split (2,105 interactions). The LLM
engine (`core2` with `gemini-3.1-flash-lite`, `sentence_transformer` embedder,
temp 0.2) was run head-to-head on the **first 10** sampled users (221 candidate
pairs, one LLM call each) — see [why](#llm-runs-on-a-subset) below.

**Head-to-head (same 10 users, identical 221-pair candidate pool):**

| Engine | NDCG@10 | MAP@10 | MRR |
|---|---|---|---|
| **LLM engine (`core2`)** | **0.3276** | **0.1762** | **0.2574** |
| Two-Tower (neural, engineered features) | 0.1449 | 0.0740 | 0.2044 |
| Item-KNN CF | 0.1279 | 0.0534 | 0.1398 |

The Two-Tower row comes from the config workflow (`run_config`), which scores the
*same* shared 10-user pool, so it drops straight into this table. The full
feature-ranker leaderboard is described below (`gbdt_hist_gbdt` reaches
NDCG@10 = 0.2947 on the same pool).

**Item-KNN CF over all 100 sampled users (appendix — does the ranking hold at scale?):**

| Engine | NDCG@10 | MAP@10 | MRR |
|---|---|---|---|
| Item-KNN CF | 0.2077 | 0.1237 | 0.2225 |

<a name="llm-runs-on-a-subset"></a>
- **Why the LLM runs on a subset**: the ranker makes one API call per candidate,
  so all 100 users would be ~2,200 calls — far beyond the Gemini free-tier daily
  quota (`gemini-3.5-flash` was **20 requests/day** at the time of writing). The
  head-to-head therefore uses 10 users on the cheaper `gemini-3.1-flash-lite`.
  Successful scores are cached in `benchmark/results/llm_score_cache.json`, so
  coverage can be extended across days by re-running:

  ```bash
  export GEMINI_API_KEY=...   # or: set -a; . ./.env; set +a
  poetry run python -m benchmark.run_benchmark --n-users 100 --n-negatives 20 \
      --llm-model gemini-3.1-flash-lite --llm-max-users 10 --llm-rpm 12
  ```

- **Read these numbers with care.** (1) The bundled `data/` is synthetically
  generated (~3K random interactions, 200 users, 100 items), so there is little
  real collaborative/semantic signal — every engine sits near the random floor
  (a uniform-random ranker scores NDCG@10 ≈ 0.30 on this pool). (2) The LLM is
  scored on only 10 users, so its lead is *indicative, not significant*. (3)
  `gemini-3.1-flash-lite` emits weakly
  discriminative scores here (most cluster near ~0.12), so the ranking is partly
  noise-driven. Point `benchmark/data_prep.py` at real logs and run the LLM on all
  users for a meaningful comparison; the harness, splits, and metrics are
  production-shaped.

A second, config-driven workflow (`benchmark/run_config.py` + `benchmark/leaderboard.py`)
compares ranker variants — LLM (embedder × provider × temperature), a neural
two-tower, and GBDT rankers — on the same shared 10-user pool (feature rankers need
no API key). Its leaderboard (`benchmark/results/leaderboard.md`):

| Config | NDCG@10 | MAP@10 | MRR |
|---|---|---|---|
| `gbdt_hist_gbdt` | 0.2947 | 0.1525 | 0.2210 |
| `two_tower` | 0.1449 | 0.0740 | 0.2044 |

## Config files

`Configs` persists/loads YAML at **`<engine_root>/docs/configs.yaml`**, where
`<engine_root>` is a folder named after the engine directly under the repo root (e.g.
`test2_reco_engine/docs/configs.yaml`). It captures `reco_engine_name`, data/store
paths, `prompt_path`, the embedding `model_name`, embedding artifact file paths,
`prompt_placeholder_sections` (maps prompt-input keys to template sections),
`user_profile_columns`/`item_catalog_columns` (parquet column mapping), `top_k`,
free-text `constraints`, source parquet folders, and `created_at`. Each engine
created via `Configs.create()` or the Builder UI gets its own such file, so multiple
named engines coexist independently.
