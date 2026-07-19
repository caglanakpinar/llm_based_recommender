# Offline Recommender Benchmark

Compares a **classic baseline recommender** against the **LLM-based recommendation
engine** (`core2`) on the datasets under [`data/`](../data), using standard
offline ranking metrics: **NDCG, MAP, MRR, Precision, Recall, HitRate** (each `@k`).

Everything lives under this `benchmark/` folder and writes artifacts to
[`benchmark/results/`](results).

---

## What it does (the 4 steps)

1. **Baseline engine** — builds classic recommenders from `data/` (Item-KNN
   collaborative filtering + Popularity + a Random floor), trained on the
   full-population *training* split.
2. **Sample 100 users** and generate recommendations with the baseline.
3. **Same 100 users through the LLM engine** — builds the item/user/user-item
   prompts, the relevance-context, the FAISS vector DB and retrieval, then calls
   the LLM ranker per user–item pair (the same pipeline as [`test_api.py`](../test_api.py),
   but seeded with the benchmark's training data and callable per user).
4. **Offline evaluation** — both engines score the *same* candidate pool per user;
   we compute ranking metrics against held-out ground truth and produce a report.

## Evaluation protocol

Standard **sampled re-ranking** (the protocol used by NCF and many RecSys papers):

- **Temporal split** — each user's interactions are sorted by time; the most
  recent `test_frac` (default 30%) is held out. Nothing from the test window is
  ever used for training (no leakage).
- **Ground truth** — a user's *relevant* items are the distinct items they had a
  **strong-positive** action on (`purchase` / `add_to_cart` / `like`) during the
  test window, that were **not** already seen in training.
- **Candidate pool** — the relevant items + `n_negatives` sampled negatives
  (items the user never interacted with). Both engines rank this *identical* pool,
  so the comparison isolates ranking quality from candidate generation.
- **Metrics** — macro-averaged over users at each cut-off `k`.

### Fairness notes

- Both engines see the **same information**: the training-split interactions only.
- The **baseline** is fit on the *full population's* training interactions (that is
  how a CF baseline is really deployed, and item–item co-occurrence needs
  population scale). The **LLM engine** builds its retrieval context from the
  sampled users' training interactions (it is content/retrieval-based, so it does
  not need the whole population — and embedding 140K rows would be needlessly slow).
- The **head-to-head** table scores every engine on the *same* users and the *same*
  candidate pool. Baselines are additionally reported over all 100 sampled users
  (appendix), since they are cheap to run at full scale.

## Why the LLM runs on a subset by default

The LLM ranker calls a local model (**Ollama**) once per user–item candidate.
At ~4–5 s/pair, all 100 users (~2,300 pairs) take a few hours, so the LLM
head-to-head is capped to `--llm-max-users 25` by default. LLM scores are cached
to `results/llm_score_cache.json`, so the run is **resumable**: re-running (or
raising the cap) reuses cached pairs and only computes new ones. Pass
`--llm-max-users 0` to run the full 100.

## Requirements

Run inside the project's **Poetry** environment (Python 3.11 — the repo's `.venv`
is 3.13 and lacks `chromadb`/`kserve`):

```bash
poetry install
```

- **LLM provider** — defaults to `ollama` with `llama3.2:latest` (free, local, no
  API key). Ensure Ollama is running and the model is pulled:
  `ollama pull llama3.2`. To use Gemini instead: `--llm-provider google
  --llm-model gemini-2.0-flash` with `GOOGLE_API_KEY` set.

## Usage

```bash
# Full run: baselines on all 100 users + LLM head-to-head on 25 users
poetry run python -m benchmark.run_benchmark --n-users 100 --n-negatives 20

# Baselines only (fast, no LLM)
poetry run python -m benchmark.run_benchmark --n-users 100 --skip-llm

# LLM head-to-head on all 100 users (slow; resumable via cache)
poetry run python -m benchmark.run_benchmark --n-users 100 --llm-max-users 0
```

### Key flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--n-users` | 100 | users to sample for evaluation |
| `--test-frac` | 0.3 | fraction of each user's most-recent interactions held out |
| `--n-negatives` | 20 | negatives per user in the candidate pool |
| `--negative-sampling` | `uniform` | `uniform` or `popularity` (harder negatives) |
| `--k` | `5,10` | ranking cut-offs |
| `--llm-provider` / `--llm-model` | `ollama` / `llama3.2:latest` | LLM backend |
| `--llm-max-users` | 25 | cap LLM head-to-head (0 = all) |
| `--skip-llm` / `--skip-baseline` | off | run only one family |
| `--seed` | 42 | reproducibility |

## Outputs (`benchmark/results/`)

| File | Contents |
| --- | --- |
| `report.md` | human-readable comparison report |
| `metrics.json` | all metrics + run metadata |
| `eval_pool.parquet` | the candidate pool (`user_id, item_id, label`) |
| `sampled_users.json` | the sampled user ids |
| `relevants.json` | per-user ground-truth relevant items |
| `llm_score_cache.json` | cached LLM scores (makes reruns cheap/resumable) |

## Module layout

| File | Responsibility |
| --- | --- |
| `data_prep.py` | load data, sample users, temporal split, ground truth + candidate pool; shared-dataset persistence |
| `baseline.py` | Popularity, Item-KNN CF, Random recommenders |
| `llm_engine.py` | build the `core2` LLM pipeline (parameterized embedder + LLM) + cached per-pair scorer |
| `metrics.py` | NDCG / MAP / MRR / Precision / Recall / HitRate |
| `evaluate.py` | shared scoring→ranking→metrics helpers |
| `configs.py` | `RankerConfig` + the sample config grid |
| `rankers/features.py` | feature builder over the already-engineered prompt features |
| `rankers/two_tower.py` | neural two-tower ranker (torch) |
| `rankers/gbdt.py` | CatBoost / sklearn-HistGBDT ranker |
| `rankers/labels.py` | labeled (pos/neg) training-pair sampler |
| `run_benchmark.py` | baseline-vs-LLM orchestrator, CLI, report |
| `run_config.py` | run ONE ranker config against the shared pool |
| `leaderboard.py` | aggregate all config metrics into one comparison table |

---

## Comparing ranker configs (LLM embedder × model × temperature, Two-Tower, CatBoost)

A second, config-driven workflow compares **ranker variants** on the *same* shared
candidate pool. Three ranker kinds:

- **`llm`** — the `core2` LLM ranker, parameterized by `(embedder, llm_provider,
  llm_model, temperature)`. Named `llm_<embedder>_<provider>_<model>_<temperature>`.
- **`two_tower`** — a user-tower/item-tower neural ranker over the engineered features.
- **`gbdt`** — a gradient-boosted ranker (CatBoost, or sklearn HistGBDT) over the
  engineered features (uses the cross/pair features too).

The Two-Tower and GBDT rankers consume the **features already created at the
prompt-engineering stage** (`core2.features.FEATURES`, via `rankers/features.py`) —
nothing is recomputed.

### Sample config grid

```bash
poetry run python -m benchmark.run_config --list
```

| Config | kind | detail |
| --- | --- | --- |
| `llm_sentence_transformer_google_gemini-3.5-flash_0.7` | llm | sentence_transformer · gemini-3.5-flash · temp 0.7 |
| `llm_sentence_transformer_google_gemini-3.5-flash_0.0` | llm | sentence_transformer · gemini-3.5-flash · temp 0.0 |
| `llm_hf_mean_pool_google_gemini-3.5-flash_0.7` | llm | hf_mean_pool · gemini-3.5-flash · temp 0.7 |
| `llm_hf_mean_pool_google_gemini-3.5-flash_0.0` | llm | hf_mean_pool · gemini-3.5-flash · temp 0.0 |
| `llm_sentence_transformer_ollama_llama3.2-latest_0.2` | llm | keyless local LLM (no API key) |
| `two_tower` | two_tower | neural two-tower over engineered features |
| `gbdt_catboost` | gbdt | CatBoost (needs `poetry add catboost`) |
| `gbdt_hist_gbdt` | gbdt | sklearn HistGradientBoosting (no extra install) |

Edit the grid or add points in [`configs.py`](configs.py) (`build_sample_configs`).

### Run configs one at a time, then compare

```bash
# feature rankers (no API key needed)
poetry run python -m benchmark.run_config --config two_tower
poetry run python -m benchmark.run_config --config gbdt_hist_gbdt

# LLM configs (Google needs GOOGLE_API_KEY; Ollama is keyless)
poetry run python -m benchmark.run_config --config llm_sentence_transformer_ollama_llama3.2-latest_0.2
export GOOGLE_API_KEY=...   # for the Gemini configs
poetry run python -m benchmark.run_config --config llm_sentence_transformer_google_gemini-3.5-flash_0.7
poetry run python -m benchmark.run_config --config llm_hf_mean_pool_google_gemini-3.5-flash_0.0

# aggregate everything into one leaderboard
poetry run python -m benchmark.leaderboard --sort ndcg@10
```

- The **shared dataset** (sampled users + candidate pool) is built once and cached
  under `results/shared/`, so every config is compared on identical users/pairs.
- Each config writes `results/configs/<name>/metrics.json`; `leaderboard.py`
  collects them into `results/leaderboard.md`.
- All configs are evaluated on the same `--llm-max-users` set (default 25) for a
  strict head-to-head; pass `--llm-max-users 0` to use all sampled users.
- **CatBoost**: `poetry add catboost` (the `gbdt_hist_gbdt` config runs without it).
- **Google Gemini**: the provider pins the model to `Configs.DEFAULT_LLM_GOOGLE_MODEL_NAME`;
  the config's model/temperature are re-applied to the client so they take effect.

## Interpreting the numbers

The bundled `data/` appears to be **synthetically generated** (random user–item
interactions, templated item/user text). Real collaborative or semantic signal is
therefore weak, so all engines land close to the random floor and absolute scores
are low — this is a property of the sample data, not the harness. Point the loader
at real interaction logs (same schema) to get a meaningful comparison; the harness,
splits, and metrics are production-shaped.
