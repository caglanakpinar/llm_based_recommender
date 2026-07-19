"""Ranker configurations for the benchmark.

A *config* is one point in the comparison grid. Three kinds of rankers exist:

* ``llm``       — the core2 LLM ranker, parameterized by embedder + LLM
                  provider/model + temperature. Identified as
                  ``llm_<embedder>_<provider>_<model>_<temperature>``.
* ``two_tower`` — a neural two-tower ranker over the engineered features.
* ``gbdt``      — a gradient-boosted ranker (CatBoost, or sklearn HistGBDT)
                  over the engineered features.

Every config scores the *same* shared candidate pool, so results are directly
comparable. Run one at a time with ``python -m benchmark.run_config --config <name>``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _slug(x) -> str:
    return str(x).replace("/", "-").replace(":", "-").replace(" ", "")


@dataclass
class RankerConfig:
    kind: str                       # "llm" | "two_tower" | "gbdt"
    name: str = ""                  # auto-derived if empty
    # --- llm knobs ---
    embedder: str = "sentence_transformer"   # see core2.embeddings EMBEDDER_REGISTRY
    llm_provider: str = "google"             # see core2.llms FREE_LLM_REGISTRY
    llm_model: str = "gemini-3.5-flash"
    temperature: float = 0.7
    max_new_tokens: int = 16
    # --- feature-ranker knobs ---
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = self.default_name()

    def default_name(self) -> str:
        if self.kind == "llm":
            return "_".join([
                "llm", _slug(self.embedder), _slug(self.llm_provider),
                _slug(self.llm_model), _slug(self.temperature),
            ])
        if self.kind == "gbdt":
            return f"gbdt_{_slug(self.params.get('backend', 'catboost'))}"
        return _slug(self.kind)


def build_sample_configs() -> list[RankerConfig]:
    """The sample candidate grid.

    Includes the three LLM examples requested (sentence_transformer @0.7 and
    hf_mean_pool @0.0 on Google gemini-3.5-flash), a couple more LLM points, a
    keyless local-Ollama LLM point, plus the two-tower and CatBoost rankers.
    """
    configs: list[RankerConfig] = []

    # ---- LLM grid: embedder x temperature on Google gemini-3.5-flash --------
    for embedder in ("sentence_transformer", "hf_mean_pool"):
        for temperature in (0.0, 0.7):
            configs.append(RankerConfig(
                kind="llm", embedder=embedder,
                llm_provider="google", llm_model="gemini-3.5-flash",
                temperature=temperature,
            ))

    # ---- Keyless local LLM (Ollama) so the pipeline runs without an API key --
    configs.append(RankerConfig(
        kind="llm", embedder="sentence_transformer",
        llm_provider="ollama", llm_model="llama3.2:latest", temperature=0.2,
    ))

    # ---- Feature-based rankers over the already-engineered features ----------
    configs.append(RankerConfig(kind="two_tower", params={
        "emb_dim": 32, "hidden": 64, "epochs": 15, "lr": 1e-3, "n_neg_per_pos": 4,
    }))
    configs.append(RankerConfig(kind="gbdt", params={
        "backend": "catboost", "iterations": 400, "depth": 6,
        "learning_rate": 0.05, "n_neg_per_pos": 4,
    }))
    # sklearn HistGradientBoosting variant — runs without installing catboost.
    configs.append(RankerConfig(kind="gbdt", params={
        "backend": "hist_gbdt", "n_neg_per_pos": 4,
    }))

    return configs


SAMPLE_CONFIGS: dict[str, RankerConfig] = {c.name: c for c in build_sample_configs()}


def get_config(name: str) -> RankerConfig:
    if name not in SAMPLE_CONFIGS:
        available = "\n  ".join(SAMPLE_CONFIGS)
        raise KeyError(f"Unknown config '{name}'. Available:\n  {available}")
    return SAMPLE_CONFIGS[name]
