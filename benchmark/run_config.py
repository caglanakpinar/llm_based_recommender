"""Run ONE ranker config against the shared benchmark pool and record its metrics.

All configs share a single sampled dataset (cached under ``results/shared/``) so
their numbers are directly comparable. Run configs one at a time from the
terminal, then aggregate with ``python -m benchmark.leaderboard``.

    python -m benchmark.run_config --list
    python -m benchmark.run_config --config two_tower
    python -m benchmark.run_config --config gbdt_hist_gbdt
    python -m benchmark.run_config --config llm_sentence_transformer_google_gemini-3.5-flash_0.7

LLM configs cap evaluation to ``--llm-max-users`` (default 25) because the LLM is
called once per candidate. Feature rankers train on all sampled users but are
evaluated on the same capped set so every config is compared on identical users.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmark.configs import SAMPLE_CONFIGS, get_config
from benchmark.data_prep import load_or_build_benchmark_data
from benchmark.evaluate import evaluate_scores

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _score_llm(config, data, head_users, head_pool, cfg_dir, args, k_list):
    from benchmark.llm_engine import build_llm_ranker, CachedLLMScorer

    head_set = set(head_users)
    ranker = build_llm_ranker(
        engine_name=config.name,
        items=data.items,
        users=data.users[data.users[data.user_col].isin(head_set)],
        train_interactions=data.train_interactions[
            data.train_interactions[data.user_col].isin(head_set)
        ],
        embedder_name=config.embedder,
        llm_provider=config.llm_provider,
        llm_model=config.llm_model,
        temperature=config.temperature,
        max_new_tokens=config.max_new_tokens,
    )
    scorer = CachedLLMScorer(ranker, cfg_dir / "llm_score_cache.json")
    pairs = list(head_pool[[data.user_col, data.item_col]].itertuples(index=False, name=None))
    return scorer.score_pairs(pairs)


def _score_feature_ranker(config, data, head_pool):
    pairs = list(head_pool[[data.user_col, data.item_col]].itertuples(index=False, name=None))
    if config.kind == "two_tower":
        from benchmark.rankers.two_tower import TwoTowerRanker
        ranker = TwoTowerRanker(
            data.items, data.users, data.train_interactions,
            engine_name=config.name, user_col=data.user_col, item_col=data.item_col,
            **{k: config.params[k] for k in
               ("emb_dim", "hidden", "epochs", "lr", "n_neg_per_pos") if k in config.params},
        )
    elif config.kind == "gbdt":
        from benchmark.rankers.gbdt import GBDTRanker
        ranker = GBDTRanker(
            data.items, data.users, data.train_interactions,
            engine_name=config.name, user_col=data.user_col, item_col=data.item_col,
            backend=config.params.get("backend", "catboost"),
            n_neg_per_pos=config.params.get("n_neg_per_pos", 4),
            **{k: v for k, v in config.params.items()
               if k in ("iterations", "depth", "learning_rate", "l2_leaf_reg")},
        )
    else:
        raise ValueError(f"Unknown feature-ranker kind '{config.kind}'")
    ranker.fit()
    return ranker.score_pairs(pairs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one benchmark ranker config")
    ap.add_argument("--config", type=str, help="config name (see --list)")
    ap.add_argument("--list", action="store_true", help="list available configs and exit")
    # shared-data params (must match across configs to stay comparable)
    ap.add_argument("--n-users", type=int, default=100)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--n-negatives", type=int, default=20)
    ap.add_argument("--min-train", type=int, default=5)
    ap.add_argument("--min-relevant", type=int, default=1)
    ap.add_argument("--negative-sampling", choices=["uniform", "popularity"], default="uniform")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rebuild-data", action="store_true")
    # eval params
    ap.add_argument("--k", type=str, default="5,10")
    ap.add_argument("--llm-max-users", type=int, default=25,
                    help="cap the evaluated user set (0 = all sampled users)")
    ap.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    args = ap.parse_args()

    if args.list or not args.config:
        print("Available configs:")
        for name, c in SAMPLE_CONFIGS.items():
            detail = (f"embedder={c.embedder} provider={c.llm_provider} model={c.llm_model} temp={c.temperature}"
                      if c.kind == "llm" else f"params={c.params}")
            print(f"  {name:52s} [{c.kind}]  {detail}")
        if not args.config:
            return
    config = get_config(args.config)

    k_list = [int(x) for x in args.k.split(",") if x.strip()]
    results_dir = Path(args.results_dir)
    shared_dir = results_dir / "shared"

    # 1. Shared dataset (built once, reused by every config) -----------------
    data = load_or_build_benchmark_data(
        shared_dir, rebuild=args.rebuild_data,
        n_users=args.n_users, test_frac=args.test_frac, n_negatives=args.n_negatives,
        min_train_interactions=args.min_train, min_relevant=args.min_relevant,
        negative_sampling=args.negative_sampling, seed=args.seed,
    )
    if args.llm_max_users and args.llm_max_users < len(data.user_ids):
        head_users = data.user_ids[: args.llm_max_users]
    else:
        head_users = data.user_ids
    head_set = set(head_users)
    head_pool = data.pool[data.pool[data.user_col].isin(head_set)].reset_index(drop=True)
    print(f"[data] sampled={len(data.user_ids)} users | evaluating config '{config.name}' "
          f"on {len(head_users)} users ({len(head_pool)} candidate pairs)")

    # 2. Score with the requested ranker -------------------------------------
    cfg_dir = results_dir / "configs" / config.name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if config.kind == "llm":
        scores = _score_llm(config, data, head_users, head_pool, cfg_dir, args, k_list)
    else:
        scores = _score_feature_ranker(config, data, head_pool)
    elapsed = time.time() - t0

    # 3. Evaluate + persist ---------------------------------------------------
    aggregate, _ = evaluate_scores(head_pool, scores, data.relevants, k_list,
                                   data.user_col, data.item_col)
    record = {
        "config": {
            "name": config.name, "kind": config.kind, "embedder": config.embedder,
            "llm_provider": config.llm_provider, "llm_model": config.llm_model,
            "temperature": config.temperature, "params": config.params,
        },
        "eval": {"n_users": len(head_users), "n_pairs": len(head_pool),
                 "k_list": k_list, "seconds": round(elapsed, 1), "seed": args.seed},
        "metrics": aggregate,
    }
    (cfg_dir / "metrics.json").write_text(json.dumps(record, indent=2))
    print(f"[done] {config.name}: "
          f"ndcg@{k_list[-1]}={aggregate.get(f'ndcg@{k_list[-1]}', 0):.4f} "
          f"map@{k_list[-1]}={aggregate.get(f'map@{k_list[-1]}', 0):.4f} "
          f"mrr={aggregate.get('mrr', 0):.4f}  ({elapsed:.1f}s)")
    print(f"[done] metrics -> {cfg_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
