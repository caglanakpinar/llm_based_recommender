"""Offline benchmark orchestrator.

Compares the classic baseline recommenders against the core2 LLM engine on a
common candidate pool using ranking metrics (NDCG / MAP / MRR / Precision /
Recall / HitRate). Both families are trained on the same training interactions
and score the identical per-user candidate pool, so the comparison isolates
ranking quality.

Run:
    python -m benchmark.run_benchmark --n-users 100 --n-negatives 20
    python -m benchmark.run_benchmark --skip-llm         # baselines only (fast)
    python -m benchmark.run_benchmark --llm-max-users 20 # LLM on a subset

Artifacts are written under ``benchmark/results/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from benchmark.data_prep import BenchmarkData, build_benchmark_data
from benchmark.baseline import ItemKNNRecommender, PopularityRecommender, RandomRecommender
from benchmark.evaluate import scores_to_rankings
from benchmark.metrics import evaluate_rankings

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_engine(name, recommender, data: BenchmarkData, pool: pd.DataFrame, k_list):
    """Fit (if needed), score the pool, and evaluate one recommender."""
    pairs = list(pool[[data.user_col, data.item_col]].itertuples(index=False, name=None))
    scores = recommender.score_pairs(pairs)
    rankings = scores_to_rankings(pool, scores, data.user_col, data.item_col)
    relevants = {u: data.relevants[u] for u in rankings if u in data.relevants}
    aggregate, per_user = evaluate_rankings(rankings, relevants, k_list)
    return {"name": name, "aggregate": aggregate, "rankings": rankings}


def _metrics_table(results: list[dict], cols: list[str]) -> list[str]:
    lines = []
    header = "| Engine | n_users | " + " | ".join(cols) + " |"
    sep = "| --- | --- | " + " | ".join(["---"] * len(cols)) + " |"
    lines.append(header)
    lines.append(sep)
    for r in results:
        agg = r["aggregate"]
        if not agg:
            continue
        n_u = int(agg.get("n_users_evaluated", 0))
        row = [r["name"], str(n_u)] + [f"{agg.get(c, 0.0):.4f}" for c in cols]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def format_report(results: list[dict], k_list: list[int], meta: dict, appendix: list[dict] | None = None) -> str:
    """Markdown comparison report."""
    # Explicit column order for the results table.
    cols = ["mrr"]
    for k in k_list:
        cols += [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"map@{k}", f"hit@{k}"]

    lines = []
    lines.append("# LLM-based vs Baseline Recommender — Offline Benchmark\n")
    lines.append("## Setup\n")
    lines.append(f"- Users sampled: **{meta['n_sampled_users']}**  |  head-to-head users (incl. LLM): **{meta['n_head_users']}**")
    lines.append(f"- Candidates per user: {meta['n_negatives']} {meta.get('negative_sampling','uniform')}-sampled negatives + strong-positive test items")
    lines.append(f"- Temporal split: last **{int(meta['test_frac']*100)}%** of each user's interactions held out for testing")
    lines.append(f"- Relevance = strong-positive test actions (purchase / add_to_cart / like), unseen in training")
    lines.append(f"- Baseline training: {meta['n_full_train_interactions']} full-population train interactions")
    lines.append(f"- LLM engine training: {meta['n_train_interactions']} sampled-users train interactions (retrieval context)")
    lines.append(f"- LLM engine: provider=`{meta['llm_provider']}` model=`{meta['llm_model']}`" if meta.get("llm_provider") else "- LLM engine: skipped")
    lines.append("")
    lines.append("## Head-to-head results (macro-averaged over users, same candidate pool)\n")
    lines += _metrics_table(results, cols)

    # Highlight the headline metric.
    headline = f"ndcg@{k_list[-1]}"
    ranked = sorted(
        [r for r in results if r["aggregate"]],
        key=lambda r: r["aggregate"].get(headline, 0.0),
        reverse=True,
    )
    if ranked:
        lines.append("### Ranking by " + headline + "\n")
        best = ranked[0]
        lines.append(f"Best: **{best['name']}** = {best['aggregate'][headline]:.4f}\n")
        for r in ranked:
            lines.append(f"- {r['name']}: {r['aggregate'][headline]:.4f}")
        lines.append("")

    if appendix and meta["n_head_users"] != meta["n_sampled_users"]:
        lines.append(f"## Appendix — baselines over all {meta['n_sampled_users']} sampled users\n")
        lines += _metrics_table(appendix, cols)

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline LLM-vs-baseline recommender benchmark")
    ap.add_argument("--n-users", type=int, default=100)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--n-negatives", type=int, default=20)
    ap.add_argument("--min-train", type=int, default=5)
    ap.add_argument("--min-relevant", type=int, default=1)
    ap.add_argument("--negative-sampling", choices=["uniform", "popularity"], default="uniform")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=str, default="5,10", help="comma-separated cut-offs")
    ap.add_argument("--engine-name", type=str, default="benchmark_llm")
    ap.add_argument("--llm-provider", type=str, default="ollama")
    ap.add_argument("--llm-model", type=str, default="llama3.2:latest")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--llm-max-users", type=int, default=25,
                    help="cap LLM head-to-head to the first N sampled users (0 = all; "
                         "local LLM is ~4-5s/pair so all-100 takes hours). Baselines always "
                         "also run on every sampled user.")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    args = ap.parse_args()

    k_list = [int(x) for x in args.k.split(",") if x.strip()]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data prep -----------------------------------------------------------
    print(f"[prep] sampling {args.n_users} users, test_frac={args.test_frac}, "
          f"n_negatives={args.n_negatives}, seed={args.seed}")
    data = build_benchmark_data(
        n_users=args.n_users,
        test_frac=args.test_frac,
        n_negatives=args.n_negatives,
        min_train_interactions=args.min_train,
        min_relevant=args.min_relevant,
        negative_sampling=args.negative_sampling,
        seed=args.seed,
    )
    print(f"[prep] {len(data.user_ids)} users sampled | pool rows={len(data.pool)} "
          f"| positives={int(data.pool['label'].sum())} | train_interactions={len(data.train_interactions)}")

    data.pool.to_parquet(results_dir / "eval_pool.parquet", index=False)
    (results_dir / "sampled_users.json").write_text(json.dumps(data.user_ids, indent=2))
    (results_dir / "relevants.json").write_text(
        json.dumps({u: sorted(v) for u, v in data.relevants.items()}, indent=2)
    )

    # Head-to-head user set: the LLM is expensive, so it may be capped to a subset.
    # Baselines are always additionally evaluated on ALL sampled users (cheap).
    run_llm = not args.skip_llm
    if run_llm and args.llm_max_users and args.llm_max_users < len(data.user_ids):
        head_users = data.user_ids[: args.llm_max_users]
    else:
        head_users = data.user_ids
    head_users_set = set(head_users)
    head_pool = data.pool[data.pool[data.user_col].isin(head_users_set)].reset_index(drop=True)
    print(f"[prep] head-to-head set: {len(head_users)} users, {len(head_pool)} candidate pairs")

    # 2. Baselines -----------------------------------------------------------
    baselines_full: list[dict] = []   # baselines over all sampled users
    baselines_head: list[dict] = []   # baselines over the head-to-head subset
    if not args.skip_baseline:
        print(f"[baseline] fitting on full-population train "
              f"({len(data.full_train_interactions)} interactions)")
        for rec_cls in (RandomRecommender, PopularityRecommender, ItemKNNRecommender):
            rec = rec_cls(user_col=data.user_col, item_col=data.item_col)
            rec.fit(data.full_train_interactions)
            res_full = run_engine(rec.name, rec, data, data.pool, k_list)
            baselines_full.append(res_full)
            print(f"[baseline/all-{len(data.user_ids)}] {rec.name:12s} "
                  f"ndcg@{k_list[-1]}={res_full['aggregate'].get(f'ndcg@{k_list[-1]}', 0):.4f} "
                  f"map@{k_list[-1]}={res_full['aggregate'].get(f'map@{k_list[-1]}', 0):.4f} "
                  f"mrr={res_full['aggregate'].get('mrr', 0):.4f}")
            if head_users_set != set(data.user_ids):
                baselines_head.append(run_engine(rec.name, rec, data, head_pool, k_list))
            else:
                baselines_head.append(res_full)

    # 3. LLM engine ----------------------------------------------------------
    head_results: list[dict] = list(baselines_head)
    llm_meta_provider = None
    if run_llm:
        from benchmark.llm_engine import build_llm_ranker, CachedLLMScorer

        ranker = build_llm_ranker(
            engine_name=args.engine_name,
            items=data.items,
            users=data.users[data.users[data.user_col].isin(head_users_set)],
            train_interactions=data.train_interactions[
                data.train_interactions[data.user_col].isin(head_users_set)
            ],
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        scorer = CachedLLMScorer(ranker, results_dir / "llm_score_cache.json")
        pairs = list(head_pool[[data.user_col, data.item_col]].itertuples(index=False, name=None))
        scores = scorer.score_pairs(pairs)
        rankings = scores_to_rankings(head_pool, scores, data.user_col, data.item_col)
        relevants = {u: data.relevants[u] for u in rankings if u in data.relevants}
        aggregate, _ = evaluate_rankings(rankings, relevants, k_list)
        head_results.append({"name": "llm_reco", "aggregate": aggregate, "rankings": rankings})
        llm_meta_provider = args.llm_provider
        print(f"[llm] llm_reco ndcg@{k_list[-1]}={aggregate.get(f'ndcg@{k_list[-1]}', 0):.4f} "
              f"map@{k_list[-1]}={aggregate.get(f'map@{k_list[-1]}', 0):.4f} "
              f"mrr={aggregate.get('mrr', 0):.4f}")

    # 4. Persist + report ----------------------------------------------------
    meta = {
        "n_sampled_users": len(data.user_ids),
        "n_head_users": len(head_users),
        "n_negatives": args.n_negatives,
        "test_frac": args.test_frac,
        "n_train_interactions": len(data.train_interactions),
        "n_full_train_interactions": len(data.full_train_interactions),
        "negative_sampling": args.negative_sampling,
        "llm_provider": llm_meta_provider,
        "llm_model": args.llm_model if llm_meta_provider else None,
        "k_list": k_list,
        "seed": args.seed,
    }
    metrics_out = {
        "head_to_head": {r["name"]: r["aggregate"] for r in head_results},
        "baselines_all_users": {r["name"]: r["aggregate"] for r in baselines_full},
    }
    (results_dir / "metrics.json").write_text(json.dumps({"meta": meta, "metrics": metrics_out}, indent=2))

    report = format_report(head_results, k_list, meta, appendix=baselines_full)
    (results_dir / "report.md").write_text(report)
    print("\n" + report)
    print(f"\n[done] artifacts written to {results_dir}")


if __name__ == "__main__":
    main()
