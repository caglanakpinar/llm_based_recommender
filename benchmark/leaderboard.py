"""Aggregate every config's metrics into one comparison table.

Reads ``results/configs/*/metrics.json`` (written by ``run_config``) and emits a
markdown leaderboard sorted by the headline metric.

    python -m benchmark.leaderboard
    python -m benchmark.leaderboard --sort ndcg@10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_records(configs_dir: Path) -> list[dict]:
    records = []
    for mpath in sorted(configs_dir.glob("*/metrics.json")):
        try:
            records.append(json.loads(mpath.read_text()))
        except Exception as exc:
            print(f"[warn] could not read {mpath}: {exc}")
    return records


def build_table(records: list[dict], sort_key: str) -> str:
    if not records:
        return "No config results found. Run `python -m benchmark.run_config --config <name>` first.\n"

    # Union of k-based metrics present.
    k_set = set()
    for r in records:
        for m in r["metrics"]:
            if "@" in m:
                k_set.add(int(m.split("@")[1]))
    k_list = sorted(k_set)
    cols = ["mrr"]
    for k in k_list:
        cols += [f"ndcg@{k}", f"map@{k}", f"precision@{k}", f"recall@{k}", f"hit@{k}"]

    records = sorted(records, key=lambda r: r["metrics"].get(sort_key, 0.0), reverse=True)

    lines = ["# Benchmark Leaderboard — ranker configs\n"]
    lines.append(f"Sorted by **{sort_key}**. Each config scores the same shared candidate pool.\n")
    header = "| Config | kind | n_users | " + " | ".join(cols) + " |"
    sep = "| --- | --- | --- | " + " | ".join(["---"] * len(cols)) + " |"
    lines += [header, sep]
    for r in records:
        agg = r["metrics"]
        name = r["config"]["name"]
        kind = r["config"]["kind"]
        n_u = int(r.get("eval", {}).get("n_users", agg.get("n_users_evaluated", 0)))
        row = [name, kind, str(n_u)] + [f"{agg.get(c, 0.0):.4f}" for c in cols]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Config legend (embedder / llm / temperature).
    lines.append("## Config details\n")
    lines.append("| Config | embedder | llm_provider | llm_model | temperature | params | secs |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in records:
        c = r["config"]
        secs = r.get("eval", {}).get("seconds", "")
        params = "" if c["kind"] == "llm" else json.dumps(c.get("params", {}))
        emb = c["embedder"] if c["kind"] == "llm" else ""
        prov = c["llm_provider"] if c["kind"] == "llm" else ""
        mdl = c["llm_model"] if c["kind"] == "llm" else ""
        temp = c["temperature"] if c["kind"] == "llm" else ""
        lines.append(f"| {c['name']} | {emb} | {prov} | {mdl} | {temp} | {params} | {secs} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate config metrics into a leaderboard")
    ap.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    ap.add_argument("--sort", type=str, default="ndcg@10")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    records = load_records(results_dir / "configs")
    table = build_table(records, args.sort)
    out = results_dir / "leaderboard.md"
    out.write_text(table)
    print(table)
    print(f"[done] leaderboard -> {out}")


if __name__ == "__main__":
    main()
