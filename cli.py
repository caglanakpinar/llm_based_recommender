"""reco-agent — terminal CLI for the LLM-based recommender.

Two commands mirror the Streamlit UI without the browser:

* ``generate-engine`` — builds a reco engine end-to-end from parquet folders and
  the same knobs the builder page exposes (embedder, LLM platform/model, top_k,
  constraints, column mappings). Persists the engine folder + FAISS ``context.index``.

* ``reco`` — generates recommendations for a user. Instead of calling the HTTP
  API it reconstructs the already-built ``RecoEngine`` predictor from the engine
  folder (the FAISS ``context.index`` is already there) and returns the results.

Run it as ``poetry run reco-agent <command> ...``.
"""

from __future__ import annotations

import json

import click

from core2.configs import Configs
from core2.embeddings import EMBEDDER_REGISTRY
from core2.llms import FREE_LLM_REGISTRY
from reco_engine import (
    build_engine,
    build_llminput_from_form,
    engine_exists,
    list_engines,
    load_engine_meta,
)
from reco_api import assemble_predictor, build_engine_prompts, build_predictor_for_engine

DEFAULT_CONSTRAINTS = (
    "- Exclude items already purchased or disliked by the Target User.\n"
    "- Prefer items in categories the user engaged with recently.\n"
    "- Balance exploitation with one exploratory item."
)
_LLM_PLATFORMS = [name for name in FREE_LLM_REGISTRY if name != "anthropic"]


def _parse_columns(raw: str | None, default: dict[str, str]) -> dict[str, str]:
    """Parse a --*-columns JSON string, falling back to the repo default map."""
    if not raw:
        return dict(default)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise click.BadParameter(f"expected a JSON object, got: {raw} ({exc})")
    if not isinstance(parsed, dict):
        raise click.BadParameter("column mapping must be a JSON object")
    return {**default, **parsed}


def _print_recommendations(result: dict) -> None:
    """Human-readable rendering of a predictor response."""
    if not isinstance(result, dict):
        click.echo(str(result))
        return
    if result.get("error"):
        raise click.ClickException(result["error"])

    recs = result.get("recommendations", [])
    click.secho(
        f"\nRecommendations for user '{result.get('user_id')}' "
        f"(returned {result.get('returned', len(recs))} of {result.get('total_scored', '?')} scored):",
        bold=True,
    )
    if not recs:
        click.echo("  (no recommendations)")
        return
    click.echo(f"  {'#':>2}  {'item_id':<14} {'score':>7}  {'category':<14} title")
    click.echo("  " + "-" * 66)
    for rank, rec in enumerate(recs, start=1):
        click.echo(
            f"  {rank:>2}  {str(rec.get('item_id','')):<14} "
            f"{float(rec.get('score', 0.0)):>7.4f}  "
            f"{str(rec.get('category','')):<14} {str(rec.get('title',''))}"
        )


@click.group()
def cli() -> None:
    """reco-agent — build and query LLM-based recommendation engines."""


@cli.command("generate-engine")
@click.option("--engine-name", "-n", required=True, help="Unique engine name.")
@click.option("--interactions", "-i", "interactions_folder", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Folder of user–item interaction parquet files.")
@click.option("--users", "-u", "users_folder", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Folder of user-profile parquet files.")
@click.option("--items", "-c", "items_folder", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Folder of item-catalog parquet files.")
@click.option("--top-k", default=3, show_default=True, type=int,
              help="Number of recommendations the engine returns.")
@click.option("--constraints", default=DEFAULT_CONSTRAINTS, help="Recommendation constraints text.")
@click.option("--llm-chat", default="", help="Extra instructions appended to the prompt.")
@click.option("--embedding-model", default=Configs.DEFAULT_EMBEDDING_MODEL_NAME,
              show_default=True, type=click.Choice(list(EMBEDDER_REGISTRY)),
              help="Embedder used to encode context prompts into the vector DB.")
@click.option("--llm-platform", default=Configs.DEFAULT_LLM_MODEL_NAME,
              show_default=True, type=click.Choice(_LLM_PLATFORMS),
              help="LLM provider used by the relevance ranker.")
@click.option("--llm-model", default="", help="Model name (empty = platform default).")
@click.option("--user-profile-columns", default=None,
              help='JSON mapping, e.g. \'{"user_id":"uid","segment":"seg"}\'.')
@click.option("--item-catalog-columns", default=None,
              help='JSON mapping, e.g. \'{"item_id":"iid","title":"name"}\'.')
@click.option("--overwrite", is_flag=True, help="Replace the engine if it already exists.")
@click.option("--run-sample/--no-run-sample", default=False,
              help="After building, run a sample recommendation for the first user.")
def generate_engine(engine_name, interactions_folder, users_folder, items_folder, top_k,
                    constraints, llm_chat, embedding_model, llm_platform, llm_model,
                    user_profile_columns, item_catalog_columns, overwrite, run_sample):
    """Build a reco engine (same inputs as the UI builder page)."""
    if engine_exists(engine_name) and not overwrite:
        raise click.ClickException(
            f"Engine '{engine_name}' already exists. Use --overwrite to replace it."
        )

    user_cols = _parse_columns(user_profile_columns, Configs.DEFAULT_USER_PROFILE_COLUMNS)
    item_cols = _parse_columns(item_catalog_columns, Configs.DEFAULT_ITEM_CATALOG_COLUMNS)

    click.echo(f"Building llminput for '{engine_name}' …")
    llminput = build_llminput_from_form(
        top_k=int(top_k),
        constraints=constraints,
        user_profile_columns=user_cols,
        item_catalog_columns=item_cols,
        interactions_parquet_folder=interactions_folder,
        users_parquet_folder=users_folder,
        items_parquet_folder=items_folder,
        llm_chat=llm_chat,
        embedding_model=embedding_model,
        llm_platform=llm_platform,
        llm_model=llm_model.strip(),
    )

    click.echo("Creating engine folder, copying parquet, rendering prompt …")
    build_engine(
        engine_name, llminput,
        interactions_parquet_folder=interactions_folder,
        users_parquet_folder=users_folder,
        items_parquet_folder=items_folder,
    )

    click.echo("Building prompts, DBs, retrieval, ranker, and FAISS context index …")
    predictor = build_predictor_for_engine(engine_name)

    configs = Configs.from_engine_name(engine_name)
    click.secho(f"✓ Engine '{engine_name}' built at {configs.engine_root}", fg="green", bold=True)
    click.echo(f"  embedder={embedding_model}  llm_platform={llm_platform}  "
               f"llm_model={llm_model or '(default)'}  top_k={top_k}")

    if run_sample:
        profile = llminput.get("user_profile") or {}
        sample_user = profile.get("user_id")
        if sample_user:
            click.echo(f"\nRunning sample recommendation for '{sample_user}' …")
            result = predictor.predict(
                {"engine_name": engine_name, "user_id": str(sample_user), "top_k": int(top_k)}
            )
            _print_recommendations(result)


@cli.command("reco")
@click.option("--engine-name", "-n", required=True, help="Name of an already-built engine.")
@click.option("--user-id", "-u", required=True, help="Target user id to recommend for.")
@click.option("--top-k", default=None, type=int, help="Number of recommendations (default: engine's top_k).")
@click.option("--use-existing-index", is_flag=True,
              help="Load the persisted FAISS context.index instead of re-embedding (faster).")
@click.option("--as-json", "as_json", is_flag=True, help="Print the raw predictor response as JSON.")
def reco(engine_name, user_id, top_k, use_existing_index, as_json):
    """Generate recommendations from an already-built engine (no HTTP API)."""
    if not engine_exists(engine_name):
        available = ", ".join(list_engines()) or "(none)"
        raise click.ClickException(f"Unknown engine '{engine_name}'. Available: {available}")

    if top_k is None:
        top_k = int((load_engine_meta(engine_name) or {}).get("top_k", 5))

    click.echo(f"Initializing RecoEngine for '{engine_name}' …")
    if use_existing_index:
        configs = Configs.from_engine_name(engine_name)
        datasets, context_prompts = build_engine_prompts(engine_name)
        predictor = assemble_predictor(
            engine_name, datasets, context_prompts,
            embedding_model_name=getattr(configs, "embedding_model", None) or Configs.DEFAULT_EMBEDDING_MODEL_NAME,
            llm_platform=getattr(configs, "llm_platform", None) or Configs.DEFAULT_LLM_MODEL_NAME,
            llm_model=(getattr(configs, "llm_model", None) or "").strip() or None,
            write=False,  # reuse the existing context.index
        )
    else:
        predictor = build_predictor_for_engine(engine_name)

    result = predictor.predict(
        {"engine_name": engine_name, "user_id": str(user_id), "top_k": int(top_k)}
    )

    if as_json:
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_recommendations(result)


@cli.command("list-engines")
def list_engines_cmd():
    """List engines that have been built."""
    engines = list_engines()
    if not engines:
        click.echo("No engines built yet. Use `reco-agent generate-engine ...`.")
        return
    click.secho("Built engines:", bold=True)
    for name in engines:
        meta = load_engine_meta(name) or {}
        click.echo(f"  {name:<28} top_k={meta.get('top_k', '?')}")


if __name__ == "__main__":
    cli()
