"""Streamlit UI for reco engine builder and recommendation generator."""

from __future__ import annotations

import json
import os
import re
from typing import Any


import streamlit as st

from core2.logger import logger
from core2.configs import Configs
from schema_defaults import DEFAULT_ITEM_CATALOG_COLUMNS, DEFAULT_USER_PROFILE_COLUMNS
from reco_engine import (
    build_engine,
    build_llminput_from_form,
    delete_engine,
    engine_exists,
    get_engine_store,
    list_engines,
    list_parquet_user_ids,
    load_engine_llminput,
    load_engine_meta,
    load_parquet_interactions,
    load_parquet_items,
    load_parquet_users,
    load_rendered_prompt,
    llminput_for_target_user,
    resolve_parquet_folder,
    save_uploaded_parquet_files,
)

from core2.datasets import DataSets
from core2.prompting import (
    RelevanceScorePrompt,
    UserPrompt,
    ItemPrompt,
    UserItemPrompt
)
from core2.embeddings import EMBEDDER_REGISTRY, describe_embedders
from core2.llms import FREE_LLM_REGISTRY
from reco_api import (
    KSERVE_API_PORT,
    KSERVE_API_URL,
    assemble_predictor,
    build_predictor_for_engine,
    start_api,
)

st.set_page_config(
    page_title="LLM Recommender",
    page_icon="🎯",
    layout="wide",
)



def _resolve_env_key(candidates: tuple[str, ...]) -> tuple[str, str | None]:
    """Return ``(value, source_env_var)`` for the first candidate env var that is
    set. ``source_env_var`` is the exact variable name the value came from (so the
    sidebar can report e.g. GEMINI_API_KEY vs GOOGLE_API_KEY), or ``None`` when
    none of the candidates are set."""
    for name in candidates:
        value = os.getenv(name)
        if value:
            return value, name
    return "", None


OPENAI_API_KEY, OPENAI_ENV_SOURCE = _resolve_env_key(("OPENAI_API_KEY",))
HUGGINGFACE_API_TOKEN, HUGGINGFACE_ENV_SOURCE = _resolve_env_key(
    ("HF_TOKEN", "HUGGINGFACE_API_TOKEN")
)
GOOGLE_API_KEY, GOOGLE_ENV_SOURCE = _resolve_env_key(("GOOGLE_API_KEY", "GEMINI_API_KEY"))
CLAUDE_API_KEY, CLAUDE_ENV_SOURCE = _resolve_env_key(("CLAUDE_KEY", "ANTHROPIC_API_KEY"))

# Consistent sample datasets shipped in the repo; used to prefill the builder's
# folder-path fields for new engines so a default build always uses matching ids.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_INTERACTIONS_FOLDER = os.path.join(_REPO_ROOT, "data", "sample_interactions")
SAMPLE_USERS_FOLDER = os.path.join(_REPO_ROOT, "data", "sample_users")
SAMPLE_ITEMS_FOLDER = os.path.join(_REPO_ROOT, "data", "sample_items")

PLATFORM_DEFAULT_MODELS = {
    "google": Configs.DEFAULT_LLM_GOOGLE_MODEL_NAME,
    "claude": Configs.DEFAULT_LLM_CLAUDE_MODEL_NAME,
    "anthropic": Configs.DEFAULT_LLM_CLAUDE_MODEL_NAME,
    "huggingface": Configs.DEFAULT_LLM_HUGGING_FACE_MODEL_NAME,
}
rag = None

# --- session state ---
if "page" not in st.session_state:
    st.session_state.page = "builder"
if "edit_engine" not in st.session_state:
    st.session_state.edit_engine = None
if "selected_engine" not in st.session_state:
    st.session_state.selected_engine = None
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = None
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "huggingface_api_key" not in st.session_state:
    st.session_state.huggingface_api_key = ""


def _go_builder(edit_name: str | None = None) -> None:
    st.session_state.page = "builder"
    st.session_state.edit_engine = edit_name


def _go_generator() -> None:
    st.session_state.page = "generator"


def _process_rendered_prompt(prompt_text: str) -> str:
    """Process rendered prompt: skip header, start from Engine Description, limit samples."""
    lines = prompt_text.split("\n")
    result_lines = []
    skip_until_engine = True
    in_item_catalog = False
    item_count = 0
    in_placeholder_ref = False

    for i, line in enumerate(lines):
        # Skip header and intro until Engine Description
        if skip_until_engine:
            if line.startswith("## Engine Description"):
                skip_until_engine = False
            else:
                continue

        # Detect Item Catalog section
        if line.startswith("## Item Catalog"):
            in_item_catalog = True
            item_count = 0
            result_lines.append(line)
            continue

        # Detect Placeholder Reference section
        if line.startswith("## 8. llminput") or line.startswith("## llminput — Placeholder Reference"):
            in_placeholder_ref = True
            result_lines.append(line)
            continue

        # If we hit next section marker, reset flags
        if line.startswith("## ") and not in_item_catalog and not in_placeholder_ref:
            result_lines.append(line)
            in_item_catalog = False
            in_placeholder_ref = False
            continue
        elif line.startswith("## ") and (in_item_catalog or in_placeholder_ref):
            in_item_catalog = False
            in_placeholder_ref = False
            result_lines.append(line)
            continue

        # In Item Catalog: limit to 1 item
        if in_item_catalog:
            if line.strip().startswith("- ") and line.strip().startswith("- {"):
                item_count += 1
                if item_count <= 1:
                    result_lines.append(line)
                continue
            elif item_count > 0 and not line.strip():
                # Keep empty lines within first item
                if item_count == 1:
                    result_lines.append(line)
                continue
            else:
                result_lines.append(line)

        # In Placeholder Reference: truncate long lines
        elif in_placeholder_ref:
            if len(line) > 100 and (":" in line or "=" in line):
                # Truncate at 100 chars
                truncated = line[:100] + "..."
                result_lines.append(truncated)
            else:
                result_lines.append(line)
        else:
            result_lines.append(line)

    return "\n".join(result_lines)


def _enhance_prompt_with_target_user(prompt: str, target_user_id: str) -> str:
    """Enhance prompt header to prominently display the target user."""
    # Find the recommendation intent section and add target user
    lines = prompt.split("\n")
    enhanced_lines = []
    for i, line in enumerate(lines):
        if "Return personalized recommendations" in line or "Return recommendations" in line:
            # Insert target user header before this line
            enhanced_lines.append(f"**🎯 TARGET USER: {target_user_id}**\n")
        enhanced_lines.append(line)
    return "\n".join(enhanced_lines)


def _format_recommendations_table(parsed_response: dict[str, Any]) -> None:
    """Display recommendations as a formatted table."""
    if "recommendations" in parsed_response:
        recs = parsed_response["recommendations"]
        if recs:
            import pandas as pd
            df_recs = pd.DataFrame(recs)
            st.dataframe(df_recs, width="stretch")
        else:
            st.info("No recommendations generated.")
    else:
        st.json(parsed_response)


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    """Extract JSON object from raw LLM text, including fenced ```json blocks."""
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


def _recommendations_to_markdown(parsed_response: dict[str, Any]) -> str:
    """Render recommendations dict as markdown for UI readability."""
    recs = parsed_response.get("recommendations")
    if not isinstance(recs, list) or not recs:
        return "No recommendations generated."

    lines = []
    for idx, rec in enumerate(recs, start=1):
        if not isinstance(rec, dict):
            lines.append(f"{idx}. {str(rec)}")
            continue

        title = rec.get("title") or rec.get("name") or rec.get("item_id") or f"Item {idx}"
        score = rec.get("score")
        reason = rec.get("reason")
        item_id = rec.get("item_id")

        header = f"{idx}. **{title}**"
        if item_id and str(item_id) != str(title):
            header += f" (`{item_id}`)"
        if score is not None:
            header += f" - score: `{score}`"
        lines.append(header)

        if reason:
            lines.append(f"   - reason: {reason}")

        signals = rec.get("signals")
        if isinstance(signals, list) and signals:
            lines.append(f"   - signals: {', '.join(str(s) for s in signals)}")

    return "\n".join(lines)


def _render_recommendations_markdown(raw: dict[str, Any] | str) -> None:
    """Always show recommendation output in markdown-first format."""
    parsed: dict[str, Any] | None = None

    if isinstance(raw, dict):
        answer = raw.get("answer")
        if isinstance(answer, str):
            parsed = _extract_json_payload(answer)
        if parsed is None and "recommendations" in raw:
            parsed = raw
    else:
        parsed = _extract_json_payload(raw)

    if parsed is not None:
        st.markdown(_recommendations_to_markdown(parsed))
        return

    if isinstance(raw, dict):
        answer = raw.get("answer")
        if isinstance(answer, str) and answer.strip():
            st.markdown(answer)
            return

        context_items = raw.get("context") if isinstance(raw.get("context"), list) else []
        if context_items:
            lines = ["No structured recommendations found. Retrieved context:"]
            for i, item in enumerate(context_items[:5], start=1):
                if not isinstance(item, dict):
                    continue
                source = item.get("source", "unknown")
                score = item.get("score", "n/a")
                text = str(item.get("text", "")).strip().replace("\n", " ")
                snippet = text[:220] + ("..." if len(text) > 220 else "")
                lines.append(f"{i}. **{source}** (score: `{score}`) - {snippet}")
            st.markdown("\n".join(lines))
            return

        st.markdown("No recommendations could be rendered from the model output.")
        return

    if isinstance(raw, str) and raw.strip():
        st.markdown(raw)
    else:
        st.markdown("No recommendations were returned.")


def _extract_recommendation_rows(raw: dict[str, Any] | str) -> list[dict[str, Any]]:
    """Extract recommendation rows from structured dict or JSON text."""
    parsed: dict[str, Any] | None = None
    if isinstance(raw, dict):
        if isinstance(raw.get("recommendations"), list):
            parsed = raw
        else:
            answer = raw.get("answer")
            if isinstance(answer, str):
                parsed = _extract_json_payload(answer)
    else:
        parsed = _extract_json_payload(raw)

    if not isinstance(parsed, dict):
        return []
    recs = parsed.get("recommendations")
    if not isinstance(recs, list):
        return []

    rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(recs, start=1):
        if not isinstance(rec, dict):
            continue
        tags = rec.get("tags")
        if isinstance(tags, list):
            tags_text = ", ".join(str(tag) for tag in tags)
        elif tags is None:
            tags_text = ""
        else:
            tags_text = str(tags)

        rows.append(
            {
                "rank": rec.get("rank", idx),
                "item_id": rec.get("item_id", ""),
                "title": rec.get("title", ""),
                "category": rec.get("category", ""),
                "tags": tags_text,
                "price": rec.get("price", ""),
                "description": rec.get("description", ""),
                "score": rec.get("score", ""),
                "reason": rec.get("reason", ""),
            }
        )
    return rows


def _ensure_predictor(engine_name: str):
    """Return a cached KServe predictor for ``engine_name``, building one from
    persisted engine artifacts (docs/configs.yaml) on first access this session."""
    predictors = st.session_state.setdefault("kserve_predictors", {})
    if predictors.get(engine_name) is not None:
        return predictors[engine_name]

    predictor = build_predictor_for_engine(engine_name)
    predictors[engine_name] = predictor
    logger.info("KServe predictor ready for engine '%s'", engine_name)
    return predictor


@st.cache_resource(show_spinner=False)
def _start_kserve_api(port: int = KSERVE_API_PORT) -> dict[str, Any]:
    """Start the KServe API once per process (st.cache_resource guards it).

    The server itself lives in ``reco_api`` so its FastAPI app is not defined in
    the module Streamlit runs — otherwise Streamlit's own uvicorn server picks
    it up and serves it on the Streamlit port.
    """
    return start_api(port)


def _get_hf_model_for_provider(openai_model: str) -> str:
    """Map OpenAI model names to HuggingFace compatible models."""
    model_map = {
        "gpt-4o-mini": "meta-llama/Llama-2-7b-chat-hf",
        "gpt-4o": "meta-llama/Llama-2-13b-chat-hf",
        "gpt-4-turbo": "mistralai/Mistral-7B-Instruct-v0.1",
        "gpt-3.5-turbo": "mistralai/Mistral-7B-Instruct-v0.1",
    }
    return model_map.get(openai_model, "meta-llama/Llama-2-7b-chat-hf")


def _call_llm(prompt: str, api_key: str, model: str, hf_api_key: str | None = None, embedding_store=None) -> str:
    """Call OpenAI when available, otherwise fall back to HuggingFace or local providers with RAG support."""
    def _fallback_call(hf_token: str) -> str:
        from core.llms import FreeLLMCaller

        errors = []
        # Use HuggingFace model mapping for non-OpenAI providers
        hf_model = _get_hf_model_for_provider(model)
        
        for provider in ("huggingface", "ollama", "replicate"):
            try:
                # Use mapped model for HuggingFace, original for others
                provider_model = hf_model if provider == "huggingface" else model
                print(f"[DEBUG] Trying {provider} with model {provider_model}...")
                # Pass embedding_store for RAG pipeline
                caller = FreeLLMCaller(api_key=hf_token, model=provider_model, provider=provider, embedding_store=embedding_store, top_k=5)
                result = caller.call(prompt)
                if result and result.strip():
                    print(f"[DEBUG] {provider} succeeded!")
                    return result
                else:
                    errors.append(f"{provider}: returned empty response")
            except Exception as e:
                error_msg = str(e)
                print(f"[DEBUG] {provider} failed: {error_msg}")
                errors.append(f"{provider}: {error_msg}")
                continue
        error_msg = "\n".join(errors)
        raise RuntimeError(
            f"No fallback LLM provider succeeded.\n\n{error_msg}\n\n"
            "Solutions:\n"
            "1. Verify HuggingFace token: huggingface-cli login\n"
            "2. Or install/start Ollama: ollama serve && ollama pull llama2\n"
            "3. Or set REPLICATE_API_TOKEN for Replicate"
        )

    def fallback_call2() -> str:
        from core.rag import LangChainRAG
        rag = LangChainRAG(index_dir=Configs.current_dir / "embeddings", llm_model=model, api_key=hf_api_key)
        return rag.answer(prompt, top_k=5)

    if api_key is None or api_key.strip() == "":
        hf_token = hf_api_key or HUGGINGFACE_API_TOKEN
        if not hf_token:
            raise ValueError("No API key available. Add OpenAI or HuggingFace API key in sidebar.")
        return _fallback_call(hf_token)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
    except Exception:
        hf_token = hf_api_key or HUGGINGFACE_API_TOKEN
        try:
            return _fallback_call(hf_token)
        except Exception:
            return fallback_call2()


def _column_mapping_inputs(
    label: str,
    defaults: dict[str, str],
    saved: dict[str, str] | None,
    *,
    optional_fields: frozenset[str] = frozenset(),
) -> dict[str, str]:
    st.markdown(f"**{label}** — parquet column names")
    cols: dict[str, str] = {}
    source = saved or defaults
    fields = list(defaults.keys())
    for field in fields:
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption(f"`{field}`" + (" (optional)" if field in optional_fields else ""))
        with col_b:
            safe_label = label.replace("{", "").replace("}", "").replace(" ", "_")
            cols[field] = st.text_input(
                field,
                value=source.get(field, defaults[field]),
                label_visibility="collapsed",
                key=f"{safe_label}_{field}",
            )
    return cols


def _parquet_upload_section(
    *,
    title: str,
    caption: str,
    dataset_kind: str,
    upload_key: str,
    meta: dict[str, Any] | None,
    session_upload_key: str,
    meta_folder_key: str,
    load_preview,
    default_folder: str = "",
) -> str | None:
    """Render browse/upload/preview for one parquet dataset kind."""
    st.subheader(title)
    st.caption(caption)

    uploaded = st.file_uploader(
        "Browse parquet files",
        type=["parquet"],
        accept_multiple_files=True,
        help="Select one or more .parquet files from your computer.",
        key=f"{dataset_kind}_uploader_{upload_key}",
    )

    folder_path = st.text_input(
        "Or folder path on this machine",
        # For a new engine (no saved meta) prefill the consistent sample folder so
        # the default build uses matching datasets instead of stale uploads.
        value=(meta or {}).get(meta_folder_key) or default_folder or "",
        help="Alternative to browse upload: path to a folder with .parquet files.",
        key=f"{dataset_kind}_folder_{upload_key}",
    )

    if uploaded:
        try:
            upload_dir = save_uploaded_parquet_files(
                uploaded, upload_key, dataset_kind=dataset_kind
            )
            st.session_state[session_upload_key] = str(upload_dir)
            df = load_preview(str(upload_dir))
            st.success(f"Uploaded {len(uploaded)} file(s) · {len(df)} rows")
            with st.expander("Preview uploaded data", expanded=False):
                st.dataframe(df.head(20), width="stretch")
        except Exception as exc:
            st.error(str(exc))

    active = resolve_parquet_folder(
        upload_dir=st.session_state.get(session_upload_key),
        manual_path=folder_path,
        fallback_path=(meta or {}).get(meta_folder_key),
    )
    if active:
        st.info(f"Active folder: `{active}`")
    return active


def _sidebar() -> None:
    engines = list_engines()

    st.sidebar.title("Reco Engines")
    if not engines:
        st.sidebar.caption("No engines yet. Create one in **Reco Engine Generator**.")
    else:
        for name in engines:
            st.sidebar.markdown(f"**{name}**")
            col_remove, col_edit = st.sidebar.columns(2)
            with col_remove:
                if st.button("Remove", key=f"remove_{name}", width="stretch"):
                    delete_engine(name)
                    if st.session_state.selected_engine == name:
                        st.session_state.selected_engine = None
                    if st.session_state.edit_engine == name:
                        st.session_state.edit_engine = None
                    st.rerun()
            with col_edit:
                if st.button("Edit", key=f"edit_{name}", width="stretch"):
                    st.session_state.selected_engine = name
                    _go_builder(name)
                    st.rerun()
            st.sidebar.divider()

    st.sidebar.title("Navigation")
    if st.sidebar.button("Reco Engine Generator", width="stretch"):
        _go_builder(None)
        st.rerun()

    if engines:
        if st.sidebar.button("Reco Generator", width="stretch"):
            _go_generator()
            st.rerun()
    else:
        st.sidebar.caption("Reco Generator unlocks after you create an engine.")

    st.sidebar.divider()
    st.sidebar.subheader("LLM settings")

    # Re-checked on every rerun: report which provider keys were found in the
    # environment and the exact variable each one came from.
    _env_status = [
        ("OpenAI API key", OPENAI_API_KEY, OPENAI_ENV_SOURCE),
        ("HuggingFace API token", HUGGINGFACE_API_TOKEN, HUGGINGFACE_ENV_SOURCE),
        ("Google API key", GOOGLE_API_KEY, GOOGLE_ENV_SOURCE),
        ("Claude API key", CLAUDE_API_KEY, CLAUDE_ENV_SOURCE),
    ]
    for label, value, source in _env_status:
        if value:
            st.sidebar.success(f"✓ {label} loaded from environment ({source})")

    st.session_state.openai_api_key = st.sidebar.text_input(
        "OpenAI API key",
        type="password",
        value=st.session_state.get("openai_api_key", OPENAI_API_KEY),
        help="Leave empty to use HuggingFace instead. Auto-loaded from OPENAI_API_KEY env var.",
    )
    st.session_state.huggingface_api_key = st.sidebar.text_input(
        "HuggingFace API key",
        type="password",
        value=st.session_state.get("huggingface_api_key", HUGGINGFACE_API_TOKEN),
        help="Used as fallback if OpenAI key is not available. Auto-loaded from HF_TOKEN env var.",
    )
    st.session_state.google_api_key = st.sidebar.text_input(
        "Google API key",
        type="password",
        value=st.session_state.get("google_api_key", GOOGLE_API_KEY),
        help="Used when LLM platform is google. Auto-loaded from GOOGLE_API_KEY/GEMINI_API_KEY env var.",
    )
    if st.session_state.google_api_key:
        os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key
    st.session_state.claude_api_key = st.sidebar.text_input(
        "Claude API key",
        type="password",
        value=st.session_state.get("claude_api_key", CLAUDE_API_KEY),
        help="Used when LLM platform is claude. Auto-loaded from CLAUDE_KEY/ANTHROPIC_API_KEY env var.",
    )
    if st.session_state.claude_api_key:
        os.environ["CLAUDE_KEY"] = st.session_state.claude_api_key
    st.session_state.openai_model = st.sidebar.text_input(
        "Model",
        value=st.session_state.get("openai_model", "gpt-4o-mini"),
        help="OpenAI model name. If using HuggingFace, it will be mapped to a compatible model.",
    )


def _builder_page() -> None:
    global rag
    edit_name = st.session_state.edit_engine
    is_edit = edit_name is not None and engine_exists(edit_name)

    st.title("Reco Engine Generator")
    if is_edit:
        st.info(f"Editing engine **{edit_name}** — update fields or replace parquet data, then regenerate.")
        meta = load_engine_meta(edit_name)
        saved = load_engine_llminput(edit_name)
    else:
        saved = None
        meta = None

    upload_key = edit_name or "draft"
    if is_edit and meta:
        if meta.get("interactions_parquet_folder") and not st.session_state.get(
            "interactions_upload_dir"
        ):
            st.session_state.interactions_upload_dir = meta["interactions_parquet_folder"]
        if meta.get("users_parquet_folder") and not st.session_state.get("users_upload_dir"):
            st.session_state.users_upload_dir = meta["users_parquet_folder"]
        if meta.get("items_parquet_folder") and not st.session_state.get("items_upload_dir"):
            st.session_state.items_upload_dir = meta["items_parquet_folder"]

    active_interactions = _parquet_upload_section(
        title="User–item interactions (parquet only)",
        caption=(
            "Required columns: `user_id`, `item_id`, `action`, `timestamp`. "
            "Optional: `value`, `session_id`, `context`."
        ),
        dataset_kind="interactions",
        upload_key=upload_key,
        meta=meta,
        session_upload_key="interactions_upload_dir",
        meta_folder_key="interactions_parquet_folder",
        load_preview=load_parquet_interactions,
        default_folder=SAMPLE_INTERACTIONS_FOLDER,
    )

    active_users = _parquet_upload_section(
        title="User profile (parquet only)",
        caption=(
            "Required column: `user_id`. "
            "Optional: `segment`, `notes`. Column names are mapped below."
        ),
        dataset_kind="users",
        upload_key=upload_key,
        meta=meta,
        session_upload_key="users_upload_dir",
        meta_folder_key="users_parquet_folder",
        load_preview=load_parquet_users,
        default_folder=SAMPLE_USERS_FOLDER,
    )

    active_items = _parquet_upload_section(
        title="Item catalog (parquet only)",
        caption=(
            "Required columns: `item_id`, `title`, `category`. "
            "Optional: `tags`, `price`, `description`. Column names are mapped below."
        ),
        dataset_kind="items",
        upload_key=upload_key,
        meta=meta,
        session_upload_key="items_upload_dir",
        meta_folder_key="items_parquet_folder",
        load_preview=load_parquet_items,
        default_folder=SAMPLE_ITEMS_FOLDER,
    )

    st.subheader("Models")
    embedder_options = list(EMBEDDER_REGISTRY.keys())
    embedder_help = describe_embedders()
    platform_options = [name for name in FREE_LLM_REGISTRY if name != "anthropic"]
    col_embed, col_platform, col_model = st.columns(3)
    with col_embed:
        embedding_model = st.selectbox(
            "Embedding model",
            embedder_options,
            index=embedder_options.index(Configs.DEFAULT_EMBEDDING_MODEL_NAME)
            if Configs.DEFAULT_EMBEDDING_MODEL_NAME in embedder_options
            else 0,
            key="embedding_model_name",
            help="Encoder used to embed context prompts into the vector DB.",
        )
        st.caption(embedder_help.get(embedding_model, ""))
    with col_platform:
        llm_platform = st.selectbox(
            "LLM platform",
            platform_options,
            index=platform_options.index(Configs.DEFAULT_LLM_MODEL_NAME)
            if Configs.DEFAULT_LLM_MODEL_NAME in platform_options
            else 0,
            key="llm_platform",
            help="Provider used by the relevance ranker (RAG LLM calls).",
        )
    with col_model:
        llm_model_name = st.text_input(
            "Model name",
            value=PLATFORM_DEFAULT_MODELS.get(llm_platform, ""),
            key=f"llm_model_name_{llm_platform}",
            help="Model for the selected platform. Leave empty to use the platform default.",
        )

    with st.form("engine_form"):
        engine_name = st.text_input(
            "Engine name",
            value=edit_name or "",
            disabled=is_edit,
            help="Unique name shown in the sidebar.",
        )

        top_k = st.number_input(
            "{top_k} — recommendations count",
            min_value=1,
            max_value=50,
            value=int((saved or {}).get("top_k", 3)),
        )

        constraints = st.text_area(
            "{constraints}",
            value=(saved or {}).get(
                "constraints",
                "- Exclude items already purchased or disliked by the Target User.\n"
                "- Prefer items in categories the user engaged with recently.\n"
                "- Balance exploitation with one exploratory item.",
            ),
            height=100,
        )

        user_profile_columns = _column_mapping_inputs(
            "{user_profile}",
            DEFAULT_USER_PROFILE_COLUMNS,
            (saved or {}).get("user_profile_columns"),
            optional_fields=frozenset({"segment", "notes"}),
        )

        item_catalog_columns = _column_mapping_inputs(
            "{item_catalog}",
            DEFAULT_ITEM_CATALOG_COLUMNS,
            (saved or {}).get("item_catalog_columns"),
            optional_fields=frozenset({"tags", "price", "description"}),
        )

        llm_chat = st.text_area(
            "LLM chat (appended to end of prompt)",
            value=(saved or {}).get("llm_chat", ""),
            height=120,
            help="Extra instructions added to Section 9 of the rendered prompt.",
        )

        st.caption("Filled `recommender_prompt.md` — generated after you build the engine.")

        submitted = st.form_submit_button(
            "Generate Recommender Engine" if not is_edit else "Regenerate Recommender Engine",
            type="primary",
            width="stretch",
        )

    if submitted:
        try:
            if not is_edit and not engine_name.strip():
                raise ValueError("Engine name is required.")
            if is_edit:
                engine_name = edit_name
            elif engine_exists(engine_name):
                raise ValueError(f"Engine '{engine_name}' already exists. Choose another name or edit it.")

            if not active_interactions:
                raise ValueError(
                    "Interaction parquet is required. Browse files or enter a folder path above."
                )
            if not active_users:
                raise ValueError(
                    "User profile parquet is required. Browse files or enter a folder path above."
                )
            if not active_items:
                raise ValueError(
                    "Item catalog parquet is required. Browse files or enter a folder path above."
                )
            load_parquet_interactions(active_interactions)
            load_parquet_users(active_users)
            load_parquet_items(active_items)

            llminput = build_llminput_from_form(
                top_k=int(top_k),
                constraints=constraints,
                user_profile_columns=user_profile_columns,
                item_catalog_columns=item_catalog_columns,
                interactions_parquet_folder=active_interactions,
                users_parquet_folder=active_users,
                items_parquet_folder=active_items,
                llm_chat=llm_chat,
                embedding_model=embedding_model,
                llm_platform=llm_platform,
                llm_model=llm_model_name.strip(),
            )

            with st.spinner("Building Recommender Engine"):
                build_engine(
                    engine_name,
                    llminput,
                    interactions_parquet_folder=active_interactions,
                    users_parquet_folder=active_users,
                    items_parquet_folder=active_items,
                )
            with st.spinner("Reading datasets, ready built engine, and generating prompt"):
                try:
                    datasets = DataSets(engine_name)
                    datasets.get_data()
                except Exception as exc:
                    st.error(f"Error reading datasets: {exc}")  

            with st.spinner("Generating rendered prompt, item, user, and context prompts"):
                item_prompts = ItemPrompt(engine_name, datasets)
                item_prompts.build_item_feature_dataset()
                print(f"[DEBUG] Item Prompt sample: {str(item_prompts.context['generated_prompt'].iloc[0])[:200]}...")
                
                user_prompts = UserPrompt(engine_name, datasets)
                user_prompts.build_user_feature_dataset()
                print(f"[DEBUG] User Prompt sample: {str(user_prompts.context['generated_prompt'].iloc[0])[:200]}...")
                
                user_item_prompts = UserItemPrompt(engine_name, datasets)
                user_item_prompts.build_user_item_feature_dataset()
                print(f"[DEBUG] User-Item Prompt sample: {str(user_item_prompts.context['generated_prompt'].iloc[0])[:200]}...")
                
                context_prompts = RelevanceScorePrompt(
                    engine_name=engine_name, datasets=datasets, item_prompts=item_prompts, user_prompts=user_prompts, user_item_prompts=user_item_prompts)
                context_prompts.generate_rag_retrieval_context()
                print(f"[DEBUG] Context/Relevance Prompt sample: {str(context_prompts.context['generated_prompt'].iloc[0])[:200]}...")
            
            with st.spinner("Building DBs, retrieval, ranker, and KServe engine"):
                predictor = assemble_predictor(
                    engine_name,
                    datasets,
                    context_prompts,
                    embedding_model_name=embedding_model,
                    llm_platform=llm_platform,
                    llm_model=llm_model_name.strip() or None,
                    write=True,
                )
                predictors = st.session_state.setdefault("kserve_predictors", {})
                predictors[engine_name] = predictor
                print(f"[DEBUG] KServe predictor cached in session state for '{engine_name}'")

            st.session_state.selected_engine = engine_name
            st.session_state.edit_engine = None
            st.session_state.interactions_upload_dir = active_interactions
            st.session_state.users_upload_dir = active_users
            st.session_state.items_upload_dir = active_items
            st.session_state.last_prompt = load_rendered_prompt(engine_name)
            st.success(f"Reco engine **{engine_name}** created.")
            st.rerun()

        except Exception as exc:
            st.error(str(exc))

    if st.session_state.last_prompt and (is_edit or list_engines()):
        with st.expander("Latest rendered prompt", expanded=False):
            processed_prompt = _process_rendered_prompt(st.session_state.last_prompt)
            st.markdown(processed_prompt)


def _render_generator_results(current_engine: str) -> None:
    """Render the last recommendation run from session state.

    Driven by session state rather than the run button, so the recommendations
    table and its CSV download button persist across the rerun that Streamlit
    fires when the download button is clicked. Only shows results that belong to
    the currently selected engine (stale runs from another engine are ignored).
    """
    raw = st.session_state.get("last_response")
    if raw is None or st.session_state.get("last_reco_engine") != current_engine:
        return

    import pandas as pd

    target_user_id = st.session_state.get("last_reco_user", "user")

    st.subheader("Target user interactions")
    interactions = st.session_state.get("last_reco_interactions") or []
    if interactions:
        st.dataframe(pd.DataFrame(interactions), width="stretch")
    else:
        st.info("No interactions found.")

    st.subheader("Recommendations")
    rows = _extract_recommendation_rows(raw)
    logger.info("Extracted recommendation rows (count=%s)", len(rows))
    if rows:
        df_recs = pd.DataFrame(rows)
        st.dataframe(df_recs, width="stretch")
        csv_bytes = df_recs.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download recommendations (CSV)",
            data=csv_bytes,
            file_name=f"recommendations_{current_engine}_{target_user_id}.csv",
            mime="text/csv",
            key="download_recs_csv",
            width="stretch",
        )
    else:
        st.info("No structured recommendation rows found.")

    with st.expander("Recommendations in Markdown", expanded=True):
        _render_recommendations_markdown(raw)

    with st.expander("Raw model output", expanded=False):
        if isinstance(raw, dict):
            st.json(raw)
        else:
            st.code(raw)


def _generator_page() -> None:
    global rag
    engines = list_engines()
    logger.info("Entered Reco Generator page (engines=%s)", len(engines))
    if not engines:
        logger.warning("Reco Generator blocked: no engines available")
        st.warning("Create a reco engine first.")
        return

    st.title("Reco Generator")

    default_idx = 0
    if st.session_state.selected_engine in engines:
        default_idx = engines.index(st.session_state.selected_engine)

    engine_name = st.selectbox("Select reco engine", engines, index=default_idx)
    st.session_state.selected_engine = engine_name
    logger.info("Selected engine: %s", engine_name)

    meta = load_engine_meta(engine_name)
    base_llminput = load_engine_llminput(engine_name)
    logger.info(
        "Loaded engine meta (users_folder=%s, interactions_folder=%s, top_k=%s)",
        meta.get("users_parquet_folder"),
        meta.get("interactions_parquet_folder"),
        meta.get("top_k"),
    )

    # Bring the KServe predictor up as soon as the generator opens, rebuilding it
    # from persisted engine artifacts when it isn't already cached this session.
    predictors = st.session_state.get("kserve_predictors", {})
    if predictors.get(engine_name) is not None:
        st.success(f"✓ KServe engine ready for **{engine_name}**")
    else:
        try:
            with st.spinner(f"Starting KServe engine for '{engine_name}'…"):
                _ensure_predictor(engine_name)
            st.success(f"✓ KServe engine ready for **{engine_name}**")
        except Exception as exc:
            logger.exception("Failed to start KServe engine for '%s'", engine_name)
            st.error(f"Failed to start KServe engine: {exc}")

    users_folder = meta.get("users_parquet_folder")
    interactions_folder = meta.get("interactions_parquet_folder")
    user_ids: list[str] = []
    if users_folder:
        try:
            user_ids = list_parquet_user_ids(users_folder)
        except Exception as exc:
            logger.exception("Failed to load user ids from folder: %s", users_folder)
            st.warning(str(exc))

    saved_profile = base_llminput.get("user_profile") or {}
    default_target = saved_profile.get("user_id", user_ids[0] if user_ids else "")
    if user_ids:
        target_user_id = st.selectbox(
            "Target user",
            user_ids,
            index=user_ids.index(default_target) if default_target in user_ids else 0,
        )
    else:
        target_user_id = st.text_input(
            "Target user",
            value=default_target or "",
            help="Enter user_id from user profile parquet.",
        )

    st.caption(
        f"top_k: `{meta.get('top_k')}` · "
        f"items: `{meta.get('num_items')}` · "
        f"users: `{meta.get('num_users')}`"
    )

    with st.expander("Override parameters for this run (optional)"):
        run_top_k = st.number_input(
            "top_k override",
            min_value=1,
            max_value=50,
            value=int(base_llminput.get("top_k", 3)),
            key="run_top_k",
        )
        run_constraints = st.text_area(
            "constraints override",
            value=base_llminput.get("constraints", ""),
            key="run_constraints",
        )
        run_llm_chat = st.text_area(
            "LLM chat override (appended to prompt)",
            value=base_llminput.get("llm_chat", ""),
            key="run_llm_chat",
        )

    col_a, col_b = st.columns(2)
    with col_a:
        run_btn = st.button("Build prompt & run recommendation", type="primary", width="stretch")
    with col_b:
        preview_btn = st.button("Preview prompt only", width="stretch")
    if run_btn or preview_btn:
        logger.info("Generator action triggered (run_btn=%s, preview_btn=%s)", run_btn, preview_btn)

    if run_btn or preview_btn:
        overrides: dict[str, Any] = {
            "top_k": int(run_top_k),
            "constraints": run_constraints,
            "llm_chat": run_llm_chat,
        }

        try:
            if not target_user_id:
                logger.error("Generator validation failed: target user missing")
                raise ValueError("Target user is required.")
            if not users_folder or not interactions_folder:
                logger.error(
                    "Generator validation failed: missing parquet paths (users=%s, interactions=%s)",
                    users_folder,
                    interactions_folder,
                )
                raise ValueError(
                    "Engine is missing user or interaction parquet paths. Rebuild the engine."
                )
            run_llminput = llminput_for_target_user(
                target_user_id=str(target_user_id),
                base_llminput=base_llminput,
                interactions_parquet_folder=interactions_folder,
                users_parquet_folder=users_folder,
            )
            overrides.update(
                {
                    "user_profile": run_llminput["user_profile"],
                    "target_user_interactions": run_llminput["target_user_interactions"],
                    "other_users_interactions": run_llminput["other_users_interactions"],
                    "generated_at": run_llminput["generated_at"],
                }
            )
            logger.info(
                "Built run llminput (target_user=%s, target_interactions=%s, peers=%s)",
                target_user_id,
                len(run_llminput.get("target_user_interactions") or []),
                len(run_llminput.get("other_users_interactions") or []),
            )

            raw: dict[str, Any] | str | None = None
            with st.spinner("Calling KServe recommendation API…"):
                try:
                    logger.info("Calling KServe API for engine '%s' with user_id '%s'", engine_name, target_user_id)
                    
                    # Build API request for KServe predictor. The engine is
                    # selected via the request body, so one endpoint serves all.
                    api_request = {
                        "engine_name": engine_name,
                        "user_id": str(target_user_id),
                        "top_k": int(run_top_k)
                    }

                    # Call the KServe predictor via HTTP API. The first call for
                    # an engine also builds its server-side predictor (embedding +
                    # store write), so allow generous headroom beyond the parallel
                    # per-candidate LLM scoring. Override with RECO_API_TIMEOUT.
                    import requests
                    api_timeout = int(os.getenv("RECO_API_TIMEOUT", "600"))
                    try:
                        logger.info(f"Calling KServe endpoint: {KSERVE_API_URL} (timeout={api_timeout}s)")
                        response = requests.post(KSERVE_API_URL, json=api_request, timeout=api_timeout)
                        response.raise_for_status()
                        raw = response.json()
                        logger.info("KServe API call succeeded")
                    except requests.exceptions.ConnectionError:
                        # Fallback: call predictor directly if server not available
                        logger.warning("KServe server not responding, using direct predictor call")
                        st.warning("⚠️ KServe server not available, using direct predictor (may be slower)")
                        # Store the predictor in session state during builder, retrieve it here
                        if "kserve_predictors" in st.session_state and engine_name in st.session_state["kserve_predictors"]:
                            predictor = st.session_state["kserve_predictors"][engine_name]
                            raw = predictor.predict(api_request)
                        else:
                            raise RuntimeError(
                                f"KServe server not available and no cached predictor for '{engine_name}'. "
                                "Ensure the engine was built successfully and the server is running."
                            )
                    
                    st.session_state.last_response = raw
                    
                except Exception as e:
                    logger.exception("KServe API call failed in generator flow")
                    st.error(f"KServe API call failed: {str(e)}")
                    st.exception(e)

                if raw is not None:
                    # Persist results so the table + CSV download survive the
                    # rerun Streamlit triggers on a download-button click (the
                    # run_btn-gated block would otherwise not re-execute).
                    st.session_state.last_response = raw
                    st.session_state.last_reco_interactions = (
                        run_llminput.get("target_user_interactions") or []
                    )
                    st.session_state.last_reco_engine = engine_name
                    st.session_state.last_reco_user = str(target_user_id)

        except Exception as exc:
            logger.exception("Generator flow failed")
            st.error(str(exc))

    # Render (persisted) results outside the run-button gate so the CSV download
    # button keeps working across reruns.
    _render_generator_results(engine_name)


def main() -> None:
    # Bring the KServe API up on startup, but only once at least one engine
    # exists — nothing to serve otherwise. st.cache_resource keeps it to a
    # single background server per process across reruns.
    if list_engines():
        try:
            _start_kserve_api()
        except Exception:
            logger.exception("Failed to start KServe API server")

    _sidebar()

    if st.session_state.page == "generator" and list_engines():
        _generator_page()
    else:
        _builder_page()


if __name__ == "__main__":
    import sys

    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is None:
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
        sys.exit(stcli.main())

    main()
