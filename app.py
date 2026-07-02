"""Streamlit UI for reco engine builder and recommendation generator."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

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

st.set_page_config(
    page_title="LLM Recommender",
    page_icon="🎯",
    layout="wide",
)

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


def _go_builder(edit_name: str | None = None) -> None:
    st.session_state.page = "builder"
    st.session_state.edit_engine = edit_name


def _go_generator() -> None:
    st.session_state.page = "generator"


def _call_llm(prompt: str, api_key: str, model: str) -> str:
    """Call OpenAI when available, otherwise fall back to local/free providers."""
    def _fallback_call() -> str:
        from core.llms import FreeLLMCaller

        for provider in ("ollama", "huggingface", "replicate"):
            try:
                caller = FreeLLMCaller(api_key=None, model=model, provider=provider)
                return caller.call(prompt)
            except Exception:
                continue
        raise RuntimeError(
            "No fallback LLM provider succeeded. Install/start Ollama or configure HF_TOKEN/REPLICATE_API_TOKEN."
        )

    if api_key is None or api_key.strip() == "":
        return _fallback_call()

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
        return _fallback_call()


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
        value=(meta or {}).get(meta_folder_key) or "",
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
                st.dataframe(df.head(20), use_container_width=True)
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
                if st.button("Remove", key=f"remove_{name}", use_container_width=True):
                    delete_engine(name)
                    if st.session_state.selected_engine == name:
                        st.session_state.selected_engine = None
                    if st.session_state.edit_engine == name:
                        st.session_state.edit_engine = None
                    st.rerun()
            with col_edit:
                if st.button("Edit", key=f"edit_{name}", use_container_width=True):
                    st.session_state.selected_engine = name
                    _go_builder(name)
                    st.rerun()
            st.sidebar.divider()

    st.sidebar.title("Navigation")
    if st.sidebar.button("Reco Engine Generator", use_container_width=True):
        _go_builder(None)
        st.rerun()

    if engines:
        if st.sidebar.button("Reco Generator", use_container_width=True):
            _go_generator()
            st.rerun()
    else:
        st.sidebar.caption("Reco Generator unlocks after you create an engine.")

    st.sidebar.divider()
    st.sidebar.subheader("LLM settings")
    st.session_state.openai_api_key = st.sidebar.text_input(
        "OpenAI API key",
        type="password",
        value=st.session_state.get("openai_api_key", ""),
    )
    st.session_state.openai_model = st.sidebar.text_input(
        "Model",
        value=st.session_state.get("openai_model", "gpt-4o-mini"),
    )


def _builder_page() -> None:
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
            "Generate embedding store" if not is_edit else "Regenerate embedding store",
            type="primary",
            use_container_width=True,
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
            )

            with st.spinner("Building embeddings (items, users, prompt chunks)…"):
                build_engine(
                    engine_name,
                    llminput,
                    interactions_parquet_folder=active_interactions,
                    users_parquet_folder=active_users,
                    items_parquet_folder=active_items,
                )

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
            st.markdown(st.session_state.last_prompt)


def _generator_page() -> None:
    engines = list_engines()
    if not engines:
        st.warning("Create a reco engine first.")
        return

    st.title("Reco Generator")

    default_idx = 0
    if st.session_state.selected_engine in engines:
        default_idx = engines.index(st.session_state.selected_engine)

    engine_name = st.selectbox("Select reco engine", engines, index=default_idx)
    st.session_state.selected_engine = engine_name

    meta = load_engine_meta(engine_name)
    base_llminput = load_engine_llminput(engine_name)

    users_folder = meta.get("users_parquet_folder")
    interactions_folder = meta.get("interactions_parquet_folder")
    user_ids: list[str] = []
    if users_folder:
        try:
            user_ids = list_parquet_user_ids(users_folder)
        except Exception as exc:
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
        run_btn = st.button("Build prompt & run recommendation", type="primary", use_container_width=True)
    with col_b:
        preview_btn = st.button("Preview prompt only", use_container_width=True)

    if run_btn or preview_btn:
        overrides: dict[str, Any] = {
            "top_k": int(run_top_k),
            "constraints": run_constraints,
            "llm_chat": run_llm_chat,
        }
        store = get_engine_store(engine_name)
        try:
            if not target_user_id:
                raise ValueError("Target user is required.")
            if not users_folder or not interactions_folder:
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

            with st.spinner("Preparing prompt…"):
                session = store.generate_prompt_embeddings(
                    llminput_overrides=overrides,
                    base_llminput=base_llminput,
                    save_session=True,
                )
            prompt = session["rendered_prompt"]
            st.session_state.last_prompt = prompt

            st.subheader("Prompt sent to LLM")
            st.code(prompt, language="markdown")

            if preview_btn:
                st.info("Prompt preview only — no LLM call.")
            elif run_btn:
                api_key = st.session_state.get("openai_api_key", "").strip()
                if not api_key:
                    st.warning("Add your OpenAI API key in the sidebar to run the LLM.")
                else:
                    with st.spinner("Calling LLM…"):
                        raw = _call_llm(
                            prompt,
                            api_key,
                            st.session_state.get("openai_model", "gpt-4o-mini"),
                        )
                    st.session_state.last_response = raw
                    st.subheader("Recommendation response")
                    try:
                        parsed = json.loads(raw)
                        st.json(parsed)
                    except json.JSONDecodeError:
                        st.code(raw)

        except Exception as exc:
            st.error(str(exc))

    elif st.session_state.last_prompt:
        st.subheader("Last prompt")
        st.code(st.session_state.last_prompt, language="markdown")
        if st.session_state.last_response:
            st.subheader("Last LLM response")
            st.code(st.session_state.last_response)


def main() -> None:
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
