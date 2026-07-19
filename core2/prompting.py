from pathlib import Path

import numpy as np
import pandas as pd

from core2.configs import Configs
from core2.datasets import DataSets
from core2.embeddings import create_embedder
from core2.features import FEATURES


def _load_prompt_cache(cache_path: Path) -> pd.DataFrame | None:
    """Return the cached prompt DataFrame if present, else None."""
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"[DEBUG] Loaded cached prompts from '{cache_path}' ({len(df)} rows)")
        return df
    return None


def _save_prompt_cache(cache_path: Path, df: pd.DataFrame) -> None:
    """Persist a built prompt DataFrame so later runs skip the pandas build."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in out.columns:
        if out[col].dtype != object:
            continue
        # fillna(0) in the feature merges leaves int 0s inside string columns
        # (e.g. session_id), which pyarrow refuses to serialize — normalize
        # mixed-type object columns to plain strings for the cache.
        non_null = out[col].dropna()
        if not non_null.empty and non_null.map(type).nunique() > 1:
            out[col] = out[col].astype(str)
    out.to_parquet(cache_path)
    print(f"[DEBUG] Saved prompt cache to '{cache_path}' ({len(out)} rows)")


class BasePrompt(Configs):
    """Base class for prompt templates used by vector databases."""
    def __init__(self, engine_name: str, prompt_path: str):
        super().__init__(engine_name)
        self.prompt_path = prompt_path or Configs.DEFAULT_RECO_PROMPT_FILE
        
    def load_prompt_template(self, args: dict[str, str] = None) -> str:
        """Load prompt template text if the configured prompt file exists."""
        prompt_path = self.resolve_repo_path(self.prompt_path)
        if prompt_path.exists():
            import string
            template = prompt_path.read_text(encoding="utf-8")
            # Use safe_substitute-style: fill known keys, leave unknown as empty
            class _SafeDict(dict):
                def __missing__(self, key):
                    return ""
            return template.format_map(_SafeDict(**(args or {})))
        return ""


class UserPrompt(Configs):
    """ User prompt generator for vector database operations. """

    def __init__(self, engine_name: str, datasets: DataSets):
        super().__init__(engine_name)
        self.users = datasets.user
        self.prompt = BasePrompt(engine_name, self.user_prompt_path)
        self.context = pd.DataFrame()

    def _prompt_from_row(self, row: pd.Series) -> str:
        args = {k: v for k, v in row.items() if pd.notnull(v)}
        return self.prompt.load_prompt_template(args)   

    def build_user_feature_dataset(self, use_cache: bool = True) -> pd.DataFrame:
        """Build feature rows for unique users from FEATURES['user_features']."""
        cache_path = self.engine_root / "cache" / "user_prompts.parquet"
        if use_cache:
            cached = _load_prompt_cache(cache_path)
            if cached is not None:
                self.context = cached
                return
        self.context = self.users.groupby(self.user_id).first().reset_index()
        kwargs = {'user_id': self.user_id}
        user_feature_funcs = FEATURES.get("user_features", {})
        for feature_name, feature_fn in user_feature_funcs.items():
            _feature_df = feature_fn(self.users, **kwargs)
            self.context = self.context.merge(
                _feature_df, on=self.user_id, how="left").fillna(0)

        self.context["generated_prompt"] = self.context.apply(self._prompt_from_row, axis=1)
        _save_prompt_cache(cache_path, self.context)

            
class ItemPrompt(Configs):
    """ Item prompt generator for vector database operations. """

    def __init__(self, engine_name, datasets: DataSets):
        super().__init__(engine_name)
        self.items = datasets.item
        self.prompt = BasePrompt(engine_name, self.item_prompt_path)
        self.context = pd.DataFrame()

    def _prompt_from_row(self, row: pd.Series) -> str:
        args = {k: v for k, v in row.items() if pd.notnull(v)}
        return self.prompt.load_prompt_template(args)   

    def build_item_feature_dataset(self, use_cache: bool = True) -> pd.DataFrame:
        """Build feature rows for unique items from FEATURES['item_features']."""
        cache_path = self.engine_root / "cache" / "item_prompts.parquet"
        if use_cache:
            cached = _load_prompt_cache(cache_path)
            if cached is not None:
                self.context = cached
                return
        self.context = self.items.groupby(self.item_id).first().reset_index()
        kwargs = {'user_id': self.user_id}
        user_feature_funcs = FEATURES.get("item_features", {})
        for feature_name, feature_fn in user_feature_funcs.items():
            _feature_df = feature_fn(self.items, **kwargs)
            self.context = self.context.merge(
                _feature_df, on=self.item_id, how="left").fillna(0)

        self.context["generated_prompt"] = self.context.apply(self._prompt_from_row, axis=1)
        _save_prompt_cache(cache_path, self.context)


class UserItemPrompt(Configs):
    """ User-Item prompt generator for vector database operations. """

    def __init__(
            self, 
            engine_name, 
            datasets: DataSets,
        ):
        super().__init__(engine_name)
        self.user_item = datasets.item_user
        self.items = datasets.item
        self.users = datasets.user
        self.prompt = BasePrompt(engine_name, self.user_item_pair_prompt_path)
        self.context = pd.DataFrame()

    def _prompt_from_row(self, row: pd.Series) -> str:
        args = {k: v for k, v in row.items() if pd.notnull(v)}
        return self.prompt.load_prompt_template(args)   

    def build_user_item_feature_dataset(self, use_cache: bool = True) -> pd.DataFrame:
        """Build feature rows for unique user-item pairs from FEATURES['user_item_features']."""
        cache_path = self.engine_root / "cache" / "user_item_prompts.parquet"
        if use_cache:
            cached = _load_prompt_cache(cache_path)
            if cached is not None:
                self.context = cached
                return
        self.context = self.user_item.groupby([self.user_id, self.item_id]).first().reset_index().merge(
            self.users, on=self.user_id, how="left"
        ).merge(
            self.items, on=self.item_id, how="left"
        )
        kwargs = {'user_id': self.user_id, 'item_id': self.item_id}
        user_item_feature_funcs = FEATURES.get("user_item_pair_features", {})
        for feature_name, feature_fn in user_item_feature_funcs.items():
            _feature_df = feature_fn(self.user_item, **kwargs)
            self.context = self.context.merge(
                _feature_df, on=[self.user_id, self.item_id], how="left").fillna(0)

        self.context["generated_prompt"] = self.context.apply(self._prompt_from_row, axis=1)
        _save_prompt_cache(cache_path, self.context)


class RelevanceScorePrompt(Configs):
    """ Relevance score prompt generator for vector database operations. """

    def __init__(self, engine_name, datasets: DataSets, user_prompts: UserPrompt, item_prompts: ItemPrompt, user_item_prompts: UserItemPrompt):
        super().__init__(engine_name)
        self.user_prompts = user_prompts.context
        self.item_prompts = item_prompts.context
        self.user_item_prompts = user_item_prompts.context
        self.item_users = datasets.item_user
        self.context = pd.DataFrame()
        self.context_wt_relevance_score = pd.DataFrame()
        self.prompt = BasePrompt(engine_name, self.relevance_score_prompt_path)

    def _prompt_from_row(self, row: pd.Series) -> str:
        args = {k: v for k, v in row.items() if pd.notnull(v)}
        return self.prompt.load_prompt_template(args)

    def generate_rag_retrieval_context(self, use_cache: bool = True) -> pd.DataFrame:
        cache_path = self.engine_root / "cache" / "rag_context.parquet"
        if use_cache:
            cached = _load_prompt_cache(cache_path)
            if cached is not None:
                self.context = cached
                self.context_wt_relevance_score = cached.copy()
                return
        # Ensure consistent dtypes by converting to string
        item_users_dup = self.item_users.drop_duplicates([self.user_id, self.item_id]).copy()
        item_users_dup[self.user_id] = item_users_dup[self.user_id].astype(str)
        item_users_dup[self.item_id] = item_users_dup[self.item_id].astype(str)
        
        user_prompts_copy = self.user_prompts.copy()
        user_prompts_copy[self.user_id] = user_prompts_copy[self.user_id].astype(str)
        
        item_prompts_copy = self.item_prompts.copy()
        item_prompts_copy[self.item_id] = item_prompts_copy[self.item_id].astype(str)
        
        user_item_prompts_copy = self.user_item_prompts.copy()
        user_item_prompts_copy[self.user_id] = user_item_prompts_copy[self.user_id].astype(str)
        user_item_prompts_copy[self.item_id] = user_item_prompts_copy[self.item_id].astype(str)
        
        self.context = item_users_dup.merge(
            user_prompts_copy[[self.user_id, "generated_prompt"]].rename(columns={"generated_prompt": "user_prompt"}),
            on=self.user_id,
            how="left"
        ).merge(
            item_prompts_copy[[self.item_id, "generated_prompt"]].rename(columns={"generated_prompt": "item_prompt"}),
            on=self.item_id,
            how="left"
        ).merge(
            user_item_prompts_copy[[self.user_id, self.item_id, "generated_prompt"]].rename(columns={"generated_prompt": "user_item_prompt"}),
            on=[self.user_id, self.item_id],
            how="left"
        )

        self.context["generated_prompt"] = self.context.apply(self._prompt_from_row, axis=1)
        relevance_score_df = (
            FEATURES
            ['user_item_pair_features']
            ['relevance_score']
            (
                self.context, 
                kwargs={'user_id': self.user_id, 'item_id': self.item_id}
            )
        )
        self.context = self.context.merge(
            relevance_score_df, on=[self.user_id, self.item_id], how="left"
        )
        self.context_wt_relevance_score = self.context.copy()
        _save_prompt_cache(cache_path, self.context)


    def build_retrieval_context(self) -> pd.DataFrame:
        db_prompts = []
        for user_id, item_id, relevance_score in self.user_prompts[[self.user_id, self.item_id, 'relevance_score']].itertuples(index=False):
            prompt = self.build_relevance_score_prompt()
            db_prompts.append({
                self.user_id: user_id,
                self.item_id: item_id,
                "generated_prompt": prompt + f"  {str(relevance_score)}"
            })
        self.context_wt_relevance_score = pd.DataFrame(db_prompts)

    def build_relevance_score_prompts(self, user_id: str, item_id: str) -> str:
        """Build a relevance score prompt for a given user-item pair."""
        user_matches = self.user_prompts[self.user_prompts[self.user_id] == user_id]['generated_prompt']
        user_row = user_matches.iloc[0] if not user_matches.empty else ""
        item_matches = self.item_prompts[self.item_prompts[self.item_id] == item_id]['generated_prompt']
        item_row = item_matches.iloc[0] if not item_matches.empty else ""
        user_item_matches = self.user_item_prompts[
            (self.user_item_prompts[self.user_id] == user_id) &
            (self.user_item_prompts[self.item_id] == item_id)
        ]['generated_prompt']
        # No prior interaction between this user and candidate item (e.g. retrieved via similarity, not history).
        user_item_row = user_item_matches.iloc[0] if not user_item_matches.empty else ""
        args = {"user_prompt": user_row, "item_prompt": item_row, "user_item_prompt": user_item_row}
        return self.prompt.load_prompt_template(args)
    