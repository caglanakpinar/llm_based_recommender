from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from core2.configs import Configs
from core2.datasets import DataSets
from core2.embeddings import create_embedder
from core2.features import FEATURES


class BasePrompt:
    """Base class for prompt templates used by vector databases."""
    def __init__(self, prompt_path: str = None):
        self.prompt_path = prompt_path or Configs.DEFAULT_RECO_PROMPT_FILE
        
    def load_prompt_template(self, args: dict[str, str] = None) -> str:
        """Load prompt template text if the configured prompt file exists."""
        prompt_path = self.resolve_repo_path(self.prompt_path)
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").format(**(args or {}))
        return ""


class UserPrompt(Configs):
    """ User prompt generator for vector database operations. """

    def __init__(self, engine_name, datasets: DataSets):
        super().__init__(engine_name)
        self.users = datasets.user
        self.prompt = BasePrompt(self.user_prompt_path)
        self.context = pd.DataFrame()

    def _prompt_from_row(self, row: pd.Series) -> str:
        args = {k: v for k, v in row.items() if pd.notnull(v)}
        return self.prompt.load_prompt_template(args)   

    def build_user_feature_dataset(self) -> pd.DataFrame:
        """Build feature rows for unique users from FEATURES['user_features']."""
        self.context = self.users.groupby(self.user_id).first().reset_index()
        kwargs = {'user_id': self.user_id}
        user_feature_funcs = FEATURES.get("user_features", {})
        for feature_name, feature_fn in user_feature_funcs.items():
            _feature_df = feature_fn(self.users, **kwargs)
            self.context = self.context.merge(
                _feature_df, on=self.user_id, how="left").fillna(0) 
            
class ItemPrompt(Configs):
    """ Item prompt generator for vector database operations. """

    def __init__(self, engine_name, datasets: DataSets):
        super().__init__(engine_name)
        self.items = datasets.item
        self.prompt = BasePrompt(self.item_prompt_path)
        self.context = pd.DataFrame()

    def _prompt_from_row(self, row: pd.Series) -> str:
        args = {k: v for k, v in row.items() if pd.notnull(v)}
        return self.prompt.load_prompt_template(args)   

    def build_item_feature_dataset(self) -> pd.DataFrame:
        """Build feature rows for unique items from FEATURES['item_features']."""
        self.context = self.items.groupby(self.item_id).first().reset_index()
        kwargs = {'user_id': self.user_id}
        user_feature_funcs = FEATURES.get("item_features", {})
        for feature_name, feature_fn in user_feature_funcs.items():
            _feature_df = feature_fn(self.items, **kwargs)
            self.context = self.context.merge(
                _feature_df, on=self.item_id, how="left").fillna(0) 
        
        self.context["generated_prompt"] = self.context.apply(self._prompt_from_row, axis=1)


class UserItemPrompt(Configs):
    """ User-Item prompt generator for vector database operations. """

    def __init__(
            self, 
            engine_name, 
            datasets: DataSets,
        ):
        super().__init__(engine_name)
        self.user_item = datasets.item_user
        self.prompt = BasePrompt(self.user_item_pair_prompt_path)
        self.context = pd.DataFrame()

    def _prompt_from_row(self, row: pd.Series) -> str:
        args = {k: v for k, v in row.items() if pd.notnull(v)}
        return self.prompt.load_prompt_template(args)   

    def build_user_item_feature_dataset(self) -> pd.DataFrame:
        """Build feature rows for unique user-item pairs from FEATURES['user_item_features']."""
        self.context = self.user_item.groupby([self.user_id, self.item_id]).first().reset_index()
        kwargs = {'user_id': self.user_id, 'item_id': self.item_id}
        user_item_feature_funcs = FEATURES.get("user_item_features", {})
        for feature_name, feature_fn in user_item_feature_funcs.items():
            _feature_df = feature_fn(self.user_item, **kwargs)
            self.context = self.context.merge(
                _feature_df, on=[self.user_id, self.item_id], how="left").fillna(0) 
        
        self.context["generated_prompt"] = self.context.apply(self._prompt_from_row, axis=1)


class RelevanceScorePrompt(Configs):
    """ Relevance score prompt generator for vector database operations. """

    def __init__(self, engine_name, datasets: DataSets, user_prompts: UserPrompt, item_prompts: ItemPrompt, user_item_prompts: UserItemPrompt):
        super().__init__(engine_name)
        self.user_prompts = user_prompts.context
        self.item_prompts = item_prompts.context
        self.user_item_prompts = user_item_prompts.context
        self.item_users = datasets.item_user
        self.context = pd.DataFrame()
        self.prompt = BasePrompt(self.relevance_score_prompt_path)

    def _prompt_from_row(self, row: pd.Series) -> str:
        args = {k: v for k, v in row.items() if pd.notnull(v)}
        return self.prompt.load_prompt_template(args)

    def generate_rag_retrieval_context(self) -> pd.DataFrame:
        self.context = self.context[[self.user_id, self.item_id]].merge(
            self.user_prompts[[self.user_id, "generated_prompt"]].rename(columns={"generated_prompt": "user_prompt"}),
            on=self.user_id,
            how="left"
        ).rename(columns={"user_prompt": "user_prompt"}).merge(
            self.item_prompts[[self.item_id, "generated_prompt"]].rename(columns={"generated_prompt": "item_prompt"}),
            on=self.item_id,
            how="left"
        ).rename(columns={"item_prompt": "item_prompt"}).merge(
            self.user_item_prompts[[self.user_id, self.item_id, "generated_prompt"]].rename(columns={"generated_prompt": "user_item_prompt"}),
            on=[self.user_id, self.item_id],
            how="left"
        ).rename(columns={"user_item_prompt": "user_item_pair_prompt"})

        self.context["generated_prompt"] = self.context.apply(self._prompt_from_row, axis=1)

    def build_relevance_score_prompt(self, user_id: str, item_id: str) -> str:
        """Build a relevance score prompt for a given user-item pair."""
        user_row = self.user_prompts[self.user_prompts[self.user_id] == user_id]['generated_prompt'].iloc[0]
        item_row = self.item_prompts[self.item_prompts[self.item_id] == item_id]['generated_prompt'].iloc[0]
        user_item_row = self.user_item_prompts[
            self.user_item_prompts[self.user_id] == user_id & 
            self.user_item_prompts[self.item_id] == item_id
        ]['generated_prompt'].iloc[0]

        if user_row.empty or item_row.empty or user_item_row.empty:
            raise ValueError(f"User or Item or User-Item pair not found for user_id={user_id}, item_id={item_id}")

        args = {**user_row.iloc[0].to_dict(), **item_row.iloc[0].to_dict(), **user_item_row.iloc[0].to_dict()}
        return self.prompt.load_prompt_template(args)
    