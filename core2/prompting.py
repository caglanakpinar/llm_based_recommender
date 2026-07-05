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
            
class ItemPPrompt(Configs):
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

    def __init__(self, engine_name, datasets: DataSets):
        super().__init__(engine_name)
        self.prompt = BasePrompt(self.relevance_score_prompt_path)

    def generate_relevance_score_prompt(self, user_features: Dict[str, Any], item_features: Dict[str, Any]) -> str:
        """Generate a relevance score prompt based on user and item features."""
        args = {**user_features, **item_features}
        return self.prompt.load_prompt_template(args)