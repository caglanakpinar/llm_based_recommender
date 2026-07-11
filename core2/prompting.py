import numpy as np
import pandas as pd

from core2.configs import Configs
from core2.datasets import DataSets
from core2.embeddings import create_embedder
from core2.features import FEATURES




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

    def build_user_feature_dataset(self) -> pd.DataFrame:
        """Build feature rows for unique users from FEATURES['user_features']."""
        self.context = self.users.groupby(self.user_id).first().reset_index()
        kwargs = {'user_id': self.user_id}
        user_feature_funcs = FEATURES.get("user_features", {})
        for feature_name, feature_fn in user_feature_funcs.items():
            _feature_df = feature_fn(self.users, **kwargs)
            self.context = self.context.merge(
                _feature_df, on=self.user_id, how="left").fillna(0) 

        self.context["generated_prompt"] = self.context.apply(self._prompt_from_row, axis=1)

            
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
        self.items = datasets.item
        self.users = datasets.user
        self.prompt = BasePrompt(engine_name, self.user_item_pair_prompt_path)
        self.context = pd.DataFrame()

    def _prompt_from_row(self, row: pd.Series) -> str:
        args = {k: v for k, v in row.items() if pd.notnull(v)}
        return self.prompt.load_prompt_template(args)   

    def build_user_item_feature_dataset(self) -> pd.DataFrame:
        """Build feature rows for unique user-item pairs from FEATURES['user_item_features']."""
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
        print(args.keys())
        return self.prompt.load_prompt_template(args)

    def generate_rag_retrieval_context(self) -> pd.DataFrame:
        # Ensure consistent dtypes by converting to string
        print(self.user_prompts.head())
        print(self.item_prompts.head())
        print(self.user_item_prompts.head())
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
        print(self.context.head())

        self.context["generated_prompt"] = self.context.apply(self._prompt_from_row, axis=1)
        self.context['relevance_score'] = FEATURES['user_item_features']['relevance_score'](self.context, kwargs={'user_id': self.user_id, 'item_id': self.item_id})


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

    def build_relevance_score_prompt(self, user_id: str, item_id: str) -> str:
        """Build a relevance score prompt for a given user-item pair."""
        user_row = self.user_prompts[self.user_prompts[self.user_id] == user_id]['generated_prompt'].iloc[0]
        item_row = self.item_prompts[self.item_prompts[self.item_id] == item_id]['generated_prompt'].iloc[0]
        user_item_row = self.user_item_prompts[
            (self.user_item_prompts[self.user_id] == user_id) & 
            (self.user_item_prompts[self.item_id] == item_id)
        ]['generated_prompt'].iloc[0]
        args = {"user_prompt": user_row, "item_prompt": item_row, "user_item_prompt": user_item_row}
        return self.prompt.load_prompt_template(args)


    