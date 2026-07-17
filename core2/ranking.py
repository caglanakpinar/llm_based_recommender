import re

import pandas as pd

from core2.configs import Configs
from core2.datasets import DataSets
from core2.prompting import RelevanceScorePrompt
from core2.prompting import UserPrompt
from core2.features import FEATURES, relevance_score
from core2.prompting import ItemPrompt
from core2.llms import create_llm
from core2.retrieval import Retrieval

class BaseRanking(Configs): 
    """Base class for ranking operations."""

    def __init__(
            self, engine_name: str, datasets: DataSets,
            context_prompts: RelevanceScorePrompt,
                 
        ):
        project_name = self.project_name_for(engine_name)
        super().__init__(project_name=project_name)
        self.items = datasets.item
        self.users = datasets.user
        self.items_users = datasets.item_user
        self.item_prompts = context_prompts.item_prompts
        self.user_prompts = context_prompts.user_prompts
        self.user_item_prompts = context_prompts.user_item_prompts
        self.context = self.items_users[[self.user_id, self.item_id]].copy()

    
class BaseRelevanceRanking(BaseRanking):
    """Base class for relevance ranking operations."""

    def __init__(
            self, engine_name: str, datasets: DataSets,
            context_prompts: RelevanceScorePrompt,
                 
        ):
        super().__init__(engine_name, datasets, context_prompts)

    def generate_scores(self) -> pd.DataFrame:
        """Generate relevance scores for user-item pairs."""
        self.context = self.context.merge(
            self.user_prompts[[self.user_id] + FEATURES['user_features']], on=self.user_id, how="left"
        ).merge(
            self.item_prompts[[self.item_id] + FEATURES['item_features']], on=self.item_id, how="left"
        ).merge(
            self.user_item_prompts[[self.user_id, self.item_id, 'generated_prompt'] + FEATURES['user_item_features']], on=[self.user_id, self.item_id], how="left"
        )
        relevance_score_df = relevance_score(
            self.context, 
            kwargs={'user_id': self.user_id, 'item_id': self.item_id}
        )
        self.context = self.context.merge(
            relevance_score_df, on=[self.user_id, self.item_id], how="left"
        )
        self.context['generated_prompt'] = self.context.apply(lambda row: row['generated_prompt'] + " " + str(row['relevance_score']), axis=1)


class LLMRanker(Configs):
    """LLM-based ranker for generating relevance scores."""

    def __init__(
            self, 
            engine_name: str, 
            datasets: DataSets,
            retrieval: Retrieval,
            context_prompts: RelevanceScorePrompt,
            llm_model_name: str = Configs.DEFAULT_LLM_MODEL_NAME

        ):
        super().__init__(engine_name)
        self.retrieval = retrieval
        self.llm_model_name = create_llm(llm_model_name, engine_name)
        self.embedder = self.retrieval.embedder._model
        self.context_prompts = context_prompts

    def generate_scores(self, user_id, item_id) -> dict:
        # 1. Generate Query prompts for user-item pairs
        user_query = self.context_prompts.build_relevance_score_prompts(user_id, item_id)
        # 2. Retrieve context documents based on query prompts
        retrieved_context = self.retrieval.retrieve_context(user_query)
        # 3. Generate relevance scores using LLM
        rag_prompt = (
            f"Context: {retrieved_context}\n"
            f"Question: {user_query}\n"
            f"Answer:"
        )
        score_response = self.llm_model_name.call(rag_prompt)
        print(f"User: {user_id}, Item: {item_id}, Score Response: {score_response}")
        return {"item_id": item_id, "relevance_score": self._parse_score(score_response)}

    @staticmethod
    def _parse_score(score_response) -> float:
        if isinstance(score_response, (int, float)):
            return float(score_response)
        match = re.search(r"-?\d*\.?\d+", str(score_response))
        return float(match.group()) if match else 0.0


