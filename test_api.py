from core2.datasets import DataSets
from core2.prompting import (
    RelevanceScorePrompt, 
    UserPrompt, 
    ItemPrompt, 
    UserItemPrompt
)
from core2.dbs import ContextVectorDB, ContextDB
from core2.ranking import LLMRanker
from core2.retrieval import Retrieval
from core2.reco_engine import BuildRecoEngine


engine_name = "test_reco_engine"
datasets = DataSets('test_reco_engine')
datasets.get_data()

print(datasets.item)

item_prompts = ItemPrompt(engine_name, datasets)
item_prompts.build_item_feature_dataset()
print(item_prompts.context['generated_prompt'])
print(f"[DEBUG] Item Prompt sample: {str(item_prompts.context['generated_prompt'].iloc[0])}...")

user_prompts = UserPrompt(engine_name, datasets)
user_prompts.build_user_feature_dataset()
print(f"[DEBUG] User Prompt sample: {str(user_prompts.context['generated_prompt'].iloc[0])[:200]}...")

user_item_prompts = UserItemPrompt(engine_name, datasets)
user_item_prompts.build_user_item_feature_dataset()
print(f"[DEBUG] User-Item Prompt sample: {str(user_item_prompts.context['generated_prompt'].iloc[0])[:200]}...")

context_prompts = RelevanceScorePrompt(
    engine_name, datasets, item_prompts, user_prompts, user_item_prompts)
context_prompts.generate_rag_retrieval_context()
print()
print(f"[DEBUG] Context/Relevance Prompt sample: {str(context_prompts.context['generated_prompt'].iloc[0])[:200]}...")