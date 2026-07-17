from core2.configs import Configs
from core2.dbs import ContextDB, ContextVectorDB
import numpy as np
import pandas as pd


class Retrieval(Configs):
    """Class for retrieval operations."""  
    def __init__(self, engine_name: str, datasets, context_prompts, context_vector_db: ContextVectorDB, context_db: ContextDB):
        super().__init__(project_name=engine_name)
        self.datasets = datasets
        self.context_prompts = context_prompts
        self.context_vector_db = context_vector_db
        self.context_db = context_db
        self.embedder = context_vector_db.embedder
        self.user_items = context_db.context if hasattr(context_db, 'context') else pd.DataFrame()
        self.candidates = self.users_last_interactions() if not self.user_items.empty else {}
        # Get column names from datasets or use defaults

    def users_last_interactions(self) -> dict:
        """Generate candidates for each user based on their last interactions."""
        if self.user_items.empty:
            return {}
        
        # Apply query function to get similar items
        self.user_items['similar_item_id'] = self.user_items.apply(
            lambda row: self.query(row[self.user_id], row[self.item_id], k=1), axis=1
        )

        # Group by user and get their candidates
        last_interactions = self.user_items.groupby(self.user_id)[[self.item_id, 'similar_item_id']].agg(list).reset_index()
        last_interactions = last_interactions.rename(columns={self.item_id: 'last_interactions'})
        print(last_interactions.head())
        # Combine interactions with similar items
        last_interactions['candidates'] = last_interactions.apply(
            lambda row: list(set(row['last_interactions'] + row['similar_item_id'][0])), 
            axis=1
        )
        return last_interactions.set_index(self.user_id).to_dict(orient='index')

    def retrieve_candidates(self, user_id: str, top_k: int = 10) -> list:
        """Retrieve candidate items for a given user_id as list of dicts."""
        if user_id not in self.candidates or not self.candidates[user_id].get('candidates'):
            return []
        
        candidates = self.candidates[user_id]['candidates'][:top_k]
        # Return list of dicts with item_id
        return [{"item_id": item_id} for item_id in candidates]

    def retrieve_context(self, query_text: str, k: int = 3) -> str:
        """Retrieve context documents based on query text."""
        try:
            # Encode the query
            query_vector = self.embedder.text_to_vector([query_text])
            query_vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
            
            # Search in context vector DB
            distances, indices = self.context_vector_db.search_vectors(query_vector, k=k)
            
            # Get the actual context documents
            indices = indices.flatten()
            if len(indices) > 0 and hasattr(self.context_db, 'context') and not self.context_db.context.empty:
                context_docs = self.context_db.context.iloc[indices[:k]]
                # Return concatenated context as string
                return " ".join([str(doc) for doc in context_docs.values.tolist()])
            return ""
        except Exception as e:
            print(f"Error in retrieve_context: {e}")
            return ""

    def query(self, user_id: str, item_id: str, k: int = 1) -> list:
        """Query similar items for a user-item pair."""
        try:
            # Try to get the generated prompt for this user-item pair
            if self.user_items.empty:
                return []
            
            mask = (self.user_items[self.user_id] == user_id) & (self.user_items[self.item_id] == item_id)
            matching = self.user_items[mask]
            
            if matching.empty or 'generated_prompt' not in matching.columns:
                return []
            
            query_prompt = matching['generated_prompt'].values[0]

            # 1. Embed the query prompt into a vector
            query_vector = self.embedder.text_to_vector([query_prompt])
            query_vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)

            if not hasattr(self.context_db, 'context') or self.context_db.context.empty:
                return []
            total_vectors = len(self.context_db.context)

            # 2. Search the context vector database for nearest vectors.
            # The query prompt is itself indexed, and rows for the same item_id
            # (from other users) tend to embed close together too, so a small
            # buffer often isn't enough to find a differing item; expand the
            # search until we find k distinct other items or exhaust the index.
            search_k = min(k + 1, total_vectors)
            retrieved_items: list = []
            while True:
                distances, indices = self.context_vector_db.search_vectors(query_vector, k=search_k)
                indices = [i for i in indices.flatten() if 0 <= i < total_vectors]
                if indices:
                    candidates = self.context_db.context.iloc[indices]
                    retrieved_items = candidates.loc[
                        candidates[self.item_id] != item_id, self.item_id
                    ].drop_duplicates().values.tolist()[:k]
                if retrieved_items or search_k >= total_vectors:
                    break
                search_k = min(search_k * 2, total_vectors)

            return retrieved_items
        except Exception as e:
            print(f"Error in query: {e}")
            return []
