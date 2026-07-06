from core2.configs import Configs
from core2.dbs import Con, ContextDB, ContextVectorDB    


class Retrieval(Configs):
    """Class for retrieval operations."""  
    def __init__(self, engine_name: str, context_prompts_db: ContextDB, context_vectors_db: ContextVectorDB):
        super().__init__(project_name=engine_name)
        self.context_prompts_db = context_prompts_db
        self.context_vectors_db = context_vectors_db
        self.embedder = context_vectors_db.embedder

    def query(self, query_texts: list[str], k: int = 10) -> tuple:
        """Query the context database and retrieve relevant documents."""
        # 1. Embed the query texts into vectors
        query_vectors = self.embedder.text_to_vector(query_texts)
        # 2. Search the context vector database for nearest vectors
        distances, indices = self.context_vectors_db.search_vectors(query_vectors, k=k)
        # 3. Retrieve the corresponding documents from the context prompts database
        indices = indices.flatten()
        retrieved_docs = self.context_prompts_db.read(indices.tolist())
        return indices, retrieved_docs
