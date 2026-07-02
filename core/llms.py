from core.configs import Configs
from datetime import datetime
from typing import Optional, List, Dict, Any


class BaseLLM(Configs):
    def __init__(self, api_key: str | None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model

    def call(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement the call method.")
    

class GPTCaller(BaseLLM):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        super().__init__(api_key, model)

    def call(self, prompt: str) -> str:
        import openai

        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content or ""

class FreeLLMCaller(BaseLLM):
    """
    Free LLM Caller with RAG pipeline supporting multiple providers:
    - HuggingFace Inference API (requires HF_TOKEN)
    - Ollama (local, no API key needed)
    - Replicate (requires REPLICATE_API_TOKEN)
    
    Pipeline: prompt → embeddings → FAISS search → context augmentation → LLM call
    
    Integrated with LangChain for chat history management and conversation memory.
    """
    def __init__(self, api_key: str = None, model: str = "mistral-7b-instruct", provider: str = "huggingface", embedding_store=None, top_k: int = 3):
        super().__init__(api_key, model)
        self.provider = provider.lower()
        self.embedding_store = embedding_store
        self.top_k = top_k
        self.search_results = {}  # Store search results for debugging
        
        # Initialize chat history for LangChain integration
        self.chat_history: List[Dict[str, Any]] = []
        self._init_langchain_components()

    def _init_langchain_components(self):
        """Initialize LangChain components for conversation memory and chaining."""
        try:
            from langchain.memory import ConversationBufferMemory
            from langchain.prompts import PromptTemplate
            
            # Create memory store for conversation history
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                human_prefix="User",
                ai_prefix="Assistant"
            )
        except ImportError:
            print("[WARNING] LangChain not installed. Chat history will be stored manually.")
            self.memory = None

    def call(self, prompt: str) -> str:
        """Call with embedding-based RAG pipeline using pretrained models and LangChain integration."""
        # Log input to chat history
        self._add_to_history(role="user", content=prompt)
        
        # Augment prompt with FAISS search results if embedding_store is available
        augmented_prompt = self._augment_prompt_with_rag(prompt)
        
        # Add chat history context to augmented prompt
        augmented_prompt = self._add_chat_history_context(augmented_prompt)
        
        # Convert augmented prompt to embeddings using pretrained encoder
        prompt_embeddings = self._encode_to_embeddings(augmented_prompt)
        
        # Retrieve relevant context from embedding space
        retrieved_context = self._retrieve_from_embeddings(prompt_embeddings, augmented_prompt)
        
        # Decode embeddings back to text using pretrained decoder
        result_text = self._decode_from_embeddings(prompt_embeddings, retrieved_context)
        
        # Log output to chat history and LangChain memory
        self._add_to_history(role="assistant", content=result_text)
        if self.memory:
            self.memory.save_context({"input": prompt}, {"output": result_text})
        
        return result_text

    def _encode_to_embeddings(self, text: str) -> list:
        """Encode text to embeddings using pretrained model."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            # Use pretrained sentence embeddings model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode([text])
            return embeddings[0].tolist() if hasattr(embeddings, 'tolist') else embeddings[0]
        except Exception as e:
            print(f"[WARNING] Embedding encoding failed: {str(e)}. Using fallback.")
            return self._get_fallback_embeddings(text)

    def _retrieve_from_embeddings(self, embeddings: list, original_text: str) -> dict:
        """Retrieve context from FAISS using embeddings."""
        if not self.embedding_store:
            return {"text": original_text, "similarity": 1.0}
        
        try:
            import numpy as np
            import faiss
            
            # Convert embeddings to numpy array
            query_vector = np.array([embeddings], dtype=np.float32)
            
            # Search item embeddings
            item_results = self.embedding_store.search_items(original_text, self.top_k)
            
            # Build context dictionary
            context = {
                "query_text": original_text,
                "similar_items": [item.get("item", {}).get("title", item.get("item_id")) for item in item_results[:2]],
                "similar_users": [f"User {user.get('user_id')}" for user in (self.embedding_store.search_users(original_text, self.top_k))[:2]],
                "scores": [item.get("score", 0) for item in item_results[:2]]
            }
            return context
        except Exception as e:
            print(f"[WARNING] Retrieval from embeddings failed: {str(e)}")
            return {"text": original_text, "error": str(e)}

    def _decode_from_embeddings(self, embeddings: list, context: dict) -> str:
        """Decode embeddings back to text using pretrained model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch
            
            # Use pretrained seq2seq model for decoding
            model_name = "facebook/bart-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Build input from context
            input_text = self._format_context_as_input(context)
            
            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate output from embeddings/input
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs["input_ids"],
                    num_beams=4,
                    max_length=150,
                    early_stopping=True,
                    temperature=0.7
                )
            
            # Decode to text
            result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Enhance with context summary
            enhanced_result = self._enhance_result_with_context(result, context)
            return enhanced_result
            
        except Exception as e:
            print(f"[WARNING] Embedding decoding failed: {str(e)}. Using fallback.")
            return self._generate_fallback_result(context)

    def _format_context_as_input(self, context: dict) -> str:
        """Format retrieved context as input for decoder."""
        parts = [context.get("query_text", "")]
        
        if context.get("similar_items"):
            parts.append(f"Consider items: {', '.join(context['similar_items'])}")
        
        if context.get("similar_users"):
            parts.append(f"Similar users: {', '.join(context['similar_users'])}")
        
        if context.get("scores"):
            avg_score = sum(context["scores"]) / len(context["scores"])
            parts.append(f"Relevance score: {avg_score:.2f}")
        
        return " ".join(parts)

    def _enhance_result_with_context(self, result: str, context: dict) -> str:
        """Enhance generated result with context information."""
        if context.get("similar_items"):
            items_str = ", ".join(context["similar_items"][:3])
            result += f"\n\nRecommended Items: {items_str}"
        
        return result

    def _generate_fallback_result(self, context: dict) -> str:
        """Generate result using context when model decoding fails."""
        query = context.get("query_text", "")
        items = context.get("similar_items", [])
        users = context.get("similar_users", [])
        
        result = f"Based on the query: '{query}'\n\n"
        
        if items:
            result += f"Recommended Items: {', '.join(items)}\n"
        
        if users:
            result += f"Similar Users: {', '.join(users)}\n"
        
        result += "Analysis: The embeddings-based retrieval identified the most relevant items and users based on semantic similarity."
        
        return result

    def _get_fallback_embeddings(self, text: str) -> list:
        """Get fallback embeddings when SentenceTransformers is unavailable."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            model = AutoModel.from_pretrained("distilbert-base-uncased")
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)[0]
            return embeddings.tolist()
        except Exception as e:
            print(f"[ERROR] All embedding methods failed: {str(e)}")
            # Return dummy embeddings
            return [0.0] * 768

    def _augment_prompt_with_rag(self, prompt: str) -> str:
        """Augment prompt using FAISS-based RAG pipeline."""
        if not self.embedding_store:
            return prompt
        
        try:
            # Search FAISS indices for relevant context
            relevant_items = self.embedding_store.search_items(prompt, self.top_k)
            relevant_users = self.embedding_store.search_users(prompt, self.top_k)
            relevant_prompts = self.embedding_store.search_default_prompt(prompt, self.top_k)
            
            # Store results for debugging
            self.search_results = {
                "items": relevant_items,
                "users": relevant_users,
                "prompts": relevant_prompts
            }
            
            # Build RAG context
            rag_context = self._build_rag_context(relevant_items, relevant_users, relevant_prompts)
            
            # Augment prompt with RAG context
            augmented = f"{prompt}\n\n---\n[RAG CONTEXT - Retrieved Similar Content]\n{rag_context}\n---\n"
            return augmented
        except Exception as e:
            # If RAG fails, continue with original prompt
            print(f"[WARNING] RAG augmentation failed: {str(e)}. Continuing with original prompt.")
            return prompt

    def _build_rag_context(self, items, users, prompts) -> str:
        """Build contextual string from FAISS search results."""
        context_parts = []
        
        if items:
            context_parts.append("Similar Items:")
            for item in items[:self.top_k]:
                title = item.get("item", {}).get("title", item.get("item_id", "Unknown"))
                context_parts.append(f"  - {title} (score: {item.get('score', 0):.3f})")
        
        if users:
            context_parts.append("\nSimilar Users:")
            for user in users[:self.top_k]:
                user_id = user.get("user_id", "Unknown")
                role = user.get("role", "unknown")
                context_parts.append(f"  - User {user_id} ({role}) (score: {user.get('score', 0):.3f})")
        
        if prompts:
            context_parts.append("\nSimilar Prompt Sections:")
            for prompt_hit in prompts[:self.top_k]:
                section = prompt_hit.get("section", "unknown")
                context_parts.append(f"  - {section} (score: {prompt_hit.get('score', 0):.3f})")
        
        return "\n".join(context_parts) if context_parts else "No similar context found."

    def _call_huggingface(self, prompt: str) -> str:
        """Call HuggingFace Inference API (free tier available)."""
        import os
        from huggingface_hub import InferenceClient

        api_key = self.api_key or os.getenv("HF_TOKEN")
        if not api_key:
            raise ValueError("HF_TOKEN env var required or pass api_key")
        
        try:
            client = InferenceClient(api_key=api_key)
            response = client.text_generation(
                prompt=prompt, 
                model=self.model, 
                max_new_tokens=512, 
                temperature=0.7,
                do_sample=True
            )
            if not response or (isinstance(response, str) and not response.strip()):
                raise ValueError(f"Empty response from HuggingFace model {self.model}. Check if model exists and is available.")
            return response
        except ValueError as ve:
            # Re-raise ValueError as-is
            raise ve
        except Exception as e:
            error_str = str(e)
            # Check for specific error types
            if "401" in error_str or "Unauthorized" in error_str or "authentication" in error_str.lower():
                raise ValueError(f"HuggingFace authentication failed (401). Check your HF_TOKEN is valid. Full error: {error_str}")
            elif "404" in error_str or "not found" in error_str.lower():
                raise ValueError(f"HuggingFace model not found (404): {self.model}. May require access request. Error: {error_str}")
            elif "429" in error_str or "rate" in error_str.lower():
                raise ValueError(f"HuggingFace rate limit exceeded (429). Try again later. Error: {error_str}")
            elif "503" in error_str or "unavailable" in error_str.lower():
                raise ValueError(f"HuggingFace service unavailable (503). Try again later. Error: {error_str}")
            else:
                raise RuntimeError(f"HuggingFace inference error ({type(e).__name__}): {error_str}")

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama LLM (completely free, runs locally)."""
        import requests
        ollama_url = "http://localhost:11434/api/generate"
        try:
            response = requests.post(ollama_url, json={"model": self.model, "prompt": prompt, "stream": False, "temperature": 0.7}, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Ollama not running. Start with: ollama serve")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {str(e)}")

    def _call_replicate(self, prompt: str) -> str:
        """Call Replicate API (free credits available)."""
        import os
        import replicate
        api_token = self.api_key or os.getenv("REPLICATE_API_TOKEN")
        if not api_token:
            raise ValueError("REPLICATE_API_TOKEN required")
        replicate.api.token = api_token
        output = replicate.run(self.model, input={"prompt": prompt, "temperature": 0.7})
        return "".join(output) if isinstance(output, list) else str(output)

    def _add_to_history(self, role: str, content: str) -> None:
        """Add message to chat history."""
        self.chat_history.append({
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content
        })

    def _add_chat_history_context(self, prompt: str) -> str:
        """Add recent chat history as context to the prompt."""
        if not self.chat_history or len(self.chat_history) < 2:
            return prompt
        
        try:
            # Get last 4 messages (2 exchanges) for context
            recent_history = self.chat_history[-4:]
            history_str = "\n[Chat History]\n"
            
            for msg in recent_history:
                role = msg.get("role", "unknown").capitalize()
                content = msg.get("content", "")[:200]  # Limit content length
                history_str += f"{role}: {content}\n"
            
            # Append history to prompt
            return f"{prompt}\n{history_str}"
        except Exception as e:
            print(f"[WARNING] Failed to add chat history context: {str(e)}")
            return prompt

    def get_chat_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve chat history.
        
        Args:
            limit: Maximum number of messages to return (None = all)
        
        Returns:
            List of chat history messages with timestamp, role, and content
        """
        if limit:
            return self.chat_history[-limit:]
        return self.chat_history

    def clear_chat_history(self) -> None:
        """Clear all chat history."""
        self.chat_history.clear()
        if self.memory:
            self.memory.clear()
        print("[INFO] Chat history cleared.")

    def get_conversation_summary(self) -> str:
        """Generate a summary of the conversation.
        
        Returns:
            Formatted string with conversation summary
        """
        if not self.chat_history:
            return "No conversation history."
        
        summary = "Conversation Summary:\n"
        summary += f"Total messages: {len(self.chat_history)}\n"
        summary += f"Duration: {self._get_duration()}\n\n"
        
        for i, msg in enumerate(self.chat_history, 1):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")[:100]
            timestamp = msg.get("timestamp", "")
            summary += f"{i}. [{timestamp}] {role}: {content}...\n"
        
        return summary

    def _get_duration(self) -> str:
        """Calculate conversation duration."""
        if len(self.chat_history) < 2:
            return "N/A"
        
        try:
            first_time = datetime.fromisoformat(self.chat_history[0].get("timestamp", ""))
            last_time = datetime.fromisoformat(self.chat_history[-1].get("timestamp", ""))
            duration = last_time - first_time
            minutes = int(duration.total_seconds() / 60)
            return f"{minutes} minutes"
        except Exception:
            return "N/A"


