"""Embedding model integration."""
from typing import List
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from .config import config


class EmbeddingModel:
    """Wrapper for embedding models."""

    def __init__(self, model_type: str = "openai"):
        """Initialize embedding model."""
        self.model_type = model_type

        if model_type == "openai":
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.model_name = config.OPENAI_EMBEDDING_MODEL
        elif model_type == "sentence_transformers":
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if self.model_type == "openai":
            response = self.client.embeddings.create(
                input=text,
                model=self.model_name
            )
            return response.data[0].embedding
        else:
            return self.model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        if self.model_type == "openai":
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            # Sort by index to maintain order
            embeddings = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in embeddings]
        else:
            return self.model.encode(texts).tolist()