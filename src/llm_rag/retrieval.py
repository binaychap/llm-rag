"""Document retrieval logic."""
from typing import List, Dict, Any
from .vector_store import VectorStore
from .embedding import EmbeddingModel
from .config import config


class Retriever:
    """Retriever for fetching relevant documents."""

    def __init__(self, embedding_model_type: str = "openai"):
        """Initialize retriever with vector store and embedding model."""
        self.vector_store = VectorStore()
        self.embedding_model = EmbeddingModel(model_type=embedding_model_type)

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        if top_k is None:
            top_k = config.TOP_K
        if threshold is None:
            threshold = config.SIMILARITY_THRESHOLD

        # Generate embedding for query
        query_embedding = self.embedding_model.embed(query)

        # Search for similar documents
        results = self.vector_store.search(
            embedding=query_embedding,
            top_k=top_k,
            threshold=threshold
        )

        return results

    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]] = None
    ) -> List[str]:
        """Add documents to the vector store."""
        # Generate embeddings for all texts
        embeddings = self.embedding_model.embed_batch(texts)

        # Add to vector store
        doc_ids = self.vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return doc_ids

    def close(self) -> None:
        """Close the retriever."""
        self.vector_store.close()