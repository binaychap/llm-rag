"""RAG Pipeline implementation."""
from typing import List, Dict, Any, Optional
from .retrieval import Retriever
from .generation import ResponseGenerator


class RAGPipeline:
    """Complete RAG (Retrieval-Augmented Generation) pipeline."""

    def __init__(self, embedding_model_type: str = "openai"):
        """Initialize RAG pipeline."""
        self.retriever = Retriever(embedding_model_type=embedding_model_type)
        self.generator = ResponseGenerator()

    def query(
        self,
        question: str,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Process a query through the RAG pipeline."""
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            threshold=threshold
        )

        # Generate response
        response = self.generator.generate(
            query=question,
            context_docs=retrieved_docs
        )

        return {
            "question": question,
            "answer": response,
            "context_documents": retrieved_docs,
            "num_retrieved": len(retrieved_docs)
        }

    def ingest_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Ingest documents into the RAG pipeline."""
        doc_ids = self.retriever.add_documents(
            texts=texts,
            metadatas=metadatas
        )
        return doc_ids

    def close(self) -> None:
        """Close the pipeline."""
        self.retriever.close()