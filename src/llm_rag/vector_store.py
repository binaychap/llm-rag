"""MongoDB Vector Store implementation."""
from typing import List, Dict, Any, Optional
import numpy as np
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from .config import config


class VectorStore:
    """MongoDB vector store for managing embeddings and documents."""

    def __init__(self):
        """Initialize MongoDB connection."""
        self.client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=5000)
        self.db = self.client[config.MONGODB_DB_NAME]
        self.collection = self.db[config.MONGODB_COLLECTION_NAME]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Create necessary indexes for vector search."""
        try:
            self.collection.create_index("embedding")
            self.collection.create_index("metadata")
        except Exception as e:
            print(f"Index creation note: {e}")

    def add_document(
        self, 
        text: str, 
        embedding: List[float], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a document with embedding to the vector store."""
        doc = {
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
        }
        result = self.collection.insert_one(doc)
        return str(result.inserted_id)

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add multiple documents to the vector store."""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        docs = [
            {
                "text": text,
                "embedding": embedding,
                "metadata": metadata,
            }
            for text, embedding, metadata in zip(texts, embeddings, metadatas)
        ]
        result = self.collection.insert_many(docs)
        return [str(id) for id in result.inserted_ids]

    def search(
        self,
        embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity."""
        embedding_array = np.array(embedding)
        
        # Retrieve all documents (for small collections)
        # For production, use MongoDB Atlas Vector Search
        docs = list(self.collection.find())
        
        results = []
        for doc in docs:
            stored_embedding = np.array(doc["embedding"])
            similarity = self._cosine_similarity(embedding_array, stored_embedding)
            
            if similarity >= threshold:
                results.append({
                    "id": str(doc["_id"]),
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {}),
                    "score": float(similarity)
                })
        
        # Sort by similarity score and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector store."""
        from bson import ObjectId
        result = self.collection.delete_one({"_id": ObjectId(doc_id)})
        return result.deleted_count > 0

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        self.collection.delete_many({})

    def get_collection_size(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count_documents({})

    def close(self) -> None:
        """Close the MongoDB connection."""
        self.client.close()