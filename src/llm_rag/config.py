"""Configuration settings for LLM RAG application."""
import os


class Config:
    """Application configuration."""

    # MongoDB settings
    MONGODB_URI: str = os.getenv(
        "MONGODB_URI", "mongodb://user:pass@127.0.0.1:27017/?directConnection=true"
    )
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "llm_rag")
    MONGODB_COLLECTION_NAME: str = os.getenv(
        "MONGODB_COLLECTION_NAME", "documents"
    )

    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )

    # Retrieval settings
    SIMILARITY_THRESHOLD: float = float(
        os.getenv("SIMILARITY_THRESHOLD", "0.7")
    )
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    # Application settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


config = Config()