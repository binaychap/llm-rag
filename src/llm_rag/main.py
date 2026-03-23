"""Main entry point for the RAG application."""
import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uvicorn
from llm_rag.api import app
from llm_rag.config import config


def run_server():
    """Run the FastAPI server."""
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG
    )


if __name__ == "__main__":
    run_server()