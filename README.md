# LLM RAG (Retrieval-Augmented Generation) System

A production-ready Python application that implements a complete Retrieval-Augmented Generation (RAG) system using FastAPI, MongoDB, OpenAI, and Sentence Transformers. This system allows you to ingest documents, generate embeddings, retrieve relevant documents, and generate AI-powered responses based on your knowledge base.

## Features

- **Document Ingestion**: Upload and store documents with embeddings
- **Vector Search**: MongoDB-based vector similarity search
- **Multiple Embedding Models**: Support for OpenAI and Sentence Transformers
- **LLM Integration**: OpenAI GPT integration for response generation
- **REST API**: FastAPI-based endpoints for easy integration
- **Conversation History**: Support for multi-turn conversations
- **Metadata Support**: Track document sources and metadata

## Prerequisites

- Python 3.12 or higher
- MongoDB 4.4+ (running locally or remote)
- OpenAI API Key
- Poetry (for dependency management)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd llm-rag
```

### 2. Install Dependencies

```bash
# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

# Run locally

```sh
poetry run python src/llm_rag/main.py
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=llm_rag
MONGODB_COLLECTION_NAME=documents

# OpenAI Configuration
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Retrieval Settings
SIMILARITY_THRESHOLD=0.7
TOP_K=5

# Application Settings
DEBUG=False
HOST=0.0.0.0
PORT=8000
```

### 4. Start MongoDB (if running locally)

```bash
# Using Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest

# Or using Homebrew (macOS)
brew services start mongodb-community
```

## Running the Application

### Start the Server

```bash
# Using Poetry
poetry run python src/llm_rag/main.py

# Or with Uvicorn directly
poetry run uvicorn src.llm_rag.api:app --reload
```

The server will start at `http://localhost:8000`

### Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### 1. Health Check

- **Endpoint**: `GET /health`
- **Description**: Check if the server is running

**Response**:

```json
{
  "status": "healthy"
}
```

### 2. Ingest Documents

- **Endpoint**: `POST /ingest`
- **Description**: Upload documents to the RAG system

**Request Body**:

```json
{
  "texts": [
    "Python is a high-level programming language",
    "Machine learning is a subset of artificial intelligence"
  ],
  "metadatas": [
    { "source": "documentation", "date": "2026-03-22" },
    { "source": "article", "date": "2026-03-22" }
  ]
}
```

**Response**:

```json
{
  "status": "success",
  "document_ids": ["507f1f77bcf86cd799439011", "507f1f77bcf86cd799439012"],
  "count": 2
}
```

### 3. Query the RAG System

- **Endpoint**: `POST /query`
- **Description**: Ask a question and retrieve answers based on ingested documents

**Request Body**:

```json
{
  "question": "What is Python used for?",
  "top_k": 5,
  "threshold": 0.7
}
```

**Response**:

```json
{
  "question": "What is Python used for?",
  "answer": "Python is a high-level programming language used for a wide range of applications including web development, data analysis, machine learning, and automation...",
  "context_documents": [
    {
      "id": "507f1f77bcf86cd799439011",
      "text": "Python is a high-level programming language",
      "metadata": { "source": "documentation" },
      "score": 0.92
    }
  ],
  "num_retrieved": 1
}
```

### 4. Get System Statistics

- **Endpoint**: `GET /stats`
- **Description**: Get information about the RAG system

**Response**:

```json
{
  "total_documents": 42,
  "embedding_model": "openai",
  "vector_dimension": 1536
}
```

## Usage Examples

### Using cURL

#### Health Check

```bash
curl http://localhost:8000/health
```

#### Ingest Documents

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Python is a programming language",
      "FastAPI is a modern web framework"
    ],
    "metadatas": [
      {"source": "doc1"},
      {"source": "doc2"}
    ]
  }'
```

#### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is FastAPI?",
    "top_k": 5,
    "threshold": 0.7
  }'
```

#### Get Statistics

```bash
curl http://localhost:8000/stats
```

### Using Python Requests

```python
import requests

API_URL = "http://localhost:8000"

# Ingest documents
response = requests.post(
    f"{API_URL}/ingest",
    json={
        "texts": ["Your document here"],
        "metadatas": [{"source": "example"}]
    }
)
print(response.json())

# Query
response = requests.post(
    f"{API_URL}/query",
    json={
        "question": "Your question here",
        "top_k": 5,
        "threshold": 0.7
    }
)
print(response.json())
```

### Using Python SDK

```python
from llm_rag import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline(embedding_model_type="openai")

# Ingest documents
doc_ids = pipeline.ingest_documents(
    texts=[
        "Python is a programming language",
        "Machine learning uses Python"
    ]
)

# Query
result = pipeline.query(
    question="What is Python used for?",
    top_k=5,
    threshold=0.7
)

print(result["answer"])
print(result["context_documents"])
```

## Project Structure

```
llm-rag/
├── src/llm_rag/
│   ├── __init__.py          # Package initialization
│   ├── main.py              # Application entry point
│   ├── api.py               # FastAPI application
│   ├── config.py            # Configuration management
│   ├── vector_store.py      # MongoDB vector store
│   ├── embedding.py         # Embedding model integration
│   ├── retrieval.py         # Document retrieval logic
│   ├── generation.py        # LLM-based generation
│   └── pipeline.py          # Complete RAG pipeline
├── tests/                   # Test files
├── docs/                    # Documentation
├── pyproject.toml           # Project configuration
├── README.md                # This file
└── .env.example             # Environment variables template
```

## Configuration

All configuration is managed through environment variables in the `.env` file:

| Variable                | Default                   | Description                     |
| ----------------------- | ------------------------- | ------------------------------- |
| MONGODB_URI             | mongodb://localhost:27017 | MongoDB connection string       |
| MONGODB_DB_NAME         | llm_rag                   | Database name                   |
| MONGODB_COLLECTION_NAME | documents                 | Collection name                 |
| OPENAI_API_KEY          | (required)                | Your OpenAI API key             |
| OPENAI_MODEL            | gpt-4                     | LLM model to use                |
| OPENAI_EMBEDDING_MODEL  | text-embedding-3-small    | Embedding model                 |
| SIMILARITY_THRESHOLD    | 0.7                       | Minimum similarity score        |
| TOP_K                   | 5                         | Number of documents to retrieve |
| DEBUG                   | False                     | Debug mode                      |
| HOST                    | 0.0.0.0                   | Server host                     |
| PORT                    | 8000                      | Server port                     |

## Troubleshooting

### MongoDB Connection Error

```
pymongo.errors.ServerSelectionTimeoutError
```

**Solution**: Ensure MongoDB is running on the configured URI:

```bash
# Check if MongoDB is running
mongo --version

# Start MongoDB (if using docker)
docker start mongodb
```

### OpenAI API Key Error

```
openai.error.AuthenticationError
```

**Solution**: Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

### Port Already in Use

```
OSError: [Errno 48] Address already in use
```

**Solution**: Change the port or kill the process:

```bash
# Use a different port
export PORT=8001

# Or kill the process
lsof -ti:8000 | xargs kill -9
```

### Import Errors

```
ImportError: No module named 'llm_rag'
```

**Solution**: Reinstall dependencies:

```bash
poetry install
```

## Development

### Run Tests

```bash
poetry run pytest tests/
```

### Code Formatting

```bash
poetry run black src/
```

### Linting

```bash
poetry run flake8 src/
```

## Performance Optimization

For production deployments:

1. **Use MongoDB Atlas Vector Search** for faster similarity search on large datasets
2. **Cache embeddings** to avoid regenerating them
3. **Use batch processing** for large document ingestion
4. **Implement rate limiting** for API endpoints
5. **Use async operations** for concurrent requests

Example with gunicorn:

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.llm_rag.api:app
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues, questions, or suggestions, please open an issue on GitHub or contact the maintainers.
