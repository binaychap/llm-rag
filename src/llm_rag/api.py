"""FastAPI application for RAG system."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from .pipeline import RAGPipeline
from .config import config

app = FastAPI(title="LLM RAG API", version="0.1.0")
pipeline = RAGPipeline(embedding_model_type="openai")


class DocumentInput(BaseModel):
    """Input model for document ingestion."""
    texts: List[str]
    metadatas: Optional[List[dict]] = None


class QueryInput(BaseModel):
    """Input model for queries."""
    question: str
    top_k: int = 5
    threshold: float = 0.7


class QueryResponse(BaseModel):
    """Response model for queries."""
    question: str
    answer: str
    context_documents: list
    num_retrieved: int


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/ingest")
async def ingest_documents(input_data: DocumentInput):
    """Ingest documents into the RAG system."""
    try:
        doc_ids = pipeline.ingest_documents(
            texts=input_data.texts,
            metadatas=input_data.metadatas
        )
        return {
            "status": "success",
            "document_ids": doc_ids,
            "count": len(doc_ids)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(input_data: QueryInput):
    """Query the RAG system."""
    try:
        result = pipeline.query(
            question=input_data.question,
            top_k=input_data.top_k,
            threshold=input_data.threshold
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get RAG system statistics."""
    try:
        vector_store = pipeline.retriever.vector_store
        collection_size = vector_store.get_collection_size()
        return {
            "total_documents": collection_size,
            "embedding_model": "openai",
            "vector_dimension": 1536  # For text-embedding-3-small
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...), chunk_size: int = 1000):
    """Ingest documents from a file with automatic chunking."""
    try:
        # Read file content
        content = await file.read()
        
        # If PDF, extract text
        if file.filename.endswith('.pdf'):
            pdf_reader = PdfReader(io.BytesIO(content))
            text = "\n".join([page.extract_text() for page in pdf_reader.pages])
        else:
            text = content.decode('utf-8')
        
        # Split into chunks
        chunks = [
            text[i:i+chunk_size] 
            for i in range(0, len(text), chunk_size)
        ]
        
        # Ingest with metadata
        doc_ids = pipeline.ingest_documents(
            texts=chunks,
            metadatas=[
                {
                    "source": file.filename,
                    "chunk": i,
                    "chunk_size": chunk_size
                } 
                for i in range(len(chunks))
            ]
        )
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks": len(chunks),
            "document_ids": doc_ids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    pipeline.close()