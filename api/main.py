# api/main.py
"""
Main FastAPI application for the RAG search engine.
This module defines the API endpoints for ingestion, querying, and status checks.
"""
import sys
from pathlib import Path
import logging

# Add project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

from index.vector_store import ChromaVectorStore
from ingest.document_parser import extract_text_from_pdf
from ingest.chunker import chunk_text
from api.llm_integrator import GeminiIntegrator # <-- NEW IMPORT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI(
    title="RAG Knowledge Base Search Engine API",
    description="An API to ingest documents and answer questions using RAG.",
    version="1.0.0"
)

# --- Global Initializations ---
try:
    db_directory = project_root / "index" / "chroma_db"
    vector_store = ChromaVectorStore(db_path=str(db_directory))
    llm_integrator = GeminiIntegrator() # <-- INITIALIZE LLM
    logging.info("Vector store and LLM initialized successfully.")
except Exception as e:
    logging.error(f"FATAL: Could not initialize components. {e}")
    raise

# --- Pydantic Models for Request/Response Bodies ---
class IngestRequest(BaseModel):
    file_path: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel): # <-- UPDATED RESPONSE MODEL
    llm_answer: str
    retrieved_chunks: List[Dict]

# --- API Endpoints ---
@app.get("/status", summary="Check API and vector store status")
def get_status():
    """Returns the health status of the API and vector store."""
    try:
        count = vector_store.collection.count()
        return {"status": "ok", "indexed_chunks": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector store connection failed: {e}")

@app.get("/list-docs", summary="List all ingested document sources")
def list_documents():
    """Returns a list of unique source document paths from the metadata."""
    try:
        all_entries = vector_store.collection.get()
        if not all_entries or not all_entries['metadatas']:
            return {"documents": []}

        unique_sources = sorted(list(set(meta['source'] for meta in all_entries['metadatas'])))
        return {"documents": unique_sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve documents: {e}")

@app.post("/ingest", summary="Ingest a new document")
def ingest_document(request: IngestRequest):
    """Processes and indexes a document from a given file path."""
    file_path = Path(request.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        logging.info(f"Starting ingestion for {file_path}...")
        doc_generator = extract_text_from_pdf(file_path)
        chunks = chunk_text(doc_generator)
        vector_store.add_documents(chunks)
        logging.info(f"Successfully ingested {file_path}.")
        return {"status": "success", "file_path": str(file_path), "chunks_added": len(chunks)}
    except Exception as e:
        logging.error(f"Ingestion failed for {file_path}. Error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

@app.post("/query", response_model=QueryResponse, summary="Query the knowledge base with RAG")
def query_index(request: QueryRequest): # <-- UPDATED ENDPOINT LOGIC
    """Searches the vector store, feeds context to an LLM, and returns a synthesized answer."""
    try:
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = vector_store.query(request.query, k=request.top_k)

        # Step 2: Generate an answer using the LLM
        llm_answer = llm_integrator.generate_answer(request.query, retrieved_chunks)

        return QueryResponse(
            llm_answer=llm_answer,
            retrieved_chunks=retrieved_chunks
        )
    except Exception as e:
        logging.error(f"Query failed for '{request.query}'. Error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")