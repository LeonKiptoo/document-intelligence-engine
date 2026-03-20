"""
DocIntel API Layer
FastAPI web service that exposes the document intelligence engine via HTTP endpoints.
Run with: uvicorn api:app --reload --port 8000
"""

import os
import sys
import logging
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

import config
from document_loader import DocumentLoader
from chunking_engine import ChunkingEngine
from embedding_engine import EmbeddingEngine
from vector_store_manager import VectorStoreManager
from retriever import Retriever
from context_builder import ContextBuilder
from generation_engine import GenerationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocIntel API",
    description="Document Intelligence Engine — ingest any document, query it with natural language.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    query: str
    answer: str
    chunks_retrieved: int
    chunks_in_context: int
    source: Optional[str] = None

class IngestResponse(BaseModel):
    message: str
    documents_loaded: int
    chunks_created: int

class StatusResponse(BaseModel):
    status: str
    vectors_in_store: int
    data_directory: str
    vector_store_directory: str

# ---------------------------------------------------------------------------
# Engine initialisation
# ---------------------------------------------------------------------------

_engines = {}

def get_engines():
    if len(_engines) == 7:
        return _engines

    _engines.clear()

    logger.info("Initialising DocIntel engines...")

    loader   = DocumentLoader()
    chunker  = ChunkingEngine()
    embedder = EmbeddingEngine()

    embedding_dim = embedder.model.get_sentence_embedding_dimension()
    logger.info(f"Embedding dimension: {embedding_dim}")

    vector_store_dir = Path(config.VECTOR_STORE_DIR)
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    vsm = VectorStoreManager(
        embedding_dim=embedding_dim,
        index_type="cosine",
        index_path=vector_store_dir / "index.faiss",
        metadata_path=vector_store_dir / "metadata.json",
    )

    retriever = Retriever(vsm, embedder)
    builder   = ContextBuilder()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in .env file")
    generator = GenerationEngine(api_key=api_key)

    _engines["loader"]    = loader
    _engines["chunker"]   = chunker
    _engines["embedder"]  = embedder
    _engines["vsm"]       = vsm
    _engines["retriever"] = retriever
    _engines["builder"]   = builder
    _engines["generator"] = generator

    index_file = vector_store_dir / "index.faiss"
    if index_file.exists():
        vsm.load()
        logger.info(f"Loaded existing vector store: {vsm.index.ntotal} vectors")

    logger.info("All engines ready.")
    return _engines


@app.on_event("startup")
async def startup_event():
    logger.info("DocIntel API ready.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_model=StatusResponse)
def root():
    """Health check."""
    try:
        engines = get_engines()
        vector_count = engines["vsm"].index.ntotal if engines["vsm"].index is not None else 0
    except Exception:
        vector_count = 0
    return StatusResponse(
        status="running",
        vectors_in_store=vector_count,
        data_directory=str(config.DATA_DIR),
        vector_store_directory=str(config.VECTOR_STORE_DIR),
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """Ingest all documents currently in the data directory."""
    engines = get_engines()
    data_dir = Path(config.DATA_DIR)

    if not data_dir.exists() or not any(data_dir.iterdir()):
        raise HTTPException(status_code=400, detail="Data directory is empty.")

    documents = engines["loader"].load_directory(str(data_dir))
    if not documents:
        raise HTTPException(status_code=400, detail="No documents could be loaded.")

    all_chunks = []
    for doc in documents:
        chunks = engines["chunker"].chunk_document(doc)
        all_chunks.extend(chunks)

    embeddings = engines["embedder"].embed_chunks(all_chunks)

    engines["vsm"].reset()
    engines["vsm"].add_embeddings(embeddings, all_chunks)
    engines["vsm"].save()

    logger.info(f"Ingested {len(documents)} documents, {len(all_chunks)} chunks")

    return IngestResponse(
        message="Ingest complete.",
        documents_loaded=len(documents),
        chunks_created=len(all_chunks),
    )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the data directory."""
    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    dest = data_dir / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "message": f"Uploaded '{file.filename}'. Call /ingest to process it.",
        "filename": file.filename,
        "size_bytes": dest.stat().st_size,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Query the document store with natural language."""
    engines = get_engines()
    vsm = engines["vsm"]

    if vsm.index is None or vsm.index.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Upload documents and call /ingest first."
        )

    chunks = engines["retriever"].retrieve(request.query, top_k=request.top_k)
    if not chunks:
        return QueryResponse(
            query=request.query,
            answer="No relevant content found in the documents.",
            chunks_retrieved=0,
            chunks_in_context=0,
        )

    build_result = engines["builder"].build_context(chunks)
    context_chunks = build_result["chunks"]
    context_tokens = build_result["statistics"] ["context_tokens"]
    context_str = build_result["context"]

    if not context_chunks:
        return QueryResponse(
            query=request.query,
            answer="Found relevant chunks but they exceeded context limits.",
            chunks_retrieved=len(chunks),
            chunks_in_context=0,
        )

    answer = engines["generator"].generate_answer(request.query, context_str)

    top_chunk = context_chunks[0]
    source_info = top_chunk.get("doc_id", "")
    meta = top_chunk.get("metadata", {})
    section = meta.get("section")
    page = meta.get("page")
    if section or page:
        source_info += f" | Section: {section} | Page: {page}"

    return QueryResponse(
        query=request.query,
        answer=answer,
        chunks_retrieved=len(chunks),
        chunks_in_context=len(context_chunks),
        source=source_info,
    )


@app.delete("/reset")
def reset():
    """Clear the entire vector store."""
    engines = get_engines()
    engines["vsm"].reset()
    store_dir = Path(config.VECTOR_STORE_DIR)
    for f in store_dir.glob("*"):
        f.unlink()
    return {"message": "Vector store cleared. Re-ingest documents to use the system."}


@app.get("/documents")
def list_documents():
    """List all documents currently in the data directory."""
    data_dir = Path(config.DATA_DIR)
    if not data_dir.exists():
        return {"documents": [], "count": 0}

    files = []
    for f in data_dir.iterdir():
        if f.is_file() and not f.name.startswith('.'):
            files.append({
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "extension": f.suffix.lower(),
            })

    return {"documents": files, "count": len(files)}