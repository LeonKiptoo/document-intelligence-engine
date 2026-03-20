"""
Configuration module for the Document Intelligence Retrieval Engine.
Centralized settings for document processing, chunking, embedding, and retrieval.
"""

import os
from pathlib import Path

# ===== Directory Configuration =====
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# ===== Document Loader Configuration =====
SUPPORTED_FORMATS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".csv": "csv",
    ".xlsx": "xlsx",
}

# ===== Chunking Configuration =====
# Token count (approximate: 1 token ≈ 4 characters for English)
MIN_CHUNK_SIZE = 100  # minimum tokens per chunk
MAX_CHUNK_SIZE = 200  # maximum tokens per chunk
CHUNK_OVERLAP = 0.10  # 10% overlap between chunks
CHAR_PER_TOKEN = 4  # approximate characters per token for estimation

# Derived chunk sizes in characters
MIN_CHUNK_CHARS = int(MIN_CHUNK_SIZE * CHAR_PER_TOKEN)
MAX_CHUNK_CHARS = int(MAX_CHUNK_SIZE * CHAR_PER_TOKEN)
OVERLAP_CHARS = int(MAX_CHUNK_CHARS * CHUNK_OVERLAP)

# ===== Embedding Configuration =====
# Using sentence-transformers for semantic embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, lightweight (22MB), good quality
EMBEDDING_DIMENSION = 384  # Output dimension of all-MiniLM-L6-v2
BATCH_SIZE = 32  # Batch size for embedding generation
DEVICE = "cpu"  # "cpu" or "cuda" - will auto-select if GPU available

# ===== Vector Store Configuration =====
FAISS_INDEX_TYPE = "cosine"  # "cosine" or "l2" for distance metric
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"

# ===== Retrieval Configuration =====
TOP_K = 5  # Number of top results to retrieve
SIMILARITY_THRESHOLD = 0.1  # Minimum similarity score
KEYWORD_BOOST = 0.1  # Weight for keyword overlap score
REDUNDANCY_THRESHOLD = 0.9  # Similarity threshold to consider chunks redundant

# ===== Context Builder Configuration =====
MAX_CONTEXT_TOKENS = 8000  # Maximum tokens in final context window
MAX_CONTEXT_CHARS = int(MAX_CONTEXT_TOKENS * CHAR_PER_TOKEN)

# ===== Logging Configuration =====
LOG_LEVEL = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

# ===== CLI Configuration =====
DEFAULT_QUERY = "What is in this document?"  # Default query if not provided
