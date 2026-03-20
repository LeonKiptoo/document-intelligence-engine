"""
Document Intelligence Retrieval Engine v2
A lightweight but high-quality retrieval layer for document intelligence systems.
"""

__version__ = "1.0.0"
__author__ = "Document Intelligence Team"

from .config import *
from .document_loader import DocumentLoader
from .chunking_engine import ChunkingEngine
from .embedding_engine import EmbeddingEngine
from .vector_store_manager import VectorStoreManager
from .retriever import Retriever
from .context_builder import ContextBuilder

__all__ = [
    "DocumentLoader",
    "ChunkingEngine",
    "EmbeddingEngine",
    "VectorStoreManager",
    "Retriever",
    "ContextBuilder",
]
