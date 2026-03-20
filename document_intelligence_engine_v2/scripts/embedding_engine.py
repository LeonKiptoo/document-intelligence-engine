"""
Embedding engine for generating semantic embeddings using sentence-transformers.
Handles batch processing and GPU/CPU device selection.
"""

import logging
from typing import List, Union, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Generates embeddings for text chunks using sentence-transformers.
    Supports batch processing and device selection.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 batch_size: int = 32, device: str = "cpu"):
        """
        Initialize the embedding engine.
        
        Args:
            model_name: Name of sentence-transformers model to use
            batch_size: Batch size for embedding generation
            device: Device to use ("cpu" or "cuda")
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        
        # Load model
        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: Union[str, List[str]], 
                   normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            text: Single string or list of strings to embed
            normalize: Whether to normalize embeddings (L2 normalization)
            
        Returns:
            numpy array of shape (1, embedding_dim) or (n, embedding_dim)
        """
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        
        if single_input:
            return embeddings[0]
        else:
            return embeddings
    
    def embed_chunks(self, chunks: List[dict], 
                     normalize: bool = True) -> List[dict]:
        """
        Generate embeddings for multiple chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            normalize: Whether to normalize embeddings
            
        Returns:
            List of chunks with added 'embedding' field
        """
        if not chunks:
            logger.warning("No chunks to embed")
            return chunks
        
        # Extract texts
        texts = [chunk.get("text", "") for chunk in chunks]
        
        # Generate embeddings
        logger.info(f"Embedding {len(texts)} chunks")
        embeddings = self.embed_text(texts, normalize=normalize)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]
        
        return chunks
    
    def get_query_embedding(self, query: str, 
                           normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a query.
        
        Args:
            query: Query string
            normalize: Whether to normalize embedding
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self.embed_text(query, normalize=normalize)
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "batch_size": self.batch_size,
        }
