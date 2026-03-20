"""
Vector store manager for storing and retrieving embeddings using FAISS.
Handles index creation, metadata storage, and persistence.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages FAISS vector index for similarity search.
    Stores embeddings and maintains metadata mapping.
    """
    
    def __init__(self, embedding_dim: int, index_type: str = "cosine",
                 index_path: Optional[Path] = None,
                 metadata_path: Optional[Path] = None):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of index ("cosine" or "l2")
            index_path: Path to save/load index
            metadata_path: Path to save/load metadata
        """
        if faiss is None:
            raise ImportError(
                "faiss not installed. Install with: pip install faiss-cpu"
            )
        
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index_path = Path(index_path) if index_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        
        # Create index
        self.index = self._create_index()
        self.metadata = {}  # Maps index position to chunk metadata
        self.vector_count = 0
    
    def _create_index(self) -> faiss.Index:
        """Create a new FAISS index."""
        if self.index_type == "cosine":
            # For cosine similarity, normalize embeddings and use L2
            index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "l2":
            index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        logger.info(f"Created FAISS index: type={self.index_type}, dim={self.embedding_dim}")
        return index
    
    def add_embeddings(self, embeddings: np.ndarray, 
                      metadata_list: List[Dict]) -> None:
        """
        Add embeddings to the index.
        
        Args:
            embeddings: numpy array of shape (n, embedding_dim)
            metadata_list: List of metadata dicts for each embedding
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Embeddings and metadata sizes don't match")
        
        # Normalize for cosine similarity if needed
        if self.index_type == "cosine":
            embeddings = self._normalize_embeddings(embeddings)
        
        # Ensure float32 type for FAISS
        embeddings = embeddings.astype(np.float32)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        for i, metadata in enumerate(metadata_list):
            self.metadata[self.vector_count + i] = metadata
        
        self.vector_count += len(embeddings)
        logger.info(f"Added {len(embeddings)} embeddings. Total: {self.vector_count}")
    
    def search(self, query_embedding: np.ndarray, 
               top_k: int = 5) -> Tuple[np.ndarray, List[int], List[Dict]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding (shape: embedding_dim)
            top_k: Number of results to return
            
        Returns:
            Tuple of (distances, indices, metadata_list)
        """
        # Normalize if needed
        if self.index_type == "cosine":
            query_embedding = self._normalize_embeddings(
                query_embedding.reshape(1, -1)
            )[0]
        
        # Ensure float32 type
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        distances = distances[0]
        indices = indices[0].tolist()
        
        # Convert distances to similarities (for L2 distance)
        if self.index_type == "cosine":
            # For L2 on normalized vectors, distance = 2 - 2*similarity
            similarities = 1 - (distances / 2)
        else:
            similarities = distances
        
        # Retrieve metadata
        metadata_list = [self.metadata.get(idx, {}) for idx in indices]
        
        return similarities, indices, metadata_list
    
    def get_size(self) -> int:
        """Get number of vectors in the index."""
        return self.index.ntotal
    
    def save(self) -> None:
        """Save index and metadata to disk."""
        if not self.index_path or not self.metadata_path:
            logger.warning("Index and metadata paths not set. Skipping save.")
            return
        
        # Save index
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        logger.info(f"Saved index to {self.index_path}")
        
        # Save metadata
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, 'w') as f:
            # Convert numpy arrays in metadata to lists
            metadata_serializable = {}
            for key, value in self.metadata.items():
                metadata_serializable[str(key)] = self._make_serializable(value)
            
            json.dump(metadata_serializable, f, indent=2)
        logger.info(f"Saved metadata to {self.metadata_path}")
    
    def load(self) -> bool:
        """
        Load index and metadata from disk.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.index_path or not self.metadata_path:
            logger.warning("Index and metadata paths not set. Skipping load.")
            return False
        
        try:
            # Load index
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"Loaded index from {self.index_path}")
            else:
                logger.warning(f"Index file not found: {self.index_path}")
                return False
            
            # Load metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    metadata_serializable = json.load(f)
                
                # Convert string keys back to integers
                self.metadata = {
                    int(key): value for key, value in metadata_serializable.items()
                }
                logger.info(f"Loaded metadata from {self.metadata_path}")
            else:
                logger.warning(f"Metadata file not found: {self.metadata_path}")
                return False
            
            self.vector_count = self.index.ntotal
            logger.info(f"Vector store loaded: {self.vector_count} vectors")
            return True
        
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings using L2 norm."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-10)
    
    @staticmethod
    def _make_serializable(obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: VectorStoreManager._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [VectorStoreManager._make_serializable(item) for item in obj]
        else:
            return obj
