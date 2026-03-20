"""
Retriever module for hybrid similarity search.
Combines semantic similarity with keyword overlap for ranking.
"""

import logging
from typing import List, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class Retriever:
    """
    Performs hybrid retrieval combining semantic similarity and keyword overlap.
    """
    
    def __init__(self, vector_store_manager, embedding_engine,
                 top_k: int = 5, similarity_threshold: float = 0.3,
                 keyword_boost: float = 0.1):
        """
        Initialize the retriever.
        
        Args:
            vector_store_manager: VectorStoreManager instance
            embedding_engine: EmbeddingEngine instance
            top_k: Number of results to retrieve
            similarity_threshold: Minimum similarity score
            keyword_boost: Weight for keyword overlap score
        """
        self.vector_store = vector_store_manager
        self.embedding_engine = embedding_engine
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.keyword_boost = keyword_boost
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Query string
            top_k: Number of results (uses default if None)
            
        Returns:
            List of retrieved chunks sorted by relevance score
        """
        if not query.strip():
            logger.warning("Empty query")
            return []
        
        top_k = top_k or self.top_k
        
        # Generate query embedding
        query_embedding = self.embedding_engine.get_query_embedding(query)
        
        # Get initial results from vector store
        similarities, indices, metadata_list = self.vector_store.search(
            query_embedding, top_k=top_k
        )
        
        # Calculate keyword overlap scores
        query_words = set(self._tokenize_query(query))
        keyword_scores = []
        
        for metadata in metadata_list:
            chunk_text = metadata.get("text", "")
            chunk_words = set(self._tokenize_query(chunk_text))
            
            # Calculate overlap
            if len(query_words) > 0:
                overlap = len(query_words & chunk_words)
                keyword_score = overlap / len(query_words)
            else:
                keyword_score = 0.0
            
            keyword_scores.append(keyword_score)
        
        # Combine scores
        combined_scores = self._combine_scores(
            similarities, keyword_scores
        )
        
        # Build results
        results = []
        for i, (idx, metadata) in enumerate(zip(indices, metadata_list)):
            result = {
                "chunk_id": metadata.get("chunk_id"),
                "doc_id": metadata.get("doc_id"),
                "text": metadata.get("text"),
                "similarity_score": float(similarities[i]),
                "keyword_score": float(keyword_scores[i]),
                "combined_score": float(combined_scores[i]),
                "chunk_index": metadata.get("chunk_index"),
            }
            results.append(result)
        
        # Sort by combined score
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Filter by similarity threshold
        results = [r for r in results if r["similarity_score"] >= self.similarity_threshold]
        
        logger.info(f"Retrieved {len(results)} chunks for query")
        return results
    
    def _combine_scores(self, similarity_scores: np.ndarray, 
                       keyword_scores: List[float]) -> np.ndarray:
        """
        Combine similarity and keyword scores.
        
        Formula: combined = similarity_score + (keyword_boost * keyword_score)
        
        Args:
            similarity_scores: Similarity scores from vector search
            keyword_scores: Keyword overlap scores
            
        Returns:
            Combined scores array
        """
        combined = similarity_scores + (self.keyword_boost * np.array(keyword_scores))
        return combined
    
    @staticmethod
    def _tokenize_query(text: str) -> List[str]:
        """
        Simple tokenization for keyword matching.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of lowercase tokens
        """
        # Simple split on whitespace and punctuation
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters, keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', '', text)
        
        # Split on whitespace
        tokens = text.split()
        
        # Filter out very short tokens
        tokens = [t for t in tokens if len(t) > 2]
        
        return tokens
