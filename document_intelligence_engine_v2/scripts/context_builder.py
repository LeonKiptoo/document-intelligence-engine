"""
Context builder for assembling structured retrieval results.
Removes redundant chunks and builds context windows.
"""

import logging
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds structured context from retrieved chunks.
    Removes redundancy and maintains document grouping.
    """
    
    def __init__(self, redundancy_threshold: float = 0.9,
                 max_context_tokens: int = 2000,
                 char_per_token: int = 4):
        """
        Initialize the context builder.
        
        Args:
            redundancy_threshold: Similarity threshold to consider chunks redundant
            max_context_tokens: Maximum tokens in final context
            char_per_token: Approximate characters per token
        """
        self.redundancy_threshold = redundancy_threshold
        self.max_context_tokens = max_context_tokens
        self.max_context_chars = int(max_context_tokens * char_per_token)
        self.char_per_token = char_per_token
    
    def build_context(self, retrieved_chunks: List[Dict],
                     query: str = "") -> Dict:
        """
        Build structured context from retrieved chunks.
        
        Args:
            retrieved_chunks: List of retrieved chunk dicts
            query: Original query (for reference)
            
        Returns:
            Dictionary with structured context:
            {
                "query": str,
                "context": str,
                "chunks": List[dict],
                "statistics": {
                    "total_chunks_retrieved": int,
                    "chunks_after_dedup": int,
                    "context_tokens": int,
                    "context_chars": int,
                }
            }
        """
        if not retrieved_chunks:
            logger.warning("No chunks to build context from")
            return {
                "query": query,
                "context": "",
                "chunks": [],
                "statistics": {
                    "total_chunks_retrieved": 0,
                    "chunks_after_dedup": 0,
                    "context_tokens": 0,
                    "context_chars": 0,
                },
            }
        
        # Remove redundant chunks
        unique_chunks = self._remove_redundant_chunks(retrieved_chunks)
        
        # Limit by token count
        final_chunks = self._limit_by_tokens(unique_chunks)
        
        # Build context string grouped by document
        context_str = self._build_context_string(final_chunks)
        
        # Calculate statistics
        context_tokens = self._estimate_tokens(context_str)
        context_chars = len(context_str)
        
        result = {
            "query": query,
            "context": context_str,
            "chunks": final_chunks,
            "statistics": {
                "total_chunks_retrieved": len(retrieved_chunks),
                "chunks_after_dedup": len(unique_chunks),
                "chunks_in_context": len(final_chunks),
                "context_tokens": context_tokens,
                "context_chars": context_chars,
            },
        }
        
        return result
    
    def _remove_redundant_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Remove redundant chunks using similarity scores.
        Keeps highest-scoring chunk among similar ones.
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            List of unique chunks
        """
        if len(chunks) <= 1:
            return chunks
        
        unique_chunks = [chunks[0]]
        
        for chunk in chunks[1:]:
            is_redundant = False
            
            for unique_chunk in unique_chunks:
                # Compare embeddings if available
                if "embedding" in chunk and "embedding" in unique_chunk:
                    similarity = self._cosine_similarity(
                        chunk["embedding"],
                        unique_chunk["embedding"]
                    )
                    
                    if similarity >= self.redundancy_threshold:
                        is_redundant = True
                        break
            
            if not is_redundant:
                unique_chunks.append(chunk)
        
        logger.info(
            f"Removed {len(chunks) - len(unique_chunks)} redundant chunks"
        )
        return unique_chunks
    
    def _limit_by_tokens(self, chunks: List[Dict]) -> List[Dict]:
        """
        Limit chunks to maximum token count.
        
        Args:
            chunks: List of chunks
            
        Returns:
            List of chunks within token limit
        """
        final_chunks = []
        total_chars = 0
        
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            chunk_chars = len(chunk_text)
            
            if total_chars + chunk_chars > self.max_context_chars:
                logger.info(
                    f"Reached max context size. "
                    f"Included {len(final_chunks)} chunks"
                )
                break
            
            final_chunks.append(chunk)
            total_chars += chunk_chars
        
        return final_chunks
    
    def _build_context_string(self, chunks: List[Dict]) -> str:
        """
        Build structured context string from chunks.
        Groups by document for readability.
        
        Args:
            chunks: List of chunks to include
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return ""
        
        # Group chunks by document
        docs = {}
        for chunk in chunks:
            doc_id = chunk.get("doc_id", "unknown")
            if doc_id not in docs:
                docs[doc_id] = []
            docs[doc_id].append(chunk)
        
        # Build context string
        context_parts = []
        
        for doc_id, doc_chunks in docs.items():
            context_parts.append(f"\n=== Document: {doc_id} ===\n")
            
            for chunk in doc_chunks:
                chunk_id = chunk.get("chunk_id", "unknown")
                text = chunk.get("text", "")
                score = chunk.get("combined_score", 0)
                
                context_parts.append(f"[{chunk_id}] (score: {score:.3f})\n")
                context_parts.append(text)
                context_parts.append("\n\n")
        
        context_str = "".join(context_parts).strip()
        return context_str
    
    @staticmethod
    def _cosine_similarity(embedding1: np.ndarray,
                          embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Normalize
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    @staticmethod
    def _estimate_tokens(text: str, char_per_token: int = 4) -> int:
        """
        Estimate token count from character count.
        """
        return max(1, len(text) // char_per_token)
