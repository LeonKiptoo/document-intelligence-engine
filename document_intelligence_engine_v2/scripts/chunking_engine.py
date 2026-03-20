"""
Chunking engine for splitting documents into overlapping chunks.
Maintains document IDs and creates unique chunk identifiers.
"""

import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)


class ChunkingEngine:
    """
    Splits documents into overlapping chunks of specified token size.
    Maintains document context and generates chunk IDs.
    """
    
    def __init__(self, min_chunk_size: int = 500, max_chunk_size: int = 800,
                 overlap_ratio: float = 0.10, char_per_token: int = 4):
        """
        Initialize the chunking engine.
        
        Args:
            min_chunk_size: Minimum chunk size in tokens
            max_chunk_size: Maximum chunk size in tokens
            overlap_ratio: Overlap ratio (0.10 = 10%)
            char_per_token: Approximate characters per token for estimation
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_ratio = overlap_ratio
        self.char_per_token = char_per_token
        
        # Convert to characters
        self.min_chunk_chars = int(min_chunk_size * char_per_token)
        self.max_chunk_chars = int(max_chunk_size * char_per_token)
        self.overlap_chars = int(self.max_chunk_chars * overlap_ratio)
    
    def chunk_document(self, doc: Dict) -> List[Dict]:
        """
        Chunk a document into overlapping parts.
        
        Args:
            doc: Document dictionary with 'text', 'doc_id', etc.
            
        Returns:
            List of chunk dictionaries with:
            {
                "chunk_id": str,
                "doc_id": str,
                "text": str,
                "chunk_index": int,
                "token_count": int,
            }
        """
        text = doc.get("text", "")
        doc_id = doc.get("doc_id", "unknown")
        
        if not text:
            logger.warning(f"Document {doc_id} has no text")
            return []
        
        chunks = self._split_text(text)
        chunk_list = []
        
        for i, chunk_text in enumerate(chunks):
            chunk = {
                "chunk_id": f"{doc_id}_chunk_{i}",
                "doc_id": doc_id,
                "text": chunk_text,
                "chunk_index": i,
                "token_count": self._estimate_tokens(chunk_text),
            }
            chunk_list.append(chunk)
        
        logger.info(f"Created {len(chunk_list)} chunks from document {doc_id}")
        return chunk_list
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Strategy:
        1. Split by sentences to maintain semantic boundaries
        2. Group sentences into chunks of target size
        3. Add overlap between chunks
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences (simple approach)
        sentences = self._split_sentences(text)
        
        if not sentences:
            return [text] if text else []
        
        chunks = []
        current_chunk = ""
        current_chars = 0
        previous_overlap = ""
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_chars = len(sentence)
            
            # Add space before sentence (except first)
            if current_chunk:
                test_chunk = current_chunk + " " + sentence
            else:
                test_chunk = sentence
            
            test_chars = len(test_chunk)
            
            # Check if adding this sentence exceeds max chunk size
            if test_chars > self.max_chunk_chars and current_chunk:
                # Save current chunk with overlap
                chunks.append(current_chunk)
                previous_overlap = self._create_overlap(current_chunk)
                
                # Start new chunk with overlap
                current_chunk = previous_overlap + " " + sentence
                current_chars = len(current_chunk)
            else:
                # Add to current chunk
                current_chunk = test_chunk
                current_chars = test_chars
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex.
        """
        # Split on common sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s.strip()]
    
    def _create_overlap(self, chunk: str) -> str:
        """
        Create overlap text from the end of a chunk.
        Returns the last portion of the chunk to use as overlap.
        """
        if len(chunk) <= self.overlap_chars:
            return chunk
        
        # Find last sentence boundary within overlap range
        overlap_start = len(chunk) - self.overlap_chars
        text_slice = chunk[overlap_start:]
        
        # Try to start from sentence boundary
        sentence_split = text_slice.rfind('. ')
        if sentence_split > 0:
            return text_slice[sentence_split + 2:]
        
        return text_slice
    
    @staticmethod
    def _estimate_tokens(text: str, char_per_token: int = 4) -> int:
        """
        Estimate token count using character count.
        Approximate: 1 token ≈ 4 characters for English text.
        """
        return max(1, len(text) // char_per_token)
