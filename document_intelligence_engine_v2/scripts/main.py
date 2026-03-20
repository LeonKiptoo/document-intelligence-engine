"""
Main CLI application for the Document Intelligence Retrieval Engine v2.
Minimal, end-to-end pipeline for ingesting and retrieving documents.
"""
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent.parent / ".env")
import os
import logging
import argparse
import sys
from pathlib import Path
import numpy as np

import config
from document_loader import DocumentLoader
from chunking_engine import ChunkingEngine
from embedding_engine import EmbeddingEngine
from vector_store_manager import VectorStoreManager
from retriever import Retriever
from context_builder import ContextBuilder
from generation_engine import GenerationEngine


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def ingest_command():
    """Ingest documents from data/ folder into FAISS."""
    data_dir = config.DATA_DIR

    print(f"\n📂 Loading documents from: {data_dir}")

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return 1

    supported_files = list(data_dir.glob("*.[pP][dD][fF]")) + \
                     list(data_dir.glob("*.[dD][oO][cC][xX]")) + \
                     list(data_dir.glob("*.[cC][sS][vV]")) + \
                     list(data_dir.glob("*.[xX][lL][sS][xX]"))

    if not supported_files:
        print(f"⚠️  No supported files found in {data_dir}")
        print(f"   Supported formats: .pdf, .docx, .csv, .xlsx")
        return 1

    try:
        # 1. Load documents
        loader = DocumentLoader()
        documents = loader.load_directory(str(data_dir))

        if not documents:
            print("❌ Failed to load documents")
            return 1

        print(f"✓ Loaded {len(documents)} document(s)")

        # 2. Chunk documents
        print("🔪 Chunking documents...")
        chunker = ChunkingEngine(
            min_chunk_size=config.MIN_CHUNK_SIZE,
            max_chunk_size=config.MAX_CHUNK_SIZE,
            overlap_ratio=config.CHUNK_OVERLAP,
        )
        chunks = chunker.chunk_documents(documents)

        if not chunks:
            print("❌ No chunks created")
            return 1

        print(f"✓ Created {len(chunks)} chunks")

        # 3. Generate embeddings
        print("🧠 Generating embeddings...")
        embedder = EmbeddingEngine(
            model_name=config.EMBEDDING_MODEL,
            batch_size=config.BATCH_SIZE,
            device=config.DEVICE,
        )
        chunks = embedder.embed_chunks(chunks)
        print(f"✓ Generated embeddings ({config.EMBEDDING_DIMENSION} dimensions)")

        # 4. Store in FAISS
        print("💾 Storing in FAISS...")
        vector_store = VectorStoreManager(
            embedding_dim=config.EMBEDDING_DIMENSION,
            index_type=config.FAISS_INDEX_TYPE,
            index_path=config.FAISS_INDEX_PATH,
            metadata_path=config.METADATA_PATH,
        )

        embeddings = np.array([c["embedding"] for c in chunks], dtype=np.float32)
        metadata_list = [
            {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "text": c["text"],
                "chunk_index": c["chunk_index"],
                "token_count": c["token_count"],
            }
            for c in chunks
        ]

        vector_store.add_embeddings(embeddings, metadata_list)
        vector_store.save()

        print(f"✓ Saved FAISS index to {config.VECTOR_STORE_DIR}")
        print(f"\n✅ Ingest complete: {len(documents)} document(s), {len(chunks)} chunk(s)")
        return 0

    except Exception as e:
        print(f"❌ Error during ingestion: {str(e)}")
        logger.exception("Ingest error")
        return 1


def retrieve_command(query: str, top_k: int = None):
    """Retrieve relevant chunks for a query."""
    top_k = top_k or config.TOP_K

    try:
        # 1. Load vector store
        vector_store = VectorStoreManager(
            embedding_dim=config.EMBEDDING_DIMENSION,
            index_type=config.FAISS_INDEX_TYPE,
            index_path=config.FAISS_INDEX_PATH,
            metadata_path=config.METADATA_PATH,
        )

        if not vector_store.load():
            print(f"\n❌ Vector index not found at: {config.VECTOR_STORE_DIR}")
            print("   Run 'python main.py ingest' first to build the index")
            return 1

        vector_count = vector_store.get_size()
        print(f"\n✓ Loaded vector store ({vector_count} chunks)")

        # 2. Initialize components
        embedder = EmbeddingEngine(
            model_name=config.EMBEDDING_MODEL,
            device=config.DEVICE,
        )
        retriever = Retriever(
            vector_store_manager=vector_store,
            embedding_engine=embedder,
            top_k=top_k,
            similarity_threshold=config.SIMILARITY_THRESHOLD,
            keyword_boost=config.KEYWORD_BOOST,
        )
        builder = ContextBuilder(
            redundancy_threshold=config.REDUNDANCY_THRESHOLD,
            max_context_tokens=config.MAX_CONTEXT_TOKENS,
        )

        # 3. Retrieve chunks
        print("🔍 Retrieving chunks...")
        retrieved_chunks = retriever.retrieve(query, top_k=top_k)

        if not retrieved_chunks:
            print("⚠️  No relevant chunks found for query")
            return 0

        # 4. Build context
        context = builder.build_context(retrieved_chunks, query)

        # 5. Print retrieval stats
        stats = context['statistics']

        print("\n" + "=" * 80)
        print("RETRIEVAL RESULTS")
        print("=" * 80)
        print(f"\n📋 Query: {query}")
        print(f"📊 Retrieved: {stats['total_chunks_retrieved']} chunks")
        print(f"   After dedup: {stats['chunks_after_dedup']} chunks")
        print(f"   In context: {stats['chunks_in_context']} chunks")
        print(f"   Context size: {stats['context_tokens']} tokens")

        # 6. Generate answer with Groq
        api_key = os.environ.get("GROQ_API_KEY", "")
        if api_key:
            print("\n" + "-" * 80)
            print("GENERATING ANSWER...")
            print("-" * 80)
            generator = GenerationEngine(api_key=api_key)
            answer = generator.generate_answer(query, context['context'])
            print("\n" + "=" * 80)
            print("ANSWER:")
            print("=" * 80)
            print(answer)
            print("=" * 80 + "\n")
        else:
            print("\n[WARNING] GEMINI_API_KEY not set. Showing raw context only.\n")
            print(context['context'])

        # 7. Detailed scores
        print("DETAILED SCORES:")
        for i, chunk in enumerate(retrieved_chunks[:top_k], 1):
            print(f"\n{i}. {chunk['chunk_id']}")
            print(f"   📍 Document: {chunk['doc_id']}")
            print(f"   🎯 Semantic similarity: {chunk['similarity_score']:.4f}")
            print(f"   🔑 Keyword overlap: {chunk['keyword_score']:.4f}")
            print(f"   ⭐ Combined score: {chunk['combined_score']:.4f}")

        print("\n" + "=" * 80 + "\n")
        return 0

    except Exception as e:
        print(f"❌ Error during retrieval: {str(e)}")
        logger.exception("Retrieval error")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Document Intelligence Engine v2 - Minimal Retrieval Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  1. Ingest documents from data/ folder:
     python main.py ingest

  2. Retrieve relevant chunks:
     python main.py retrieve "What is the main topic?"

  3. Retrieve with custom top-k:
     python main.py retrieve "What is the main topic?" --top-k 3
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Load documents from data/ and build FAISS index"
    )

    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Retrieve relevant chunks for a query"
    )
    retrieve_parser.add_argument(
        "query",
        type=str,
        help="Query string"
    )
    retrieve_parser.add_argument(
        "--top-k",
        type=int,
        default=config.TOP_K,
        help=f"Number of results to retrieve (default: {config.TOP_K})"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "ingest":
        return ingest_command()
    elif args.command == "retrieve":
        return retrieve_command(args.query, top_k=args.top_k)

    return 1


if __name__ == "__main__":
    sys.exit(main())

