import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from reasoning_engine import generate_answer

# =========================
# PATH CONFIGURATION
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")

VECTORS_PATH = os.path.join(VECTOR_DIR, "vectors.npy")
METADATA_PATH = os.path.join(VECTOR_DIR, "metadata.json")

# =========================
# LOAD MODELS & DATA
# =========================

MODEL_NAME = "all-MiniLM-L6-v2"

print("[INFO] Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("[INFO] Loading vector store...")
vectors = np.load(VECTORS_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"[INFO] Loaded {len(metadata)} sections.")

# =========================
# DOCUMENT TYPE DETECTION
# =========================

def detect_query_domain(query: str):
    q = query.lower()

    legal_keywords = ["agreement", "nda", "contract", "clause", "party", "confidential"]
    medical_keywords = ["diabetes", "disease", "patient", "treatment", "health"]
    data_keywords = ["employee", "salary", "dataset", "record"]

    if any(k in q for k in legal_keywords):
        return "legal"
    if any(k in q for k in medical_keywords):
        return "medical"
    if any(k in q for k in data_keywords):
        return "data"

    return "general"


def detect_doc_domain(source_file: str):
    name = source_file.lower()

    if "agreement" in name or "nda" in name or "contract" in name:
        return "legal"
    if "diabetes" in name or "research" in name:
        return "medical"
    if "employee" in name or "data" in name:
        return "data"

    return "general"

# =========================
# CORE RETRIEVAL LOGIC
# =========================

def retrieve_evidence(query, top_k=5, min_score=0.20):
    query_domain = detect_query_domain(query)

    query_vec = model.encode([query])
    similarities = cosine_similarity(query_vec, vectors)[0]

    ranked_indices = np.argsort(similarities)[::-1]

    results = []
    seen_sources = {}

    for idx in ranked_indices:
        score = float(similarities[idx])
        if score < min_score:
            break

        item = metadata[idx]
        source = item["source_file"]

        # 🔥 DOCUMENT FILTERING LAYER
        doc_domain = detect_doc_domain(source)
        if query_domain != "general" and doc_domain != query_domain:
            continue

        seen_sources[source] = seen_sources.get(source, 0) + 1
        if seen_sources[source] > 3:
            continue

        results.append({
            "score": round(score, 3),
            "source": source,
            "title": item.get("title", "Section"),
            "text": item["text"]
        })

        if len(results) >= top_k:
            break

    return results

# =========================
# INTERACTIVE LOOP
# =========================

def main():
    print("\n[INFO] Query engine ready. Type 'exit' to quit.\n")

    while True:
        query = input("Ask a question: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("[INFO] Session ended.")
            break

        evidence = retrieve_evidence(query)

        if not evidence:
            print("\n[INFO] No strong evidence found.\n")
            continue

        # Pass structured contexts (NOT string)
        answer = generate_answer(query, evidence)

        print("\n=== FINAL ANSWER ===\n")
        print(answer)

        print("\n=== SUPPORTING EVIDENCE ===\n")
        for i, item in enumerate(evidence, 1):
            print(f"[{i}] Source: {item['source']} | Score: {item['score']}")
            print(f"{item['text'][:300]}...\n")


if __name__ == "__main__":
    main()
