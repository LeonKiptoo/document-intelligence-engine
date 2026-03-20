import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------
# PATH CONFIGURATION
# ---------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PARSED_DIR = os.path.join(BASE_DIR, "structured_docs_parsed")
VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")

os.makedirs(VECTOR_DIR, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------
# LOAD MODEL
# ---------------------------

print("[INFO] Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# ---------------------------
# UTIL FUNCTIONS
# ---------------------------

def extract_text(section):
    if isinstance(section, dict):
        content = section.get("content", "")
        return extract_text(content)

    elif isinstance(section, list):
        texts = []
        for item in section:
            texts.append(extract_text(item))
        return " ".join(texts).strip()

    elif isinstance(section, str):
        return section.strip()

    else:
        return ""

# ---------------------------
# READ DOCUMENTS
# ---------------------------

texts = []
metadata = []

print("[INFO] Reading parsed documents...")

if not os.path.exists(PARSED_DIR):
    print(f"[ERROR] Parsed directory not found: {PARSED_DIR}")
    exit()

files = [f for f in os.listdir(PARSED_DIR) if f.endswith(".json")]

print(f"[INFO] Found {len(files)} parsed files.")

for file in files:
    path = os.path.join(PARSED_DIR, file)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARNING] Could not read {file}: {e}")
        continue

    if not isinstance(data, list):
        print(f"[WARNING] Skipping {file} (not a list structure)")
        continue

    for section in data:
        content = extract_text(section)

        if len(content) > 30:
            texts.append(content)
            metadata.append({
                "source_file": file,
                "text": content
            })

print(f"[INFO] Total usable text sections collected: {len(texts)}")

if len(texts) == 0:
    print("[ERROR] No text extracted. Stopping.")
    exit()

# ---------------------------
# GENERATE EMBEDDINGS
# ---------------------------

print("[INFO] Generating embeddings...")
vectors = model.encode(texts, show_progress_bar=True)

vectors = np.array(vectors)

# ---------------------------
# SAVE VECTOR STORE
# ---------------------------

np.save(os.path.join(VECTOR_DIR, "vectors.npy"), vectors)

with open(os.path.join(VECTOR_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("[INFO] Embedding generation complete.")
print(f"[INFO] Vectors saved: {vectors.shape}")
print("[INFO] Metadata saved.")
print("[SUCCESS] Vector store ready.")
