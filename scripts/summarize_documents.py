# summarize_documents.py
import json
from pathlib import Path
import re
from collections import Counter

# -------------------------------
# CONFIGURATION
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # go to docintel root
STRUCTURED_DIR = BASE_DIR / "structured_docs"
SUMMARY_DIR = BASE_DIR / "summaries"
SUMMARY_DIR.mkdir(exist_ok=True)

NUM_SENTENCES = 5  # number of sentences in the summary

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def clean_text(text):
    """Remove extra spaces, newlines, and non-standard characters."""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_sentences(text):
    """Basic sentence splitter."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

def score_sentences(text):
    """
    Simple frequency-based scoring:
    - Count word frequencies
    - Score sentences by sum of word frequencies
    """
    words = re.findall(r'\w+', text.lower())
    freq = Counter(words)
    
    sentences = split_sentences(text)
    sentence_scores = []
    for s in sentences:
        score = sum(freq.get(w.lower(), 0) for w in re.findall(r'\w+', s))
        sentence_scores.append((score, s))
    return sentence_scores

def summarize_text(text, num_sentences=NUM_SENTENCES):
    text = clean_text(text)
    sentence_scores = score_sentences(text)
    # sort by score descending
    top_sentences = sorted(sentence_scores, key=lambda x: x[0], reverse=True)[:num_sentences]
    # keep original order
    top_sentences_sorted = sorted(top_sentences, key=lambda x: text.find(x[1]))
    return " ".join([s[1] for s in top_sentences_sorted])

# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main():
    print("[INFO] Starting summarization...")
    for json_file in STRUCTURED_DIR.glob("*.json"):
        print(f"[INFO] Processing {json_file.name}...")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
        except Exception as e:
            print(f"[ERROR] Could not read {json_file.name}: {e}")
            continue
        
        # Combine all text chunks
        full_text = " ".join([chunk.get("text", "") for chunk in chunks if chunk.get("text")])
        if not full_text.strip():
            print(f"[WARNING] No text found in {json_file.name}")
            continue
        
        summary = summarize_text(full_text, NUM_SENTENCES)
        
        # Save summary as txt
        output_file = SUMMARY_DIR / f"{json_file.stem}_summary.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"[INFO] Saved summary to {output_file.name}")
    
    print("[INFO] Summarization complete.")

if __name__ == "__main__":
    main()
