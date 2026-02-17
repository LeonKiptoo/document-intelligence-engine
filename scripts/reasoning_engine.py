import re
from typing import List, Dict


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def normalize(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [normalize(s) for s in sentences if len(s.strip()) > 40]


# ---------------------------------------------------------
# Document type detection (STRUCTURAL, not keyword spam)
# ---------------------------------------------------------

def detect_document_type(text: str) -> str:
    t = text.lower()

    legal_markers = [
        "non disclosure agreement",
        "confidential information",
        "receiving party",
        "disclosing party",
        "shall",
        "hereinafter",
    ]

    research_markers = [
        "abstract",
        "methodology",
        "dataset",
        "results",
        "discussion",
        "this study aims",
        "research",
    ]

    legal_score = sum(1 for m in legal_markers if m in t)
    research_score = sum(1 for m in research_markers if m in t)

    if legal_score >= 2:
        return "Legal Document"
    if research_score >= 2:
        return "Research Document"

    return "General Document"


# ---------------------------------------------------------
# Section filtering (kill noise early)
# ---------------------------------------------------------

def is_noise(sentence: str) -> bool:
    s = sentence.lower()

    noise_markers = [
        "acknowledg",
        "gratitude",
        "supervisor",
        "kaggle",
        "creativecommons",
        "here is the link",
        "page",
        "post office box",
        "signature",
        "date:",
    ]

    return any(m in s for m in noise_markers)


# ---------------------------------------------------------
# Core reasoning engine
# ---------------------------------------------------------

def generate_answer(question: str, retrieved_sections: List[Dict]) -> str:
    if not retrieved_sections:
        return "No relevant content found."

    # 1️⃣ Choose ONE primary document
    primary = retrieved_sections[0]
    document_text = primary["text"]

    doc_type = detect_document_type(document_text)

    # 2️⃣ Build document-wide sentence pool
    all_sentences = []
    for section in retrieved_sections:
        all_sentences.extend(split_sentences(section["text"]))

    # 3️⃣ Remove noise
    clean_sentences = [s for s in all_sentences if not is_noise(s)]

    # 4️⃣ Intent-aware filtering
    q = question.lower()

    if "cover" in q or "about" in q or "what does" in q:
        focus_terms = ["purpose", "aim", "objective", "confidential", "information", "predict"]
    else:
        focus_terms = q.split()

    scored = []
    for s in clean_sentences:
        score = sum(1 for t in focus_terms if t in s.lower())
        if score > 0:
            scored.append((score, s))

    if not scored:
        return "Relevant content found but could not generate a precise answer."

    scored.sort(reverse=True, key=lambda x: x[0])

    # 5️⃣ Synthesize answer (NOT raw extraction)
    bullets = []
    used = set()

    for _, s in scored:
        if s not in used:
            bullets.append(s.rstrip(" ;."))
            used.add(s)
        if len(bullets) == 4:
            break

    header = (
        "DOCUMENT ANALYSIS RESULT\n"
        "-------------------------\n"
        f"Document Type: {doc_type}\n\n"
        "Answer Summary:"
    )

    body = "\n".join(f"{i+1}. {b}." for i, b in enumerate(bullets))

    return f"{header}\n{body}"
