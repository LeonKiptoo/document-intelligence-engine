import json
import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "structured_docs"
OUTPUT_DIR = BASE_DIR / "structured_docs_parsed"

OUTPUT_DIR.mkdir(exist_ok=True)

SECTION_PATTERN = re.compile(r"^\s*(\d+(\.\d+)*|[A-Z][A-Z\s]{3,}|[A-Z][a-z]+)\s*$")

def detect_section(line):
    return bool(SECTION_PATTERN.match(line.strip()))

def parse_document(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    sections = []
    current_section = {
        "title": "Uncategorized",
        "content": []
    }

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue

        lines = text.split("\n")

        for line in lines:
            clean = line.strip()

            if detect_section(clean):
                if current_section["content"]:
                    sections.append(current_section)

                current_section = {
                    "title": clean,
                    "content": []
                }
            else:
                current_section["content"].append(clean)

    if current_section["content"]:
        sections.append(current_section)

    return sections

def process_all():
    print("[INFO] Starting structural parsing...")

    for file in INPUT_DIR.glob("structured_*.json"):
        print(f"[INFO] Parsing {file.name}")

        sections = parse_document(file)

        output_file = OUTPUT_DIR / file.name.replace("structured_", "parsed_")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(sections, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Saved {len(sections)} sections to {output_file.name}")

    print("[INFO] Structural parsing complete.")

if __name__ == "__main__":
    process_all()
