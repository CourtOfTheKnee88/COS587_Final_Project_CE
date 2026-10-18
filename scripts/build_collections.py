import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / "task1_train_files_2026" / "cases"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed"

if not DEFAULT_DATA_DIR.is_dir():
    raise FileNotFoundError(f"Task 1 training corpus directory not found: {DEFAULT_DATA_DIR}")

all_files = sorted(path for path in DEFAULT_DATA_DIR.glob("*.txt"))
if not all_files:
    raise FileNotFoundError(f"No .txt case files found in the training corpus directory: {DEFAULT_DATA_DIR}")

def clean_coliee_text(text: str) -> str:
    # Normalize whitespaces but preserve the suppressed tags
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def get_paragraphs(text: str) -> List[Tuple[int, str]]:
    paragraph_markers = list(
        re.finditer(r"(?m)^[ \t]*\[[ \t]*(\d+)[ \t]*\][ \t]*(?:\r?\n|$)", text)
    )
    paragraphs = []

    for index, marker in enumerate(paragraph_markers):
        end = paragraph_markers[index + 1].start() if index + 1 < len(paragraph_markers) else len(text)
        paragraph_text = clean_coliee_text(text[marker.end():end])
        if paragraph_text:
            paragraphs.append((int(marker.group(1)), paragraph_text))

    return paragraphs


def extract_case_year(file_path: Path) -> Optional[int]:
    """
    Reads the first 30 lines of a COLIEE text file and attempts to extract 
    the decision year using regular expressions.
    """
    # Regex targets 4-digit years adjacent to structural terms like Date or Dated
    # It accounts for formats like: "Date: 20151124", "Date: 2012-05-14", or "Dated: April 4, 1999"
    date_patterns = [
        r"(?:Date|Dated)\s*:\s*(\d{4})\d{4}",          # Formats like 20151124 -> 2015
        r"(?:Date|Dated)[^:\n]*:\s*.*?(\d{4})",        # Formats like Date of Decision: May 12, 2010
        r"\b(19\d{2}|20\d{2})\b"                       # General fallback for any safe year block
    ]
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # We only need the header block to extract metadata
            header_lines = [next(f).strip() for _ in range(30)]
            header_text = "\n".join(header_lines)
            
        for pattern in date_patterns:
            match = re.search(pattern, header_text, re.IGNORECASE)
            if match:
                year = int(match.group(1))
                # Validate the year is reasonable for the dataset scale
                if 1900 <= year <= 2026: 
                    return year
                    
        return None
    except (FileNotFoundError, StopIteration):
        return None

# Setting up documents as structured JSONs
collection_documents = []

for filename in all_files:
    with open(filename, "r", encoding="utf-8") as f:
        raw_content = f.read()
        
    case_date = extract_case_year(filename)

    for paragraph_number, paragraph_text in get_paragraphs(raw_content):
        collection_documents.append({
            "doc_id": filename.name, # e.g. "000001.txt"
            "para_id": f"{filename.stem}_{paragraph_number}",
            "date": case_date,
            "text": paragraph_text
        })

# Export to a JSONL file format
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with open(DEFAULT_OUTPUT_DIR / "coliee_corpus.jsonl", "w", encoding="utf-8") as out:
    for doc in collection_documents:
        out.write(json.dumps(doc) + "\n")
