import json
import re
from pathlib import Path
from typing import Optional

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
    # Optional: Replace specific case header artifacts if present
    # text = re.sub(r'Date: \d+', '', text) 
    return text.strip()

# Will need to be update to actually parse dates, does not actually get the dates from the documents yet
def extract_case_date(text: str) -> Optional[str]:
    for line in text.splitlines():
        match = re.match(r'^[ \t]*Date[ \t]*:[ \t]*(.+?)[ \t]*$', line, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


# Setting up documents as structured JSONs
collection_documents = []

for filename in all_files:
    with open(filename, "r", encoding="utf-8") as f:
        raw_content = f.read()
        
    cleaned_content = clean_coliee_text(raw_content)
    case_date = extract_case_date(raw_content)
    
    collection_documents.append({
        "id": filename.name, # e.g. "000001.txt"
        "date": case_date,
        "contents": cleaned_content
    })

# Export to a JSONL file format
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with open(DEFAULT_OUTPUT_DIR / "coliee_corpus.jsonl", "w", encoding="utf-8") as out:
    for doc in collection_documents:
        out.write(json.dumps(doc) + "\n")
