
"""
This file handles building the inverted index for our COLIEE legal
document retrieval project.

Instead of reading through all the original case files, the indexer
uses the processed JSONL collection that we already created. Each
record contains a paragraph from a legal case, along with its
paragraph ID, case ID, and text.

The indexer uses our tokenizer to break the text into tokens and
keeps track of which terms appear in each paragraph and how often.
It also calculates paragraph lengths and the average length across
the collection.

This information will be used by our BM25 retrieval model to help
find relevant legal cases based on a search query.

To build the index, run:
    python -m src.index.index

The finished index will be saved in data/processed/.
"""

import json
from pathlib import Path
from collections import Counter
from src.preprocessing.tokenize import tokenize_text

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file)

    print("Saved index to: ", file_path)

class CorpusIndex:
    def __init__(self):
        self.num_docs = 0
        self.avg_doc_length = 0.0
        self.doc_lengths = []
        self._postings = {}
        self._para_ids = []
        self._para_numbers = {}
        self._case_ids = []

    def vocabulary(self):
        return self._postings.keys()

    def df(self, term):
        if term not in self._postings:
            return 0

        return len(self._postings[term][0])

    def postings(self, term):
        if term not in self._postings:
            return [], []

        return self._postings[term]

    def para_id(self, doc_number):
        return self._para_ids[doc_number]

    def para_number(self, para_id):
        return self._para_numbers[para_id]

    def case_id(self, doc_number):
        return self._case_ids[doc_number]

    def to_dict(self):
        return {
            "num_docs": self.num_docs,
            "avg_doc_length": self.avg_doc_length,
            "doc_lengths": self.doc_lengths,
            "postings": self._postings,
            "para_ids": self._para_ids,
            "case_ids": self._case_ids
        }

    @classmethod
    def from_dict(cls, data):
        index = cls()
        index.num_docs = data["num_docs"]
        index.avg_doc_length = data["avg_doc_length"]
        index.doc_lengths = data["doc_lengths"]
        index._postings = data["postings"]
        index._para_ids = data["para_ids"] if "para_ids" in data else data["doc_ids"]
        index._case_ids = data["case_ids"]
        index._para_numbers = {
            para_id: doc_number
            for doc_number, para_id in enumerate(index._para_ids)
        }

        return index

def load_index(file_path, force_reindex=False):
    index_path = Path(file_path)

    if force_reindex:
        collection_path = index_path.with_name("coliee_corpus.jsonl")
        return build_and_save_index(collection_path, index_path)

    with open(index_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return CorpusIndex.from_dict(data)

def build_index_from_jsonl(file_path):
    """So here I am building an inverted index from a preprocessed JSONL collection. Fingers crossed."""
    index = CorpusIndex()

    with open(file_path, "r", encoding="utf-8-sig") as file:
        for line in file:
            record = json.loads(line)
            case_id = record["doc_id"]
            para_id = record["para_id"]
            tokens = tokenize_text(record["text"])
            doc_number = index.num_docs
            index._para_ids.append(para_id)
            index._case_ids.append(case_id)
            index._para_numbers[para_id] = doc_number
            index.doc_lengths.append(len(tokens))
            term_frequencies = Counter(tokens)

            for term, frequency in term_frequencies.items():
                if term not in index._postings:
                    index._postings[term] = ([], [])

                index._postings[term][0].append(doc_number)
                index._postings[term][1].append(frequency)

            index.num_docs += 1

    if index.num_docs > 0:
        index.avg_doc_length = sum(index.doc_lengths) / index.num_docs

    return index

def build_and_save_index(collection_path, index_path):
    """Build an index from a collection and then save it to eventually return it."""
    index = build_index_from_jsonl(collection_path)
    save_json(index.to_dict(), index_path)

    return index

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    processed_dir = repo_root / "data" / "processed"

    # Build the index from our processed legal paragraphs.
    print("Building COLIEE paragraph index...")
    coliee_index = build_and_save_index(
        processed_dir / "coliee_corpus.jsonl",
        processed_dir / "coliee_index.json"
    )

    print("Paragraphs indexed:", coliee_index.num_docs)
    print("Unique terms in index:", len(coliee_index.vocabulary()))
    print("Average paragraph length:", coliee_index.avg_doc_length)
