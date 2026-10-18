
"""
Tokenizer for our COLIEE legal document retrieval project.

This takes text from legal documents or search queries and breaks it
into tokens that we can use when building and searching the index.

The preprocessing settings are stored in config/preprocessing.yaml
so we can change things like stopwords and stemming when testing
different retrieval methods.
"""

import re
import sys
from pathlib import Path
from functools import lru_cache

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "preprocessing.yaml"

MATH_SPAN = re.compile(r"\$[^$]*\$")
LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+")
URL = re.compile(r"https?://\S+|www\.\S+")
CITATION = re.compile(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]")

# Match Unicode letters and numbers, excluding underscores.
TOKEN_PATTERN = re.compile(r"[^\W_]+")
WORD_PATTERN = re.compile(r"\w+")


def load_config(config_path=DEFAULT_CONFIG_PATH):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def clean_text(text, remove_math=False, remove_urls=False,
               remove_citations=False):
    """Optionally remove unwanted formatting from legal text."""
    if remove_math:
        if text.count("$") % 2 == 0:
            text = MATH_SPAN.sub(" ", text)
        text = LATEX_COMMAND.sub(" ", text)

    if remove_urls:
        text = URL.sub(" ", text)

    if remove_citations:
        text = CITATION.sub(" ", text)

    return text


def load_stopwords(config):
    """Build a stopword set from the selected source."""
    source = config.get("source", "none")

    if source == "none":
        words = set()
    elif source == "nltk":
        from nltk.corpus import stopwords
        words = set(stopwords.words("english"))
    elif source == "custom":
        path = REPO_ROOT / config["custom_file"]
        with open(path, "r", encoding="utf-8") as file:
            words = {
                line.strip().lower()
                for line in file if line.strip()
            }
    else:
        raise ValueError(f"Unknown stopword source: {source}")

    words.update(word.lower() for word in (config.get("extra") or []))
    words.difference_update(
        word.lower() for word in (config.get("keep") or [])
    )

    return words


def build_stemmer(name):
    """Create an optional cached stemmer."""
    if name == "none":
        return None

    from nltk.stem import PorterStemmer, SnowballStemmer

    if name == "porter":
        stemmer = PorterStemmer()
    elif name == "snowball":
        stemmer = SnowballStemmer("english")
    else:
        raise ValueError(f"Unknown stemmer: {name}")

    return lru_cache(maxsize=50000)(stemmer.stem)


class Tokenizer:
    """Reusable tokenizer for documents and queries."""

    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        config = load_config(config_path)
        cleaning = config.get("cleaning") or {}
        settings = config.get("tokenizer") or {}

        self.cleaning = {
            "remove_math": cleaning.get("remove_math", False),
            "remove_urls": cleaning.get("remove_urls", False),
            "remove_citations": cleaning.get("remove_citations", False),
        }

        self.lowercase = settings.get("lowercase", True)
        self.min_token_length = settings.get("min_token_length", 1)
        self.drop_numeric = settings.get("drop_numeric", False)

        self.token_pattern = (
            TOKEN_PATTERN
            if settings.get("split_underscores", True)
            else WORD_PATTERN
        )

        self.stopwords = load_stopwords(settings.get("stopwords") or {})
        self.stem = build_stemmer(settings.get("stemming", "none"))

    def tokenize(self, text):
        """Return an ordered list of processed tokens."""
        text = clean_text(text, **self.cleaning)

        if self.lowercase:
            text = text.lower()

        tokens = [
            token
            for token in self.token_pattern.findall(text)
            if len(token) >= self.min_token_length
            and not (self.drop_numeric and token.isdigit())
            and token not in self.stopwords
        ]

        if self.stem is not None:
            tokens = [self.stem(token) for token in tokens]

        return tokens


_default_tokenizer = None


def tokenize_text(text):
    """Tokenize text using the default COLIEE configuration."""
    global _default_tokenizer

    if _default_tokenizer is None:
        _default_tokenizer = Tokenizer()

    return _default_tokenizer.tokenize(text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m src.preprocessing.tokenize "<text>"')
        sys.exit(1)

    print(tokenize_text(" ".join(sys.argv[1:])))
"""
This file handles building the inverted indexes for our paper and
paragraph collections.

Instead of reading and processing all the original papers again,
the indexer uses the preprocessed JSONL files that we already created.
Each document has an ID and a list of tokens that we can use to
build the index.

For each document, we keep track of which terms appear, how often
they appear, and which document they came from. We also calculate
document lengths and the average length across the collection.

All of this information gets stored in posting lists so our BM25
and TF-IDF models can use it when searching for relevant documents.

To build both indexes, run:
    python -m src.index.indexer

The finished indexes are saved in data/processed/.
"""

import json
from collections import Counter

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
        self._doc_ids = []
        self._doc_numbers = {}

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

    def doc_id(self, doc_number):
        return self._doc_ids[doc_number]

    def doc_number(self, doc_id):
        return self._doc_numbers[doc_id]

    def to_dict(self):
        return {
            "num_docs": self.num_docs,
            "avg_doc_length": self.avg_doc_length,
            "doc_lengths": self.doc_lengths,
            "postings": self._postings,
            "doc_ids": self._doc_ids
        }

    @classmethod
    def from_dict(cls, data):
        index = cls()
        index.num_docs = data["num_docs"]
        index.avg_doc_length = data["avg_doc_length"]
        index.doc_lengths = data["doc_lengths"]
        index._postings = data["postings"]
        index._doc_ids = data["doc_ids"]
        index._doc_numbers = {
            doc_id: doc_number
            for doc_number, doc_id in enumerate(index._doc_ids)
        }

        return index

def load_index(file_path, force_reindex=False):
    if force_reindex:
        from pathlib import Path

        index_path = Path(file_path)

        collection_name = {
            "paper_index.json": "papers.jsonl",
            "paragraph_index.json": "paragraphs.jsonl"
        }

        collection_path = index_path.with_name(
            collection_name[index_path.name]
        )

        return build_and_save_index(collection_path, index_path)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return CorpusIndex.from_dict(data)

def build_index_from_jsonl(file_path):
    """So here I am building an inverted index from a preprocessed JSONL collection. Fingers crossed."""
    index = CorpusIndex()

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            doc_id = record["doc_id"]
            tokens = record["tokens"]
            doc_number = index.num_docs

            index._doc_ids.append(doc_id)
            index._doc_numbers[doc_id] = doc_number
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
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    processed_dir = repo_root / "data" / "processed"

    # Build the paper index from the preprocessed collection.
    print("Building paper index...")
    paper_index = build_and_save_index(
        processed_dir / "papers.jsonl",
        processed_dir / "paper_index.json"
    )

    print("Papers indexed:", paper_index.num_docs)
    print("Unique terms in paper index:", len(paper_index.vocabulary()))
    print("Average paper length:", paper_index.avg_doc_length)

    del paper_index

    # Build the paragraph index from the preprocessed collection.
    print("Building paragraph index...")
    paragraph_index = build_and_save_index(
        processed_dir / "paragraphs.jsonl",
        processed_dir / "paragraph_index.json"
    )

    print("Paragraphs indexed:", paragraph_index.num_docs)
    print("Unique terms in paragraph index:", len(paragraph_index.vocabulary()))
    print("Average paragraph length:", paragraph_index.avg_doc_length)