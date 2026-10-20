
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

# Match Unicode letters and numbers, excluding underscores.
TOKEN_PATTERN = re.compile(r"[^\W_]+")
WORD_PATTERN = re.compile(r"\w+")


def load_config(config_path=DEFAULT_CONFIG_PATH):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}

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
        settings = config.get("tokenizer") or {}
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