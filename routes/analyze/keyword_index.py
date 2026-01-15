"""
Keyword Index - Simple keyword-based search for cached documents.

Uses TF-IDF-like scoring for keyword relevance.
"""

import re
import json
import math
from pathlib import Path
from collections import Counter
from typing import Optional

# Stop words to ignore
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'this', 'that', 'these', 'those', 'it', 'its', 'i', 'you', 'he',
    'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
    'his', 'her', 'our', 'their', 'what', 'which', 'who', 'whom', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here',
}


def tokenize(text: str) -> list[str]:
    """Split text into tokens, lowercase, remove punctuation."""
    # Lowercase and split on non-alphanumeric
    tokens = re.findall(r'\b[a-z]{2,}\b', text.lower())
    # Remove stop words
    return [t for t in tokens if t not in STOP_WORDS]


def extract_keywords(text: str, top_n: int = 50) -> list[tuple[str, int]]:
    """Extract top keywords from text by frequency."""
    tokens = tokenize(text)
    counts = Counter(tokens)
    return counts.most_common(top_n)


def extract_ngrams(text: str, n: int = 2, top_n: int = 30) -> list[tuple[str, int]]:
    """Extract top n-grams (phrases) from text."""
    tokens = tokenize(text)
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i+n])
        ngrams.append(ngram)
    counts = Counter(ngrams)
    return counts.most_common(top_n)


class KeywordIndex:
    """
    Simple inverted index for keyword search.

    Structure:
    {
        "documents": {
            "doc_id": {
                "title": "...",
                "keywords": {"word": count, ...},
                "total_tokens": N
            }
        },
        "inverted": {
            "keyword": ["doc_id1", "doc_id2", ...]
        },
        "doc_count": N
    }
    """

    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.index = self._load()

    def _load(self) -> dict:
        """Load index from disk."""
        if self.index_path.exists():
            with open(self.index_path) as f:
                return json.load(f)
        return {'documents': {}, 'inverted': {}, 'doc_count': 0}

    def _save(self):
        """Save index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f)

    def add_document(self, doc_id: str, title: str, content: str):
        """Add a document to the index."""
        # Extract keywords
        keywords = dict(extract_keywords(content, top_n=100))
        total_tokens = len(tokenize(content))

        # Store document info
        self.index['documents'][doc_id] = {
            'title': title,
            'keywords': keywords,
            'total_tokens': total_tokens
        }

        # Update inverted index
        for keyword in keywords:
            if keyword not in self.index['inverted']:
                self.index['inverted'][keyword] = []
            if doc_id not in self.index['inverted'][keyword]:
                self.index['inverted'][keyword].append(doc_id)

        self.index['doc_count'] = len(self.index['documents'])
        self._save()

    def search(self, query: str, top_n: int = 10) -> list[tuple[str, float]]:
        """
        Search for documents matching query.
        Returns list of (doc_id, score) tuples.
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Score each document
        scores = {}
        for token in query_tokens:
            if token not in self.index['inverted']:
                continue

            # IDF: log(N / df) where df = docs containing term
            df = len(self.index['inverted'][token])
            idf = math.log(self.index['doc_count'] / df) if df > 0 else 0

            for doc_id in self.index['inverted'][token]:
                doc_info = self.index['documents'].get(doc_id, {})
                # TF: term frequency in document
                tf = doc_info.get('keywords', {}).get(token, 0)
                # Normalize by doc length
                total = doc_info.get('total_tokens', 1)
                tf_norm = tf / total if total > 0 else 0

                # TF-IDF score
                score = tf_norm * idf
                scores[doc_id] = scores.get(doc_id, 0) + score

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def get_related_keywords(self, keyword: str, top_n: int = 10) -> list[str]:
        """Find keywords that often appear with given keyword."""
        if keyword not in self.index['inverted']:
            return []

        # Get docs containing keyword
        doc_ids = self.index['inverted'][keyword]

        # Count co-occurring keywords
        cooccur = Counter()
        for doc_id in doc_ids:
            doc_info = self.index['documents'].get(doc_id, {})
            for kw in doc_info.get('keywords', {}):
                if kw != keyword:
                    cooccur[kw] += 1

        return [kw for kw, _ in cooccur.most_common(top_n)]

    def get_document_keywords(self, doc_id: str) -> list[str]:
        """Get keywords for a specific document."""
        doc_info = self.index['documents'].get(doc_id, {})
        keywords = doc_info.get('keywords', {})
        # Return sorted by frequency
        return [kw for kw, _ in sorted(keywords.items(), key=lambda x: x[1], reverse=True)]


def get_project_index(project_name: Optional[str] = None) -> KeywordIndex:
    """Get keyword index for a project."""
    from .research import get_cache_dir
    cache_dir = get_cache_dir(project_name)
    index_path = cache_dir / "_keywords.json"
    return KeywordIndex(index_path)
