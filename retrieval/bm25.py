import os
import re
import json
from typing import List, Dict, Any, Optional

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank-bm25 package is required. Install it with: py -m pip install rank-bm25")

from retrieval.act_matcher import matches_act_filter

# Standard stopwords to filter out low-information English tokens
DEFAULT_STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'into',
    'that', 'this', 'these', 'those', 'and', 'or', 'not', 'it', 'its',
    'shall', 'may', 'any', 'which', 'who', 'whom', 'whose', 'where', 'when'
}


def tokenize_legal_text(text: str, stopwords: set = DEFAULT_STOPWORDS) -> List[str]:
    """
    Tokenizes legal text for BM25 indexing and search:
    - Normalizes to lowercase.
    - Extracts alphanumeric words and section numbers.
    - Filters out standard non-informative stopwords.
    """
    if not text:
        return []
    words = re.findall(r'\b[a-z0-9]+\b', text.lower())
    return [w for w in words if w not in stopwords and len(w) > 1]


class BM25Retriever:
    """
    Reusable BM25 Lexical Retriever for VidhiSetu legal chunks.
    """

    def __init__(self, chunks_path: str = "data/processed/bns_chunks.json"):
        self.chunks_path = chunks_path
        self.chunks: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.is_indexed: bool = False

    def build_index(self) -> int:
        """
        Loads bns_chunks.json and builds the BM25 index over the chunks.
        Returns the number of indexed chunks.
        """
        if not os.path.exists(self.chunks_path):
            raise FileNotFoundError(f"Chunks file not found at: {self.chunks_path}")

        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        # Build tokenized corpus including chapter, section title, and chunk text for indexing
        self.corpus_tokens = []
        for c in self.chunks:
            indexable_text = f"{c.get('chapter', '')} {c.get('section_title', '')} {c.get('text', '')}"
            tokens = tokenize_legal_text(indexable_text)
            self.corpus_tokens.append(tokens)

        self.bm25 = BM25Okapi(self.corpus_tokens)
        self.is_indexed = True

        return len(self.chunks)

    def search(self, query: str, top_k: int = 5, act_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Searches the BM25 index for the top_k most relevant chunks for the given query.
        Optionally filters candidates by act_filter.
        """
        if not self.is_indexed or self.bm25 is None:
            self.build_index()

        # Handle empty query safely
        if not query or not query.strip():
            return []

        query_tokens = tokenize_legal_text(query)
        if not query_tokens:
            return []

        # Compute BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Filter and rank indices with positive scores
        scored_indices = [(idx, float(scores[idx])) for idx in range(len(scores)) if scores[idx] > 0.0]
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        if act_filter:
            scored_indices = [
                (idx, s) for idx, s in scored_indices
                if matches_act_filter(
                    act_filter,
                    self.chunks[idx].get("act", ""),
                    self.chunks[idx].get("document_id", ""),
                    self.chunks[idx].get("source", ""),
                    self.chunks[idx].get("chunk_id", "")
                )
            ]

        results = []
        for idx, score in scored_indices[:top_k]:
            chunk = self.chunks[idx]
            result = {
                "chunk_id": chunk.get("chunk_id", ""),
                "document_id": chunk.get("document_id", ""),
                "act": chunk.get("act", ""),
                "section_number": chunk.get("section_number", 0),
                "section_title": chunk.get("section_title", ""),
                "chapter": chunk.get("chapter", ""),
                "page_start": chunk.get("page_start", 0),
                "page_end": chunk.get("page_end", 0),
                "source": chunk.get("source", ""),
                "score": round(score, 4),
                "text": chunk.get("text", "")
            }
            results.append(result)

        return results


if __name__ == "__main__":
    retriever = BM25Retriever()
    count = retriever.build_index()
    print(f"[BM25Retriever] Indexed {count} chunks successfully.")
    sample_res = retriever.search("murder punishment", top_k=3)
    print(f"Sample search results count: {len(sample_res)}")
