import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple

from retrieval.bm25 import BM25Retriever
from retrieval.semantic import BGEM3SemanticRetriever
from retrieval.act_matcher import matches_act_filter


def parse_exact_section_query(query: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Parses natural language query to detect exact section lookups.
    Examples:
        'Section 73 of Indian Contract Act' -> (73, 'Indian Contract Act')
        'Section 3 Arms Act' -> (3, 'Arms Act')
        'Section 105 BNS' -> (105, 'BNS')
        'Section 302' -> (302, None)
    """
    if not query:
        return None, None

    # Pattern for section numbers and optional act hints
    pattern = re.compile(
        r'\b(?:Section|Sec\.|Article|Art\.)\s*(\d+[A-Z]?)\b(?:\s+(?:of|in)\b)?(?:\s+([A-Za-z0-9\s]+))?',
        re.IGNORECASE
    )
    match = pattern.search(query)
    if not match:
        return None, None

    sec_str = match.group(1)
    raw_act = match.group(2).strip() if match.group(2) else None

    try:
        sec_num = int(re.sub(r'[A-Z]$', '', sec_str))
    except ValueError:
        sec_num = None

    if raw_act:
        act_hint = re.sub(r'^\s*(?:of|in|the|under)\s+', '', raw_act, flags=re.IGNORECASE).strip()
        if not act_hint:
            act_hint = raw_act
    else:
        act_hint = None

    return sec_num, act_hint


class HybridRetriever:
    """
    Hybrid Retriever combining BM25 Lexical Search and BGE-M3 Semantic Search
    using Reciprocal Rank Fusion (RRF) with exact legal metadata matching.
    """

    def __init__(
        self,
        chunks_path: str = "data/processed/all_legal_chunks.json",
        embeddings_path: str = "data/processed/all_legal_embeddings.npy",
        model_name: str = "BAAI/bge-m3"
    ):
        self.chunks_path = chunks_path
        self.embeddings_path = embeddings_path
        self.model_name = model_name

        self.bm25_retriever = BM25Retriever(chunks_path=self.chunks_path)
        self.semantic_retriever = BGEM3SemanticRetriever(
            chunks_path=self.chunks_path,
            embeddings_path=self.embeddings_path,
            model_name=self.model_name
        )

        self.is_initialized: bool = False

    def initialize(self):
        """Builds or loads BM25 index and BGE-M3 embeddings."""
        self.bm25_retriever.build_index()
        self.semantic_retriever.build_embeddings(force_rebuild=False)
        self.is_initialized = True

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 30,
        rrf_k: float = 60.0,
        weights: Tuple[float, float] = (1.0, 1.0),
        exact_match_boost: float = 1.0,
        act_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs hybrid retrieval using Reciprocal Rank Fusion (RRF).

        Args:
            query: Natural language question or section lookup.
            top_k: Number of final results to return.
            candidate_k: Number of candidates to fetch from BM25 and Semantic search.
            rrf_k: Smoothing constant for RRF formula (default 60.0).
            weights: Weight tuple (bm25_weight, semantic_weight).
            exact_match_boost: Score boost for exact section metadata matches.
            act_filter: Optional filter restricting candidates to a specific Act.

        Returns:
            Ranked list of deduplicated chunk dictionaries with full metadata.
        """
        if not query or not query.strip():
            return []

        if not self.is_initialized:
            self.initialize()

        w_bm25, w_sem = weights

        # 1. Candidate Retrieval with consistent act_filter
        bm25_candidates = self.bm25_retriever.search(query, top_k=candidate_k, act_filter=act_filter)
        semantic_candidates = self.semantic_retriever.search(query, top_k=candidate_k, act_filter=act_filter)

        # 2. Check for exact section query
        target_sec, detected_act = parse_exact_section_query(query)
        effective_act = act_filter or detected_act

        # 3. Reciprocal Rank Fusion Map: chunk_id -> metadata & accumulated RRF score
        fused_scores: Dict[str, Dict[str, Any]] = {}

        # Process BM25 ranks
        for rank, item in enumerate(bm25_candidates, 1):
            c_id = item["chunk_id"]
            if c_id not in fused_scores:
                fused_scores[c_id] = {
                    "chunk": item,
                    "rrf_score": 0.0,
                    "bm25_rank": rank,
                    "semantic_rank": None,
                    "bm25_score": item.get("score", 0.0),
                    "semantic_score": 0.0,
                    "exact_match": False
                }
            fused_scores[c_id]["rrf_score"] += w_bm25 / (rrf_k + rank)
            fused_scores[c_id]["bm25_rank"] = rank

        # Process Semantic ranks
        for rank, item in enumerate(semantic_candidates, 1):
            c_id = item["chunk_id"]
            if c_id not in fused_scores:
                fused_scores[c_id] = {
                    "chunk": item,
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "bm25_score": 0.0,
                    "semantic_score": item.get("score", 0.0),
                    "exact_match": False
                }
            else:
                fused_scores[c_id]["semantic_rank"] = rank
                fused_scores[c_id]["semantic_score"] = item.get("score", 0.0)

            fused_scores[c_id]["rrf_score"] += w_sem / (rrf_k + rank)

        # 4. Apply Exact Section Metadata Boost & Metadata Lookup
        if target_sec is not None:
            for item in self.bm25_retriever.chunks:
                c_id = item["chunk_id"]
                c_sec = item.get("section_number", None)
                c_act = item.get("act", "").lower()
                c_src = item.get("source", "").lower()

                if c_sec == target_sec:
                    match_act = True
                    if effective_act:
                        match_act = matches_act_filter(
                            effective_act,
                            item.get("act", ""),
                            item.get("document_id", ""),
                            item.get("source", ""),
                            c_id
                        )

                    if match_act:
                        if c_id not in fused_scores:
                            fused_scores[c_id] = {
                                "chunk": item,
                                "rrf_score": 0.0,
                                "bm25_rank": None,
                                "semantic_rank": None,
                                "bm25_score": 0.0,
                                "semantic_score": 0.0,
                                "exact_match": True
                            }
                        fused_scores[c_id]["rrf_score"] += exact_match_boost
                        fused_scores[c_id]["exact_match"] = True

        # 5. If act_filter is provided, ensure all returned candidates match the filter
        if act_filter:
            fused_scores = {
                cid: entry for cid, entry in fused_scores.items()
                if matches_act_filter(
                    act_filter,
                    entry["chunk"].get("act", ""),
                    entry["chunk"].get("document_id", ""),
                    entry["chunk"].get("source", ""),
                    cid
                )
            }

        # 6. Sort fused results descending by RRF score
        sorted_chunks = sorted(fused_scores.values(), key=lambda x: x["rrf_score"], reverse=True)

        # 7. Format final results
        final_results = []
        for entry in sorted_chunks[:top_k]:
            chunk = entry["chunk"]
            res = {
                "chunk_id": chunk.get("chunk_id", ""),
                "document_id": chunk.get("document_id", ""),
                "act": chunk.get("act", ""),
                "section_number": chunk.get("section_number", 0),
                "section_title": chunk.get("section_title", ""),
                "chapter": chunk.get("chapter", ""),
                "page_start": chunk.get("page_start", 0),
                "page_end": chunk.get("page_end", 0),
                "source": chunk.get("source", ""),
                "rrf_score": round(entry["rrf_score"], 6),
                "bm25_rank": entry["bm25_rank"],
                "semantic_rank": entry["semantic_rank"],
                "exact_match": entry["exact_match"],
                "text": chunk.get("text", "")
            }
            final_results.append(res)

        return final_results


if __name__ == "__main__":
    retriever = HybridRetriever()
    retriever.initialize()
    res = retriever.search("Section 73 of Indian Contract Act", top_k=3)
    print(json.dumps(res, indent=2))
