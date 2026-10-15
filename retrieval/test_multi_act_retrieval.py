import os
import sys
import json

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from retrieval.bm25 import BM25Retriever
from retrieval.semantic import BGEM3SemanticRetriever


def run_retrieval_tests():
    chunks_path = "data/processed/all_legal_chunks.json"
    embeddings_path = "data/processed/all_legal_embeddings.npy"

    if not os.path.exists(chunks_path):
        print(f"Error: Chunks file not found at {chunks_path}")
        return

    print("==================================================")
    print("VIDHISETU MULTI-ACT RETRIEVAL TEST")
    print("==================================================")

    # 1. Initialize BM25 Lexical Retriever
    print("\n--- Initializing BM25 Retriever ---")
    bm25 = BM25Retriever(chunks_path=chunks_path)
    bm25_count = bm25.build_index()
    print(f"BM25 Index Built over {bm25_count} legal chunks.")

    # 2. Initialize BGE-M3 Semantic Retriever
    print("\n--- Initializing BGE-M3 Semantic Retriever ---")
    semantic = BGEM3SemanticRetriever(
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        model_name="BAAI/bge-m3"
    )
    semantic.build_embeddings(force_rebuild=False)
    print("BGE-M3 Semantic Retriever Ready.")

    test_queries = [
        ("Code of Civil Procedure", "temporary injunction procedure balance of convenience"),
        ("Indian Contract Act", "breach of contract compensation for loss or damage"),
        ("Indian Penal Code", "punishment for murder and culpable homicide"),
        ("Arms Act", "licence for acquisition and possession of firearms")
    ]

    for category, query in test_queries:
        print("\n" + "=" * 60)
        print(f"QUERY [{category}]: \"{query}\"")
        print("=" * 60)

        # Execute BM25 search
        bm25_results = bm25.search(query, top_k=2)
        print("\n--- BM25 Lexical Top Match ---")
        if bm25_results:
            top_bm = bm25_results[0]
            print(f"Act Name:       {top_bm.get('act', 'Unknown Act')}")
            print(f"Section Number: Section {top_bm.get('section_number', 'N/A')}")
            print(f"Section Title:  {top_bm.get('section_title', '')}")
            print(f"Source File:    {top_bm.get('source', '')}")
            print(f"Page Number:    Pages {top_bm.get('page_start', 1)} - {top_bm.get('page_end', 1)}")
            print(f"BM25 Score:     {top_bm.get('score', 0.0)}")
            snippet = top_bm.get('text', '')[:200].replace('\n', ' ')
            print(f"Retrieved Text: {snippet}...")
        else:
            print("No BM25 results found.")

        # Execute BGE-M3 Semantic search
        sem_results = semantic.search(query, top_k=2)
        print("\n--- BGE-M3 Semantic Top Match ---")
        if sem_results:
            top_sem = sem_results[0]
            print(f"Act Name:       {top_sem.get('act', 'Unknown Act')}")
            print(f"Section Number: Section {top_sem.get('section_number', 'N/A')}")
            print(f"Section Title:  {top_sem.get('section_title', '')}")
            print(f"Source File:    {top_sem.get('source', '')}")
            print(f"Page Number:    Pages {top_sem.get('page_start', 1)} - {top_sem.get('page_end', 1)}")
            print(f"BGE-M3 Score:   {top_sem.get('score', 0.0)}")
            snippet = top_sem.get('text', '')[:200].replace('\n', ' ')
            print(f"Retrieved Text: {snippet}...")
        else:
            print("No BGE-M3 semantic results found.")


if __name__ == "__main__":
    run_retrieval_tests()
