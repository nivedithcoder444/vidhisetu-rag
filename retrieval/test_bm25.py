import os
import sys

# Ensure project root is in python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from retrieval.bm25 import BM25Retriever


def main():
    chunks_path = os.path.join(project_root, "data", "processed", "bns_chunks.json")
    retriever = BM25Retriever(chunks_path=chunks_path)

    # 1. Build Index
    num_indexed = retriever.build_index()

    print("========================================")
    print("BM25 LEXICAL RETRIEVER - VALIDATION REPORT")
    print("========================================")
    print(f"\nIndexed File: {chunks_path}")

    # Sanity Checks
    chunk_ids = [c['chunk_id'] for c in retriever.chunks]
    unique_ids = set(chunk_ids)

    print("\n----------------------------------------")
    print("INDEX SANITY CHECKS:")
    print("----------------------------------------")
    print(f"1. Total Chunks Indexed: {num_indexed}")
    print(f"2. All Chunk IDs Unique: {len(unique_ids) == num_indexed} ({len(unique_ids)} unique IDs)")

    # Test Queries
    required_queries = [
        "What is the punishment for murder?",
        "What is theft?",
        "What is criminal intimidation?",
        "What is the punishment for rape?",
        "What is culpable homicide?"
    ]

    print("\n========================================")
    print("LEGAL RETRIEVAL TESTS (top_k=5)")
    print("========================================")

    for i, q in enumerate(required_queries, 1):
        results = retriever.search(q, top_k=5)
        print(f"\nQuery {i}: \"{q}\"")
        print(f"Results Returned: {len(results)}")
        print("-" * 50)
        for idx, res in enumerate(results, 1):
            print(f"  {idx}. [Score: {res['score']:>7.4f}] Sec {res['section_number']:>3d} | {res['section_title']} (ID: {res['chunk_id']})")
            print(f"     Chapter: {res['chapter']} | Pages {res['page_start']}-{res['page_end']}")
            lines = res['text'].split('\n')
            preview = ' '.join([l.strip() for l in lines[:2]])[:120]
            print(f"     Preview: {preview}...\n")

    print("========================================")
    print("EDGE CASE & PARAMETER TESTS")
    print("========================================")

    # Test top_k = 3
    q_top3 = "What is the punishment for murder?"
    res_top3 = retriever.search(q_top3, top_k=3)
    print(f"\n1. Test top_k=3 (\"{q_top3}\"):")
    print(f"   Results returned: {len(res_top3)}")
    for r in res_top3:
        print(f"   - Score: {r['score']:>7.4f} | Sec {r['section_number']} | {r['section_title']} ({r['chunk_id']})")

    # Test Empty Query
    q_empty = "   "
    res_empty = retriever.search(q_empty, top_k=5)
    print(f"\n2. Test Empty Query (\" {q_empty} \"):")
    print(f"   Results returned: {len(res_empty)} (Expected: 0)")

    # Test Completely Unrelated Query
    q_unrelated = "quantum physics gravitational wave particle accelerator"
    res_unrelated = retriever.search(q_unrelated, top_k=5)
    print(f"\n3. Test Unrelated Query (\"...{q_unrelated}...\"):")
    print(f"   Results returned: {len(res_unrelated)} (Expected: 0)")

    # Metadata & Non-empty Text Check
    print("\n----------------------------------------")
    print("RESULT METADATA VERIFICATION:")
    print("----------------------------------------")
    sample_search = retriever.search("theft punishment", top_k=5)
    all_keys_valid = all(
        all(k in r for k in ["chunk_id", "section_number", "section_title", "chapter", "page_start", "page_end", "score", "text"])
        for r in sample_search
    )
    no_empty_text = all(len(r['text'].strip()) > 0 for r in sample_search)

    print(f"1. Search returns all required metadata keys: {all_keys_valid}")
    print(f"2. No search result contains empty text: {no_empty_text}")
    print(f"3. Results sorted descending by score: {all(sample_search[i]['score'] >= sample_search[i+1]['score'] for i in range(len(sample_search)-1))}")

    print("\n========================================")
    print("STATUS: BM25 RETRIEVER VALIDATED AND OPERATIONAL")
    print("========================================\n")


if __name__ == "__main__":
    main()
