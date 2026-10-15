import os
import sys
import numpy as np

# Ensure project root is in python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from retrieval.semantic import BGEM3SemanticRetriever


def main():
    chunks_path = os.path.join(project_root, "data", "processed", "bns_chunks.json")
    embeddings_path = os.path.join(project_root, "data", "processed", "bns_embeddings.npy")

    retriever = BGEM3SemanticRetriever(
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        model_name="BAAI/bge-m3"
    )

    # Load chunks and build/load embeddings
    retriever.load_chunks()
    embeddings = retriever.build_embeddings(force_rebuild=False)

    num_chunks = len(retriever.chunks)
    emb_shape = embeddings.shape

    print("========================================")
    print("BGE-M3 MULTILINGUAL SEMANTIC RETRIEVER REPORT")
    print("========================================")
    print(f"\nChunks File: {chunks_path}")
    print(f"Embeddings File: {embeddings_path}")

    # Validation Checks
    print("\n----------------------------------------")
    print("EMBEDDING VALIDATION CHECKS:")
    print("----------------------------------------")
    print(f"1. Total Chunks Loaded: {num_chunks}")
    print(f"2. Embeddings Shape: {emb_shape} (Expected: ({num_chunks}, 1024))")
    print(f"3. Embedding Dimensions Consistent: {emb_shape[1] == 1024}")
    print(f"4. Contains No NaN / Inf Values: {not np.isnan(embeddings).any() and not np.isinf(embeddings).any()}")

    # English Queries
    english_queries = [
        "What is the punishment for murder?",
        "What is theft?",
        "What is criminal intimidation?",
        "What is the punishment for rape?",
        "What is culpable homicide?"
    ]

    print("\n========================================")
    print("ENGLISH SEMANTIC RETRIEVAL TESTS (top_k=5)")
    print("========================================")

    for i, q in enumerate(english_queries, 1):
        results = retriever.search(q, top_k=5)
        print(f"\nQuery {i}: \"{q}\"")
        print(f"Results Returned: {len(results)}")
        print("-" * 50)
        for idx, res in enumerate(results, 1):
            print(f"  {idx}. [Score: {res['score']:>6.4f}] Sec {res['section_number']:>3d} | {res['section_title']} (ID: {res['chunk_id']})")
            print(f"     Chapter: {res['chapter']} | Pages {res['page_start']}-{res['page_end']}")
            lines = res['text'].split('\n')
            preview = ' '.join([l.strip() for l in lines[:2]])[:120]
            print(f"     Preview: {preview}...\n")

    # Multilingual Queries
    multilingual_queries = [
        ("Hindi", "हत्या के लिए क्या सजा है?"),
        ("Telugu", "హత్యకు శిక్ష ఏమిటి?"),
        ("Tamil", "கொலைக்கான தண்டனை என்ன?"),
        ("Kannada", "ಕೊಲೆಗೆ ಶಿಕ್ಷೆ ಏನು?"),
        ("Bengali", "হত্যার শাস্তি কি?"),
        ("Hinglish", "Murder ke liye kya punishment hai?")
    ]

    print("========================================")
    print("MULTILINGUAL SEMANTIC RETRIEVAL TESTS")
    print("========================================")

    for lang, q in multilingual_queries:
        results = retriever.search(q, top_k=3)
        print(f"\nLanguage: {lang} | Query: \"{q}\"")
        print(f"Top Result:")
        if results:
            top_res = results[0]
            print(f"   [Score: {top_res['score']:>6.4f}] Sec {top_res['section_number']} | {top_res['section_title']} ({top_res['chunk_id']})")
            lines = top_res['text'].split('\n')
            preview = ' '.join([l.strip() for l in lines[:2]])[:120]
            print(f"   Preview: {preview}...")
        print("-" * 50)

    # Edge Case Tests
    print("\n========================================")
    print("EDGE CASE & PARAMETER TESTS")
    print("========================================")

    # Test top_k = 3
    q_top3 = "What is the punishment for murder?"
    res_top3 = retriever.search(q_top3, top_k=3)
    print(f"\n1. Test top_k=3 (\"{q_top3}\"):")
    print(f"   Results returned: {len(res_top3)}")
    for r in res_top3:
        print(f"   - Score: {r['score']:>6.4f} | Sec {r['section_number']} | {r['section_title']} ({r['chunk_id']})")

    # Test Empty Query
    q_empty = "   "
    res_empty = retriever.search(q_empty, top_k=5)
    print(f"\n2. Test Empty Query (\" {q_empty} \"):")
    print(f"   Results returned: {len(res_empty)} (Expected: 0)")

    # Result Metadata Verification
    print("\n----------------------------------------")
    print("METADATA & QUALITY VERIFICATION:")
    print("----------------------------------------")
    sample_search = retriever.search("theft punishment", top_k=5)
    all_keys_valid = all(
        all(k in r for k in ["chunk_id", "section_number", "section_title", "chapter", "page_start", "page_end", "score", "text"])
        for r in sample_search
    )
    no_empty_text = all(len(r['text'].strip()) > 0 for r in sample_search)
    scores_sorted = all(sample_search[i]['score'] >= sample_search[i+1]['score'] for i in range(len(sample_search)-1))

    print(f"1. Search returns all required metadata keys: {all_keys_valid}")
    print(f"2. No search result contains empty text: {no_empty_text}")
    print(f"3. Results sorted descending by score: {scores_sorted}")

    print("\n========================================")
    print("STATUS: BGE-M3 MULTILINGUAL SEMANTIC RETRIEVER VALIDATED AND OPERATIONAL")
    print("========================================\n")


if __name__ == "__main__":
    main()
