import json
import re
from rank_bm25 import BM25Okapi

with open('data/processed/bns_chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'into', 'that', 'this', 'these', 'those', 'and', 'or', 'not', 'it', 'its', 'shall', 'may', 'any'}

def tokenize(text):
    words = re.findall(r'\b[a-z0-9]+\b', text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]

# Include chapter, section_title, and text in corpus indexing for rich lexical matching
corpus = [tokenize(f"{c.get('chapter', '')} {c.get('section_title', '')} {c.get('text', '')}") for c in chunks]
bm25 = BM25Okapi(corpus)

queries = [
    "What is the punishment for murder?",
    "What is theft?",
    "What is criminal intimidation?",
    "What is the punishment for rape?",
    "What is culpable homicide?"
]

for q in queries:
    q_tokens = tokenize(q)
    scores = bm25.get_scores(q_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    print(f"=== QUERY: {q} ===")
    for idx in top_indices:
        c = chunks[idx]
        print(f"Score: {scores[idx]:.4f} | Sec {c['section_number']:>3d} | Title: {c['section_title']} (ID: {c['chunk_id']})")
    print()
