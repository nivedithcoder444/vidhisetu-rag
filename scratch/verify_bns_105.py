import json
import sys
sys.path.insert(0, ".")
from retrieval.hybrid import HybridRetriever

sys.stdout.reconfigure(encoding='utf-8')
r = HybridRetriever()
r.initialize()
res = r.search('Section 105 BNS', top_k=3)
for i, chunk in enumerate(res, 1):
    print(f'=== RESULT {i} ===')
    print('Chunk ID:', chunk.get('chunk_id'))
    print('Act:', chunk.get('act'))
    print('Section:', chunk.get('section_number'), '-', chunk.get('section_title'))
    print(f'Pages: {chunk.get("page_start")} to {chunk.get("page_end")}')
    print('RRF Score:', chunk.get('rrf_score'))
    print('Exact Match:', chunk.get('exact_match'))
    print('Text snippet:', chunk.get('text', '')[:300].replace('\n', ' '))
