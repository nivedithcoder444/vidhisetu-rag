import json
import re

with open('data/processed/bns_sections.json', 'r', encoding='utf-8') as f:
    sections = json.load(f)

def split_section_into_legal_blocks(text: str):
    """Splits section text into structural legal blocks (subsections, definitions, etc.)"""
    lines = text.split('\n')
    blocks = []
    current_block = []

    # Block header patterns: (1), (2), Explanation, Illustration, Provided that
    block_start_re = re.compile(
        r'^\s*(\(\d+\)|Explanation(?:\s+\d+)?\.|Illustrations?\.|Provided\s+that)',
        re.IGNORECASE
    )

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if block_start_re.match(s) and current_block:
            blocks.append('\n'.join(current_block))
            current_block = [line]
        else:
            current_block.append(line)

    if current_block:
        blocks.append('\n'.join(current_block))

    return blocks

all_chunks = []
max_chars = 1800

for sec in sections:
    sec_num = int(sec['section'])
    sec_title = sec['section_title']
    chapter = sec['chapter']
    page_start = sec['page_start']
    page_end = sec['page_end']
    full_text = sec['text']

    if len(full_text) <= max_chars:
        # Keep section as a single chunk
        chunk_text = full_text
        all_chunks.append({
            "chunk_id": f"BNS-S{sec_num:03d}-C001",
            "section_number": sec_num,
            "section_title": sec_title,
            "chapter": chapter,
            "page_start": page_start,
            "page_end": page_end,
            "chunk_index": 0,
            "text": chunk_text
        })
    else:
        # Split into legal blocks and aggregate into chunks <= max_chars
        blocks = split_section_into_legal_blocks(full_text)
        chunk_idx = 0
        current_chunk_blocks = []
        current_len = 0

        for b in blocks:
            b_len = len(b)
            if current_len + b_len > max_chars and current_chunk_blocks:
                c_text = '\n'.join(current_chunk_blocks)
                all_chunks.append({
                    "chunk_id": f"BNS-S{sec_num:03d}-C{chunk_idx+1:03d}",
                    "section_number": sec_num,
                    "section_title": sec_title,
                    "chapter": chapter,
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_index": chunk_idx,
                    "text": c_text
                })
                chunk_idx += 1
                current_chunk_blocks = [b]
                current_len = b_len
            else:
                current_chunk_blocks.append(b)
                current_len += b_len

        if current_chunk_blocks:
            c_text = '\n'.join(current_chunk_blocks)
            all_chunks.append({
                "chunk_id": f"BNS-S{sec_num:03d}-C{chunk_idx+1:03d}",
                "section_number": sec_num,
                "section_title": sec_title,
                "chapter": chapter,
                "page_start": page_start,
                "page_end": page_end,
                "chunk_index": chunk_idx,
                "text": c_text
            })

print(f"Loaded {len(sections)} sections.")
print(f"Generated {len(all_chunks)} chunks.")

# Distribution of chunks per section
sec_chunk_counts = {}
for c in all_chunks:
    s = c['section_number']
    sec_chunk_counts[s] = sec_chunk_counts.get(s, 0) + 1

multi_chunk_secs = {s: cnt for s, cnt in sec_chunk_counts.items() if cnt > 1}
print(f"Sections with 1 chunk: {len(sections) - len(multi_chunk_secs)}")
print(f"Sections with >1 chunks: {len(multi_chunk_secs)}")
print("Sample multi-chunk sections:")
for s in sorted(multi_chunk_secs.keys())[:10]:
    print(f"  Sec {s}: {multi_chunk_secs[s]} chunks")
