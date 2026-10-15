import os
import re
import json
from typing import List, Dict, Any, Optional


def split_section_into_legal_blocks(text: str) -> List[str]:
    """
    Splits section text into coherent structural legal blocks based on legal boundaries:
    - Subsections: (1), (2), (3)...
    - Definitions: (1), (2), (3)...
    - Explanations: Explanation, Explanation 1, Explanation 2...
    - Illustrations: Illustration, Illustrations...
    - Provisos: Provided that, Provided further that...

    Preserves subsection/clause integrity without cutting inside sentences.
    """
    lines = text.split('\n')
    blocks = []
    current_block = []

    # Regex for top-level structural legal boundaries
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


def chunk_section(
    sec_record: Dict[str, Any],
    max_chunk_chars: int = 1800,
    prefix_id: str = "BNS"
) -> List[Dict[str, Any]]:
    """
    Chunks a single legal section into one or more structured chunk records.

    If section text <= max_chunk_chars, keeps it as a single chunk.
    If section text > max_chunk_chars, splits at structural legal boundaries.
    """
    raw_sec_num = sec_record['section']
    try:
        sec_num_int = int(re.sub(r'[A-Za-z]+$', '', str(raw_sec_num)))
    except ValueError:
        sec_num_int = 0

    sec_title = sec_record.get('section_title', '')
    chapter = sec_record.get('chapter', '')
    page_start = sec_record.get('page_start', 0)
    page_end = sec_record.get('page_end', 0)
    full_text = sec_record.get('text', '').strip()
    act = sec_record.get('act', '')
    source = sec_record.get('source', '')
    doc_id = sec_record.get('document_id', prefix_id)

    if not full_text:
        return []

    # Single chunk case
    if len(full_text) <= max_chunk_chars:
        chunk_id = f"{prefix_id}-S{sec_num_int:03d}-C001"
        return [{
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "act": act,
            "section_number": sec_num_int,
            "section_title": sec_title,
            "chapter": chapter,
            "page_start": page_start,
            "page_end": page_end,
            "chunk_index": 0,
            "source": source,
            "text": full_text
        }]

    # Multi-chunk case for long sections
    blocks = split_section_into_legal_blocks(full_text)
    chunks = []
    chunk_idx = 0
    current_blocks = []
    current_len = 0

    for b in blocks:
        b_len = len(b)
        if current_len + b_len > max_chunk_chars and current_blocks:
            c_text = '\n'.join(current_blocks).strip()
            if c_text:
                c_id = f"{prefix_id}-S{sec_num_int:03d}-C{chunk_idx + 1:03d}"
                chunks.append({
                    "chunk_id": c_id,
                    "document_id": doc_id,
                    "act": act,
                    "section_number": sec_num_int,
                    "section_title": sec_title,
                    "chapter": chapter,
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_index": chunk_idx,
                    "source": source,
                    "text": c_text
                })
                chunk_idx += 1
            current_blocks = [b]
            current_len = b_len
        else:
            current_blocks.append(b)
            current_len += b_len

    if current_blocks:
        c_text = '\n'.join(current_blocks).strip()
        if c_text:
            c_id = f"{prefix_id}-S{sec_num_int:03d}-C{chunk_idx + 1:03d}"
            chunks.append({
                "chunk_id": c_id,
                "document_id": doc_id,
                "act": act,
                "section_number": sec_num_int,
                "section_title": sec_title,
                "chapter": chapter,
                "page_start": page_start,
                "page_end": page_end,
                "chunk_index": chunk_idx,
                "source": source,
                "text": c_text
            })

    return chunks


def create_bns_chunks(
    sections_path: str = "data/processed/bns_sections.json",
    output_path: Optional[str] = "data/processed/bns_chunks.json",
    max_chunk_chars: int = 1800,
    prefix_id: str = "BNS"
) -> List[Dict[str, Any]]:
    """
    Loads section records from bns_sections.json, creates legal-aware chunks,
    saves the output to bns_chunks.json, and returns the chunk list.
    """
    if not os.path.exists(sections_path):
        raise FileNotFoundError(f"Processed sections file not found at {sections_path}")

    with open(sections_path, 'r', encoding='utf-8') as f:
        sections = json.load(f)

    all_chunks = []
    for sec in sections:
        sec_chunks = chunk_section(sec, max_chunk_chars=max_chunk_chars, prefix_id=prefix_id)
        all_chunks.extend(sec_chunks)

    # Save output if output_path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=4, ensure_ascii=False)

    return all_chunks


if __name__ == "__main__":
    chunks = create_bns_chunks()
    print(f"[Chunker] Generated {len(chunks)} chunks into data/processed/bns_chunks.json")
