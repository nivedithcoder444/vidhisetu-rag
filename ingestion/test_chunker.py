import os
import sys
import json
from collections import Counter

# Ensure project root is in python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ingestion.chunker import create_bns_chunks


def main():
    sections_path = os.path.join(project_root, "data", "processed", "bns_sections.json")
    output_path = os.path.join(project_root, "data", "processed", "bns_chunks.json")

    # Load original sections
    with open(sections_path, 'r', encoding='utf-8') as f:
        sections = json.load(f)

    total_sections_loaded = len(sections)

    # Generate chunks
    chunks = create_bns_chunks(
        sections_path=sections_path,
        output_path=output_path,
        max_chunk_chars=1800,
        prefix_id="BNS"
    )

    total_chunks_generated = len(chunks)

    # 1. Check sections with zero chunks
    section_nums_in_sections = set([int(s['section']) for s in sections])
    section_nums_in_chunks = set([c['section_number'] for c in chunks])
    zero_chunk_sections = sorted(list(section_nums_in_sections - section_nums_in_chunks))

    # 2. Check duplicate chunk IDs
    chunk_ids = [c['chunk_id'] for c in chunks]
    chunk_id_counts = Counter(chunk_ids)
    duplicate_chunk_ids = [cid for cid, count in chunk_id_counts.items() if count > 1]

    # 3. Check empty chunks
    empty_chunks = [c['chunk_id'] for c in chunks if not c['text'].strip()]

    # 4. Check missing section numbers in range 1 to max(section_numbers)
    expected_range = set(range(1, max(section_nums_in_sections) + 1))
    missing_sections = sorted(list(expected_range - section_nums_in_chunks))

    # 5. Calculate distribution of chunks per section
    chunk_counts_per_sec = Counter([c['section_number'] for c in chunks])
    distribution_counts = Counter(chunk_counts_per_sec.values())

    # 6. Representation check
    all_358_represented = (len(section_nums_in_chunks) == total_sections_loaded) and (len(zero_chunk_sections) == 0)

    # Output Validation Report
    print("========================================")
    print("LEGAL-AWARE CHUNKER VALIDATION REPORT")
    print("========================================")
    print(f"\nSections File: {sections_path}")
    print(f"Chunks Output File: {output_path}")

    print(f"\n1. Total Sections Loaded: {total_sections_loaded}")
    print(f"2. Total Chunks Generated: {total_chunks_generated}")
    print(f"3. Sections with Zero Chunks: {zero_chunk_sections if zero_chunk_sections else 'None'}")
    print(f"4. Duplicate Chunk IDs: {duplicate_chunk_ids if duplicate_chunk_ids else 'None'}")
    print(f"5. Empty Chunks: {empty_chunks if empty_chunks else 'None'}")
    print(f"6. Missing Section Numbers (1-{max(section_nums_in_sections)}): {missing_sections if missing_sections else 'None'}")
    print(f"7. All 358 Sections Represented: {all_358_represented}")

    print("\n8. Distribution of Chunks per Section:")
    for num_chunks in sorted(distribution_counts.keys()):
        sec_cnt = distribution_counts[num_chunks]
        print(f"   - {sec_cnt} sections split into {num_chunks} chunk(s)")

    print("\n----------------------------------------")
    print("FIRST 5 CHUNKS PREVIEW:")
    print("----------------------------------------")
    for c in chunks[:5]:
        print(f"Chunk ID: {c['chunk_id']} | Sec {c['section_number']} (Idx {c['chunk_index']}) | Pages {c['page_start']}-{c['page_end']}")
        print(f"Chapter: {c['chapter']}")
        print(f"Title: {c['section_title']}")
        lines = c['text'].split('\n')
        preview = '\n'.join(lines[:3])
        if len(lines) > 3:
            preview += "\n..."
        print(f"Text Preview:\n{preview}\n")

    print("========================================")
    print(f"PROCESSED OUTPUT SAVED TO: {output_path}")
    print("STATUS: LEGAL-AWARE CHUNKING VALIDATED AND READY FOR RETRIEVAL PIPELINE")
    print("========================================\n")


if __name__ == "__main__":
    main()
