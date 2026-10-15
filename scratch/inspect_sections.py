import json

with open('data/processed/all_legal_sections.json', 'r', encoding='utf-8') as f:
    sections = json.load(f)

print(f"Total sections: {len(sections)}")

# Find sections for BNS
bns_secs = [s for s in sections if 'bns' in s.get('document_id', '').lower() or 'bns' in s.get('act', '').lower()]
print(f"BNS sections count: {len(bns_secs)}")

# Check section 105 and 358 in BNS
for s in bns_secs:
    if str(s.get('section')) in ['105', '358']:
        print(f"Sec {s.get('section')}: Pages {s.get('page_start')}-{s.get('page_end')}, Title: {s.get('section_title')}, Text len: {len(s.get('text', ''))}")
        print("Text preview:", repr(s.get('text', '')[:200]))

# Check sections with excessively broad page ranges across all Acts (> 10 pages)
broad_secs = [s for s in sections if (s.get('page_end', 0) - s.get('page_start', 0)) > 10]
print(f"Total sections with page range > 10 pages: {len(broad_secs)}")
for bs in broad_secs[:15]:
    print(f"  {bs.get('act')} Sec {bs.get('section')}: Pages {bs.get('page_start')}-{bs.get('page_end')} (text len={len(bs.get('text', ''))})")
