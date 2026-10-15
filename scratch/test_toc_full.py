import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import re
from ingestion.loader import load_pdf

pages = load_pdf('data/raw/bns_2023.pdf')

toc_text = '\n'.join([pages[p-1]['text'] for p in range(1, 14)])
lines = [l.strip() for l in toc_text.split('\n') if l.strip()]

toc_map = {}
i = 0
while i < len(lines):
    line = lines[i]
    m = re.match(r'^(\d+)\.\s*(.*)', line)
    if m:
        sec_num = int(m.group(1))
        title_parts = [m.group(2).strip()] if m.group(2).strip() else []
        j = i + 1
        while j < len(lines):
            next_l = lines[j]
            # Stop if next line is a new section number like "10." or "10. Title"
            if re.match(r'^\d+\.\s*', next_l) or next_l.startswith('CHAPTER') or next_l.startswith('CLAUSES') or next_l.startswith('Of ') or re.match(r'^\([ivx]+\)$', next_l):
                break
            title_parts.append(next_l)
            j += 1
        title = ' '.join(title_parts).strip()
        # Clean up any trailing page numbers or marginal codes
        title = re.sub(r'\s+', ' ', title)
        toc_map[sec_num] = title
        i = j
    else:
        i += 1

print(f"Extracted {len(toc_map)} section titles from TOC.")
print("\nSample full titles:")
for s in [10, 50, 100, 150, 356]:
    print(f"Sec {s:3d}: {repr(toc_map.get(s))}")
