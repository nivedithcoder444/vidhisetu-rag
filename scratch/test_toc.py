import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import re
from ingestion.loader import load_pdf

pages = load_pdf('data/raw/bns_2023.pdf')

# 1. Parse TOC from pages 1 to 13
toc_text = '\n'.join([pages[p-1]['text'] for p in range(1, 14)])
lines = [l.strip() for l in toc_text.split('\n') if l.strip()]

toc_map = {}
for i, line in enumerate(lines):
    m = re.match(r'^(\d+)\.\s*(.*)', line)
    if m:
        sec_num = int(m.group(1))
        title_part = m.group(2).strip()
        # If title_part is empty, look at next line
        if not title_part and i + 1 < len(lines):
            title_part = lines[i+1]
        
        # Clean up title if it contains next section numbers
        if title_part:
            # truncate at next number or CLAUSES if any accidentally concatenated
            title_part = re.split(r'\s+\d+\.\s+', title_part)[0]
            toc_map[sec_num] = title_part

print(f"Extracted {len(toc_map)} section titles from TOC.")
for s in [1, 2, 3, 4, 5, 10, 50, 100, 150, 200, 250, 300, 356]:
    print(f"Sec {s:3d}: {toc_map.get(s)}")
