import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import re
from ingestion.loader import load_pdf

pages = load_pdf('data/raw/bns_2023.pdf')

# Test chapter and section tracking
chapter_re = re.compile(r'^\s*(CHAPTER\s+[I|V|X|L|C|D|M]+)\s*$', re.MULTILINE | re.IGNORECASE)

# Let's inspect chapter lines in pages 14 to 115
for p in range(14, 116):
    text = pages[p-1]['text']
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for idx, l in enumerate(lines):
        if re.match(r'^CHAPTER\s+[I|V|X|L|C|D|M]+$', l, re.IGNORECASE):
            ch_num = l
            ch_title = lines[idx+1] if idx+1 < len(lines) else ''
            print(f"Page {p:3d} | {ch_num} -> {ch_title}")
