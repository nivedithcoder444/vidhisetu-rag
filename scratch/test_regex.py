import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import re
from ingestion.loader import load_pdf

pages = load_pdf('data/raw/bns_2023.pdf')

sec_header_re = re.compile(r'^\s*(\d+)\.\s*([A-Za-z\(\'\"])', re.MULTILINE)

matches = []
for p in range(14, 116):
    text = pages[p-1]['text']
    for m in sec_header_re.finditer(text):
        sec_num = int(m.group(1))
        if sec_num > 356:
            continue
        matches.append({
            'page': p,
            'sec': sec_num,
            'pos': m.start(),
            'line': m.group(0).strip()
        })

seq_matches = []
expected = 1
for m in matches:
    if m['sec'] == expected:
        seq_matches.append(m)
        expected += 1
    elif m['sec'] < expected:
        pass
    else:
        print(f"Discontinuity! Expected {expected}, got {m['sec']} on page {m['page']}, line: {repr(m['line'][:50])}")

print(f"Sequential sections matched: {len(seq_matches)} / 356")
