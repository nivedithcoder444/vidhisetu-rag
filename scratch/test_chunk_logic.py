import json
import re

with open('data/processed/bns_sections.json', 'r', encoding='utf-8') as f:
    sections = json.load(f)

# Inspect section 2 definitions
sec2 = [s for s in sections if s['section'] == '2'][0]
print("Section 2 length:", len(sec2['text']))
lines2 = sec2['text'].split('\n')
defs = [l for l in lines2 if re.match(r'^\(\d+\)', l.strip())]
print(f"Section 2 has {len(defs)} definitions:")
for d in defs[:5]:
    print("  ", d[:60])
