import os
import sys
import re
import json
import pymupdf

sys.stdout.reconfigure(encoding='utf-8')
doc = pymupdf.open('data/raw/extracted/constitution.pdf')
pages = [{'page': p + 1, 'text': doc[p].get_text()} for p in range(len(doc))]

start_page_idx = 0
for i, p in enumerate(pages):
    if re.search(r'PART\s+I\b', p['text'], re.IGNORECASE) and re.search(r'\b1\.\s+Name and territory', p['text']):
        start_page_idx = i
        break

sched_re = re.compile(
    r'^\s*(?:\[?\d+\[?)?(?:THE\s+FIRST\s+SCHEDULE|FIRST\s+SCHEDULE|THE\s+SCHEDULE\b|SCHEDULE\s+I\b|THE\s+SCHEDULES\b)',
    re.MULTILINE
)

body_tokens = []
for idx, p_dict in enumerate(pages[start_page_idx:]):
    text = p_dict['text']
    if (start_page_idx + idx) > (len(pages) * 0.15):
        m_sched = sched_re.search(text)
        if m_sched:
            text = text[:m_sched.start()]
            for line in text.split('\n'):
                if line.strip():
                    body_tokens.append({'page': p_dict['page'], 'line': line.strip()})
            break
    for line in text.split('\n'):
        if line.strip():
            body_tokens.append({'page': p_dict['page'], 'line': line.strip()})

pat = re.compile(
    r'^\s*(?:[^\w\s]*\d*\[|\*+)?\s*(?:Section|Article|Sec\.)?\s*(\d+[A-Za-z]{0,2})\.\s*([A-Za-z\(\'\"—])'
)

section_starts = []
seen = set()
for idx, token in enumerate(body_tokens):
    m = pat.match(token['line'])
    if m:
        s_str = m.group(1)
        if s_str not in seen:
            seen.add(s_str)
            section_starts.append({'sec': s_str, 'token_idx': idx, 'page': token['page']})

broad = []
for i, s in enumerate(section_starts):
    start_t = s['token_idx']
    end_t = section_starts[i+1]['token_idx'] if i+1 < len(section_starts) else len(body_tokens)
    p_start = body_tokens[start_t]['page']
    p_end = body_tokens[end_t-1]['page']
    if (p_end - p_start) > 8:
        broad.append({'sec': s['sec'], 'p_start': p_start, 'p_end': p_end})

print(f"Total articles found: {len(section_starts)}")
print(f"Articles with page span > 8: {len(broad)}")
if broad:
    print(broad)
