import os
import sys
import re
import json
import pymupdf

project_root = r"C:\Users\nived\OneDrive\Desktop\VidhiSetu"
manifest = json.load(open(os.path.join(project_root, "data/metadata/manifest.json")))
sched_re = re.compile(
    r'^\s*(?:\[?\d+\[?)?(?:THE\s+FIRST\s+SCHEDULE|FIRST\s+SCHEDULE|THE\s+SCHEDULE\b|SCHEDULE\s+I\b|THE\s+SCHEDULES\b)',
    re.MULTILINE
)

broad_list = []
for item in manifest:
    if item['file_type'] != 'pdf':
        continue
    pdf_path = os.path.join(project_root, item['target_path'])
    doc = pymupdf.open(pdf_path)
    pages = [{'page': p + 1, 'text': doc[p].get_text()} for p in range(len(doc))]
    doc.close()

    # Body start
    start_page_idx = 0
    enactment_re = re.compile(r'\b(BE it enacted|BE IT ENACTED|Be it enacted|An Act to|AN ACT TO|WHEREAS it is expedient)\b', re.MULTILINE)
    if 'constitution' in item['extracted_filename'].lower():
        for i, p in enumerate(pages):
            if re.search(r'PART\s+I\b', p['text'], re.IGNORECASE) and re.search(r'\b1\.\s+Name and territory', p['text']):
                start_page_idx = i
                break
    elif len(pages) > 1:
        for i in range(min(40, len(pages))):
            if enactment_re.search(pages[i]['text']):
                start_page_idx = i
                break

    body_tokens = []
    for idx, p_dict in enumerate(pages[start_page_idx:]):
        text = p_dict['text']
        # End markers: Objects & Reasons or Schedules
        m_end = re.search(r'^\s*(?:STATEMENT\s+OF\s+OBJECTS\s+AND\s+REASONS|FINANCIAL\s+MEMORANDUM|MEMORANDUM\s+REGARDING)', text, re.MULTILINE | re.IGNORECASE)
        m_sched = sched_re.search(text)
        if m_end or m_sched:
            cut_idx = min(m.start() for m in [m_end, m_sched] if m)
            text = text[:cut_idx]
            for line in text.split('\n'):
                if line.strip():
                    body_tokens.append({'page': p_dict['page'], 'line': line.strip()})
            break
        for line in text.split('\n'):
            if line.strip():
                body_tokens.append({'page': p_dict['page'], 'line': line.strip()})

    sec_start_re = re.compile(r'^\s*(?:Section|Article|Sec\.)?\s*(\d+[A-Z]?)\.\s*([A-Za-z\(\'\"])', re.MULTILINE | re.IGNORECASE)
    section_starts = []
    seen = set()
    for idx, token in enumerate(body_tokens):
        m = sec_start_re.match(token['line'])
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            section_starts.append({'sec': m.group(1), 'token_idx': idx, 'page': token['page']})

    for i, s in enumerate(section_starts):
        start_t = s['token_idx']
        end_t = section_starts[i+1]['token_idx'] if i+1 < len(section_starts) else len(body_tokens)
        p_start = body_tokens[start_t]['page']
        p_end = body_tokens[end_t-1]['page']
        if (p_end - p_start) > 8:
            broad_list.append({'file': item['extracted_filename'], 'sec': s['sec'], 'p_start': p_start, 'p_end': p_end})

print(f"Total sections with page span > 8: {len(broad_list)}")
if broad_list:
    print(json.dumps(broad_list, indent=2))
