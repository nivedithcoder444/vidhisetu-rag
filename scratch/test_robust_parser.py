import os
import sys
import json
import pymupdf
import re

sys.path.insert(0, os.path.abspath("."))
from ingestion.section_parser import clean_section_text, extract_toc_titles

def robust_parse_pdf(pdf_path, doc_id, act_title, year):
    doc = pymupdf.open(pdf_path)
    pages = [{"page": p + 1, "text": doc[p].get_text()} for p in range(len(doc))]
    doc.close()

    # Detect body start page
    start_page_idx = 0
    enactment_re = re.compile(
        r'\b(BE it enacted|BE IT ENACTED|Be it enacted|An Act to|AN ACT TO|WHEREAS it is expedient)\b',
        re.MULTILINE
    )
    if "constitution" in os.path.basename(pdf_path).lower():
        for i, p in enumerate(pages):
            txt = p['text']
            if re.search(r'PART\s+I\b', txt, re.IGNORECASE) and re.search(r'\b1\.\s+Name and territory', txt):
                start_page_idx = i
                break
    elif len(pages) > 1:
        for i in range(min(40, len(pages))):
            txt = pages[i]['text']
            if enactment_re.search(txt):
                start_page_idx = i
                break

    # Extract TOC titles from pages before start_page_idx
    toc_titles = extract_toc_titles(pages, max_toc_page=start_page_idx) if start_page_idx > 0 else {}

    # Tokenize body lines
    body_tokens = []
    stop_parsing = False
    end_marker_re = re.compile(
        r'STATEMENT\s+OF\s+OBJECTS\s+AND\s+REASONS|FINANCIAL\s+MEMORANDUM|MEMORANDUM\s+REGARDING',
        re.IGNORECASE
    )

    for idx, p_dict in enumerate(pages[start_page_idx:]):
        if stop_parsing:
            break
        p_num = p_dict['page']
        text = p_dict['text']
        
        # Only check end markers in the last 30% of pages to avoid footnotes
        if (start_page_idx + idx) > (len(pages) * 0.7):
            m_end = re.search(r'^\s*(?:STATEMENT\s+OF\s+OBJECTS\s+AND\s+REASONS|FINANCIAL\s+MEMORANDUM)', text, re.MULTILINE | re.IGNORECASE)
            if m_end:
                text = text[:m_end.start()]
                stop_parsing = True

        for line in text.split('\n'):
            if line.strip():
                body_tokens.append({'page': p_num, 'line': line.strip()})

    sec_start_re = re.compile(
        r'^\s*(?:Section|Article|Sec\.)?\s*(\d+[A-Z]?)\.\s*([A-Za-z\(\'\"])',
        re.MULTILINE | re.IGNORECASE
    )

    section_starts = []
    current_chapter = None
    seen_sections = set()

    for idx, token in enumerate(body_tokens):
        line_str = token['line']

        # Chapter detection
        if re.match(r'^CHAPTER\s+[I|V|X|L|C|D|M\d]+', line_str, re.IGNORECASE) or re.match(r'^PART\s+[I|V|X|L|C|D|M\d]+', line_str, re.IGNORECASE):
            current_chapter = line_str

        m = sec_start_re.match(line_str)
        if m:
            s_str = m.group(1)
            # Avoid duplicate section declarations within the body
            if s_str not in seen_sections:
                seen_sections.add(s_str)
                section_starts.append({
                    'section': s_str,
                    'token_idx': idx,
                    'page': token['page'],
                    'chapter': current_chapter,
                    'title_line': line_str[:120]
                })

    if len(section_starts) == 0:
        return [{
            "document_id": doc_id,
            "act": act_title,
            "year": year,
            "chapter": "General",
            "section": "1",
            "section_title": act_title,
            "text": '\n'.join([t['line'] for t in body_tokens]),
            "page_start": 1,
            "page_end": len(pages),
            "source": os.path.relpath(pdf_path, ".").replace('\\', '/'),
            "anomalies": []
        }]

    total_found = len(section_starts)
    sections_records = []

    for idx, sec in enumerate(section_starts):
        s_num_str = sec['section']
        start_t = sec['token_idx']
        chapter = sec['chapter'] or "General"

        if idx + 1 < total_found:
            end_t = section_starts[idx + 1]['token_idx']
        else:
            end_t = len(body_tokens)

        sec_tokens = body_tokens[start_t:end_t]
        sec_lines = [t['line'] for t in sec_tokens]
        page_start = sec_tokens[0]['page']
        page_end = sec_tokens[-1]['page']
        sec_text = clean_section_text(sec_lines)

        # Title: Prefer TOC canonical title, fallback to title_line
        sec_title = toc_titles.get(s_num_str, "")
        if not sec_title:
            raw_title = sec['title_line']
            sec_title = re.sub(r'^\s*(?:Section|Article|Sec\.)?\s*\d+[A-Z]?\.\s*', '', raw_title).strip()
        if not sec_title:
            sec_title = f"{act_title} Section {s_num_str}"

        # Anomaly detection (Requirement 2)
        anomalies = []
        if (page_end - page_start) > 8:
            anomalies.append(f"Excessively broad page range ({page_start}-{page_end})")
        # Check if text contains multiple section headings
        other_secs = re.findall(r'\n\s*(\d+[A-Z]?)\.\s+[A-Z]', sec_text)
        if len(other_secs) > 2:
            anomalies.append(f"Possible embedded section headings detected: {other_secs[:5]}")

        sections_records.append({
            "document_id": doc_id,
            "act": act_title,
            "year": year,
            "chapter": chapter,
            "section": s_num_str,
            "section_title": sec_title,
            "text": sec_text,
            "page_start": page_start,
            "page_end": page_end,
            "source": os.path.relpath(pdf_path, ".").replace('\\', '/'),
            "anomalies": anomalies
        })

    return sections_records

# Test on key acts
for act_file, doc_id, title, year in [
    ("data/raw/extracted/bns.pdf", "BNS", "Bharatiya Nyaya Sanhita", 2023),
    ("data/raw/extracted/arms act 1959.pdf", "ARMS_ACT_1959", "Arms Act 1959", 1959),
    ("data/raw/extracted/indian contract act 1872.pdf", "INDIAN_CONTRACT_ACT_1872", "Indian Contract Act 1872", 1872)
]:
    res = robust_parse_pdf(act_file, doc_id, title, year)
    print(f"\n{title} -> Total parsed sections: {len(res)}")
    broad = [s for s in res if (s['page_end'] - s['page_start']) > 8]
    print(f"  Broad page range sections (>8 pages): {len(broad)}")
    if doc_id == "BNS":
        s105 = [s for s in res if s['section'] == '105'][0]
        print(f"  BNS Sec 105: Pages {s105['page_start']}-{s105['page_end']}, Title: {s105['section_title']}")
        print(f"  BNS Sec 105 Text len: {len(s105['text'])}, Preview: {repr(s105['text'][:150])}")
        s358 = [s for s in res if s['section'] == '358'][0]
        print(f"  BNS Sec 358: Pages {s358['page_start']}-{s358['page_end']}, Text len: {len(s358['text'])}")
