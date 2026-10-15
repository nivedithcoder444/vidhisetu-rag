import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import re
import json
from typing import List, Dict, Any
from ingestion.loader import load_pdf


def clean_section_text(raw_lines: List[str]) -> str:
    cleaned = []
    margin_numbers = {'5', '10', '15', '20', '25', '30', '35', '40', '45', '50'}
    
    for line in raw_lines:
        s = line.strip()
        if not s:
            continue
        if s in margin_numbers:
            continue
        if s.startswith('AS INTRODUCED IN LOK SABHA') or s.startswith('Bill No.') or s.startswith('THE BHARATIYA NYAYA SANHITA'):
            continue
        if re.match(r'^\d+$', s) and len(s) <= 3:  # Standalone page number
            continue
        cleaned.append(s)

    text = '\n'.join(cleaned)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    lines_norm = []
    for l in text.split('\n'):
        l_norm = re.sub(r'[ \t]+', ' ', l).strip()
        if l_norm:
            lines_norm.append(l_norm)
            
    return '\n'.join(lines_norm)


def parse_bns_sections(
    pdf_path: str = "data/raw/bns_2023.pdf",
    document_id: str = "BNS_2023",
    act_name: str = "Bharatiya Nyaya Sanhita",
    year: int = 2023
) -> List[Dict[str, Any]]:
    pages = load_pdf(pdf_path)

    # 1. Parse TOC for canonical section titles (Pages 1 to 13)
    toc_text = '\n'.join([pages[p-1]['text'] for p in range(1, 14)])
    toc_lines = [l.strip() for l in toc_text.split('\n') if l.strip()]

    toc_titles = {}
    i = 0
    while i < len(toc_lines):
        line = toc_lines[i]
        m = re.match(r'^(\d+)\.\s*(.*)', line)
        if m:
            sec_num = int(m.group(1))
            title_parts = [m.group(2).strip()] if m.group(2).strip() else []
            j = i + 1
            while j < len(toc_lines):
                next_l = toc_lines[j]
                if re.match(r'^\d+\.\s*', next_l) or next_l.startswith('CHAPTER') or next_l.startswith('CLAUSES') or next_l.startswith('Of ') or re.match(r'^\([ivx]+\)$', next_l):
                    break
                title_parts.append(next_l)
                j += 1
            title = ' '.join(title_parts).strip()
            title = re.sub(r'\s+', ' ', title)
            toc_titles[str(sec_num)] = title
            i = j
        else:
            i += 1

    # 2. Extract line tokens with page numbers from body (Page 14 onwards)
    sec_start_re = re.compile(r'^\s*(\d+)\.\s*([A-Za-z\(\'\"])', re.MULTILINE)

    body_tokens = []
    stop_parsing = False
    for p_dict in pages[13:]:
        if stop_parsing:
            break
        p_num = p_dict['page']
        text = p_dict['text']
        if 'STATEMENT OF OBJECTS AND REASONS' in text:
            text = text.split('STATEMENT OF OBJECTS AND REASONS')[0]
            stop_parsing = True
        
        for line in text.split('\n'):
            if line.strip():
                body_tokens.append({'page': p_num, 'line': line})

    # Find section start indices in body_tokens
    section_starts = []
    expected_sec = 1
    current_chapter = None

    for idx, token in enumerate(body_tokens):
        line_str = token['line'].strip()
        
        if re.match(r'^CHAPTER\s+[I|V|X|L|C|D|M]+$', line_str, re.IGNORECASE):
            ch_num = line_str.upper()
            ch_title = ''
            for k in range(idx + 1, min(idx + 4, len(body_tokens))):
                candidate = body_tokens[k]['line'].strip()
                if candidate and not re.match(r'^\d+\.', candidate) and not candidate.startswith('CHAPTER'):
                    ch_title = candidate
                    break
            current_chapter = f"{ch_num} - {ch_title}".strip(" -")

        m = sec_start_re.match(line_str)
        if m:
            s_num = int(m.group(1))
            if s_num == expected_sec:
                section_starts.append({
                    'section': str(s_num),
                    'token_idx': idx,
                    'page': token['page'],
                    'chapter': current_chapter
                })
                expected_sec += 1

    sections_records = []
    total_found = len(section_starts)

    for idx, sec in enumerate(section_starts):
        s_num_str = sec['section']
        start_t_idx = sec['token_idx']
        chapter = sec['chapter']

        if idx + 1 < total_found:
            end_t_idx = section_starts[idx + 1]['token_idx']
        else:
            end_t_idx = len(body_tokens)

        sec_tokens = body_tokens[start_t_idx:end_t_idx]
        sec_lines = [t['line'] for t in sec_tokens]
        page_start = sec_tokens[0]['page']
        page_end = sec_tokens[-1]['page']
        
        sec_text = clean_section_text(sec_lines)
        sec_title = toc_titles.get(s_num_str, "")

        record = {
            "document_id": document_id,
            "act": act_name,
            "year": year,
            "chapter": chapter,
            "section": s_num_str,
            "section_title": sec_title,
            "text": sec_text,
            "page_start": page_start,
            "page_end": page_end,
            "source": pdf_path
        }
        sections_records.append(record)

    return sections_records


if __name__ == "__main__":
    records = parse_bns_sections()
    sec356 = records[355]
    print('Sec 356 start:', sec356['page_start'], 'end:', sec356['page_end'])
