import os
import re
import json
from typing import List, Dict, Any, Optional
from ingestion.loader import load_pdf


def clean_section_text(raw_lines: List[str]) -> str:
    """
    Cleans extracted raw line tokens for a legal section:
    - Removes marginal line numbers (5, 10, 15...).
    - Removes page header/footer boilerplate.
    - Removes standalone page numbers.
    - Fixes hyphenated word breaks at line ends.
    - Normalizes multiple spaces within lines while preserving sub-sections and clauses.
    """
    cleaned = []
    margin_numbers = {'5', '10', '15', '20', '25', '30', '35', '40', '45', '50'}
    
    for line in raw_lines:
        s = line.strip()
        if not s:
            continue
        if s in margin_numbers:
            continue
        if (s.startswith('AS INTRODUCED IN LOK SABHA') or 
            s.startswith('Bill No.') or 
            s.startswith('THE BHARATIYA NYAYA SANHITA') or
            s.startswith('THE BHARATIYA NAGARIK') or
            s.startswith('THE BHARATIYA SAKSHYA')):
            continue
        if re.match(r'^\d+$', s) and len(s) <= 3:  # Standalone page number
            continue
        cleaned.append(s)

    text = '\n'.join(cleaned)
    # Fix hyphenation across line breaks (e.g., "punish-\nment" -> "punishment")
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    lines_norm = []
    for l in text.split('\n'):
        l_norm = re.sub(r'[ \t]+', ' ', l).strip()
        if l_norm:
            lines_norm.append(l_norm)
            
    return '\n'.join(lines_norm)


def extract_toc_titles(pages: List[Dict[str, Any]], max_toc_page: int = 13) -> Dict[str, str]:
    """
    Parses Table of Contents / Arrangement of Clauses (pages 1 to max_toc_page)
    to build a canonical mapping of section_number -> section_title.
    """
    toc_pages = [p['text'] for p in pages if p['page'] <= max_toc_page]
    toc_text = '\n'.join(toc_pages)
    toc_lines = [l.strip() for l in toc_text.split('\n') if l.strip()]

    toc_titles = {}
    i = 0
    while i < len(toc_lines):
        line = toc_lines[i]
        m = re.match(r'^(\d+[A-Z]?)\.\s*(.*)', line)
        if m:
            sec_num = m.group(1)
            title_parts = [m.group(2).strip()] if m.group(2).strip() else []
            j = i + 1
            while j < len(toc_lines):
                next_l = toc_lines[j]
                if (re.match(r'^\d+[A-Z]?\.\s*', next_l) or 
                    next_l.startswith('CHAPTER') or 
                    next_l.startswith('CLAUSES') or 
                    next_l.startswith('Of ') or 
                    re.match(r'^\([ivx]+\)$', next_l)):
                    break
                title_parts.append(next_l)
                j += 1
            title = ' '.join(title_parts).strip()
            title = re.sub(r'\s+', ' ', title)
            toc_titles[sec_num] = title
            i = j
        else:
            i += 1

    return toc_titles


def parse_sections(
    pdf_path: str,
    document_id: str = "BNS_2023",
    act_name: str = "Bharatiya Nyaya Sanhita",
    year: int = 2023,
    output_path: Optional[str] = "data/processed/bns_sections.json"
) -> List[Dict[str, Any]]:
    """
    Parses legal sections from a legislation PDF using PyMuPDF page extraction.

    Supports:
    - BNS 2023 (Bharatiya Nyaya Sanhita)
    - BNSS 2023 (Bharatiya Nagarik Suraksha Sanhita)
    - BSA 2023 (Bharatiya Sakshya Adhiniyam)
    - IPC 1860 (Indian Penal Code)
    - CrPC 1973 (Code of Criminal Procedure)
    - Constitution of India & other Acts
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    pages = load_pdf(pdf_path)

    # Detect body start page (where enactment/body starts after TOC)
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

    # Extract TOC section titles strictly from front-matter pages (before body)
    toc_titles = extract_toc_titles(pages, max_toc_page=start_page_idx) if start_page_idx > 0 else {}

    # Tokenize body lines with page metadata
    sec_start_re = re.compile(r'^\s*(\d+[A-Za-z]{0,2})\.\s*([A-Za-z\(\'\"])', re.MULTILINE)

    body_tokens = []
    stop_parsing = False

    for idx, p_dict in enumerate(pages[start_page_idx:]):
        if stop_parsing:
            break
        p_num = p_dict['page']
        text = p_dict['text']

        # Only check end markers in the last 30% of pages to avoid footnotes
        if (start_page_idx + idx) > (len(pages) * 0.7):
            m_end = re.search(
                r'^\s*(?:STATEMENT\s+OF\s+OBJECTS\s+AND\s+REASONS|FINANCIAL\s+MEMORANDUM|MEMORANDUM\s+REGARDING)',
                text,
                re.MULTILINE | re.IGNORECASE
            )
            if m_end:
                text = text[:m_end.start()]
                stop_parsing = True

        for line in text.split('\n'):
            if line.strip():
                body_tokens.append({'page': p_num, 'line': line})

    # Detect chapters and section starts
    section_starts = []
    expected_numeric = 1
    current_chapter = None

    for idx, token in enumerate(body_tokens):
        line_str = token['line'].strip()

        # Chapter detection
        if re.match(r'^CHAPTER\s+[I|V|X|L|C|D|M]+$', line_str, re.IGNORECASE):
            ch_num = line_str.upper()
            ch_title = ''
            for k in range(idx + 1, min(idx + 4, len(body_tokens))):
                candidate = body_tokens[k]['line'].strip()
                if candidate and not re.match(r'^\d+[A-Za-z]{0,2}\.', candidate) and not candidate.startswith('CHAPTER'):
                    ch_title = candidate
                    break
            current_chapter = f"{ch_num} - {ch_title}".strip(" -")

        # Section start matching
        m = sec_start_re.match(line_str)
        if m:
            s_str = m.group(1)
            try:
                s_num = int(re.sub(r'[A-Za-z]+$', '', s_str))
                if s_num == expected_numeric or s_str not in [s['section'] for s in section_starts]:
                    section_starts.append({
                        'section': s_str,
                        'token_idx': idx,
                        'page': token['page'],
                        'chapter': current_chapter
                    })
                    if s_num == expected_numeric:
                        expected_numeric += 1
            except ValueError:
                pass

    # Build section records
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

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sections_records, f, indent=4, ensure_ascii=False)

    return sections_records


if __name__ == "__main__":
    bns_sections = parse_sections("data/raw/bns_2023.pdf")
    print(f"[SectionParser] Parsed {len(bns_sections)} sections into data/processed/bns_sections.json")
