import os
import sys
import zipfile
import hashlib
import json
import io
import re
import csv
import numpy as np
from typing import List, Dict, Any, Tuple

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pymupdf
from ingestion.section_parser import clean_section_text, extract_toc_titles
from ingestion.chunker import chunk_section
from retrieval.bm25 import BM25Retriever
from retrieval.semantic import BGEM3SemanticRetriever


def compute_sha256_file(file_path: str) -> str:
    """Computes SHA256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_sha256_bytes(data: bytes) -> str:
    """Computes SHA256 hash of byte content."""
    return hashlib.sha256(data).hexdigest()


def sanitize_filename(filename: str) -> str:
    """Sanitizes extracted filenames to prevent path traversal (ZipSlip)."""
    clean_name = os.path.basename(filename)
    clean_name = re.sub(r'[^\w\s\.\-\(\)]', '_', clean_name)
    return clean_name


# =====================================================================
# STEP 1: LOCATE AND EXTRACT ARCHIVE
# =====================================================================
def step_1_extract_archive(
    zip_path: str = "AI Agent.zip",
    extract_dir: str = "data/raw/extracted",
    manifest_path: str = "data/metadata/manifest.json"
) -> List[Dict[str, Any]]:
    """Locates and extracts AI Agent.zip into data/raw/extracted/ and generates manifest."""
    abs_zip = os.path.abspath(zip_path)
    if not os.path.exists(abs_zip):
        raise FileNotFoundError(f"Archive not found at {abs_zip}. Please place AI Agent.zip in the project root.")

    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    extracted_manifest = []

    with zipfile.ZipFile(abs_zip, 'r') as z:
        infolist = z.infolist()
        for info in infolist:
            if info.is_dir():
                continue

            safe_name = sanitize_filename(info.filename)
            target_path = os.path.abspath(os.path.join(extract_dir, safe_name))

            # Prevent ZipSlip vulnerability
            if not target_path.startswith(os.path.abspath(extract_dir)):
                print(f"[Security Warning] Skipped unsafe zip entry: {info.filename}")
                continue

            # Extract file content
            content = z.read(info.filename)
            sha256_hash = compute_sha256_bytes(content)

            # Avoid overwriting identical files, handle duplicate names if different
            if os.path.exists(target_path):
                existing_sha = compute_sha256_file(target_path)
                if existing_sha != sha256_hash:
                    base, ext = os.path.splitext(safe_name)
                    safe_name = f"{base}_{sha256_hash[:8]}{ext}"
                    target_path = os.path.abspath(os.path.join(extract_dir, safe_name))

            with open(target_path, 'wb') as out_f:
                out_f.write(content)

            file_type = "pdf" if safe_name.endswith('.pdf') else ("csv" if safe_name.endswith('.csv') else "other")

            entry = {
                "original_zip_entry": info.filename,
                "extracted_filename": safe_name,
                "target_path": os.path.relpath(target_path, project_root).replace('\\', '/'),
                "file_size": info.file_size,
                "sha256": sha256_hash,
                "file_type": file_type
            }
            extracted_manifest.append(entry)

    with open(manifest_path, 'w', encoding='utf-8') as mf:
        json.dump(extracted_manifest, mf, indent=4, ensure_ascii=False)

    print(f"[Step 1] Extracted {len(extracted_manifest)} files into {extract_dir}. Manifest saved to {manifest_path}.")
    return extracted_manifest


# =====================================================================
# STEP 2: AUDIT EXTRACTED DOCUMENTS
# =====================================================================
def step_2_audit_documents(
    manifest: List[Dict[str, Any]],
    audit_path: str = "data/metadata/audit.json"
) -> List[Dict[str, Any]]:
    """Audits every extracted file, recording page count, text characters, scanned status, etc."""
    audit_results = []

    for item in manifest:
        rel_path = item["target_path"]
        abs_path = os.path.join(project_root, rel_path)

        file_type = item["file_type"]
        page_count = 0
        total_chars = 0
        is_scanned = False
        status = "OK"
        error_msg = None

        try:
            if file_type == "pdf":
                doc = pymupdf.open(abs_path)
                page_count = len(doc)
                total_chars = sum(len(doc[p].get_text()) for p in range(page_count))
                doc.close()
                if page_count > 0 and total_chars < 100:
                    is_scanned = True
            elif file_type == "csv":
                with open(abs_path, 'r', encoding='utf-8', errors='ignore') as csv_f:
                    reader = csv.reader(csv_f)
                    page_count = sum(1 for _ in reader)  # count rows as "pages"
                    csv_f.seek(0)
                    total_chars = len(csv_f.read())
        except Exception as e:
            status = "ERROR"
            error_msg = str(e)

        audit_entry = {
            "filename": item["extracted_filename"],
            "target_path": rel_path,
            "file_type": file_type,
            "file_size": item["file_size"],
            "sha256": item["sha256"],
            "page_count": page_count,
            "total_chars": total_chars,
            "is_scanned": is_scanned,
            "needs_ocr": is_scanned,
            "status": status,
            "error_msg": error_msg
        }
        audit_results.append(audit_entry)

    os.makedirs(os.path.dirname(audit_path), exist_ok=True)
    with open(audit_path, 'w', encoding='utf-8') as af:
        json.dump(audit_results, af, indent=4, ensure_ascii=False)

    print(f"[Step 2] Audited {len(audit_results)} documents. Audit saved to {audit_path}.")
    return audit_results


# =====================================================================
# STEP 3: EXTRACT & STRUCTURE LEGAL CONTENT
# =====================================================================
CANONICAL_ACT_TITLES = {
    "bns": ("Bharatiya Nyaya Sanhita, 2023", 2023),
    "bnss": ("Bharatiya Nagarik Suraksha Sanhita, 2023", 2023),
    "bsa": ("Bharatiya Sakshya Adhiniyam, 2023", 2023),
    "constitution": ("Constitution of India", 1950),
    "code of civil procedure 1908": ("Code of Civil Procedure, 1908", 1908),
    "indian contract act 1872": ("Indian Contract Act, 1872", 1872),
    "arms act 1959": ("Arms Act, 1959", 1959),
    "pocso act 2012": ("Protection of Children from Sexual Offences Act, 2012", 2012),
    "prvention of money laundering act 2002": ("Prevention of Money Laundering Act, 2002", 2002),
    "narcotics act 1985": ("Narcotic Drugs and Psychotropic Substances Act, 1985", 1985),
    "sc and st act 1989": ("Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act, 1989", 1989),
    "specifc relief act 1963": ("Specific Relief Act, 1963", 1963),
    "transfer of property act 1882": ("Transfer of Property Act, 1882", 1882),
    "unlawful activities(prevention) act 1967": ("Unlawful Activities (Prevention) Act, 1967", 1967),
    "disaster management act 2005": ("Disaster Management Act, 2005", 2005),
    "dowry prohibition": ("Dowry Prohibition Act, 1961", 1961),
    "extradition act 1962": ("Extradition Act, 1962", 1962),
    "fugitve economic offenders act 2018": ("Fugitive Economic Offenders Act, 2018", 2018),
    "immoral traffic (prevention) act 1956": ("Immoral Traffic (Prevention) Act, 1956", 1956),
    "indian easements acts 1882": ("Indian Easements Act, 1882", 1882),
    "juvenile justice act 2015": ("Juvenile Justice (Care and Protection of Children) Act, 2015", 2015),
    "limitation act 1963": ("Limitation Act, 1963", 1963),
    "national security act 1980": ("National Security Act, 1980", 1980),
    "nia act 2008": ("National Investigation Agency Act, 2008", 2008),
    "official secrets act": ("Official Secrets Act, 1923", 1923),
    "prevention of corruption act": ("Prevention of Corruption Act, 1988", 1988),
    "prisoners act 1900": ("Prisoners Act, 1900", 1900),
    "prisons act 1894": ("Prisons Act, 1894", 1894),
    "probation of offenders acts 1958": ("Probation of Offenders Act, 1958", 1958),
    "protection of civil rights act 1955": ("Protection of Civil Rights Act, 1955", 1955),
    "registration act 1908": ("Registration Act, 1908", 1908),
    "criminal procedure (identification)act 2022": ("Criminal Procedure (Identification) Act, 2022", 2022),
}


def get_act_metadata_from_filename(filename: str) -> Tuple[str, str, int]:
    """Generates stable document_id, canonical act title, and year from filename."""
    base = os.path.splitext(filename)[0].lower()
    
    # Stable Document ID (preserves exact chunk IDs across runs)
    doc_id = re.sub(r'[^\w]', '_', base).upper()
    doc_id = re.sub(r'_+', '_', doc_id).strip('_')

    if base in CANONICAL_ACT_TITLES:
        title, year = CANONICAL_ACT_TITLES[base]
        return doc_id, title, year

    # Year extraction
    m_year = re.search(r'\b(18\d{2}|19\d{2}|20\d{2})\b', base)
    year = int(m_year.group(1)) if m_year else 0

    # Clean title
    clean_title = base.replace('_', ' ').replace('-', ' ').strip()
    clean_title = re.sub(r'\s+', ' ', clean_title).title()

    return doc_id, clean_title, year


def parse_pdf_into_sections(pdf_path: str, doc_id: str, act_title: str, year: int) -> List[Dict[str, Any]]:
    """
    Parses an arbitrary Legal Act PDF into structured section records,
    reliably skipping Table of Contents and flagging parsing anomalies.
    """
    doc = pymupdf.open(pdf_path)
    pages = [{"page": p + 1, "text": doc[p].get_text()} for p in range(len(doc))]
    doc.close()

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

    # Extract TOC canonical titles from pages before start_page_idx
    toc_titles = extract_toc_titles(pages, max_toc_page=start_page_idx) if start_page_idx > 0 else {}

    sched_re = re.compile(
        r'^\s*(?:\[?\d+\[?)?(?:THE\s+FIRST\s+SCHEDULE|FIRST\s+SCHEDULE|THE\s+SCHEDULE\b|SCHEDULE\s+I\b|THE\s+SCHEDULES\b)',
        re.MULTILINE
    )

    # Tokenize body lines ONLY from the enactment body
    body_tokens = []
    stop_parsing = False

    for idx, p_dict in enumerate(pages[start_page_idx:]):
        if stop_parsing:
            break
        p_num = p_dict['page']
        text = p_dict['text']

        # End markers: Objects and Reasons or Schedules after sections
        # Avoid checking in the first 15% of pages to protect preamble/footnotes
        if (start_page_idx + idx) > (len(pages) * 0.15):
            m_end = re.search(
                r'^\s*(?:STATEMENT\s+OF\s+OBJECTS\s+AND\s+REASONS|FINANCIAL\s+MEMORANDUM|MEMORANDUM\s+REGARDING)',
                text,
                re.MULTILINE | re.IGNORECASE
            )
            m_sched = sched_re.search(text)
            if m_end or m_sched:
                cut_idx = min(m.start() for m in [m_end, m_sched] if m)
                text = text[:cut_idx]
                stop_parsing = True

        for line in text.split('\n'):
            if line.strip():
                body_tokens.append({'page': p_num, 'line': line.strip()})

    sec_start_re = re.compile(
        r'^\s*(?:[^\w\s]*\d*\[|\*+)?\s*(?:Section|Article|Sec\.)?\s*(\d+[A-Za-z]{0,2})\.\s*([A-Za-z\(\'\"—])',
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
            "source": os.path.relpath(pdf_path, project_root).replace('\\', '/'),
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
            sec_title = re.sub(r'^\s*(?:Section|Article|Sec\.)?\s*\d+[A-Za-z]{0,2}\.\s*', '', raw_title).strip()
        if not sec_title:
            sec_title = f"{act_title} Section {s_num_str}"

        # Anomaly detection (Requirement 2)
        anomalies = []
        if (page_end - page_start) > 8:
            anomalies.append(f"Excessively broad page range ({page_start}-{page_end})")
        other_secs = re.findall(r'\n\s*(\d+[A-Za-z]{0,2})\.\s+[A-Z]', sec_text)
        if len(other_secs) > 2:
            anomalies.append(f"Possible embedded section headings: {other_secs[:5]}")

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
            "source": os.path.relpath(pdf_path, project_root).replace('\\', '/'),
            "anomalies": anomalies
        })

    return sections_records


def parse_ipc_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Parses ipc_sections.csv into structured legal section records."""
    sec_records = []
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            sec_raw = row.get('Section', f'{i}')
            sec_clean = sec_raw.replace('IPC_', '').strip()
            offense = row.get('Offense', '').strip()
            punishment = row.get('Punishment', '').strip()
            description = row.get('Description', '').strip()

            full_text = f"Section {sec_clean}: {offense}\n\n{description}\n\nPunishment: {punishment}"

            sec_records.append({
                "document_id": "IPC_CSV",
                "act": "Indian Penal Code (IPC Dataset)",
                "year": 1860,
                "chapter": "IPC Offences & Punishments",
                "section": sec_clean,
                "section_title": offense or f"IPC Section {sec_clean}",
                "text": full_text,
                "page_start": 1,
                "page_end": 1,
                "source": os.path.relpath(csv_path, project_root).replace('\\', '/')
            })

    return sec_records


def step_3_extract_and_structure(
    manifest: List[Dict[str, Any]],
    output_sections_path: str = "data/processed/all_legal_sections.json"
) -> List[Dict[str, Any]]:
    """Extracts and structures legal content from all PDFs and CSVs in manifest."""
    all_sections = []

    for item in manifest:
        abs_path = os.path.join(project_root, item["target_path"])
        file_type = item["file_type"]
        filename = item["extracted_filename"]

        if file_type == "csv" and "ipc_sections" in filename.lower():
            ipc_secs = parse_ipc_csv(abs_path)
            all_sections.extend(ipc_secs)
            print(f"[Step 3] Parsed IPC CSV ({len(ipc_secs)} sections)")
        elif file_type == "pdf":
            doc_id, act_title, year = get_act_metadata_from_filename(filename)
            pdf_secs = parse_pdf_into_sections(abs_path, doc_id, act_title, year)
            all_sections.extend(pdf_secs)
            print(f"[Step 3] Parsed {filename:35s} -> {len(pdf_secs):4d} sections")

    os.makedirs(os.path.dirname(output_sections_path), exist_ok=True)
    with open(output_sections_path, 'w', encoding='utf-8') as sf:
        json.dump(all_sections, sf, indent=4, ensure_ascii=False)

    print(f"[Step 3] Extracted total {len(all_sections)} legal sections across all Acts and CSV dataset.")
    return all_sections


# =====================================================================
# STEP 4: PREPARE RAG INDEX (CHUNKS, EMBEDDINGS, BM25)
# =====================================================================
def step_4_prepare_rag_index(
    sections: List[Dict[str, Any]],
    chunks_path: str = "data/processed/all_legal_chunks.json",
    embeddings_path: str = "data/processed/all_legal_embeddings.npy"
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Chunks all legal sections, generates BGE-M3 embeddings, and builds BM25 index."""
    all_chunks = []

    for sec in sections:
        doc_id = sec.get("document_id", "ACT")
        c_list = chunk_section(sec, max_chunk_chars=1800, prefix_id=doc_id)
        all_chunks.extend(c_list)

    os.makedirs(os.path.dirname(chunks_path), exist_ok=True)
    with open(chunks_path, 'w', encoding='utf-8') as cf:
        json.dump(all_chunks, cf, indent=4, ensure_ascii=False)

    print(f"[Step 4] Created {len(all_chunks)} legal chunks. Saved to {chunks_path}.")

    # Generate or load BGE-M3 embeddings
    semantic_retriever = BGEM3SemanticRetriever(
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        model_name="BAAI/bge-m3"
    )
    embeddings = semantic_retriever.build_embeddings(force_rebuild=False)

    # Build BM25 index over all_legal_chunks.json
    bm25_retriever = BM25Retriever(chunks_path=chunks_path)
    bm25_count = bm25_retriever.build_index()

    print(f"[Step 4] BGE-M3 embeddings generated: Shape {embeddings.shape}.")
    print(f"[Step 4] BM25 Index built over {bm25_count} chunks.")

    return all_chunks, embeddings


def run_full_ingestion():
    """Runs end-to-end dataset ingestion pipeline."""
    print("========================================")
    print("VIDHISETU DATASET INGESTION PIPELINE")
    print("========================================")

    # Step 1
    manifest = step_1_extract_archive()

    # Step 2
    audit = step_2_audit_documents(manifest)

    # Step 3
    sections = step_3_extract_and_structure(manifest)

    # Step 4
    chunks, embeddings = step_4_prepare_rag_index(sections)

    # Step 5 Verification Summary
    total_discovered = len(manifest)
    total_extracted = len(manifest)
    pdf_count = sum(1 for a in audit if a["file_type"] == "pdf" and a["status"] == "OK")
    ocr_count = sum(1 for a in audit if a["needs_ocr"])
    failed_count = sum(1 for a in audit if a["status"] != "OK")

    print("\n========================================")
    print("INGESTION VERIFICATION SUMMARY")
    print("========================================")
    print(f"1. Archive Files Discovered: {total_discovered}")
    print(f"2. Files Extracted:          {total_extracted}")
    print(f"3. PDFs Processed:           {pdf_count}")
    print(f"4. Files Requiring OCR:      {ocr_count}")
    print(f"5. Failed Files:             {failed_count}")
    print(f"6. Legal Sections Parsed:    {len(sections)}")
    print(f"7. Chunks Generated:         {len(chunks)}")
    print(f"8. Embeddings Generated:     {embeddings.shape[0]} (Shape: {embeddings.shape})")
    print(f"9. BM25 Records Indexed:     {len(chunks)}")
    print("========================================\n")


if __name__ == "__main__":
    run_full_ingestion()
