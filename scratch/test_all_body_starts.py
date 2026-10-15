import os
import sys
import json
import pymupdf
import re

sys.path.insert(0, os.path.abspath("."))
from ingestion.section_parser import clean_section_text, extract_toc_titles

def detect_body_start_page(doc, filename):
    """
    Detects the 0-indexed page where the enactment body starts after any Table of Contents.
    """
    # Special handling for single-page documents
    if len(doc) <= 1:
        return 0

    # Look for enactment formula
    enactment_re = re.compile(
        r'\b(BE it enacted|BE IT ENACTED|Be it enacted|An Act to|AN ACT TO|WHEREAS it is expedient)\b',
        re.MULTILINE
    )
    
    # Check for Constitution
    if "constitution" in filename.lower():
        for i, page in enumerate(doc):
            txt = page.get_text()
            if re.search(r'PART\s+I\b', txt, re.IGNORECASE) and re.search(r'\b1\.\s+Name and territory', txt):
                return i
        return 0

    for i in range(min(40, len(doc))):
        txt = doc[i].get_text()
        if enactment_re.search(txt):
            return i

    return 0

print("Testing body start page detection on all 33 PDFs:")
extracted_dir = 'data/raw/extracted'
pdf_files = [f for f in os.listdir(extracted_dir) if f.endswith('.pdf')]

for f in sorted(pdf_files):
    path = os.path.join(extracted_dir, f)
    doc = pymupdf.open(path)
    start_idx = detect_body_start_page(doc, f)
    print(f"{f:40s} | Pages: {len(doc):3d} | Body Starts on Page: {start_idx + 1:2d}")
    doc.close()
