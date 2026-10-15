import os
import pymupdf
import re

extracted_dir = 'data/raw/extracted'
pdf_files = [f for f in os.listdir(extracted_dir) if f.endswith('.pdf')]

print(f"Scanning {len(pdf_files)} PDFs for Enactment / Body Start Page...")

for f in sorted(pdf_files):
    path = os.path.join(extracted_dir, f)
    doc = pymupdf.open(path)
    start_page = 0
    enact_phrase = ""
    for p_idx, page in enumerate(doc):
        txt = page.get_text()
        # Look for enactment formula
        m = re.search(r'(BE it enacted|BE IT ENACTED|Be it enacted|An Act to|AN ACT TO|WHEREAS it is expedient)', txt)
        if m:
            start_page = p_idx + 1
            enact_phrase = m.group(1)
            break
    
    # Also check if page 1 has "ARRANGEMENT OF SECTIONS" or "ARRANGEMENT OF CLAUSES"
    has_arr = False
    if len(doc) > 0:
        p1_txt = doc[0].get_text()
        if "ARRANGEMENT OF SECTIONS" in p1_txt.upper() or "ARRANGEMENT OF CLAUSES" in p1_txt.upper() or "CONTENTS" in p1_txt.upper():
            has_arr = True

    print(f"{f:40s} | Total pages: {len(doc):3d} | Has TOC: {str(has_arr):5s} | Enactment Page: {start_page:2d} ({enact_phrase[:20]})")
    doc.close()
