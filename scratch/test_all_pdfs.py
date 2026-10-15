import os
import sys
import json
import pymupdf
import re

sys.path.insert(0, os.path.abspath("."))
from scratch.test_robust_parser import robust_parse_pdf
from ingestion.ingest_dataset import get_act_metadata_from_filename

extracted_dir = 'data/raw/extracted'
pdf_files = [f for f in os.listdir(extracted_dir) if f.endswith('.pdf')]

total_sections = 0
broad_count = 0
anomalies_count = 0

print(f"{'Act Filename':40s} | {'Secs':5s} | {'Broad (>8p)':11s} | {'Anomalies':10s}")
print("-" * 75)

for f in sorted(pdf_files):
    path = os.path.join(extracted_dir, f)
    doc_id, act_title, year = get_act_metadata_from_filename(f)
    secs = robust_parse_pdf(path, doc_id, act_title, year)
    total_sections += len(secs)
    broad = sum(1 for s in secs if (s['page_end'] - s['page_start']) > 8)
    anom = sum(1 for s in secs if len(s['anomalies']) > 0)
    broad_count += broad
    anomalies_count += anom
    print(f"{f:40s} | {len(secs):5d} | {broad:11d} | {anom:10d}")

print("-" * 75)
print(f"TOTAL PARSED SECTIONS ACROSS ALL 33 PDFS: {total_sections}")
print(f"TOTAL SECTIONS WITH BROAD PAGE RANGE (>8 pages): {broad_count}")
