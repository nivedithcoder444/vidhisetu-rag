import os
from typing import List, Dict, Any

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf


def load_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Loads a PDF document using PyMuPDF and extracts text page-by-page.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        A list of dictionaries representing pages:
        [
            {
                "page": 1,
                "text": "..."
            },
            ...
        ]
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    pages = []

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_text = page.get_text()
            pages.append({
                "page": page_index + 1,
                "text": page_text
            })
    finally:
        doc.close()

    return pages
