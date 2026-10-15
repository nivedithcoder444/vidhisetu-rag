import os
import sys

# Ensure project root is in python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ingestion.downloader import download_bns_2023
from ingestion.loader import load_pdf


def main():
    pdf_path = os.path.join(project_root, "data", "raw", "bns_2023.pdf")
    
    # Ensure PDF is downloaded
    if not os.path.exists(pdf_path):
        download_bns_2023(target_path=pdf_path)

    pages = load_pdf(pdf_path)

    print("PDF loaded successfully!")
    print(f"Total pages: {len(pages)}")
    print("\n--- FIRST PAGE ---\n")
    if pages:
        print(pages[0]["text"])


if __name__ == "__main__":
    main()
