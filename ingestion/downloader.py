import os
import json
import ssl
import urllib.request
from datetime import datetime

# Authoritative BNS 2023 Metadata
BNS_METADATA = {
    "document_id": "BNS_2023",
    "title": "Bharatiya Nyaya Sanhita, 2023",
    "act_no": "Act No. 45 of 2023",
    "source_url": "https://www.indiacode.nic.in/handle/123456789/21267",
    "download_url": "https://prsindia.org/files/bills_acts/bills_parliament/2023/Bharatiya_Nyaya_Sanhita,_2023.pdf",
    "source_domain": "indiacode.nic.in",
    "local_file": "data/raw/bns_2023.pdf"
}


def verify_pdf(file_path: str) -> bool:
    """Check if the file exists and has valid PDF header bytes."""
    if not os.path.exists(file_path):
        return False
    with open(file_path, 'rb') as f:
        header = f.read(5)
    return header.startswith(b'%PDF')


def update_metadata(metadata_path: str = "data/metadata/sources.json", doc_meta: dict = None):
    """Save or update document metadata in metadata/sources.json."""
    if doc_meta is None:
        doc_meta = BNS_METADATA

    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    sources = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                sources = json.load(f)
        except Exception:
            sources = {}

    doc_meta["retrieved_at"] = datetime.now().isoformat()
    sources[doc_meta["document_id"]] = doc_meta

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(sources, f, indent=4, ensure_ascii=False)


def download_bns_2023(target_path: str = "data/raw/bns_2023.pdf", force: bool = False) -> str:
    """
    Downloads the official Bharatiya Nyaya Sanhita, 2023 PDF file.
    Skips download if file exists and is valid, unless force=True.
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    if not force and verify_pdf(target_path):
        file_size = os.path.getsize(target_path)
        print(f"[Downloader] File already exists: {target_path} ({file_size} bytes)")
        update_metadata(doc_meta=BNS_METADATA)
        return target_path

    print(f"[Downloader] Downloading official BNS 2023 PDF from {BNS_METADATA['download_url']}...")
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    req = urllib.request.Request(BNS_METADATA['download_url'], headers=headers)
    with urllib.request.urlopen(req, context=ctx, timeout=30) as response, open(target_path, 'wb') as out_file:
        out_file.write(response.read())

    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Download failed, file not created at {target_path}")

    if not verify_pdf(target_path):
        os.remove(target_path)
        raise ValueError(f"Downloaded file at {target_path} is not a valid PDF")

    file_size = os.path.getsize(target_path)
    print(f"[Downloader] PDF downloaded successfully: {target_path} ({file_size} bytes)")
    update_metadata(doc_meta=BNS_METADATA)
    return target_path


if __name__ == "__main__":
    download_bns_2023()
