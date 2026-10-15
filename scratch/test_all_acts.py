import zipfile
import re
import fitz as pymupdf

with zipfile.ZipFile('AI Agent.zip', 'r') as z:
    for name in sorted(z.namelist()):
        if name.endswith('.pdf'):
            data = z.read(name)
            doc = pymupdf.open(stream=data, filetype='pdf')
            full_text = '\n'.join(doc[p].get_text() for p in range(len(doc)))
            
            # Test regex patterns
            secs = re.findall(r'^\s*(\d+[A-Z]?)\.\s*([A-Za-z\(\"\'\–\—])', full_text, re.MULTILINE)
            secs_word = re.findall(r'^\s*Section\s+(\d+[A-Z]?)', full_text, re.IGNORECASE | re.MULTILINE)
            articles = re.findall(r'^\s*Article\s+(\d+[A-Z]?)', full_text, re.IGNORECASE | re.MULTILINE)
            
            total = len(secs) + len(secs_word) + len(articles)
            print(f"{name:45s} | Pages: {len(doc):3d} | Sections detected: {total:4d}")
