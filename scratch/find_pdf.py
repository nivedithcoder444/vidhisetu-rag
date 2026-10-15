import re

content_file = r'C:\Users\nived\.gemini\antigravity\brain\8796d365-bc08-44aa-aa67-a095823944fe\.system_generated\steps\224\content.md'

with open(content_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Find relative or absolute pdf links
links = re.findall(r'href=["\']([^"\']+\.pdf)["\']', text, re.IGNORECASE)
print(f"Found {len(links)} pdf links:")
for l in set(links):
    if l.startswith('/'):
        print('https://prsindia.org' + l)
    else:
        print(l)
