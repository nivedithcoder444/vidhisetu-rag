import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
pat = re.compile(
    r'^\s*(?:[^\w\s]*\d*\[|\*+)?\s*(?:Section|Article|Sec\.)?\s*(\d+[A-Za-z]{0,2})\.\s*([A-Za-z\(\'\"—])'
)
samples = [
    '1[370. Temporary provisions with respect to the State of Jammu',
    '1[371. Special provision with respect to the States of',
    '369. Temporary power to Parliament',
    '243ZA. Elections to the Municipalities.',
    'Section 73. Compensation for loss',
    '1. Short title, commencement'
]
for s in samples:
    m = pat.match(s)
    print(s[:35], '-> Match:', m.group(1) if m else None)
