import sys, os
sys.path.insert(0, os.path.abspath("."))
from ingestion.section_parser import parse_sections

arms_secs = parse_sections('data/raw/extracted/arms act 1959.pdf', document_id='ARMS_ACT_1959', act_name='Arms Act 1959', year=1959, output_path=None)
print('Arms Act parsed sections count:', len(arms_secs))
for s in arms_secs[:5]:
    print(f"  Sec {s['section']}: Pages {s['page_start']}-{s['page_end']}, Title: {s['section_title']}, Text len: {len(s['text'])}")
sec3 = [s for s in arms_secs if s['section'] == '3']
if sec3:
    print('Arms Act Sec 3 text preview:', repr(sec3[0]['text'][:300]))

contract_secs = parse_sections('data/raw/extracted/indian contract act 1872.pdf', document_id='INDIAN_CONTRACT_ACT_1872', act_name='Indian Contract Act 1872', year=1872, output_path=None)
print('Contract Act parsed sections count:', len(contract_secs))
sec73 = [s for s in contract_secs if s['section'] == '73']
if sec73:
    print(f"Contract Act Sec 73 Pages: {sec73[0]['page_start']}-{sec73[0]['page_end']}, Title: {sec73[0]['section_title']}")
    print('Contract Act Sec 73 text preview:', repr(sec73[0]['text'][:300]))
