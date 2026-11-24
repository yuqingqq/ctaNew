"""Embed the checked quote projection; no new measurement or file fetching."""
import json
from pathlib import Path
H=Path(__file__).resolve().parent
d=json.loads((H/'maker3_quotes.json').read_text())
assert json.loads((H/'maker3_quote_checks.json').read_text())['checks_pass']
for r in d['rows']:
 r['first60_fills']=[{'t':f['t'],'q':f['q']} for f in r['first60_fills']]
s=(H/'maker3-quotes-template.html').read_text().replace('__DATA__',json.dumps(d,separators=(',',':')).replace('</',r'<\/'))
assert '__DATA__' not in s and len(s.encode())<1000000
(H/'maker3-quotes.html').write_text(s)
print('Generated',len(s.encode()),'bytes; full 17-order membership retained')
