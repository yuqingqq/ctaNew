import gzip,json
from pathlib import Path
H=Path(__file__).resolve().parent
x=json.loads(gzip.decompress((H/'data.json.gz').read_bytes()))
rows=[]
for m in x['cases']:
 rows.append(dict(m=m['id'],gate=m['gate'],resume=m['resume'],first_fills={k:a['first_fill'] for k,a in m['arms'].items()},first_orders={k:a['first_orders'] for k,a in m['arms'].items()},maker_first=min((f['t'] for f in m['makers']),default=None),maker_first_passive=min((f['t'] for f in m['makers'] if f['role']=='maker'),default=None)))
(H/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
