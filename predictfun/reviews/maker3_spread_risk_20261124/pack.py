import base64,gzip,json
from pathlib import Path
import core as C
data=json.loads(gzip.decompress((C.OUT/'data.json.gz').read_bytes()));data['risk']=json.loads((C.H/'risk.json').read_text())
payload=gzip.compress(json.dumps(data,separators=(',',':'),allow_nan=False).encode(),mtime=0)
template=(C.H/'view.html').read_text();script=(C.H/'view.js').read_text();html=template.replace('__PAYLOAD__',base64.b64encode(payload).decode()).replace('__SCRIPT__',script)
assert '__PAYLOAD__' not in html and '__SCRIPT__' not in html
assert json.loads(gzip.decompress(base64.b64decode(html.split('<script id="payload" type="application/octet-stream">')[1].split('</script>')[0])))==data
p=C.H.parent/'maker_dashboard_20261124/maker3-research.html';p.write_text(html)
C.dump(C.H/'chart.json',dict(path=str(p.relative_to(C.R)),sha256=C.sha(p),bytes=p.stat().st_size,markets=len(data['cases']),trend=data['trend'],source_sha256=C.sha(C.OUT/'data.json.gz'),risk_sha256=C.sha(C.H/'risk.json')))
print(p,p.stat().st_size,'bytes; payload exact round trip')
