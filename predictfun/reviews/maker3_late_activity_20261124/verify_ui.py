"""Check small annotation update without rerunning browser/render or economics."""
import base64,gzip,hashlib,json,re,resource,urllib.request
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2];U=R/'predictfun/reviews/maker_dashboard_20261124'
resource.setrlimit(resource.RLIMIT_CPU,(5,6))
out={}
for file in ['dashboard.html','opening.html']:
    s=(U/file).read_text()
    match=re.search(r'window.MAKER3_TAKES=(.*?);</script>',s);assert match
    notes=json.loads(match.group(1));assert len(notes)==13
    rows=[v for k,v in notes.items() if k.startswith('3789484|')]
    assert len(rows)==7 and sum(r['action']=='BUY NO' for r in rows)==5
    assert 'Native active action' in s and 'YES-equivalent side' in s
    served=urllib.request.urlopen('http://127.0.0.1:8765/'+file,timeout=3).read()
    assert served==(U/file).read_bytes()
    out[file]=dict(bytes=len(served),native_annotations=len(notes),served_exact=True)
s=(U/'dashboard.html').read_text()
raw=gzip.decompress(base64.b64decode(re.search(r'window.MM_DATA_GZIP="(.*?)"',s).group(1)))
assert raw==gzip.decompress((U/'data.json.gz').read_bytes())
s=(U/'opening.html').read_text()
raw2=gzip.decompress(base64.b64decode(re.search(r'<script[^>]*id="data"[^>]*>(.*?)</script>',s,re.S).group(1).strip()))
assert raw2==gzip.decompress((R/'predictfun/reviews/opening_chart_20261124/data.json.gz').read_bytes())
out['unchanged_decoded_payloads']=[hashlib.sha256(raw).hexdigest(),hashlib.sha256(raw2).hexdigest()]
(H/'ui_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
