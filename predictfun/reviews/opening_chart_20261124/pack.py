import base64,gzip,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2];UI=R/'predictfun/reviews/maker_dashboard_20261124'
raw=(H/'data.json.gz').read_bytes();enc=base64.b64encode(raw).decode()
html=(UI/'opening.template.html').read_text().replace('__DATA__',enc)
html=html.replace('<link rel="stylesheet" href="opening.css">','<style>'+(UI/'opening.css').read_text()+'</style>').replace('<script src="opening.js"></script>','<script>'+(UI/'opening.js').read_text()+'</script>')
# The running server is rooted at the dashboard directory: use an in-root copy.
html=html.replace('../opening_chart_20261124/report.md','opening_report.md')
assert '__DATA__' not in html and len(html)<1_048_576
(UI/'opening.html').write_text(html)
print('self-contained page bytes',len(html.encode()),'data sha256',hashlib.sha256(gzip.decompress(raw)).hexdigest())
