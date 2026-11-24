"""Portable UI packaging; decoded data identical to verified projections."""
from pathlib import Path
import gzip,base64,hashlib
h=Path(__file__).resolve().parent
html=(h/'index.html').read_text();raw=gzip.decompress((h/'data.json.gz').read_bytes());enc=base64.b64encode(gzip.compress(raw,compresslevel=9,mtime=0)).decode();extra=(h/'extra.js').read_text().removeprefix('window.MM_EXTRA=').rstrip(';\n');ee=base64.b64encode(gzip.compress(extra.encode(),compresslevel=9,mtime=0)).decode()
app=(h/'app.js').read_text().replace("const allowed=s=>", "window.MM_EXTRA=JSON.parse(await new Response(new Blob([Uint8Array.from(atob(window.MM_EXTRA_GZIP),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).text());\nconst allowed=s=>")
html=html.replace('<link rel="stylesheet" href="style.css">','<style>'+(h/'style.css').read_text()+'</style>').replace('<script src="data.js"></script>','<script>window.MM_DATA_GZIP="'+enc+'";</script>').replace('<script src="extra.js"></script>','<script>window.MM_EXTRA_GZIP="'+ee+'";</script>').replace('<script src="app.js"></script>','<script>'+app+'</script>')
assert len(html.encode())<1048576
assert gzip.decompress(base64.b64decode(enc))==raw
(h/'dashboard.html').write_text(html);print('portable HTML bytes',len(html.encode()),'decoded data SHA',hashlib.sha256(raw).hexdigest())
