import build as B
p=B.HERE/'fast_sources_v3.json';B.need(not p.exists(),'write once');B.dump(p,{str(x.relative_to(B.ROOT)):B.sha(x) for x in [B.HERE/y for y in ('fast_v3.py','amendment_v3.md','freeze_v3.py')]})
