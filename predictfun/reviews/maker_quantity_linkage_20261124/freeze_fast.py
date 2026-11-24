import json
import build as B
ps=[B.HERE/x for x in ('fast.py','amendment_v2.md','profile_pilot.py','freeze_fast.py')]
p=B.HERE/'fast_sources.json';B.need(not p.exists(),'write once');B.dump(p,{str(x.relative_to(B.ROOT)):B.sha(x) for x in ps})
