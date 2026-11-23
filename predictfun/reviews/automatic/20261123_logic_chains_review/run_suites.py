"""Pinned synthetic checks only; run once through PF_SLOT=codex/run_light."""
import json
import os
import resource
import subprocess as sp
import time
from pathlib import Path
H=Path(__file__).resolve().parent
P=Path('/tmp/pf-chains-review-8d68808d/predictfun/reviews/maker_trace_20261122/tools')
PY='/home/yuqing/pricer-sol/venv/bin/python'
env=dict(os.environ, MT_RUN='v2')
out=[]
for name in ('chain_context','order_estimator','chain_post_fill','chain_pull','chain_take'):
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    with (H/(name+'_selftest.log')).open('w') as f:
        r=sp.run([PY,str(P/(name+'_selftest.py'))],env=env,stdout=f,stderr=sp.STDOUT,timeout=20)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    out.append(dict(name=name,rc=r.returncode,cpu_s=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime))
    assert r.returncode==0, out
    assert sum(x['cpu_s'] for x in out)<30, 'review synthetic CPU ceiling'
(H/'suites.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
