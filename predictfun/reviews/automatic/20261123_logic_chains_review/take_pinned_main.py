"""Work around receipt relative_to(REPO) in detached checkout, with pinned-byte checks."""
import hashlib
import os
import subprocess as sp
from pathlib import Path
ROOT=Path('/home/yuqing/ctaNew')
PIN=Path('/tmp/pf-chains-review-8d68808d')
TOOL='predictfun/reviews/maker_trace_20261122/tools/'
files=[TOOL+n+'.py' for n in ('chain_take','chain_take_selftest','chain_context','order_estimator','chain_pull','closes')]
files += ['predictfun/research/'+n+'.py' for n in ('pf_mt_common','pf_mt_econ','pf_mt_lifecycle','pf_mt_decision')]
files += ['predictfun/reviews/maker_trace_20261122/order_estimator_contract.md']
def verify():
    for f in files:
        assert (ROOT/f).read_bytes()==(PIN/f).read_bytes(), f
verify()
print('Pinned closure byte-equal before main-path test; only metadata path differs.',flush=True)
r=sp.run(['/home/yuqing/pricer-sol/venv/bin/python',str(ROOT/(TOOL+'chain_take_selftest.py'))],env=dict(os.environ,MT_RUN='v2'),timeout=20)
verify()
print('Pinned closure byte-equal after test.',flush=True)
raise SystemExit(r.returncode)
