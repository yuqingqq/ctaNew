"""Wait for the declared build driver, then run the already-declared read-only audits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORK=Path('/tmp/cta-default-pull-20261111');ROOT=Path('/home/yuqing/ctaNew')
sys.path.insert(0,str(WORK))
from predictfun.research import pf_continuation_spec as S
from predictfun.research.pf_continuation_v3 import configure
configure()


def main(pid):
    identity=Path(f'/proc/{pid}/cmdline')
    while True:
        p=json.loads((S.OUT/'progress.json').read_text())
        if p['state']=='failed':raise SystemExit('Main driver failed; audits not run.')
        try:command=identity.read_bytes()
        except FileNotFoundError:command=b''
        if not command:
            if (S.OUT/'result.json').exists():break
            raise SystemExit('Main driver ended without result; audits not run.')
        if b'continuation_driver_v3.py' not in command:raise SystemExit('Driver PID identity changed.')
        time.sleep(10)
    env=dict(os.environ,PYTHONPATH=str(WORK),PYTHONSAFEPATH='1',PF_SLOT='codex',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    for module,artifact in [('pf_continuation_audit','audit.json'),('pf_continuation_value_detail','value_detail.json')]:
        if (S.OUT/artifact).exists():
            if json.loads((S.OUT/artifact).read_text())['status']!='PASS':raise ValueError('failed existing audit')
            continue
        p=S.OUT/'logs'/(module+'.log')
        if p.exists():raise ValueError('prior audit attempt preserved')
        with p.open('x') as f:
            rc=subprocess.call([str(ROOT/'predictfun/research/run_light.sh'),module,'--','/home/yuqing/pricer-sol/venv/bin/python3','-m','predictfun.research.'+module,'run'],cwd=WORK,env=env,stdout=f,stderr=subprocess.STDOUT)
        if rc:
            S.progress(module,'failed',f'Audit rc={rc}; see {p}. No automatic retry.')
            raise SystemExit(rc)
    S.progress('awaiting_human_review','awaiting_review','All declared builds, forecasts, artifact reconstruction and component diagnostics are complete. Results ready for interpretation; no candidate promoted.')

if __name__=='__main__':main(int(sys.argv[1]))
