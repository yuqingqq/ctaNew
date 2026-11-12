"""Durable sequential driver; numerical workers occupy only the Codex research slot."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time

WORK=Path('/tmp/cta-default-pull-20261111')
ROOT=Path('/home/yuqing/ctaNew')
OUT=ROOT/'data/predict_fun/research/continuation_20261112'
PYTHON='/home/yuqing/pricer-sol/venv/bin/python3'
DAYS=['20261023','20261024','20261025','20261026','20261027','20261031','20261101','20261102','20261103']


def run():
    import sys
    sys.path.insert(0,str(WORK))
    from predictfun.research import pf_continuation_spec as S
    S.check()
    tasks=[('smoke_20261023',['-m','predictfun.research.pf_continuation_build','20261023','smoke'])]
    tasks += [('build_'+d,['-m','predictfun.research.pf_continuation_build',d]) for d in DAYS]
    tasks += [('result',['-m','predictfun.research.pf_continuation_model'])]
    env=dict(os.environ,PYTHONPATH=str(WORK),PF_SLOT='codex',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    for tag,args in tasks:
        receipt=OUT/(tag+'.json')
        if receipt.exists():
            r=json.loads(receipt.read_text())
            if r['status']!='PASS':raise ValueError('cannot resume failed receipt')
            continue
        path=OUT/'logs'/(tag+'.log');path.parent.mkdir(exist_ok=True)
        if path.exists():raise ValueError('failed/partial stage preserved; supersede before retry: '+tag)
        S.progress(tag,'running','Launching one worker in Codex slot/core 8. Other research slot is left available.')
        with path.open('x') as log:
            rc=subprocess.call([str(ROOT/'predictfun/research/run_light.sh'),'continuation_'+tag,'--',PYTHON,*args],cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
        if rc or not receipt.exists():
            S.progress(tag,'failed',f'Worker rc={rc}; log {path}; original output retained. No automatic numerical retry.')
            raise SystemExit(rc or 1)
    S.progress('awaiting_review','complete','All declared builds and forecasts complete. Human review required before any further experiment or promotion.')

if __name__=='__main__':run()
