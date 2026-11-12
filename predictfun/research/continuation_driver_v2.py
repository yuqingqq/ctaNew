"""Run the unchanged declared study with corrected fork event ordering."""
import json
import os
from pathlib import Path
import subprocess
import sys

WORK=Path('/tmp/cta-default-pull-20261111');ROOT=Path('/home/yuqing/ctaNew')
sys.path.insert(0,str(WORK))
from predictfun.research import pf_continuation_spec as S
from predictfun.research.pf_continuation_v2 import configure
configure()


def run():
    S.check()
    tasks=[('smoke_20261023',['build','20261023','smoke'])]
    tasks += [('build_'+d,['build',d]) for d in S.DAYS]
    tasks += [('result',['fit'])]
    env=dict(os.environ,PYTHONPATH=str(WORK),PYTHONSAFEPATH='1',PF_SLOT='codex',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    for tag,args in tasks:
        receipt=S.OUT/(tag+'.json')
        if receipt.exists():
            if json.loads(receipt.read_text())['status']!='PASS':raise ValueError('failed receipt')
            continue
        log=S.OUT/'logs'/(tag+'.log');log.parent.mkdir(exist_ok=True)
        if log.exists():raise ValueError('partial stage preserved; supersede before retry')
        S.progress(tag,'running','v2 fork ordering; one worker in Codex slot/core 8.')
        with log.open('x') as f:
            rc=subprocess.call([str(ROOT/'predictfun/research/run_light.sh'),'continuation_v2_'+tag,'--','/home/yuqing/pricer-sol/venv/bin/python3','-m','predictfun.research.pf_continuation_v2',*args],cwd=WORK,env=env,stdout=f,stderr=subprocess.STDOUT)
        if rc or not receipt.exists():
            S.progress(tag,'failed',f'Worker rc={rc}; log {log}; output retained, no automatic retry.')
            raise SystemExit(rc or 1)
    S.progress('awaiting_review','complete','All v2 builds and forecasts complete; human review pending.')

if __name__=='__main__':run()
