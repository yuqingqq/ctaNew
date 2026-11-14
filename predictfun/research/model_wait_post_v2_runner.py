"""Supervised v2 pipeline. Unit ExecStopPost must call `resume` as a second guard.

Synthetic tests run first. Historical work waits for a committed freeze manifest.
Every worker uses codex/core8 via run_light. Failed artifacts are preserved;
no stage is overwritten or automatically retried. Frozen source identity is checked
before and after every historical worker.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from model_wait_post_driver import busy,resume_size
DAYS=('20261023','20261024','20261025','20261026','20261027','20261031','20261101','20261102','20261103')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(a):
    a.out.mkdir(parents=True,exist_ok=True);logs=a.out/'logs';logs.mkdir(exist_ok=True);state=a.out/'runner.json'
    def update(status,**kw):
        tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(dict(status=status,pid=os.getpid(),asof=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw),indent=2)+'\n');tmp.replace(state)
    env=dict(os.environ,PF_SLOT='codex',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    def job(name,args):
        update('waiting_for_slot',stage=name)
        while busy():time.sleep(10)
        log=logs/(name+'.log')
        if log.exists():raise RuntimeError('existing attempted stage/log: '+name)
        update('running',stage=name)
        with log.open('x') as f:
            rc=subprocess.call([str(HERE/'run_light.sh'),'wait_post_v2_'+name,'--',sys.executable,str(HERE/'pf_wait_post_v2.py'),*args],
                cwd=HERE.parents[1],env=env,stdout=f,stderr=subprocess.STDOUT)
        if rc:raise RuntimeError(f'{name} failed, exit{rc}; preserved outputs; no retry')
    try:
        job('selftest',['selftest'])
        manifest=HERE.parent/'reviews/wait_post_v2_20261114/freeze.json';deadline=time.monotonic()+1200
        while not manifest.exists():
            update('waiting_for_committed_freeze')
            if time.monotonic()>deadline:raise RuntimeError('freeze wait timed out')
            time.sleep(10)
        frozen=json.loads(manifest.read_text())
        def check():
            for name,h in frozen['files'].items():
                if sha(HERE.parents[1]/name)!=h:raise RuntimeError('frozen source changed: '+name)
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE.parents[1],text=True).strip()
            if subprocess.call(['git','merge-base','--is-ancestor',frozen['source_commit'],commit],cwd=HERE.parents[1])!=0:
                raise RuntimeError('source commit not in checkout history')
        check();job('smoke',['rows',DAYS[0],'--kind','flat','--max-windows','2','--out',str(a.out/'smoke')]);check()
        jobs=[]
        for d in DAYS[:2]:jobs.append(('rows_flat_'+d,['rows',d,'--kind','flat']))
        jobs.append(('bootstrap',['bootstrap']))
        for d in DAYS[2:4]:jobs.append(('rows_learned_'+d,['rows',d,'--kind','learned']))
        for d in DAYS[4:]:
            jobs.extend([('rows_learned_'+d,['rows',d,'--kind','learned']),('fit_'+d,['fit',d]),('eval_'+d,['eval',d])])
        jobs.extend([('portfolio_null',['null']),('report',['report'])])
        for name,args in jobs:
            check();job(name,args+['--out',str(a.out/'full')]);check()
        result=a.out/'full/report/result.json'
        if not result.exists():raise RuntimeError('no final result')
        update('complete',result=str(result),result_sha256=sha(result))
    except BaseException as exc:
        update('failed',error=str(exc));raise
    finally:resume_size(a.resume_size_record)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','resume']);p.add_argument('--out',type=Path)
    p.add_argument('--resume-size-record',type=Path,required=True);a=p.parse_args()
    if a.command=='resume':resume_size(a.resume_size_record)
    else:run(a)
