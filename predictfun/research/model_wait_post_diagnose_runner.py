"""One supervised diagnostic worker on the assigned codex/core8 slot.

No source changes after freeze; no automatic overwrite/retry. The original size
queue yields after its current worker and resumes on completion/failure. Core9
belongs to the other session. ExecStopPost also invokes the resume helper.
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
ROOT=HERE.parents[1]
REVIEW=ROOT/'predictfun/reviews/wait_post_diagnosis_20261114/v3'
sys.path.insert(0,str(HERE))
from model_wait_post_driver import resume_size
DAYS=('20261027','20261031','20261101','20261102','20261103')


def busy():
    count=0;occupied=False
    for p in Path('/proc').glob('[0-9]*'):
        try:
            a=(p/'cmdline').read_bytes().split(b'\0')
            if not a or not Path(os.fsdecode(a[0])).name.startswith('python'):continue
            cmd=b' '.join(a)
            if b'predictfun.research.' not in cmd and b'predictfun/research/pf_' not in cmd:continue
            count+=1
            affinity=next(z.split(':',1)[1].strip() for z in (p/'status').read_text().splitlines() if z.startswith('Cpus_allowed_list:'))
            occupied|=affinity=='8'
        except (FileNotFoundError,ProcessLookupError,PermissionError,StopIteration):continue
    return occupied or count>=2


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(out,resume_record):
    out.mkdir();logs=out/'logs';logs.mkdir();state=out/'runner.json';freeze=json.loads((REVIEW/'freeze.json').read_text())
    def update(status,**kw):
        p=state.with_suffix('.tmp');p.write_text(json.dumps(dict(status=status,asof=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),**kw),indent=2)+'\n');p.replace(state)
    def check():
        for n,h in freeze['files'].items():
            if digest(ROOT/n)!=h:raise RuntimeError('diagnostic frozen source changed: '+n)
        subprocess.run(['git','merge-base','--is-ancestor',freeze['source_commit'],'HEAD'],cwd=ROOT,check=True)
    jobs=[('selftest',['selftest']),('saved',['saved'])]+[('replay_'+d,['replay',d]) for d in DAYS]+[('finish',['finish'])]
    env=dict(os.environ,PF_SLOT='codex',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    try:
        for name,args in jobs:
            update('waiting_for_slot',stage=name)
            while busy():time.sleep(5)
            check();update('running',stage=name)
            with (logs/(name+'.log')).open('x') as f:
                rc=subprocess.call([str(HERE/'run_light.sh'),'wait_post_diagnose_'+name,'--',sys.executable,str(HERE/'pf_wait_post_diagnose.py'),*args,'--out',str(out)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            if rc:raise RuntimeError(name+' failed; exit'+str(rc))
            check()
        p=out/'report/result.json';update('complete',result=str(p),result_sha256=digest(p))
    except BaseException as exc:update('failed',error=str(exc));raise
    finally:resume_size(resume_record)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--resume-size-record',type=Path,required=True);a=p.parse_args();run(a.out,a.resume_size_record)
