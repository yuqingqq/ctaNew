"""Durable serial entry-repair job; immutable attempts and source/input checks."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from model_wait_post_driver import busy,digest,resume_size,write
DAYS=('20261025','20261026','20261027','20261031','20261101','20261102','20261103')


def run(a):
    out=a.out;out.mkdir();logs=out/'logs';logs.mkdir()
    freeze=json.loads(a.freeze.read_text())
    def check(day=None):
        for p,h in freeze['files'].items():
            if digest(ROOT/p)!=h:raise ValueError('changed frozen source: '+p)
        for p,h in freeze['inputs'].items():
            if digest(p)!=h:raise ValueError('changed frozen input: '+p)
        if day:
            for p,h in freeze['raw_inputs'][day].items():
                if digest(p)!=h:raise ValueError('changed admissible raw input: '+p)
    env=dict(os.environ,PF_SLOT='codex',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONSAFEPATH='1')
    steps=[('selftest','selftest',None,None)]
    if a.mode=='smoke':
        steps += [('validity_smoke','validity',DAYS[0],2),('density_smoke','build',DAYS[0],2)]
    else:
        steps += [('validity_'+d,'validity',d,None) for d in DAYS]
        steps += [('build_'+d,'build',d,None) for d in DAYS]
        steps += [('fit_'+d,'measure',d,None) for d in DAYS[2:]]
        steps += [('report','report',None,None)]
    def status(s,**kw):write(out/'runner.json',dict(status=s,asof=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),mode=a.mode,freeze_sha256=digest(a.freeze),**kw))
    try:
        for name,cmd,day,limit in steps:
            status('waiting_for_slot',stage=name)
            while busy():time.sleep(3)
            check(day if cmd in ('validity','build') else None);status('running',stage=name)
            args=[str(HERE/'run_light.sh'),'entry_repair_'+name,'--',sys.executable,str(HERE/'pf_entry_repair.py'),cmd]
            if day:args.append(day)
            args += ['--out',str(out)]
            if limit:args += ['--max-windows',str(limit)]
            with (logs/(name+'.log')).open('x') as f:rc=subprocess.call(args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            if rc:raise RuntimeError(f'{name} failed with {rc}; attempt preserved')
            check(day if cmd in ('validity','build') else None)
        status('complete',completed_stages=[x[0] for x in steps])
    except BaseException as e:
        status('failed',error=str(e));raise
    finally:resume_size(a.resume_record)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','full'],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--freeze',type=Path,required=True);p.add_argument('--resume-record',type=Path);a=p.parse_args()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    run(a)
