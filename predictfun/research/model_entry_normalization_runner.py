"""Serial, frozen, read-only audit in the Codex light slot."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from model_wait_post_driver import busy,digest,resume_size,write


def run(out,resume):
    out.mkdir();logs=out/'logs';logs.mkdir()
    freeze=json.loads((ROOT/'predictfun/reviews/entry_normalization_20261115/freeze.json').read_text())
    def check():
        for p,h in freeze['files'].items():
            if digest(ROOT/p)!=h:raise ValueError('source changed: '+p)
        for p,h in freeze['inputs'].items():
            if digest(p)!=h:raise ValueError('input changed: '+p)
    def status(s,**kw):write(out/'runner.json',dict(status=s,asof=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))
    env=dict(os.environ,PF_SLOT='codex',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    try:
        for command in ('selftest','audit'):
            status('waiting_for_slot',stage=command)
            while busy():time.sleep(3)
            check();status('running',stage=command)
            with (logs/(command+'.log')).open('x') as f:
                rc=subprocess.call([str(HERE/'run_light.sh'),'entry_normalization_'+command,'--',sys.executable,
                   str(HERE/'pf_entry_normalization_audit.py'),command,'--out',str(out/'audit')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            if rc:raise RuntimeError(f'{command} failed: {rc}; attempt preserved')
            check()
        status('complete',result=str(out/'audit/result.json'))
    except BaseException as exc:
        status('failed',error=str(exc));raise
    finally:resume_size(resume)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--resume-record',type=Path,required=True)
    a=p.parse_args();signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    run(a.out,a.resume_record)
