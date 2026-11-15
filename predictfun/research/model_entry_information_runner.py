"""Run the frozen entry-information study serially in the Codex light slot."""
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
from model_wait_post_driver import busy, digest, resume_size, write

DAYS=('20261025','20261026','20261027','20261031','20261101','20261102','20261103')


def run(out, resume):
    out.mkdir(); logs=out/'logs'; logs.mkdir()
    freeze=json.loads((ROOT/'predictfun/reviews/entry_information_20261115/freeze_v2.json').read_text())
    def check(day=None):
        for p,h in freeze['files'].items():
            if digest(ROOT/p)!=h:raise ValueError('source changed: '+p)
        for p,h in freeze['inputs'].items():
            if digest(p)!=h:raise ValueError('label input changed: '+p)
        if day:
            for p,h in freeze['raw_inputs'][day].items():
                if digest(p)!=h:raise ValueError('raw input changed: '+p)
    def status(s,**kw):write(out/'runner.json',dict(status=s,asof=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))
    env=dict(os.environ,PF_SLOT='codex',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONSAFEPATH='1')
    steps=[('selftest','selftest',None)]+[('build_'+d,'build',d) for d in DAYS]
    steps += [('measure_'+d,'measure',d) for d in DAYS[2:]]+[('report','report',None)]
    try:
        for name,command,day in steps:
            status('waiting_for_slot',stage=name)
            while busy():time.sleep(3)
            check(day if command=='build' else None);status('running',stage=name)
            args=[str(HERE/'run_light.sh'),'entry_information_'+name,'--',sys.executable,str(HERE/'pf_entry_information.py'),command]
            if day:args.append(day)
            args += ['--out',str(out)]
            with (logs/(name+'.log')).open('x') as f:
                rc=subprocess.call(args,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            if rc:raise RuntimeError(f'{name} failed with {rc}; attempt preserved')
            check(day if command=='build' else None)
        status('complete',result=str(out/'report/result.json'))
    except BaseException as exc:
        status('failed',error=str(exc));raise
    finally:resume_size(resume)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--resume-record',type=Path,required=True)
    a=p.parse_args();signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    run(a.out,a.resume_record)
