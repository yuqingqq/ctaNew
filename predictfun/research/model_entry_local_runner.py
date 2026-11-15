"""Durable single-core local-entry build, chronological fitting and full replay."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from model_entry_local_freeze import digest
from model_wait_post_driver import busy,write

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
DAYS=('20261025','20261026','20261027','20261031','20261101','20261102','20261103')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True);p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--mode',choices=['smoke','full'],required=True);a=p.parse_args()
    out=a.out;out.mkdir(); logs=out/'logs';logs.mkdir(); freeze=json.loads(a.freeze.read_text())
    def check(day=None):
        for rel,h in freeze['files'].items():
            if digest(ROOT/rel)!=h: raise ValueError('changed source '+rel)
        for name,h in freeze['inputs'].items():
            if digest(name)!=h: raise ValueError('changed input '+name)
        if day:
            for name,h in freeze['raw_inputs'][day].items():
                if digest(name)!=h: raise ValueError('changed raw input '+name)
    env=dict(os.environ,PF_SLOT='codex',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONSAFEPATH='1')
    steps=[('selftest','selftest',None,None)]
    if a.mode=='smoke': steps += [('build_smoke','build',DAYS[0],2)]
    else:
        steps += [('build_'+d,'build',d,None) for d in DAYS]
        steps += [('fit_'+d,'fit',d,None) for d in DAYS[2:]]
        steps += [('replay_'+d,'replay',d,None) for d in DAYS[2:]]
        steps += [('report','report',None,None)]
    def status(s,**kw): write(out/'runner.json',dict(status=s,asof=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),mode=a.mode,freeze_sha256=digest(a.freeze),**kw))
    try:
        for name,command,day,limit in steps:
            status('waiting_for_slot',stage=name)
            while busy(): time.sleep(3)
            rawday=day if command in ('build','replay') else None;check(rawday)
            status('running',stage=name)
            args=[str(HERE/'run_light.sh'),'entry_local_'+name,'--',sys.executable,str(HERE/'pf_entry_local.py'),command]
            if day: args.append(day)
            args += ['--out',str(out)]
            if limit: args += ['--max-windows',str(limit)]
            with (logs/(name+'.log')).open('x') as stream: rc=subprocess.call(args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if rc: raise RuntimeError(f'{name} failed with rc={rc}; attempt preserved')
            check(rawday)
        status('complete',completed_stages=[x[0] for x in steps])
    except BaseException as e:
        status('failed',error=str(e)); raise


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    main()
