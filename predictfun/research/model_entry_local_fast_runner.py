"""Freeze and execute an equivalent fast replay of the already fitted models."""
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
REVIEW=ROOT/'predictfun/reviews/entry_local_20261115'
DATA=Path('/home/yuqing/ctaNew/data/predict_fun/research/entry_local_20261115_v3')
DAYS=('20261025','20261026','20261027','20261031','20261101','20261102','20261103');TESTS=DAYS[2:]


def freeze():
    original=REVIEW/'freeze_v3.json'; f=json.loads(original.read_text());f['model_freeze_sha256']=digest(original)
    f['model_source_commit']=f['source_commit'];f['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for name in ('pf_entry_local_fast.py','model_entry_local_fast_runner.py'):
        p=Path('predictfun/research')/name;f['files'][str(p)]=digest(ROOT/p)
    for rel,h in f['files'].items():
        if digest(ROOT/rel)!=h: raise ValueError('model source changed '+rel)
        b=subprocess.check_output(['git','show',f['source_commit']+':'+rel],cwd=ROOT)
        import hashlib
        if hashlib.sha256(b).hexdigest()!=h: raise ValueError('uncommitted runtime source '+rel)
    for d in DAYS:
        for name in ('receipt.json','rows.parquet'):
            p=DATA/('rows_'+d)/name;f['inputs'][str(p)]=digest(p)
    for d in TESTS:
        fp=DATA/('fit_'+d);r=json.loads((fp/'receipt.json').read_text())
        for name in ('receipt.json','predictions.parquet',*r['models']):
            p=fp/name;f['inputs'][str(p)]=digest(p)
    f.update(asof=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),runtime_only=True,new_candidates=0,
        runtime_design='Native single-row prediction, same normal prediction/all trees/float32/one thread. Verify all evaluation rows and 24 complete replay cases against the original scorer before the full replay.',
        supersedes_runtime=dict(runner=str(DATA/'runner.json'),reason='Original replay superseded for scalar prediction overhead after all seven builds and five fits completed. No models or labels changed.'))
    for p,h in f['inputs'].items():
        if digest(p)!=h: raise ValueError('changed frozen input '+p)
    with (REVIEW/'runtime_freeze_v4.json').open('x') as stream:json.dump(f,stream,indent=2);stream.write('\n')
    print('Frozen equivalent replay sources and',len(f['inputs']),'input identities')


def run(a):
    f=json.loads(a.freeze.read_text());out=a.out;out.mkdir();logs=out/'logs';logs.mkdir()
    def check(day=None):
        for p,h in f['files'].items():
            if digest(ROOT/p)!=h: raise ValueError('changed source '+p)
        for p,h in f['inputs'].items():
            if digest(p)!=h: raise ValueError('changed input '+p)
        if day:
            for p,h in f['raw_inputs'][day].items():
                if digest(p)!=h: raise ValueError('changed raw input '+p)
    env=dict(os.environ,PF_SLOT='codex',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONSAFEPATH='1')
    steps=[('parity','parity',None,out/'parity')]+[('replay_'+d,'replay',d,out) for d in TESTS]+[('report','report',None,out)]
    def status(s,**kw):write(out/'runner.json',dict(status=s,asof=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),freeze_sha256=digest(a.freeze),**kw))
    try:
        for name,command,day,target in steps:
            status('waiting_for_slot',stage=name)
            while busy():time.sleep(3)
            rawday=TESTS[0] if command=='parity' else day;check(rawday);status('running',stage=name)
            args=[str(HERE/'run_light.sh'),'entry_local_fast_'+name,'--',sys.executable,str(HERE/'pf_entry_local_fast.py'),command]
            if day:args.append(day)
            args += ['--out',str(target)]
            with (logs/(name+'.log')).open('x') as stream:rc=subprocess.call(args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if rc:raise RuntimeError(f'{name} failed rc={rc}; attempt preserved')
            check(rawday)
        status('complete',completed_stages=[x[0] for x in steps])
    except BaseException as e:status('failed',error=str(e));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run']);p.add_argument('--out',type=Path);p.add_argument('--freeze',type=Path);a=p.parse_args()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    if a.command=='freeze':freeze()
    else:run(a)
