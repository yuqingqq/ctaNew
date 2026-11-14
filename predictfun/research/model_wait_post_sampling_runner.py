"""Assigned-slot supervisor; old size worker finishes, queue then resumes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from model_wait_post_driver import busy,resume_size
REVIEW=ROOT/'predictfun/reviews/wait_post_signal_data_20261114/sampling'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(out,record):
    out.mkdir();logs=out/'logs';logs.mkdir();freeze=json.loads((REVIEW/'freeze.json').read_text())
    def status(state,**kw):
        p=out/'runner.tmp';p.write_text(json.dumps(dict(status=state,asof=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),**kw),indent=2)+'\n');p.replace(out/'runner.json')
    def check():
        for n,h in freeze['files'].items():
            if digest(ROOT/n)!=h:raise RuntimeError('frozen source changed: '+n)
        subprocess.run(['git','merge-base','--is-ancestor',freeze['source_commit'],'HEAD'],cwd=ROOT,check=True)
    try:
        check();r=json.loads(record.read_text());proc=Path('/proc')/str(r['pid'])
        if (proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()!=r['cmdline'] or (proc/'stat').read_text().split()[21]!=r['starttime']:raise RuntimeError('size queue identity differs')
        os.kill(r['pid'],signal.SIGSTOP)
        env=dict(os.environ,PF_SLOT='codex',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        for name in ('selftest','probes'):
            status('waiting_for_slot',stage=name)
            while busy():time.sleep(5)
            check();status('running',stage=name)
            with (logs/(name+'.log')).open('x') as f:
                rc=subprocess.call([str(HERE/'run_light.sh'),'wait_post_sampling_'+name,'--',sys.executable,str(HERE/'pf_wait_post_sampling.py'),name,'--out',str(out)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            if rc:raise RuntimeError(name+' failed: '+str(rc))
            check()
        status('complete',results={n:digest(out/n/'result.json') for n in ('probes',)})
    except BaseException as exc:status('failed',error=str(exc));raise
    finally:resume_size(record)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--resume-size-record',type=Path,required=True);a=p.parse_args();run(a.out,a.resume_size_record)
