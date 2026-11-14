"""Supervised bounded review runner. Invoke under a user systemd service.

The unit must also invoke `resume` in ExecStopPost so a killed interpreter cannot
strand the older queue. Failed/abandoned attempts are preserved as receipts.
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
from model_wait_post_driver import resume_size


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','resume'])
    p.add_argument('--out',type=Path);p.add_argument('--resume-size-record',type=Path,required=True)
    a=p.parse_args()
    if a.command=='resume': resume_size(a.resume_size_record); return
    a.out.mkdir(); status=a.out/'supervisor.json'
    def write(state,**more):
        temp=status.with_suffix('.tmp'); temp.write_text(json.dumps(dict(status=state,pid=os.getpid(),asof=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**more),indent=2)+'\n');temp.replace(status)
    try:
        manifest=json.loads((HERE.parent/'reviews/wait_post_review_20261114/freeze.json').read_text())
        for name,sha in manifest['source_sha256'].items():
            if hashlib.sha256((HERE.parents[1]/name).read_bytes()).hexdigest()!=sha: raise RuntimeError('source differs: '+name)
        write('running',phase='pilot')
        env=dict(os.environ,PF_SLOT='codex',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        with (a.out/'pilot.log').open('x') as log:
            rc=subprocess.call([str(HERE/'run_light.sh'),'wait_post_review_pilot','--',sys.executable,
                str(HERE/'pf_wait_post_review.py'),'pilot','--out',str(a.out/'pilot')],cwd=HERE.parents[1],env=env,stdout=log,stderr=subprocess.STDOUT)
        if rc: raise RuntimeError('pilot failed with exit '+str(rc))
        if not (a.out/'pilot/result.json').exists(): raise RuntimeError('no result receipt')
        write('complete',result=str(a.out/'pilot/result.json'))
    except BaseException as exc:
        write('failed',error=str(exc));raise
    finally: resume_size(a.resume_size_record)


if __name__=='__main__':main()
