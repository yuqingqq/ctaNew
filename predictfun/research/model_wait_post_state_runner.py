"""Frozen single-slot state experiment, with immutable attempts and queue cleanup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from model_wait_post_driver import busy, resume_size

DAYS = ('20261025', '20261026', '20261027', '20261031', '20261101', '20261102', '20261103')
REVIEW = ROOT / 'predictfun/reviews/wait_post_state_20261115'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(out, record):
    out.mkdir(); logs = out / 'logs'; logs.mkdir()
    freeze = json.loads((REVIEW / 'freeze.json').read_text())
    def status(state, **kw):
        p = out / 'runner.tmp'
        p.write_text(json.dumps(dict(status=state, asof=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), pid=os.getpid(), **kw), indent=2) + '\n')
        p.replace(out / 'runner.json')
    def check():
        for name, sha in freeze['files'].items():
            if digest(ROOT / name) != sha:
                raise RuntimeError('frozen source differs: ' + name)
        for name, sha in freeze['inputs'].items():
            if digest(name) != sha:
                raise RuntimeError('frozen input differs: ' + name)
        subprocess.run(['git', 'merge-base', '--is-ancestor', freeze['source_commit'], 'HEAD'], cwd=ROOT, check=True)
    def job(name, args):
        status('waiting_for_slot', stage=name)
        while busy(): time.sleep(5)
        check(); status('running', stage=name)
        with (logs / (name + '.log')).open('x') as f:
            command = [str(HERE / 'run_light.sh'), 'wait_post_state_' + name, '--', sys.executable, str(HERE / 'pf_wait_post_state.py'), *args]
            rc = subprocess.call(command, cwd=ROOT, env=dict(os.environ, PF_SLOT='codex', OMP_NUM_THREADS='1',
                                 OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1'), stdout=f, stderr=subprocess.STDOUT)
        if rc:
            raise RuntimeError(f'{name} failed: {rc}; attempted outputs preserved')
        check()
    try:
        job('selftest', ['selftest'])
        job('smoke', ['rows', DAYS[0], '--max-windows', '2', '--out', str(out / 'smoke')])
        for day in DAYS:
            job('rows_' + day, ['rows', day, '--out', str(out / 'full')])
            if day in DAYS[2:]:
                job('fit_' + day, ['fit', day, '--out', str(out / 'full')])
        job('report', ['report', '--out', str(out / 'full')])
        status('complete', result=str(out / 'full/report/result.json'), result_sha256=digest(out / 'full/report/result.json'))
    except BaseException as exc:
        status('failed', error=str(exc)); raise
    finally:
        resume_size(record)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['run', 'resume']); p.add_argument('--out', type=Path)
    p.add_argument('--resume-size-record', type=Path, required=True); a = p.parse_args()
    if a.command == 'resume': resume_size(a.resume_size_record)
    else: run(a.out, a.resume_size_record)
