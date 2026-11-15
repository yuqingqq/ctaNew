"""Run the cleanup comparison after final labels, then resume the frozen comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from model_wait_post_driver import busy, resume_size


def run(out, record):
    out.mkdir(); logs = out / 'logs'; logs.mkdir()
    freeze = json.loads((ROOT / 'predictfun/reviews/wait_post_state_20261115/clean_freeze.json').read_text())
    def check():
        for name, sha in freeze['files'].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
                raise ValueError('frozen diagnostic source differs')
        for name, sha in freeze['inputs'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
                raise ValueError('frozen diagnostic input differs')
    def status(value, **kw):
        p = out / 'runner.tmp'; p.write_text(json.dumps(dict(status=value, asof=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw), indent=2) + '\n')
        p.replace(out / 'runner.json')
    try:
        status('waiting_for_final_label_stage')
        parent_status = Path('/home/yuqing/ctaNew/data/predict_fun/research/wait_post_state_20261115_v1/runner.json')
        while True:
            state = json.loads(parent_status.read_text())
            if state['status'] == 'failed': raise RuntimeError('label producer failed')
            if state.get('stage') in ('rows_20261103', 'fit_20261103', 'report'): break
            if state['status'] == 'complete': raise RuntimeError('producer already complete; do not signal an expired supervisor')
            time.sleep(2)
        check(); r = json.loads(record.read_text()); p = Path('/proc') / str(r['pid'])
        if (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode() != r['cmdline'] or (p / 'stat').read_text().split()[21] != r['starttime']:
            raise ValueError('supervisor identity differs')
        os.kill(r['pid'], signal.SIGSTOP)
        for command in ('selftest', 'probes'):
            status('waiting_for_slot', stage=command)
            while busy(): time.sleep(3)
            check(); status('running', stage=command)
            with (logs / (command + '.log')).open('x') as f:
                rc = subprocess.call([str(HERE / 'run_light.sh'), 'wait_post_clean_' + command, '--', sys.executable,
                                      str(HERE / 'pf_wait_post_state_clean.py'), command, '--out', str(out / 'probes')],
                                     cwd=ROOT, env=dict(os.environ, PF_SLOT='codex', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1'), stdout=f, stderr=subprocess.STDOUT)
            if rc: raise RuntimeError(f'{command} failed: {rc}; attempt preserved')
            check()
        status('complete', result=str(out / 'probes/result.json'))
    except BaseException as exc:
        status('failed', error=str(exc)); raise
    finally:
        resume_size(record)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['run', 'resume']); p.add_argument('--out', type=Path)
    p.add_argument('--resume-record', type=Path, required=True); a = p.parse_args()
    if a.command == 'resume': resume_size(a.resume_record)
    else: run(a.out, a.resume_record)
