"""Briefly yield our comparison supervisor at a worker boundary, then resume it."""
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
    freeze = json.loads((ROOT / 'predictfun/reviews/wait_post_state_20261115/noise_freeze.json').read_text())
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
        check(); r = json.loads(record.read_text()); p = Path('/proc') / str(r['pid'])
        if (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode() != r['cmdline'] or (p / 'stat').read_text().split()[21] != r['starttime']:
            raise ValueError('supervisor identity differs')
        os.kill(r['pid'], signal.SIGSTOP)
        for command in ('selftest', 'diagnose'):
            status('waiting_for_slot', stage=command)
            while busy(): time.sleep(3)
            check(); status('running', stage=command)
            with (logs / (command + '.log')).open('x') as f:
                rc = subprocess.call([str(HERE / 'run_light.sh'), 'wait_post_noise_' + command, '--', sys.executable,
                                      str(HERE / 'pf_wait_post_state_noise.py'), command, '--out', str(out / 'diagnosis')],
                                     cwd=ROOT, env=dict(os.environ, PF_SLOT='codex', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1'), stdout=f, stderr=subprocess.STDOUT)
            if rc: raise RuntimeError(f'{command} failed: {rc}; attempt preserved')
            check()
        status('complete', result=str(out / 'diagnosis/result.json'))
    except BaseException as exc:
        status('failed', error=str(exc)); raise
    finally:
        resume_size(record)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['run', 'resume']); p.add_argument('--out', type=Path)
    p.add_argument('--resume-record', type=Path, required=True); a = p.parse_args()
    if a.command == 'resume': resume_size(a.resume_record)
    else: run(a.out, a.resume_record)
