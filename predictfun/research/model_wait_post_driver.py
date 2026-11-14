"""Durable, single-slot wait/post launcher; preserves every attempted stage.

Waits for core 8, uses run_light for each worker, and always resumes an explicitly
recorded older size queue after its own work finishes or fails. --mode smoke is
the bounded preflight; --mode full runs the declared consumed-date pipeline.
"""
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
ROOT = Path('/home/yuqing/ctaNew')
DAYS = ('20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102', '20261103')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(path)


def busy():
    count = 0; core_busy = False
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args = (p/'cmdline').read_bytes().split(b'\0')
            if not args or not Path(os.fsdecode(args[0])).name.startswith('python'): continue
            cmd = b' '.join(args)
            if b'predictfun.research.' not in cmd and b'predictfun/research/pf_' not in cmd: continue
            count += 1
            affinity = next(x.split(':', 1)[1].strip() for x in (p/'status').read_text().splitlines() if x.startswith('Cpus_allowed_list:'))
            core_busy |= affinity == '8'
        except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration): continue
    return core_busy or count >= 2


def resume_size(path):
    if path is None: return
    record = json.loads(path.read_text()); pid = record['pid']; proc = Path('/proc')/str(pid)
    if not proc.exists(): return
    cmd = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode()
    if cmd != record['cmdline'] or (proc/'stat').read_text().split()[21] != record['starttime']:
        raise RuntimeError('refusing to signal a different process')
    os.kill(pid, signal.SIGCONT)


def run(args):
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    logs = out/('driver_'+args.mode); logs.mkdir()
    freeze = json.loads((HERE.parent/'reviews/wait_post_20261114/freeze.json').read_text())
    def check():
        for rel, sha in freeze['files'].items():
            if digest(HERE.parents[1]/rel) != sha: raise RuntimeError('frozen source changed: '+rel)
    check()
    steps = [('selftest', [str(HERE/'pf_entry_value.py'), 'selftest']),
             ('wait_post_selftest', [str(HERE/'pf_wait_post.py'), 'selftest'])]
    if args.mode == 'smoke':
        steps += [('rows_smoke', [str(HERE/'pf_wait_post.py'), 'rows', DAYS[0], '--max-windows', '2', '--out', str(out/'smoke')])]
    else:
        steps += [('rows_'+d, [str(HERE/'pf_wait_post.py'), 'rows', d, '--out', str(out/'full')]) for d in DAYS]
        for d in DAYS[4:]:
            steps += [('fit_'+d, [str(HERE/'pf_wait_post.py'), 'fit', d, '--out', str(out/'full')]),
                      ('eval_'+d, [str(HERE/'pf_wait_post.py'), 'eval', d, '--out', str(out/'full')])]
        steps += [('report', [str(HERE/'pf_wait_post.py'), 'report', '--out', str(out/'full')])]
    env = dict(os.environ, PF_SLOT='codex', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONSAFEPATH='1')
    try:
        for name, command in steps:
            write(logs/'status.json', dict(status='waiting_for_slot', stage=name, pid=os.getpid(), updated=time.time()))
            while busy(): time.sleep(10)
            check()
            write(logs/'status.json', dict(status='running', stage=name, pid=os.getpid(), updated=time.time()))
            with (logs/(name+'.log')).open('x') as stream:
                rc = subprocess.call([str(HERE/'run_light.sh'), 'wait_post_'+name, '--', sys.executable, *command],
                                     cwd=HERE.parents[1], env=env, stdout=stream, stderr=subprocess.STDOUT)
            if rc: raise RuntimeError(f'{name} failed with exit {rc}; outputs preserved, no retry')
        check(); write(logs/'status.json', dict(status='complete', stages=[n for n, _ in steps], pid=os.getpid(), updated=time.time()))
    except BaseException as exc:
        write(logs/'status.json', dict(status='failed', error=str(exc), pid=os.getpid(), updated=time.time()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    p.add_argument('--out', type=Path, required=True); p.add_argument('--resume-size-record', type=Path)
    a = p.parse_args()
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        run(a)
    finally:
        resume_size(a.resume_size_record)
