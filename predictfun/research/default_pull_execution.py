"""Two-slot execution metadata adapter; frozen numerical sources stay unchanged."""
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
ROOT = Path('/home/yuqing/ctaNew')
OUT = ROOT / 'data/predict_fun/research/default_pull_20261111_v4'
CORE = 8
CAP = 2


def numerical(argv):
    if not argv or not Path(argv[0]).name.startswith('python'):
        return False
    args = argv[1:]
    while args and args[0] in ('-u', '-I', '-B', '-E', '-s', '-S'):
        args = args[1:]
    if not args:
        return False
    if args[0] == '-m':
        return len(args) > 1 and args[1].startswith('predictfun.research.')
    script = args[0]
    if 'predictfun/research/pf_' not in script or not script.endswith('.py'):
        return False
    if script.endswith('pf_default_pull_progress.py'):
        return False
    if script.endswith('pf_default_pull_known_empty.py') and len(args) > 1 and args[1] in ('progress', 'slot'):
        return False
    return True


def cores(value):
    result = set()
    for part in value.split(','):
        ends = part.split('-')
        lo = int(ends[0]); hi = int(ends[-1])
        if len(ends) > 2 or lo > hi:
            raise ValueError('invalid CPU affinity')
        result.update(range(lo, hi + 1))
    return result


def reasons(jobs):
    result = []
    if len(jobs) >= CAP:
        result.append('two-process research cap reached')
    if any(CORE in cores(j['cpus']) for j in jobs):
        result.append('Codex core 8 is occupied')
    return result


def observe():
    jobs = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            argv = p.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
            if not numerical(argv):
                continue
            status = p.joinpath('status').read_text().splitlines()
            cpus = next(s.split(':', 1)[1].strip() for s in status if s.startswith('Cpus_allowed_list:'))
            jobs.append(dict(pid=int(p.name), cpus=cpus, command=argv))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return dict(slot='codex', core=CORE, cap=CAP, jobs=jobs, wait_reasons=reasons(jobs))


def controls():
    if reasons([]) or reasons([dict(cpus='9')]):
        raise ValueError('available slot refused')
    for jobs in ([dict(cpus='8')], [dict(cpus='9'), dict(cpus='10')], [dict(cpus='0-15')]):
        if not reasons(jobs):
            raise ValueError('occupied slot or process cap accepted')
    if not numerical(['python3', '-I', '/x/predictfun/research/pf_b21_rest_default.py']):
        raise ValueError('script job missed')
    if not numerical(['python3', '-m', 'predictfun.research.pf_default_pull_known_empty', 'run']):
        raise ValueError('module job missed')
    for argv in (['python3', 'predictfun/research/default_pull_watch.py'],
                 ['python3', 'predictfun/research/pf_default_pull_known_empty.py', 'slot'],
                 ['python3', '/tmp/b21_launch.py', '/x/predictfun/research/pf_b21_rest_default.py']):
        if numerical(argv):
            raise ValueError('metadata launcher misclassified as numeric worker')
    try:
        cores('9-8')
    except ValueError:
        pass
    else:
        raise ValueError('invalid affinity accepted')


def progress(*args):
    spec = importlib.util.spec_from_file_location('frozen_progress', HERE / 'pf_default_pull_progress.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.OUT = OUT
    module.update(*args)
    # Only presentation/operations metadata changes; stage history stays intact.
    for base in (WORK, ROOT):
        for rel in ('predictfun/handoffs/20261111_default_pull_progress.md', 'predictfun/HANDOFF.md'):
            p = base / rel
            text = p.read_text().replace(
                'One numeric process; consumed dates only; no live changes.',
                'One Codex worker on core 8; two research workers allowed globally (user 2026-11-12); consumed dates only; no live changes.'
            ).replace(
                'one light numeric process; no live changes/protected-day reads.',
                'Codex slot/core 8, global cap two research workers; no live changes/protected-day reads.'
            )
            module.atomic(p, text)
        p = base / 'predictfun/STATUS.yml'; text = p.read_text()
        text = text.replace('# /DEFAULT_PULL_LOOP', '  process_cap: 2\n  execution_slot: codex\n  cpu_core: 8\n# /DEFAULT_PULL_LOOP', 1)
        module.atomic(p, text)


if __name__ == '__main__':
    command = sys.argv[1]
    if command == 'controls':
        controls(); print('TWO_SLOT_CONTROLS_PASS')
    elif command == 'slot':
        state = observe(); print(json.dumps(state)); sys.exit(bool(state['wait_reasons']))
    elif command == 'progress':
        progress(*sys.argv[2:])
    else:
        raise ValueError('unknown execution metadata command')
