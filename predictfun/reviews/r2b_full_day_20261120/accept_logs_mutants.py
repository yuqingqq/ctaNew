"""Mutation check of accept_logs.py: each planted error in the log predicate must be caught BY A CHECK, meaning at least one falsifier case
returns the wrong verdict. A mutant that crashes is not caught. Precondition: the unchanged copy gives every case its expected verdict.
Usage: accept_logs_mutants.py"""
import importlib.util
import sys
import tempfile
from pathlib import Path

SRC = (Path(__file__).resolve().parent / 'accept_logs.py').read_text()
MUTANTS = [('no exclusion of the acceptance logs', 'for p in every if not p.name.startswith(OWN)}', 'for p in every}'),
           ('no declared-presence check', 'missing = sorted(set(declared) - set(seen))', 'missing = []'),
           ('first footer only', 'for a, b in v)))', 'for a, b in v[:1])))'),
           ('a footer not required', 'not (len(v) >= 1 and all(', 'not (all('),
           ('rc not checked', 'int(a) == 0 and int(b) <= CAP', 'int(b) <= CAP'),
           ('strict cap', 'int(b) <= CAP for', 'int(b) < CAP for'),
           ('cap 1537', 'CAP = 1536', 'CAP = 1537'),
           ('exclusion of every log', 'if not p.name.startswith(OWN)}', "if not p.name.endswith('.log')}"),
           ('missing logs do not fail', 'ok=not missing and not bad', 'ok=not bad')]


def load(text, name):
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / 'accept_logs.py'; p.write_text(text)
        spec = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        return m.cases()


base = load(SRC, 'm_base'); pre = all(w == g for _, w, g in base)
print(f'unchanged copy: {sum(w == g for _, w, g in base)} / {len(base)} cases as expected: {pre}')
caught = []
for i, (name, old, new) in enumerate(MUTANTS):
    assert SRC.count(old) == 1, f'mutation target not unique: {name}'
    try:
        r = load(SRC.replace(old, new), f'm{i}'); hit = [c[0] for c in r if c[1] != c[2]]; how = f'caught by {hit}' if hit else 'NOT CAUGHT'
    except Exception as e:                                   # a crash is not a catch
        hit, how = [], f'CRASHED ({type(e).__name__}: {e}), not caught'
    caught.append(bool(hit)); print(f'  {name}: {how}')
ok = pre and all(caught)
print(f'accept_logs mutation check {"passed" if ok else "FAILED"}: the unchanged copy passes, {sum(caught)} / {len(MUTANTS)} planted errors are caught by a check')
sys.exit(0 if ok else 1)
