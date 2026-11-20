"""ONE MUTANT, ONE PROCESS (review 4393ad19 B6; 2026-11-20, session aea5bfb9). The per-mutant runner behind pf_mutant_queue.sh: every planted
error costs one process and one slot core. A harness shard held two processes (itself and its child) for one core under the 5-process cap.
It imports a mutation harness (MUTANTS, SRC, FAIL, SELFTEST; its own main is guarded) and builds the copy with the one replacement, or the
unchanged copy for 'control'. The copy is executed as the module under its real path, in this fresh process, and its selftest is run.
The last stdout line is one JSON record, with the class:
 - CAUGHT: the selftest raised its own failure (SystemExit carrying the harness's FAIL message);
 - ESCAPED: the selftest returned for a mutant (it passed);
 - PASSED / CONTROL_FAILED: the unchanged copy's selftest returned / failed;
 - HARNESS_FAILURE: any other exception, or a target not found exactly once.
A killed or timed-out process writes no record, and pf_mutant_summary.py counts the missing record as a failure, never a catch.
Usage: pf_mutant_one.py <harness file> <index | control>"""
import contextlib
import hashlib
import importlib
import importlib.util
import io
import json
import sys
import time
import types
from pathlib import Path


def sources(H):
    """The module, its selftest and the harness (sha256 prefixes) this record ran on."""
    d = Path(H.SRC).parent
    return {n: hashlib.sha256((d / n).read_bytes()).hexdigest()[:16] for n in (Path(H.SRC).name, H.SELFTEST + '.py', Path(H.__file__).name)}


def main(harness, which):
    t0 = time.time(); hp = Path(harness).resolve(); sys.path.insert(0, str(hp.parent))
    spec = importlib.util.spec_from_file_location(hp.stem, hp); H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
    src = Path(H.SRC); base = src.read_text(); rec = dict(index=which, sources=sources(H))
    if which == 'control':
        rec['name'], text = 'control (the unchanged copy)', base
    else:
        name, old, new = H.MUTANTS[int(which)]; rec['name'] = name
        if base.count(old) != 1:
            return dict(rec, cls='HARNESS_FAILURE', seconds=round(time.time() - t0, 1), why='the target does not occur exactly once')
        text = base.replace(old, new)
    mod = types.ModuleType(src.stem); mod.__file__ = str(src); sys.modules[src.stem] = mod; out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            exec(compile(text, str(src), 'exec'), mod.__dict__)
            importlib.import_module(H.SELFTEST).run(mod)
        lines = out.getvalue().strip().splitlines(); cls, why = ('PASSED' if which == 'control' else 'ESCAPED'), (lines[-1][:200] if lines else '')
    except SystemExit as e:
        msg = str(e); cls = ('CONTROL_FAILED' if which == 'control' else 'CAUGHT') if H.FAIL in msg else 'HARNESS_FAILURE'; why = msg[:200]
    except BaseException as e:                                                              # a crash is never a catch
        cls, why = 'HARNESS_FAILURE', f'crash: {e!r}'[:200]
    return dict(rec, cls=cls, seconds=round(time.time() - t0, 1), why=why)


if __name__ == '__main__':
    r_ = main(sys.argv[1], sys.argv[2]); sys.stdout.flush(); print(json.dumps(r_, sort_keys=True), flush=True)
    sys.exit(0 if r_['cls'] in ('CAUGHT', 'PASSED') else 1)
