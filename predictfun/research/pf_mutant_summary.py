"""SUMMARY of a per-mutant run (pf_mutant_one.py jobs; review 4393ad19 B6; 2026-11-20, session aea5bfb9). It reads <log dir>/<prefix>_control.log
and <prefix>_<i>.log for every mutant index i of the harness, and takes each log's last JSON record.
It passes iff:
 - the control PASSED;
 - every index has exactly one record, and it is CAUGHT;
 - every record ran on one set of sources, equal to the files on disk now.
A missing record (a killed or timed-out job), an escape, a crash, a failed control or mixed sources each fail. Prints one line per mutant
(class, seconds, the catching message), the totals and the CPU seconds.
--selftest: these rules on synthetic records. All caught passes; each of the five failures above fails.
Usage: pf_mutant_summary.py <harness file> <log dir> <prefix> | --selftest"""
import importlib.util
import json
import sys
from pathlib import Path


def last_record(p):
    if not p.exists():
        return None
    for ln in reversed(p.read_text().splitlines()):
        if ln.startswith('{') and '"cls"' in ln:
            try:
                return json.loads(ln)
            except ValueError:
                return None
    return None


def judge(n_mut, recs, now, names=None):
    """recs: {'control' | i: record or None}; names: the harness's mutant names -> (ok, failures). A record must carry its own index and name
    (review 38febc0b item 4: a copied record at another file must not count)."""
    bad = []; c = recs.get('control')
    if not c or c.get('cls') != 'PASSED' or c.get('index') != 'control':
        bad.append(f"control: {c and c.get('cls')} / {c and c.get('index')}")
    for i in range(n_mut):
        r = recs.get(i)
        if r is None:
            bad.append(f'{i}: no record (killed / timed out)')
        elif str(r.get('index')) != str(i) or (names is not None and r.get('name') != names[i]):
            bad.append(f"{i}: the record is mutant {r.get('index')} ({r.get('name')}), not {i}")
        elif r.get('cls') != 'CAUGHT':
            bad.append(f"{i} {r.get('name')}: {r.get('cls')} - {r.get('why', '')[:120]}")
    srcs = {json.dumps(r.get('sources'), sort_keys=True) for r in recs.values() if r}
    if len(srcs) != 1 or (now is not None and json.loads(next(iter(srcs))) != now):
        bad.append(f'sources: {len(srcs)} distinct source sets, or not the files on disk now')
    return not bad, bad


def selftest():
    S = dict(a='1', b='2'); ok = lambda **k: dict(dict(cls='CAUGHT', sources=S, name='m'), **k); nm = ['m0', 'm1']
    good = {'control': ok(cls='PASSED', index='control', name='control'), 0: ok(index='0', name='m0'), 1: ok(index='1', name='m1')}
    cases = {'all caught': (good, True), 'a missing record': ({k: v for k, v in good.items() if k != 1}, False), 'an escape': ({**good, 1: ok(cls='ESCAPED')}, False),
             'a crash': ({**good, 1: ok(cls='HARNESS_FAILURE')}, False), 'a failed control': ({**good, 'control': ok(cls='CONTROL_FAILED')}, False),
             'mixed sources': ({**good, 1: ok(index='1', name='m1', sources=dict(a='1', b='3'))}, False),
             'a copied record (another index)': ({**good, 1: dict(good[0])}, False), 'a renamed record': ({**good, 1: ok(index='1', name='other')}, False)}
    cases['an escape'] = ({**good, 1: ok(index='1', name='m1', cls='ESCAPED')}, False); cases['a crash'] = ({**good, 1: ok(index='1', name='m1', cls='HARNESS_FAILURE')}, False)
    cases['a failed control'] = ({**good, 'control': ok(cls='CONTROL_FAILED', index='control')}, False)
    wrong = [n for n, (recs, want) in cases.items() if judge(2, recs, S, nm)[0] != want]
    if wrong:
        sys.exit(f'MUTANT SUMMARY SELFTEST FAILED: {wrong}')
    print(f'mutant summary selftest passed: {len(cases)} cases (all caught passes; a missing record, an escape, a crash, a failed control, mixed sources, a copied and a renamed record fail)')


def main(harness, logdir, prefix):
    hp = Path(harness).resolve(); sys.path.insert(0, str(hp.parent))
    spec = importlib.util.spec_from_file_location(hp.stem, hp); H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
    import pf_mutant_one as O
    d = Path(logdir); recs = {'control': last_record(d / f'{prefix}_control.log')}
    recs.update({i: last_record(d / f'{prefix}_{i}.log') for i in range(len(H.MUTANTS))})
    ok, bad = judge(len(H.MUTANTS), recs, O.sources(H), [m[0] for m in H.MUTANTS])
    for i in range(len(H.MUTANTS)):
        r = recs[i] or {}
        print(f"  {i} {H.MUTANTS[i][0]}: {r.get('cls', 'NO RECORD')} ({r.get('seconds', '?')} s) - {str(r.get('why', ''))[:150]}")
    c = recs['control'] or {}
    cpu = sum(r.get('seconds', 0) for r in recs.values() if r)
    print(f"control: {c.get('cls')} ({c.get('seconds', '?')} s); sources {c.get('sources')}")
    n_c = sum(1 for i in range(len(H.MUTANTS)) if (recs[i] or {}).get('cls') == 'CAUGHT')
    print(f"mutation check {'passed' if ok else 'FAILED'}: the control passes its selftest, {n_c} / {len(H.MUTANTS)} planted code errors are caught by a check "
          f"({cpu / 60:.1f} CPU min){'' if ok else ' | ' + ' | '.join(bad[:8])}")
    return 0 if ok else 1


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        selftest()
    else:
        sys.exit(main(*sys.argv[1:4]))
