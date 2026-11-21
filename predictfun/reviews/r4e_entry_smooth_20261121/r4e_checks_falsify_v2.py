"""R4E FALSIFIER v2: SUPERSEDES r4e_checks.falsify (frozen in 602fdfc5 / 5c2ec2ca) (P-2026-005; 2026-11-21, session aea5bfb9). The checks
themselves (r4e_checks.main: A-I) ran unchanged and wrote checks.json. v1's falsifier then crashed (logs/r4e_checks.log): its fill
corruption looked only in the first ok world of partition 0, which had no fill in any episode (StopIteration), and nothing was written.
v2 = v1's cases on copies of the real inputs, with each corruption searching every partition for its target. A case records whether its
corruption was applied; an unapplied corruption fails the falsifier, never passes it.
 - C: a label changed in the evaluation predictions;
 - D: a stored predicate flipped;
 - F: a fill quantity changed in an episode trace, and a control episode's value changed;
 - G: a fill changed in a reached pair's arm ledger;
 - H: a mutant record copied over another index.
Each check is true on the clean copy and false after each corruption. Writes checks_falsifier.json.
Usage (run_light): r4e_checks_falsify_v2.py"""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r4e_checks as V1                                  # noqa: E402  frozen: the checks C, D, F, G, H and the readers

N, NR, LS, C, pd = V1.N, V1.NR, V1.LS, V1.C, V1.pd


def safe(f, *a):
    try:
        return f(*a)
    except Exception:                                                                     # noqa: BLE001  (a corrupted copy that cannot be read is false)
        return False


def corrupt_parts(td, name, fn):
    """A copy of every replay part; fn(rows) is applied to the first partition where it finds its target (the part rewritten with a matching
    record hash) -> (directory, applied)."""
    g = td / name / 'parts'; g.mkdir(parents=True)
    for q in (NR.OUT / 'parts').glob('part_*'):
        shutil.copy(q, g / q.name)
    if fn is None:
        return td / name, True
    for p in sorted(g.glob('part_*[0-9].json')):
        k = p.stem; rec = json.loads(p.read_text()); rows = LS.gz_rows((g / f'{k}.jsonl.gz').read_bytes())
        if fn(rows):
            b = LS.gz_bytes(rows); (g / f'{k}.jsonl.gz').write_bytes(b); rec['sha256'] = hashlib.sha256(b).hexdigest(); p.write_text(json.dumps(rec))
            return td / name, True
    return td / name, False


def fill_bump(rows):
    for r in rows:
        if r['status'] == 'ok':
            for e in r['episodes'].values():
                tr = e.get('trace') or {}
                if tr.get('fill_list'):
                    tr['fill_list'][0][3] = float(tr['fill_list'][0][3]) + 1.; return True
    return False


def ctrl_value(rows):
    for r in rows:
        if r['status'] == 'ok':
            e = r['episodes']['PI0']; e['trace']['value'] += 1e-6; e['value'] += 1e-6; return True
    return False


def arm_fill(rows):
    for r in rows:
        if r['status'] == 'ok':
            for x in r.get('reached', []):
                if x.get('status') == 'ok':
                    for a_ in x['result']['arms'].values():
                        if a_.get('fill_list'):
                            a_['fill_list'][0][3] = float(a_['fill_list'][0][3]) + 1.; return True
    return False


def main(out=N.OUT):
    out = Path(out); res = {}
    with tempfile.TemporaryDirectory(dir=str(out)) as td:
        td = Path(td); o = td / 'CD'; (o / 'eval').mkdir(parents=True); shutil.copy(out / 'fit.json', o / 'fit.json')
        for q in (out / 'eval').iterdir():
            shutil.copy(q, o / 'eval' / q.name)
        clean = safe(V1.check_C, o, {}); D = pd.read_csv(o / 'eval' / 'predictions.csv.gz'); D.loc[D.index[0], 'Y'] = D.loc[D.index[0], 'Y'] + 1.
        D.to_csv(o / 'eval' / 'predictions.csv.gz', index=False); res['C'] = dict(clean=clean, corrupted=safe(V1.check_C, o, {}), applied=True)
        for q in (out / 'eval').iterdir():
            shutil.copy(q, o / 'eval' / q.name)
        clean = safe(V1.check_D, o, {}); P = pd.read_csv(o / 'eval' / 'predicates.csv'); P.loc[P.index[0], 'screen_pass'] = not bool(P.loc[P.index[0], 'screen_pass'])
        P.to_csv(o / 'eval' / 'predicates.csv', index=False); res['D'] = dict(clean=clean, corrupted=safe(V1.check_D, o, {}), applied=True)
        cdir, _ = corrupt_parts(td, 'clean', None); fdir, fa = corrupt_parts(td, 'Ffill', fill_bump); vdir, va = corrupt_parts(td, 'Fval', ctrl_value)
        gdir, ga = corrupt_parts(td, 'Garm', arm_fill)
        res['F'] = dict(clean=safe(V1.check_F, out, {}, cdir), trace_fill=safe(V1.check_F, out, {}, fdir), control_value=safe(V1.check_F, out, {}, vdir), applied=bool(fa and va))
        res['G'] = dict(clean=safe(V1.check_G, out, {}, cdir), arm_fill=safe(V1.check_G, out, {}, gdir), applied=bool(ga))
        h = td / 'H'
        for tag in ('entry', 'entry_eval', 'entry_replay'):
            shutil.copytree(V1.LOGS / f'mutants_{tag}', h / f'mutants_{tag}')
        saved = V1.LOGS
        try:
            V1.LOGS = h; clean = safe(V1.check_H, {}); shutil.copy(h / 'mutants_entry_eval' / 'mut_entry_eval_0.log', h / 'mutants_entry_eval' / 'mut_entry_eval_1.log')
            res['H'] = dict(clean=clean, corrupted=safe(V1.check_H, {}), applied=True)
        finally:
            V1.LOGS = saved
    ok = all(v['clean'] is True and v['applied'] is True and all(x is False for k, x in v.items() if k not in ('clean', 'applied')) for v in res.values())
    (HERE / 'checks_falsifier.json').write_text(json.dumps(dict(cases=res, falsifier_ok=ok, supersedes='r4e_checks.falsify (crashed: StopIteration)',
                                                              source_sha256=C.sha(Path(__file__)), asof=C.stamp()), indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(cases=res, falsifier_ok=ok)))


if __name__ == '__main__':
    main()
