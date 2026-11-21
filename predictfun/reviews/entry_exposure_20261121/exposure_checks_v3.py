"""ENTRY-EXPOSURE-v1: THE DECLARED CHECKS, v3 (2026-11-21, session aea5bfb9). Supersedes checks_v2.json in band. Check H's comparison of the
frozen counter with its v2 copy (`overrides_counter_v1`) cast both columns with astype(str) and compared them elementwise with ==; under
pandas 3.0.1 astype(str) keeps NaN as missing, so the 12 world rows, NaN in both tables, compared unequal (checks_v2.json: H false, "the v2
forks.csv differs from the frozen one outside the override readings"). logs/exposure_h_diag.log: no column differs, the 12 elementwise
mismatches are exactly the 12 world rows with NaN in both, and the NaN-aware comparison is equal. v3 compares NaN-aware (counter_equal);
everything else is exposure_checks_v2 unchanged (its split check C rule, H's other conditions, G2).
SELFTEST (selftest_v3()): counter_equal is true for equal columns with NaN in the same rows, false for one changed counter, a NaN against a
number, a dropped row or a shifted row. FALSIFIER (falsify_v3()): the v2 falsifier's cases on the v2 outputs, with H clean under v3, plus
H_counter (one frozen-counter copy changed in the v2 forks.csv). Writes checks_v3.json, checks_falsifier_v3.json.
Usage (run_light): exposure_checks_v3.py [selftest]  (no argument: the checks, then the falsifier)"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import exposure_checks_v2 as V2                          # noqa: E402  (imports the frozen checks and applies the v2 row rule)

X, M, V, C, pd, LS = V2.X, V2.M, V2.V, V2.C, V2.pd, V2.LS
OUT2 = V2.OUT2


def counter_equal(a, b):
    """Two counter columns equal row for row, NaN equal to NaN only (header)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(a.shape == b.shape and np.array_equal(a, b, equal_nan=True))


def check_H(out, R):
    """v2's check H with the NaN-aware counter comparison (header)."""
    out = Path(out); F = X.forks(out); S = pd.read_csv(out / 'summary.csv', float_precision='round_trip'); S['day'] = S.day.astype(str)
    js = json.loads((out / 'exposure.json').read_text()); bad = []; logged = {}; changed = 0
    for k, rec, b in X.parts(out):
        for r in LS.gz_rows(b):
            if r.get('arms') is not None:
                key = (str(r['day']), int(r['m']), int(r['t_ms']), int(r['side'])); logged[key] = len(r['shadow']); changed += int(len(r['shadow']) != int(r['overrides']))
    got = {(r.day, int(r.m), int(r.t_ms), int(r.side)): int(r.overrides) for r in F.itertuples() if (r.day, int(r.m), int(r.t_ms), int(r.side)) in logged}
    if got != logged or not logged:
        bad.append('forks.csv overrides are not the logged overrides')
    ok = F[F.status == 'ok']
    for (day, s), g in ok.groupby(['day', 'side']):
        row = S[(S.day == day) & (S.side == s)]
        if len(row) != 1 or int(row.iloc[0].overrides_forks) != int((g.overrides > 0).sum()):
            bad.append(f'{day}/{s}: overrides_forks')
    v1 = M.OUT / 'exposure.json'; F1 = X.forks(M.OUT); cols = [c for c in F1.columns if c not in ('overrides', 'keep_baseline_equal')]
    same_rest = bool(len(F1) == len(F) and list(F1.columns) == [c for c in F.columns if c != 'overrides_counter_v1']
                     and F1[cols].astype(str).equals(F[cols].astype(str)) and counter_equal(F1.overrides, F.overrides_counter_v1))
    same_pred = json.loads(v1.read_text())['predicates'] == js.get('predicates')
    if not same_rest:
        bad.append('the v2 forks.csv differs from the frozen one outside the override readings')
    if not same_pred:
        bad.append('the v2 predicates differ from the frozen ones')
    if js.get('version') != V.VERSION or js.get('supersedes') != V.SUPERSEDES or js.get('superseded_sha256') != C.sha(v1) or js.get('overrides_changed_forks') != changed:
        bad.append('exposure.json: supersession fields')
    R['H'] = dict(bad=bad, forks_with_arms=len(logged), changed=changed, same_rest=same_rest, same_predicates=same_pred, overrides_forks_v2=int((ok.overrides > 0).sum()),
                  overrides_forks_v1=int((ok.overrides_counter_v1 > 0).sum()))
    return not bad


def main_v3(out=OUT2):
    out = Path(out); R = {}; ck = {}
    for k, f in (('A_identity', X.check_A), ('B_reproduction', X.check_B), ('C_raw', X.check_C), ('D_summaries', X.check_D), ('E_predicates', X.check_E),
                 ('F_resources', X.check_F), ('H_overrides', check_H)):
        try:
            ck[k] = f(out, R)
        except Exception as e:                                                              # noqa: BLE001  (a check that cannot run is false)
            ck[k] = False; R[k[0]] = dict(error=repr(e)[:300])
    for k, f in (('G_suites', X.check_G), ('G2_suite_v2', V2.check_G2)):
        try:
            ck[k] = f(R)
        except Exception as e:                                                              # noqa: BLE001
            ck[k] = False; R[k] = dict(error=repr(e)[:300])
    J = dict(checks=ck, all_true=all(ck.values()), readings=R, supersedes='checks_v2.json (check H compared NaN counters elementwise)',
             source_sha256=C.sha(Path(__file__)), v2_checks_sha256=C.sha(HERE / 'exposure_checks_v2.py'), frozen_checks_sha256=C.sha(HERE / 'exposure_checks.py'), asof=C.stamp())
    (HERE / 'checks_v3.json').write_text(json.dumps(J, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(checks=ck, all_true=J['all_true']), indent=1))


def falsify_v3(out=OUT2):
    out = Path(out); res = {}

    def safe(f, *a):
        try:
            return f(*a)
        except Exception:                                                                 # noqa: BLE001
            return False
    with tempfile.TemporaryDirectory(dir=str(M.OUT)) as td:
        td = Path(td)

        def copy_(name):
            d = td / name; (d / 'parts').mkdir(parents=True)
            for q in out.iterdir():
                if q.is_file():
                    shutil.copy(q, d / q.name)
            for q in (out / 'parts').iterdir():
                if q.is_file():
                    (d / 'parts' / q.name).symlink_to(q.resolve())
            return d

        def csv_fn(d, name, fn):
            D = pd.read_csv(d / name, float_precision='round_trip'); applied = fn(D); D.to_csv(d / name, index=False); return applied

        def drop(D):
            D.drop(index=D.index[D.status == 'ok'][0], inplace=True); return True
        def flag(D):
            D.loc[D.index[D.status == 'ok'][0], 'prefix_ok'] = False; return True
        def comp(D):
            i = D.index[D.status == 'ok'][0]; D.loc[i, 'D_initial'] += 1.; D.loc[i, 'D_rest'] -= 1.; return True
        def summ(D):
            D.loc[D.index[0], 'mean_D'] += 1.; return True
        def ovr(D):
            D.loc[D.index[D.status == 'ok'][0], 'overrides'] += 1; return True
        def ctr(D):
            D.loc[D.index[D.status == 'ok'][0], 'overrides_counter_v1'] += 1; return True

        def wrong_gid(r):
            g = int(r['init_KEEP']['gid0']); other = [int(o['gid']) for o in r['arms']['POST_KEEP_INITIAL']['orders'] if int(o['gid']) != g]
            if not other:
                return False
            r['init_KEEP']['gid0'] = other[0]; return True

        def fill_price(r):
            fl = r['arms']['POST_KEEP_INITIAL']['fill_list']
            if not fl:
                return False
            fl[0][4] = float(fl[0][4]) + 1.; return True

        def deficit(r):
            if not r['shadow']:
                return False
            r['overrides'] = len(r['shadow']) - 1; return True
        clean = copy_('clean')
        for name, chk, fn, tgt in (('A', X.check_A, drop, 'forks.csv'), ('B', X.check_B, flag, 'forks.csv'), ('C_components', X.check_C, comp, 'forks.csv'),
                                   ('D', X.check_D, summ, 'summary.csv'), ('H', check_H, ovr, 'forks.csv'), ('H_counter', check_H, ctr, 'forks.csv')):
            d = copy_(name); ap = csv_fn(d, tgt, fn); res[name] = dict(clean=safe(chk, clean, {}), corrupted=safe(chk, d, {}), applied=bool(ap))
        for name, fn in (('C_wrong_gid', wrong_gid), ('C_fill_price', fill_price), ('C_counter_deficit', deficit)):
            d = copy_(name); ap = X._corrupt_part(d, out, fn); res[name] = dict(clean=safe(X.check_C, clean, {}), corrupted=safe(X.check_C, d, {}), applied=bool(ap),
                                                                                 a_still_true=safe(X.check_A, d, {}))
        d = copy_('E'); j = json.loads((d / 'exposure.json').read_text()); j['predicates']['diagnosis'] = 'flipped'; (d / 'exposure.json').write_text(json.dumps(j))
        res['E'] = dict(clean=safe(X.check_E, clean, {}), corrupted=safe(X.check_E, d, {}), applied=True)
        saved = X.LOGS
        try:
            h = td / 'H_logs'; shutil.copytree(saved, h); X.LOGS = h; clF = safe(X.check_F, out, {})
            (h / 'over_budget.log').write_text('2026-11-21T00:00:00Z tag=over_budget rc=0 peak_rss_mib=100 wall_s=7300 cpu_user_s=7300.0\n')
            res['F'] = dict(clean=clF, corrupted=safe(X.check_F, out, {}), applied=True)
            for nm, sub, pre, fn in (('G', 'mutants_exp', 'mut_exp', X.check_G), ('G2', 'mutants_sum', 'mut_sum', V2.check_G2)):
                h2 = td / f'{nm}_logs'; shutil.copytree(saved, h2); X.LOGS = h2; cl = safe(fn, {})
                shutil.copy(h2 / sub / f'{pre}_0.log', h2 / sub / f'{pre}_1.log'); res[nm] = dict(clean=cl, corrupted=safe(fn, {}), applied=True)
        finally:
            X.LOGS = saved
    ok = all(v['clean'] is True and v['corrupted'] is False and v['applied'] and v.get('a_still_true', True) for v in res.values())
    (HERE / 'checks_falsifier_v3.json').write_text(json.dumps(dict(cases=res, falsifier_ok=ok, asof=C.stamp()), indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(cases=res, falsifier_ok=ok)))


def selftest_v3():
    """counter_equal on hand columns (header)."""
    n = np.nan; a = pd.Series([3., n, 5., n]); bad = []
    cases = ((a, pd.Series([3., n, 5., n]), True, 'NaN in the same rows'), (a, pd.Series([3., n, 6., n]), False, 'one changed counter'),
             (a, pd.Series([3., 4., 5., n]), False, 'a NaN against a number'), (a, pd.Series([3., n, 5.]), False, 'a dropped row'),
             (a, pd.Series([n, 3., n, 5.]), False, 'a shifted row'))
    for x, y, want, lab in cases:
        if counter_equal(x, y) is not want:
            bad.append(f'counter_equal: {lab}')
    s = pd.Series([1.0, n]).astype(str)
    if bool((s.values == s.values).all()):
        bad.append('the positive control: elementwise == on astype(str) no longer drops NaN pairs under this pandas (the v2 failure would not reproduce)')
    if bad:
        sys.exit('EXPOSURE CHECKS V3 SELFTEST FAILED: ' + '; '.join(bad))
    print(f'exposure checks v3 selftest passed: counter_equal on {len(cases)} hand cases; the v2 comparison\'s NaN failure reproduces under pandas {pd.__version__}')


if __name__ == '__main__':
    if sys.argv[1:] == ['selftest']:
        selftest_v3()
    else:
        main_v3(); falsify_v3()
