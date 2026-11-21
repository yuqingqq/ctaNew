"""ENTRY-EXPOSURE-v1: THE DECLARED CHECKS, v2 (2026-11-21, session aea5bfb9). Supersedes exposure_checks.py's results in band, because the
frozen check C failed on a reported counter (declaration_v2.md): the KEEP wrapper's `overrides` counter included the causal probe's decide
calls on its copies. The frozen checks A-G run unchanged on the v2 outputs (OUT/v2: the frozen file names, the manifest's byte copy, the
partitions linked), except check C's shadow rule, split:
 (i) every logged override is of the forced initial gid with a shadow PULL (as frozen);
 (ii) the v2 override count equals the log's (H);
 (iii) the frozen counter exceeds the log by at least 0 and at most twice the KEEP arm's probe-checked calls (the diagnosed inflation;
       a deficit or a larger excess is a failure).
Added: H overrides (forks.csv's overrides are the partitions' logged overrides; summary.csv's overrides_forks recomputed; exposure.json
names what it supersedes, with the frozen exposure.json's hash and the changed-fork count; and the supersession changed nothing else: the
frozen and v2 forks.csv are equal in every other column and row, and the v2 predicates equal the frozen ones exactly); G2 the v2 summary's
mutation suite.
FALSIFIER (falsify_v2()): the frozen falsifier's cases on the v2 outputs, plus H (one fork's overrides changed) and G2 (a copied record).
SELFTEST (selftest_v2()): the frozen checker's synthetic selftest with the v2 row rule, and the v2 rule refuses a counter below the log, an
excess beyond the probe bound and a shadow entry of another gid.
Usage (run_light): exposure_checks_v2.py [selftest]  (no argument: the checks, then the falsifier)"""
import copy
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import exposure_checks as X                              # noqa: E402  frozen checks
import pf_entry_exposure as M                            # noqa: E402  frozen
import pf_entry_exposure_summary as V                    # noqa: E402  the v2 summary

C, pd, LS = M.C, M.pd, M.LS
OUT2 = V.OUT2
SHADOW_V1 = 'KEEP: the shadow log is not its overrides'
_raw_v1 = X.raw_row_bad


def raw_row_bad_v2(r):
    """The frozen row reconciliation with the shadow rule split (header)."""
    bad = [m for m in _raw_v1(r) if m != SHADOW_V1]; s = int(r['side'])
    dl = [d for d in r['arms']['POST_KEEP_INITIAL']['decision_list'] if float(d[0]) == 0. and int(d[1]) == s and d[3] == 'POST']
    gk = int(dl[0][4]) if len(dl) == 1 and dl[0][4] is not None else None
    if any(gk is None or int(x['gid']) != gk or x['shadow_action'] != 'PULL' for x in r['shadow']):
        bad.append('KEEP: a logged override not of the initial gid with a shadow PULL')
    chk = int((r['arms']['POST_KEEP_INITIAL']['causal'] or {}).get('checked', 0)); ex = int(r['overrides']) - len(r['shadow'])
    if not (0 <= ex <= 2 * chk):
        bad.append('KEEP: the frozen counter differs from the log beyond the probe bound')
    return bad


X.raw_row_bad = raw_row_bad_v2                           # check_C and the falsifier resolve the row rule at call time


def check_H(out, R):
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
                     and F1[cols].astype(str).equals(F[cols].astype(str)) and (F1.overrides.astype(str).values == F.overrides_counter_v1.astype(str).values).all())
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


def check_G2(R):
    import importlib.util
    import pf_mutant_one as O
    import pf_mutant_summary as S
    h = 'pf_entry_exposure_summary_mutants.py'; spec = importlib.util.spec_from_file_location(h[:-3], X.RES / h); H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
    d = X.LOGS / 'mutants_sum'; files = {'control': d / 'mut_sum_control.log', **{i: d / f'mut_sum_{i}.log' for i in range(len(H.MUTANTS))}}
    recs = {k: S.last_record(p) for k, p in files.items()}
    one = all(p.exists() and sum(1 for ln in p.read_text().splitlines() if ln.startswith('{') and '"cls"' in ln) == 1 for p in files.values())
    extra = sorted({x.name for x in d.glob('mut_sum_*.log')} - {p.name for p in files.values()})
    ok, bad = S.judge(len(H.MUTANTS), recs, O.sources(H), [m_[0] for m_ in H.MUTANTS])
    R['G2'] = dict(planted=len(H.MUTANTS), judge_ok=ok, failures=bad[:5], one_record_per_file=one, extra_files=extra)
    return bool(ok and one and not extra)


def main_v2(out=OUT2):
    out = Path(out); R = {}; ck = {}
    for k, f in (('A_identity', X.check_A), ('B_reproduction', X.check_B), ('C_raw', X.check_C), ('D_summaries', X.check_D), ('E_predicates', X.check_E),
                 ('F_resources', X.check_F), ('H_overrides', check_H)):
        try:
            ck[k] = f(out, R)
        except Exception as e:                                                              # noqa: BLE001  (a check that cannot run is false)
            ck[k] = False; R[k[0]] = dict(error=repr(e)[:300])
    for k, f in (('G_suites', X.check_G), ('G2_suite_v2', check_G2)):
        try:
            ck[k] = f(R)
        except Exception as e:                                                              # noqa: BLE001
            ck[k] = False; R[k] = dict(error=repr(e)[:300])
    J = dict(checks=ck, all_true=all(ck.values()), readings=R, supersedes='checks.json (the frozen check C failed on the probe-inflated override counter)',
             source_sha256=C.sha(Path(__file__)), frozen_checks_sha256=C.sha(HERE / 'exposure_checks.py'), asof=C.stamp())
    (HERE / 'checks_v2.json').write_text(json.dumps(J, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(checks=ck, all_true=J['all_true']), indent=1))


def falsify_v2(out=OUT2):
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
                                   ('D', X.check_D, summ, 'summary.csv'), ('H', check_H, ovr, 'forks.csv')):
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
            for nm, sub, pre, fn in (('G', 'mutants_exp', 'mut_exp', X.check_G), ('G2', 'mutants_sum', 'mut_sum', check_G2)):
                h2 = td / f'{nm}_logs'; shutil.copytree(saved, h2); X.LOGS = h2; cl = safe(fn, {})
                shutil.copy(h2 / sub / f'{pre}_0.log', h2 / sub / f'{pre}_1.log'); res[nm] = dict(clean=cl, corrupted=safe(fn, {}), applied=True)
        finally:
            X.LOGS = saved
    ok = all(v['clean'] is True and v['corrupted'] is False and v['applied'] and v.get('a_still_true', True) for v in res.values())
    (HERE / 'checks_falsifier_v2.json').write_text(json.dumps(dict(cases=res, falsifier_ok=ok, asof=C.stamp()), indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(cases=res, falsifier_ok=ok)))


def selftest_v2():
    """The frozen checker's synthetic selftest under the v2 row rule, then the v2 rule's own refusals on a synthetic row (header)."""
    import os
    X.selftest(); bad = []
    import pf_r2_labels_selftest as LT
    L, R2 = M.L, M.R2; wj = json.loads((Path(os.environ['PF_TEACHER_SYN']) / 'world.json').read_text()); day = '20261024'
    W = L.worlds_of_day(day, Path(wj['kp']), 2); _, prs, fair, depth, ff = L.world_inputs(day, W, R2.KP.Syn(), R2.X.syn_inputs, R2.X.syn_features())
    Mst, sst = LT.stubs(-12.), LT.OpenSupport(); r0 = None
    for w in W:
        wb = L.world_bind(day, w, prs, fair, depth, ff)[0]
        if wb is None:
            continue
        sel = dict(day=day, side=0, decision_ms=0, source='neutral', shadow_source='entry', offset_ms=0, kind='post')
        ctx = L.fork_for(sel, wb, Mst, sst); st = L.pair_bound(ctx, L.teacher_factory(Mst, sst, wb, every=M.AUDIT_EVERY), 'post')
        f = dict(day=day, m=int(w['m']), t_ms=int(round(float(w['t']))), hour=0, side=0, oid=1, status='ok', sel=sel, fork_fingerprint=ctx['fingerprint'], Y=st['Y'],
                 arms={n: {h_: a[h_] for h_ in ('value', 'fills_sha256', 'decisions_sha256', 'requests_sha256')} for n, a in st['arms'].items()})
        r0 = json.loads(json.dumps(M.fork_package(f, wb, Mst, sst), default=str)); break
    if r0 is None or r0['status'] != 'ok' or not r0['shadow'] or raw_row_bad_v2(r0):
        bad.append(f"a clean synthetic row with logged overrides: {None if r0 is None else (r0['status'], len(r0['shadow']), raw_row_bad_v2(r0))}")
    else:
        chk = int((r0['arms']['POST_KEEP_INITIAL']['causal'] or {}).get('checked', 0))
        for lab, fn in (('a counter below the log', lambda x: x.update(overrides=len(x['shadow']) - 1)),
                        ('an excess beyond the probe bound', lambda x: x.update(overrides=len(x['shadow']) + 2 * chk + 1)),
                        ('a shadow entry of another gid', lambda x: x['shadow'][0].update(gid=int(x['shadow'][0]['gid']) + 1))):
            x = copy.deepcopy(r0); fn(x)
            if not raw_row_bad_v2(x):
                bad.append(f'{lab}: not flagged')
        x = copy.deepcopy(r0); x['overrides'] = len(x['shadow']) + min(2 * chk, 1)
        if raw_row_bad_v2(x):
            bad.append('an excess inside the probe bound was flagged')
    if bad:
        sys.exit('EXPOSURE CHECKS V2 SELFTEST FAILED: ' + '; '.join(bad))
    print('exposure checks v2 selftest passed: the frozen synthetic selftest under the v2 rule; a counter below the log, an excess beyond the probe '
          'bound and a shadow entry of another gid are flagged; an excess inside the bound is not')


if __name__ == '__main__':
    if sys.argv[1:] == ['selftest']:
        selftest_v2()
    else:
        main_v2(); falsify_v2()
