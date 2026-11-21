"""ENTRY-TARGET-AUDIT-v1: THE DECLARED CHECKS (review d6d5d55e; 2026-11-21, session aea5bfb9). Declared in the freeze, before the audit runs.
Each check recomputes from the saved rows and the accepted inputs; a check that cannot run is false, with its reason.
 A identity: audit.json names the two declared datasets; every POST row of each accepted dataset appears exactly once (the count from its
   accepted.json statuses, unique keys); the non-ok statuses reconcile with accepted.json's POST statuses.
 B accounting: every accounted row has a declared class, Y = V_initial + D_focal + D_opposite within 1e-6 c (recomputed here), the first
   four classes carry |Y| <= 1e-6 c and meaningful > 1e-6 c; no identity_failure or forced_generation_not_unique row.
 C predictions: every ok Oct 26-27 row is joined and equal to R4E's saved predictions by the exact key (re-joined here); every ok Oct 24-25
   row carries in-sample predictions.
 D reached: the reached rows are R4E's 760 ok pairs, every one accounted with the identity.
 E summaries: the sets' totals (rows, weight, meaningful share, no-fill share, absolute V_initial and Y) and the predicates recomputed here
   equal groups.csv and audit.json.
 F resources: every run_light job of this receipt peaked below 2,048 MiB; the round's CPU is within 1 core-hour.
 G suites: pf_mutant_summary.judge on every record of the audit harness against the final sources; one record per file; no other file.
FALSIFIER (falsify()): B, A, C, E and G on copies of their real inputs are true clean and false after one corruption each (a V_initial; a
 dropped row; a joined prediction; a stored predicate; a copied mutant record), each corruption applied on a real row.
Usage (run_light): audit_checks.py  (the checks, then the falsifier)"""
import glob
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; RES = Path('/home/yuqing/ctaNew/predictfun/research'); sys.path.insert(0, str(RES))
import pf_entry_audit as M                               # noqa: E402

C, pd, NE, N = M.C, M.pd, M.NE, M.N
LOGS = HERE / 'logs'
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')


def rows(out):
    return NE.read_r3(Path(out) / 'rows.csv.gz')


def check_A(out, R):
    D = rows(out); js = json.loads((Path(out) / 'audit.json').read_text()); bad = []
    for k, s in M.SETS.items():
        acc = json.loads((Path(s['dir']) / 'accepted.json').read_text()); g = D[D.set == k]
        post = {kk: v for kk, v in acc['statuses'].items() if kk.split()[1] == 'post'}
        if js['datasets'][k] != s['sha'] or acc['dataset_sha256'] != s['sha'] or len(g) != sum(post.values()) or g.duplicated(['day', 'm', 't_ms', 'side', 'pick']).any():
            bad.append(f'{k} identity / count / duplicates')
        got = g.assign(st=g.status.where(~g.status.isin(['identity_failure', 'forced_generation_not_unique']), 'ok')).groupby(['day', 'st']).size()
        want = {(kk.split()[0], kk.split(maxsplit=2)[2]): v for kk, v in post.items()}
        if {(str(d), st): int(n) for (d, st), n in got.items()} != want:
            bad.append(f'{k} statuses')
    R['A'] = dict(bad=bad, rows=int(len(D)))
    return not bad


def check_B(out, R):
    D = rows(out); ok = D[D.status.isin(['ok', 'identity_failure', 'forced_generation_not_unique'])]
    gap = np.abs(ok.Y - (ok.v_initial + ok.d_focal + ok.d_opposite)); y = ok.Y.abs()
    cons = np.where(ok.cls == 'meaningful', y > M.MEANINGFUL, y <= M.MEANINGFUL)
    R['B'] = dict(accounted=int(len(ok)), max_gap=float(gap.max()) if len(ok) else None, failures=int((ok.status != 'ok').sum()), bad_class=int((~ok.cls.isin(M.CLASSES)).sum()),
                  inconsistent=int((~cons).sum()))
    return bool(len(ok) and (ok.status == 'ok').all() and ok.cls.isin(M.CLASSES).all() and bool(cons.all()) and float(gap.max()) <= M.TOL)


def check_C(out, R):
    D = rows(out); ev = D[(D.set == 'eval') & (D.status == 'ok')]; tr = D[(D.set == 'train') & (D.status == 'ok')]
    P = NE.read_r3(N.REPO / M.R4E_PRED); P['day'] = P.day.astype(str); key = ['day', 'm', 't_ms', 'side', 'pick']
    j = ev.assign(day=ev.day.astype(str)).merge(P[key + ['pred_SMOOTH', 'pred_RICH']], on=key, how='left', suffixes=('', '_r4e'))
    same = bool(len(j) == len(ev) and np.array_equal(j.pred_SMOOTH.values, j.pred_SMOOTH_r4e.values) and np.array_equal(j.pred_RICH.values, j.pred_RICH_r4e.values))
    R['C'] = dict(eval_ok=int(len(ev)), joined=int((ev.pred_status == 'joined').sum()), train_ok=int(len(tr)), in_sample=int((tr.pred_status == 'in_sample').sum()), equal_to_r4e=same)
    return bool(len(ev) and (ev.pred_status == 'joined').all() and (tr.pred_status == 'in_sample').all() and same)


def check_D(out, R):
    Rr = pd.read_csv(Path(out) / M.FILES['reached_rows'], float_precision='round_trip'); ok = Rr[Rr.status == 'ok']
    gap = np.abs(ok.Y - (ok.v_initial + ok.d_focal + ok.d_opposite))
    n_r4e = int((pd.read_csv('/home/yuqing/ctaNew/predictfun/reviews/r4e_entry_smooth_20261121/replay/reached.csv.gz' if Path('/home/yuqing/ctaNew/predictfun/reviews/r4e_entry_smooth_20261121/replay/reached.csv.gz').exists()
                          else '/home/yuqing/ctaNew/predictfun/reviews/r4e_entry_smooth_20261121/replay/reached.csv').status == 'ok').sum())
    R['D'] = dict(rows=int(len(Rr)), ok=int(len(ok)), r4e_ok=n_r4e, max_gap=float(gap.max()) if len(ok) else None)
    return bool(len(ok) == n_r4e == 760 and len(Rr) == len(ok) and float(gap.max()) <= M.TOL)


def check_E(out, R):
    D = rows(out); G = pd.read_csv(Path(out) / 'groups.csv', float_precision='round_trip'); js = json.loads((Path(out) / 'audit.json').read_text()); bad = []
    for k in M.SETS:
        g = D[(D.set == k) & (D.status == 'ok')]; w = g.w.astype(float); W = w.sum(); y = g.Y.astype(float)
        mine = dict(rows=len(g), w_sum=W, share_meaningful=float(w[y.abs() > M.MEANINGFUL].sum() / W), init_no_fill=float(w[g.init_fill == 'no_fill'].sum() / W),
                    abs_v_initial=float((w * g.v_initial.abs()).sum()), abs_Y=float((w * y.abs()).sum()))
        row = G[(G.set == k) & (G.day == 'all')].iloc[0]
        if any(abs(float(row[f]) - float(v)) > 1e-9 * max(1., abs(float(v))) for f, v in mine.items()):
            bad.append(f'{k} totals')
        if M.predicates(dict(mine))['diagnosis'] != js['predicates'][k]['diagnosis']:
            bad.append(f'{k} predicates')
    R['E'] = dict(bad=bad, predicates={k: js['predicates'][k]['diagnosis'] for k in M.SETS})
    return not bad


def check_F(R):
    peaks, cpu, bad = [], 0., []
    for p in glob.glob(str(LOGS / '**' / '*.log'), recursive=True):
        for ln in Path(p).read_text().splitlines():
            m = RL.match(ln)
            if m:
                peaks.append(int(m.group(4))); cpu += float(m.group(6)); bad += [m.group(2)] if int(m.group(4)) >= 2048 else []
    R['F'] = dict(jobs=len(peaks), peak_mib=max(peaks) if peaks else None, core_hours=cpu / 3600., bad=bad)
    return bool(peaks and not bad and cpu / 3600. <= 1.)


def check_G(R):
    import importlib.util
    import pf_mutant_one as O
    import pf_mutant_summary as S
    h = 'pf_entry_audit_mutants.py'; spec = importlib.util.spec_from_file_location(h[:-3], RES / h); H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
    d = LOGS / 'mutants_audit'; files = {'control': d / 'mut_audit_control.log', **{i: d / f'mut_audit_{i}.log' for i in range(len(H.MUTANTS))}}
    recs = {k: S.last_record(p) for k, p in files.items()}
    one = all(p.exists() and sum(1 for ln in p.read_text().splitlines() if ln.startswith('{') and '"cls"' in ln) == 1 for p in files.values())
    extra = sorted({x.name for x in d.glob('mut_audit_*.log')} - {p.name for p in files.values()})
    ok, bad = S.judge(len(H.MUTANTS), recs, O.sources(H), [m_[0] for m_ in H.MUTANTS])
    R['G'] = dict(planted=len(H.MUTANTS), judge_ok=ok, failures=bad[:5], one_record_per_file=one, extra_files=extra)
    return bool(ok and one and not extra)


def main(out=M.OUT):
    out = Path(out); R = {}; ck = {}
    for k, f in (('A_identity', check_A), ('B_accounting', check_B), ('C_predictions', check_C), ('D_reached', check_D), ('E_summaries', check_E)):
        try:
            ck[k] = f(out, R)
        except Exception as e:
            ck[k] = False; R[k[0]] = dict(error=repr(e)[:300])
    for k, f in (('F_resources', check_F), ('G_suites', check_G)):
        try:
            ck[k] = f(R)
        except Exception as e:
            ck[k] = False; R[k[0]] = dict(error=repr(e)[:300])
    J = dict(checks=ck, all_true=all(ck.values()), readings=R, source_sha256=C.sha(Path(__file__)), asof=C.stamp())
    (HERE / 'checks.json').write_text(json.dumps(J, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(checks=ck, all_true=J['all_true']), indent=1))


def falsify(out=M.OUT):
    global LOGS
    out = Path(out); res = {}

    def safe(f, *a):
        try:
            return f(*a)
        except Exception:                                                                 # noqa: BLE001
            return False
    with tempfile.TemporaryDirectory(dir=str(out)) as td:
        td = Path(td)

        def copy(name, fn=None, target='rows.csv.gz'):
            d = td / name; d.mkdir()
            for q in out.iterdir():
                if q.is_file():
                    shutil.copy(q, d / q.name)
            if fn is not None:
                if target.endswith('.json'):
                    j = json.loads((d / target).read_text()); applied = fn(j); (d / target).write_text(json.dumps(j))
                else:
                    D = NE.read_r3(d / target); applied = fn(D); D.to_csv(d / target, index=False)
                return d, applied
            return d, True

        def v_bump(D):
            i = D.index[(D.status == 'ok') & (D.cls == 'meaningful')][0]; D.loc[i, 'v_initial'] += 1.; return True
        def drop(D):
            D.drop(index=D.index[(D.set == 'train') & (D.status == 'ok')][0], inplace=True); return True
        def pred(D):
            i = D.index[(D.set == 'eval') & (D.status == 'ok')][0]; D.loc[i, 'pred_SMOOTH'] += 1e-9; return True
        def diag(j):
            k = next(iter(j['predicates'])); j['predicates'][k]['diagnosis'] = 'flipped'; return True
        clean, _ = copy('clean')
        for name, chk, fn, tgt in (('B', check_B, v_bump, 'rows.csv.gz'), ('A', check_A, drop, 'rows.csv.gz'), ('C', check_C, pred, 'rows.csv.gz'), ('E', check_E, diag, 'audit.json')):
            d, ap = copy(f'{name}x', fn, tgt); res[name] = dict(clean=safe(chk, clean, {}), corrupted=safe(chk, d, {}), applied=bool(ap))
        h = td / 'H'; shutil.copytree(LOGS / 'mutants_audit', h / 'mutants_audit'); saved = LOGS
        try:
            LOGS = h; cl = safe(check_G, {}); shutil.copy(h / 'mutants_audit' / 'mut_audit_0.log', h / 'mutants_audit' / 'mut_audit_1.log')
            res['G'] = dict(clean=cl, corrupted=safe(check_G, {}), applied=True)
        finally:
            LOGS = saved
    ok = all(v['clean'] is True and v['corrupted'] is False and v['applied'] for v in res.values())
    (HERE / 'checks_falsifier.json').write_text(json.dumps(dict(cases=res, falsifier_ok=ok, asof=C.stamp()), indent=1, sort_keys=True) + '\n')
    print(json.dumps(dict(cases=res, falsifier_ok=ok)))


if __name__ == '__main__':
    main(); falsify()
