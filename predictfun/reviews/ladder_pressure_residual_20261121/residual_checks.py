"""INDEPENDENT CHECKS of PRESSURE-RESIDUAL-TIMING-AUDIT-v1 (declaration.md; session aea5bfb9, 2026-11-21). Recomputed from the stored evidence (the
ablation receipt's FULL_11 ledgers and command arrays, the pressure receipt's cache arrays, the frozen edges) with this checker's own status code;
the residual module is not imported (its outputs are read as data).
 A identity: frozen files committed; the manifest's ablation part records = the data files = the committed copies; the parent day records;
   the output hashes in audit.json;
 B membership: fills.csv holds every FULL_11 fill of the 19 complete markets exactly once and generations.csv every order once;
 F statuses: this checker's own classification of every fill (pending by request reason before a strict removal; before T1 + 37; a timely
   evaluable ON a mismatch; all feasible ticks non-evaluable; else unflagged) and its feasible / evaluable / ON counts equal the stored ones;
 F values: every fill's edge, +50, +100, eligible +1 s, movement, late and END recomputed; per market the sums equal the FULL_11 arm row
   and the ledger value;
 F tables and reading: the status table, the generation statuses and the reading (fractions by day, the answer by the declared rule)
   recomputed from this checker's statuses and values;
 I resources: this round <= 0.12 core-hours; the ladder line (its six earlier receipts and the review logs) <= 2.0.
falsify(): each corruption of a copy must fail its family while the clean evidence passes. Usage: residual_checks.py | residual_checks.py rehearse (the
saved-input pilot's outputs before the full audit: code paths; B_membership false by design)."""
import collections
import copy
import glob
import gzip
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_pressure_residual_20261121'; D = REPO / 'data/predict_fun/research/ladder_pressure_residual_v1_20261121'
AD = REPO / 'data/predict_fun/research/ladder_pressure_ablation_v1_20261121'; AR = REPO / 'predictfun/reviews/ladder_pressure_ablation_20261121'
PD_ = REPO / 'data/predict_fun/research/ladder_pressure_v1_20261121'; RUN = REPO / 'data/predict_fun/research/ladder_rest_v1_20261121'; EDGE_FILE = RUN / 'measure' / 'audit.json'
COMMITTED = ['predictfun/research/pf_ladder_pressure_residual.py', 'predictfun/research/pf_ladder_pressure_residual_selftest.py', 'predictfun/research/pf_ladder_pressure_residual_mutants.py',
             'predictfun/reviews/ladder_pressure_residual_20261121/declaration.md', 'predictfun/reviews/ladder_pressure_residual_20261121/manifest.json',
             'predictfun/reviews/ladder_pressure_residual_20261121/launch_residual.sh', 'predictfun/reviews/ladder_pressure_residual_20261121/residual_checks.py',
             'predictfun/reviews/ladder_pressure_residual_20261121/tools/pilot_projection_residual.py']
LINE = ['ladder_rest_20261121', 'ladder_audit_20261121', 'ladder_guard_20261121', 'ladder_pressure_20261121', 'ladder_pressure_lag_20261121', 'ladder_pressure_ablation_20261121']
REVIEWS = ['20261121_ladder_rest_review', '20261121_ladder_audit_review', '20261121_ladder_guard_review', '20261121_ladder_pressure_review', '20261121_ladder_pressure_lag_review',
           '20261121_ladder_pressure_ablation_review']
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
KINDS, ACTIONS = ('live', 'slot'), ('POST', 'WAIT', 'KEEP', 'PULL')
REASONS = ('post', 'keep', 'inventory_pull', 'restricted', 'zero_desired', 'exposure_full', 'no_room', 'cap', 'near_end', 'own_cross_blocked', 'unavailable_price', 'fallback', 'pressure_pull',
           'pressure_withhold')
TIMING = ('pending_pressure', 'pending_inventory', 'pending_structural', 'before_feasible_effect'); SG = {0: 1., 1: -1.}; LAT = 37.; CAP, LINE_CAP, MIN_NEG = .12, 2., 30


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def blob(rel):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True); return r.stdout if r.returncode == 0 else None


def gz(p):
    return [json.loads(x) for x in gzip.decompress(Path(p).read_bytes()).decode().splitlines() if x]


def close(a, b, tol=1e-6):
    a = float('nan') if a is None else float(a); b = float('nan') if b is None else float(b)
    return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= tol * max(1., abs(a), abs(b))


def load(scope='full'):
    gm = json.loads((R / 'manifest.json').read_text()); E = dict(gm=gm, led={}, cmd={}, cache={}, scope=scope)
    for day in gm['days']:
        z = np.load(PD_ / 'days' / f'{day}_arrays.npz')
        for part in (('pilot',) if scope == 'pilot' else ('pilot', 'rest')):
            rec = json.loads((AD / part / f'{day}.json').read_text())
            if not rec['members']:
                continue
            za = np.load(AD / part / f'{day}_arrays.npz')
            for r in gz(AD / part / f'{day}.jsonl.gz'):
                m = int(r['m']); E['led'][(day, m)] = r['FULL_11']; E['cmd'][(day, m)] = za[f'{m}__FULL_11__commands']
                E['cache'][(day, m)] = {k: {s: z[f'{m}__cache__{k}{s}'] for s in (0, 1)} for k in ('x', 'age', 'status')}
    for n in ('fills', 'generations', 'status_table', 'generation_status'):
        E[n] = pd.read_csv(D / scope / f'{n}.csv', dtype={'day': str}, float_precision='round_trip')
    E['audit'] = json.loads((D / scope / 'audit.json').read_text()); E['arms'] = pd.read_csv(AD / 'measure' / 'arms.csv', dtype={'day': str}, float_precision='round_trip')
    E['last'] = {s: float(json.loads(EDGE_FILE.read_text())['edges'][f'live|{s}|adv_bn_r250'][-1]) for s in (0, 1)}; return E


def on(x, last):
    return bool(np.isfinite(x) and x > 0. and x >= last)


def my_status(t_f, o, cmds, cz, last):
    """This checker's own classification (header)."""
    s = int(o['side']); lv = [(float(c[0]), REASONS[int(c[8])], ACTIONS[int(c[7])]) for c in cmds if KINDS[int(c[2])] == 'live']
    if o['req'] is not None and float(o['req']) < t_f:
        if not t_f < float(o['remove']):
            return 'event_or_action_mismatch', (0, 0, 0)
        if o['why'] == 'structural':
            return 'pending_structural', (0, 0, 0)
        rs = [r for t, r, a in lv if a == 'PULL' and t == float(o['req'])]
        return ({'pressure_pull': 'pending_pressure', 'inventory_pull': 'pending_inventory'}.get(rs[0], 'event_or_action_mismatch') if len(rs) == 1 else 'event_or_action_mismatch'), (0, 0, 0)
    if not lv or t_f < lv[0][0] + LAT:
        return 'before_feasible_effect', (0, 0, 0)
    n = ev = hits = 0
    for t, r, a in lv:
        if t + LAT > t_f:
            break
        k = int(round(t / 50.)); good = r != 'fallback' and int(cz['status'][s][k]) == 0; n += 1; ev += good; hits += int(good and on(float(cz['x'][s][k]), last[s]))
    return ('event_or_action_mismatch' if hits else 'nonevaluable' if not ev else 'unflagged_later'), (n, ev, hits)


def my_rows(E):
    out = []
    for (day, m), L_ in E['led'].items():
        od = {int(o['gid']): o for o in L_['orders']}; cg = collections.defaultdict(list); end, mark = float(L_['end']), float(L_['mark'])
        for c in E['cmd'][(day, m)]:
            if int(c[3]) >= 0:
                cg[int(c[3])].append(c)
        for i, (f, mm) in enumerate(zip(L_['fills'], L_['fill_mids'])):
            o = od[int(f[1])]; st, (n, ev, hits) = my_status(float(f[0]), o, cg.get(int(f[1]), []), E['cache'][(day, m)], E['last']); t, s, q, p = float(f[0]), int(f[2]), float(f[3]), float(f[4])
            el = t <= end - 1000. and mm[6] is not None; v = lambda j: np.nan if mm[j] is None else SG[s] * q * (float(mm[j]) - p)
            out.append(dict(day=day, m=m, fill=i, status=st, n_feasible=n, n_evaluable=ev, n_on=hits, q=q, edge=v(0), mo50=v(2), mo100=v(4), mo1s=v(6) if el else np.nan, eligible=el,
                            move_end=SG[s] * q * (mark - float(mm[6])) if el else 0., late_end=0. if el else SG[s] * q * (mark - p), end=SG[s] * q * (mark - p), side=s,
                            rung=int(round(float(o['rung'])))))
    return pd.DataFrame(out)


def fam_A(E):
    gm = E['gm']; c = dict(A_committed=all(blob(p) == (REPO / p).read_bytes() for p in COMMITTED))
    c['A_inputs'] = all(sha(AD / p / f'{d}.json') == gm['ablation']['part_records'][f'{p}|{d}'] and blob(f'predictfun/reviews/ladder_pressure_ablation_20261121/outputs/{p}/{d}.json') == (AD / p / f'{d}.json').read_bytes()
                        for p in ('pilot', 'rest') for d in gm['days']) and all(sha(PD_ / 'days' / f'{d}.json') == gm['parent_day_records'][d] for d in gm['days'])
    c['A_outputs'] = all(E['audit']['outputs'][n] == sha(D / E['scope'] / n) for n in ('fills.csv', 'generations.csv', 'status_table.csv', 'generation_status.csv'))
    return c, {}


def fam_B(E):
    F, G = E['fills'], E['generations']; want = sorted((d, m, i) for (d, m), L_ in E['led'].items() for i in range(len(L_['fills']))); got = sorted(zip(F.day, F.m.astype(int), F.fill.astype(int)))
    gw = sorted((d, m, int(o['gid'])) for (d, m), L_ in E['led'].items() for o in L_['orders']); gg = sorted(zip(G.day, G.m.astype(int), G.gid.astype(int)))
    return dict(B_membership=len(E['led']) == 19 and got == want and gg == gw), dict(B=dict(markets=len(E['led']), fills=len(want), generations=len(gw)))


def fam_F(E):
    me = my_rows(E); F = E['fills'].copy(); key = ['day', 'm', 'fill']; J = me.merge(F, on=key, suffixes=('', '_s'), how='left'); c = {}
    c['F_statuses'] = bool(len(J) == len(F) and (J.status == J.status_s).all() and all((J[k] == J[f'{k}_s']).all() for k in ('n_feasible', 'n_evaluable', 'n_on')))
    c['F_values'] = bool(all(all(close(a, b) for a, b in zip(J[k], J[f'{k}_s'])) for k in ('q', 'edge', 'mo50', 'mo100', 'mo1s', 'move_end', 'late_end', 'end')))
    A_ = E['arms'][E['arms'].arm == 'FULL_11'].set_index(['day', 'm']); rb = 0
    for (d, m), L_ in E['led'].items():
        g = me[(me.day == d) & (me.m == m)]; a = A_.loc[(d, m)]
        rb += int(not (len(g) == a.fills and close(g.q.sum(), a.all_q) and close(g.end.sum(), a.all_end) and close(g.end.sum(), L_['value']) and close(g.mo1s.fillna(0.).sum(), a.all_id_mo1s)
                       and close(g.edge.fillna(0.).sum(), a.all_edge) and close((g.mo1s.fillna(0.) + g.move_end + g.late_end).sum(), g.end.sum())))
    c['F_reconcile'] = rb == 0
    T = E['status_table']; tb = 0
    for k_, g in me.groupby(['status', 'day', 'side', 'rung']):
        row = T[(T.status == k_[0]) & (T.day == k_[1]) & (T.side == k_[2]) & (T.rung == k_[3])]; el = g[g.eligible]; mo = el.mo1s.astype(float)
        exp = dict(fills=len(g), shares=g.q.sum(), edge=g.edge.dropna().sum(), mo1s=mo.sum(), mo1s_pos=mo[mo > 0].sum(), mo1s_neg=mo[mo < 0].sum(), move_end=g.move_end.sum(),
                   late_end=g.late_end.sum(), end=g.end.sum(), mo1s_valid_q=el.q.sum())
        tb += int(len(row) != 1 or not all(close(row[k].iloc[0], v) for k, v in exp.items()))
    c['F_table'] = tb == 0 and len(T) == me.groupby(['status', 'day', 'side', 'rung']).ngroups
    rd = E['audit']['reading']; el = me[me.eligible].copy(); el['neg'] = el.mo1s.clip(upper=0.); days = sorted(me.day.unique()); okr = True; ft, fu = [], []
    for d in days + ['all']:
        g = el if d == 'all' else el[el.day == d]; tot = g.neg.sum(); a_t = g[g.status.isin(TIMING)].neg.sum() / tot; a_u = g[g.status == 'unflagged_later'].neg.sum() / tot
        okr &= close(rd['by_day'][d]['frac_timing'], a_t) and close(rd['by_day'][d]['frac_unflagged'], a_u) and close(rd['by_day'][d]['total']['neg'], tot)
        if d != 'all':
            ft.append(a_t); fu.append(a_u)
    mism = int((me.status == 'event_or_action_mismatch').sum()); thin = any(int((el[el.day == d].neg < 0).sum()) < MIN_NEG for d in days)
    ans = 'AUDIT_FAILED' if mism else 'THIN' if thin else 'TIMING_LIMITED_DOMINANT' if all(x > .5 for x in ft) else 'UNFLAGGED_EXPOSURE_DOMINANT' if all(x > .5 for x in fu) else 'MIXED'
    c['F_reading'] = bool(okr and rd['answer'] == ans and rd['mismatches'] == mism)
    return c, dict(F_answer=ans, F_mismatches=mism, F_status_counts=me.status.value_counts().to_dict())


def cpu(paths):
    tot, peak = 0., 0
    for p in paths:
        for ln in Path(p).read_text().splitlines():
            m_ = RL.match(ln)
            if m_:
                tot += float(m_.group(6)); peak = max(peak, int(m_.group(4)))
    return tot / 3600., peak


def fam_I(E, root=None):
    own, peak = cpu(glob.glob(str((root or R) / 'logs' / '**' / '*.log'), recursive=True))
    line, _ = cpu([p for n in LINE for p in glob.glob(str(REPO / 'predictfun/reviews' / n / 'logs' / '**' / '*.log'), recursive=True)])
    rev, _ = cpu([p for n in REVIEWS for p in glob.glob(str(REPO / 'predictfun/reviews/automatic' / n / '*.log'))])
    return dict(I_resources=own <= CAP and peak < 1536 and own + line + rev <= LINE_CAP), dict(I=dict(round_core_hours=own, line_core_hours=own + line + rev, peak_mib=peak))


FAMS = dict(A=fam_A, B=fam_B, F=fam_F, I=fam_I)


def run_checks(E):
    chk, det = {}, {}
    for k in FAMS:
        c, d = FAMS[k](E); chk.update(c); det.update(d)
    return dict(checks=chk, details=det, all_ok=bool(all(chk.values())))


def _set(E, name, df):
    E[name] = df


def _fake(E):
    E['_root'] = Path(tempfile.mkdtemp(dir=str(D))); (E['_root'] / 'logs').mkdir()
    (E['_root'] / 'logs' / 'fake.log').write_text('2026-11-21T00:00:00Z tag=fake rc=0 peak_rss_mib=2000 wall_s=600.0 cpu_user_s=600.0\n')


CASES = [
    ('a_status_changed', 'F', 'F_statuses', lambda E: _set(E, 'fills', E['fills'].assign(status=np.where(E['fills'].index == 0, 'unflagged_later' if E['fills'].status.iat[0] != 'unflagged_later' else 'nonevaluable', E['fills'].status)))),
    ('a_value_changed', 'F', 'F_values', lambda E: _set(E, 'fills', E['fills'].assign(end=E['fills'].end + np.where(E['fills'].index == 0, 1., 0.)))),
    ('a_fill_duplicated', 'B', 'B_membership', lambda E: _set(E, 'fills', pd.concat([E['fills'], E['fills'].iloc[[0]]], ignore_index=True))),
    ('a_table_cell_changed', 'F', 'F_table', lambda E: _set(E, 'status_table', E['status_table'].assign(mo1s_neg=E['status_table'].mo1s_neg + np.where(E['status_table'].index == 0, -1., 0.)))),
    ('the_reading_changed', 'F', 'F_reading', lambda E: E.__setitem__('audit', dict(E['audit'], reading=dict(E['audit']['reading'], answer='TIMING_LIMITED_DOMINANT' if E['audit']['reading']['answer'] != 'TIMING_LIMITED_DOMINANT' else 'MIXED')))),
    ('an_output_hash_changed', 'A', 'A_outputs', lambda E: E.__setitem__('audit', dict(E['audit'], outputs=dict(E['audit']['outputs'], **{'fills.csv': '0' * 64})))),
    ('resources_over_the_cap', 'I', 'I_resources', _fake),
]


def falsify(E, base):
    import shutil
    out = {}
    for name, fam, target, fn in CASES:
        X_ = dict(E); fn(X_)
        try:
            c, _ = FAMS[fam](X_, X_['_root']) if fam == 'I' else FAMS[fam](X_)
        finally:
            if '_root' in X_:
                shutil.rmtree(X_['_root'], ignore_errors=True)
        out[name] = dict(family=fam, target=target, clean=bool(base['checks'][target]), refused=not c[target])
    return dict(cases=out, falsifier_ok=bool(base['all_ok'] and all(v['clean'] and v['refused'] for v in out.values())))


def main():
    scope = 'pilot' if len(sys.argv) > 1 and sys.argv[1] == 'rehearse' else 'full'; dest = D / 'pilot' if scope == 'pilot' else R; tag = '_rehearsal' if scope == 'pilot' else ''
    E = load(scope); res = run_checks(E); (dest / f'checks{tag}.json').write_text(json.dumps(res, indent=1, default=str) + '\n'); fz = falsify(E, res)
    (dest / f'checks_falsifier{tag}.json').write_text(json.dumps(fz, indent=1) + '\n')
    print(json.dumps(dict(all_ok=res['all_ok'], failed=[k for k, v in res['checks'].items() if not v], falsifier_ok=fz['falsifier_ok'], cases={k: v['refused'] for k, v in fz['cases'].items()},
                          details=res['details']), indent=1, default=str)[:3000])


if __name__ == '__main__':
    main()
