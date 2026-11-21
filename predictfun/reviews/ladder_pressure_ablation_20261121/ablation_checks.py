"""INDEPENDENT CHECKS of PRESSURE-ACTION-ATTRIBUTION-v1 (declaration.md; session aea5bfb9, 2026-11-21). Recomputed from the stored evidence: this
round's part records and arrays, the pressure receipt's (parent) records and arrays, the accepted baseline's records and the frozen edge file. The
ablation module is imported for rule_sha() only; pressure_checks.py (793c706b) and lag_checks_v2.py (9cc6161a) for their pure helpers
(recompute, matched_split, bits, close, on; uptime, inventory, queue); no measurement code of the ablation, lag or pressure module is used.
 A identity: frozen files committed; the rule sha256 = the manifest's = every part record's; the parent rule and day records (= the manifest's
   sha256 = the committed receipt copies); the edges; sources; outputs;
 B membership: the parts are disjoint and cover the 19 complete baseline markets; every (day, part) has its record with the manifest's
   members; every market carries the three run arms; the pilot = the first two complete markets of each date;
 C controls: FULL_11 = the parent's pressure arm (fingerprint, fills, orders, requests, value, log, command array, queue records); the parent's
   disabled arm (BASELINE_00) = the accepted baseline;
 D the rule per arm from the parent cache at the current tick: an intervention only at an ON tick with the cached bits and a pressure reason;
   FULL_11 neither KEEPs (but a fallback) nor POSTs (or cap / near-END WAITs after a POST) at an ON tick; PULL_ONLY_10 never withholds and
   never KEEPs at an ON tick (its POSTs during pressure are allowed); WITHHOLD_ONLY_01 never pulls for pressure and never POSTs at an ON tick
   (its KEEPs are allowed); no gid requested twice;
 E accounting: each run arm's cash, position and value from its fills;
 F reconcile: per arm (four) every arms.csv field a reading uses (markouts, END identity terms, rung, interventions, exact uptime, exact
   inventory, the queue fields and pair summaries); the END identity; lost / added per contrast; the contrasts, classifications, market
   counts, top-3 shares, the interaction and all_arms_negative recomputed from the ledgers and compared with the stored readings;
 H queue: the activation-clock snapshot of every non-refused order, a pull entry for every pressure pull, a first-review entry for every
   reviewed gid, and every reentry.csv row field by field against the ledger (review 7fca0db4's finding);
 I resources: this round <= 0.35 core-hours; the ladder line's recorded use (its five earlier receipts, this one and the review logs)
   <= 2.0 core-hours; peaks <= 1,536 MiB.
falsify(): each corruption of a copy must fail its family while the clean evidence passes, incl. a coordinated markout-and-contrast case and a
changed re-entry price. Usage: ablation_checks.py | ablation_checks.py rehearse <dir> (pilot copy; B_membership false by design)."""
import collections
import copy
import glob
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_pressure_ablation_20261121'; D = REPO / 'data/predict_fun/research/ladder_pressure_ablation_v1_20261121'
PD_ = REPO / 'data/predict_fun/research/ladder_pressure_v1_20261121'; PR = REPO / 'predictfun/reviews/ladder_pressure_20261121'
RUN = REPO / 'data/predict_fun/research/ladder_rest_v1_20261121'; EDGE_FILE = RUN / 'measure' / 'audit.json'
COMMITTED = ['predictfun/research/pf_ladder_pressure_ablation.py', 'predictfun/research/pf_ladder_pressure_ablation_selftest.py', 'predictfun/research/pf_ladder_pressure_ablation_mutants.py',
             'predictfun/reviews/ladder_pressure_ablation_20261121/declaration.md', 'predictfun/reviews/ladder_pressure_ablation_20261121/manifest.json',
             'predictfun/reviews/ladder_pressure_ablation_20261121/launch_ablation.sh', 'predictfun/reviews/ladder_pressure_ablation_20261121/ablation_checks.py',
             'predictfun/reviews/ladder_pressure_ablation_20261121/tools/pilot_projection_ablation.py']
LINE = ['ladder_rest_20261121', 'ladder_audit_20261121', 'ladder_guard_20261121', 'ladder_pressure_20261121', 'ladder_pressure_lag_20261121']
REVIEWS = ['20261121_ladder_rest_review', '20261121_ladder_audit_review', '20261121_ladder_guard_review', '20261121_ladder_pressure_review', '20261121_ladder_pressure_lag_review']
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
STEP, SG, CAP, LINE_CAP = 50., {0: 1., 1: -1.}, .35, 2.
ARMS = ('BASELINE_00', 'PULL_ONLY_10', 'WITHHOLD_ONLY_01', 'FULL_11'); RUN_ARMS = ('FULL_11', 'PULL_ONLY_10', 'WITHHOLD_ONLY_01')
BITS = {'BASELINE_00': (0, 0), 'PULL_ONLY_10': (1, 0), 'WITHHOLD_ONLY_01': (0, 1), 'FULL_11': (1, 1)}
CONTRASTS = {'cancel_given_withhold': ('FULL_11', 'WITHHOLD_ONLY_01'), 'withhold_given_cancel': ('FULL_11', 'PULL_ONLY_10'),
             'cancel_alone': ('PULL_ONLY_10', 'BASELINE_00'), 'withhold_alone': ('WITHHOLD_ONLY_01', 'BASELINE_00')}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


PC = _load('pressure_checks', PR / 'pressure_checks.py'); LC2 = _load('lag_checks_v2', REPO / 'predictfun/reviews/ladder_pressure_lag_20261121/lag_checks_v2.py')
sys.path.insert(0, str(REPO / 'predictfun/research'))
import pf_ladder_pressure_ablation as AB                  # noqa: E402  rule_sha only
P, M = AB.P, AB.M
bits, close, on, sha, git_blob, gz = PC.bits, PC.close, PC.on, PC.sha, PC.git_blob, PC.gz
PARTS_ = ('pilot', 'rest')


def load():
    gm = json.loads((R / 'manifest.json').read_text()); E = dict(gm=gm, prec={}, recs=[], parent={}, parr={}, base={}, arr={})
    for day in gm['days']:
        E['parent'].update({(day, int(r['m'])): r for r in gz(PD_ / 'days' / f'{day}.jsonl.gz')}); E['base'].update({(day, int(r['m'])): r for r in gz(RUN / 'days' / f'{day}.jsonl.gz')})
        z = np.load(PD_ / 'days' / f'{day}_arrays.npz'); E['parr'][day] = {k: z[k] for k in z.files}
        for part in PARTS_:
            rec = json.loads((D / part / f'{day}.json').read_text()); E['prec'][(day, part)] = rec
            if rec['members']:
                E['recs'] += [dict(r, _day=day, _part=part) for r in gz(D / part / f'{day}.jsonl.gz')]
                z = np.load(D / part / f'{day}_arrays.npz'); E['arr'][(day, part)] = {k: z[k] for k in z.files}
    for n in ('arms', 'lost_added', 'reentry', 'control'):
        E[n] = pd.read_csv(D / 'measure' / f'{n}.csv', dtype={'day': str}, float_precision='round_trip')
    E['maudit'] = json.loads((D / 'measure' / 'audit.json').read_text()); E['edges_raw'] = json.loads(EDGE_FILE.read_text())['edges']
    return E


def cache(E, day, m):
    a = E['parr'][day]; return {k: {s: a[f'{m}__cache__{k}{s}'] for s in (0, 1)} for k in ('x', 'age', 'status')}


def ledger(E, r, arm):
    return E['parent'][(r['_day'], int(r['m']))]['disabled'] if arm == 'BASELINE_00' else r[arm]


def cmds(E, r, arm):
    return E['parr'][r['_day']][f"{int(r['m'])}__disabled__commands"] if arm == 'BASELINE_00' else E['arr'][(r['_day'], r['_part'])][f"{int(r['m'])}__{arm}__commands"]


def fam_A(E):
    gm = E['gm']; pm = json.loads((PR / 'manifest.json').read_text()); c = {}; recs = list(E['prec'].values())
    c['A_committed'] = all(git_blob(p) == (REPO / p).read_bytes() for p in COMMITTED)
    c['A_rule'] = gm['rule_sha256'] == AB.rule_sha() and all(r['rule_sha256'] == gm['rule_sha256'] for r in recs)
    c['A_parent'] = gm['parent']['rule_sha256'] == pm['rule_sha256'] == P.rule_sha() and all(
        sha(PD_ / 'days' / f'{d}.json') == gm['parent']['day_records'][d] and git_blob(f'predictfun/reviews/ladder_pressure_20261121/outputs/days/{d}.json') == (PD_ / 'days' / f'{d}.json').read_bytes()
        and all(r.get('parent_day_record_sha256', gm['parent']['day_records'][d]) == gm['parent']['day_records'][d] for (dd, _), r in E['prec'].items() if dd == d) for d in gm['days'])
    c['A_edges'] = sha(EDGE_FILE) == gm['edges_sha256'] and all(gm['edges'][str(s)] == [float(v) for v in E['edges_raw'][f'live|{s}|adv_bn_r250']] for s in (0, 1))
    c['A_sources'] = all(r['sources'] == gm['sources'] for r in recs)
    c['A_outputs'] = all(sha(D / part / f'{day}.jsonl.gz') == r['sha256']['gz'] and sha(D / part / f'{day}_arrays.npz') == r['sha256']['npz'] for (day, part), r in E['prec'].items() if r['members']) and all(
        E['maudit']['outputs'][n] == sha(D / 'measure' / n) for n in ('arms.csv', 'control.csv', 'lost_added.csv', 'reentry.csv'))
    return c, {}


def fam_B(E):
    gm = E['gm']; comp = {day: sorted(int(m) for (d, m), r in E['base'].items() if d == day and r['status'] == 'complete') for day in gm['days']}
    pl, rs = gm['parts']['pilot'], gm['parts'].get('rest', {}); got = collections.Counter((r['_day'], int(r['m'])) for r in E['recs'])
    disjoint = all(not set(pl[d]) & set(rs.get(d, [])) for d in gm['days'])
    cover = all(sorted(pl[d] + rs.get(d, [])) == comp[d] for d in gm['days'])
    recs_ok = all(E['prec'][(d, p)]['members'] == [int(x) for x in gm['parts'][p][d]] for d in gm['days'] for p in PARTS_)
    complete_ok = sorted(got) == sorted((d, m) for d, v in comp.items() for m in v) and all(v == 1 for v in got.values()) and all(all(a in r for a in RUN_ARMS) for r in E['recs'])
    return dict(B_membership=bool(disjoint and cover and recs_ok and complete_ok), B_pilot=all([int(m) for m in pl[d]] == comp[d][:2] for d in gm['days'])), {}


def fam_C(E):
    bad = bb = 0
    for r in E['recs']:
        day, m = r['_day'], int(r['m']); pr_ = E['parent'][(day, m)]; pp, z = pr_['pressure'], r['FULL_11']
        bad += int(not (z['fingerprint'] == pp['fingerprint'] and z['fills'] == pp['fills'] and z['orders'] == pp['orders'] and z['requests'] == pp['requests'] and z['value'] == pp['value']
                        and z['guard_log'] == pp['guard_log'] and z['queue_venue'] == pp['queue_venue'] and z['queue_known'] == pp['queue_known'] and r['control_all']
                        and np.array_equal(E['arr'][(day, r['_part'])][f'{m}__FULL_11__commands'], E['parr'][day][f'{m}__pressure__commands'], equal_nan=True)))
        b, d_ = E['base'][(day, m)]['ledger'], pr_['disabled']
        bb += int(not (d_['fills'] == b['fills'] and d_['orders'] == b['orders'] and d_['requests'] == b['requests'] and d_['value'] == b['value'] and d_['fingerprint'] == b['fingerprint']))
    return dict(C_full_equals_parent=bad == 0, C_baseline_equals_original=bb == 0), dict(C_full_mismatch=bad, C_baseline_mismatch=bb)


def fam_D(E):
    last = {s: float(E['gm']['edges'][str(s)][-1]) for s in (0, 1)}; KI, AI, RS = M.CMD_KINDS, M.CMD_ACTIONS, P.REASONS; bad = collections.Counter()
    for r in E['recs']:
        day, m = r['_day'], int(r['m']); c = cache(E, day, m); nt = len(c['x'][0])
        ON = {s: np.array([c['status'][s][k] == 0 and on(float(c['x'][s][k]), last[s]) for k in range(nt)], bool) for s in (0, 1)}
        for arm in RUN_ARMS:
            Cb, Wb = BITS[arm]
            for z in r[arm]['guard_log']:
                k, s = int(round(z[0] / STEP)), int(z[1])
                ok_ = c['status'][s][k] == 0 and bits(z[5]) == bits(c['x'][s][k]) and on(float(z[5]), last[s]) and ((z[10] == 'pressure_pull' and Cb) or (z[10] == 'pressure_withhold' and Wb))
                bad[f'{arm}_log'] += int(not ok_)
            for c_ in cmds(E, r, arm):
                k, s, kind, act, why = int(round(c_[0] / STEP)), int(c_[1]), KI[int(c_[2])], AI[int(c_[7])], RS[int(c_[8])]; onk = bool(ON[s][k])
                bad[f'{arm}_keep'] += int(Cb and onk and kind == 'live' and act == 'KEEP' and why != 'fallback')
                bad[f'{arm}_post'] += int(Wb and onk and kind == 'slot' and (act == 'POST' or (act == 'WAIT' and why in ('cap', 'near_end'))))
                bad[f'{arm}_reason'] += int((why == 'pressure_pull' and not (Cb and onk)) or (why == 'pressure_withhold' and not (Wb and onk)))
            rq = collections.Counter(e[2] for e in r[arm]['requests']); bad[f'{arm}_repeat'] += sum(1 for v in rq.values() if v > 1)
    return dict(D_rule=sum(bad.values()) == 0), dict(D_bad={k: v for k, v in bad.items() if v})


def fam_E(E):
    bad = 0
    for r in E['recs']:
        for arm in RUN_ARMS:
            L_ = r[arm]; cash = -sum(SG[int(f[2])] * float(f[3]) * float(f[4]) for f in L_['fills']); pos = sum(SG[int(f[2])] * float(f[3]) for f in L_['fills'])
            bad += int(not (close(cash, L_['cash']) and close(pos, L_['position']) and close(cash + pos * L_['mark'], L_['value'])))
    return dict(E_accounting=bad == 0), {}


def classify(v):
    v = list(v); return 'POSITIVE_EVERY_DAY' if all(x > 0 for x in v) else 'NEGATIVE_EVERY_DAY' if all(x < 0 for x in v) else 'MIXED'


def fam_F(E):  # noqa: C901
    A_ = E['arms']; rb = ib = 0; rc = {}
    for r in E['recs']:
        end = float(r['path_info']['end_rel'])
        for arm in ARMS:
            L_ = ledger(E, r, arm); x = collections.defaultdict(float, PC.recompute(L_, end)); x.update(LC2.uptime(L_)); x.update(LC2.inventory(L_)); q_, _ = LC2.queue(L_, cmds(E, r, arm)); x.update(q_)
            rc[(r['_day'], int(r['m']), arm)] = x; row = A_[(A_.day == r['_day']) & (A_.m == int(r['m'])) & (A_.arm == arm)]
            keys = list(PC.RECON) + list(LC2.UP) + list(LC2.INV) + list(LC2.QF) + list(LC2.PF)
            rb += int(len(row) != 1 or not all(close(row[k].iloc[0], x[k]) for k in keys))
            ib += int(not (close(x['all_id_mo1s'] + x['all_id_move'] + x['all_id_late'], x['all_end']) and close(x['all_end'], x['value'])))
    c = dict(F_reconcile=rb == 0); LA = E['lost_added']; lb = 0
    for r in E['recs']:
        for name, (a1, a0) in CONTRASTS.items():
            fb, fg = ledger(E, r, a0), ledger(E, r, a1); lo = [f for f, u in zip(fb['fills'], PC.matched_split(fb['fills'], fg['fills'])) if u]
            ad = [f for f, u in zip(fg['fills'], PC.matched_split(fg['fills'], fb['fills'])) if u]; en = lambda F_, mk: sum(SG[int(f[2])] * float(f[3]) * (mk - float(f[4])) for f in F_)
            row = LA[(LA.day == r['_day']) & (LA.m == int(r['m'])) & (LA.contrast == name)]
            lb += int(len(row) != 1 or not (int(row.lost_n.iloc[0]) == len(lo) and int(row.added_n.iloc[0]) == len(ad) and close(row.lost_end.iloc[0], en(lo, float(fb['mark'])))
                                             and close(row.added_end.iloc[0], en(ad, float(fg['mark']))) and close(float(fg['value']) - float(fb['value']), en(ad, float(fg['mark'])) - en(lo, float(fb['mark'])))))
    c['F_identity'] = ib == 0 and lb == 0
    days = sorted({r['_day'] for r in E['recs']}); T = lambda arm, d, k: sum(v[k] for (dd, m, a), v in rc.items() if a == arm and (d is None or dd == d))
    per = lambda arm, d: T(arm, d, 'all_cut_mo1s') / T(arm, d, 'all_cut_valid_q'); pr = E['maudit']['predicates']; okr = True
    for name, (a, b) in CONTRASTS.items():
        p_ = pr['contrasts'][name]; dE = {d: T(a, d, 'value') - T(b, d, 'value') for d in days}; d1 = {d: T(a, d, 'all_id_mo1s') - T(b, d, 'all_id_mo1s') for d in days}
        dp = {d: per(a, d) - per(b, d) for d in days}; mE = {(dd, m): v['value'] - rc[(dd, m, b)]['value'] for (dd, m, aa), v in rc.items() if aa == a}; ab = sum(abs(v) for v in mE.values())
        top3 = sum(sorted((abs(v) for v in mE.values()), reverse=True)[:3]) / ab if ab > 0 else np.nan
        okr &= (p_['arms'] == [a, b] and p_['END']['answer'] == classify(dE.values()) and all(close(p_['END']['by_day'][d], dE[d]) for d in days) and close(p_['END']['total'], T(a, None, 'value') - T(b, None, 'value'))
                and p_['mo1s']['answer'] == classify(d1.values()) and all(close(p_['mo1s']['by_day'][d], d1[d]) for d in days) and p_['per_share']['answer'] == classify(dp.values())
                and all(close(p_['per_share']['by_day'][d], dp[d]) for d in days) and close(p_['per_share']['total'], per(a, None) - per(b, None))
                and p_['markets_better'] == sum(1 for v in mE.values() if v > 1e-9) and p_['markets_worse'] == sum(1 for v in mE.values() if v < -1e-9)
                and p_['markets_equal'] == sum(1 for v in mE.values() if abs(v) <= 1e-9) and close(p_['top3_abs_share'], top3))
    it = lambda k, d: T('FULL_11', d, k) - T('PULL_ONLY_10', d, k) - T('WITHHOLD_ONLY_01', d, k) + T('BASELINE_00', d, k)
    for k, sk in (('value', 'END'), ('all_id_mo1s', 'mo1s')):
        okr &= pr['interaction'][sk]['answer'] == classify(it(k, d) for d in days) and all(close(pr['interaction'][sk]['by_day'][d], it(k, d)) for d in days) and close(pr['interaction'][sk]['total'], it(k, None))
    okr &= pr['all_arms_negative']['total'] == all(T(a, None, 'value') < 0 for a in ARMS) and all(pr['all_arms_negative']['by_day'][d] == all(T(a, d, 'value') < 0 for a in ARMS) for d in days)
    c['F_readings'] = bool(okr)
    return c, dict(F_reconcile_bad=rb, F_identity_bad=ib + lb, F_totals={a: T(a, None, 'value') for a in ARMS})


RE_FIELDS = ('side', 'pull_t', 'pull_price', 'pull_rung', 'pull_venue_ahead', 'pull_venue_rem', 'next_gid', 'next_submit', 'next_act', 'next_price', 'next_rung', 'next_activation_ahead',
             'next_refused', 'gap_ms', 'same_price', 'same_rung', 'ahead_difference')


def reentry_expect(L_):
    """This checker's own next-same-side rows from the ledger."""
    qv = L_['queue_venue']; od = {int(o['gid']): o for o in L_['orders']}; out = {}
    for g, v in qv['pull'].items():
        o = od[int(g)]; row = dict(side=int(o['side']), pull_t=float(v[0]), pull_price=float(o['price']), pull_rung=float(o['rung']), pull_venue_ahead=float(v[1]), pull_venue_rem=float(v[2]))
        later = [x for x in L_['orders'] if int(x['side']) == int(o['side']) and float(x['submit']) > float(v[0])]
        if later:
            n_ = min(later, key=lambda x: (float(x['submit']), int(x['gid']))); a2 = float(qv['ahead0'][str(int(n_['gid']))][1])
            row.update(next_gid=int(n_['gid']), next_submit=float(n_['submit']), next_act=float(n_['act']), next_price=float(n_['price']), next_rung=float(n_['rung']), next_activation_ahead=a2,
                       next_refused=bool(n_['refused']), gap_ms=float(n_['submit']) - float(v[0]), same_price=float(n_['price']) == float(o['price']), same_rung=float(n_['rung']) == float(o['rung']),
                       ahead_difference=a2 - float(v[1]))
        else:
            row.update(next_gid=None, next_submit=np.nan, next_act=np.nan, next_price=np.nan, next_rung=np.nan, next_activation_ahead=np.nan, next_refused=None, gap_ms=np.nan,
                       same_price=None, same_rung=None, ahead_difference=np.nan)
        out[int(g)] = row
    return out


def same(a, b):
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return b is None or (isinstance(b, float) and np.isnan(b)) or (isinstance(b, str) and b == '')
    if isinstance(a, bool):
        return str(b) in (str(a), str(float(a)))
    return close(a, b)


def fam_H(E):
    bad = 0; RP = E['reentry']
    for r in E['recs']:
        for arm in RUN_ARMS:
            L_ = r[arm]; qv = L_['queue_venue']; cm = cmds(E, r, arm)
            bad += sum(1 for o in L_['orders'] if not o['refused'] and str(o['gid']) not in qv['ahead0'])
            bad += sum(1 for z in L_['guard_log'] if z[10] == 'pressure_pull' and str(int(z[3])) not in qv['pull'])
            rev = {int(c_[3]) for c_ in cm if int(c_[2]) == M.CMD_KINDS.index('live') and int(c_[3]) >= 0 and P.REASONS[int(c_[8])] != 'fallback'}
            bad += sum(1 for g in rev if str(g) not in qv['first'])
            ex = reentry_expect(L_); rows = RP[(RP.day == r['_day']) & (RP.m == int(r['m'])) & (RP.arm == arm)]
            bad += int(len(rows) != len(ex) or sorted(rows.pull_gid.astype(int)) != sorted(ex))
            for row in rows.to_dict('records'):
                e = ex.get(int(row['pull_gid']))
                bad += int(e is None or not all(same(e[f], row[f]) for f in RE_FIELDS))
    return dict(H_queue=bad == 0), dict(H_queue_bad=bad)


def logs_cpu(paths):
    cpu, peak = 0., 0
    for p in paths:
        for ln in Path(p).read_text().splitlines():
            m_ = RL.match(ln)
            if m_:
                cpu += float(m_.group(6)); peak = max(peak, int(m_.group(4)))
    return cpu / 3600., peak


def fam_I(E, root=None):
    own, peak = logs_cpu(glob.glob(str((root or R) / 'logs' / '**' / '*.log'), recursive=True))
    line, _ = logs_cpu([p for n in LINE for p in glob.glob(str(REPO / 'predictfun/reviews' / n / 'logs' / '**' / '*.log'), recursive=True)])
    rev, _ = logs_cpu([p for n in REVIEWS for p in glob.glob(str(REPO / 'predictfun/reviews/automatic' / n / '*.log'))])
    return dict(I_resources=own <= CAP and peak < 1536 and own + line + rev <= LINE_CAP), dict(I=dict(round_core_hours=own, line_core_hours=own + line + rev, peak_mib=peak))


FAMS = dict(A=fam_A, B=fam_B, C=fam_C, D=fam_D, E=fam_E, F=fam_F, H=fam_H, I=fam_I)


def run_checks(E):
    chk, det = {}, {}
    for k in FAMS:
        c, d = FAMS[k](E); chk.update(c); det.update(d)
    return dict(checks=chk, details=det, all_ok=bool(all(chk.values())))


# ---------------------------------------------------------------------------------------------------------------------- falsifier
def _rec(E, i, fn):
    E['recs'] = list(E['recs']); E['recs'][i] = copy.deepcopy(E['recs'][i]); fn(E['recs'][i])


def _first(E, arm, pred):
    return next(i for i, r in enumerate(E['recs']) if pred(r[arm]))


def _coordinated(E):
    A_ = E['arms'].copy(); i = A_.index[A_.arm == 'FULL_11'][0]; A_.loc[i, 'all_id_mo1s'] += 1000.; E['arms'] = A_; pr = copy.deepcopy(E['maudit']['predicates'])
    d = str(A_.loc[i, 'day']); pr['contrasts']['cancel_given_withhold']['mo1s']['by_day'][d] += 1000.; pr['contrasts']['cancel_given_withhold']['mo1s']['total'] += 1000.
    E['maudit'] = dict(E['maudit'], predicates=pr)


def _post_at_on(E):
    i = _first(E, 'WITHHOLD_ONLY_01', lambda L_: any(z[10] == 'pressure_withhold' for z in L_['guard_log'])); r = E['recs'][i]; key = (r['_day'], r['_part']); nm = f"{int(r['m'])}__WITHHOLD_ONLY_01__commands"
    cm = E['arr'][key][nm].copy(); j = int(np.nonzero(cm[:, 8] == P.REASONS.index('pressure_withhold'))[0][0]); cm[j, 7] = M.CMD_ACTIONS.index('POST'); cm[j, 8] = P.REASONS.index('post')
    E['arr'] = dict(E['arr']); E['arr'][key] = dict(E['arr'][key]); E['arr'][key][nm] = cm


def _fake_logs(E):
    E['_root'] = Path(tempfile.mkdtemp(dir=str(D))); (E['_root'] / 'logs').mkdir()
    (E['_root'] / 'logs' / 'fake.log').write_text('2026-11-21T00:00:00Z tag=fake rc=0 peak_rss_mib=2000 wall_s=3000.0 cpu_user_s=3000.0\n')


CASES = [
    ('coordinated_markout_and_contrast', 'F', 'F_reconcile', _coordinated),
    ('a_reentry_price_changed', 'H', 'H_queue', lambda E: E.__setitem__('reentry', E['reentry'].assign(next_price=E['reentry'].next_price + np.where(E['reentry'].index == E['reentry'].next_price.first_valid_index(), 1., 0.)))),
    ('a_pull_only_withhold', 'D', 'D_rule', lambda E: _rec(E, _first(E, 'PULL_ONLY_10', lambda L_: L_['guard_log']), lambda r: r['PULL_ONLY_10']['guard_log'][0].__setitem__(10, 'pressure_withhold'))),
    ('a_withhold_only_post_at_an_on_tick', 'D', 'D_rule', _post_at_on),
    ('full_fill_dropped', 'C', 'C_full_equals_parent', lambda E: _rec(E, 0, lambda r: r['FULL_11'].__setitem__('fills', r['FULL_11']['fills'][:-1]))),
    ('cash_changed', 'E', 'E_accounting', lambda E: _rec(E, 0, lambda r: r['PULL_ONLY_10'].__setitem__('cash', r['PULL_ONLY_10']['cash'] + 1.))),
    ('lost_end_changed', 'F', 'F_identity', lambda E: E.__setitem__('lost_added', E['lost_added'].assign(lost_end=E['lost_added'].lost_end + np.where(E['lost_added'].index == 0, 1., 0.)))),
    ('stored_interaction_answer_changed', 'F', 'F_readings', lambda E: E.__setitem__('maudit', dict(E['maudit'], predicates=dict(E['maudit']['predicates'], interaction=dict(
        E['maudit']['predicates']['interaction'], END=dict(E['maudit']['predicates']['interaction']['END'], answer='POSITIVE_EVERY_DAY' if E['maudit']['predicates']['interaction']['END']['answer'] != 'POSITIVE_EVERY_DAY' else 'MIXED')))))),
    ('a_market_dropped', 'B', 'B_membership', lambda E: E.__setitem__('recs', E['recs'][:-1])),
    ('a_record_rule_changed', 'A', 'A_rule', lambda E: E.__setitem__('prec', dict(E['prec'], **{next(iter(E['prec'])): dict(E['prec'][next(iter(E['prec']))], rule_sha256='0' * 64)}))),
    ('resources_over_the_cap', 'I', 'I_resources', _fake_logs),
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
        out[name] = dict(family=fam, target=target, clean=bool(base['checks'][target]), refused=not c[target], flipped=sorted(k for k, v in c.items() if base['checks'].get(k) and not v))
    return dict(cases=out, falsifier_ok=bool(base['all_ok'] and all(v['clean'] and v['refused'] for v in out.values())))


def main():
    global D, PARTS_
    rehearse = len(sys.argv) > 2 and sys.argv[1] == 'rehearse'
    if rehearse:
        D, PARTS_ = Path(sys.argv[2]), ('pilot',)
    E = load(); res = run_checks(E); dest = D if rehearse else R; tag = '_rehearsal' if rehearse else ''
    (dest / f'checks{tag}.json').write_text(json.dumps(res, indent=1, default=str) + '\n'); fz = falsify(E, res)
    (dest / f'checks_falsifier{tag}.json').write_text(json.dumps(fz, indent=1) + '\n')
    if rehearse:
        print(json.dumps(dict(checks=res['checks'], cases={k: (v['clean'], v['refused']) for k, v in fz['cases'].items()}), indent=1)); return
    print(json.dumps(dict(all_ok=res['all_ok'], failed=[k for k, v in res['checks'].items() if not v], falsifier_ok=fz['falsifier_ok'],
                          cases={k: (v['refused'], v['flipped']) for k, v in fz['cases'].items()}, details=res['details']), indent=1, default=str)[:5000])


if __name__ == '__main__':
    main()
