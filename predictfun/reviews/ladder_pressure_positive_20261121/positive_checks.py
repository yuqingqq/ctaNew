"""INDEPENDENT CHECKS of POSITIVE-PRESSURE-GUARD-v1 (declaration.md; session aea5bfb9, 2026-11-21). Recomputed from the stored evidence (this round's
part records and arrays, the pressure receipt's records and arrays, the accepted baseline's records, the audit's states, the frozen edges). The
positive module is imported for rule_sha() only; pressure_checks.py and lag_checks_v2.py for their pure helpers.
 A identity: frozen files committed; the rule sha256 = the manifest's = every part record's; the parent rule and day records; edges; sources;
   outputs;  B membership: the parts cover the 19 complete markets once each with both run arms;  C controls: Q5_CONTROL = the parent's
   pressure arm (fingerprint, fills, orders, requests, value, log, command array, queue); the parent's disabled arm = the accepted baseline;
 D the challenger's rule from the parent cache: an intervention only at an ok tick with float32 x > 0 (the cached bits) and a pressure reason; no
   non-fallback KEEP and no POST (or cap / near-END WAIT after a POST) at such a tick; a pressure reason only at such a tick; no gid requested
   twice;  E accounting: each run arm's cash, position and value from its fills;
 F reconcile: per arm (three) every arms.csv field a reading uses (markouts, END identity terms, rung, interventions, exact uptime, exact inventory,
   the queue fields); the END identity; lost / added (POSITIVE vs Q5_CONTROL); the readings recomputed (END / eligible +1 s / per share by day
   and their classes, economic, stop_candidate, all_arms_negative, market counts); the screen recomputed (this checker's own join, population and
   mask from the parent's disabled-arm commands, the frozen null of the manifest);
 I resources: this round <= 0.14 core-hours; the ladder line (its seven earlier receipts and the review logs) <= 2.0.
falsify(): one corruption per family must fail while the clean evidence passes. Usage: positive_checks.py | positive_checks.py rehearse (the Oct
24 pilot: code paths; B_membership false by design)."""
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

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_pressure_positive_20261121'; D = REPO / 'data/predict_fun/research/ladder_pressure_positive_v1_20261121'
PD_ = REPO / 'data/predict_fun/research/ladder_pressure_v1_20261121'; PR = REPO / 'predictfun/reviews/ladder_pressure_20261121'
RUN = REPO / 'data/predict_fun/research/ladder_rest_v1_20261121'; AUD = REPO / 'data/predict_fun/research/ladder_audit_v1_20261121'; EDGE_FILE = RUN / 'measure' / 'audit.json'
COMMITTED = ['predictfun/research/pf_ladder_pressure_positive.py', 'predictfun/research/pf_ladder_pressure_positive_selftest.py', 'predictfun/research/pf_ladder_pressure_positive_mutants.py',
             'predictfun/reviews/ladder_pressure_positive_20261121/declaration.md', 'predictfun/reviews/ladder_pressure_positive_20261121/manifest.json',
             'predictfun/reviews/ladder_pressure_positive_20261121/launch_positive.sh', 'predictfun/reviews/ladder_pressure_positive_20261121/positive_checks.py',
             'predictfun/reviews/ladder_pressure_positive_20261121/tools/pilot_projection_positive.py']
LINE = ['ladder_rest_20261121', 'ladder_audit_20261121', 'ladder_guard_20261121', 'ladder_pressure_20261121', 'ladder_pressure_lag_20261121', 'ladder_pressure_ablation_20261121',
        'ladder_pressure_residual_20261121']
REVIEWS = ['20261121_ladder_rest_review', '20261121_ladder_audit_review', '20261121_ladder_guard_review', '20261121_ladder_pressure_review', '20261121_ladder_pressure_lag_review',
           '20261121_ladder_pressure_ablation_review', '20261121_ladder_pressure_residual_review']
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
STEP, SG, CAP, LINE_CAP, MASK_ON = 50., {0: 1., 1: -1.}, .14, 2., 10.
ARMS = ('BASELINE', 'Q5_CONTROL', 'POSITIVE'); RUN_ARMS = ('Q5_CONTROL', 'POSITIVE')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


PC = _load('pressure_checks', PR / 'pressure_checks.py'); LC2 = _load('lag_checks_v2', REPO / 'predictfun/reviews/ladder_pressure_lag_20261121/lag_checks_v2.py')
sys.path.insert(0, str(REPO / 'predictfun/research'))
import pf_ladder_pressure_positive as PP                  # noqa: E402  rule_sha only
P, M = PP.P, PP.M
bits, close, on, sha, git_blob, gz = PC.bits, PC.close, PC.on, PC.sha, PC.git_blob, PC.gz
PARTS_ = ('pilot', 'rest')


def load():
    gm = json.loads((R / 'manifest.json').read_text()); E = dict(gm=gm, prec={}, recs=[], parent={}, parr={}, base={}, arr={})
    for day in gm['days']:
        E['parent'].update({(day, int(r['m'])): r for r in gz(PD_ / 'days' / f'{day}.jsonl.gz')}); E['base'].update({(day, int(r['m'])): r for r in gz(RUN / 'days' / f'{day}.jsonl.gz')})
        z = np.load(PD_ / 'days' / f'{day}_arrays.npz'); E['parr'][day] = {k: z[k] for k in z.files}
        for part in PARTS_:
            if not gm['parts'][part][day]:
                continue
            rec = json.loads((D / part / f'{day}.json').read_text()); E['prec'][(day, part)] = rec
            E['recs'] += [dict(r, _day=day, _part=part) for r in gz(D / part / f'{day}.jsonl.gz')]; z = np.load(D / part / f'{day}_arrays.npz'); E['arr'][(day, part)] = {k: z[k] for k in z.files}
    for n in ('arms', 'lost_added', 'control'):
        E[n] = pd.read_csv(D / 'measure' / f'{n}.csv', dtype={'day': str}, float_precision='round_trip')
    E['maudit'] = json.loads((D / 'measure' / 'audit.json').read_text()); E['screen'] = json.loads((D / 'screen.json').read_text())
    E['states'] = pd.read_csv(AUD / 'states.csv', dtype={'day': str}, float_precision='round_trip'); E['edges_raw'] = json.loads(EDGE_FILE.read_text())['edges']
    return E


def cache(E, day, m):
    a = E['parr'][day]; return {k: {s: a[f'{m}__cache__{k}{s}'] for s in (0, 1)} for k in ('x', 'age', 'status')}


def ledger(E, r, arm):
    return E['parent'][(r['_day'], int(r['m']))]['disabled'] if arm == 'BASELINE' else r[arm]


def cmds(E, r, arm):
    return E['parr'][r['_day']][f"{int(r['m'])}__disabled__commands"] if arm == 'BASELINE' else E['arr'][(r['_day'], r['_part'])][f"{int(r['m'])}__{arm}__commands"]


def fam_A(E):
    gm = E['gm']; pm = json.loads((PR / 'manifest.json').read_text()); recs = list(E['prec'].values())
    c = dict(A_committed=all(git_blob(p) == (REPO / p).read_bytes() for p in COMMITTED), A_rule=gm['rule_sha256'] == PP.rule_sha() and all(r['rule_sha256'] == gm['rule_sha256'] for r in recs))
    c['A_parent'] = gm['parent']['rule_sha256'] == pm['rule_sha256'] == P.rule_sha() and all(sha(PD_ / 'days' / f'{d}.json') == gm['parent']['day_records'][d] and git_blob(
        f'predictfun/reviews/ladder_pressure_20261121/outputs/days/{d}.json') == (PD_ / 'days' / f'{d}.json').read_bytes() for d in gm['days'])
    c['A_edges'] = sha(EDGE_FILE) == gm['edges_sha256'] and all(gm['edges'][str(s)] == [float(v) for v in E['edges_raw'][f'live|{s}|adv_bn_r250']] for s in (0, 1))
    c['A_sources'] = all(r['sources'] == gm['sources'] for r in recs)
    c['A_outputs'] = all(sha(D / p / f'{d}.jsonl.gz') == r['sha256']['gz'] and sha(D / p / f'{d}_arrays.npz') == r['sha256']['npz'] for (d, p), r in E['prec'].items()) and all(
        E['maudit']['outputs'][n] == sha(D / 'measure' / n) for n in ('arms.csv', 'control.csv', 'lost_added.csv'))
    return c, {}


def fam_B(E):
    gm = E['gm']; comp = {d: sorted(int(m) for (dd, m), r in E['base'].items() if dd == d and r['status'] == 'complete') for d in gm['days']}
    got = collections.Counter((r['_day'], int(r['m'])) for r in E['recs'])
    ok_ = (all(sorted(gm['parts']['pilot'][d] + gm['parts']['rest'][d]) == comp[d] for d in gm['days']) and sorted(got) == sorted((d, m) for d, v in comp.items() for m in v)
           and all(v == 1 for v in got.values()) and all(all(a in r for a in RUN_ARMS) for r in E['recs']))
    return dict(B_membership=bool(ok_)), {}


def fam_C(E):
    bad = bb = 0
    for r in E['recs']:
        day, m = r['_day'], int(r['m']); pr_ = E['parent'][(day, m)]; pp, z = pr_['pressure'], r['Q5_CONTROL']
        bad += int(not (z['fingerprint'] == pp['fingerprint'] and z['fills'] == pp['fills'] and z['orders'] == pp['orders'] and z['requests'] == pp['requests'] and z['value'] == pp['value']
                        and z['guard_log'] == pp['guard_log'] and z['queue_venue'] == pp['queue_venue'] and z['queue_known'] == pp['queue_known'] and r['control_all']
                        and np.array_equal(E['arr'][(day, r['_part'])][f'{m}__Q5_CONTROL__commands'], E['parr'][day][f'{m}__pressure__commands'], equal_nan=True)))
        b, d_ = E['base'][(day, m)]['ledger'], pr_['disabled']
        bb += int(not (d_['fills'] == b['fills'] and d_['value'] == b['value'] and d_['fingerprint'] == b['fingerprint']))
    return dict(C_q5_equals_parent=bad == 0, C_baseline_equals_original=bb == 0), dict(C_q5_mismatch=bad)


def fam_D(E):
    KI, AI, RS = M.CMD_KINDS, M.CMD_ACTIONS, P.REASONS; bad = collections.Counter()
    for r in E['recs']:
        c = cache(E, r['_day'], int(r['m'])); ON = {s: (c['status'][s] == 0) & np.isfinite(c['x'][s]) & (c['x'][s] > 0) for s in (0, 1)}
        for z in r['POSITIVE']['guard_log']:
            k, s = int(round(z[0] / STEP)), int(z[1])
            bad['log'] += int(not (ON[s][k] and bits(z[5]) == bits(c['x'][s][k]) and z[10] in ('pressure_pull', 'pressure_withhold')))
        for c_ in cmds(E, r, 'POSITIVE'):
            k, s, kind, act, why = int(round(c_[0] / STEP)), int(c_[1]), KI[int(c_[2])], AI[int(c_[7])], RS[int(c_[8])]; onk = bool(ON[s][k])
            bad['keep_post'] += int(onk and ((kind == 'live' and act == 'KEEP' and why != 'fallback') or (kind == 'slot' and (act == 'POST' or (act == 'WAIT' and why in ('cap', 'near_end'))))))
            bad['reason'] += int(why in ('pressure_pull', 'pressure_withhold') and not onk)
        rq = collections.Counter(e[2] for e in r['POSITIVE']['requests']); bad['repeat'] += sum(1 for v in rq.values() if v > 1)
    return dict(D_rule=sum(bad.values()) == 0), dict(D_bad={k: v for k, v in bad.items() if v})


def fam_E(E):
    bad = 0
    for r in E['recs']:
        for arm in RUN_ARMS:
            L_ = r[arm]; cash = -sum(SG[int(f[2])] * float(f[3]) * float(f[4]) for f in L_['fills']); pos = sum(SG[int(f[2])] * float(f[3]) for f in L_['fills'])
            bad += int(not (close(cash, L_['cash']) and close(pos, L_['position']) and close(cash + pos * L_['mark'], L_['value'])))
    return dict(E_accounting=bad == 0), {}


def my_screen(E):
    """This checker's own join, population and mask; then the frozen null of the manifest on the encoded population."""
    S = E['states'][E['states'].kind == 'live'].copy(); idx = {}
    for (d, m), r in E['parent'].items():
        for c_ in E['parr'][d][f'{m}__disabled__commands']:
            if int(c_[2]) == M.CMD_KINDS.index('live') and int(c_[3]) >= 0:
                idx[(d, m, int(c_[3]), float(c_[0]))] = (int(c_[7]), int(c_[8]))
    keep = [idx.get((d, int(m), int(g), float(t))) == (M.CMD_ACTIONS.index('KEEP'), P.REASONS.index('keep')) for d, m, g, t in zip(S.day, S.m, S.gid, S.t)]
    last = {s: float(E['edges_raw'][f'live|{s}|adv_bn_r250'][-1]) for s in (0, 1)}; x = S.adv_bn_r250.astype(float).values
    pop = np.array(keep) & np.isfinite(x) & np.array([not on(float(np.float32(v)), last[int(s)]) if np.isfinite(v) else False for v, s in zip(x, S.side)])
    H = S[pop].copy(); H['adv_bn_r250'] = np.where(H.adv_bn_r250.astype(float).values > 0., MASK_ON, 0.)
    return PC.selection_recompute(H, E['gm'], E['edges_raw']), dict(q5_off=int(pop.sum()), mask_on=int((H.adv_bn_r250 > 0).sum()))


def fam_F(E):  # noqa: C901
    A_ = E['arms']; rb = ib = 0; rc = {}
    for r in E['recs']:
        end = float(r['path_info']['end_rel'])
        for arm in ARMS:
            L_ = ledger(E, r, arm); x = collections.defaultdict(float, PC.recompute(L_, end)); x.update(LC2.uptime(L_)); x.update(LC2.inventory(L_)); q_, _ = LC2.queue(L_, cmds(E, r, arm)); x.update(q_)
            rc[(r['_day'], int(r['m']), arm)] = x; row = A_[(A_.day == r['_day']) & (A_.m == int(r['m'])) & (A_.arm == arm)]
            rb += int(len(row) != 1 or not all(close(row[k].iloc[0], x[k]) for k in list(PC.RECON) + list(LC2.UP) + list(LC2.INV) + list(LC2.QF)))
            ib += int(not (close(x['all_id_mo1s'] + x['all_id_move'] + x['all_id_late'], x['all_end']) and close(x['all_end'], x['value'])))
    c = dict(F_reconcile=rb == 0); LA = E['lost_added']; lb = 0
    for r in E['recs']:
        fb, fg = r['Q5_CONTROL'], r['POSITIVE']; lo = [f for f, u in zip(fb['fills'], PC.matched_split(fb['fills'], fg['fills'])) if u]
        ad = [f for f, u in zip(fg['fills'], PC.matched_split(fg['fills'], fb['fills'])) if u]; en = lambda F_, mk: sum(SG[int(f[2])] * float(f[3]) * (mk - float(f[4])) for f in F_)
        row = LA[(LA.day == r['_day']) & (LA.m == int(r['m']))]
        lb += int(len(row) != 1 or not (int(row.lost_n.iloc[0]) == len(lo) and int(row.added_n.iloc[0]) == len(ad) and close(row.lost_end.iloc[0], en(lo, float(fb['mark'])))
                                         and close(row.added_end.iloc[0], en(ad, float(fg['mark']))) and close(float(fg['value']) - float(fb['value']), en(ad, float(fg['mark'])) - en(lo, float(fb['mark'])))))
    c['F_identity'] = ib == 0 and lb == 0
    sr, cnt = my_screen(E); sc = E['screen']
    c['F_screen'] = bool(sr['answer'] == sc['answer'] and sr['selected'] == sc['selected'] and sr['states'] == sc['states'] and close(sr['J'], sc['J']) and close(sr['null_p05'], sc['null_p05'])
                         and sr['movable_strata'] == sc['movable_strata'] and cnt['q5_off'] == sc['population']['q5_off'] and cnt['mask_on'] == sc['population']['mask_on'])
    days = sorted({r['_day'] for r in E['recs']}); T = lambda arm, d, k: sum(v[k] for (dd, m, a), v in rc.items() if a == arm and (d is None or dd == d))
    cl = lambda v: 'POSITIVE_EVERY_DAY' if all(x > 0 for x in v) else 'NEGATIVE_EVERY_DAY' if all(x < 0 for x in v) else 'MIXED'; pr = E['maudit']['predicates']
    dE = {d: T('POSITIVE', d, 'value') - T('Q5_CONTROL', d, 'value') for d in days}; d1 = {d: T('POSITIVE', d, 'all_id_mo1s') - T('Q5_CONTROL', d, 'all_id_mo1s') for d in days}
    per = lambda arm, d: T(arm, d, 'all_cut_mo1s') / T(arm, d, 'all_cut_valid_q'); dp = {d: per('POSITIVE', d) - per('Q5_CONTROL', d) for d in days}
    mE = [v['value'] - rc[(dd, m, 'Q5_CONTROL')]['value'] for (dd, m, a), v in rc.items() if a == 'POSITIVE']
    eA, mA = cl(dE.values()), cl(d1.values()); econ = 'IMPROVES' if eA == mA == 'POSITIVE_EVERY_DAY' else 'WORSENS' if eA == mA == 'NEGATIVE_EVERY_DAY' else 'MIXED'
    c['F_readings'] = bool(pr['END']['answer'] == eA and all(close(pr['END']['by_day'][d], dE[d]) for d in days) and pr['mo1s']['answer'] == mA and all(close(pr['mo1s']['by_day'][d], d1[d]) for d in days)
                           and all(close(pr['per_share']['by_day'][d], dp[d]) for d in days) and pr['economic'] == econ and pr['screen'] == sr['answer']
                           and pr['stop_candidate'] == (not (econ == 'IMPROVES' and sr['answer'] == 'SELECTION_BEYOND_EXPOSURE'))
                           and pr['all_arms_negative']['total'] == all(T(a, None, 'value') < 0 for a in ARMS) and pr['END']['markets_better'] == sum(1 for v in mE if v > 1e-9)
                           and pr['END']['markets_worse'] == sum(1 for v in mE if v < -1e-9))
    return c, dict(F_reconcile_bad=rb, F_identity_bad=ib + lb, F_economic=econ, F_screen=sr['answer'], F_totals={a: T(a, None, 'value') for a in ARMS})


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


FAMS = dict(A=fam_A, B=fam_B, C=fam_C, D=fam_D, E=fam_E, F=fam_F, I=fam_I)


def run_checks(E):
    chk, det = {}, {}
    for k in FAMS:
        c, d = FAMS[k](E); chk.update(c); det.update(d)
    return dict(checks=chk, details=det, all_ok=bool(all(chk.values())))


def _rec(E, i, fn):
    E['recs'] = list(E['recs']); E['recs'][i] = copy.deepcopy(E['recs'][i]); fn(E['recs'][i])


def _keep_at_on(E):
    i = next(j for j, r in enumerate(E['recs']) if any(z[10] == 'pressure_pull' for z in r['POSITIVE']['guard_log'])); r = E['recs'][i]; key = (r['_day'], r['_part'])
    nm = f"{int(r['m'])}__POSITIVE__commands"; cm = E['arr'][key][nm].copy(); j = int(np.nonzero(cm[:, 8] == P.REASONS.index('pressure_pull'))[0][0])
    cm[j, 7] = M.CMD_ACTIONS.index('KEEP'); cm[j, 8] = P.REASONS.index('keep'); E['arr'] = dict(E['arr']); E['arr'][key] = dict(E['arr'][key]); E['arr'][key][nm] = cm


def _rule(E):
    k = next(iter(E['prec'])); E['prec'] = dict(E['prec']); E['prec'][k] = dict(E['prec'][k], rule_sha256='0' * 64)


def _fake(E):
    E['_root'] = Path(tempfile.mkdtemp(dir=str(D))); (E['_root'] / 'logs').mkdir()
    (E['_root'] / 'logs' / 'fake.log').write_text('2026-11-21T00:00:00Z tag=fake rc=0 peak_rss_mib=2000 wall_s=900.0 cpu_user_s=900.0\n')


CASES = [
    ('a_markout_changed', 'F', 'F_reconcile', lambda E: (E.__setitem__('arms', E['arms'].assign(all_id_mo1s=E['arms'].all_id_mo1s + np.where(E['arms'].index == E['arms'].index[E['arms'].arm == 'POSITIVE'][0], 500., 0.))))),
    ('a_keep_at_an_on_tick', 'D', 'D_rule', _keep_at_on),
    ('q5_fill_dropped', 'C', 'C_q5_equals_parent', lambda E: _rec(E, 0, lambda r: r['Q5_CONTROL'].__setitem__('fills', r['Q5_CONTROL']['fills'][:-1]))),
    ('cash_changed', 'E', 'E_accounting', lambda E: _rec(E, 0, lambda r: r['POSITIVE'].__setitem__('cash', r['POSITIVE']['cash'] + 1.))),
    ('lost_end_changed', 'F', 'F_identity', lambda E: E.__setitem__('lost_added', E['lost_added'].assign(lost_end=E['lost_added'].lost_end + np.where(E['lost_added'].index == 0, 1., 0.)))),
    ('stored_screen_J_changed', 'F', 'F_screen', lambda E: E.__setitem__('screen', dict(E['screen'], J=E['screen']['J'] + 1.))),
    ('stored_economic_changed', 'F', 'F_readings', lambda E: E.__setitem__('maudit', dict(E['maudit'], predicates=dict(E['maudit']['predicates'], economic='IMPROVES' if E['maudit']['predicates']['economic'] != 'IMPROVES' else 'MIXED')))),
    ('a_market_dropped', 'B', 'B_membership', lambda E: E.__setitem__('recs', E['recs'][:-1])),
    ('a_record_rule_changed', 'A', 'A_rule', _rule),
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
    global D, PARTS_
    rehearse = len(sys.argv) > 1 and sys.argv[1] == 'rehearse'
    if rehearse:
        D, PARTS_ = D / 'rehearsal', ('pilot',)
    E = load(); res = run_checks(E); dest = D if rehearse else R; tag = '_rehearsal' if rehearse else ''
    (dest / f'checks{tag}.json').write_text(json.dumps(res, indent=1, default=str) + '\n'); fz = falsify(E, res)
    (dest / f'checks_falsifier{tag}.json').write_text(json.dumps(fz, indent=1) + '\n')
    print(json.dumps(dict(all_ok=res['all_ok'], failed=[k for k, v in res['checks'].items() if not v], falsifier_ok=fz['falsifier_ok'], cases={k: (v['clean'], v['refused']) for k, v in fz['cases'].items()},
                          details=res['details']), indent=1, default=str)[:3500])


if __name__ == '__main__':
    main()
