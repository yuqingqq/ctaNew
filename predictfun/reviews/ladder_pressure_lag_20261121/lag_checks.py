"""INDEPENDENT CHECKS of PRESSURE-LEAD-SENSITIVITY-v1 (declaration.md; session aea5bfb9, 2026-11-21). Recomputed from the stored evidence: this round's
day records and arrays, the pressure receipt's (parent) records and arrays, the accepted baseline's records, the audit's states and the frozen
edge file. The lag module is imported for rule_sha() only; pressure_checks.py (frozen, 793c706b) for its pure helpers (recompute, matched_split,
selection_recompute, bits, close, on); no measurement code of the lag or pressure module is used.
 A identity: frozen files committed; the rule sha256 = the manifest's = every day record's; the parent rule = the parent manifest's = the
   frozen pressure module's; the parent day records = the manifest's sha256 = the committed receipt copies; the edges; sources; outputs;
 B membership: lag0 and lag1 for each of the 19 complete markets; the pilot = the first two complete markets of each date;
 C controls: lag0 = the parent's pressure arm (fingerprint, fills, orders, requests, value, the parent's log fields, the command array, the
   queue records); the parent's disabled arm = the accepted baseline (fills, orders, requests, value, fingerprint);
 D the lag rule from the parent cache arrays: every lag1 intervention at tick k >= 1 with the cached bits at k - 1 (status ok, ON by this
   checker's q5 test), source time t - 50, effective age = the cached age at k - 1 + 50, a pressure reason; no lag1 KEEP (but a fallback) or
   POST (or a cap / near-END WAIT after a POST) at a lagged-ON tick of its side; every lag1 pressure PULL / WAIT at a lagged-ON tick; no gid
   requested twice;
 E accounting: each arm's cash, position and value from its fills;
 F reconcile: per arm (baseline, lag0, lag1) every arms.csv field a reading uses - the markout / END / identity / rung / intervention fields,
   exact uptime from each order's activation, removal and full fill (review 909c671e finding 1), the exact inventory integrals from the fills,
   the queue fields and the next-same-side pair summaries; the END identity; lost / added for both pairs; reentry.csv; the readings (lag1 and
   lag0 vs baseline: END, per share, participation including exact active time, inventory; the retained gain; lag1 - lag0; the reading and
   its failed components) with this checker's own selection answer; the lagged selection screen recomputed (the join from the parent arrays,
   the manifest's null settings) with its answer derived here;
 G the join: every live state's zero-lag cached value equals its saved value bit for bit; the warm-up / non-ok counts;
 H queue: the activation-clock snapshot of every non-refused order, a pull entry for every pressure pull, a first-review entry for every
   reviewed gid, one reentry.csv row per pull;
 I resources: this round <= 0.50 core-hours; peaks <= 1,536 MiB.
falsify(): each corruption of a copy must fail its family while the clean evidence passes; it includes the review's coordinated case (every
lag1 both-side uptime +25 pp with the stored reading moved with it) and an inventory case. Usage: lag_checks.py | lag_checks.py rehearse <dir>
(the pre-run rehearsal on the pilot copy: check booleans and falsifier outcomes only; B_membership fails there by design)."""
import collections
import copy
import glob
import gzip
import hashlib
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_pressure_lag_20261121'; D = REPO / 'data/predict_fun/research/ladder_pressure_lag_v1_20261121'
PD_ = REPO / 'data/predict_fun/research/ladder_pressure_v1_20261121'; PR = REPO / 'predictfun/reviews/ladder_pressure_20261121'
RUN = REPO / 'data/predict_fun/research/ladder_rest_v1_20261121'; AUD = REPO / 'data/predict_fun/research/ladder_audit_v1_20261121'
EDGE_FILE = RUN / 'measure' / 'audit.json'
COMMITTED = ['predictfun/research/pf_ladder_pressure_lag.py', 'predictfun/research/pf_ladder_pressure_lag_selftest.py', 'predictfun/research/pf_ladder_pressure_lag_mutants.py',
             'predictfun/reviews/ladder_pressure_lag_20261121/declaration.md', 'predictfun/reviews/ladder_pressure_lag_20261121/manifest.json',
             'predictfun/reviews/ladder_pressure_lag_20261121/launch_lag.sh', 'predictfun/reviews/ladder_pressure_lag_20261121/lag_checks.py',
             'predictfun/reviews/ladder_pressure_lag_20261121/tools/pilot_projection_lag.py']
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
STEP, SG, TOL, S_LIM, CAP = 50., {0: 1., 1: -1.}, 1e-9, 100., .50
spec = importlib.util.spec_from_file_location('pressure_checks', PR / 'pressure_checks.py'); PC = importlib.util.module_from_spec(spec); spec.loader.exec_module(PC)
sys.path.insert(0, str(REPO / 'predictfun/research'))
import pf_ladder_pressure_lag as LG                       # noqa: E402  rule_sha only
P, M = LG.P, LG.M                                         # encodings and the parent rule_sha only
bits, close, on, sha, git_blob, gz = PC.bits, PC.close, PC.on, PC.sha, PC.git_blob, PC.gz


def load():
    gm = json.loads((R / 'manifest.json').read_text()); E = dict(gm=gm, days={}, recs=[], parent={}, parr={}, base={}, arr={})
    for day in gm['days']:
        E['days'][day] = json.loads((D / 'days' / f'{day}.json').read_text()); E['recs'] += [dict(r, _day=day) for r in gz(D / 'days' / f'{day}.jsonl.gz')]
        E['parent'].update({(day, int(r['m'])): r for r in gz(PD_ / 'days' / f'{day}.jsonl.gz')}); E['base'].update({(day, int(r['m'])): r for r in gz(RUN / 'days' / f'{day}.jsonl.gz')})
        z = np.load(D / 'days' / f'{day}_arrays.npz'); E['arr'][day] = {k: z[k] for k in z.files}
        z = np.load(PD_ / 'days' / f'{day}_arrays.npz'); E['parr'][day] = {k: z[k] for k in z.files}
    E['pilot_days'] = {day: json.loads((D / 'pilot' / f'{day}.json').read_text()) for day in gm['days'] if (D / 'pilot' / f'{day}.json').exists()}
    for n in ('arms', 'lost_added', 'reentry', 'control'):
        E[n] = pd.read_csv(D / 'measure' / f'{n}.csv', dtype={'day': str}, float_precision='round_trip')
    E['maudit'] = json.loads((D / 'measure' / 'audit.json').read_text()); E['selection'] = json.loads((D / 'selection.json').read_text())
    E['parent_selection'] = json.loads((PD_ / 'selection.json').read_text())
    E['states'] = pd.read_csv(AUD / 'states.csv', dtype={'day': str}, float_precision='round_trip'); E['edges_raw'] = json.loads(EDGE_FILE.read_text())['edges']
    return E


def cache(E, day, m):
    a = E['parr'][day]; return {k: {s: a[f'{m}__cache__{k}{s}'] for s in (0, 1)} for k in ('x', 'age', 'status')}


def arm_ledger(E, r, arm):
    return E['parent'][(r['_day'], int(r['m']))]['disabled'] if arm == 'baseline' else r[arm]


def arm_cmds(E, r, arm):
    day, m = r['_day'], int(r['m'])
    return E['parr'][day][f'{m}__disabled__commands'] if arm == 'baseline' else E['arr'][day][f'{m}__{arm}__commands']


# ---------------------------------------------------------------------------------------------------------------------- independent pieces
def uptime(L_):
    """Exact active shares from each order's activation, removal and full fill (this checker's own interval code)."""
    end = float(L_['end']); fl = collections.defaultdict(list)
    for f in L_['fills']:
        fl[int(f[1])].append((float(f[0]), int(f[5]), float(f[3])))
    iv = {0: [], 1: []}
    for o in L_['orders']:
        if o['refused']:
            continue
        cum, full = 0., np.inf
        for t, _, q in sorted(fl.get(int(o['gid']), [])):
            cum += q
            if cum >= float(o['size']) - TOL:
                full = t; break
        a, b = float(o['act']), min(float(o['remove']), full, end)
        if b > a:
            iv[int(o['side'])].append((a, b))

    def merge(v):
        out = []
        for a, b in sorted(v):
            if out and a <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], b))
            else:
                out.append((a, b))
        return out
    u0, u1 = merge(iv[0]), merge(iv[1]); both = 0.
    for a, b in u0:
        for c, d in u1:
            both += max(0., min(b, d) - max(a, c))
    ln = lambda u: sum(b - a for a, b in u)
    return dict(bid_exact=ln(u0) / end, ask_exact=ln(u1) / end, both_exact=both / end)


def inventory(L_):
    end = float(L_['end']); ev = sorted((min(max(float(f[0]), 0.), end), SG[int(f[2])] * float(f[3])) for f in L_['fills']); I_, t0, area, high, peak = 0., 0., 0., 0., 0.
    for t, dq in ev + [(end, 0.)]:
        area += abs(I_) * (t - t0); high += (t - t0) * (abs(I_) >= S_LIM - 1e-9); t0 = t; I_ += dq; peak = max(peak, abs(I_))
    return dict(mean_abs_inv=area / end, peak_abs_inv=peak, share_abs_ge_S=high / end, terminal_inv=float(L_['position']))


def queue(L_, cm):
    qv, qk = L_['queue_venue'], L_['queue_known']; od = {int(o['gid']): o for o in L_['orders']}; mean = lambda z: float(np.mean(z)) if len(z) else np.nan
    pairs = []
    for g, v in qv['pull'].items():
        o = od[int(g)]; later = [x for x in L_['orders'] if int(x['side']) == int(o['side']) and float(x['submit']) > float(v[0])]
        if later:
            n_ = min(later, key=lambda x: (float(x['submit']), int(x['gid']))); a2 = float(qv['ahead0'][str(int(n_['gid']))][1])
            pairs.append(dict(pull_gid=int(g), next_gid=int(n_['gid']), same_price=float(n_['price']) == float(o['price']), same_rung=float(n_['rung']) == float(o['rung']), diff=a2 - float(v[1])))
        else:
            pairs.append(dict(pull_gid=int(g), next_gid=None))
    have = [p for p in pairs if p['next_gid'] is not None]; sp = [p for p in have if p['same_price']]
    return dict(q_activation_ahead_mean=mean([v[1] for v in qv['ahead0'].values()]), q_first_review_venue_ahead_mean=mean([v[1] for v in qv['first'].values()]),
                q_first_review_known_ahead_mean=mean([v['first'] for v in qk.values() if v['first'] is not None]), q_pulls=float(len(qv['pull'])),
                q_pull_venue_ahead_mean=mean([v[1] for v in qv['pull'].values()]), pairs_pulls=float(len(pairs)), pairs_with_next=float(len(have)), pairs_same_price=float(len(sp)),
                pairs_same_rung=float(sum(1 for p in have if p['same_rung'])), pairs_ahead_difference_mean=mean([p['diff'] for p in have]),
                pairs_same_price_ahead_difference_mean=mean([p['diff'] for p in sp])), pairs


# ---------------------------------------------------------------------------------------------------------------------- families
def fam_A(E):
    gm = E['gm']; pm = json.loads((PR / 'manifest.json').read_text()); c = {}
    c['A_committed'] = all(git_blob(p) == (REPO / p).read_bytes() for p in COMMITTED)
    c['A_rule'] = gm['rule_sha256'] == LG.rule_sha() and all(d['rule_sha256'] == gm['rule_sha256'] for d in list(E['days'].values()) + list(E['pilot_days'].values()))
    c['A_parent'] = (gm['parent']['rule_sha256'] == pm['rule_sha256'] == P.rule_sha() and all(
        sha(PD_ / 'days' / f'{d}.json') == gm['parent']['day_records'][d] == E['days'][d]['parent_day_record_sha256'] and git_blob(f'predictfun/reviews/ladder_pressure_20261121/outputs/days/{d}.json')
        == (PD_ / 'days' / f'{d}.json').read_bytes() for d in gm['days']))
    c['A_edges'] = sha(EDGE_FILE) == gm['edges_sha256'] and all(gm['edges'][str(s)] == [float(v) for v in E['edges_raw'][f'live|{s}|adv_bn_r250']] for s in (0, 1))
    c['A_sources'] = all(d['sources'] == gm['sources'] for d in list(E['days'].values()) + list(E['pilot_days'].values()))
    c['A_outputs'] = all(sha(D / 'days' / f'{day}.jsonl.gz') == d['sha256']['gz'] and sha(D / 'days' / f'{day}_arrays.npz') == d['sha256']['npz'] for day, d in E['days'].items()) and all(
        E['maudit']['outputs'][n] == sha(D / 'measure' / n) for n in ('arms.csv', 'control.csv', 'lost_added.csv', 'reentry.csv'))
    return c, {}


def fam_B(E):
    gm = E['gm']; comp = {day: sorted(int(m) for (d, m), r in E['base'].items() if d == day and r['status'] == 'complete') for day in gm['days']}
    got = sorted((r['_day'], int(r['m'])) for r in E['recs'])
    return dict(B_membership={d: sorted(int(m) for m in v) for d, v in gm['complete'].items()} == comp and got == sorted((d, m) for d, v in comp.items() for m in v)
                and all('lag0' in r and 'lag1' in r for r in E['recs']),
                B_pilot=all([int(m) for m in gm['pilot'][d]] == comp[d][:2] for d in gm['days']) and all(d_['members'] == [int(m) for m in gm['pilot'][day]] for day, d_ in E['pilot_days'].items())), {}


def fam_C(E):
    bad = bb = 0
    for r in E['recs']:
        day, m = r['_day'], int(r['m']); pr_ = E['parent'][(day, m)]; pp, z0 = pr_['pressure'], r['lag0']
        bad += int(not (z0['fingerprint'] == pp['fingerprint'] and z0['fills'] == pp['fills'] and z0['orders'] == pp['orders'] and z0['requests'] == pp['requests'] and z0['value'] == pp['value']
                        and [e[:11] for e in z0['guard_log']] == pp['guard_log'] and z0['queue_venue'] == pp['queue_venue'] and z0['queue_known'] == pp['queue_known']
                        and np.array_equal(E['arr'][day][f'{m}__lag0__commands'], E['parr'][day][f'{m}__pressure__commands'], equal_nan=True) and r['control_all']))
        b, d_ = E['base'][(day, m)]['ledger'], pr_['disabled']
        bb += int(not (d_['fills'] == b['fills'] and d_['orders'] == b['orders'] and d_['requests'] == b['requests'] and d_['value'] == b['value'] and d_['fingerprint'] == b['fingerprint']))
    return dict(C_zero_lag_equals_parent=bad == 0, C_baseline_equals_original=bb == 0), dict(C_zero_lag_mismatch=bad, C_baseline_mismatch=bb)


def fam_D(E):
    last = {s: float(E['gm']['edges'][str(s)][-1]) for s in (0, 1)}; KI, AI, RS = M.CMD_KINDS, M.CMD_ACTIONS, P.REASONS; lb = cb = rb = 0; n_on = 0
    for r in E['recs']:
        day, m = r['_day'], int(r['m']); c = cache(E, day, m); nt = len(c['x'][0])
        ON = {s: np.r_[False, np.array([c['status'][s][j] == 0 and on(float(c['x'][s][j]), last[s]) for j in range(nt - 1)], bool)] for s in (0, 1)}
        for z in r['lag1']['guard_log']:
            k, s = int(round(z[0] / STEP)), int(z[1]); j = k - 1
            lb += int(not (j >= 0 and c['status'][s][j] == 0 and bits(z[5]) == bits(c['x'][s][j]) and on(float(z[5]), last[s]) and z[10] in ('pressure_pull', 'pressure_withhold')
                           and z[11] == z[0] - STEP and close(z[12], float(c['age'][s][j]) + STEP, 1e-9) and close(z[6], float(c['age'][s][j]), 1e-9)))
        for c_ in arm_cmds(E, r, 'lag1'):
            k, s, kind, act, why = int(round(c_[0] / STEP)), int(c_[1]), KI[int(c_[2])], AI[int(c_[7])], RS[int(c_[8])]; onk = bool(ON[s][k]); n_on += int(onk)
            cb += int(onk and ((kind == 'live' and act == 'KEEP' and why != 'fallback') or (kind == 'slot' and (act == 'POST' or (act == 'WAIT' and why in ('cap', 'near_end'))))))
            cb += int(why in ('pressure_pull', 'pressure_withhold') and not onk)
        rq = collections.Counter(e[2] for e in r['lag1']['requests']); rb += sum(1 for v in rq.values() if v > 1)
    return dict(D_log=lb == 0, D_commands=cb == 0, D_requests=rb == 0), dict(D_log_bad=lb, D_command_escapes=cb, D_repeat_requests=rb, D_commands_at_lagged_on_ticks=n_on)


def fam_E(E):
    bad = 0
    for r in E['recs']:
        for arm in ('lag0', 'lag1'):
            L_ = r[arm]; cash = -sum(SG[int(f[2])] * float(f[3]) * float(f[4]) for f in L_['fills']); pos = sum(SG[int(f[2])] * float(f[3]) for f in L_['fills'])
            bad += int(not (close(cash, L_['cash']) and close(pos, L_['position']) and close(cash + pos * L_['mark'], L_['value'])))
    return dict(E_accounting=bad == 0), {}


UP, INV = ('both_exact', 'bid_exact', 'ask_exact'), ('mean_abs_inv', 'peak_abs_inv', 'share_abs_ge_S', 'terminal_inv')
QF = ('q_activation_ahead_mean', 'q_first_review_venue_ahead_mean', 'q_first_review_known_ahead_mean', 'q_pulls', 'q_pull_venue_ahead_mean')
PF = ('pairs_pulls', 'pairs_with_next', 'pairs_same_price', 'pairs_same_rung', 'pairs_ahead_difference_mean', 'pairs_same_price_ahead_difference_mean')


def selection_lagged(E):
    S = E['states'].copy(); live = np.nonzero((S.kind == 'live').values)[0]; vals = S.adv_bn_r250.astype(float).values.copy(); mis = warm = nok = 0
    for i in live:
        c = cache(E, str(S.day.iat[i]), int(S.m.iat[i])); s = int(S.side.iat[i]); k = int(round(float(S.t.iat[i]) / STEP)); mis += int(bits(c['x'][s][k]) != bits(S.adv_bn_r250.iat[i]))
        if k < 1:
            warm += 1; vals[i] = np.nan
        elif int(c['status'][s][k - 1]) != 0:
            nok += 1; vals[i] = np.nan
        else:
            vals[i] = float(c['x'][s][k - 1])
    S['adv_bn_r250'] = vals; return PC.selection_recompute(S, E['gm'], E['edges_raw']), dict(live_states=len(live), zero_lag_mismatched=mis, warmup=warm, non_ok=nok)


def fam_F(E):  # noqa: C901
    A_ = E['arms']; c, det = {}, {}; rb = ib = 0; rc = {}
    for r in E['recs']:
        end = float(r['path_info']['end_rel'])
        for arm in ('baseline', 'lag0', 'lag1'):
            L_ = arm_ledger(E, r, arm); x = dict(PC.recompute(L_, end)); x.update(uptime(L_)); x.update(inventory(L_))
            if arm != 'baseline':
                q_, _ = queue(L_, arm_cmds(E, r, arm)); x.update(q_)
            rc[(r['_day'], int(r['m']), arm)] = x; row = A_[(A_.day == r['_day']) & (A_.m == int(r['m'])) & (A_.arm == arm)]
            keys = list(PC.RECON) + list(UP) + list(INV) + (list(QF) + list(PF) if arm != 'baseline' else [])
            rb += int(len(row) != 1 or not all(close(row[k].iloc[0], x[k]) for k in keys))
            ib += int(not (close(x['all_id_mo1s'] + x['all_id_move'] + x['all_id_late'], x['all_end']) and close(x['all_end'], x['value'])))
    c['F_reconcile'] = rb == 0; det['F_reconcile_bad'] = rb
    LA = E['lost_added']; lb = 0
    for r in E['recs']:
        for pair, (a1, a2) in (('lag1_vs_baseline', ('baseline', 'lag1')), ('lag1_vs_lag0', ('lag0', 'lag1'))):
            fb, fg = arm_ledger(E, r, a1), arm_ledger(E, r, a2); lo = [f for f, u in zip(fb['fills'], PC.matched_split(fb['fills'], fg['fills'])) if u]
            ad = [f for f, u in zip(fg['fills'], PC.matched_split(fg['fills'], fb['fills'])) if u]; en = lambda F_, mk: sum(SG[int(f[2])] * float(f[3]) * (mk - float(f[4])) for f in F_)
            row = LA[(LA.day == r['_day']) & (LA.m == int(r['m'])) & (LA.pair == pair)]
            lb += int(len(row) != 1 or not (int(row.lost_n.iloc[0]) == len(lo) and int(row.added_n.iloc[0]) == len(ad) and close(row.lost_end.iloc[0], en(lo, float(fb['mark'])))
                                             and close(row.added_end.iloc[0], en(ad, float(fg['mark']))) and close(float(fg['value']) - float(fb['value']), en(ad, float(fg['mark'])) - en(lo, float(fb['mark'])))))
    c['F_identity'] = ib == 0 and lb == 0; det['F_identity_bad'] = ib + lb
    sr, js = selection_lagged(E); sel = E['selection']; det['F_selection'] = sr
    c['F_selection'] = (sr['answer'] == sel['answer'] and sel['seed'] == E['gm']['null']['seed'] and sel['draws'] == E['gm']['null']['draws'] and sr['selected'] == sel['selected']
                        and sr['states'] == sel['states'] and close(sr['J'], sel['J']) and close(sr['null_p05'], sel['null_p05']) and close(sr['null_mean'], sel['null_mean'])
                        and sr['movable_strata'] == sel['movable_strata'] and all(close(sr[k], sel[k]) for k in ('valid_window_q', 'total_window_q', 'excluded_window_q'))
                        and all(sel['join'][k] == js[k] for k in js))
    days = sorted({r['_day'] for r in E['recs']}); tot = lambda day, arm, k: sum(v[k] for (d, m, a), v in rc.items() if (d == day or day == 'all') and a == arm)
    mn = lambda day, arm, k: float(np.mean([v[k] for (d, m, a), v in rc.items() if a == arm and (d == day or day == 'all')]))
    cat = lambda v: 'IMPROVES_EVERY_DAY' if all(x > 0 for x in v) else 'WORSENS_EVERY_DAY' if all(x < 0 for x in v) else 'MIXED'
    per = lambda d, arm, side: tot(d, arm, f'{side}_cut_mo1s') / tot(d, arm, f'{side}_cut_valid_q'); pr = E['maudit']['predicates']; ok_r = True
    for key, cand, sel_ans in (('lag1_vs_baseline', 'lag1', sr['answer']), ('lag0_vs_baseline', 'lag0', E['parent_selection']['answer'])):
        p_ = pr[key]; dEND = {d: tot(d, cand, 'value') - tot(d, 'baseline', 'value') for d in days}; dps = {s: {d: per(d, cand, s) - per(d, 'baseline', s) for d in days} for s in ('all', 'bid', 'ask')}
        dq = {d: tot(d, cand, 'all_q') - tot(d, 'baseline', 'all_q') for d in days}
        ok_r &= (p_['END_paired']['answer'] == cat(list(dEND.values())) and all(close(p_['END_paired']['by_day'][d], dEND[d]) for d in days)
                 and all(p_['mo1s_per_share'][s]['answer'] == cat(list(dps[s].values())) and all(close(p_['mo1s_per_share'][s]['by_day'][d], dps[s][d]) for d in days) for s in dps)
                 and all(close(p_['participation']['filled_q_delta_by_day'][d], dq[d]) for d in days) and p_['selection'] == sel_ans
                 and all(close(p_['participation'][f'{u}_delta'], mn('all', cand, u) - mn('all', 'baseline', u)) for u in UP)
                 and close(p_['inventory']['mean_abs_delta'], mn('all', cand, 'mean_abs_inv') - mn('all', 'baseline', 'mean_abs_inv'))
                 and close(p_['inventory']['share_abs_ge_S_delta'], mn('all', cand, 'share_abs_ge_S') - mn('all', 'baseline', 'share_abs_ge_S'))
                 and close(p_['inventory']['terminal_abs_delta'], float(np.mean([abs(v['terminal_inv']) for (d, m, a), v in rc.items() if a == cand]))
                           - float(np.mean([abs(v['terminal_inv']) for (d, m, a), v in rc.items() if a == 'baseline']))))
        if key == 'lag1_vs_baseline':
            fails = ([f"END_paired: {cat(list(dEND.values()))}"] if cat(list(dEND.values())) != 'IMPROVES_EVERY_DAY' else []) + \
                    ([f"mo1s_per_share: {cat(list(dps['all'].values()))}"] if cat(list(dps['all'].values())) != 'IMPROVES_EVERY_DAY' else []) + \
                    ([f"selection: {sr['answer']}"] if sr['answer'] != 'SELECTION_BEYOND_EXPOSURE' else [])
    rt = pr['retained']
    for d in days + ['all']:
        b0, z0, z1 = tot(d, 'baseline', 'value'), tot(d, 'lag0', 'value'), tot(d, 'lag1', 'value'); ps = {a: per(d, a, 'all') for a in ('baseline', 'lag0', 'lag1')}
        ok_r &= close(rt['END'][d], (z1 - b0) / (z0 - b0)) and close(rt['per_share'][d], (ps['lag1'] - ps['baseline']) / (ps['lag0'] - ps['baseline']))
    ok_r &= all(close(rt['lag1_minus_lag0_END'][d], tot(d, 'lag1', 'value') - tot(d, 'lag0', 'value')) for d in days)
    ok_r &= pr['failed'] == fails and pr['reading'] == ('survives_50ms_screen' if not fails else 'not_demonstrated_at_50ms')
    c['F_readings'] = bool(ok_r); det['F_readings'] = dict(reading='survives_50ms_screen' if not fails else 'not_demonstrated_at_50ms', failed=fails)
    return c, det


def fam_G(E):
    _, js = selection_lagged(E); return dict(G_join=js['zero_lag_mismatched'] == 0 and js['live_states'] == int((E['states'].kind == 'live').sum())), dict(G_join=js)


def fam_H(E):
    bad = 0; RP = E['reentry']
    for r in E['recs']:
        for arm in ('lag0', 'lag1'):
            L_ = r[arm]; qv = L_['queue_venue']; cm = arm_cmds(E, r, arm)
            bad += sum(1 for o in L_['orders'] if not o['refused'] and str(o['gid']) not in qv['ahead0'])
            bad += sum(1 for z in L_['guard_log'] if z[10] == 'pressure_pull' and str(int(z[3])) not in qv['pull'])
            rev = {int(c_[3]) for c_ in cm if int(c_[2]) == M.CMD_KINDS.index('live') and int(c_[3]) >= 0 and P.REASONS[int(c_[8])] != 'fallback'}
            bad += sum(1 for g in rev if str(g) not in qv['first'])
            _, pairs = queue(L_, cm); rows = RP[(RP.day == r['_day']) & (RP.m == int(r['m'])) & (RP.arm == arm)]
            bad += int(len(rows) != len(pairs) or sorted(rows.pull_gid.astype(int)) != sorted(p['pull_gid'] for p in pairs))
    return dict(H_queue=bad == 0), dict(H_queue_bad=bad)


def fam_I(E, root=None):
    cpu, peak = 0., 0
    for p in glob.glob(str((root or R) / 'logs' / '**' / '*.log'), recursive=True):
        for ln in Path(p).read_text().splitlines():
            m_ = RL.match(ln)
            if m_:
                cpu += float(m_.group(6)); peak = max(peak, int(m_.group(4)))
    return dict(I_resources=cpu / 3600. <= CAP and peak < 1536), dict(I=dict(core_hours=cpu / 3600., peak_mib=peak))


FAMS = dict(A=fam_A, B=fam_B, C=fam_C, D=fam_D, E=fam_E, F=fam_F, G=fam_G, H=fam_H, I=fam_I)


def run_checks(E, fams=None):
    chk, det = {}, {}
    for k in (fams or FAMS):
        c, d = FAMS[k](E); chk.update(c); det.update(d)
    return dict(checks=chk, details=det, all_ok=bool(all(chk.values())))


# ---------------------------------------------------------------------------------------------------------------------- falsifier
def _rec(E, i, fn):
    E['recs'] = list(E['recs']); E['recs'][i] = copy.deepcopy(E['recs'][i]); fn(E['recs'][i])


def _first(E, arm, pred):
    return next(i for i, r in enumerate(E['recs']) if pred(r[arm]))


def _coordinated_uptime(E):
    """Review 909c671e's case: every lag1 both-side uptime +25 pp, with the stored reading moved with it."""
    A_ = E['arms'].copy(); A_.loc[A_.arm == 'lag1', 'both_exact'] += .25; E['arms'] = A_; pr = copy.deepcopy(E['maudit']['predicates'])
    pr['lag1_vs_baseline']['participation']['both_exact_delta'] += .25; E['maudit'] = dict(E['maudit'], predicates=pr)


def _keep_escape(E):
    i = _first(E, 'lag1', lambda L_: any(z[10] == 'pressure_pull' for z in L_['guard_log'])); r = E['recs'][i]; day, m = r['_day'], int(r['m']); key = f'{m}__lag1__commands'
    cm = E['arr'][day][key].copy(); j = int(np.nonzero(cm[:, 8] == P.REASONS.index('pressure_pull'))[0][0]); cm[j, 7] = M.CMD_ACTIONS.index('KEEP'); cm[j, 8] = P.REASONS.index('keep')
    E['arr'] = dict(E['arr']); E['arr'][day] = dict(E['arr'][day]); E['arr'][day][key] = cm


def _join_bad(E):
    S = E['states'].copy(); i = int(np.nonzero((S.kind == 'live').values)[0][0]); S.loc[S.index[i], 'adv_bn_r250'] = float(S.adv_bn_r250.iat[i]) + 1.; E['states'] = S


def _fake_logs(E):
    E['_root'] = Path(tempfile.mkdtemp(dir=str(D))); (E['_root'] / 'logs').mkdir()
    (E['_root'] / 'logs' / 'fake.log').write_text('2026-11-21T00:00:00Z tag=fake rc=0 peak_rss_mib=2000 wall_s=3000.0 cpu_user_s=3000.0\n')


CASES = [
    ('coordinated_uptime_and_reading', 'F', 'F_reconcile', _coordinated_uptime),
    ('lag1_inventory_changed', 'F', 'F_reconcile', lambda E: E.__setitem__('arms', E['arms'].assign(mean_abs_inv=E['arms'].mean_abs_inv + np.where(E['arms'].arm == 'lag1', 1., 0.)))),
    ('a_log_source_time_without_the_lag', 'D', 'D_log', lambda E: _rec(E, _first(E, 'lag1', lambda L_: L_['guard_log']), lambda r: r['lag1']['guard_log'][0].__setitem__(11, r['lag1']['guard_log'][0][0]))),
    ('a_keep_at_a_lagged_on_tick', 'D', 'D_commands', _keep_escape),
    ('zero_lag_fill_dropped', 'C', 'C_zero_lag_equals_parent', lambda E: _rec(E, 0, lambda r: r['lag0'].__setitem__('fills', r['lag0']['fills'][:-1]))),
    ('cash_changed', 'E', 'E_accounting', lambda E: _rec(E, 0, lambda r: r['lag1'].__setitem__('cash', r['lag1']['cash'] + 1.))),
    ('lost_end_changed', 'F', 'F_identity', lambda E: E.__setitem__('lost_added', E['lost_added'].assign(lost_end=E['lost_added'].lost_end + np.where(E['lost_added'].index == 0, 1., 0.)))),
    ('stored_selection_J_changed', 'F', 'F_selection', lambda E: E.__setitem__('selection', dict(E['selection'], J=E['selection']['J'] + 1.))),
    ('stored_reading_changed', 'F', 'F_readings', lambda E: E.__setitem__('maudit', dict(E['maudit'], predicates=dict(E['maudit']['predicates'], reading='survives_50ms_screen'
                                                                                                                  if E['maudit']['predicates']['reading'] != 'survives_50ms_screen' else 'not_demonstrated_at_50ms')))),
    ('a_saved_state_value_changed', 'G', 'G_join', _join_bad),
    ('a_pull_without_its_queue_entry', 'H', 'H_queue', lambda E: _rec(E, _first(E, 'lag1', lambda L_: L_['queue_venue']['pull']),
                                                                       lambda r: r['lag1']['queue_venue']['pull'].pop(next(iter(r['lag1']['queue_venue']['pull']))))),
    ('a_member_dropped', 'B', 'B_membership', lambda E: E.__setitem__('recs', E['recs'][:-1])),
    ('a_record_rule_changed', 'A', 'A_rule', lambda E: E.__setitem__('days', dict(E['days'], **{sorted(E['days'])[0]: dict(E['days'][sorted(E['days'])[0]], rule_sha256='0' * 64)}))),
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
    global D
    rehearse = len(sys.argv) > 2 and sys.argv[1] == 'rehearse'
    if rehearse:
        D = Path(sys.argv[2])
    E = load(); res = run_checks(E); dest = D if rehearse else R; tag = '_rehearsal' if rehearse else ''
    (dest / f'checks{tag}.json').write_text(json.dumps(res, indent=1, default=str) + '\n'); fz = falsify(E, res)
    (dest / f'checks_falsifier{tag}.json').write_text(json.dumps(fz, indent=1) + '\n')
    if rehearse:
        print(json.dumps(dict(checks=res['checks'], cases={k: (v['clean'], v['refused']) for k, v in fz['cases'].items()}), indent=1)); return
    print(json.dumps(dict(all_ok=res['all_ok'], failed=[k for k, v in res['checks'].items() if not v], falsifier_ok=fz['falsifier_ok'],
                          cases={k: (v['refused'], v['flipped']) for k, v in fz['cases'].items()}, details=res['details']), indent=1, default=str)[:5000])


if __name__ == '__main__':
    main()
