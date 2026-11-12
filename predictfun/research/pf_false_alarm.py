"""FA, the guard's false alarms and what they cost (2026-11-12, session ce68ebe0). USER: "I think the gap are mostly 2 things, one our
model detects the signals in time, but it fires quite some false alarms, two, we don't get the long term good fills measurement" /
"Yes, proceed".

Neutral reference path (CLAUDE.md rule 1): one 5-sh order per side resting at the known best and never pulled (pf_kp_s4 rest lane with
no trigger = B21's Z-none, run through pf_qr_queue_race.sim3 for per-order fills), 9 consumed replay days, first 60 s, M 0 / M 50 /
M 250. Rows: every 10th grid step (500 ms, so the near windows below tile and each fill counts once in 'near') at which an order of
ours is live and not yet fully filled. Per row, from what v3 sees at the
step: the guard ALARM = v3 would not quote this side (state off or v <= PM90); v = -1 (start price not yet known) is a status. Outcome,
from THIS order's own fills only (no re-placed order), valued at 1 s (sg x (mid(anchor + 1 s) - price), $ = shares x c / 100):
 near  = fills in (t + 22, t + 522] ms - the guard's own horizon (a pull at t takes effect at t + 22 ms)
 later = fills in (t + 522, t + 5000] ms - the flow the 500 ms target cannot see
Near classes: HARM (near value < 0), QUIET (no near fill), GOOD (near value > 0). An alarm is TRUE if HARM, FALSE if QUIET or GOOD.
Primary population (FINDINGS C1): rows where the order is >= 100 ms old at t (every fill counted is then on an order >= 100 ms old);
rows on younger orders printed beside. Cancel-pending fills do not occur on this path except at price-limit / window-end cancels
(counted, C3).

'False alarm' is defined by the outcome, so its near value is >= 0 by construction: sums over false-alarm rows are CEILINGS (what a
perfect classifier could keep), not decision values. The decision-level comparison of never pulling against pulling on the guard is
B21 / S4's (Z-none against Z-guard), not repeated here.
Readings DECLARED BEFORE THE FIRST RUN (evaluated in code; day rule = >= 7 of 9 days; primary population, M 0 and M 50):
 R340 PRECISION (descriptive): the share of alarm rows that are FALSE (QUIET or GOOD), pooled; MOSTLY-FALSE if > 80% at M 0 and
      M 50, else NOT-MOSTLY-FALSE. Recall (the share of HARM rows that the guard alarms on) printed.
 R341 DISCRIMINATION: per day, AUC of the alarm score (-v) for HARM against GOOD among rows with a near fill, and for ANY near fill
      against QUIET; ACTIVITY-NOT-HARM if the activity AUC exceeds the harm AUC on >= 7 of 9 days at M 50, else NOT-SHOWN.
 R342 CEILING (within the guard's 500 ms): $/day of near value on FALSE-alarm rows (>= 0 by construction: the most a perfect
      classifier could keep inside 500 ms) against |$/day| of near value on TRUE-alarm rows (the harm the alarms avoid), pooled at
      M 0 and M 50; CEILING-SMALL if the first is < 25% of the second at both, else CEILING-LARGE.
 R343 LATER (beyond 500 ms, the user's long-term point): the mean later value per row (c per 5-sh order) on FALSE-alarm rows, per day:
      > 0 by the day rule at M 0 and M 50 -> GOOD-FLOW-AFTER-FALSE-ALARMS; < 0 -> BAD-FLOW-AFTER-FALSE-ALARMS; else NOT-SHOWN. The same
      mean on no-alarm and true-alarm rows printed beside.
What the outcomes mean, fixed in advance: MOSTLY-FALSE = most alarms see no harm inside the guard's own horizon; ACTIVITY-NOT-HARM =
the guard reacts to busy moments more than to harmful ones; CEILING-LARGE = a better classifier inside 500 ms could keep real money;
GOOD-FLOW-AFTER-FALSE-ALARMS = good fills follow false alarms beyond the guard's horizon (the long-term value the target misses);
BAD-FLOW-... = after a false alarm, staying still loses over the next 4.5 s (the danger arrives later than the target sees).
Controls (any failure -> no table): the never-pull lane equals pf_kp_s4.sim2 'rest' with no trigger (fills and orders) on every
window-side and config; synthetic row cases (a fill at t + 10 ms is excluded, at t + 30 near, at t + 600 later, at t + 5100 neither;
a fill of a later order is excluded) with planted errors (window edges shifted, other orders' fills counted) that must fail.
Usage: research/run_fa.sh; python3 predictfun/research/pf_false_alarm.py selftest | day <yyyymmdd> | report
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402
import pf_qr_queue_race as Q                              # noqa: E402

NAN = float('nan'); INF = float('inf')
DAYS9 = K.DAYS9; OUTA = f'{K.IO}/fa_20261112'
CFG = [('M', 0.0), ('M', 50.0), ('M', 250.0)]
NEAR = (22.0, 522.0); LATER = (522.0, 5000.0); ROW_EVERY = 10; AGE_MIN = 100.0
MUT_EDGE = MUT_OTHER = 0


def row_outcomes(t, oid, fills_by_order):
    """Near and later value ($) and near fill count of order oid's own fills after t. fills_by_order: oid -> [(time, value $)]."""
    lo1, hi1 = NEAR; lo2, hi2 = LATER
    if MUT_EDGE:
        lo1 = 0.0
    near = 0.0; later = 0.0; nn = 0
    src = [x for k_, v_ in fills_by_order.items() for x in v_] if MUT_OTHER else fills_by_order.get(oid, [])
    for tf_, val in src:
        if t + lo1 < tf_ <= t + hi1:
            near += val; nn += 1
        elif t + lo2 < tf_ <= t + hi2:
            later += val
    return near, later, nn


def selftest():
    bad = []
    fb = {0: [(1010.0, -0.1), (1030.0, 0.2), (1600.0, 0.3), (6100.0, 9.0)], 1: [(1050.0, 5.0)]}
    near, later, nn = row_outcomes(1000.0, 0, fb)
    if not (abs(near - 0.2) < 1e-12 and abs(later - 0.3) < 1e-12 and nn == 1):
        bad.append(f'row windows {near} {later} {nn}')
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_EDGE', 'MUT_OTHER'):
        globals()[nm] = 1
        try:
            out[nm] = len(selftest()) > 0
        finally:
            globals()[nm] = 0
    return out


def auc(score, pos):
    from scipy.stats import rankdata
    pos = np.asarray(pos, bool); n1 = pos.sum(); n0 = (~pos).sum()
    if n1 == 0 or n0 == 0:
        return NAN
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def run_day(day):
    bad = selftest(); mut = mutation_check()
    if bad or not all(mut.values()):
        sys.exit(f'SELFTEST FAILED -> no output: {bad} {mut}')
    os.makedirs(OUTA, exist_ok=True)
    D = K.load_day(day); st = D['stat']; st.update(window_sides=0, reg_differ=0, rows=0, rows_young=0, rows_v_withheld=0, pend_fills=0)
    T0 = time.time(); ROWS = {f'{q}|{s:.0f}': [] for q, s in CFG}
    for w in K.window_sides(D):
        st['window_sides'] += 1; side = w['side']; tf = w['tf']; cache = w.setdefault('cache', {}); taul = w['tg'].tolist()
        aux = [False] * w['n']
        for q_, shf in CFG:
            tt = w['anc'] - shf; key = f'{q_}|{shf:.0f}'
            a = (side, tf, w['opp'], w['P5'], w['S5'], cache, w['tg'], w['vkl'], w['stn'], -INF, w['best'], w['bopp'], tt, w['ptl'], w['shl'], q_, w['te'])
            F, O, fx, _ = Q.sim3('rest', *a, aux=aux); F2, O2 = K.sim2('rest', *a, aux=aux)
            st['reg_differ'] += int(F != F2 or Q.otuples(O, w['te']) != O2)
            r = K.vals(w, F, tt); fb = {}
            for f, x, qq, m1 in zip(F, fx, r['q'], r['m1']):
                fb.setdefault(x[0], []).append((float(tt[f[0]]), float(qq * m1) / 100.0 if np.isfinite(m1) else 0.0))
                st['pend_fills'] += int(f[5] == 'pend')
            ends = [(o['ta'], min(o['tr'], o['tfill']), o['id']) for o in O if not o['ref']]
            for k in range(0, w['n'], ROW_EVERY):
                t = taul[k]; cur = [e for e in ends if e[0] <= t < e[1]]
                if not cur:
                    continue
                ta, _, oid = cur[-1]; v = w['vkl'][k]
                if v == -1.0:
                    st['rows_v_withheld'] += 1; continue
                alarm = not (bool(w['stv3'][k]) and v > w['P90'])
                near, later, nn = row_outcomes(t, oid, fb)
                st['rows'] += 1; st['rows_young'] += int(t - ta < AGE_MIN)
                ROWS[key].append((float(v), int(alarm), near, later, nn, float(t - ta)))
        w.pop('cache', None)
    if st['reg_differ']:
        sys.exit(f'FA CONTROL FAILED on {day} -> no result: never-pull lane differs from sim2 on {st["reg_differ"]} window-side-configs')
    np.savez_compressed(f'{OUTA}/rows_{day}.npz', **{k_.replace('|', '_'): np.array(v_, float) for k_, v_ in ROWS.items()})
    json.dump(dict(day=day, days=len(D['W0']) / 288.0, stat=st, planted=mut), open(f'{OUTA}/day_{day}.json', 'w'), default=float)
    print(f'{day}: done in {time.time() - T0:.0f} s; rows {st["rows"]}, control reg_differ 0', flush=True)


def report():
    L = []; RES = dict(as_of=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), days=DAYS9)
    DJ = {d: json.load(open(f'{OUTA}/day_{d}.json')) for d in DAYS9}; R = {d: dict(np.load(f'{OUTA}/rows_{d}.npz')) for d in DAYS9}
    st = {k_: sum(DJ[d]['stat'].get(k_, 0) for d in DAYS9) for k_ in ('window_sides', 'rows', 'rows_young', 'rows_v_withheld', 'pend_fills')}
    L.append(f'FA the guard\'s false alarms on the never-pull path - 9 replay days, first 60 s; as of {RES["as_of"]}; control passed; statuses {st}')
    cols = dict(v=0, alarm=1, near=2, later=3, nn=4, age=5)
    for key in [f'{q}_{s:.0f}' for q, s in CFG]:
        L.append(''); L.append(f'{key.replace("_", " ")}:')
        for pop, sel in (('orders >= 100 ms old (primary)', lambda x: x[:, 5] >= AGE_MIN), ('orders < 100 ms old', lambda x: x[:, 5] < AGE_MIN)):
            X = np.concatenate([R[d][key][sel(R[d][key])] for d in DAYS9 if len(R[d][key])])
            al = X[:, 1] == 1; harm = X[:, 2] < 0; quiet = X[:, 4] == 0; good = X[:, 2] > 0
            n = len(X); na = al.sum()
            fa = (al & ~harm).sum() / max(na, 1); rec = (al & harm).sum() / max(harm.sum(), 1)
            usd = lambda m: X[m, 2].sum()                    # near only: rows every 500 ms tile the near windows (each fill once)
            L.append(f'  {pop}: rows {n}, alarm share {na / max(n, 1):.1%}; among alarms: HARM {(al & harm).sum() / max(na, 1):.1%}, QUIET {(al & quiet).sum() / max(na, 1):.1%}, '
                     f'GOOD {(al & good).sum() / max(na, 1):.1%} -> false {fa:.1%}; recall of HARM {rec:.1%}')
            L.append(f'    $/day near (each fill once): false-alarm rows {usd(al & ~harm) / 9:.2f}, true-alarm rows {usd(al & harm) / 9:.2f}, no-alarm rows {usd(~al) / 9:.2f}; '
                     f'later value per row (c per order): alarm {100 * X[al, 3].mean():+.3f}, no alarm {100 * X[~al, 3].mean():+.3f}; '
                     f'near per row: alarm {100 * X[al, 2].mean():+.3f}, no alarm {100 * X[~al, 2].mean():+.3f}')
            RES[f'{key}|{pop[:6]}'] = dict(rows=int(n), alarm_share=float(na / max(n, 1)), false_share=float(fa), recall=float(rec),
                                           usd_false=float(usd(al & ~harm) / 9), usd_true=float(usd(al & harm) / 9), usd_noalarm=float(usd(~al) / 9))
    P = lambda d, key: R[d][key][R[d][key][:, 5] >= AGE_MIN]
    fs = []
    for key in ('M_0', 'M_50'):
        X = np.concatenate([P(d, key) for d in DAYS9]); al = X[:, 1] == 1; harm = X[:, 2] < 0
        fs.append((al & ~harm).sum() / max(al.sum(), 1))
    RES['R340'] = dict(label='MOSTLY-FALSE' if min(fs) > 0.8 else 'NOT-MOSTLY-FALSE', false_share_M0_M50=fs)
    L.append(''); L.append(f'R340 PRECISION: {RES["R340"]["label"]}; false share of alarms M 0 / M 50 ' + ' / '.join(f'{x:.1%}' for x in fs))
    ah = []; aa = []
    for d in DAYS9:
        X = P(d, 'M_50'); wf = X[:, 4] > 0
        ah.append(auc(-X[wf, 0], X[wf, 2] < 0)); aa.append(auc(-X[:, 0], X[:, 4] > 0))
    n_ = int(np.sum(np.array(aa) > np.array(ah)))
    RES['R341'] = dict(label='ACTIVITY-NOT-HARM' if n_ >= 7 else 'NOT-SHOWN', auc_harm=ah, auc_activity=aa, days=n_)
    L.append(f'R341 DISCRIMINATION (M 50): {RES["R341"]["label"]}; AUC harm vs good among near fills ' + ' '.join(f'{x:.3f}' for x in ah)
             + ' | AUC any near fill vs quiet ' + ' '.join(f'{x:.3f}' for x in aa))
    ok_small = []; tab = {}
    for key in ('M_0', 'M_50'):
        X = np.concatenate([P(d, key) for d in DAYS9]); al = X[:, 1] == 1; harm = X[:, 2] < 0
        fa_ = X[al & ~harm, 2].sum() / 9.0; ta_ = X[al & harm, 2].sum() / 9.0
        tab[key] = (fa_, ta_); ok_small.append(fa_ < 0.25 * abs(ta_))
    RES['R342'] = dict(label='CEILING-SMALL' if all(ok_small) else 'CEILING-LARGE', usd_false_true={k_: list(v_) for k_, v_ in tab.items()})
    L.append(f'R342 CEILING: {RES["R342"]["label"]}; $/day near value on false-alarm rows / on true-alarm rows: ' +
             '; '.join(f'{k_.replace("_", " ")} {v_[0]:+.2f} / {v_[1]:+.2f}' for k_, v_ in tab.items()))
    pos = []; neg = []; per = {}
    for key in ('M_0', 'M_50'):
        m_ = []
        for d in DAYS9:
            X = P(d, key); al = X[:, 1] == 1; harm = X[:, 2] < 0; m = al & ~harm
            m_.append(100.0 * X[m, 3].mean() if m.any() else NAN)
        m_ = np.array(m_); per[key] = m_.tolist(); pos.append(int(np.sum(m_ > 0))); neg.append(int(np.sum(m_ < 0)))
    RES['R343'] = dict(label='GOOD-FLOW-AFTER-FALSE-ALARMS' if min(pos) >= 7 else ('BAD-FLOW-AFTER-FALSE-ALARMS' if min(neg) >= 7 else 'NOT-SHOWN'),
                       days_pos=pos, days_neg=neg, per_day=per)
    side_ = []
    for key in ('M_0', 'M_50'):
        X = np.concatenate([P(d, key) for d in DAYS9]); al = X[:, 1] == 1; harm = X[:, 2] < 0
        side_.append(f'{key.replace("_", " ")}: false {100 * X[al & ~harm, 3].mean():+.3f}, true {100 * X[al & harm, 3].mean():+.3f}, no alarm {100 * X[~al, 3].mean():+.3f}')
    L.append(f'R343 LATER: {RES["R343"]["label"]}; days mean later value after false alarms > 0 at M 0 / M 50 {pos}, < 0 {neg}; pooled c per order: ' + '; '.join(side_))
    json.dump(RES, open(f'{OUTA}/fa.json', 'w'), indent=1, default=float)
    open(f'{OUTA}/fa.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        bad = selftest(); mut = mutation_check(); print('failing:', bad, 'planted errors caught:', mut); sys.exit(0 if not bad and all(mut.values()) else 1)
    elif cmd == 'day':
        run_day(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit(__doc__)
