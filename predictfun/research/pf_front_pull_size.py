"""FRONT-PULL-SIZE (2026-11-13, session ce68ebe0; USER: "it looks we need to increase our size and price levels for testing? Yes can you try
to test and check if these next steps can bridge the gaps"): the top makers' order shape (ORDERS, 1d6040b2: they rest 30-100 sh orders,
keep them through fills, quote ~6 prices per side per minute) in our replay, with FRONT-PULL's pulls.

Simulator: S4's rest lane (pf_kp_s4.py), 9 consumed replay days, first 60 s, clocks / fill rules / trade times / latencies of B21 / S4. The
rest lane keeps an order through partial fills and re-places it only once a full fill is known (+100 ms). pf_kp_s4.py gains two inert
switches through sim_arm: size (the order's size; default 5 sh) and offset (the rest lane places its order d ticks behind the known best
and holds it up to 3 + d ticks behind; default 0). Off, every S4 / B21 / FRONT-PULL path is unchanged (self-tests and C1).
Signal and lines: FRONT-PULL's (df320ba3): P = the per-day AV model (fitted without the day), lines = the 50 / 90% quantiles of P over the
training days; FPQ's tier rule (<= 25 sh known ahead -> 90% line, else 50%; entry blocked at the 50% line).
Arms (per held-out day; fill clocks M 0 / 50 / 250 and U 50, the front bound, descriptive):
  J-v3            the v3 quote line, 5 sh (reference)
  Z-none-S, Z-guard-S, FPQ-S    one order per side at the best, S = 5 / 30 / 100 sh: never pulled / pulled at v <= -0.10 / FPQ
  LZ-none-30, LZ-guard-30, LFPQ-30   a ladder of 6 rungs per side (d = 0..5 ticks behind the best), 30 sh each, each rung simulated as its
                  own order. Rung 0 = the single-level arm. Rungs d >= 1 are placed without an entry block and pulled only while they are
                  AT the known best (the level ahead has gone), by the same trigger (v <= -0.10 / FPQ's tier rule)
Values per fill as FRONT-PULL (S4's): 1 / 5 / 60 s and settlement; order-age split (< 100 / 100-500 / >= 500 ms), w/o young, B1a-weighted;
swept, pending, front shares; per order: the number of fills (orders filled >= 3 times, as ORDERS); per window: |net| / gross shares at
the end of the minute (both sides, all rungs).
Readings DECLARED BEFORE THE FIRST RUN (9 consumed days; day rule >= 7 of 9; w/o young at 1 s unless stated):
 R410 SIZE: FPQ-30 against FPQ-5, c/sh at M 0 and at M 50: higher on >= 7 days at both -> SIZE-HELPS-PER-SHARE; lower on >= 7 at both ->
      SIZE-HURTS-PER-SHARE; else NOT-SHOWN (FPQ-100 printed).
 R411 LADDER: LFPQ-30 against FPQ-30, $/day at M 0 and at M 50: higher on >= 7 at both -> LADDER-HELPS; lower -> LADDER-HURTS; else NOT-SHOWN.
 R412 VS-V3: LFPQ-30 against J-v3, $/day at 1 s and 5 s x M 0 / 50 / 250 -> BEATS-V3 if 6 of 6, else 'k of 6'.
 R413 POSITIVE: LFPQ-30's and FPQ-30's pooled $/day > 0 at M 0 / 50 / 250 -> 'k of 3' each.
 Printed, not readings: every arm by clock; day-clustered LFPQ-30 minus J-v3 and FPQ-30 minus FPQ-5 (t, 8 df); LZ-guard-30 beside LFPQ-30.
Controls (any failure -> no output): S4's / B21's / FRONT-PULL's self-tests; switch cases on a synthetic book (a 30-sh order is filled by
two trades and stays, a 5-sh order once; an order with offset 1 rests one tick behind; falsifier: offset 0 and 1 differ; defaults equal
S4's plain rest lane); C1 at 5 and 30 sh (the never-pull callback path equals Z-none); P aligned with FRONT-PULL's fit rows (|diff| <= 1e-9);
age bins add up; memory self-guard (2 GiB).
Limits: replay rule M (B1a; not validated, a fifth of live's fills contradict the shown queue); rungs are independent orders (no shared
inventory control, no CAP / RO); bigger orders carry more inventory to settlement (reported); our orders are not in the public book, so
a 100-sh order changes nothing in others' behaviour; consumed days, design after ORDERS (exploratory).
Usage (run_light, PF_SLOT=claude): pf_front_pull_size.py selftest | eval <day> | report; driver run_front_pull_size.sh
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402
import pf_front_pull as FP                                # noqa: E402

INF = float('inf')
DAYS9 = K.DAYS9
OUTS = f'{K.IO}/front_pull_size_20261113'
EVAL = [('M', 0.0), ('M', 50.0), ('M', 250.0), ('U', 50.0)]
SIZES = (5.0, 30.0, 100.0)
RUNGS = range(6)
LAD_SIZE = 30.0
ARMS = ['J-v3'] + [f'{a}-{int(s)}' for s in SIZES for a in ('Z-none', 'Z-guard', 'FPQ')] + ['LZ-none-30', 'LZ-guard-30', 'LFPQ-30']
MEM_CAP_MIB = 2048.0


def switch_selftest():
    """size and offset on a synthetic bid book under rule U (nothing ahead): trades at 50 at 300 and 400 ms."""
    bad = []
    fr = [(-100, [(50, 100)], 52)]; tr = [(300, 50, 10), (400, 50, 10)]
    tf = np.array([f[0] for f in fr], float); PB = np.full((1, 5), -1); SB = np.zeros((1, 5)); PB[0, 0] = 50; SB[0, 0] = 100
    ba = np.array([52]); tau = np.arange(0.0, 1000.0, 50.0); n = len(tau); vv = [1.0] * n; st = [True] * n
    best = [50] * n; bo = [52] * n; tt = np.array([x[0] for x in tr], float); tp = [x[1] for x in tr]; tq = [float(x[2]) for x in tr]

    def run(size=None, offset=0):
        s0 = K.SIZE
        try:
            K.SIZE = s0 if size is None else size; K.REST_OFFSET = offset
            return K.sim2('rest', 0, tf, ba, PB, SB, {}, tau, vv, st, -INF, best, bo, tt, tp, tq, 'U', 1000.0, aux=[False] * n)
        finally:
            K.SIZE = s0; K.REST_OFFSET = 0
    F30, O30 = run(30.0); F5, O5 = run(5.0); Fo, Oo = run(30.0, 1); F0, O0 = run()
    if not (len(F30) == 2 and abs(sum(f[2] for f in F30) - 20.0) < 1e-9 and len([o for o in O30 if not o[3]]) == 1):
        bad.append(f'30-sh order: fills {[(f[0], f[2]) for f in F30]}, orders {len(O30)}')
    if not (len(F5) == 1 and abs(F5[0][2] - 5.0) < 1e-9):
        bad.append(f'5-sh order: fills {[(f[0], f[2]) for f in F5]}')
    if not (len(Fo) == 0 and all(f[1] == 49 for f in Fo) and any(True for _ in Oo)):
        bad.append(f'offset 1: fills {Fo}')
    if Fo == F30:
        bad.append('falsifier: offset 0 and 1 give the same path')
    if (F0, O0) != K.sim2('rest', 0, tf, ba, PB, SB, {}, tau, vv, st, -INF, best, bo, tt, tp, tq, 'U', 1000.0, aux=[False] * n):
        bad.append('defaults differ from the plain rest lane')
    return bad


def all_selftests():
    FP.all_selftests()
    b = switch_selftest()
    if b:
        sys.exit(f'SELFTEST FAILED -> no output: {b}')


def rung_dec(P, v, l_back, l_front, kind, side):
    """Rungs d >= 1: pulled only while AT the known best, by the arm's trigger."""
    def dec(k, t, cur):
        f = K.own_feats(k, t, cur)
        if not (f[1] == f[1]) or f[1] != 0:
            return False
        if kind == 'guard':
            return bool(v[k] == v[k] and v[k] <= K.GUARD_PULL)
        a0, mn = f[2], cur['_kpmin']
        ah = min(a0, mn) if (a0 == a0 and mn == mn) else a0
        p = P[k]
        return bool(p == p and p >= FP.fp_line(ah, l_back, l_front))
    return dec


def order_fill_counts(F, O, tt):
    """Fills per live order (one order per lane at a time: a fill belongs to the order live at its trade time at its price)."""
    out = []
    ts = np.array([tt[f[0]] for f in F], float) if F else np.zeros(0); ps = np.array([f[1] for f in F]) if F else np.zeros(0)
    for o in O:
        if o[3]:
            continue
        out.append(int(((ts >= o[0]) & (ts <= o[1])).sum()))
    return out


def fpq_dec(P, l50, l90):
    """FPQ's resting-order rule (FRONT-PULL run_fpq): the line by the known queue ahead."""
    def dec(k, t, cur):
        f = K.own_feats(k, t, cur)
        a0, mn = f[2], cur['_kpmin']
        ah = min(a0, mn) if (a0 == a0 and mn == mn) else a0
        p = P[k]
        return bool(p == p and p >= FP.fp_line(ah, l50, l90))
    return dec


def lane(w, q, shf, kind, size, offset, P, vv, vg, l50, l90):
    """One order lane; returns (fills, orders, pulls, queue positions)."""
    if kind == 'none':
        F, O, p = K.sim_arm(w, q, shf, 'rest', size=size, offset=offset)
    elif offset == 0 and kind == 'guard':
        F, O, p = K.sim_arm(w, q, shf, 'rest', aux=vg, size=size)
    elif offset == 0:
        F, O, p = K.sim_arm(w, q, shf, 'rest', aux=(P >= l50).tolist(), decide=fpq_dec(P, l50, l90), idle_only=1, size=size)
    else:
        F, O, p = K.sim_arm(w, q, shf, 'rest', decide=rung_dec(P, vv, l50, l90, kind, w['side']), size=size, offset=offset)
    return F, O, p, list(K.LAST_POS)


NAME = {'none': 'Z-none', 'guard': 'Z-guard', 'fpq': 'FPQ'}
LNAME = {'none': 'LZ-none-30', 'guard': 'LZ-guard-30', 'fpq': 'LFPQ-30'}


def fp_eval(day):
    import lightgbm as lgb
    all_selftests()
    fit = json.load(open(f'{FP.OUTF}/fit.json'))['days'][day]; L = fit['lines']; l50, l90 = L['0.50'], L['0.90']
    mdl = lgb.Booster(model_file=f'{FP.OUTF}/models/av_{day}.txt')
    z = np.load(f'{FP.MOVE_DIR}/day_{day}.npz'); zm = z['B|m'].astype(np.int64); zt = z['B|t']; zs = z['B|side'].astype(int)
    pz = np.load(f'{FP.OUTF}/p_rows_{day}.npy')
    ROWS = {(int(m_), int(s_)): (zt[ix], pz[ix]) for (m_, s_), ix in pd.DataFrame(dict(m=zm, s=zs)).groupby(['m', 's']).indices.items()}
    D = K.load_day(day)
    acc = {(a, q, s): FP.new_acc() for a in ARMS for q, s in EVAL}
    for a_ in acc.values():
        a_.update(orders_live=0, orders_ge3=0, orders_filled=0)
    NET = {k_: {} for k_ in acc}; GROSS = {k_: 0.0 for k_ in acc}
    st = D['stat']; st.update(window_sides=0, c1_differ=0, align_rows=0, align_bad=0)
    T0 = time.time(); nb = len(K.G32)

    def rec(key, w, F, O, tt, pulls, pos):
        FP.add(acc[key], w, F, O, tt, pulls, pos)
        cnt = order_fill_counts(F, O, tt); a_ = acc[key]
        a_['orders_live'] += len(cnt); a_['orders_filled'] += sum(c >= 1 for c in cnt); a_['orders_ge3'] += sum(c >= 3 for c in cnt)
        sg = 1.0 if w['side'] == 0 else -1.0
        q_ = float(sum(f[2] for f in F)); NET[key][w['m']] = NET[key].get(w['m'], 0.0) + sg * q_; GROSS[key] += q_
    for w in K.window_sides(D):
        st['window_sides'] += 1; n = w['n']
        Xw = np.column_stack([w['X'], np.full(n, float(w['side']))]).astype(np.float32)
        P = mdl.predict(Xw) if n else np.zeros(0)
        vk = w['X'][:, nb] if n else np.zeros(0)
        P = np.where(~np.isfinite(vk) | (vk == -1.0) | (np.asarray(w['best']) < 0) | (w['jk'] < 0), np.nan, P)
        if (w['m'], w['side']) in ROWS:
            tz_, pz_ = ROWS[(w['m'], w['side'])]
            ii = np.searchsorted(w['tg'], tz_); ok = (ii < n) & (w['tg'][np.clip(ii, 0, n - 1)] == tz_)
            st['align_rows'] += len(tz_); st['align_bad'] += int((~ok).sum()) + int((~(np.abs(P[ii[ok]] - pz_[ok]) <= 1e-9)).sum())
        vv = np.asarray(w['vkl'], float); vg = (vv <= K.GUARD_PULL).tolist()
        for q, shf in EVAL:
            tt = w['anc'] - shf
            F, O, _ = K.sim_arm(w, q, shf, 'J-v3'); rec(('J-v3', q, shf), w, F, O, tt, 0, list(K.LAST_POS))
            for S in SIZES:
                for kind in ('none', 'guard', 'fpq'):
                    F, O, p, pos = lane(w, q, shf, kind, S, 0, P, vv, vg, l50, l90)
                    rec((f'{NAME[kind]}-{int(S)}', q, shf), w, F, O, tt, p, pos)
                    if S == LAD_SIZE:
                        rec((LNAME[kind], q, shf), w, F, O, tt, p, pos)          # rung 0 of the ladder
                    if kind == 'none' and S in (5.0, LAD_SIZE):
                        Fk, Ok, _ = K.sim_arm(w, q, shf, 'rest', decide=K.decide_record, size=S)
                        st['c1_differ'] += int(Fk != F or Ok != O)
            for d in RUNGS:
                if d == 0:
                    continue
                for kind in ('none', 'guard', 'fpq'):
                    F, O, p, pos = lane(w, q, shf, kind, LAD_SIZE, d, P, vv, vg, l50, l90)
                    rec((LNAME[kind], q, shf), w, F, O, tt, p, pos)
        if st['window_sides'] % 40 == 0:
            r_ = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r_:.0f} MiB', flush=True)
            if r_ > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r_:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    if st['c1_differ'] or st['align_bad'] or st['align_rows'] != len(zm):
        sys.exit(f'FRONT-PULL-SIZE CONTROL FAILED -> no result: C1 {st["c1_differ"]}, alignment {st["align_bad"]} of {st["align_rows"]} ({len(zm)})')
    for key in acc:
        acc[key]['net_abs'] = float(sum(abs(x) for x in NET[key].values())); acc[key]['gross_win'] = float(GROSS[key])
    out = dict(day=day, days=len(D['W0']) / 288.0, stat=st, arms={f'{a}|{q}|{s:.0f}': v for (a, q, s), v in acc.items()})
    os.makedirs(OUTS, exist_ok=True)
    json.dump(out, open(f'{OUTS}/eval_{day}.json', 'w'), default=float)
    print(f'{day}: evaluated in {time.time() - T0:.0f} s; {st}', flush=True)


def fp_report():
    from scipy.stats import t as tdist
    import pf_fv3_check as FV
    E = {d: json.load(open(f'{OUTS}/eval_{d}.json')) for d in DAYS9}
    old = [nm for nm, lo, _ in FP.ABINS if lo >= 100.0]; bad = {}
    for d in DAYS9:
        for key, a in E[d]['arms'].items():
            if abs(sum(a[f'q|{nm}'] for nm, _, _ in FP.ABINS) - a['q']) > 1e-6 or abs(sum(a[f'v1|{nm}'] for nm, _, _ in FP.ABINS) - a['v1']) > 1e-6:
                bad[f'bins {d} {key}'] = 1
    if bad:
        sys.exit(f'FRONT-PULL-SIZE REPORT CONTROL FAILED -> no table: {bad}')
    usd = lambda x, dd: x / 100.0 / dd

    def cell(a, dd):
        r = dict(all1=usd(a['v1'], dd), wo1=usd(sum(a[f'v1|{nm}'] for nm in old), dd), wo5=usd(sum(a[f'v5|{nm}'] for nm in old), dd),
                 st=usd(a['vst'], dd), q=a['q'] / dd, pulls=a['pulls'] / dd)
        for nm, _, _ in FP.ABINS:
            r[f'b|{nm}'] = usd(a[f'v1|{nm}'], dd)
        for k_, wt in FV.B1A.items():
            r[f'w_{k_}'] = sum(r[f'b|{b[0]}'] * p for b, p in zip(FV.BINS, wt))
        qo = sum(a[f'q|{nm}'] for nm in old)
        r['cps_wo1'] = sum(a[f'v1|{nm}'] for nm in old) / qo if qo else float('nan')
        for c in ('q_thr', 'q_pend', 'q_front5'):
            r[c] = a[c] / a['q'] if a['q'] else float('nan')
        r['ge3'] = a['orders_ge3'] / a['orders_filled'] if a['orders_filled'] else float('nan')
        r['net_gross'] = a['net_abs'] / a['gross_win'] if a['gross_win'] else float('nan')
        return r
    per = {d: {k_: cell(a, E[d]['days']) for k_, a in E[d]['arms'].items()} for d in DAYS9}
    alld = sum(E[d]['days'] for d in DAYS9)
    pooled = {k_: cell({c: sum(E[d]['arms'][k_][c] for d in DAYS9) for c in E[DAYS9[0]]['arms'][k_]}, alld) for k_ in E[DAYS9[0]]['arms']}
    dv = lambda arm, c, col: np.array([per[d][f'{arm}|{c}'][col] for d in DAYS9])
    R = {}
    up = [int(np.sum(dv('FPQ-30', c, 'cps_wo1') > dv('FPQ-5', c, 'cps_wo1'))) for c in ('M|0', 'M|50')]
    dn = [int(np.sum(dv('FPQ-30', c, 'cps_wo1') < dv('FPQ-5', c, 'cps_wo1'))) for c in ('M|0', 'M|50')]
    R['R410'] = 'SIZE-HELPS-PER-SHARE' if min(up) >= 7 else ('SIZE-HURTS-PER-SHARE' if min(dn) >= 7 else 'NOT-SHOWN'); R['R410_days_up_down'] = [up, dn]
    up = [int(np.sum(dv('LFPQ-30', c, 'wo1') > dv('FPQ-30', c, 'wo1'))) for c in ('M|0', 'M|50')]
    dn = [int(np.sum(dv('LFPQ-30', c, 'wo1') < dv('FPQ-30', c, 'wo1'))) for c in ('M|0', 'M|50')]
    R['R411'] = 'LADDER-HELPS' if min(up) >= 7 else ('LADDER-HURTS' if min(dn) >= 7 else 'NOT-SHOWN'); R['R411_days_up_down'] = [up, dn]
    k6 = sum(pooled[f'LFPQ-30|{q}|{s:.0f}'][u] > pooled[f'J-v3|{q}|{s:.0f}'][u] for q, s in K.EVAL_CFG for u in ('wo1', 'wo5'))
    R['R412'] = 'BEATS-V3' if k6 == 6 else f'{k6} of 6'
    for arm in ('LFPQ-30', 'FPQ-30'):
        R[f'R413 {arm}'] = f"{sum(pooled[f'{arm}|{q}|{s:.0f}']['wo1'] > 0 for q, s in K.EVAL_CFG)} of 3"
    IV = {}
    for a1, a0 in (('LFPQ-30', 'J-v3'), ('FPQ-30', 'FPQ-5'), ('LFPQ-30', 'LZ-guard-30')):
        for q, s in EVAL:
            c = f'{q}|{s:.0f}'; x = dv(a1, c, 'wo1') - dv(a0, c, 'wo1'); h = float(tdist.ppf(0.975, 8) * x.std(ddof=1) / 3.0)
            IV[f'{a1} - {a0} {c}'] = dict(mean=float(x.mean()), lo=float(x.mean() - h), hi=float(x.mean() + h), days_better=int((x > 0).sum()))
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    json.dump(dict(as_of=asof, readings=R, pooled=pooled, per_day=per, intervals=IV), open(f'{OUTS}/front_pull_size.json', 'w'), indent=1, default=float)
    T = [f'FRONT-PULL-SIZE, 9 consumed replay days, first 60 s; as of {asof}; controls passed (self-tests, switches, C1 at 5 / 30 sh, P aligned, bins)',
         'Readings: ' + json.dumps(R), '',
         'pooled $/day at 1 s: all | < 100 | 100-500 | >= 500 ms | w/o young | B1a M0 / M50 | w/o young 5 s | settle | c/sh w/o young | sh/day | '
         'pulls/day | swept | pending | front <= 5 sh | orders filled >= 3x | |net| / gross']
    for q, s in EVAL:
        for a in ARMS:
            y = pooled[f'{a}|{q}|{s:.0f}']
            T.append(f"  {q}{s:<3.0f} {a:<12s} {y['all1']:+8.2f} | {y['b|< 100 ms']:+8.2f} | {y['b|100-500 ms']:+8.2f} | {y['b|>= 500 ms']:+8.2f} | {y['wo1']:+8.2f} | "
                     f"{y['w_M0']:+7.2f} / {y['w_M50']:+7.2f} | {y['wo5']:+8.2f} | {y['st']:+9.2f} | {y['cps_wo1']:+.2f} | {y['q']:7.0f} | {y['pulls']:6.0f} | "
                     f"{y['q_thr']:.0%} | {y['q_pend']:.0%} | {y['q_front5']:.0%} | {y['ge3']:.1%} | {y['net_gross']:.0%}")
    T += ['', 'per day, $/day w/o young at 1 s (M 50): LFPQ-30 | FPQ-30 | FPQ-5 | LZ-guard-30 | Z-guard-30 | J-v3']
    for d in DAYS9:
        p_ = per[d]
        T.append(f"  {d} " + ' | '.join(f"{p_[f'{a}|M|50']['wo1']:+8.2f}" for a in ('LFPQ-30', 'FPQ-30', 'FPQ-5', 'LZ-guard-30', 'Z-guard-30', 'J-v3')))
    T += ['', 'day-clustered differences, $/day w/o young at 1 s (mean [95% t-interval], days better):']
    for k_, y in IV.items():
        T.append(f"  {k_}: {y['mean']:+.2f} [{y['lo']:+.2f}, {y['hi']:+.2f}], {y['days_better']} / 9")
    open(f'{OUTS}/front_pull_size.txt', 'w').write('\n'.join(T) + '\n'); print('\n'.join(T))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        all_selftests(); print('self-tests PASS (B21, S1, S4, FRONT-PULL, size / offset switches)')
    elif cmd == 'eval' and len(sys.argv) > 2:
        fp_eval(sys.argv[2])
    elif cmd == 'report':
        fp_report()
    else:
        sys.exit(__doc__)
