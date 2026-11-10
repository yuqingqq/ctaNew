"""B1a (USER "Yes" 2026-11-10; specification HANDOFF 2026-11-10 04:56Z, session 62f3ce33): live's OWN order paths
through the research fill model, compared order by order and leg by leg with what live was filled. A historical
reference, not a calibration: no delay, minimum resting age or queue parameter is fitted or selected (fill-clock audit).
One consumed partial day (the 220 windows of 2026-11-08, cut 23:00Z) -> point estimates only, no intervals (G < 5).

Version 2 (2026-11-10, after v1's output was seen; declared before v2 ran). v1 (7ef3de6e, output kept as b1a_v1.*)
matched a replay fill to live's own leg by the maker LEG ROW. Within one taker order the replay serves the legs at one
price in an arbitrary row order, so a replay fill from the same taker order as live's own leg counted as not REAL (post
hoc check of v1: 33 of 59 uncredited own legs inside their order's interval had a replay fill from the same taker
order). v2 matches on the TAKER ORDER (the replay's own time unit). Every reading, threshold, control and config is
unchanged; v1's leg-row numbers are printed beside v2's. Added, descriptive (no reading): live's own legs at M 0 P1 by
outcome in the replay (before activation / after the order left / credited / order filled earlier / not credited, with
the volume traded at our price before the leg at least or below the queue ahead at activation).

Population  accepted live orders (core_orders_{v6,basis} of pf_live_vs_replay: t_Accepted present, not refused) of
            tokyo-6 and basis-tokyo-1 in the 220 cohort windows, whole windows (orders are decided 1-240 s in).
            tokyo-5 (11-07) has no research tape in this cohort: a status; nothing of it is read.
Order path  side = lane % 2 (0 = Up bid, 1 = Up ask); price = the core tick (= the research legs' tick; checked).
            P1 (primary): live at the venue from t_Accepted (receipt of the ack, the latest possible activation) until
               t_CancelDone (receipt of the cancel ack), or the last fill notice if the order was filled in full, or the
               window's end, whichever is first.
            P0 (bound): from t_req (the decision) until t_cancel (the cancel request), same terminal rules.
Fill rule   that of pf_review_queue_reach.sim (pf_step4 fill model v2): queue ahead = shown size at our price when the
            order goes live (raised to the next frame's if larger and <= 100 ms later); M caps it by every frame strictly
            before a trade; a trade at our price takes the queue ahead first, then us; a trade through our price fills us
            and clears the queue. L = M without the cap; U = queue ahead 0. Several live orders on one side are served in
            the order they went live. No post-only check (the venue accepted every order).
Trades      every maker leg (legs_timed2 + legs_tk), one time per taker order (earliest WS-timed leg, else earliest E2
            estimate), minus SHIFT ms (the unknown match-to-report lag; not fitted). Untimed legs: status.
Configs     M 0 / M 50 / M 250 / L 50 / U 50 on P1; M 50 on P0.
Identity    live's own legs = the 99 REST maker legs of the fill-clock audit (own_rest_legs / clock_cases; hashes checked
            against its committed receipt), matched to legs_timed2 rows on the nine content keys. Eligible = the audit's
            unique or identical-clock legs that the replay can time. A replay fill is REAL if it comes from the taker
            order of an eligible own leg of the same order (v2; credited up to that leg's shares, in time order). Fills
            from the taker order of an ineligible own leg of the same order are a status, outside precision.
Value       per fill sg*(mid(anchor + h) - price), h = 1 s, 5 s, anchor unshifted (as in the review); live's eligible own
            legs are valued the same way at their research anchor.

Readings DECLARED BEFORE ANY RESULT (evaluated in code; 'n/a' below the stated minimum):
 R24 volume (per config): rho = replay shares on live's orders / live's filled shares on them. OVER-CREDITS if
     rho > 1.25; UNDER-CREDITS if rho < 0.80; else MATCHED.
 R25 identity (per config; n/a under 30 eligible own legs): recall = REAL shares / eligible own-leg shares;
     precision = REAL shares / replay shares outside the status. AGREES if both >= 0.6; FAILS if either < 0.3; else
     PARTIAL.
 R26 young orders (M 0 and M 50 on P1; n/a if a bucket has < 100 replay shares in a config): precision of replay fills
     on orders < 100 ms old (age = shifted trade time - activation) against >= 500 ms old. YOUNG-PHANTOM if
     precision(< 100 ms) < 0.5 x precision(>= 500 ms) in both configs; YOUNG-REAL if precision(< 100 ms) >=
     precision(>= 500 ms) in both; else MIXED. Null (an instrument diagnostic; no inference from one day): 500
     permutations of the age bucket among the M 50 replay fills within run x side; statistic precision(>= 500 ms) -
     precision(< 100 ms); null p95 / p99 and the share of draws at or above the observed value are reported.
 R27 lag bound (eligible own legs): U = research anchor - t_req. The order could not trade before its decision, so
     that trade's match-to-anchor lag is below U. CONSTANT-250-CONTRADICTED if own-leg shares with 0 <= U < 250 ms are
     >= 25% of eligible own-leg shares; CONSTANT-50-CONTRADICTED likewise for 0 <= U < 50 ms. U < 0 is a status
     (clock inconsistent). This bounds the lags of trades that filled us, not of other trades.
 R28 value (per config on P1; n/a under 30 eligible own legs): d = replay c/sh at 1 s on live's orders - live's c/sh
     at 1 s on its eligible own legs. REPLAY-OVERSTATES if d > +0.5; REPLAY-UNDERSTATES if d < -0.5; else MATCHES.
 R29 cancel-pending (M 50 on P1; n/a if both are under 20 sh): replay shares from trades at or after the cancel request
     against eligible own-leg shares whose anchor is at or after it (both at the config's shift). PEND-OVERCREDITED if
     replay > 2 x own; PEND-UNDERCREDITED if replay < 0.5 x own; else PEND-MATCHED.
Descriptive, no reading: orders filled live / in the replay (2 x 2); replay shares and precision by age; live's own legs
by age; live and replay (M 50) fill rate by the volume traded at our price while resting minus the queue ahead at
activation (orders resting >= 500 ms, P1, shift 0); P0 against P1.

Controls (any failure -> no table):
 K1a 2,000 random synthetic window-sides (both sides; rules M, L, U) through pf_review_queue_reach.sim: its orders,
     replayed here from the time they went live to the time they left, give the same fills (trade, price, shares,
     through, age) and the same queue ahead at activation: 0 differences.
 K1b the 0-60 s trade statuses rebuilt here equal the committed queue_reach.json statuses (753 untimed legs /
     9,131.366 sh; 94,783 timed legs / 1,074,957.357 sh) to 1e-6.
 K2 the audit's artifacts match its committed receipt hashes; every own leg has a legs_timed2 row on the nine keys; own
     legs' side and price equal their order's lane side and core tick (100%); the replay's anchor equals the audit's
     engine clock Z within 1 ms for every eligible own leg; live's filled shares equal the REST total (849.867 sh).
 K3 planted: an own leg after activation with no queue ahead is credited and REAL; one before activation is not
     credited (status); a trade of another maker is credited and not REAL (recall 0.5, precision 0.5 in that case).
 K4 refusals: unsorted trades or frames; an order decided at or after the cutoff. An order with no time to rest is a
     status, not replayed.
 K5 planted errors in the order replay (frame cap off; activation +10 ms; trades through our price ignored; own orders
     served in reverse): each must give at least one K1a difference or fail a synthetic case (S1 two orders at one
     price: the earlier is served first; S3 removal; S4 queue ahead at activation, frame cap, rules M / L / U).
 `--selftest` runs the synthetic controls only (K1a, K3, K4, K5); it reads no data.

Usage: predictfun/research/run_light.sh b1a_live_orders -- python3 predictfun/research/pf_b1a_live_orders.py <out.json>
"""
import bisect
import glob
import hashlib
import json
import sys

import numpy as np
import pandas as pd

import pf_review_queue_reach as QR

RP = QR.RP; S1 = QR.S1; INF = float('inf')
LV = f'{RP}/live_vs_replay'
FC = '/home/yuqing/ctaNew/data/predict_fun/research/fill_clock_20261109'
FC_SHA = {'own_rest_legs.parquet': '118d3f51cf4e67def6088e7b00f56c1439306528cf72ae6f6946c35799edb955',
          'clock_cases.parquet': '2f1a580bb6bb3689bf39b52ddbf5a7a443ea004e0d281274510f1193eb108a19'}
CUTOFF_MS = 1794178800000.0                               # 2026-11-08 23:00Z (pf_live_vs_replay.CUTOFF_MS)
RUNS = ('v6', 'basis')
CFG = [('M', 0.0, 'P1'), ('M', 50.0, 'P1'), ('M', 250.0, 'P1'), ('L', 50.0, 'P1'), ('U', 50.0, 'P1'), ('M', 50.0, 'P0')]
KEYS = ['mid', 'exec_s', 'first_recv_ns', 'side', 'ptick', 'shares', 'price_executed', 'taker_index', 'taker_quote']
QR_STAT = dict(untimed_legs=753, untimed_shares=9131.366331840098, legs_0_60=94783, shares_0_60=1074957.3571880765)
LIVE_SH = 849.867319
AGE_B = [('< 100 ms', -INF, 100.0), ('100-500 ms', 100.0, 500.0), ('>= 500 ms', 500.0, INF)]
MUT = dict(cap_off=False, act_plus=0.0, thr_off=False, reverse=False)   # planted-error switches (K5 only)


def sim_orders(side, tf, P5, S5, cache, ords, tt, tp, tq, qrule):
    """Replay given orders (dicts pt, ta, tr, size) against the trades under pf_review_queue_reach.sim's rule.
    Returns fills (order index, trade index, shares, through, age ms) and the queue ahead at activation per order."""
    for a in (tf, tt):
        if len(a) > 1 and (np.diff(a) < 0).any():
            raise ValueError('REFUSED: frames or trades not time-sorted')
    tfl = tf.tolist(); nf = len(tfl); ttl = tt.tolist()

    def lv(pt):                                           # as in QR.sim: shown size at pt per frame; nan beyond 5 levels
        if pt not in cache:
            last = P5[:, 4]
            cache[pt] = np.where((last >= 0) & ((pt < last) if side == 0 else (pt > last)), np.nan, (S5 * (P5 == pt)).sum(1))
        return cache[pt]
    st = []
    for n in sorted(range(len(ords)), key=lambda k: (ords[k]['ta'], k)):
        o = ords[n]; ta = o['ta'] + MUT['act_plus']; fi = bisect.bisect_right(tfl, ta) - 1; a = 0.0
        if fi >= 0:
            x = lv(o['pt'])[fi]; a = float(x) if np.isfinite(x) else 0.0
            if fi + 1 < nf and tfl[fi + 1] - ta <= 100.0:
                d2 = lv(o['pt'])[fi + 1]
                if np.isfinite(d2) and d2 > a:
                    a = float(d2)
        st.append(dict(n=n, pt=o['pt'], ta=ta, tr=o['tr'], rem=o['size'], ahead=0.0 if qrule == 'U' else a, ci=fi, a0=a))
    st.sort(key=lambda o: (o['ta'], o['n']))
    fills = []; act = []; ptr = 0
    for i, t in enumerate(ttl):
        while ptr < len(st) and st[ptr]['ta'] <= t:
            act.append(st[ptr]); ptr += 1
        if not act:
            continue
        p = tp[i]; left = tq[i]
        for o in (reversed(act) if MUT['reverse'] else act):
            if o['rem'] <= 1e-12 or t >= o['tr']:
                continue
            if qrule == 'M' and not MUT['cap_off']:
                j = bisect.bisect_left(tfl, t) - 1
                if j > o['ci']:
                    seg = lv(o['pt'])[o['ci'] + 1:j + 1]; seg = seg[np.isfinite(seg)]
                    if len(seg) and seg.min() < o['ahead']:
                        o['ahead'] = float(seg.min())
                    o['ci'] = j
            thr = ((p < o['pt']) if side == 0 else (p > o['pt'])) and not MUT['thr_off']
            if p == o['pt']:
                use = min(left, o['ahead']); o['ahead'] -= use; q = min(left - use, o['rem'])
            elif thr:
                q = min(left, o['rem']); o['ahead'] = 0.0
            else:
                q = 0.0
            if q > 1e-12:
                left -= q; o['rem'] -= q
                fills.append((o['n'], i, q, bool(thr), t - o['ta']))
        if len(act) > 8:
            act = [o for o in act if o['rem'] > 1e-12 and o['tr'] > t]
    return fills, {o['n']: o['a0'] for o in st}


def fuzz_case(rng, side):
    """A random window-side: frames with 5 levels a side, trades at, behind and through the touch, a guard path."""
    nf = int(rng.integers(3, 50)); t_end = float(rng.integers(800, 3000))
    tf = np.sort(rng.uniform(-150.0, t_end, nf)); tf[0] = -150.0
    b1 = np.clip(50 + np.cumsum(rng.integers(-1, 2, nf)), 6, 88); a1 = b1 + rng.integers(1, 4, nf)
    gb = np.cumsum(rng.integers(1, 3, (nf, 5)), 1) - 1; ga = np.cumsum(rng.integers(1, 3, (nf, 5)), 1) - 1
    PB = (b1[:, None] - gb).astype('int64'); PA = (a1[:, None] + ga).astype('int64')
    SB = rng.integers(0, 120, (nf, 5)).astype(float); SA = rng.integers(0, 120, (nf, 5)).astype(float)
    nt = int(rng.integers(0, 70)); tt = np.sort(rng.uniform(-100.0, t_end + 250.0, nt))
    fi = np.maximum(np.searchsorted(tf, tt, side='right') - 1, 0)
    tp = ((PB[fi, 0] - rng.integers(-1, 3, nt)) if side == 0 else (PA[fi, 0] + rng.integers(-1, 3, nt))).tolist()
    tq = rng.uniform(0.5, 40.0, nt).tolist()
    tau = np.arange(0.0, t_end, 50.0); v = np.cumsum(rng.normal(0.0, 0.3, len(tau)))
    s = (np.cumsum(rng.random(len(tau)) < 0.05) % 2 == 0)
    ki = np.searchsorted(tf, tau, side='right') - 1
    P5, S5, opp = (PB, SB, PA[:, 0]) if side == 0 else (PA, SA, PB[:, 0])
    best = np.where(ki >= 0, P5[np.maximum(ki, 0), 0], -1)
    return tf, opp, P5, S5, tau, v, s, best, tt, tp, tq, t_end


def fuzz(n_cases, seed=20261110):
    """K1a: number of differences between QR.sim and the order replay of QR.sim's own orders."""
    rng = np.random.default_rng(seed); diff = 0; n_fills = 0
    for c in range(n_cases):
        side = c % 2
        tf, opp, P5, S5, tau, v, s, best, tt, tp, tq, t_end = fuzz_case(rng, side)
        for qrule in 'MLU':
            F, O = QR.sim(side, tf, opp, P5, S5, {}, tau, v.tolist(), s.tolist(), 0.0, best.tolist(), tt, tp, tq, qrule, t_end)
            want = {(i, round(float(tt[i]) - age, 6)): (pt, round(q, 9), bool(th)) for i, pt, q, th, age, _ph, _a in F}
            filled = {}
            for i, pt, q, th, age, _ph, _a in F:
                k = round(float(tt[i]) - age, 6); filled[k] = filled.get(k, 0.0) + q
            ords = []; a0w = []
            for ta, gone, a0, ref in O:
                if ref:
                    continue
                k = int(round((ta - QR.L_PLACE) / 50.0))
                if abs(tau[k] + QR.L_PLACE - ta) > 1e-9:
                    diff += 1; continue
                full = filled.get(round(ta, 6), 0.0) >= QR.SIZE - 1e-9
                ords.append(dict(pt=int(best[k]), ta=ta, tr=gone + 1e-3 if full else gone, size=QR.SIZE)); a0w.append(a0)
            keep = tt <= t_end + 200.0; ttk = tt[keep]          # QR.sim applies no trade after t_end + 200 ms
            G, A0 = sim_orders(side, tf, P5, S5, {}, ords, ttk, [p for p, k_ in zip(tp, keep) if k_],
                               [q for q, k_ in zip(tq, keep) if k_], qrule)
            got = {(i, round(float(ttk[i]) - age, 6)): (ords[n]['pt'], round(q, 9), th) for n, i, q, th, age in G}
            diff += int(want != got) + sum(1 for n, a in enumerate(a0w) if abs(A0[n] - a) > 1e-9)
            n_fills += len(F)
    return diff, n_fills


def credit(fl, own, key='grp'):
    """REAL shares per replay fill. fl: replay fills (cid, row, grp, q) of one config, in time order; own: eligible own
    legs (cid, leg, shares, rows / grps = sets of candidate leg rows / their taker orders). key 'grp' (v2) matches on
    the taker order, 'row' (v1) on the maker leg row. Each own leg is credited up to its shares, in time order."""
    cap = {(r.cid, r.leg): r.shares for r in own.itertuples()}
    cand = {}
    for r in own.itertuples():
        for x in getattr(r, key + 's'):
            cand.setdefault((r.cid, x), []).append(r.leg)
    real = np.zeros(len(fl))
    for n, (c, x, q) in enumerate(zip(fl.cid.values, fl[key].values, fl.q.values)):
        for leg in cand.get((c, x), ()):
            u = min(q - real[n], cap[(c, leg)])
            if u > 1e-12:
                real[n] += u; cap[(c, leg)] -= u
    return real


def _one(ords, frames, trades, qrule, side=0):
    """Synthetic case: frames (t, [5 prices], [5 sizes]); trades (t, price, shares)."""
    tf = np.array([f[0] for f in frames], float); P5 = np.array([f[1] for f in frames], 'int64'); S5 = np.array([f[2] for f in frames], float)
    tt = np.array([x[0] for x in trades], float)
    return sim_orders(side, tf, P5, S5, {}, ords, tt, [x[1] for x in trades], [float(x[2]) for x in trades], qrule)


def synth():
    """Known-answer cases for the order replay and the credit rule (K3, K4, S1, S3, S4); returns failed case names."""
    bad = []; L5 = [50, 49, 48, 47, 46]; Z5 = [0, 0, 0, 0, 0]
    o = lambda pt, ta, tr, size=5.0: dict(pt=pt, ta=ta, tr=tr, size=size)
    F, _ = _one([o(50, 100, 1000), o(50, 120, 1000)], [(-100, L5, Z5)], [(200, 50, 5)], 'M')
    if [(n, i, q) for n, i, q, _t, _a in F] != [(0, 0, 5.0)]:
        bad.append('S1 service order')
    F, _ = _one([o(50, 100, 150)], [(-100, L5, Z5)], [(149, 50, 5), (200, 50, 5)], 'M')
    if [(n, i, q) for n, i, q, _t, _a in F] != [(0, 0, 5.0)]:
        bad.append('S3 removal')
    fr = [(-100, L5, [30, 0, 0, 0, 0]), (160, L5, [50, 0, 0, 0, 0])]; tr = [(200, 50, 40), (300, 50, 20)]
    want = {'M': [(0, 1, 5.0)], 'L': [(0, 1, 5.0)], 'U': [(0, 0, 5.0)]}
    for q in 'MLU':
        F, A0 = _one([o(50, 100, 1000)], fr, tr, q)
        if [(n, i, x) for n, i, x, _t, _a in F] != want[q] or A0[0] != 50.0:
            bad.append(f'S4 {q}')
    fr2 = [(-100, L5, [30, 0, 0, 0, 0]), (150, L5, [10, 0, 0, 0, 0])]        # the frame cap lowers the queue ahead (M only)
    for q, w in (('M', [(0, 0, 2.0)]), ('L', [])):        # M: ahead 30 -> 10 by the frame at 150; 12 sh -> 2 to us
        F, _ = _one([o(50, 100, 1000)], fr2, [(200, 50, 12)], q)
        if [(n, i, x) for n, i, x, _t, _a in F] != w:
            bad.append(f'S4 cap {q}')
    F, _ = _one([o(50, 100, 1000)], [(-100, L5, [99, 0, 0, 0, 0])], [(200, 49, 3)], 'M')   # through our price: filled
    if [(n, i, x, t) for n, i, x, t, _a in F] != [(0, 0, 3.0, True)]:
        bad.append('S5 through')
    F, _ = _one([o(50, 100, 1000, 10.0)], [(-100, L5, Z5)], [(50, 50, 5), (200, 50, 5), (300, 50, 5)], 'M')
    fl = pd.DataFrame(dict(cid=[7] * len(F), row=[i for _n, i, _q, _t, _a in F], q=[q for _n, _i, q, _t, _a in F]))
    fl['grp'] = fl.row                                    # each synthetic trade is its own taker order
    own = pd.DataFrame(dict(cid=[7, 7], leg=[0, 1], shares=[5.0, 5.0], rows=[{0}, {1}], grps=[{0}, {1}], anc=[50.0, 200.0]))
    real = credit(fl, own); before = int((own.anc < 100.0).sum())
    if list(fl.row) != [1, 2] or list(real) != [5.0, 0.0] or real.sum() / own.shares.sum() != 0.5 or real.sum() / fl.q.sum() != 0.5 or before != 1:
        bad.append('K3 planted credit')
    for name, args in (('K4 unsorted trades', ([o(50, 100, 1000)], [(-100, L5, Z5)], [(300, 50, 5), (200, 50, 5)], 'M')),
                       ('K4 unsorted frames', ([o(50, 100, 1000)], [(100, L5, Z5), (-100, L5, Z5)], [(200, 50, 5)], 'M'))):
        try:
            _one(*args); bad.append(name)
        except ValueError:
            pass
    try:
        population(pd.DataFrame(dict(t_Accepted=[1.0], refused=[None], market=[1], t_req=[CUTOFF_MS], lane=[0], filled=[0.0],
                                     size=[5.0], t_Filled_last=[np.nan], t_CancelDone=[np.nan], t_cancel=[np.nan])), {1: 0.0}, 'x')
        bad.append('K4 cutoff')
    except ValueError:
        pass
    return bad


def population(K, W, run):
    """Accepted live orders of one run in the cohort windows, with both paths. Refuses an order decided at the cutoff."""
    a = K[K.t_Accepted.notna() & K.refused.isna() & K.market.isin(list(W))].copy()
    if len(a) and float(a.t_req.max()) >= CUTOFF_MS:
        raise ValueError('REFUSED: an order decided at or after the cutoff')
    a['side'] = (a.lane % 2).astype(int); a['ws'] = a.market.map(W).astype(float); a['run'] = run
    full = ((a.filled >= a['size'] - 1e-6) & a.t_Filled_last.notna()).values; end = (a.ws + 300000.0).values
    for P, t0, tc in (('P1', 't_Accepted', 't_CancelDone'), ('P0', 't_req', 't_cancel')):
        tr = a[tc].fillna(pd.Series(end, index=a.index)).values
        tr = np.where(full, np.minimum(tr, a.t_Filled_last.values), tr)
        a[f'ta_{P}'] = a[t0].values; a[f'tr_{P}'] = np.minimum(tr, end)
    return a


def controls_synthetic():
    bad = synth(); d0, nf = fuzz(2000); mut = {}
    for k, val in (('cap_off', True), ('act_plus', 10.0), ('thr_off', True), ('reverse', True)):
        old = MUT[k]; MUT[k] = val
        try:
            mut[k] = dict(fuzz_diffs=fuzz(400)[0], synth_failed=len(synth()))
        finally:
            MUT[k] = old
    ok = not bad and d0 == 0 and nf > 0 and all(v['fuzz_diffs'] > 0 or v['synth_failed'] > 0 for v in mut.values()) and not synth() and fuzz(50)[0] == 0
    return ok, dict(synth_failed=bad, fuzz_cases=2000, fuzz_fills=nf, fuzz_diffs=d0, planted=mut)


def wq(x, w, q):
    """Weighted quantile."""
    o = np.argsort(x); c = np.cumsum(w[o]) / w.sum()
    return float(x[o][min(np.searchsorted(c, q), len(x) - 1)])


def label(c):
    r, s, P = CFG[c]
    return f'{r} {int(s)} {P}'


def main():
    if '--selftest' in sys.argv:
        ok, rep = controls_synthetic(); print(json.dumps(rep, indent=1)); sys.exit(0 if ok else 1)
    out = sys.argv[1]
    ok, ksyn = controls_synthetic()
    if not ok:
        sys.exit(f'CONTROL FAILED -> no table: {ksyn}')
    sc = json.load(open(f'{RP}/scope.json')); W0 = {int(w['market']): w['ws'] * 1000.0 for w in sc['windows']}; days = len(W0) / 288.0
    k2 = dict(hashes_ok=all(hashlib.sha256(open(f'{FC}/{f}', 'rb').read()).hexdigest() == h for f, h in FC_SHA.items()))
    core = pd.concat([population(pd.read_parquet(f'{LV}/core_orders_{r}.parquet'), W0, r) for r in RUNS], ignore_index=True)
    k2['side_convention'] = bool(((core.lane % 2 == 0) == core.up).all())
    k2['live_shares'] = float(core.filled.sum()); k2['live_shares_ok'] = abs(k2['live_shares'] - LIVE_SH) < 1e-3

    LTF = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=KEYS + ['t_ms', 'how', 't_e2'])
    tk = pd.read_parquet(f'{S1}/legs_tk.parquet', columns=['tk']).tk.values
    if len(tk) != len(LTF):
        sys.exit('REFUSED: legs_tk not aligned with legs_timed2')
    grp = np.where(tk >= 0, tk, -1 - np.arange(len(LTF)))   # as pf_review_queue_reach: one time per taker order
    ws_t = np.where(LTF.how.values > 0, LTF.t_ms.values.astype(float), np.nan)
    e2_t = np.where((LTF.how.values == 0) & np.isfinite(LTF.t_e2.values), LTF.t_e2.values, np.nan)
    a_ = pd.Series(ws_t).groupby(grp, sort=False).transform('min').values
    b_ = pd.Series(e2_t).groupby(grp, sort=False).transform('min').values
    LTF['anc'] = np.where(np.isfinite(a_), a_, b_); LTF['row'] = np.arange(len(LTF)); LTF['grp'] = grp
    L0 = LTF[LTF.mid.isin(W0)].copy(); L0['ws'] = L0.mid.map(W0)
    unt = L0[~np.isfinite(L0.anc.values) & (L0.exec_s * 1000.0 >= L0.ws) & (L0.exec_s * 1000.0 < L0.ws + 60000.0)]
    T0 = L0[np.isfinite(L0.anc.values)]; in60 = T0[(T0.anc >= T0.ws) & (T0.anc < T0.ws + 60000.0)]
    k1b = dict(untimed_legs=len(unt), untimed_shares=float(unt.shares.sum()), legs_0_60=len(in60), shares_0_60=float(in60.shares.sum()))
    k1b_ok = all(abs(k1b[k] - v) <= 1e-6 for k, v in QR_STAT.items())
    uf = L0[~np.isfinite(L0.anc.values) & (L0.exec_s * 1000.0 >= L0.ws) & (L0.exec_s * 1000.0 < L0.ws + 300000.0)]
    LT = T0[(T0.anc >= T0.ws - 2000.0) & (T0.anc <= T0.ws + 302000.0)]
    LI = LT.groupby(['mid', 'side'], sort=False).indices

    CC = pd.read_parquet(f'{FC}/clock_cases.parquet'); own = CC.reset_index().rename(columns={'index': 'leg'})
    j = own[['leg'] + KEYS].merge(LTF[KEYS + ['anc', 'row']], on=KEYS, how='left')
    g = j.groupby('leg').agg(n=('row', 'count'), amin=('anc', 'min'), amax=('anc', 'max'))
    g['rows'] = j.groupby('leg').row.apply(lambda x: set(int(v) for v in x.dropna()))
    g['grps'] = g.rows.apply(lambda s: set(int(grp[v]) for v in s))
    own = own.join(g, on='leg').merge(core[['cid', 'side', 'tick', 't_req', 't_Accepted', 't_cancel', 'tr_P1', 'ta_P1', 'run']]
                                      .rename(columns={'side': 'o_side', 'run': 'o_run'}), on='cid', how='left')
    k2.update(own_legs=len(own), own_shares=float(own.shares.sum()), own_with_rows=int((own.n > 0).sum()),
              own_with_order=int(own.o_side.notna().sum()), own_side_price_ok=bool(((own.side == own.o_side) & (own.ptick == own.tick)).all()))
    timed = np.isfinite(own.amin.values) & (own.amin.values == own.amax.values)
    own['status'] = np.where(~own.leg_clock_status.isin(['unique_leg', 'identical_clock_candidates']), 'ambiguous clock (audit)',
                             np.where(~timed, 'untimed in the replay', 'eligible'))
    E = own[own.status == 'eligible'].copy(); E['anc'] = E.amin
    k2['eligible_legs'] = len(E); k2['anc_eq_Z'] = int(((E.anc - E.Z).abs() <= 1.0).sum())
    k2_ok = (k2['hashes_ok'] and k2['side_convention'] and k2['live_shares_ok'] and k2['own_with_rows'] == len(own)
             and k2['own_with_order'] == len(own) and k2['own_side_price_ok'] and k2['anc_eq_Z'] == len(E))
    if not (k1b_ok and k2_ok):
        sys.exit(f'CONTROL FAILED -> no table: K1b {k1b} ok={k1b_ok}; K2 {k2}')
    AMB = {(int(r.cid), x) for r in own[own.status != 'eligible'].itertuples() for x in r.grps}

    cols = ['mid', 't_ms'] + [f'{s}{i}_{x}' for s in 'ba' for i in range(1, 6) for x in 'ps']
    parts = []
    for fn in sorted(glob.glob(f'{S1}/books/part_*.parquet')):
        d = pd.read_parquet(fn, columns=cols); d = d[d.mid.isin(W0)]
        rel = d.t_ms.values - d.mid.map(W0).values
        parts.append(d[(rel >= -5000.0) & (rel <= 305000.0)])
    B = pd.concat(parts, ignore_index=True); del parts
    BI = B.groupby('mid', sort=False).indices; OG = core.groupby(['market', 'side'], sort=False).indices
    stat = dict(no_frame_windows=0, untimed_legs_0_300=int(len(uf)), untimed_shares_0_300=float(uf.shares.sum()),
                no_rest={P: int((core[f'tr_{P}'] <= core[f'ta_{P}']).sum()) for P in ('P1', 'P0')},
                own_status=own.status.value_counts().to_dict(), own_status_shares=own.groupby('status').shares.sum().to_dict(),
                tokyo5='excluded: no research tape for 11-07 in this cohort (not read)')
    rows = []; qrows = []; E['mo1'] = np.nan; E['mo5'] = np.nan
    for m in sorted(W0, key=W0.get):
        if m not in BI:
            stat['no_frame_windows'] += 1; continue
        b = B.iloc[BI[m]].sort_values('t_ms', kind='stable'); tf = b.t_ms.values.astype(float)
        bb = b.b1_p.values.astype('int64'); ba = b.a1_p.values.astype('int64')
        mff = pd.Series(np.where((bb >= 0) & (ba > bb), (ba + bb) / 2.0, np.nan)).ffill().values

        def mid_at(t):
            i = np.searchsorted(tf, t, side='right') - 1
            return np.where(i >= 0, mff[np.maximum(i, 0)], np.nan)
        lev = {0: (b[[f'b{i}_p' for i in range(1, 6)]].values.astype('int64'), b[[f'b{i}_s' for i in range(1, 6)]].values.astype(float)),
               1: (b[[f'a{i}_p' for i in range(1, 6)]].values.astype('int64'), b[[f'a{i}_s' for i in range(1, 6)]].values.astype(float))}
        for side in (0, 1):
            sg = 1.0 if side == 0 else -1.0
            ke = (E.o_run.notna()) & (E.cid.isin(core.cid[(core.market == m)])) & (E.side == side)
            if ke.any():
                E.loc[ke, 'mo1'] = sg * (mid_at(E.anc[ke].values + 1000.0) - E.ptick[ke].values)
                E.loc[ke, 'mo5'] = sg * (mid_at(E.anc[ke].values + 5000.0) - E.ptick[ke].values)
            if (m, side) not in OG:
                continue
            O = core.iloc[OG[(m, side)]]
            if (m, side) in LI:
                l = LT.iloc[LI[(m, side)]]
                o_ = np.lexsort(((-l.ptick.values if side == 0 else l.ptick.values), l.anc.values))
                anc = l.anc.values[o_]; pt = l.ptick.values[o_].astype('int64'); sh = l.shares.values[o_].astype(float); rw = l.row.values[o_]
                gw = l.grp.values[o_]
            else:
                anc = np.zeros(0); pt = np.zeros(0, 'int64'); sh = np.zeros(0); rw = np.zeros(0, 'int64'); gw = np.zeros(0, 'int64')
            ptl = pt.tolist(); shl = sh.tolist(); cache = {}; mid1 = mid_at(anc + 1000.0); mid5 = mid_at(anc + 5000.0)
            for c, (rule, shf, P) in enumerate(CFG):
                OO = O[(O[f'tr_{P}'] > O[f'ta_{P}']).values]
                ords = [dict(pt=int(x), ta=float(a), tr=float(r_), size=float(s_)) for x, a, r_, s_ in
                        zip(OO.tick.values, OO[f'ta_{P}'].values, OO[f'tr_{P}'].values, OO['size'].values)]
                F, A0 = sim_orders(side, tf, lev[side][0], lev[side][1], cache, ords, anc - shf, ptl, shl, rule)
                cid_ = OO.cid.values; tc_ = OO.t_cancel.values; tk_ = OO.tick.values; run_ = OO.run.values
                for n, i, q, th, age in F:
                    rows.append((c, run_[n], m, side, int(cid_[n]), int(rw[i]), int(gw[i]), q, th, age, bool(anc[i] - shf >= tc_[n]),
                                 sg * (mid1[i] - tk_[n]), sg * (mid5[i] - tk_[n]), float(anc[i])))
                if c == 0:
                    for n, (x, a, r_) in enumerate(zip(OO.tick.values, OO.ta_P1.values, OO.tr_P1.values)):
                        k = (pt == x) & (anc >= a) & (anc < r_)
                        qrows.append((int(cid_[n]), float(sh[k].sum()), A0[n], float(r_ - a), float(OO.filled.values[n])))
    FL = pd.DataFrame(rows, columns=['c', 'run', 'mid', 'side', 'cid', 'row', 'grp', 'q', 'thr', 'age', 'pend', 'mo1', 'mo5', 'anc'])
    QT = pd.DataFrame(qrows, columns=['cid', 'V', 'a0', 'rest', 'live_filled'])
    f0 = FL[FL.c == 0]; a0 = QT.set_index('cid').a0; dec = []            # v2, descriptive: own legs at M 0 P1
    for r in E.merge(core[['cid', 'size', 'market']], on='cid').itertuples():
        ff = f0[f0.cid == r.cid]
        if r.anc < r.ta_P1:
            c = 'before activation'
        elif r.anc >= r.tr_P1:
            c = 'after the order left'
        elif ff.grp.isin(r.grps).any():
            c = 'credited (same taker order)'
        elif ff.q[ff.anc < r.anc].sum() >= r.size - 1e-9:
            c = 'order filled earlier by other trades'
        else:
            V = float(LT.shares[(LT.mid == r.market) & (LT.side == r.side) & (LT.ptick == r.ptick) & (LT.anc >= r.ta_P1) & (LT.anc <= r.anc)].sum())
            c = ('not credited; volume at our price >= queue ahead' if V >= a0.get(r.cid, np.nan)
                 else 'not credited; volume at our price < queue ahead')
        dec.append((c, float(r.shares)))
    D_ = pd.DataFrame(dec, columns=['cat', 'sh'])
    stat['own_legs_M0_P1'] = {k: dict(legs=int(len(v)), shares=float(v.sh.sum())) for k, v in D_.groupby('cat')}
    report(out, CFG, FL, QT, core, E, AMB, stat, dict(K1a_K3_K4_K5=ksyn, K1b=k1b, K2=k2), days)


def report(out, CFG, FL, QT, core, E, AMB, stat, controls, days):
    nan = float('nan'); res = {}; FB = {}
    for c in range(len(CFG)):
        rule, shf, P = CFG[c]
        f = FL[FL.c == c].sort_values(['cid', 'anc', 'row'], kind='stable').reset_index(drop=True)
        rep = core[core[f'tr_{P}'] > core[f'ta_{P}']]; live_sh = float(rep.filled.sum()); e = E[E.cid.isin(rep.cid)]
        f['real'] = credit(f, e) if len(f) else np.zeros(0)
        f['real_row'] = credit(f, e, 'row') if len(f) else np.zeros(0)
        f['amb'] = np.array([(a, b) in AMB for a, b in zip(f.cid.values, f.grp.values)], bool)
        x = f[~f.amb]; S = float(f.q.sum()); Sx = float(x.q.sum()); R = float(f.real.sum()); own_sh = float(e.shares.sum())
        ages = {}
        for lab, lo, hi in AGE_B:
            k = ((x.age >= lo) & (x.age < hi)).values; s_ = float(x.q.values[k].sum()); r_ = float(x.real.values[k].sum())
            ages[lab] = dict(shares=s_, real=r_, precision=r_ / s_ if s_ else nan, mo1=QR.wmean(x.mo1.values[k], x.q.values[k]))
        lf = rep.set_index('cid').filled; rf = f.groupby('cid').q.sum().reindex(lf.index).fillna(0.0); cells = {}
        for nm, kk in (('both', (lf > 1e-9) & (rf > 1e-9)), ('live only', (lf > 1e-9) & (rf <= 1e-9)),
                       ('replay only', (lf <= 1e-9) & (rf > 1e-9)), ('neither', (lf <= 1e-9) & (rf <= 1e-9))):
            cells[nm] = dict(orders=int(kk.sum()), live_sh=float(lf[kk].sum()), replay_sh=float(rf[kk].sum()))
        res[label(c)] = dict(orders=int(len(rep)), fills=int(len(f)), replay_sh=S, live_sh=live_sh, rho=S / live_sh, status_sh=S - Sx,
                             real_sh=R, real_row_sh_v1=float(f.real_row.sum()), own_eligible_legs=int(len(e)), own_eligible_sh=own_sh, recall=R / own_sh if own_sh else nan,
                             precision=R / Sx if Sx else nan, mo1=QR.wmean(f.mo1.values, f.q.values), mo5=QR.wmean(f.mo5.values, f.q.values),
                             mo1_real=QR.wmean(f.mo1.values, f.real.values), mo1_other=QR.wmean(f.mo1.values, (f.q - f.real).values),
                             usd1_day=float(np.nansum(f.q.values * f.mo1.values)) / 100.0 / days,
                             thr_pct=100.0 * float(f.q[f.thr].sum()) / S if S else nan, ages=ages, cells=cells,
                             pend_replay_sh=float(x.q[x.pend].sum()), pend_own_sh=float(e.shares[(e.anc - shf) >= e.t_cancel].sum()))
        FB[c] = f
    w = E.shares.values; al = (E.anc - E.t_Accepted).values; U = (E.anc - E.t_req).values
    live = dict(legs=int(len(E)), shares=float(w.sum()), mo1=QR.wmean(E.mo1.values, w), mo5=QR.wmean(E.mo5.values, w),
                ages={lab: dict(shares=float(w[(al >= lo) & (al < hi)].sum()), mo1=QR.wmean(E.mo1.values[(al >= lo) & (al < hi)], w[(al >= lo) & (al < hi)]))
                      for lab, lo, hi in AGE_B})
    lag = dict(quantiles_ms={str(q): wq(U, w, q) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}, share_U_negative=float(w[U < 0].sum() / w.sum()),
               share_0_50=float(w[(U >= 0) & (U < 50)].sum() / w.sum()), share_0_250=float(w[(U >= 0) & (U < 250)].sum() / w.sum()))
    x = FB[1][~FB[1].amb].reset_index(drop=True); bk = np.select([x.age.values < 100.0, x.age.values < 500.0], [0, 1], 2)
    qv = x.q.values; rv = x.real.values

    def dstat(b_):
        y = b_ == 0; o = b_ == 2
        return float(rv[o].sum() / qv[o].sum() - rv[y].sum() / qv[y].sum()) if qv[o].sum() > 0 and qv[y].sum() > 0 else nan
    obs = dstat(bk); rng = np.random.default_rng(20261110); strata = list(x.groupby(['run', 'side']).indices.values()); nul = []
    for _ in range(500):
        b2 = bk.copy()
        for idx in strata:
            b2[idx] = rng.permutation(b2[idx])
        nul.append(dstat(b2))
    nul = np.array(nul)
    null = dict(draws=500, observed=obs, p95=float(np.nanquantile(nul, 0.95)), p99=float(np.nanquantile(nul, 0.99)),
                share_at_or_above=float(np.mean(nul >= obs)))
    q = QT[QT.rest >= 500.0].copy(); q['rep'] = q.cid.map(FB[1].groupby('cid').q.sum()).fillna(0.0); q['x'] = q.V - q.a0; qt = {}
    for lab, k in (('x <= 0', q.x <= 0), ('0 < x <= 25', (q.x > 0) & (q.x <= 25)), ('x > 25', q.x > 25)):
        qt[lab] = dict(orders=int(k.sum()), live_fill_rate=float((q.live_filled[k] > 1e-9).mean()) if k.any() else nan,
                       replay_fill_rate=float((q.rep[k] > 1e-9).mean()) if k.any() else nan)
    rd = dict(R24={lab: 'OVER-CREDITS' if v['rho'] > 1.25 else ('UNDER-CREDITS' if v['rho'] < 0.80 else 'MATCHED') for lab, v in res.items()},
              R25={lab: 'n/a' if v['own_eligible_legs'] < 30 else ('AGREES' if v['recall'] >= 0.6 and v['precision'] >= 0.6 else
                                                                   ('FAILS' if v['recall'] < 0.3 or v['precision'] < 0.3 else 'PARTIAL'))
                   for lab, v in res.items()})
    yo = []
    for c in (0, 1):
        a = res[label(c)]['ages']; y, o = a['< 100 ms'], a['>= 500 ms']
        yo.append(None if y['shares'] < 100 or o['shares'] < 100 else (y['precision'], o['precision']))
    rd['R26'] = 'n/a' if None in yo else ('YOUNG-PHANTOM' if all(y < 0.5 * o for y, o in yo) else ('YOUNG-REAL' if all(y >= o for y, o in yo) else 'MIXED'))
    r27 = [t for t, s_ in (('CONSTANT-250-CONTRADICTED', lag['share_0_250']), ('CONSTANT-50-CONTRADICTED', lag['share_0_50'])) if s_ >= 0.25]
    rd['R27'] = r27 if r27 else ['NOT-CONTRADICTED']
    rd['R28'] = {lab: 'n/a' if len(E) < 30 else ('REPLAY-OVERSTATES' if v['mo1'] - live['mo1'] > 0.5 else
                                                 ('REPLAY-UNDERSTATES' if v['mo1'] - live['mo1'] < -0.5 else 'MATCHES'))
                 for lab, v in res.items() if lab.endswith('P1')}
    rp, op = res['M 50 P1']['pend_replay_sh'], res['M 50 P1']['pend_own_sh']
    rd['R29'] = 'n/a' if rp < 20 and op < 20 else ('PEND-OVERCREDITED' if rp > 2 * op else ('PEND-UNDERCREDITED' if rp < 0.5 * op else 'PEND-MATCHED'))
    J = dict(as_of=pd.Timestamp.now(tz='UTC').isoformat(), windows=int(round(days * 288)), days=days,
             population={r: dict(orders=int((core.run == r).sum()), live_sh=float(core.filled[core.run == r].sum())) for r in RUNS},
             statuses=stat, controls=controls, configs=res, live_own=live, lag_bound=lag, null_R26=null, queue_table=qt, readings=rd)
    json.dump(J, open(out, 'w'), indent=1, default=float)
    f2 = lambda v: '   n/a' if v != v else f'{v:+6.2f}'
    print(f'B1a as of {J["as_of"]}: {J["windows"]} windows ({days:.3f} window-days), consumed day 2026-11-08; point estimates only')
    print(f'population: {len(core):,} accepted live orders, {float(core.filled.sum()):.3f} sh filled; statuses {json.dumps(stat, default=float)}')
    print(f'controls: K1a fuzz {controls["K1a_K3_K4_K5"]["fuzz_cases"]} cases, {controls["K1a_K3_K4_K5"]["fuzz_fills"]} fills, '
          f'{controls["K1a_K3_K4_K5"]["fuzz_diffs"]} differences; planted {json.dumps(controls["K1a_K3_K4_K5"]["planted"])}; K1b {controls["K1b"]}; K2 {controls["K2"]}')
    print(f'live own legs (eligible): {live["legs"]} legs, {live["shares"]:.1f} sh, c/sh 1 s {f2(live["mo1"])}, 5 s {f2(live["mo5"])}; '
          f'by age at P1: ' + '; '.join(f'{k} {v["shares"]:.1f} sh {f2(v["mo1"])}' for k, v in live['ages'].items()))
    print(f'lag bound U = anchor - decision (eligible own legs, share-weighted): ' + ', '.join(f'q{k} {v:.0f} ms' for k, v in lag['quantiles_ms'].items())
          + f'; U < 0 {100 * lag["share_U_negative"]:.1f}%, 0-50 ms {100 * lag["share_0_50"]:.1f}%, 0-250 ms {100 * lag["share_0_250"]:.1f}%')
    print('live own legs at M 0 P1 by outcome in the replay (legs / sh): ' + '; '.join(f'{k} {v["legs"]} / {v["shares"]:.1f}'
                                                                                 for k, v in stat['own_legs_M0_P1'].items()))
    print('config        orders  fills  replay sh  rho   status  REAL sh (v1 row)  recall  precision  c/sh 1s  5s     REAL  other  $/day1s  thr%  pend rep/own')
    for lab, v in res.items():
        print(f'{lab:12s} {v["orders"]:7,d} {v["fills"]:6,d} {v["replay_sh"]:10.1f} {v["rho"]:5.2f} {v["status_sh"]:7.1f} {v["real_sh"]:8.1f} ({v["real_row_sh_v1"]:5.1f}) '
              f'{v["recall"]:7.3f} {v["precision"]:9.3f}  {f2(v["mo1"])} {f2(v["mo5"])} {f2(v["mo1_real"])} {f2(v["mo1_other"])} {v["usd1_day"]:8.2f} '
              f'{v["thr_pct"]:5.1f} {v["pend_replay_sh"]:6.1f}/{v["pend_own_sh"]:.1f}')
    print('by age at the replay fill (shares | REAL | precision | c/sh 1 s):')
    for lab, v in res.items():
        print(f'  {lab:12s} ' + '; '.join(f'{k}: {a["shares"]:.1f} | {a["real"]:.1f} | {a["precision"]:.3f} | {f2(a["mo1"])}' for k, a in v['ages'].items()))
    print('orders, live vs replay (orders, live sh, replay sh):')
    for lab, v in res.items():
        print(f'  {lab:12s} ' + '; '.join(f'{k} {c_["orders"]:,} / {c_["live_sh"]:.1f} / {c_["replay_sh"]:.1f}' for k, c_ in v['cells'].items()))
    print(f'R26 null (M 50 P1, 500 draws within run x side): observed {obs:+.3f}, null p95 {null["p95"]:+.3f}, p99 {null["p99"]:+.3f}, '
          f'share >= observed {null["share_at_or_above"]:.3f}')
    print('queue table (P1, shift 0, orders resting >= 500 ms; x = volume at our price while resting - queue ahead at activation): '
          + '; '.join(f'{k}: {v["orders"]:,} orders, live {v["live_fill_rate"]:.3f}, replay M 50 {v["replay_fill_rate"]:.3f}' for k, v in qt.items()))
    print('readings: ' + json.dumps(rd))


if __name__ == '__main__':
    main()
