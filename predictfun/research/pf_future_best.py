"""FUTURE-BEST (2026-11-13, session ce68ebe0). USER: "Pull the risk ones, and keep our protection side at the future best price, it should be
ideally much better. Yes two sides will be very different. It looks the repost side they are similar to us".
After PULL-DEPTH (bc4aa5a2): pulling only the rungs the danger move is predicted to reach beat pulling every rung on WS-timed trades (PD10
+$23, PD25 +$44 a day at M 50), but that rule pulls the price the move reaches, so the kept rung sits one price BEHIND the hit side's new
best, and an order the market comes to while the guard is still on (d = 0) is pulled: we still joined the new best ~320 ms after it formed
(-0.17 to -0.22 c/sh) while the top 5 earned +0.30 c/sh there.

The USER's rule, on the side the guard protects (each side's own guard; the other side's guard is mostly off, so it stays as LDG):
  FB10 / FB25  while the guard is on, a resting order d ticks behind the known best is pulled if the move is likely to go THROUGH its
               price: P(mv >= d + 1) >= tau (PULL-DEPTH: P(mv >= d)), so the order at the predicted landing price (the future best) stays
               in its queue. d + 1 > 5 uses P_5 (overstates the risk: more pulls). An order placed behind the best (rungs 1-5) that the
               market has come to (d = 0) stays unless P(mv >= 1) >= tau; rung 0's orders (placed at the best) at d = 0, and any d < 0,
               are pulled as today. Entries stay blocked while the guard is on (as LDG).
  P            PULL-DEPTH's leave-one-day-out P(mv >= j within 1 s), j = 1..5 (pred_<day>, each day's model fitted without it; unchanged);
               tau = 0.10 / 0.25 as PULL-DEPTH (no new choice).
  OFB          the same rule with the realized move (PULL-DEPTH's mv1000 rows) instead of P: an UPPER BOUND with perfect foresight of the
               depth (uses the future; not a candidate).
  references   LDG (every rung pulled on the guard), PD10 / PD25 (PULL-DEPTH's rule), recomputed.
9 consumed replay days (10-23..27, 10-31, 11-01..03), first 60 s; the guard ladder (30 sh, rungs d = 0..5, S4's rest lane with idle-only
triggers and a decision callback, as PULL-DEPTH). Fill clocks M 0 / M 50 / M 250. Values: c/sh at 1 s / 5 s from the trade's anchor; $/day
by order age (< 100 / 100-500 / >= 500 ms) and trade timing (C12: WS = the taker order has a WS-timed leg; E2 = estimated); cancel-pending
shares; per at-best fill whether our order was at the price before it became the best (PRE).
New-best levels (PULL-DEPTH's events, unchanged): adverse = the first level that becomes the guarded side's best by fallback in
(onset - 100 ms, onset + 1 s] after a guard onset; calm fallback = fallback births with no onset on either side in [birth - 1 s,
birth + 100 ms). Our M 50 WS fills there while the level is the best (<= 2 s): shares, c/sh, STOP, and PRE = our order live before the birth.

Readings DECLARED BEFORE THE FIRST RUN (evaluated in code; day rule >= 7 of 9; a day with < 30 fills in a compared cell does not count):
 R500 FB-VS-PD (FB10 vs PD10, FB25 vs PD25, each): $/day w/o young (order >= 100 ms) at 1 s on WS-timed trades, higher at M 0 AND M 50 on
      >= 7 days -> BETTER; lower at both on >= 7 -> WORSE; else NOT-SHOWN. M 250 printed, with whether the pooled sign holds there (C2).
 R501 FB-VS-LDG (FB10, FB25): the same against LDG -> HELPS / HURTS / NOT-SHOWN.
 R502 NEW-BEST-FRONT (FB10 vs PD10, FB25 vs PD25): at adverse levels, our aged M 50 WS c/sh at 1 s, FB minus PD > 0 on >= 7 days ->
      FRONT-PAYS; < 0 on >= 7 -> FRONT-DOES-NOT-PAY; else NOT-SHOWN. Shares, $/day there, STOP and PRE shares printed; calm fallback too.
 R503 ALL-TRADES (C12, printed): all-trade $/day at M 50 (WS + E2), FB minus PD and minus LDG, pooled and days higher.
 OFB is printed beside FB as the upper bound; no reading.
What the outcomes mean, fixed in advance: R500 BETTER = keeping the order at the predicted landing price beats keeping it one price behind
(the USER's expectation); WORSE = the move goes past the prediction often enough that orders at the landing price are run over; R502
FRONT-PAYS = being in the new best's queue before it forms pays at that level; OFB minus FB = what a better depth prediction could add.
A pass is a candidate on consumed days, not a result (C8).
Controls (any failure -> no output): (i) FB with tau = 0 (the callback engaged, every order pulled) equals LDG fill for fill and order for
order on every window-side, rung and clock (off-parity through the override); FB25 differs from PD25 on some window-sides; (ii) LDG, PD10,
PD25 equal PULL-DEPTH's fills row for row (M 0 / M 50, every column but its P tercile); (iii) P and the realized move aligned with
PULL-DEPTH's rows (same window, side, step); (iv) synthetic cases (bid and ask, the arrival rule for rung 0 and deeper rungs, the oracle,
guard off, missing prediction, tau = 0) with three planted errors (the +1 shift dropped, the arrival exemption given to rung 0, the oracle's
sign flipped) that must each fail a case, and PULL-DEPTH's own self-test; (v) memory self-guard.
Limits fixed in advance: replay rule M (B1a; C5), replay volumes inflated (C4); consumed days, exploratory (C8); tau from PULL-DEPTH; the
rungs are independent orders; the lane is the guard ladder, not the frozen pm_ladder v3 (C11); no order-level validation (C10).
Usage (run_light, PF_SLOT=claude): pf_future_best.py selftest | eval <day> | report | posthoc (added after the report); driver run_future_best.sh
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
import pf_qr_queue_race as QR                             # noqa: E402  (book runs)
import pf_queue_place as QP                               # noqa: E402  (run_at)
import pf_timing_split as TS                              # noqa: E402  (WS / E2 flag)
import pf_pull_depth as PD                                # noqa: E402  (P, the PD rule, events, reference fills)

DAYS9 = K.DAYS9
OUTD = f'{K.IO}/future_best_20261113'
PDD = PD.OUTD
JS = PD.JS
TAUS = {'PD10': 0.10, 'PD25': 0.25, 'FB10': 0.10, 'FB25': 0.25}
ARMS = ('LDG', 'PD10', 'PD25', 'FB10', 'FB25', 'OFB')
CFGS = [('M', 0.0), ('M', 50.0), ('M', 250.0)]
SIZE = 30.0; RUNGS = range(6)
LEVEL_MS = 2000.0; MIN_FILLS = 30
AGEB = (('< 100 ms', -1.0, 100.0), ('100-500 ms', 100.0, 500.0), ('>= 500 ms', 500.0, 1e18))
INF = float('inf'); NAN = float('nan')
MUT_NO_SHIFT = MUT_ARRIVE_R0 = MUT_ORACLE_SIGN = 0        # planted errors (self-test only)
MEM_CAP_MIB = 2048.0
FCOLS = ['day', 'cfg', 'arm', 'rung', 'src', 'depth', 'thr', 'young', 'pend', 'q', 'v1', 'v5', 'pre', 'type', 'age']


def fb_dec(vg, best, side, Pm, offset, tau=None, mv=None):
    """While the guard is on: pull a resting order if the move is likely to go through its price, P(mv >= d + 1) >= tau (oracle: the
    realized mv >= d + 1). d = 0 for an order placed behind the best (offset >= 1): the same test; d = 0 for rung 0, and d < 0: pull."""
    def dec(k, t, cur):
        if not vg[k]:
            return False
        b = best[k]
        if b < 0:
            return True
        d = (b - cur['pt']) if side == 0 else (cur['pt'] - b)
        if d < 0 or (d == 0 and offset == 0 and not MUT_ARRIVE_R0):
            return True
        if MUT_NO_SHIFT:
            if d == 0:
                return True
            j = d
        else:
            j = d + 1
        if mv is not None:
            m_ = mv[k]
            if not (m_ == m_):
                return True
            return bool(m_ < j) if MUT_ORACLE_SIGN else bool(m_ >= j)
        p = Pm[k, min(j, len(JS)) - 1]
        return bool((not (p == p)) or p >= tau)
    return dec


def selftest():
    bad = []
    c = lambda pt: dict(pt=pt)
    Pm = np.array([[0.9, 0.3, 0.05, 0.01, 0.0]])            # tau 0.25: the move likely reaches 48 (2 ticks), not 47 -> keep 48
    g = fb_dec([True], [50], 0, Pm, 2, tau=0.25)
    if [g(0, 0.0, c(p)) for p in (50, 49, 48, 47)] != [True, True, False, False]:
        bad.append(f'bid FB25 {[g(0, 0.0, c(p)) for p in (50, 49, 48, 47)]} (keep the landing price 48)')
    pd25 = PD.pd_dec([True], [50], 0, Pm, tau=0.25)
    if [pd25(0, 0.0, c(p)) for p in (50, 49, 48, 47)] != [True, True, True, False]:
        bad.append('PD25 reference')
    ga = fb_dec([True], [50], 1, Pm, 3, tau=0.25)
    if [ga(0, 0.0, c(p)) for p in (50, 51, 52, 53)] != [True, True, False, False]:
        bad.append('ask FB25')
    P2 = np.array([[0.2, 0.05, 0.01, 0.0, 0.0]])            # arrival: the market came to our order (d = 0), P(mv >= 1) < tau
    if fb_dec([True], [50], 0, P2, 1, tau=0.25)(0, 0.0, c(50)) is not False:
        bad.append('arrival: a rung-1 order at the new best must stay when P(mv >= 1) < tau')
    if fb_dec([True], [50], 0, P2, 0, tau=0.25)(0, 0.0, c(50)) is not True:
        bad.append('rung 0 at the best must be pulled (the guard as today)')
    if fb_dec([True], [50], 0, Pm, 1, tau=0.25)(0, 0.0, c(51)) is not True:
        bad.append('d < 0 must be pulled')
    mv = np.array([2.0])
    o = fb_dec([True], [50], 0, Pm, 3, mv=mv)
    if [o(0, 0.0, c(p)) for p in (50, 49, 48, 47)] != [True, True, False, False]:
        bad.append(f'oracle {[o(0, 0.0, c(p)) for p in (50, 49, 48, 47)]}')
    if fb_dec([True], [50], 0, Pm, 3, mv=np.array([np.nan]))(0, 0.0, c(47)) is not True:
        bad.append('oracle: missing move must pull')
    if fb_dec([False], [50], 0, Pm, 2, tau=0.25)(0, 0.0, c(48)) is not False:
        bad.append('guard off must keep')
    if fb_dec([True], [50], 0, np.array([[np.nan] * 5]), 2, tau=0.25)(0, 0.0, c(47)) is not True:
        bad.append('missing prediction must pull')
    P5 = np.array([[0.9, 0.9, 0.9, 0.9, 0.3]])
    if fb_dec([True], [50], 0, P5, 5, tau=0.25)(0, 0.0, c(45)) is not True:
        bad.append('d + 1 > 5 must use P_5')
    z = fb_dec([True], [50], 0, np.zeros((1, 5)), 4, tau=0.0)
    if not all(z(0, 0.0, c(p)) for p in (50, 49, 48, 47, 46, 45)):
        bad.append('tau = 0 must pull every order')
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_NO_SHIFT', 'MUT_ARRIVE_R0', 'MUT_ORACLE_SIGN'):
        globals()[nm] = 1
        try:
            out[nm] = bool(selftest())
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    PD.all_selftests()
    b = selftest(); m = mutation_check()
    if b or not all(m.values()):
        sys.exit(f'FUTURE-BEST SELFTEST FAILED -> no output: {b} planted errors caught {m}')
    return m


def same_rows(X, Y):
    """Row-for-row equality of two fill tables (floats bit-equal, NaN = NaN)."""
    if len(X) != len(Y):
        return False
    for c_ in X.columns:
        a, b = X[c_].to_numpy(), Y[c_].to_numpy()
        if a.dtype.kind == 'f' or b.dtype.kind == 'f':
            if not np.array_equal(a.astype(float), b.astype(float), equal_nan=True):
                return False
        elif not (a.astype(str) == b.astype(str)).all():
            return False
    return True


def eval_day(day):
    import pf_gap_top_makers as GP
    import pf_depth_timing as DT
    mut = all_selftests()
    T0 = time.time(); D = K.load_day(day); st = D['stat']; S1d = f'{K.IO}/step1/{day}'
    tkf = pd.read_parquet(f'{S1d}/legs_tk.parquet', columns=['tk']).tk.values
    LT = D['LT']; ix = LT.index.values
    grp = np.where(tkf[ix] >= 0, tkf[ix], -1 - ix)
    WS = pd.Series(TS.ws_groups(LT.how.values, grp), index=LT.index)
    GRP = pd.Series(grp, index=LT.index)
    TPG = pd.Series(LT.ptick.values).groupby(grp).agg(list).to_dict()
    PR = pd.read_parquet(f'{PDD}/pred_{day}.parquet'); RW = pd.read_parquet(f'{PDD}/rows_{day}.parquet', columns=['m', 'side', 'k', 'mv1000'])
    st.update(window_sides=0, ctrl_differ=0, fb25_differs=0, clock_ok=1, levfills=0,
              align_bad=int(len(PR) != len(RW) or not (PR[['m', 'side', 'k']].values == RW[['m', 'side', 'k']].values).all()))
    PRI = PR.groupby(['m', 'side']).indices
    EV = pd.read_parquet(f'{PDD}/events_{day}.parquet'); EV = EV[EV.kind.isin(['adverse', 'calm fallback'])]
    EVI = {}
    for e in EV.itertuples(index=False):
        m_ = int(e.eid.split('|')[1])
        EVI.setdefault((m_, int(e.side)), {}).setdefault(int(e.p), []).append((float(e.tb), float(e.tb) + min(float(e.life), LEVEL_MS), str(e.kind)))
    st['events'] = EV.kind.value_counts().to_dict()
    fills = []; lev = []

    def ttype(r_, x, p):
        return DT.trade_type(x, p, TPG[GRP.at[r_]]) if GRP.at[r_] in TPG else 'n/a'
    for w in K.window_sides(D):
        st['window_sides'] += 1; m, side, sg, n = w['m'], w['side'], w['sg'], w['n']
        tf, mff = w['tf_mff']; tfl = tf.tolist(); P5, opp = w['P5'], w['opp']
        Pm = np.full((n, len(JS)), np.nan); mv = np.full(n, np.nan)
        if (m, side) in PRI:
            ii = PRI[(m, side)]; kk = PR.k.values[ii]
            Pm[kk] = PR[[f'P{j}' for j in JS]].values[ii]; mv[kk] = RW.mv1000.values[ii]
        vg = (np.asarray(w['vkl'], float) <= K.GUARD_PULL).tolist()
        b0, valid, rs, kind = QR.runs(tf, P5, opp, side)
        if (m, side) in D['LI']:
            l = D['LT'].iloc[D['LI'][(m, side)]]
            o_ = np.lexsort(((-l.ptick.values if side == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
        else:
            lrow = np.zeros(0, 'int64')
        src = WS.reindex(lrow).values.astype(bool) if len(lrow) else np.zeros(0, bool)
        evs = EVI.get((m, side), {})
        val = lambda t, h, pt: sg * (mff[max(int(np.searchsorted(tf, t + h, side='right')) - 1, 0)] - pt)   # as PULL-DEPTH, bit for bit
        for q, shf in CFGS:
            tt = w['anc'] - shf; cfg = f'{q}{int(shf)}'
            for d in RUNGS:
                res = {'LDG': K.sim_arm(w, q, shf, 'rest', aux=vg, offset=d, size=SIZE)}
                Fc, Oc, _ = K.sim_arm(w, q, shf, 'rest', aux=vg, decide=fb_dec(vg, w['best'], side, Pm, d, tau=0.0), idle_only=1, offset=d, size=SIZE)
                st['ctrl_differ'] += int(Fc != res['LDG'][0] or Oc != res['LDG'][1])
                for arm in ARMS[1:]:
                    if arm.startswith('PD'):
                        dec = PD.pd_dec(vg, w['best'], side, Pm, tau=TAUS[arm])
                    elif arm == 'OFB':
                        dec = fb_dec(vg, w['best'], side, Pm, d, mv=mv)
                    else:
                        dec = fb_dec(vg, w['best'], side, Pm, d, tau=TAUS[arm])
                    res[arm] = K.sim_arm(w, q, shf, 'rest', aux=vg, decide=dec, idle_only=1, offset=d, size=SIZE)
                st['fb25_differs'] += int(res['FB25'][0] != res['PD25'][0] or res['FB25'][1] != res['PD25'][1])
                for arm, (F, O, pulls) in res.items():
                    st[f'pulls|{arm}|{cfg}'] = st.get(f'pulls|{arm}|{cfg}', 0) + int(pulls)
                    for f in F:
                        i, pt, qq, thr, age, ph = f[0], int(f[1]), f[2], f[3], f[4], f[5]
                        t = float(w['anc'][i])
                        if not (w['ws'] <= t < w['ws'] + 61000.0):
                            continue
                        d_ = GP.depth_of(tfl, P5, opp, side, t, pt)
                        pre = 'n/a'
                        if d_ == 0:
                            ru = QP.run_at(tfl, b0, valid, rs, kind, float(tt[i]), pt)
                            if ru is not None and ru[1] != 'censored':
                                pre = 'PRE' if float(tt[i]) - age < ru[0] else 'POST'
                        ty = 'THROUGH' if thr else (ttype(lrow[i], side, pt) if src[i] else 'n/a')
                        v1 = val(t, 1000.0, pt); v5 = val(t, 5000.0, pt)
                        fills.append((int(day), cfg, arm, d, 'WS' if src[i] else 'E2', -9 if d_ is None else d_, bool(thr), age < 100.0, ph == 'pend',
                                      float(qq), v1, v5, pre, ty, float(age)))
                        if cfg == 'M50' and src[i] and pt in evs:
                            for tb, tend, kev in evs[pt]:
                                if tb < t <= tend:
                                    lev.append((int(day), kev, arm, d, float(qq), v1, float(tt[i] - age - tb), ty, age >= 100.0, ph == 'pend'))
                                    break
        if (K.L_PLACE, K.L_CANCEL, K.L_PENDCANCEL) != (PD.BG.CLK0['L_PLACE'], PD.BG.CLK0['L_CANCEL'], PD.BG.CLK0['L_PENDCANCEL']):
            st['clock_ok'] = 0
        w.pop('cache', None)
        if st['window_sides'] % 80 == 0:
            r2 = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r2:.0f} MiB', flush=True)
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    F = pd.DataFrame(fills, columns=FCOLS)
    # control ii: the references equal PULL-DEPTH's fills row for row
    R0 = pd.read_parquet(f'{PDD}/fills_{day}.parquet')
    ref_bad = {}
    for arm in ('LDG', 'PD10', 'PD25'):
        for cfg in ('M0', 'M50'):
            X = F[(F.arm == arm) & (F.cfg == cfg)][FCOLS[:-1]].reset_index(drop=True)
            Y = R0[(R0.arm == arm) & (R0.cfg == cfg)][FCOLS[:-1]].reset_index(drop=True)
            if not same_rows(X, Y):
                ref_bad[f'{arm} {cfg}'] = (len(X), len(Y))
    st['ref_bad'] = ref_bad; st['fills'] = len(F); st['levfills'] = len(lev)
    if st['ctrl_differ'] or st['align_bad'] or ref_bad or not st['clock_ok'] or not st['fb25_differs']:
        sys.exit(f'FUTURE-BEST CONTROL FAILED on {day} -> no result: { {k_: v_ for k_, v_ in st.items() if not isinstance(v_, (list,))} }')
    st['planted_caught'] = mut
    os.makedirs(OUTD, exist_ok=True)
    F.to_parquet(f'{OUTD}/fills_{day}.parquet', index=False)
    pd.DataFrame(lev, columns=['day', 'kind', 'arm', 'rung', 'q', 'v1', 'gap', 'type', 'aged', 'pend']).to_parquet(f'{OUTD}/levfills_{day}.parquet', index=False)
    json.dump(dict(stat=st, days=len(D['W0']) / 288.0), open(f'{OUTD}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: {json.dumps({k_: v_ for k_, v_ in st.items() if not isinstance(v_, dict)}, default=float)} in {time.time() - T0:.0f} s', flush=True)


def report():
    stats = {d: json.load(open(f'{OUTD}/day_{d}.json')) for d in DAYS9}
    A = pd.concat([pd.read_parquet(f'{OUTD}/fills_{d}.parquet') for d in DAYS9], ignore_index=True)
    nbad = int((~np.isfinite(A.v1)).sum()); A = A[np.isfinite(A.v1)]
    LV = pd.concat([pd.read_parquet(f'{OUTD}/levfills_{d}.parquet') for d in DAYS9], ignore_index=True); LV = LV[np.isfinite(LV.v1)]
    dd = {d: stats[d]['days'] for d in DAYS9}; ndays = sum(dd.values())
    cps = lambda x: float((x.q * x.v1).sum() / x.q.sum()) if x.q.sum() else NAN
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

    def cell(arm, cfg, srcs=('WS',), d=None, h='v1', aged=True):
        x = A[(A.arm == arm) & (A.cfg == cfg) & A.src.isin(srcs)]
        if aged:
            x = x[~x.young]
        if d:
            x = x[x.day == int(d)]
        return float((x.q * x[h]).sum()) / 100.0 / (dd[d] if d else ndays)
    R = {}
    L = [f'FUTURE-BEST, 9 consumed replay days, first 60 s; as of {asof}; controls passed (tau = 0 = LDG on every window-side, LDG / PD10 / PD25 = '
         f'PULL-DEPTH row for row, P and the move aligned, planted errors caught); fills without a 1 s value (status, excluded): {nbad}', '',
         'PART 1: $/day at 1 s, w/o young (order >= 100 ms): WS-timed | E2 | all trades || WS all ages | WS 5 s | pulls a day | WS c/sh |'
         ' cancel-pending share of WS shares, its $/day']
    for cfg in ('M0', 'M50', 'M250'):
        for arm in ARMS:
            pl = sum(stats[d]['stat'].get(f'pulls|{arm}|{cfg}', 0) for d in DAYS9) / ndays
            x = A[(A.arm == arm) & (A.cfg == cfg) & (A.src == 'WS') & ~A.young]; xp = x[x.pend]
            L.append(f'  {cfg:<4} {arm:<4} {cell(arm, cfg):+8.1f} | {cell(arm, cfg, ("E2",)):+8.1f} | {cell(arm, cfg, ("WS", "E2")):+8.1f} ||'
                     f' {cell(arm, cfg, aged=False):+8.1f} | {cell(arm, cfg, h="v5"):+8.1f} | {pl:7.0f} | {cps(x):+.3f} |'
                     f' {xp.q.sum() / max(x.q.sum(), 1e-9):4.0%} {float((xp.q * xp.v1).sum()) / 100.0 / ndays:+7.1f}')
        L.append('')
    R['R500'] = {}; R['R501'] = {}
    for fb, ref, key in (('FB10', 'PD10', 'R500'), ('FB25', 'PD25', 'R500'), ('FB10', 'LDG', 'R501'), ('FB25', 'LDG', 'R501')):
        up = {c: sum(cell(fb, c, d=d) > cell(ref, c, d=d) for d in DAYS9) for c in ('M0', 'M50', 'M250')}
        dn = {c: sum(cell(fb, c, d=d) < cell(ref, c, d=d) for d in DAYS9) for c in ('M0', 'M50', 'M250')}
        good, badl = ('BETTER', 'WORSE') if key == 'R500' else ('HELPS', 'HURTS')
        lab = good if min(up['M0'], up['M50']) >= 7 else (badl if min(dn['M0'], dn['M50']) >= 7 else 'NOT-SHOWN')
        diff = {c: cell(fb, c) - cell(ref, c) for c in ('M0', 'M50', 'M250')}
        holds = (diff['M250'] > 0) == (diff['M50'] > 0)
        R[key][f'{fb} vs {ref}'] = dict(label=lab, up=up, down=dn, diff=diff, sign_holds_at_M250=bool(holds))
        L.append(f'  {key} {fb} vs {ref}: {lab} (WS $/day: M 0 {diff["M0"]:+.1f}, higher {up["M0"]}/9; M 50 {diff["M50"]:+.1f}, higher {up["M50"]}/9;'
                 f' M 250 {diff["M250"]:+.1f}, higher {up["M250"]}/9, sign {"holds" if holds else "FLIPS"})')
    L.append('  per day, WS $/day M 50: ' + ' | '.join(ARMS))
    for d in DAYS9:
        L.append(f'    {d} ' + ' | '.join(f'{cell(a_, "M50", d=d):+7.1f}' for a_ in ARMS))
    R['R503'] = {}
    for fb in ('FB10', 'FB25', 'OFB'):
        for ref in ('PD10' if fb == 'FB10' else 'PD25', 'LDG'):
            dif = cell(fb, 'M50', ('WS', 'E2')) - cell(ref, 'M50', ('WS', 'E2'))
            hi = sum(cell(fb, 'M50', ('WS', 'E2'), d=d) > cell(ref, 'M50', ('WS', 'E2'), d=d) for d in DAYS9)
            R['R503'][f'{fb} vs {ref}'] = dict(diff=dif, higher_days=hi)
            L.append(f'  R503 all trades M 50: {fb} minus {ref} {dif:+.1f} $/day, higher on {hi}/9')
    # C1 / C12: the age split by trade-anchor source
    L += ['', 'C1 / C12: $/day at 1 s by order age, M 50: WS (< 100 | 100-500 | >= 500 ms) || E2 (same)']
    for arm in ARMS:
        x = A[(A.arm == arm) & (A.cfg == 'M50')]
        parts = []
        for s_ in ('WS', 'E2'):
            y = x[x.src == s_]
            parts.append(' | '.join(f'{float((y.q * y.v1)[(y.age >= lo) & (y.age < hi)].sum()) / 100.0 / ndays:+8.1f}' for _, lo, hi in AGEB))
        L.append(f'  {arm:<4} {parts[0]} || {parts[1]}')
    # where it comes from
    L += ['', 'At the best vs behind (M 50, WS, w/o young): at-best $/day (PRE | POST) | behind $/day | at-best PRE share of shares | PRE c/sh |'
          ' PRE STOP | POST c/sh']
    for arm in ARMS:
        x = A[(A.arm == arm) & (A.cfg == 'M50') & (A.src == 'WS') & ~A.young]; ab = x[x.depth == 0]; pr = ab[ab.pre == 'PRE']; po = ab[ab.pre == 'POST']
        u = lambda y: float((y.q * y.v1).sum()) / 100.0 / ndays
        L.append(f'  {arm:<4} {u(ab):+7.1f} ({u(pr):+7.1f} | {u(po):+7.1f}) | {u(x[x.depth > 0]):+7.1f} | {pr.q.sum() / max(ab.q.sum(), 1e-9):5.1%} |'
                 f' {cps(pr):+.3f} | {pr.q[pr.type == "STOP"].sum() / max(pr.q.sum(), 1e-9):4.0%} | {cps(po):+.3f}')
    # new-best levels
    ev = {k_: int(sum(stats[d]['stat']['events'].get(k_, 0) for d in DAYS9)) for k_ in ('adverse', 'calm fallback')}
    L += ['', f'PART 2: our fills at the new best (M 50, WS; levels {ev}): arm | fills | shares | $/day aged | aged c/sh | STOP | PRE share'
          ' (our order live before the birth) | PRE c/sh | POST c/sh | our order live after the birth, ms median (POST)']
    for kev in ('adverse', 'calm fallback'):
        for arm in ARMS:
            y = LV[(LV.kind == kev) & (LV.arm == arm)]; ya = y[y.aged]; pr = ya[ya.gap < 0]; po = ya[ya.gap >= 0]
            L.append(f'  {kev:<13} {arm:<4} {len(y):>6} | {y.q.sum():>8.0f} | {float((ya.q * ya.v1).sum()) / 100.0 / ndays:+7.1f} | {cps(ya):+.3f} |'
                     f' {ya.q[ya.type == "STOP"].sum() / max(ya.q.sum(), 1e-9):4.0%} | {pr.q.sum() / max(ya.q.sum(), 1e-9):4.0%} | {cps(pr):+.3f} |'
                     f' {cps(po):+.3f} | {np.median(y.gap[y.gap >= 0]) if (y.gap >= 0).any() else NAN:6.0f}')
    R['R502'] = {}
    adv = LV[(LV.kind == 'adverse') & LV.aged]
    for fb, ref in (('FB10', 'PD10'), ('FB25', 'PD25')):
        up = dn = 0; per = []
        for d in DAYS9:
            a_ = adv[(adv.day == int(d)) & (adv.arm == fb)]; b_ = adv[(adv.day == int(d)) & (adv.arm == ref)]
            if len(a_) >= MIN_FILLS and len(b_) >= MIN_FILLS:
                up += int(cps(a_) > cps(b_)); dn += int(cps(a_) < cps(b_)); per.append(round(cps(a_) - cps(b_), 3))
        lab = 'FRONT-PAYS' if up >= 7 else ('FRONT-DOES-NOT-PAY' if dn >= 7 else 'NOT-SHOWN')
        R['R502'][f'{fb} vs {ref}'] = dict(label=lab, up=up, down=dn, fb=cps(adv[adv.arm == fb]), ref=cps(adv[adv.arm == ref]), per_day=per)
        L.append(f'  R502 {fb} vs {ref} at adverse levels: c/sh {R["R502"][f"{fb} vs {ref}"]["fb"]:+.3f} vs {R["R502"][f"{fb} vs {ref}"]["ref"]:+.3f}'
                 f' (higher {up}, lower {dn} of 9; per day {per}) -> {lab}')
    os.makedirs(OUTD, exist_ok=True)
    json.dump(R, open(f'{OUTD}/future_best.json', 'w'), indent=1, default=float)
    open(f'{OUTD}/future_best.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


def posthoc():
    """POST HOC (added after the report, descriptive, no readings): (a) C2 for PULL-DEPTH's arms, which were not run at M 250 there: PD10 /
    PD25 minus LDG, WS $/day w/o young, per clock and days higher; (b) our aged fills at adverse levels (M 50, WS) by trade type: share of
    shares and c/sh for STOP / THROUGH, PRE (order live before the birth) and POST."""
    stats = {d: json.load(open(f'{OUTD}/day_{d}.json')) for d in DAYS9}; dd = {d: stats[d]['days'] for d in DAYS9}; ndays = sum(dd.values())
    A = pd.concat([pd.read_parquet(f'{OUTD}/fills_{d}.parquet') for d in DAYS9], ignore_index=True); A = A[np.isfinite(A.v1) & ~A.young & (A.src == 'WS')]
    LV = pd.concat([pd.read_parquet(f'{OUTD}/levfills_{d}.parquet') for d in DAYS9], ignore_index=True); LV = LV[np.isfinite(LV.v1) & LV.aged]
    cps = lambda x: float((x.q * x.v1).sum() / x.q.sum()) if x.q.sum() else NAN
    usd = lambda x, d=None: float((x.q * x.v1).sum()) / 100.0 / (dd[d] if d else ndays)
    L = ['FUTURE-BEST post hoc (descriptive): (a) PULL-DEPTH arms minus LDG, WS $/day w/o young at 1 s, per clock (days higher of 9)']
    for arm in ('PD10', 'PD25', 'FB10', 'FB25'):
        parts = []
        for cfg in ('M0', 'M50', 'M250'):
            x = A[(A.arm == arm) & (A.cfg == cfg)]; y = A[(A.arm == 'LDG') & (A.cfg == cfg)]
            hi = sum(usd(x[x.day == int(d)], d) > usd(y[y.day == int(d)], d) for d in DAYS9)
            parts.append(f'{cfg} {usd(x) - usd(y):+8.1f} ({hi}/9)')
        L.append(f'  {arm:<4} ' + ' | '.join(parts))
    L += ['', '(b) adverse levels, M 50, WS, aged: arm | PRE: share of our shares there, STOP share, c/sh STOP / THROUGH | POST: the same']
    for arm in ARMS:
        y = LV[(LV.kind == 'adverse') & (LV.arm == arm)]; tot = max(y.q.sum(), 1e-9); parts = []
        for nm, z in (('PRE', y[y.gap < 0]), ('POST', y[y.gap >= 0])):
            st_ = z[z.type == 'STOP']; th = z[z.type == 'THROUGH']
            parts.append(f'{nm} {z.q.sum() / tot:4.0%}, STOP {st_.q.sum() / max(z.q.sum(), 1e-9):4.0%}, {cps(st_):+.3f} / {cps(th):+.3f}')
        L.append(f'  {arm:<4} ' + ' | '.join(parts))
    open(f'{OUTD}/posthoc.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        print(f'selftest passed; planted errors caught: {all_selftests()}')
    elif cmd == 'eval' and len(sys.argv) > 2:
        eval_day(sys.argv[2])
    elif cmd == 'report':
        report()
    elif cmd == 'posthoc':
        posthoc()
    else:
        sys.exit(__doc__)
