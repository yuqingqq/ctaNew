"""ORDER-HARM WHY (2026-11-14, session ce68ebe0), POST HOC and descriptive (chosen after seeing POLICY v2's NOT-ACCEPTED result; it changes
no verdict there). USER: "yes check why it suffers loss and performs worse. Check against top makers, is it timing issue, then check the
model why it picks the wrong time".

DATA: POLICY v2's five test days (10-27, 10-31, 11-01, 11-02, 11-03; consumed), each window's first 60 s, the headline cap hard 50, fill
clocks M 0 / 50 / 250 (a trade reaches our orders 0 / 50 / 250 ms before its anchor).
OURS (simulated): LDG, FB25C, OHQ0, OHQM re-run on POLICY v2's engine, folds and decision rules, with wrappers that only RECORD each
decision on the order (step, time, A_hat, keep). RECONCILIATION: for every day x clock x arm the fill rows equal POLICY v2's saved fills
(its 23 columns, row for row), else no output.
THEIRS (real): every maker leg on the public tape in the same window-sides' first 60 s; makers ranked per day by shares in t_rel 0-240 s
(TOP-COMPARE / B12: top 5, 6-20, rest); legs aligned to the simulator's (control); value sg x (mid(a + h) - price), the same mid series.
STATE OF A TRADE (one function for our fills and theirs, at the leg's anchor a, venue time):
  phase  our guard ladder's guard at the last 50 ms step <= a - 50 ms (TOP-COMPARE's lag; 'none' if that step is > 100 ms old; 'masked'
         at steps without a start price): ON by time since the episode onset (< 100 / 100-250 / 250-500 / 500-1000 / >= 1000 ms); OFF by
         time since the last clear (AFTER < 250 / AFTER 250-1000 ms; else CALM);
  move   dfv250 at a: the Binance-implied fair price's change over the last 250 ms, signed against the maker side (> 0 = against):
         unknown / <= 0 / 0-0.5 / 0.5-1.5 / > 1.5 c;
  depth  the fill price against the maker side's best in the frame strictly before a: at best / behind / inside / no frame.
TIMING CLASS OF OUR FILL (engine time t = a - clock; exclusive, in this order): PEND (our cancel was in flight); KEPT-ON (the guard on at
the last step before t, the order kept by >= 1 decision in this episode); OTHER-ON (the guard on, no decision: edge cases, counted);
KEPT-AFTER (guard off, the order kept through an earlier episode); REPOST (placed <= 100 ms after a guard episode cleared); CALM.
READINGS (computed; descriptive, no acceptance):
 W1 LOSS MAP: whole-portfolio $/day (each fill marked at the common 60.2 s mid, as R652) and $/day at 1 s, by timing class x trade
    source, per arm and clock; per clock, the class carrying the most negative OHQ0 - LDG and OHQ0 - FB25C difference.
 W2 TIMING vs TOP 5: per-share value at 1 s, legs anchored in the first 60 s, WS-timed (exact timing; all trades printed beside), ours
    without fills on orders < 100 ms old (C1). gap = ours - top 5 = MIX + WITHIN + RESIDUAL over the phase x move x depth states:
    MIX = sum_s (w_ours(s) - w_top5(s)) v_top5(s) (filled in different states), WITHIN = sum_s w_ours(s) (v_ours(s) - v_top5(s)) (same
    state, worse fills), RESIDUAL = sum over states the top 5 never fill of w_ours v_ours (exact). Per arm x clock: TIMING if
    MIX < 0 and |MIX| >= |WITHIN|; WITHIN-STATE if WITHIN < 0 and |WITHIN| > |MIX|; else NEITHER. Marginal tables by phase and by move.
 W3 THE MODEL'S KEEPS (OHQ0, OHQM): fills on KEPT orders (KEPT-ON, KEPT-AFTER) by what the model saw at its last keep decision before the
    fill: SEEN (dfv250 > 0.5 c against at that decision), LATER (<= 0.5 there, > 0.5 at the trade), NO-MOVE (<= 0.5 at both), UNKNOWN;
    $/day (60.2 s mark, 1 s), sh/day, mean A_hat, the time from that decision to the fill, size at better prices at that decision; per
    arm x clock the class with the largest loss at the 60.2 s mark among losing kept fills (computed).
 W4 THE MODEL'S VIEW (ORDER-HARM v3's test-day label rows and predictions, no replay): decisions by the move at the decision x rung group:
    keep share (A_hat > 0), mean A_hat, the realized target under the final policy (G) and the keep-all advantage (KAV - Pe);
    LABEL-OK if kept decisions' mean G > 0 on >= 4 of 5 days; MISSES-MOVES if kept decisions with dfv250 > 0.5 c have mean G < 0 on
    >= 4 of 5 days (both may hold; neither -> printed as such).
 W5 HORIZON (printed): c/sh at 1 s, 5 s and the 60.2 s mark per timing class (ours, M 50) and per phase (top 5).
CONTROLS (any failure -> no output): the reconciliation; legs aligned on every window-side; synthetic cases for the phase, move, class and
decomposition functions with planted errors (no lag; KEPT without a decision; PEND not first; MIX / WITHIN weights swapped); upstream
selftests (POLICY v2, TOP-COMPARE).
LIMITS: ours simulated (rule M unvalidated, B1a; volumes inflated, C4), theirs real (C9); consumed days (C8); first minute; their orders'
age and queue unknown (states only); fees and rebates not counted.
Usage (run_light, PF_SLOT=claude): pf_order_harm_why.py selftest | day <day> | report; driver run_order_harm_why.sh
"""
import bisect
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402
import pf_timing_split as TS                              # noqa: E402
import pf_gap_top_makers as GP                            # noqa: E402
import pf_depth_timing as DT                              # noqa: E402
import pf_future_best as FB                               # noqa: E402
import pf_influence as IN                                 # noqa: E402
import pf_order_harm_v2 as OH                             # noqa: E402
import pf_order_harm_v3 as V3                             # noqa: E402
import pf_order_harm_client_lane as CL                    # noqa: E402
import pf_joint_ladder_v3 as J3                           # noqa: E402
import pf_order_harm_policy as P1                         # noqa: E402
import pf_order_harm_policy_v2 as P2                      # noqa: E402
import pf_top_compare as TC                               # noqa: E402

TESTS = P2.TESTS
OUTD = f'{K.IO}/order_harm_why_20261114'
ARMS = ('LDG', 'FB25C', 'OHQ0', 'OHQM'); MODEL_ARMS = ('OHQ0', 'OHQM'); CAPN = 'hard50'; CAPV = 50 * J3.SIZE
CFGS = J3.CFGS; NR = J3.NR
LAG = 50.0; MAX_AGE = 100.0; REPOST_WIN = 100.0; MOVE = 0.5
ON_B = (100.0, 250.0, 500.0, 1000.0); AFTER_B = (250.0, 1000.0)
PHASES = ('masked', 'none', 'CALM', 'AFTER<250', 'AFTER250-1000', 'ON<100', 'ON100-250', 'ON250-500', 'ON500-1000', 'ON>=1000')
MOVES = ('unknown', '<=0', '0-0.5', '0.5-1.5', '>1.5')
DEPTHS = ('at best', 'behind', 'inside', 'no frame')
CLASSES = ('CALM', 'REPOST', 'KEPT-ON', 'KEPT-AFTER', 'PEND', 'OTHER-ON')
SAW = ('SEEN', 'LATER', 'NO-MOVE', 'UNKNOWN')
IDFV = V3.FEAT.index('dfv250')
MEM_CAP_MIB = 2048.0
NAN = float('nan'); INF = float('inf')
MUT_NOLAG = MUT_KEPT_NODEC = MUT_PEND_LAST = MUT_SWAP = 0  # planted errors (selftest only)


# ---------------------------------------------------------------- state and class functions
def side_guard(w, vg):
    """Per side: the guard flag, masked flag, episode onset time and the latest clear time at each 50 ms step."""
    tg = np.asarray(w['tg'], float); vv = np.asarray(w['vkl'], float); n = len(tg)
    on = np.asarray(vg, bool); masked = vv == OH.MASK_V
    onset_t = np.full(n, NAN); clear_t = np.full(n, -INF); k0 = -1; lc = -INF
    for k in range(n):
        if on[k] and not masked[k] and (k == 0 or not on[k - 1] or masked[k - 1]):
            k0 = k                                        # an episode starts at its first unmasked guard-on step
        if (not on[k]) and k > 0 and on[k - 1] and not masked[k - 1]:
            lc = tg[k]                                    # the end of the masked start is not a clear
        onset_t[k] = tg[k0] if on[k] else NAN
        clear_t[k] = lc
    return dict(tg=tg, tgl=tg.tolist(), on=on, masked=masked, onset_t=onset_t, clear_t=clear_t)


def phase_at(G, a):
    """The trade's phase at anchor a: the guard at the last step <= a - 50 ms (none if that step is > 100 ms old)."""
    t_ = a - (0.0 if MUT_NOLAG else LAG)
    g = bisect.bisect_right(G['tgl'], t_) - 1
    if g < 0 or t_ - G['tg'][g] > MAX_AGE:
        return 'none'
    if G['masked'][g]:
        return 'masked'
    if G['on'][g]:
        x = G['tg'][g] - G['onset_t'][g]
        return 'ON<100' if x < ON_B[0] else ('ON100-250' if x < ON_B[1] else ('ON250-500' if x < ON_B[2] else
                                                                             ('ON500-1000' if x < ON_B[3] else 'ON>=1000')))
    x = G['tg'][g] - G['clear_t'][g]
    return 'AFTER<250' if x < AFTER_B[0] else ('AFTER250-1000' if x < AFTER_B[1] else 'CALM')


def move_bucket(v):
    if not np.isfinite(v):
        return 'unknown'
    return '<=0' if v <= 0 else ('0-0.5' if v <= MOVE else ('0.5-1.5' if v <= 1.5 else '>1.5'))


def depth_class(d_):
    if d_ is None or d_ == -9:
        return 'no frame'
    return 'inside' if d_ < 0 else ('at best' if d_ == 0 else 'behind')


def dfv_at(PBm, side, a):
    rt, pv = PBm[0], PBm[1]
    p1 = IN.pbn_at(rt, pv, a); p0 = IN.pbn_at(rt, pv, a - 250.0)
    return IN.adv_of(side) * (p1 - p0) if (np.isfinite(p1) and np.isfinite(p0)) else NAN


def timing_class(G, t, ph, hist, ts):
    """Our fill at engine time t: PEND, KEPT-ON, OTHER-ON, KEPT-AFTER, REPOST, CALM (in this order). hist: the order's decisions
    (k, t_dec, a_hat, keep) - only those strictly before t count (a step's decisions come after its trades)."""
    if ph == 'pend' and not MUT_PEND_LAST:
        return 'PEND'
    g = bisect.bisect_left(G['tgl'], t) - 1
    on = g >= 0 and bool(G['on'][g])
    hb = [h for h in hist if h[1] < t]
    if on:
        k0t = G['onset_t'][g]
        if any(h[3] and G['tg'][h[0]] >= k0t for h in hb) or (MUT_KEPT_NODEC and not hb):
            return 'KEPT-ON'
        if ph == 'pend':
            return 'PEND'
        return 'OTHER-ON'
    if ph == 'pend':
        return 'PEND'
    if any(h[3] for h in hb):
        return 'KEPT-AFTER'
    gs = bisect.bisect_right(G['tgl'], ts) - 1
    if gs >= 0 and ts - G['clear_t'][gs] <= REPOST_WIN:
        return 'REPOST'
    return 'CALM'


def decompose(ours, theirs):
    """ours / theirs: {state: (shares, value sum)}. Returns gap, MIX, WITHIN, RESIDUAL in c/sh (value sums are c x sh)."""
    qo = sum(v[0] for v in ours.values()); qt = sum(v[0] for v in theirs.values())
    if qo <= 0 or qt <= 0:
        return dict(gap=NAN, mix=NAN, within=NAN, resid=NAN, ours=NAN, top5=NAN)
    vo_bar = sum(v[1] for v in ours.values()) / qo; vt_bar = sum(v[1] for v in theirs.values()) / qt
    mix = within = resid = 0.0
    for s in set(ours) | set(theirs):
        wo = ours.get(s, (0.0, 0.0))[0] / qo; wt = theirs.get(s, (0.0, 0.0))[0] / qt
        vt = theirs[s][1] / theirs[s][0] if s in theirs and theirs[s][0] > 0 else None
        vo = ours[s][1] / ours[s][0] if s in ours and ours[s][0] > 0 else None
        if MUT_SWAP:
            wo, wt = wt, wo
        if vt is None:
            if vo is not None:
                resid += wo * vo                          # states the top 5 never fill: gap = MIX + WITHIN + RESIDUAL exactly
            continue
        mix += (wo - wt) * vt
        if vo is not None:
            within += wo * (vo - vt)
    return dict(gap=vo_bar - vt_bar, mix=mix, within=within, resid=resid, ours=vo_bar, top5=vt_bar)


def timing_label(dd):
    if not np.isfinite(dd['mix']):
        return 'NO-DATA'
    if dd['mix'] < 0 and abs(dd['mix']) >= abs(dd['within']):
        return 'TIMING'
    if dd['within'] < 0 and abs(dd['within']) > abs(dd['mix']):
        return 'WITHIN-STATE'
    return 'NEITHER'


def saw_class(dfv_dec, dfv_trade):
    if not (np.isfinite(dfv_dec) and np.isfinite(dfv_trade)):
        return 'UNKNOWN'
    if dfv_dec > MOVE:
        return 'SEEN'
    return 'LATER' if dfv_trade > MOVE else 'NO-MOVE'


def selftest():
    bad = []
    w = dict(tg=np.arange(0.0, 1500.0, 50.0), vkl=[0.5] * 30)
    vg = [False] * 30
    for k in range(6, 18):
        vg[k] = True                                      # guard on 300-850 ms, clears at step 18 (900 ms)
    w['vkl'][0] = OH.MASK_V; vg[0] = True                 # a masked first step (guard flag on, start price unknown)
    G = side_guard(w, vg)
    cases = [(40.0, 'none'), (60.0, 'masked'), (360.0, 'ON<100'), (520.0, 'ON100-250'), (860.0, 'ON500-1000'), (960.0, 'AFTER<250'),
             (1200.0, 'AFTER250-1000'), (1460.0, 'AFTER250-1000')]
    got = [phase_at(G, a) for a, _ in cases]
    if got != [c for _, c in cases]:
        bad.append(f'phase {got}')
    w2 = dict(tg=np.arange(0.0, 3000.0, 50.0), vkl=[0.5] * 60); G2 = side_guard(w2, [False] * 60)
    if phase_at(G2, 2000.0) != 'CALM':
        bad.append('calm phase')
    if [move_bucket(x) for x in (NAN, -0.2, 0.0, 0.3, 0.5, 1.0, 2.0)] != ['unknown', '<=0', '<=0', '0-0.5', '0-0.5', '0.5-1.5', '>1.5']:
        bad.append('move bucket')
    if [depth_class(x) for x in (None, -9, -1, 0, 2)] != ['no frame', 'no frame', 'inside', 'at best', 'behind']:
        bad.append('depth class')
    # classes: decisions (k, t, a, keep); guard on steps 6-17
    kept = [(6, 300.0, 0.4, True), (7, 350.0, 0.2, True)]
    ccases = [((380.0, 'hold', kept, 100.0), 'KEPT-ON'), ((380.0, 'pend', kept, 100.0), 'PEND'), ((380.0, 'hold', [], 100.0), 'OTHER-ON'),
              ((310.0, 'hold', [(6, 300.0, 0.4, True)], 100.0), 'KEPT-ON'), ((300.0, 'hold', [(6, 300.0, 0.4, True)], 100.0), 'CALM'),
              ((1000.0, 'hold', kept, 100.0), 'KEPT-AFTER'), ((1000.0, 'hold', [], 950.0), 'REPOST'), ((1000.0, 'hold', [], 200.0), 'CALM'),
              ((1000.0, 'pend', kept, 100.0), 'PEND')]
    gotc = [timing_class(G, *a) for a, _ in ccases]
    if gotc != [c for _, c in ccases]:
        bad.append(f'class {gotc}')
    # decomposition: two states; theirs 50/50 at +1 / -1 (mean 0); ours 80% in the bad state at the same values -> gap -0.6, all MIX
    th = {'good': (50.0, 50.0), 'bad': (50.0, -50.0)}; ou = {'good': (20.0, 20.0), 'bad': (80.0, -80.0)}
    dd = decompose(ou, th)
    if not (abs(dd['gap'] + 0.6) < 1e-9 and abs(dd['mix'] + 0.6) < 1e-9 and abs(dd['within']) < 1e-9) or timing_label(dd) != 'TIMING':
        bad.append(f'decomposition mix {dd}')
    ou2 = {'good': (50.0, 0.0), 'bad': (50.0, -100.0)}       # the same mix, every state 1 c worse -> all WITHIN
    dd2 = decompose(ou2, th)
    if not (abs(dd2['mix']) < 1e-9 and abs(dd2['within'] + 1.0) < 1e-9) or timing_label(dd2) != 'WITHIN-STATE':
        bad.append(f'decomposition within {dd2}')
    if [saw_class(1.0, 0.0), saw_class(0.2, 1.0), saw_class(0.2, 0.1), saw_class(NAN, 1.0)] != ['SEEN', 'LATER', 'NO-MOVE', 'UNKNOWN']:
        bad.append('saw class')
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_NOLAG', 'MUT_KEPT_NODEC', 'MUT_PEND_LAST', 'MUT_SWAP'):
        globals()[nm] = 1
        try:
            out[nm] = bool(selftest())
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    P2.all_selftests(); TC.all_selftests()
    b = selftest(); m = mutation_check()
    if b or not all(m.values()):
        sys.exit(f'ORDER-HARM WHY SELFTEST FAILED -> no output: {b} planted errors caught {m}')
    print(f'order-harm why selftests passed; planted errors caught: {m}', flush=True)


# ---------------------------------------------------------------- the arms with recording wrappers (decisions unchanged)
def arm_dec_rec(arm, W, ctx, wi, pred, F_, cnt):
    if arm == 'LDG':
        return None, 0
    out = {}
    for s in (0, 1):
        for d in range(NR):
            if arm == 'FB25C':
                fb = FB.fb_dec(ctx[s]['vg'], W[s]['best'], s, ctx[s]['Pm'], d, tau=0.25)

                def keep_fn(k, t, cur, fb=fb):
                    kp = not fb(k, t, cur)
                    cur.setdefault('_h', []).append((k, t, NAN, kp))
                    return kp
            else:
                theta = 0.0 if arm == 'OHQ0' else F_['theta_M']

                def keep_fn(k, t, cur, s=s, d=d, MK=wi['MK'][s], theta=theta):
                    if k not in MK:                       # POLICY v2's model_keep: no inputs -> pull (fail closed)
                        cnt['noinput'] += 1
                        cur.setdefault('_h', []).append((k, t, NAN, False))
                        return False
                    a = pred(s, d, cur, k, t)
                    kp = a > theta
                    cur.setdefault('_h', []).append((k, t, a, kp))
                    return kp
            out[(s, d)] = P2.cadence(keep_fn, ctx[s]['vg'], wi['onset'][s], cnt, d)
    return out, 1


# ---------------------------------------------------------------- one test day
def day_run(day):
    all_selftests()
    T0 = time.time(); F_ = P2.fold(day); D = K.load_day(day); DI = P1.day_inputs(day, D)
    S1d = f'{K.IO}/step1/{day}'
    full = pd.read_parquet(f'{S1d}/legs_timed2.parquet', columns=['mid', 'signer', 'shares', 't_rel'])
    tkf = pd.read_parquet(f'{S1d}/legs_tk.parquet', columns=['tk']).tk.values
    if len(tkf) != len(full):
        sys.exit('REFUSED: legs_tk not aligned with legs_timed2')
    inw = full.mid.isin(list(D['W0'])).values
    GR = GP.rank_groups(full.signer.values[inw], full.shares.values[inw].astype(float), full.t_rel.values[inw].astype(float))
    SIG = full.signer.values.astype('int64'); del full
    LT = D['LT']; ix = LT.index.values
    grp = np.where(tkf[ix] >= 0, tkf[ix], -1 - ix)
    WSg = pd.Series(TS.ws_groups(LT.how.values, grp), index=LT.index); TKS = pd.Series(grp, index=LT.index)
    SV = pd.read_parquet(f'{P2.OUTD}/fills_{day}.parquet')
    SV = SV[(SV.capk == CAPN) & SV.arm.isin(ARMS)]
    st = dict(windows=0, window_sides=0, align_differ=0, legs=0, runs=0, noinput=0, kept_fills=0, viol=0, lookahead_checks=0, lookahead_bad=0)
    theirs = {}; rows = []; recon = []
    for W in P1.window_pairs(D):
        st['windows'] += 1; m = int(W[0]['m']); n = W[0]['n']
        ctx = J3.window_ctx(W, F_['PO'], F_['POI'], n); wi = P1.window_inputs(W, ctx, DI)
        pst = dict(inputs=0, lookahead_checks=0, lookahead_bad=0); pred = P2.make_pred(W, wi, F_['B'], pst)
        PBm = DI['PB'].get(m, ([], np.zeros(0), np.zeros(0), np.zeros(0)))
        tend = W[0]['te']; jte = int(np.searchsorted(ctx[0]['tf'], tend, side='right')) - 1; mid_end = ctx[0]['mff'][max(jte, 0)]
        yup = 100.0 * W[0]['yup']
        jtm = int(np.searchsorted(ctx[0]['tf'], tend + CL.TAIL_MS, side='right')) - 1; mark = float(ctx[0]['mff'][jtm])
        Gs = {}; src = {}; tkv = {}; tkp = {}; lrows = {}; PH = {}; MV = {}
        for s in (0, 1):
            w = W[s]; Gs[s] = side_guard(w, ctx[s]['vg'])
            if (m, s) in D['LI']:
                l = LT.iloc[D['LI'][(m, s)]]
                o_ = np.lexsort(((-l.ptick.values if s == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
                st['window_sides'] += 1; st['align_differ'] += int(not np.array_equal(l.anc.values[o_], w['anc']))
            else:
                lrow = np.zeros(0, 'int64')
            lrows[s] = lrow
            src[s] = WSg.reindex(lrow).values.astype(bool) if len(lrow) else np.zeros(0, bool)
            tkv[s] = TKS.reindex(lrow).values if len(lrow) else np.zeros(0, 'int64'); tkp[s] = {}
            for t_, p_ in zip(tkv[s], w['ptl']):
                tkp[s].setdefault(int(t_), []).append(int(p_))
            anc = w['anc']
            PH[s] = [phase_at(Gs[s], float(a)) for a in anc]
            MV[s] = np.array([dfv_at(PBm, s, float(a)) for a in anc]) if len(anc) else np.zeros(0)
            # THEIRS: every maker leg in the first 60 s
            sg = ctx[s]['sg']; tf, mff, tfl = ctx[s]['tf'], ctx[s]['mff'], ctx[s]['tfl']
            for i in range(len(lrow)):
                a = float(anc[i])
                if not (w['ws'] <= a < w['te']):
                    continue
                st['legs'] += 1; p = int(w['ptl'][i]); q = float(w['shl'][i])
                dc = depth_class(GP.depth_of(tfl, w['P5'], w['opp'], s, a, p))
                v1 = sg * (mff[max(int(np.searchsorted(tf, a + 1000.0, side='right')) - 1, 0)] - p)
                v5 = sg * (mff[max(int(np.searchsorted(tf, a + 5000.0, side='right')) - 1, 0)] - p)
                if not np.isfinite(v1):
                    continue
                who = GR.get(int(SIG[lrow[i]]), 'rest'); typ = DT.trade_type(s, p, tkp[s][int(tkv[s][i])])
                key = (who, 'WS' if src[s][i] else 'E2', PH[s][i], move_bucket(MV[s][i]), dc, typ)
                e = theirs.setdefault(key, [0, 0.0, 0.0, 0.0, 0.0])
                e[0] += 1; e[1] += q; e[2] += q * v1; e[3] += q * (v5 if np.isfinite(v5) else 0.0); e[4] += q * sg * (mark - p)
        # OURS: the four arms at hard 50, every clock
        for q_, shf in CFGS:
            cfg = f'{q_}{int(shf)}'
            for arm in ARMS:
                cnt = dict(keep=[0] * NR, pull=[0] * NR, noinput=0)
                dec, idle = arm_dec_rec(arm, W, ctx, wi, pred, F_, cnt)
                r = CL.sim_joint(W, q_, shf, CAPV, {0: ctx[0]['vg'], 1: ctx[1]['vg']}, decide=dec, idle_only=idle, mode='hard',
                                 cooldown=CL.REPOST_MS)
                st['runs'] += 1; st['noinput'] += cnt['noinput']; st['viol'] += sum(r['stat']['viol'].values())
                terminal = CL.portfolio_terminal(r, ctx[0]['tf'], ctx[0]['mff'])
                for s in (0, 1):
                    c2 = ctx[s]; sg = c2['sg']; tf, mff, tfl = c2['tf'], c2['mff'], c2['tfl']; anc = W[s]['anc']
                    for d in range(NR):
                        byk = {}
                        for o in r['order_states'][(s, d)]:
                            byk.setdefault(int(o['pt']), []).append(o)
                        for f in r['fills'][(s, d)]:
                            i, pt, qq, thr_, age, ph = f[0], int(f[1]), f[2], f[3], f[4], f[5]
                            t = float(anc[i]); t_exec = t - shf
                            vend = sg * (terminal['mark'] - pt)
                            d_ = GP.depth_of(tfl, W[s]['P5'], W[s]['opp'], s, t, pt)
                            v1 = sg * (mff[max(int(np.searchsorted(tf, t + 1000.0, side='right')) - 1, 0)] - pt)
                            v5 = sg * (mff[max(int(np.searchsorted(tf, t + 5000.0, side='right')) - 1, 0)] - pt)
                            typ = P2.fill_type(s, pt, bool(thr_), tkp[s][int(tkv[s][i])])
                            recon.append((int(day), cfg, arm, CAPN, s, d, 'WS' if src[s][i] else 'E2', -9 if d_ is None else d_, bool(thr_),
                                          age < 100.0, ph == 'pend', float(qq), v1, v5, vend, sg * (yup - pt), float(age),
                                          sg * (mid_end - pt), m, t_exec, terminal['time'], pt, typ))
                            cand = [o for o in byk.get(pt, []) if abs((t_exec - o['ta']) - age) < 1e-6]
                            if len(cand) != 1:
                                sys.exit(f'REFUSED: fill not matched to one order ({len(cand)}) on {day} m {m}')
                            o = cand[0]; hist = o.get('_h', [])
                            cls = timing_class(Gs[s], t_exec, ph, hist, float(o['ts']))
                            hb = [h for h in hist if h[1] < t_exec and h[3]]
                            a_last = dfv_last = t_last = sb_last = dn_last = NAN; n_keep = len(hb)
                            if hb and arm in MODEL_ARMS:
                                kl, tl, al, _ = hb[-1]
                                a_last = float(al); t_last = float(tl)
                                xk = wi['MK'][s].get(kl)
                                dfv_last = float(xk[IDFV]) if xk is not None else NAN
                                ordf = pred.order_x(s, d, o, kl, tl, W[s]['S5'])
                                sb_last = float(ordf[OH.ORD_FEAT.index('size_better')]); dn_last = float(ordf[OH.ORD_FEAT.index('depth_now')])
                                st['kept_fills'] += 1
                            rows.append((int(day), cfg, arm, m, s, d, i, t, t_exec, pt, float(qq), bool(thr_), float(age), age < 100.0,
                                         ph == 'pend', 'WS' if src[s][i] else 'E2', depth_class(d_), typ, v1, v5, vend, cls, PH[s][i],
                                         float(MV[s][i]), n_keep, a_last, dfv_last, t_last, sb_last, dn_last,
                                         bool(W[s]['ws'] <= t < W[s]['te'])))
        st['lookahead_checks'] += pst['lookahead_checks']; st['lookahead_bad'] += pst['lookahead_bad']
        for s in (0, 1):
            W[s].pop('cache', None)
        if st['windows'] % 24 == 0:
            r2 = K.rss_mib(); print(f'{day}: {st["windows"]} windows, {st["runs"]} runs, {time.time() - T0:.0f} s, rss {r2:.0f} MiB', flush=True)
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    if st['align_differ'] or st['viol'] or st['lookahead_bad'] or not st['lookahead_checks']:
        sys.exit(f'ORDER-HARM WHY CONTROL FAILED on {day} -> no output: {st}')
    RC = pd.DataFrame(recon, columns=list(SV.columns)); bad = {}
    for q_, shf in CFGS:
        cfg = f'{q_}{int(shf)}'
        for arm in ARMS:
            X = RC[(RC.cfg == cfg) & (RC.arm == arm)].reset_index(drop=True)
            Y = SV[(SV.cfg == cfg) & (SV.arm == arm)].reset_index(drop=True)
            if not FB.same_rows(X, Y):
                bad[f'{cfg} {arm}'] = (len(X), len(Y))
    if bad:
        sys.exit(f'ORDER-HARM WHY RECONCILIATION FAILED on {day} -> no output: {bad}')
    st['reconciled_rows'] = len(RC)
    cols = ['day', 'cfg', 'arm', 'm', 'side', 'rung', 'i', 'anc', 't_exec', 'pt', 'q', 'thr', 'age', 'young', 'pend', 'src', 'depth', 'typ',
            'v1', 'v5', 'vend', 'cls', 'phase', 'dfv_trade', 'n_keep', 'a_last', 'dfv_last', 't_last', 'sb_last', 'dn_last', 'in60']
    os.makedirs(OUTD, exist_ok=True)
    pd.DataFrame(rows, columns=cols).to_parquet(f'{OUTD}/ours_{day}.parquet', index=False)
    pd.DataFrame([dict(who=k[0], src=k[1], phase=k[2], move=k[3], depth=k[4], typ=k[5], legs=v[0], q=v[1], v1=v[2], v5=v[3], vend=v[4])
                  for k, v in theirs.items()]).to_parquet(f'{OUTD}/theirs_{day}.parquet', index=False)
    json.dump(dict(stat=st, days=len(D['W0']) / 288.0, **CL.stamp()), open(f'{OUTD}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: {st} in {time.time() - T0:.0f} s', flush=True)


# ---------------------------------------------------------------- report
def report():
    all_selftests()
    S = {d: json.load(open(f'{OUTD}/day_{d}.json')) for d in TESTS}
    for r_ in S.values():
        CL.require_contract(r_)
    dd_ = {d: S[d]['days'] for d in TESTS}; ndays = sum(dd_.values())
    O = pd.concat([pd.read_parquet(f'{OUTD}/ours_{d}.parquet') for d in TESTS], ignore_index=True)
    T = pd.concat([pd.read_parquet(f'{OUTD}/theirs_{d}.parquet').assign(day=int(d)) for d in TESTS], ignore_index=True)
    O['usd_end'] = O.q * O.vend / 100.0; O['usd1'] = O.q * O.v1 / 100.0
    st_ = {k_: sum(S[d]['stat'][k_] for d in TESTS) for k_ in ('windows', 'legs', 'runs', 'reconciled_rows', 'kept_fills', 'noinput')}
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    L = [f'ORDER-HARM WHY (POST HOC, descriptive; after POLICY v2\'s NOT-ACCEPTED): test days {", ".join(TESTS)}, first 60 s, hard 50; as of {asof}; '
         f'controls: our fill rows equal POLICY v2\'s saved fills ({st_["reconciled_rows"]} rows, 4 arms x 3 clocks x 5 days), legs aligned on '
         f'every window-side, accounting violations 0; their legs {st_["legs"]}; our runs {st_["runs"]}', '']
    out = {}
    # W1 LOSS MAP
    L.append('W1 LOSS MAP: $/day at the 60.2 s mark (whole portfolio) | $/day at 1 s | sh/day, by timing class (all trades; WS / E2 split '
             'beside the mark total)')
    for cfg in ('M0', 'M50', 'M250'):
        L.append(f'  {cfg}: class: ' + ' || '.join(ARMS))
        for c in CLASSES:
            parts = []
            for arm in ARMS:
                x = O[(O.cfg == cfg) & (O.arm == arm) & (O.cls == c)]
                ws_ = x[x.src == 'WS'].usd_end.sum() / ndays; e2_ = x[x.src == 'E2'].usd_end.sum() / ndays
                parts.append(f'{x.usd_end.sum() / ndays:+8.1f} | {x.usd1.sum() / ndays:+8.1f} | {x.q.sum() / ndays:7.0f} (WS {ws_:+7.1f} E2 {e2_:+7.1f})')
            L.append(f'    {c:<10} ' + ' || '.join(parts))
        tot = [O[(O.cfg == cfg) & (O.arm == arm)].usd_end.sum() / ndays for arm in ARMS]
        L.append(f'    {"total":<10} ' + ' || '.join(f'{v:+8.1f}' for v in tot))
        for base in ('LDG', 'FB25C'):
            diffs = {c: (O[(O.cfg == cfg) & (O.arm == 'OHQ0') & (O.cls == c)].usd_end.sum()
                         - O[(O.cfg == cfg) & (O.arm == base) & (O.cls == c)].usd_end.sum()) / ndays for c in CLASSES}
            worst = min(diffs, key=lambda c: diffs[c]); tdiff = sum(diffs.values())
            out[f'W1 {cfg} OHQ0-{base}'] = dict(worst=worst, diffs=diffs, total=tdiff)
            L.append(f'    OHQ0 - {base}: ' + ' | '.join(f'{c} {diffs[c]:+.1f}' for c in CLASSES) + f' | total {tdiff:+.1f} -> most negative: {worst}')
        L.append('')
    # W2 TIMING vs TOP 5
    L.append('W2 TIMING vs TOP 5: per-share value at 1 s (c/sh), legs in the first 60 s; ours w/o young (C1). gap = MIX (state mix, valued at the '
             'top 5\'s per-state value) + WITHIN (same state) + RESIDUAL; state = phase x move x depth')
    t5 = T[T.who == 'top 5']

    def tstate(x, val):
        g = x.groupby(['phase', 'move', 'depth'])
        return {k: (float(v.q.sum()), float(v[val].sum())) for k, v in g}
    for srcs, nm in ((('WS',), 'WS-timed'), (('WS', 'E2'), 'all trades')):
        th = tstate(t5[t5.src.isin(srcs)], 'v1')
        L.append(f'  {nm}: top 5 {sum(v[1] for v in th.values()) / max(sum(v[0] for v in th.values()), 1e-9):+.3f} c/sh')
        for cfg in ('M0', 'M50', 'M250'):
            for arm in ARMS:
                x = O[(O.cfg == cfg) & (O.arm == arm) & O.in60 & ~O.young & O.src.isin(srcs)].assign(vq=lambda z: z.q * z.v1)
                x = x.assign(dfv_b=[move_bucket(v) for v in x.dfv_trade.values])
                ou = {k: (float(v.q.sum()), float(v.vq.sum())) for k, v in x.groupby(['phase', 'dfv_b', 'depth'])}
                dd = decompose(ou, th); lab = timing_label(dd)
                out[f'W2 {nm} {cfg} {arm}'] = dict(label=lab, **dd)
                L.append(f'    {cfg:<4} {arm:<5}: ours {dd["ours"]:+.3f} gap {dd["gap"]:+.3f} = MIX {dd["mix"]:+.3f} + WITHIN {dd["within"]:+.3f} '
                         f'+ RESIDUAL {dd["resid"]:+.3f} -> {lab}')
    L.append('')
    L.append('  by phase (WS-timed, 1 s): share of shares | c/sh  - top 5 (real) || ours w/o young at M 50: ' + ' | '.join(ARMS))
    t5w = t5[t5.src == 'WS']; qt = t5w.q.sum()
    O50 = O[(O.cfg == 'M50') & O.in60 & ~O.young & (O.src == 'WS')]
    for ph in PHASES:
        a = t5w[t5w.phase == ph]
        parts = [f'{a.q.sum() / qt:6.1%} {a.v1.sum() / max(a.q.sum(), 1e-9):+6.3f}']
        for arm in ARMS:
            x = O50[O50.arm == arm]; xx = x[x.phase == ph]
            parts.append(f'{xx.q.sum() / max(x.q.sum(), 1e-9):6.1%} {(xx.q * xx.v1).sum() / max(xx.q.sum(), 1e-9):+6.3f}')
        L.append(f'    {ph:<14} ' + ' || '.join(parts))
    L.append('  by move at the trade (WS-timed, 1 s): share of shares | c/sh - top 5 (real) || ours w/o young at M 50: ' + ' | '.join(ARMS))
    O50 = O50.assign(dfv_b=[move_bucket(v) for v in O50.dfv_trade.values])
    for mv in MOVES:
        a = t5w[t5w.move == mv]
        parts = [f'{a.q.sum() / qt:6.1%} {a.v1.sum() / max(a.q.sum(), 1e-9):+6.3f}']
        for arm in ARMS:
            x = O50[O50.arm == arm]; xx = x[x.dfv_b == mv]
            parts.append(f'{xx.q.sum() / max(x.q.sum(), 1e-9):6.1%} {(xx.q * xx.v1).sum() / max(xx.q.sum(), 1e-9):+6.3f}')
        L.append(f'    {mv:<14} ' + ' || '.join(parts))
    L.append('')
    # W3 THE MODEL'S KEEPS
    L.append('W3 THE MODEL\'S KEEPS: fills on KEPT orders (KEPT-ON + KEPT-AFTER) by what the model saw at its last keep decision before the '
             'fill: $/day 60.2 s mark | $/day 1 s | sh/day | mean A_hat | median ms from that decision to the fill | mean size at better (sh)')
    for cfg in ('M0', 'M50', 'M250'):
        for arm in MODEL_ARMS:
            x = O[(O.cfg == cfg) & (O.arm == arm) & O.cls.isin(['KEPT-ON', 'KEPT-AFTER'])]
            x = x.assign(saw=[saw_class(a_, b_) for a_, b_ in zip(x.dfv_last.values, x.dfv_trade.values)])
            parts = []; loss = {}
            for c in SAW:
                y = x[x.saw == c]
                loss[c] = float(y[y.usd_end < 0].usd_end.sum()) / ndays
                parts.append(f'{c} {y.usd_end.sum() / ndays:+7.1f} | {y.usd1.sum() / ndays:+7.1f} | {y.q.sum() / ndays:6.0f} | '
                             f'{y.a_last.mean() if len(y) else NAN:+.3f} | {np.median(y.t_exec - y.t_last) if len(y) else NAN:4.0f} | '
                             f'{y.sb_last.mean() if len(y) else NAN:6.0f}')
            worst = min(loss, key=lambda c: loss[c])
            out[f'W3 {cfg} {arm}'] = dict(worst=worst, loss=loss, total=float(x.usd_end.sum()) / ndays)
            L.append(f'  {cfg:<4} {arm:<5} kept total {x.usd_end.sum() / ndays:+7.1f}: ' + ' || '.join(parts) + f' -> largest loss: {worst}')
    L.append('')
    # W4 THE MODEL'S VIEW (labels)
    L.append('W4 THE MODEL\'S VIEW (v3 test-day label rows): rung group x move at the decision: rows | keep share | mean A_hat | mean G (realized, '
             'final policy) | mean keep-all advantage; kept decisions\' mean G per day')
    ok_days = 0; miss_days = 0; per = []
    agg = {}
    for d in TESTS:
        P = pd.read_parquet(f'{V3.OUTD}/pred_{d}.parquet', columns=['m', 'side', 'k', 'rung_i', 'a_final', 'target', 'realized'])
        R = pd.read_parquet(f'{V3.OUTD}/rows_{d}.parquet', columns=['m', 'side', 'k', 'rung_i', 'dfv250'])
        for c in ('m', 'side', 'k', 'rung_i'):
            if not np.array_equal(P[c].values, R[c].values):
                sys.exit(f'REFUSED: v3 predictions not aligned with its rows on {c} ({d})')
        mvb = np.array([move_bucket(v) for v in R.dfv250.values]); kp = P.a_final.values > 0
        gk = P.target.values[kp]; mk = float(gk.mean()) if len(gk) else NAN
        sel = kp & np.isin(mvb, ['0.5-1.5', '>1.5']); gm = float(P.target.values[sel].mean()) if sel.any() else NAN
        ok_days += int(mk > 0); miss_days += int(gm < 0); per.append((d, mk, gm, int(kp.sum()), int(sel.sum())))
        rg = np.where(P.rung_i.values == 0, '0', np.where(P.rung_i.values <= 2, '1-2', '3-5'))
        DF = pd.DataFrame(dict(rg=rg, mv=mvb, kp=kp, a=P.a_final.values, g=P.target.values, kav=P.realized.values))
        for k_, v_ in DF.groupby(['rg', 'mv']):
            e = agg.setdefault(k_, [0, 0, 0.0, 0.0, 0.0])
            e[0] += len(v_); e[1] += int(v_.kp.sum()); e[2] += float(v_.a.sum()); e[3] += float(v_.g.sum()); e[4] += float(v_.kav.sum())
        del P, R, DF
    for rg in ('0', '1-2', '3-5'):
        L.append(f'  rung {rg}: ' + ' | '.join(f'{mv} {agg[(rg, mv)][0]:7d} {agg[(rg, mv)][1] / agg[(rg, mv)][0]:.2f} '
                                                f'{agg[(rg, mv)][2] / agg[(rg, mv)][0]:+.3f} {agg[(rg, mv)][3] / agg[(rg, mv)][0]:+.3f} '
                                                f'{agg[(rg, mv)][4] / agg[(rg, mv)][0]:+.3f}' for mv in MOVES if (rg, mv) in agg))
    lab_ok = 'LABEL-OK' if ok_days >= 4 else 'LABEL-NOT-OK'
    lab_mm = 'MISSES-MOVES' if miss_days >= 4 else 'MOVES-OK'
    out['W4'] = dict(label_ok=lab_ok, misses_moves=lab_mm, per_day=per)
    L.append('  per day: ' + ' | '.join(f'{d[4:]} kept G {mk:+.3f} (n {nk}), kept in moves > 0.5 c G {gm:+.3f} (n {ns})' for d, mk, gm, nk, ns in per))
    L.append(f'  -> {lab_ok} (kept decisions\' mean G > 0 on {ok_days}/5 days); {lab_mm} (kept decisions in moves > 0.5 c with mean G < 0 on '
             f'{miss_days}/5 days)')
    L.append('')
    # W5 HORIZON
    L.append('W5 HORIZON: c/sh at 1 s | 5 s | 60.2 s mark - ours by timing class at M 50 (all trades, all ages) || top 5 by phase (all trades)')
    for c in CLASSES:
        parts = []
        for arm in ARMS:
            x = O[(O.cfg == 'M50') & (O.arm == arm) & (O.cls == c)]; qq = x.q.sum()
            parts.append(f'{arm} {(x.q * x.v1).sum() / max(qq, 1e-9):+6.3f} {(x.q * x.v5).sum() / max(qq, 1e-9):+6.3f} '
                         f'{(x.q * x.vend).sum() / max(qq, 1e-9):+6.3f}')
        L.append(f'  {c:<10} ' + ' | '.join(parts))
    for ph in PHASES:
        a = t5[t5.phase == ph]; qq = a.q.sum()
        L.append(f'  top 5 {ph:<14} {a.v1.sum() / max(qq, 1e-9):+6.3f} {a.v5.sum() / max(qq, 1e-9):+6.3f} {a.vend.sum() / max(qq, 1e-9):+6.3f} '
                 f'({qq / ndays:9.0f} sh/day)')
    txt = '\n'.join(L)
    open(f'{OUTD}/order_harm_why.txt', 'w').write(txt + '\n')
    json.dump(dict(readings=out, **CL.stamp()), open(f'{OUTD}/order_harm_why.json', 'w'), indent=1, default=str)
    print(txt, flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        all_selftests()
    elif cmd == 'day':
        day_run(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit('usage: pf_order_harm_why.py selftest | day <day> | report')
