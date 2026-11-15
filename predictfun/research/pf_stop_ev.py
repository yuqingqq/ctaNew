"""STOP-EV STAGE 1 - DATA (2026-11-15, session aea5bfb9): the label data of the expected-value keep rule, built and checked before any
model (USER 2026-11-15: "Again, we should have a few preprocessing steps, first, data processing, check its coverage, data quality, etc,
then model, then the pipeline"). Design: predictfun/handoffs/20261115_stop_ev_design.md, sections 7-8. No model and no policy here.

ROWS: the STOP-MODEL's rows (every 50 ms grid step of the first 60 s, both sides, valid known book; ref = the side's best in the last
  frame KNOWN at t), keys (m, side, k). LEVELS j = 1..6: price p_j = ref - (j - 1) ticks on the side (bids down, asks up).
CLOCK: the lane's primary fill clock M 50 (trade time tt = anchor - 50 ms). A decision at t can avoid the legs with tt >= t + 22 ms
  (K.L_CANCEL: a cancel sent at t lands at t + 22 and the lane fills an order only while tt < its cancel time); horizon (t, t + 1 s].
LABELS per row and level (float32; statuses instead of silent drops):
  vis_j     visible size at p_j in the frame KNOWN at t (the lane's lv: 0 = inside the shown levels but empty; NaN = beyond the 5th
            shown level -> status VIS_UNKNOWN; p_j outside 1..99 -> status PRICE_OUT, every value NaN)
  COMPONENTS (market flow, no queue) over the side's public legs with tt in [t + 22, t + 1000]: shares and value at p_j of
    as_j / asv_j   legs AT p_j whose taker order STOPPED there (its worst price on the side is p_j; DT.trade_type)
    at_j / atv_j   legs AT p_j whose taker order went deeper (THROUGH)
    au_j / auv_j   legs AT p_j without a taker key (type unknown; the lane's fill_type calls them STOP)
    bt_j / btv_j   legs BEYOND p_j (the level was taken; a front order at p_j fills at p_j)
    value = shares x sg x (mid(anchor + 1 s) - p_j), cents x shares (order_steps' markout: the last venue frame <= anchor + 1 s)
  EXACT ORDER FILLS = THE LABEL (a chronological, quantity-conserving ledger; the components above are diagnostics only - Codex reviews
    e315abe8 / ffa2025e: window-average markouts erase the queue position's value). The lane's rule M (CL.sim_joint's apply) for ONE
    order of Z shares resting at p_j with queue ahead q at t, CONTINUATION declared: the order is held until t + 1 s, no cancel, no
    replacement (a rolling local keep-value surrogate, not the value of keeping only until the next 50 ms decision). Legs in (t, t + 22)
    are applied (they drain the queue / fill: unavoidable under keep and pull alike) but not counted; a leg AT p_j drains the queue, then
    fills us; a leg BEYOND fills us by min(its size, our remainder) and clears the queue; before each leg the queue is capped by the
    visible size at p_j in every venue frame since the known frame; each tranche is valued at its own mark (anchor + 1 s; frames are
    loaded to ws + 125 s). SUPPORT: a fresh order (remainder Z) at a hypothetical queue position on the grid below; the lane's own
    order states (rec, with partial remainders) give the support of real states. One order per level: shared trade volume with our
    other rungs is the joint lane's (E7's SHARED status measures it):
    f0_j / v0_j    front (q = 0): shares, 1 s value; v0x5_j its 5 s value; f0w_j / v0w_j the part from WS-anchored legs
    fh_j / vh_j    middle (q = 0.5 vis_j);  fb_j / vb_j  back (q = vis_j)   (NaN when vis_j is unknown)
    v0m0_j / v0m250_j   the front's 1 s value on the M 0 / M 250 clocks (timing sensitivity)
  INCOMPLETE flags inc_j (bits; the value is kept, the row is flagged, the report counts): 1 = a counted leg's 1 s mark frame is more than
    500 ms old (feed gap); 2 = an UNTIMED leg (no anchor) at or beyond p_j may fall in (t, t + 1 s] (its possible time = its exec second
    + the 0.1-99.9% band of timed legs' anchor - exec second offsets on the day) - missing flow would change both the queue and the fills.
LEG TABLE legs_<day>: every leg the lane sees (row id in step 1's legs_timed2, anchor, price, shares, WS / E2 source, taker group, type,
  depth = ticks behind the venue best in the frame strictly before the anchor, beyond the 5th shown level, 1 s / 5 s mids).
RECONCILIATION TABLE rec_<day>: the keep-all client lane with the CORRECTED queue initialization (pf_stop_queue_activation_lane, contract
  activation-snapshot-queue-v1: queue ahead from the last snapshot at or before activation; the old lane's next-frame uplift was
  look-ahead; M 50, 6 rungs x Z sh a side, no cap, re-post at once = align's KA) order states
  every 5th step: the lane's own queue ahead / remaining size, the lane's fills of that order in [t + 22, t + 1 s], the label simulated
  from that state; status ENDED (the order's cancel lands by t + 1 s: drift / cross / window end) and SHARED (a leg that reaches it also
  filled our better-priced order on the side; the label takes the whole leg); CLEAN = neither.
DATA CHECKS, declared before the first run (E1-E7: any failure -> no output for the day; E8 in the report; any failure -> DATA-FAIL and
  the model waits). Every check ships a positive control that must fire on the same real data:
 E1 POPULATION   label keys and ref equal the STOP-MODEL rows' (m, side, k, best) on the day (symmetric difference 0).
 E2 LEGS         every first-minute step-1 leg (exec second in [ws, ws + 60 s)) of an included window is USED or carries a status
                 (UNTIMED: no anchor; OUT: anchor outside [ws - 2 s, ws + 61 s]); unaccounted 0; the table holds each leg once and only
                 legs of included windows with an in-range anchor; anchors equal an independent recomputation. Exact-content duplicates
                 in step 1 (every column and the taker key equal) are counted, not removed; above 0.1% of first-minute shares -> FLAG
                 named in the verdict (a maker's two equal orders in one transaction look the same; the lane sees the same legs).
 E3 TYPES        per-leg STOP / THROUGH / UNK equal an independent recomputation over the whole day's taker groups (mismatch 0; groups
                 with legs in two markets or on both sides are a status).
 E4 VISIBLE      vis_j equals an independent recomputation from the raw book files at an independently computed known frame on >= 2,000
                 sampled (row, level) per day (mismatch 0); control: the venue frame at t differs somewhere.
 E5 MARKOUT      the legs' 1 s / 5 s mids equal an independent recomputation from the raw book files on >= 2,000 sampled legs (mismatch
                 0); control: the mid at anchor + 0.9 s differs somewhere.
 E6 BRUTE        on >= 2,000 sampled (row, level) the components and the front order's fill f0 equal a recomputation from the leg table
                 with E3's independent types (mismatch 0); control: the components over (t, t + 1 s] differ somewhere.
 E7 RECONCILE    the recording corrected lane equals the plain corrected lane fill for fill, no lane violations; on >= 1,000 CLEAN order-steps
                 the label from
                 the lane's own state equals the lane's fills (shares 1e-9, value 1e-6 c x sh) on 100%; controls: the cap removed and
                 the early window each differ somewhere. ENDED / SHARED shares and value gaps are printed (the single-order limit).
 E8 SANITY (descriptive, not a pass condition - Codex ffa2025e: a selected maker population cannot define the data's correctness)
                 STOP legs at the best (leg level, first minute, each leg once, 1 s) per share beside LEARN-PULL's +0.50 c for the top
                 makers' front fills; sweep fills at the best (beyond legs at j = 1, valued at p_1). A sign opposite to the expectation
                 (STOP <= 0 or sweep >= 0, dev pooled) -> FLAG named in the verdict, explained before stage 2.
READINGS (B4 discipline: tables for the dev days 10-23..26 only; test days get counts and the E-predicates):
 b COVERAGE   by level / side / time in window / volatility regime (step 5's sigma, dev terciles): rows, VIS_UNKNOWN / PRICE_OUT /
              INCOMPLETE, any STOP / THROUGH-at-price / sweep flow, fill share at the front / middle / back, rows with vis 0 (the grid's
              three positions identical - stage 2 deduplicates); book move through the level (STOP-MODEL D1 >= j) without a through trade
              and a through trade without that move; legs beyond the 5th shown level; the lane's states with a partial remainder (support).
 e TIMING     E2-anchored share of the front's fill shares and value by level; front value at M 0 / 50 / 250. Declared: E2 share of the
              front's fill shares at j = 1 > 30% -> E2-DEPENDENT (stage 2 reads by source, C12), else WS-DOMINANT.
 g SHAPE      by level: zero share of the front value, its quantiles (c per order), the top 1% of rows' share of |value|, STOP flow /
              visible queue. Declared loss rule at j = 1: zero share >= 50% -> HURDLE (P(fill) and the value given a fill) else DIRECT;
              top-1% share of |value| >= 30% -> ROBUST (Huber / clipped) else L2.
 q QUEUE      the lane's queue ahead / known visible size at its price by depth (stage 3's estimate; how real the grid is).
 r GRID       mean value per order at the front / middle / back by level (descriptive).
CONTROLS (synthetic; any failure -> no output): the single-order rule on hand cases (queue drain, beyond fill, visible-size cap,
  unavoidable legs, size cap, ask sign); types against DT.trade_type; the known-frame visible size; the population; components and the
  front against brute force; the INCOMPLETE flags against brute force (a feed gap, an untimed leg); the reconciliation on a toy two-sided
  lane window (cap binding, SHARED, ENDED); planted errors each caught: rule-M cap off, legs before t + 22 counted, THROUGH counted as
  STOP, venue-frame visible size, ask markout sign, invalid-book rows kept, INCOMPLETE flags off.
STAGE 2 NOTES (from the reviews, not built here): inputs use client-known state only (never the simulator's ahead / rem, delayed trade
  reports or unacknowledged fills); evaluation on signed value of neutral actions with matched controls and >= 200-draw nulls, day
  clusters; grid copies are correlated samples (weights / dedupe); baselines conditional on price, level, q and size; joint cap in stage 3.
LIMITS: one order per level (the joint lane's shared legs: SHARED status, measured); rule M is the replay's queue model (B1a: not
  validated on live orders); E2 anchors +-250 ms (C12); consumed days (exploratory); rows overlap (1 s windows every 50 ms) - per-row
  statistics describe decision states, never strategy P&L.
Usage (run_light, PF_SLOT=claude): pf_stop_ev.py selftest | day <day> | report; driver run_stop_ev.sh. PF_EV_SIZE (default 30) -> _s<Z>.
"""
import bisect
import collections
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_stop_model as M                                 # noqa: E402  (loader settings, rows, boundary cut, the lanes)
import pf_stop_queue_activation_lane as QA                # noqa: E402  (the corrected queue initialization, Codex 00971693 / a815aa8a)

K, CL, P1, QR, TS, DT = M.K, M.CL, M.P1, M.QR, M.TS, M.DT
DAYS9 = tuple(K.DAYS9); DEV = tuple(M.DEV_DAYS)
OUTD = f'{K.IO}/stop_ev_20261115'
JS = (1, 2, 3, 4, 5, 6); NR = int(CL.NR)
H_EV = 1000.0; L_C = float(K.L_CANCEL); SHF = 50.0; SHF_ALT = (0.0, 250.0)
Z = float(os.environ.get('PF_EV_SIZE', '30')); ZTAG = '' if Z == 30.0 else f'_s{int(Z)}'
COLS_L = ('vis', 'as', 'asv', 'at', 'atv', 'au', 'auv', 'bt', 'btv', 'f0', 'v0', 'v0x5', 'f0w', 'v0w', 'fh', 'vh', 'fb', 'vb', 'v0m0', 'v0m250')
NL = len(COLS_L); NCOL = NL * len(JS)
LCOLS = [f'{c}_{j}' for j in JS for c in COLS_L]; SCOLS = [f'st_{j}' for j in JS]
ST_OK, ST_PRICE_OUT, ST_VIS_UNKNOWN = 0, 1, 2
T_STOP, T_THR, T_UNK = 0, 1, 2
REC_DT = (('m', 'int64'), ('side', 'int8'), ('rung', 'int8'), ('k', 'int32'), ('pt', 'int16'), ('depth', 'int16'), ('ahead', 'float32'), ('visk', 'float32'),
          ('rem', 'float32'), ('Fl', 'float32'), ('Vl', 'float32'), ('Fb', 'float32'), ('Vb', 'float32'), ('ended', 'bool'), ('shared', 'bool'), ('match', 'bool'))
REC_COLS = tuple(c for c, _ in REC_DT)
EVERY_REC = 5; SAMPLE_WIN = 23; SAMPLE_ROW = 13; SAMPLE_BRUTE = 7
TOL_F = 1e-9; TOL_V = 1e-6; N_MIN = 2000; N_CLEAN_MIN = 1000; DUP_MAX = 0.001
MEM_CAP_MIB = 2048.0
INF = float('inf'); NAN = float('nan')
STALE_MS = 500.0; INC_STALE = 1; INC_UNTIMED = 2; ICOLS = [f'inc_{j}' for j in JS]
MUT_NO_CAP = MUT_WIN_EARLY = MUT_THR_AS_STOP = MUT_VENUE_VIS = MUT_ASK_SIGN = MUT_POP_ALL = MUT_INC_OFF = 0     # planted errors (selftest only)
MUTS = ('MUT_NO_CAP', 'MUT_WIN_EARLY', 'MUT_THR_AS_STOP', 'MUT_VENUE_VIS', 'MUT_ASK_SIGN', 'MUT_POP_ALL', 'MUT_INC_OFF')


# ---------------------------------------------------------------- the single-order rule and the legs
def leg_types(side, pr, grp):
    """Per leg: STOP if its taker order's worst price on the side is this leg's price, THROUGH if the order went deeper (DT.trade_type);
    UNK without a taker key (grp < 0)."""
    pr = np.asarray(pr, 'int64'); grp = np.asarray(grp, 'int64'); out = np.full(len(pr), T_UNK, 'int8'); ok = grp >= 0
    if ok.any():
        g = pd.Series(pr[ok]).groupby(grp[ok])
        worst = (g.transform('min') if side == 0 else g.transform('max')).values
        out[ok] = np.where(pr[ok] == worst, T_STOP, T_THR)
        if MUT_THR_AS_STOP:
            out[ok] = T_STOP
    return out


def sgn(side):
    return -1.0 if (side == 1 and not MUT_ASK_SIGN) else 1.0


def sim_one(L, idx, side, p, q0, rem, ci, lvi, tfl, t_cnt, cap=True):
    """CL.sim_joint's apply for ONE order at price p (queue ahead q0, rem shares, rule-M frame index ci at the decision) over the legs idx
    (time order, in (t, t + H], at or beyond p). Before a leg the queue is capped by the smallest visible size at p (lvi: NaN -> inf) over
    the venue frames after ci and before the leg (applying the cap only at the legs that reach the order gives the same minimum). Counts
    fills with tt >= t_cnt: (shares, 1 s value, 5 s value, WS shares, WS 1 s value), cents x shares."""
    sg = sgn(side); ahead = q0; F = V = V5 = Fw = Vw = 0.0
    tt = L['ttl']; pr = L['prl']; sh = L['shl']; m1 = L['m1l']; m5 = L['m5l']; src = L['srcl']
    for i in idx:
        ti = tt[i]
        if cap and ahead > 0.0 and not MUT_NO_CAP:
            j = bisect.bisect_left(tfl, ti) - 1
            if j > ci:
                mn = min(lvi[ci + 1:j + 1])
                if mn < ahead:
                    ahead = mn
                ci = j
        left = sh[i]
        if pr[i] == p:
            use = left if left < ahead else ahead; ahead -= use; q = min(left - use, rem)
        else:
            q = min(left, rem); ahead = 0.0
        if q > 1e-12:
            rem -= q
            if ti >= t_cnt or MUT_WIN_EARLY:
                v1 = q * sg * (m1[i] - p); F += q; V += v1; V5 += q * sg * (m5[i] - p)
                if src[i]:
                    Fw += q; Vw += v1
            if rem <= 1e-12:
                break
    return F, V, V5, Fw, Vw


class SideCtx:
    """One window-side's legs (in the lane's order: anchor, then price priority) and visible sizes, with every clock's leg lists."""

    def __init__(self, w, lrow, grp, src):
        self.w = w; self.side = int(w['side']); self.sg = 1.0 if self.side == 0 else -1.0
        self.tf = np.asarray(w['tf'], float); self.tfl = self.tf.tolist()
        self.anc = np.asarray(w['anc'], float); self.pr = np.asarray(w['ptl'], 'int64'); self.sh = np.asarray(w['shl'], float)
        mff = np.asarray(w['tf_mff'][1], float); n = len(self.anc)
        j1 = np.maximum(np.searchsorted(self.tf, self.anc + 1000.0, side='right') - 1, 0)
        j5 = np.maximum(np.searchsorted(self.tf, self.anc + 5000.0, side='right') - 1, 0)
        self.m1 = mff[j1] if n else np.zeros(0); self.m5 = mff[j5] if n else np.zeros(0)
        self.a1 = (self.anc + 1000.0 - self.tf[j1]) if n else np.zeros(0); self.a5 = (self.anc + 5000.0 - self.tf[j5]) if n else np.zeros(0)   # mark ages
        self.src = np.asarray(src, bool); self.grp = np.asarray(grp, 'int64'); self.row = np.asarray(lrow, 'int64')
        self.ty = leg_types(self.side, self.pr, self.grp)
        base = dict(prl=self.pr.tolist(), shl=self.sh.tolist(), m1l=self.m1.tolist(), m5l=self.m5.tolist(), srcl=self.src.tolist(), tyl=self.ty.tolist(),
                    stl=(self.a1 > STALE_MS).tolist())
        self.L = {s: dict(base, ttl=(self.anc - s).tolist()) for s in (SHF,) + SHF_ALT}
        self._lv = {}; self._lvi = {}

    def lv(self, p):
        """The lane's shown size at p per frame (CL.sim_joint lv): NaN beyond the 5th shown level, 0 inside the shown range if absent."""
        if p not in self._lv:
            P5 = np.asarray(self.w['P5']); S5 = np.asarray(self.w['S5'], float); last = P5[:, 4]
            self._lv[p] = np.where((last >= 0) & ((p < last) if self.side == 0 else (p > last)), np.nan, (S5 * (P5 == p)).sum(1))
        return self._lv[p]

    def lvi(self, p):
        if p not in self._lvi:
            a = self.lv(p); self._lvi[p] = np.where(np.isnan(a), INF, a).tolist()
        return self._lvi[p]

    def leg_table(self, m, ws):
        P5 = np.asarray(self.w['P5']); n = len(self.anc)
        jf = np.searchsorted(self.tf, self.anc, side='left') - 1; ok = jf >= 0; jj = np.maximum(jf, 0)
        b = np.where(ok, P5[jj, 0], -1) if n else np.zeros(0, 'int64'); last = np.where(ok, P5[jj, 4], -1) if n else np.zeros(0, 'int64')
        depth = np.where(ok & (b >= 0), (b - self.pr) if self.side == 0 else (self.pr - b), -99)
        bvis = ok & (last >= 0) & ((self.pr < last) if self.side == 0 else (self.pr > last))
        return dict(row=self.row, m=np.full(n, m, 'int64'), side=np.full(n, self.side, 'int8'), anc=self.anc, arel=self.anc - ws, pr=self.pr,
                    sh=self.sh, src=self.src, grp=self.grp, ty=self.ty, depth=depth.astype('int16'), bvis=bvis, m1=self.m1, m5=self.m5,
                    m1_age=self.a1.astype(np.float32), m5_age=self.a5.astype(np.float32))


def rel_legs(L, lo, hi, side, p):
    pr = L['prl']
    return [i for i in range(lo, hi) if (pr[i] <= p if side == 0 else pr[i] >= p)]


def side_rows(S, st, unt=()):
    """Label rows of one window-side (the STOP-MODEL population). Returns keys, X (rows x NCOL float32), statuses and INCOMPLETE flags
    (rows x 6 int8). unt: the side's untimed legs as (earliest tt, latest tt, price)."""
    w = S.w; n = int(w['n']); side = S.side; sg = sgn(side)
    if not n:
        return None
    _, valid, _, _ = QR.runs(w['tf'], w['P5'], w['opp'], side)
    jk = np.asarray(w['jk'], 'int64'); best = np.asarray(w['best'], 'int64'); tg = np.asarray(w['tg'], float)
    L = S.L[SHF]; ttl, prl, shl, m1l, tyl, stl = L['ttl'], L['prl'], L['shl'], L['m1l'], L['tyl'], L['stl']
    keep, X, STS, INC = [], [], [], []
    for k in range(n):
        st['steps'] += 1; j0 = int(jk[k])
        if not (j0 >= 0 and bool(valid[j0]) and best[k] >= 0) and not MUT_POP_ALL:
            st['invalid_book'] += 1; continue
        t = float(tg[k]); ref = int(best[k]); tc = t + L_C
        jv = (bisect.bisect_right(S.tfl, t) - 1) if MUT_VENUE_VIS else j0
        lo = bisect.bisect_right(ttl, t); hi = bisect.bisect_right(ttl, t + H_EV)
        alt = {a: (bisect.bisect_right(S.L[a]['ttl'], t), bisect.bisect_right(S.L[a]['ttl'], t + H_EV)) for a in SHF_ALT}
        x = [NAN] * NCOL; sts = [ST_OK] * len(JS); inc = [0] * len(JS)
        uk = [u for u in unt if u[0] <= t + H_EV and u[1] > t]
        for j in JS:
            b = (j - 1) * NL; p = ref - (j - 1) if side == 0 else ref + (j - 1)
            if p < 1 or p > 99:
                sts[j - 1] = ST_PRICE_OUT; continue
            vis = float(S.lv(p)[jv]) if jv >= 0 else NAN
            rel = rel_legs(L, lo, hi, side, p)
            c = [0.0] * 8
            for i in rel:
                if ttl[i] < tc and not MUT_WIN_EARLY:
                    continue
                if stl[i]:
                    inc[j - 1] |= INC_STALE
                q = shl[i]; v = q * sg * (m1l[i] - p)
                if prl[i] == p:
                    o = 2 * tyl[i]; c[o] += q; c[o + 1] += v
                else:
                    c[6] += q; c[7] += v
            if any((u[2] <= p if side == 0 else u[2] >= p) for u in uk):
                inc[j - 1] |= INC_UNTIMED
            if MUT_INC_OFF:
                inc[j - 1] = 0
            f = sim_one(L, rel, side, p, 0.0, Z, j0, None, S.tfl, tc, cap=False) if rel else (0.0, 0.0, 0.0, 0.0, 0.0)
            x[b] = vis; x[b + 1:b + 9] = c; x[b + 9:b + 14] = f
            if vis == vis:
                for o_, cq in ((14, 0.5), (16, 1.0)):
                    q0 = cq * vis
                    g = f if (q0 <= 0.0 or not rel) else sim_one(L, rel, side, p, q0, Z, j0, S.lvi(p), S.tfl, tc, cap=True)
                    x[b + o_] = g[0]; x[b + o_ + 1] = g[1]
            else:
                sts[j - 1] = ST_VIS_UNKNOWN
            for o_, a in ((18, 0.0), (19, 250.0)):
                La = S.L[a]; ra = rel_legs(La, alt[a][0], alt[a][1], side, p)
                x[b + o_] = sim_one(La, ra, side, p, 0.0, Z, j0, None, S.tfl, tc, cap=False)[1] if ra else 0.0
        keep.append(k); X.append(x); STS.append(sts); INC.append(inc)
    if not keep:
        return None
    ks = np.asarray(keep, 'int64')
    return dict(m=np.full(len(ks), int(w['m']), 'int64'), side=np.full(len(ks), side, 'int64'), k=ks, t=tg[ks], trel=(tg[ks] - float(w['ws'])).astype(np.float32),
                best=best[ks], X=np.asarray(X, np.float64).astype(np.float32), S=np.asarray(STS, 'int8'), I=np.asarray(INC, 'int8'))


# ---------------------------------------------------------------- E7: the keep-all lane's own order states
def rec_window(W, SS, st, ctl):
    """E7 on one window: keep-all client lane (align's KA, corrected queue initialization) with a recording callback that never pulls (its
    fills must equal the plain run's); per recorded order state the
    lane's fills of that order in [t + 22, t + 1 s] against the label simulated from the lane's own queue ahead / remaining size / frame."""
    n = int(W[0]['n']); aux = {0: [False] * n, 1: [False] * n}; R = []; objs = {}

    def mk(s, d):
        def dec(k, t, cur):
            if k % EVERY_REC == 0:
                R.append((s, d, k, float(t), int(cur['pt']), float(cur['ahead']), int(cur['ci']), float(cur['rem']), id(cur))); objs[id(cur)] = cur
            return False
        return dec
    r0 = QA.sim_joint_g(W, 'M', SHF, None, aux, decide=None, gate=None, idle_only=0, size=Z, mode='none', cooldown=CL.REPOST_MS)
    r = QA.sim_joint_g(W, 'M', SHF, None, aux, decide={(s, d): mk(s, d) for s in (0, 1) for d in range(NR)}, gate=None, idle_only=0, size=Z,
                       mode='none', cooldown=CL.REPOST_MS)
    st['rec_windows'] += 1; st['rec_regress_differ'] += int(r['fills'] != r0['fills'] or r['endpos'] != r0['endpos'])
    st['rec_viol'] += int(sum(r['stat']['viol'].values()))
    used = {0: {}, 1: {}}                                # side -> leg index -> prices of our orders that leg filled
    for (s, d), F in r['fills'].items():
        for f in F:
            used[s].setdefault(int(f[0]), []).append(int(f[1]))
    tf = np.asarray(W[0]['tf_mff'][0], float); mff = np.asarray(W[0]['tf_mff'][1], float)
    cols = collections.defaultdict(list)
    for (s, d, k, t, pt, ahead, ci, rem, oid) in R:
        o = objs[oid]; S = SS[s]; L = S.L[SHF]; tc = t + L_C; th = t + H_EV
        Fl = Vl = 0.0
        for (tq, qq) in o['fl']:
            if tc <= tq <= th:
                j1 = max(int(np.searchsorted(tf, tq + SHF + 1000.0, side='right')) - 1, 0)
                Fl += qq; Vl += qq * S.sg * (float(mff[j1]) - pt)
        rel = rel_legs(L, bisect.bisect_right(L['ttl'], t), bisect.bisect_right(L['ttl'], th), s, pt)
        lvi = S.lvi(pt)
        Fb, Vb = sim_one(L, rel, s, pt, ahead, rem, ci, lvi, S.tfl, tc, cap=True)[:2] if rel else (0.0, 0.0)
        ended = bool(o['tr'] <= th)
        shared = any(any(((p2 > pt) if s == 0 else (p2 < pt)) for p2 in used[s].get(i, ())) for i in rel)
        match = abs(Fb - Fl) <= TOL_F and abs(Vb - Vl) <= TOL_V
        if not ended and not shared and rel:
            if ctl['cap'] < 50 and ahead > 0.0:
                ctl['cap'] += int(abs(sim_one(L, rel, s, pt, ahead, rem, ci, lvi, S.tfl, tc, cap=False)[0] - Fl) > TOL_F)
            if ctl['early'] < 50:
                ctl['early'] += int(abs(sim_one(L, rel, s, pt, ahead, rem, ci, lvi, S.tfl, t, cap=True)[0] - Fl) > TOL_F)
        jk = int(W[s]['jk'][k]); b = int(W[s]['best'][k])
        for c_, v_ in (('side', s), ('rung', d), ('k', k), ('pt', pt), ('depth', ((b - pt) if s == 0 else (pt - b)) if b >= 0 else -99),
                       ('ahead', ahead), ('visk', float(S.lv(pt)[jk]) if jk >= 0 else NAN), ('rem', rem), ('Fl', Fl), ('Vl', Vl), ('Fb', Fb), ('Vb', Vb),
                       ('ended', ended), ('shared', shared), ('match', match)):
            cols[c_].append(v_)
    cols['m'] = [int(W[0]['m'])] * len(R)
    return {c: np.asarray(cols[c], dt) for c, dt in REC_DT}      # compact per window (Python lists of ~0.8M states a day would hold ~0.4 GB)


# ---------------------------------------------------------------- independent checks on the real day (E1-E6)
def e1_population(day, keys):
    R = pd.read_parquet(f'{M.OUTD}/rows_{day}.parquet', columns=['m', 'side', 'k', 'best'])

    def cmp(A):
        J = A.merge(R, on=['m', 'side', 'k'], how='outer', suffixes=('', '_sm'), indicator=True)
        both = J._merge.values == 'both'
        return dict(label_only=int((J._merge.values == 'left_only').sum()), stop_model_only=int((J._merge.values == 'right_only').sum()),
                    best_differ=int((J.best.values[both] != J.best_sm.values[both]).sum()), joined=int(both.sum()))
    x = cmp(keys); ctl = cmp(keys.iloc[1:])
    x['control_fires'] = bool(ctl['stop_model_only'] >= 1)
    x['pass'] = bool(x['label_only'] == 0 and x['stop_model_only'] == 0 and x['best_differ'] == 0 and x['control_fires'])
    return x


def e23_legs(day, wins, TAB):
    """E2 (legs reconciled with step 1, duplicates) and E3 (types) from an independent read of the day's legs."""
    T = pd.read_parquet(f'{K.IO}/step1/{day}/legs_timed2.parquet')
    tk = pd.read_parquet(f'{K.IO}/step1/{day}/legs_tk.parquet', columns=['tk']).tk.values.astype('int64')
    if len(tk) != len(T):
        return dict(pass_e2=False, pass_e3=False, why='legs_tk not aligned')
    rid = np.arange(len(T)); grp = np.where(tk >= 0, tk, -1 - rid)
    wsT = np.where(T.how.values > 0, T.t_ms.values.astype(float), np.nan)
    e2T = np.where((T.how.values == 0) & np.isfinite(T.t_e2.values.astype(float)), T.t_e2.values.astype(float), np.nan)
    a = pd.Series(wsT).groupby(grp).transform('min').values; b = pd.Series(e2T).groupby(grp).transform('min').values
    anc = np.where(np.isfinite(a), a, b)
    WS = pd.Series(wins, dtype=float); inc = T.mid.isin(WS.index).values; ws = T.mid.map(WS).values.astype(float)
    ex = T.exec_s.values.astype(float) * 1000.0
    fm = inc & (ex >= ws) & (ex < ws + 60000.0)
    unt = fm & ~np.isfinite(anc)
    inr = inc & np.isfinite(anc) & (anc >= ws - 2000.0) & (anc <= ws + 61000.0)
    out = fm & np.isfinite(anc) & ~inr; used = fm & inr
    tr = TAB.row.values.astype('int64'); trs = np.unique(tr)
    missing = np.setdiff1d(rid[used], trs); extra = np.setdiff1d(trs, rid[inr])
    sh = T.shares.values.astype(float)
    e2 = dict(first_minute_legs=int(fm.sum()), used=int(used.sum()), untimed=int(unt.sum()), out_of_range=int(out.sum()),
              shares_first_minute=float(sh[fm].sum()), shares_used=float(sh[used].sum()), shares_untimed=float(sh[unt].sum()), shares_out=float(sh[out].sum()),
              unaccounted=int(fm.sum() - used.sum() - unt.sum() - out.sum()), missing_from_table=int(len(missing)), table_rows_not_admissible=int(len(extra)),
              table_duplicate_rows=int(len(tr) - len(trs)), table_rows=int(len(tr)), table_tail_rows=int((~fm[tr]).sum()),
              anchor_differ=int((TAB.anc.values != anc[tr]).sum()))
    fmt = np.flatnonzero(fm[tr])                         # control: drop one first-minute leg from the table -> it must be missing
    e2['control_fires'] = bool(len(fmt) and len(np.setdiff1d(rid[used], np.unique(np.delete(tr, fmt[0])))) >= 1)
    C = [c for c in T.columns]; Tf = T.loc[fm, C].assign(_tk=tk[fm])
    dmask = Tf.duplicated(keep='first').values
    e2['exact_dup_legs'] = int(dmask.sum()); e2['exact_dup_share_of_shares'] = float(sh[fm][dmask].sum() / max(sh[fm].sum(), 1e-12))
    e2['pass'] = bool(e2['unaccounted'] == 0 and e2['missing_from_table'] == 0 and e2['table_rows_not_admissible'] == 0 and e2['table_duplicate_rows'] == 0
                      and e2['anchor_differ'] == 0 and e2['control_fires'])
    e2['flag_dup'] = bool(e2['exact_dup_share_of_shares'] > DUP_MAX)
    mp = tk >= 0
    G = pd.DataFrame(dict(tk=tk[mp], side=T.side.values[mp].astype('int64'), pr=T.ptick.values[mp].astype('int64'), mid=T.mid.values[mp]))
    nmid = G.groupby('tk').mid.transform('nunique').values; nside = G.groupby('tk').side.transform('nunique').values
    g2 = G.groupby(['tk', 'side']).pr
    worst = np.where(G.side.values == 0, g2.transform('min').values, g2.transform('max').values)
    ty = np.full(len(T), T_UNK, 'int8'); ty[mp] = np.where(G.pr.values == worst, T_STOP, T_THR)
    multi = np.zeros(len(T), bool); multi[mp] = (nmid > 1) | (nside > 1)
    ok3 = ~multi[tr]; tyt = TAB.ty.values.astype('int8')
    mism = int((tyt[ok3] != ty[tr][ok3]).sum())
    flip = tyt.copy()
    if len(flip):
        i0 = int(np.flatnonzero(ok3)[0]) if ok3.any() else 0; flip[i0] = T_THR if flip[i0] != T_THR else T_STOP
    e3 = dict(compared=int(ok3.sum()), mismatch=mism, multi_market_or_side_legs=int((~ok3).sum()), unk_legs=int((tyt == T_UNK).sum()),
              control_fires=bool(int((flip[ok3] != ty[tr][ok3]).sum()) >= 1))
    e3['pass'] = bool(mism == 0 and e3['control_fires'] and e3['compared'] > 0)
    return e2, e3, pd.Series(ty[tr], index=TAB.index)


def e456_books(day, wins, ROWS, ROWS4, TAB, TYI):
    """E4 (visible sizes, rows ROWS4), E5 (mids, legs TAB) from one independent read of the raw book files; E6 (components, rows ROWS)
    from the leg table with E3's independent types."""
    S1d = f'{K.IO}/step1/{day}'; mids = sorted(set(ROWS.m.values.tolist()))
    cols = ['mid', 't_ms', 'recv_ns'] + [f'{s}{i}_{x}' for s in 'ba' for i in range(1, 6) for x in 'ps']
    parts = []
    for fn in sorted(glob.glob(f'{S1d}/books/part_*.parquet')):
        if K.SKIP_FOREIGN_PARTS and not K.part_allowed(fn, day):
            continue
        d = pd.read_parquet(fn, columns=cols); d = d[d.mid.isin(mids)]
        rel = d.t_ms.values - d.mid.map(wins).values
        parts.append(d[(rel >= -K.PREROLL_MS) & (rel <= 125000.0)])
    B = pd.concat(parts, ignore_index=True); del parts
    ck = json.load(open(f'{S1d}/clock_rl_d2.json')); SH, DD = float(ck['shift_ms']), float(ck['d_dec'])
    e4 = dict(compared=0, mismatch=0, venue_differ=0); e5 = dict(compared=0, mismatch=0, early_differ=0)
    for m in mids:
        b = B[B.mid.values == m].sort_values('t_ms', kind='stable')
        tf = b.t_ms.values.astype(float); kf = np.maximum.accumulate(np.maximum(tf + K.PF_MIN, b.recv_ns.values / 1e6 - SH) + DD)
        bb = b.b1_p.values.astype('int64'); ba = b.a1_p.values.astype('int64'); ok = (bb >= 0) & (ba > bb)
        mid = pd.Series(np.where(ok, (ba + bb) / 2.0, np.nan)).ffill().values
        for side, pfx in ((0, 'b'), (1, 'a')):
            P = b[[f'{pfx}{i}_p' for i in range(1, 6)]].values.astype('int64'); Sz = b[[f'{pfx}{i}_s' for i in range(1, 6)]].values.astype(float)
            r = ROWS4[(ROWS4.m.values == m) & (ROWS4.side.values == side)]
            for t, ref, row in zip(r.t.values, r.best.values, r.index.values):
                jkn = int(np.searchsorted(kf, t, side='right')) - 1; jvn = int(np.searchsorted(tf, t, side='right')) - 1
                for j in JS:
                    p = int(ref) - (j - 1) if side == 0 else int(ref) + (j - 1)
                    if p < 1 or p > 99 or jkn < 0:
                        continue
                    def vis_at(jj):
                        last = P[jj, 4]
                        if last >= 0 and ((p < last) if side == 0 else (p > last)):
                            return np.float32(np.nan)
                        return np.float32(Sz[jj][P[jj] == p].sum())
                    v = vis_at(jkn); lab = ROWS4.at[row, f'vis_{j}']
                    e4['compared'] += 1; e4['mismatch'] += int(not ((np.isnan(v) and np.isnan(lab)) or v == lab))
                    if jvn >= 0:
                        vv = vis_at(jvn); e4['venue_differ'] += int(not ((np.isnan(vv) and np.isnan(lab)) or vv == lab))
            lg = TAB[(TAB.m.values == m) & (TAB.side.values == side)]
            for a1, m1, m5 in zip(lg.anc.values, lg.m1.values, lg.m5.values):
                i1 = max(int(np.searchsorted(tf, a1 + 1000.0, side='right')) - 1, 0); i5 = max(int(np.searchsorted(tf, a1 + 5000.0, side='right')) - 1, 0)
                i9 = max(int(np.searchsorted(tf, a1 + 900.0, side='right')) - 1, 0)
                e5['compared'] += 1; e5['mismatch'] += int(not (mid[i1] == m1 and mid[i5] == m5)); e5['early_differ'] += int(mid[i9] != m1)
    e4['pass'] = bool(e4['compared'] >= N_MIN and e4['mismatch'] == 0 and e4['venue_differ'] >= 1)
    e5['pass'] = bool(e5['compared'] >= N_MIN and e5['mismatch'] == 0 and e5['early_differ'] >= 1)
    e6 = dict(compared=0, mismatch=0, early_differ=0)
    TB = TAB.assign(tyi=TYI.values, tt=TAB.anc.values - SHF)
    names = ('as', 'asv', 'at', 'atv', 'au', 'auv', 'bt', 'btv')
    for (m, side), r in ROWS.groupby(['m', 'side'], sort=False):
        lg = TB[(TB.m.values == m) & (TB.side.values == side)]; tt = lg.tt.values; pr = lg.pr.values; sh = lg.sh.values; m1 = lg.m1.values; ty = lg.tyi.values
        sg = 1.0 if side == 0 else -1.0
        for row in r.index.values[::SAMPLE_BRUTE]:
            t = float(ROWS.at[row, 't']); ref = int(ROWS.at[row, 'best'])
            for j in JS:
                p = ref - (j - 1) if side == 0 else ref + (j - 1)
                if p < 1 or p > 99:
                    continue
                def comp(lo):
                    w_ = ((tt >= lo) if lo > t else (tt > t)) & (tt <= t + H_EV)
                    at_ = w_ & (pr == p); bt_ = w_ & ((pr < p) if side == 0 else (pr > p)); v = sh * sg * (m1 - p); out = []
                    for msk in (at_ & (ty == T_STOP), at_ & (ty == T_THR), at_ & (ty == T_UNK), bt_):
                        out += [sh[msk].sum(), v[msk].sum()]
                    return np.asarray(out, float)
                lab = np.asarray([ROWS.at[row, f'{c}_{j}'] for c in names], float)
                good = comp(t + L_C); early = comp(t)
                tol = 1e-3 + 1e-5 * np.abs(good)
                rel = (tt > t) & (tt <= t + H_EV) & ((pr <= p) if side == 0 else (pr >= p))     # the front order: min(Z, all) - min(Z, before t + 22)
                f0 = min(Z, float(sh[rel].sum())) - min(Z, float(sh[rel & (tt < t + L_C)].sum()))
                e6['compared'] += 1; e6['mismatch'] += int(np.any(np.abs(good - lab) > tol) or abs(f0 - float(ROWS.at[row, f'f0_{j}'])) > 1e-3)
                e6['early_differ'] += int(np.any(np.abs(early - lab) > tol))
    e6['pass'] = bool(e6['compared'] >= N_MIN and e6['mismatch'] == 0 and e6['early_differ'] >= 1)
    return e4, e5, e6


# ---------------------------------------------------------------- one day
def untimed_legs(day, D):
    """The day's UNTIMED legs (no WS / E2 anchor; the loader drops them with a count) per (m, side) as (earliest tt, latest tt, price):
    exec second + the 0.1-99.9% band of the timed legs' anchor - exec second offsets, on the M 50 clock (independent read of step 1)."""
    T = pd.read_parquet(f'{K.IO}/step1/{day}/legs_timed2.parquet', columns=['mid', 'exec_s', 'side', 'ptick', 'how', 't_ms', 't_e2'])
    tk = pd.read_parquet(f'{K.IO}/step1/{day}/legs_tk.parquet', columns=['tk']).tk.values.astype('int64')
    rid = np.arange(len(T)); grp = np.where(tk >= 0, tk, -1 - rid)
    wsT = np.where(T.how.values > 0, T.t_ms.values.astype(float), np.nan)
    e2T = np.where((T.how.values == 0) & np.isfinite(T.t_e2.values.astype(float)), T.t_e2.values.astype(float), np.nan)
    a = pd.Series(wsT).groupby(grp).transform('min').values; b = pd.Series(e2T).groupby(grp).transform('min').values
    anc = np.where(np.isfinite(a), a, b)
    off = D['LT'].anc.values - D['LT'].exec_s.values.astype(float) * 1000.0
    lo, hi = (float(x) for x in np.quantile(off, [0.001, 0.999]))
    u = ~np.isfinite(anc) & T.mid.isin(list(D['W0'])).values; out = {}
    for m, sd, e, pt in zip(T.mid.values[u], T.side.values[u], T.exec_s.values[u].astype(float) * 1000.0, T.ptick.values[u]):
        out.setdefault((int(m), int(sd)), []).append((e + lo - SHF, e + hi - SHF, int(pt)))
    return out, dict(band_ms=[lo, hi], untimed_legs_day_windows=int(u.sum()))


def mem_guard(what):
    M.trim(); r = K.rss_mib()
    if r > MEM_CAP_MIB:
        sys.exit(f'REFUSED: resident memory {r:.0f} MiB > {MEM_CAP_MIB:.0f} MiB ({what})')


def rows_frame(parts):
    return pd.concat([pd.DataFrame({'m': p['m'], 'side': p['side'], 'k': p['k'], 't': p['t'], 'trel': p['trel'], 'best': p['best']})
                      .join(pd.DataFrame(p['X'], columns=LCOLS)).join(pd.DataFrame(p['S'], columns=SCOLS)).join(pd.DataFrame(p['I'], columns=ICOLS))
                      for p in parts], ignore_index=True)


def day_build(day):
    import pyarrow as pa
    import pyarrow.parquet as pq
    all_selftests()
    if day not in DAYS9:
        sys.exit(f'REFUSED: {day} is not a consumed replay day')
    if M.QUNK != 'zero':
        sys.exit('REFUSED: E7 runs the lane with an unknown queue = 0 (QUNK zero), the primary assumption')
    T0 = time.time(); D = K.load_day(day); cut = M.boundary_cut(day)
    tkf = pd.read_parquet(f'{K.IO}/step1/{day}/legs_tk.parquet', columns=['tk']).tk.values
    LT = D['LT']; ix = LT.index.values; g_all = np.where(tkf[ix] >= 0, tkf[ix], -1 - ix)
    WSg = pd.Series(TS.ws_groups(LT.how.values, g_all), index=LT.index); GRP = pd.Series(g_all, index=LT.index)
    st = collections.Counter(); ctl = dict(cap=0, early=0); UNT, ust = untimed_legs(day, D)
    os.makedirs(OUTD, exist_ok=True); tmp = f'{OUTD}/rows_{day}{ZTAG}.parquet.tmp'; writer = None; buf = []
    legs, recs, samp, wins, keys = [], [], [], {}, []
    for W in P1.window_pairs(D):
        if W[0]['ws'] < cut:
            st['boundary_windows'] += 1; continue
        m = int(W[0]['m']); ws = float(W[0]['ws']); st['windows'] += 1; wins[m] = ws; SS = {}
        for s in (0, 1):
            w = W[s]
            if (m, s) in D['LI']:
                l = LT.iloc[D['LI'][(m, s)]]
                o_ = np.lexsort(((-l.ptick.values if s == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
            else:
                lrow = np.zeros(0, 'int64')
            if len(lrow) != len(w['ptl']) or (len(lrow) and not (np.array_equal(LT.loc[lrow, 'ptick'].values, np.asarray(w['ptl']))
                                                                  and np.array_equal(LT.loc[lrow, 'anc'].values, np.asarray(w['anc'])))):
                sys.exit(f'REFUSED: {day} window {m} side {s}: leg rows not aligned with the window-side legs')
            src = WSg.reindex(lrow).values.astype(bool) if len(lrow) else np.zeros(0, bool)
            grp = GRP.reindex(lrow).values.astype('int64') if len(lrow) else np.zeros(0, 'int64')
            S = SideCtx(w, lrow, grp, src); SS[s] = S
            st['clock_roundtrip_differ'] += int(not np.array_equal((S.anc - SHF) + SHF, S.anc))
            st['legs_mid_nan'] += int(np.isnan(S.m1).sum() + np.isnan(S.m5).sum())
            out = side_rows(S, st, UNT.get((m, s), ()))
            if out is not None:
                buf.append(out); keys.append(pd.DataFrame({'m': out['m'], 'side': out['side'], 'k': out['k'], 'best': out['best']}))
                st['rows'] += len(out['k'])
                for j in JS:
                    st[f'price_out_{j}'] += int((out['S'][:, j - 1] == ST_PRICE_OUT).sum()); st[f'vis_unknown_{j}'] += int((out['S'][:, j - 1] == ST_VIS_UNKNOWN).sum())
                    st[f'inc_stale_{j}'] += int((out['I'][:, j - 1] & INC_STALE > 0).sum()); st[f'inc_untimed_{j}'] += int((out['I'][:, j - 1] & INC_UNTIMED > 0).sum())
                if st['windows'] % SAMPLE_WIN == 1:
                    samp.append(out)
            legs.append(S.leg_table(m, ws))
        recs.append(rec_window(W, SS, st, ctl))
        if len(buf) >= 80:
            tb = pa.Table.from_pandas(rows_frame(buf), preserve_index=False)
            writer = writer or pq.ParquetWriter(tmp, tb.schema); writer.write_table(tb); buf = []
        if st['windows'] % 40 == 0:
            mem_guard(f'day {day}')
    if buf:
        tb = pa.Table.from_pandas(rows_frame(buf), preserve_index=False); writer = writer or pq.ParquetWriter(tmp, tb.schema); writer.write_table(tb); buf = []
    if writer is not None:
        writer.close()
    load_stat = D['stat']; del D, LT, WSg, GRP; mem_guard(f'day {day} loop done')
    TAB = pd.DataFrame({c: np.concatenate([x[c] for x in legs]) for c in legs[0]}); del legs
    REC = pd.DataFrame({c: np.concatenate([np.asarray(x[c]) for x in recs if len(x['m'])]) for c in REC_COLS}); del recs
    KEYS = pd.concat(keys, ignore_index=True); del keys
    ROWS = rows_frame(samp); del samp
    checks = {}
    checks['E1'] = e1_population(day, KEYS); del KEYS
    e2, e3, TYI = e23_legs(day, wins, TAB); checks['E2'] = e2; checks['E3'] = e3; mem_guard(f'day {day} legs checks')
    smp = TAB[TAB.m.isin(set(ROWS.m.values.tolist()))]
    ROWS4 = ROWS.groupby(['m', 'side'], sort=False, group_keys=False).apply(lambda g: g.iloc[::SAMPLE_ROW])
    e4, e5, e6 = e456_books(day, wins, ROWS, ROWS4, smp, TYI.reindex(smp.index))
    checks['E4'] = e4; checks['E5'] = e5; checks['E6'] = e6
    cl = ~REC.ended.values & ~REC.shared.values
    e7 = dict(order_steps=int(len(REC)), clean=int(cl.sum()), clean_mismatch=int((cl & ~REC.match.values).sum()), ended=int(REC.ended.sum()),
              shared=int(REC.shared.sum()), mismatch_ended_or_shared=int((~cl & ~REC.match.values).sum()),
              value_gap_ended_or_shared=float((REC.Vl.values - REC.Vb.values)[~cl].sum()), value_lane_ended_or_shared=float(REC.Vl.values[~cl].sum()),
              value_lane_all=float(REC.Vl.values.sum()), regress_windows=int(st['rec_windows']), regress_differ=int(st['rec_regress_differ']),
              lane_violations=int(st['rec_viol']), control_cap_differ=int(ctl['cap']), control_early_differ=int(ctl['early']))
    e7['pass'] = bool(e7['regress_differ'] == 0 and e7['lane_violations'] == 0 and e7['clean'] >= N_CLEAN_MIN and e7['clean_mismatch'] == 0
                      and e7['control_cap_differ'] >= 1 and e7['control_early_differ'] >= 1)
    checks['E7'] = e7
    fails = [e for e, x in checks.items() if not x['pass']]
    if st['clock_roundtrip_differ'] or st['legs_mid_nan']:
        fails.append(f"clock round trip {st['clock_roundtrip_differ']} / legs without a mid {st['legs_mid_nan']}")
    rep = dict(day=day, size=Z, stat=dict(st), load_stat=load_stat, checks=checks, flags=dict(dup=checks['E2']['flag_dup']), untimed=ust,
               queue_contract=QA.QUEUE_CONTRACT, clock=f'M {SHF:g}', horizon_ms=H_EV, seconds=round(time.time() - T0, 1), preroll_ms=K.PREROLL_MS,
               skip_foreign_parts=K.SKIP_FOREIGN_PARTS)
    if fails:
        if os.path.exists(tmp):
            os.remove(tmp)
        json.dump(rep, open(f'{OUTD}/refused_{day}{ZTAG}.json', 'w'), indent=1, default=str)
        sys.exit(f'REFUSED (no output for {day}): {fails}; detail in refused_{day}{ZTAG}.json')
    os.replace(tmp, f'{OUTD}/rows_{day}{ZTAG}.parquet')
    TAB.to_parquet(f'{OUTD}/legs_{day}{ZTAG}.parquet', index=False); REC.to_parquet(f'{OUTD}/rec_{day}{ZTAG}.parquet', index=False)
    json.dump(rep, open(f'{OUTD}/day_{day}{ZTAG}.json', 'w'), indent=1, default=str)
    print(f'{day}: rows {st["rows"]}, windows {st["windows"]} (boundary {st["boundary_windows"]}), invalid book {st["invalid_book"]}; '
          + '; '.join(f'{e} ' + ('PASS' if x['pass'] else 'FAIL') for e, x in checks.items()) + f'; {rep["seconds"]} s', flush=True)


# ---------------------------------------------------------------- report (E8 + readings)
def wavg(v, w):
    v = np.asarray(v, float); w = np.asarray(w, float); s = w.sum()
    return float(v.sum() / s) if s > 0 else NAN


def report():
    J = {d: json.load(open(f'{OUTD}/day_{d}{ZTAG}.json')) for d in DAYS9 if os.path.exists(f'{OUTD}/day_{d}{ZTAG}.json')}
    fails = [f'{d}: no output (refused or not run)' for d in DAYS9 if d not in J]
    fails += [f'{d}: {e}' for d, x in J.items() for e, y in x['checks'].items() if not y['pass']]
    flags = [f'{d}: step-1 exact duplicates {x["checks"]["E2"]["exact_dup_share_of_shares"]:.3%} of first-minute shares' for d, x in J.items() if x['flags']['dup']]
    T = [f'STOP-EV STAGE 1 - DATA (Z = {Z:g} sh, clock M {SHF:g}, horizon {H_EV:g} ms; {len(J)} of 9 days built)', '']
    T.append('DATA CHECKS E1-E7 per day (PASS / FAIL; counts)')
    for d, x in J.items():
        c = x['checks']
        T.append(f"  {d}: " + ' '.join(f"{e} {'PASS' if y['pass'] else 'FAIL'}" for e, y in c.items())
                 + f" | rows {x['stat'].get('rows', 0)}, legs used {c['E2']['used']} / untimed {c['E2']['untimed']} / out {c['E2']['out_of_range']}"
                 + f", dup share {c['E2']['exact_dup_share_of_shares']:.4%}, E4 {c['E4']['compared']}, E5 {c['E5']['compared']}, E6 {c['E6']['compared']}"
                 + f", E7 clean {c['E7']['clean']} / ended {c['E7']['ended']} / shared {c['E7']['shared']} of {c['E7']['order_steps']}")
    R = dict(days=list(J), fails=list(fails))
    # E8 sanity: leg level (STOP legs at the best, first minute) and sweep fills at the best (beyond legs at j = 1)
    e8a = {}; e8b = {}
    for d in J:
        L = pd.read_parquet(f'{OUTD}/legs_{d}{ZTAG}.parquet', columns=['side', 'arel', 'pr', 'sh', 'ty', 'depth', 'm1'])
        x = L[(L.arel.values >= 0) & (L.arel.values < 60000.0) & (L.ty.values == T_STOP) & (L.depth.values == 0)]
        sg = np.where(x.side.values == 0, 1.0, -1.0)
        e8a[d] = (float((x.sh.values * sg * (x.m1.values - x.pr.values)).sum()), float(x.sh.values.sum()))
        X = pd.read_parquet(f'{OUTD}/rows_{d}{ZTAG}.parquet', columns=['bt_1', 'btv_1'])
        e8b[d] = (float(np.nansum(X.btv_1.values.astype(float))), float(np.nansum(X.bt_1.values.astype(float))))
    dev = [d for d in DEV if d in J]
    a_pool = sum(e8a[d][0] for d in dev) / max(sum(e8a[d][1] for d in dev), 1e-12) if dev else NAN
    b_pool = sum(e8b[d][0] for d in dev) / max(sum(e8b[d][1] for d in dev), 1e-12) if dev else NAN
    a_days = sum(int(e8a[d][1] > 0 and e8a[d][0] > 0) for d in J); b_days = sum(int(e8b[d][1] > 0 and e8b[d][0] < 0) for d in J)
    e8 = dict(stop_at_best_c_per_sh_dev=a_pool, stop_at_best_positive_days=a_days, sweep_at_best_c_per_sh_dev=b_pool, sweep_negative_days=b_days)
    e8['flag'] = bool(not (a_pool > 0) or not (b_pool < 0))
    if e8['flag']:
        flags.append(f'E8 sign opposite to the expectation (STOP at the best {a_pool:+.3f}, sweeps {b_pool:+.3f} c/sh)')
    R['E8'] = e8
    T += ['', f"E8 SANITY (descriptive): STOP legs at the best {a_pool:+.3f} c/sh (dev pooled; the top makers' front fills on stopping trades +0.50, "
              f"LEARN-PULL), positive on {a_days} of {len(J)} days; sweep fills at the best {b_pool:+.3f} c/sh, negative on {b_days} of {len(J)} days"
              + (' -> FLAG' if e8['flag'] else '')]
    # E7 pooled (all days: a validity table, not a result)
    if J:
        e7 = {k_: sum(J[d]['checks']['E7'][k_] for d in J) for k_ in ('order_steps', 'clean', 'clean_mismatch', 'ended', 'shared', 'mismatch_ended_or_shared')}
        gap = sum(J[d]['checks']['E7']['value_gap_ended_or_shared'] for d in J); lane = sum(J[d]['checks']['E7']['value_lane_all'] for d in J)
        T.append(f"E7 RECONCILIATION pooled: {e7['clean']} clean order-steps of {e7['order_steps']}, clean mismatches {e7['clean_mismatch']}; "
                 f"ENDED {e7['ended']}, SHARED {e7['shared']}, mismatching among them {e7['mismatch_ended_or_shared']}; their value gap (lane - label) "
                 f"{gap / 100.0:+,.0f} $ against the lane's 1 s fill value {lane / 100.0:+,.0f} $ (the single-order label's limit)")
        R['E7_pooled'] = dict(e7, value_gap=gap, value_lane=lane)
    # readings on the dev days
    T += ['', f'READINGS (dev days {", ".join(dev)} only; test days: counts and predicates above)']
    agg = {j: collections.Counter() for j in JS}; sub = {j: [] for j in JS}; ratio = {j: [] for j in JS}
    side_t = {(j, s): collections.Counter() for j in JS for s in (0, 1)}; tb = {(j, b): collections.Counter() for j in (1, 2) for b in range(4)}
    leg_bvis = [0.0, 0.0]; QQ = []; reg = []
    for d in dev:
        cols = ['m', 'side', 'k', 'trel'] + SCOLS + ICOLS + [f'{c}_{j}' for j in JS for c in ('vis', 'as', 'at', 'au', 'bt', 'f0', 'v0', 'v0x5', 'f0w', 'v0w', 'fh',
                                                                                         'vh', 'fb', 'vb', 'v0m0', 'v0m250')]
        X = pd.read_parquet(f'{OUTD}/rows_{d}{ZTAG}.parquet', columns=cols)
        SMR = pd.read_parquet(f'{M.OUTD}/rows_{d}.parquet', columns=['m', 'side', 'k', 'sigma', 'D1'])
        X = X.merge(SMR, on=['m', 'side', 'k'], how='left', validate='one_to_one'); del SMR
        reg.append(pd.DataFrame({'sigma': X.sigma.values.astype(np.float32), **{f'{c}_{j}': X[f'{c}_{j}'].values.astype(np.float32) for j in (1, 2) for c in ('v0', 'f0')}}))
        sd = X.side.values; bkt = np.minimum((X.trel.values // 15000.0).astype(int), 3)
        for j in JS:
            stj = X[f'st_{j}'].values; okp = stj != ST_PRICE_OUT; a = agg[j]
            a['rows'] += len(stj); a['price_out'] += int((~okp).sum()); a['vis_unknown'] += int((stj == ST_VIS_UNKNOWN).sum())
            g = lambda c: X[f'{c}_{j}'].values.astype(float)
            asj, atj, buj, btj, f0, v0 = g('as'), g('at'), g('au'), g('bt'), g('f0'), g('v0')
            fh, fb, inc, d1 = g('fh'), g('fb'), X[f'inc_{j}'].values, X.D1.values.astype(float)
            for nm, msk in (('any_stop', asj > 0), ('any_thr_at', atj > 0), ('any_unk_at', buj > 0), ('any_sweep', btj > 0), ('front_fill', f0 > 0),
                            ('mid_fill', fh > 0), ('back_fill', fb > 0), ('vis_zero', g('vis') == 0), ('inc_stale', (inc & INC_STALE) > 0),
                            ('inc_untimed', (inc & INC_UNTIMED) > 0), ('book_through', d1 >= j), ('gone_no_trade', (d1 >= j) & ~(btj > 0)),
                            ('trade_no_move', (btj > 0) & (d1 < j)), ('d1_missing', ~np.isfinite(d1))):
                a[nm] += int((msk & okp).sum())
            for nm, c in (('f0', 'f0'), ('v0', 'v0'), ('v0x5', 'v0x5'), ('f0w', 'f0w'), ('v0w', 'v0w'), ('v0m0', 'v0m0'), ('v0m250', 'v0m250')):
                a[nm] += float(np.nansum(g(c)[okp]))
            for nm in ('vh', 'vb'):
                vv = g(nm); fin = np.isfinite(vv); a[nm] += float(vv[fin].sum()); a[f'n_{nm}'] += int(fin.sum())
                a[f'v0_on_{nm}'] += float(v0[fin].sum())
            a['zero_v0'] += int(((np.abs(v0) < 1e-9) & okp).sum()); a['n_v0'] += int(okp.sum())
            sub[j].append(v0[okp][::2].astype(np.float32))
            vis = g('vis'); vk = okp & np.isfinite(vis) & (vis > 0)
            ratio[j].append((asj[vk] / vis[vk])[::4].astype(np.float32))
            for s in (0, 1):
                ms = okp & (sd == s); c_ = side_t[(j, s)]; c_['rows'] += int(ms.sum()); c_['v0'] += float(np.nansum(v0[ms])); c_['fill'] += int((ms & (f0 > 0)).sum())
            if j in (1, 2):
                for bb in range(4):
                    ms = okp & (bkt == bb); c_ = tb[(j, bb)]; c_['rows'] += int(ms.sum()); c_['v0'] += float(np.nansum(v0[ms])); c_['fill'] += int((ms & (f0 > 0)).sum())
        del X
        L = pd.read_parquet(f'{OUTD}/legs_{d}{ZTAG}.parquet', columns=['arel', 'sh', 'bvis'])
        fm = (L.arel.values >= 0) & (L.arel.values < 60000.0); leg_bvis[0] += float(L.sh.values[fm & L.bvis.values].sum()); leg_bvis[1] += float(L.sh.values[fm].sum())
        QQ.append(pd.read_parquet(f'{OUTD}/rec_{d}{ZTAG}.parquet', columns=['depth', 'ahead', 'visk', 'ended', 'rem']))
    RD = {}
    if dev:
        T.append('b COVERAGE by level (share of level-steps; price-out excluded):  rows | VIS_UNKNOWN | any STOP flow | THROUGH-at-price | '
                 'no-key-at-price | sweep | front fill')
        for j in JS:
            a = agg[j]; n_ = max(a['rows'] - a['price_out'], 1)
            T.append(f"  j{j}: {a['rows']:9d} | {a['vis_unknown'] / n_:6.1%} | {a['any_stop'] / n_:6.1%} | {a['any_thr_at'] / n_:6.1%} | "
                     f"{a['any_unk_at'] / n_:6.1%} | {a['any_sweep'] / n_:6.1%} | {a['front_fill'] / n_:6.1%}")
        T.append(f"  legs beyond the 5th shown level: {leg_bvis[0] / max(leg_bvis[1], 1e-12):.2%} of first-minute leg shares")
        T.append('  INCOMPLETE / grid / book move vs trade (share of level-steps): stale mark | untimed leg may fall in | middle fill | back fill | '
                 'vis 0 (grid copies identical) | book through the level (D1 >= j) | gone without a through trade | through trade without that move')
        for j in JS:
            a = agg[j]; n_ = max(a['rows'] - a['price_out'], 1)
            T.append(f"  j{j}: {a['inc_stale'] / n_:6.2%} | {a['inc_untimed'] / n_:6.2%} | {a['mid_fill'] / n_:6.1%} | {a['back_fill'] / n_:6.1%} | "
                     f"{a['vis_zero'] / n_:6.1%} | {a['book_through'] / n_:6.1%} | {a['gone_no_trade'] / n_:6.1%} | {a['trade_no_move'] / n_:6.1%}"
                     + (f" (D1 missing {a['d1_missing']})" if a['d1_missing'] else ''))
        RG = pd.concat(reg, ignore_index=True); del reg
        sq = np.nanquantile(RG.sigma.values, [1 / 3, 2 / 3]); tr_ = np.digitize(RG.sigma.values, sq); tr_[~np.isfinite(RG.sigma.values)] = -1
        T.append(f'  by volatility regime (step 5 sigma terciles {sq[0]:.2e} / {sq[1]:.2e}; front value c per order, front fill share), low / mid / high:')
        for j in (1, 2):
            T.append(f'    j{j}: ' + ' | '.join(f"{np.nanmean(RG[f'v0_{j}'].values[tr_ == b]):+.3f}, {np.mean(RG[f'f0_{j}'].values[tr_ == b] > 0):.1%}"
                                             for b in range(3)) + f" (sigma missing {int((tr_ < 0).sum())})")
        del RG
        T.append('  by side (front value c per order | front fill share), bid / ask:')
        for j in JS:
            b_, a_ = side_t[(j, 0)], side_t[(j, 1)]
            T.append(f"    j{j}: {b_['v0'] / max(b_['rows'], 1):+.3f} | {b_['fill'] / max(b_['rows'], 1):.1%}   /   {a_['v0'] / max(a_['rows'], 1):+.3f} | "
                     f"{a_['fill'] / max(a_['rows'], 1):.1%}")
        T.append('  by time in window (0-15 / 15-30 / 30-45 / 45-60 s; front value c per order, front fill share):')
        for j in (1, 2):
            T.append(f'    j{j}: ' + ' | '.join(f"{tb[(j, b)]['v0'] / max(tb[(j, b)]['rows'], 1):+.3f}, {tb[(j, b)]['fill'] / max(tb[(j, b)]['rows'], 1):.1%}" for b in range(4)))
        e2s = {j: 1.0 - agg[j]['f0w'] / agg[j]['f0'] if agg[j]['f0'] > 0 else NAN for j in JS}
        e2v = {j: (agg[j]['v0'] - agg[j]['v0w']) for j in JS}
        timing = 'E2-DEPENDENT' if e2s[1] > 0.30 else 'WS-DOMINANT'
        T.append(f'e TIMING (front order): E2-anchored share of fill shares / value (c per order, E2 part) / front value at M 0 / 50 / 250 (c per order)')
        for j in JS:
            a = agg[j]; n_ = max(a['n_v0'], 1)
            T.append(f"  j{j}: {e2s[j]:6.1%} / {e2v[j] / n_:+.4f} / {a['v0m0'] / n_:+.4f} / {a['v0'] / n_:+.4f} / {a['v0m250'] / n_:+.4f}")
        T.append(f'  declared (E2 share at j1 > 30% -> E2-DEPENDENT): {timing} ({e2s[1]:.1%})')
        T.append('g SHAPE of the front value (c per order): zero share | p1 / p5 / p25 / p50 / p75 / p95 / p99 | top-1% share of |value| | '
                 'STOP flow / visible queue p50 / p90 / p99 (rows with vis > 0)')
        shp = {}
        for j in JS:
            v = np.concatenate(sub[j]) if sub[j] else np.zeros(0, np.float32); a = agg[j]
            z0 = a['zero_v0'] / max(a['n_v0'], 1); q = np.quantile(v, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]) if len(v) else [NAN] * 7
            av = np.sort(np.abs(v))[::-1]; top = float(av[:max(len(av) // 100, 1)].sum() / max(av.sum(), 1e-12)) if len(av) else NAN
            rr = np.concatenate(ratio[j]) if ratio[j] else np.zeros(0); rq = np.quantile(rr, [0.5, 0.9, 0.99]) if len(rr) else [NAN] * 3
            shp[j] = dict(zero=z0, q=[float(x) for x in q], top1=top, ratio_q=[float(x) for x in rq])
            T.append(f'  j{j}: {z0:6.1%} | ' + ' / '.join(f'{x:+.2f}' for x in q) + f' | {top:6.1%} | ' + ' / '.join(f'{x:.2f}' for x in rq))
        loss = ('HURDLE' if shp[1]['zero'] >= 0.5 else 'DIRECT') + ' + ' + ('ROBUST' if shp[1]['top1'] >= 0.30 else 'L2')
        T.append(f"  declared loss rule (j1: zero share >= 50% -> HURDLE; top-1% share >= 30% -> ROBUST): {loss}")
        Q = pd.concat(QQ, ignore_index=True) if QQ else pd.DataFrame(columns=['depth', 'ahead', 'visk', 'ended']); del QQ
        T.append("q QUEUE (keep-all lane's orders, corrected initialization, every 5th step): by depth behind the known best - steps | queue ahead / "
                 "visible p10 / p50 / p90 | at the front (ahead 0) | at or behind the visible size | visible unknown | partial remainder (support)")
        qd = {}
        for dd in range(0, 6):
            x = Q[Q.depth.values == dd]; vk = x.visk.values.astype(float); ah = x.ahead.values.astype(float); fin = np.isfinite(vk) & (vk > 0)
            rq = np.quantile(ah[fin] / vk[fin], [0.1, 0.5, 0.9]) if fin.any() else [NAN] * 3
            qd[dd] = dict(steps=int(len(x)), q=[float(v) for v in rq], front=float((ah == 0).mean()) if len(x) else NAN,
                          back=float((ah[fin] >= vk[fin]).mean()) if fin.any() else NAN, vis_unknown=float((~np.isfinite(vk)).mean()) if len(x) else NAN,
                          partial=float((x.rem.values.astype(float) < Z - 1e-6).mean()) if len(x) else NAN)
            T.append(f"  d{dd}: {len(x):8d} | " + ' / '.join(f'{v:.2f}' for v in rq) + f" | {qd[dd]['front']:.1%} | {qd[dd]['back']:.1%} | {qd[dd]['vis_unknown']:.1%}"
                     f" | {qd[dd]['partial']:.1%}")
        T.append('r GRID: mean value per order, front (all rows) | front / middle / back on rows with the visible size known (c per order; c/sh = / Z)')
        for j in JS:
            a = agg[j]
            T.append(f"  j{j}: {a['v0'] / max(a['n_v0'], 1):+.3f} | {a['v0_on_vh'] / max(a['n_vh'], 1):+.3f} / {a['vh'] / max(a['n_vh'], 1):+.3f} / "
                     f"{a['vb'] / max(a['n_vb'], 1):+.3f}")
        RD = dict(agg={j: dict(agg[j]) for j in JS}, shape=shp, loss=loss, timing=timing, e2_share=e2s, queue=qd,
                  beyond_visible_share=leg_bvis[0] / max(leg_bvis[1], 1e-12))
    R['readings_dev'] = RD
    verdict = ('DATA-PASS' if not flags else 'DATA-PASS-WITH-FLAGS') if not fails else 'DATA-FAIL'
    T += ['', f"VERDICT: {verdict}" + ('' if not fails else ' - ' + '; '.join(fails)) + ('' if not flags else ' | FLAGS: ' + '; '.join(flags))]
    R['verdict'] = verdict; R['flags'] = flags
    open(f'{OUTD}/stop_ev_data{ZTAG}.txt', 'w').write('\n'.join(T) + '\n')
    json.dump(R, open(f'{OUTD}/stop_ev_data{ZTAG}.json', 'w'), indent=1, default=str)
    print('\n'.join(T), flush=True)


# ---------------------------------------------------------------- self-tests (synthetic, no data files)
def toy_w(side, legs, n=40, drop=None, invalid=None, known_lag=60.0, mid2=50.5, t_mid=600.0, gap=None):
    """A synthetic window-side: venue frames every 25 ms known `known_lag` ms later, a contiguous 5-level book at 50..46 (bids) /
    52..56 (asks), 100 shares a level; drop = (t0, level, size) from frame time t0; invalid = (t0, t1): a crossed book in those frames;
    legs = [(anchor, price, shares, taker group)]; mid 51, then mid2 from t_mid. Returns (w, leg rows, groups, WS flags)."""
    tg = np.arange(0.0, 50.0 * n, 50.0); tf = np.arange(-1000.0, 50.0 * n + 3000.0, 25.0)
    if gap:
        tf = tf[(tf < gap[0]) | (tf >= gap[1])]           # a feed gap: no frames in [gap0, gap1)
    nf = len(tf)
    lvl = np.array([50, 49, 48, 47, 46]) if side == 0 else np.array([52, 53, 54, 55, 56])
    P5 = np.tile(lvl, (nf, 1)).astype('int64'); S5 = np.full((nf, 5), 100.0)
    if drop:
        S5[tf >= drop[0], drop[1]] = drop[2]
    opp = np.full(nf, 52 if side == 0 else 50, 'int64')
    if invalid:
        opp[(tf >= invalid[0]) & (tf < invalid[1])] = 50 if side == 0 else 52
    kf = tf + known_lag; jk = np.searchsorted(kf, tg, side='right') - 1
    valid = (opp > P5[:, 0]) if side == 0 else (P5[:, 0] > opp)
    best = [int(P5[j, 0]) if (j >= 0 and valid[j]) else -1 for j in jk]
    mff = np.where(tf >= t_mid, mid2, 51.0)
    lg = sorted(legs, key=lambda x: (x[0], -x[1] if side == 0 else x[1]))
    anc = np.asarray([x[0] for x in lg], float); pt = [int(x[1]) for x in lg]; sh = [float(x[2]) for x in lg]
    w = dict(m=1, side=side, sg=1.0 if side == 0 else -1.0, ws=0.0, te=float(50.0 * n), tf=tf, kf=kf, jk=jk, tg=tg, n=n, opp=opp, P5=P5, S5=S5,
             best=best, bopp=[52 if side == 0 else 50] * n, vkl=[0.0] * n, anc=anc, ptl=pt, shl=sh, tf_mff=(tf, mff))
    return w, np.arange(len(lg), dtype='int64'), np.asarray([x[3] for x in lg], 'int64'), np.asarray([x[3] % 2 == 0 for x in lg], bool)


def brute_front(L, side, p, t, z):
    """Front order (nothing ahead): every leg at or beyond p fills it up to z; counted = cumulative fill by t + 1 s - by t + 22 ms."""
    tt, pr, sh = L['ttl'], L['prl'], L['shl']
    rel = [i for i in range(len(tt)) if t < tt[i] <= t + H_EV and (pr[i] <= p if side == 0 else pr[i] >= p)]
    tot = sum(sh[i] for i in rel); pre = sum(sh[i] for i in rel if tt[i] < t + L_C)
    return min(z, tot) - min(z, pre)


def selftest():
    bad = []
    # (1) the single-order rule on hand cases (bid at 50; frames every 100 ms; the visible size at 50 drops to 15 from frame 3)
    tfl = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]; lvi = [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
    Lh = dict(ttl=[10.0, 30.0, 150.0, 320.0, 400.0], prl=[50, 50, 49, 50, 50], shl=[5.0, 8.0, 6.0, 20.0, 30.0], m1l=[49.0] * 5, m5l=[48.0] * 5, srcl=[True] * 5)
    f = sim_one(Lh, range(5), 0, 50, 20.0, 30.0, 0, lvi, tfl, 22.0)
    if abs(f[0] - 30.0) > 1e-9 or abs(f[1] + 30.0) > 1e-9 or abs(f[2] + 60.0) > 1e-9:
        bad.append(f'hand case A (drain, beyond fill, size cap) {f}')
    Lb = dict(ttl=[320.0], prl=[50], shl=[20.0], m1l=[49.0], m5l=[49.0], srcl=[False])
    f = sim_one(Lb, [0], 0, 50, 30.0, 30.0, 0, lvi, tfl, 22.0)
    if abs(f[0] - 5.0) > 1e-9 or f[3] != 0.0:
        bad.append(f'hand case B (visible-size cap) {f}')
    Lc = dict(ttl=[10.0, 30.0], prl=[50, 50], shl=[5.0, 8.0], m1l=[49.0] * 2, m5l=[49.0] * 2, srcl=[True] * 2)
    f = sim_one(Lc, [0, 1], 0, 50, 0.0, 30.0, 0, lvi, tfl, 22.0)
    if abs(f[0] - 8.0) > 1e-9:
        bad.append(f'hand case C (legs before t + 22 not counted) {f}')
    Ld = dict(ttl=[100.0], prl=[52], shl=[10.0], m1l=[51.0], m5l=[51.0], srcl=[True])
    f = sim_one(Ld, [0], 1, 52, 0.0, 30.0, 0, [100.0] * 6, tfl, 22.0)
    if abs(f[1] - 10.0) > 1e-9:
        bad.append(f'hand case D (ask sold at 52, mid 51 after 1 s: +1 c/sh) {f}')
    # (2) types against DT.trade_type
    rng = np.random.default_rng(7)
    for side in (0, 1):
        pr = rng.integers(40, 60, 200); grp = rng.integers(-3, 30, 200); ty = leg_types(side, pr, grp)
        for i in range(200):
            if grp[i] < 0:
                ok = ty[i] == T_UNK
            else:
                ok = ty[i] == (T_STOP if DT.trade_type(side, int(pr[i]), pr[grp == grp[i]].tolist()) == 'STOP' else T_THR)
            if not ok:
                bad.append(f'types side {side} leg {i}'); break
    # (3) a window-side: population, known-frame visible size, components and front against brute force
    for side in (0, 1):
        p0 = 50 if side == 0 else 52; dpx = -1 if side == 0 else 1
        legs = [(float(a), int(p0 + dpx * int(rng.integers(0, 4))), float(rng.integers(1, 40)), int(rng.integers(-2, 25))) for a in rng.uniform(0.0, 2500.0, 90)]
        legs += [(1012.0, p0, 7.0, 40), (1012.0, p0 + dpx, 9.0, 40)]        # one taker order through the best (THROUGH at p0)
        w, lrow, grp, src = toy_w(side, legs, drop=(510.0, 0, 20.0), invalid=(900.0, 1000.0), gap=(1600.0, 2300.0))
        unt = [(950.0, 1350.0, p0 + dpx)]                 # an untimed leg one tick behind the best, possible tt 950..1350 ms
        S = SideCtx(w, lrow, grp, src); st = collections.Counter(); out = side_rows(S, st, unt)
        n_inv = sum(1 for k in range(w['n']) if not (w['jk'][k] >= 0 and w['best'][k] >= 0))
        if not (S.a1 > STALE_MS).any():
            bad.append(f'toy side {side}: the feed gap made no stale mark (the INCOMPLETE check would be vacuous)')
        if out is None or len(out['k']) != w['n'] - n_inv or n_inv == 0:
            bad.append(f'population side {side}: rows {None if out is None else len(out["k"])}, n {w["n"]}, invalid {n_inv}'); continue
        X = out['X']; pos = {int(k): i for i, k in enumerate(out['k'])}
        if 11 in pos and abs(X[pos[11], 0] - 100.0) > 1e-6:
            bad.append(f'visible size side {side}: known frame expected 100 at t 550, got {X[pos[11], 0]}')
        L = S.L[SHF]; sg = 1.0 if side == 0 else -1.0; tyind = {}
        for i in range(len(L['prl'])):
            g_ = int(S.grp[i]); tyind[i] = T_UNK if g_ < 0 else (T_STOP if DT.trade_type(side, L['prl'][i], [L['prl'][q] for q in range(len(L['prl'])) if S.grp[q] == g_]) == 'STOP' else T_THR)
        for r_, k in enumerate(out['k']):
            t = float(w['tg'][k]); ref = int(w['best'][k])
            for j in JS:
                p = ref - (j - 1) if side == 0 else ref + (j - 1); b = (j - 1) * NL
                c = [0.0] * 8
                for i in range(len(L['ttl'])):
                    if not (t + L_C <= L['ttl'][i] <= t + H_EV):
                        continue
                    q = L['shl'][i]; v = q * sg * (L['m1l'][i] - p)
                    if L['prl'][i] == p:
                        c[2 * tyind[i]] += q; c[2 * tyind[i] + 1] += v
                    elif (L['prl'][i] < p if side == 0 else L['prl'][i] > p):
                        c[6] += q; c[7] += v
                if np.any(np.abs(np.asarray(c) - X[r_, b + 1:b + 9]) > 1e-3):
                    bad.append(f'components side {side} k {k} j {j}: {c} vs {X[r_, b + 1:b + 9].tolist()}'); break
                fb = brute_front(L, side, p, t, Z)
                if abs(fb - X[r_, b + 9]) > 1e-4:
                    bad.append(f'front side {side} k {k} j {j}: {fb} vs {X[r_, b + 9]}'); break
                reach = lambda q_: (q_ <= p if side == 0 else q_ >= p)
                e_inc = (INC_STALE if any(t + L_C <= L['ttl'][i] <= t + H_EV and reach(L['prl'][i]) and S.a1[i] > STALE_MS for i in range(len(L['ttl']))) else 0) \
                    | (INC_UNTIMED if any(u[0] <= t + H_EV and u[1] > t and reach(u[2]) for u in unt) else 0)
                if int(out['I'][r_, j - 1]) != e_inc:
                    bad.append(f'INCOMPLETE flags side {side} k {k} j {j}: {int(out["I"][r_, j - 1])} vs {e_inc}'); break
            if bad:
                break
    # (4) the reconciliation on a toy two-sided lane window (cap binding, SHARED, ENDED)
    global EVERY_REC
    keep_every = EVERY_REC; EVERY_REC = 1
    try:
        bid = [(710.0, 50, 30.0, 2), (910.0, 50, 25.0, 4), (1210.0, 47, 40.0, 6), (1260.0, 49, 12.0, 8)]
        ask = [(300.0, 52, 15.0, 3), (800.0, 53, 40.0, 5), (1500.0, 52, 18.0, 7)]
        (w0, r0_, g0, s0), (w1, r1_, g1, s1) = toy_w(0, bid, drop=(500.0, 0, 20.0)), toy_w(1, ask)
        W = {0: w0, 1: w1}; SS = {0: SideCtx(w0, r0_, g0, s0), 1: SideCtx(w1, r1_, g1, s1)}; st = collections.Counter(); ctl = dict(cap=0, early=0)
        c = rec_window(W, SS, st, ctl)
        cl = [not e and not s_ for e, s_ in zip(c['ended'], c['shared'])]
        nclean = sum(cl); cm = sum(1 for x, m_ in zip(cl, c['match']) if x and not m_)
        fills_clean = sum(1 for x, f_ in zip(cl, c['Fl']) if x and f_ > 0)
        ask_clean = sum(1 for x, s_, f_ in zip(cl, c['side'], c['Fl']) if x and s_ == 1 and f_ > 0)
        if st['rec_regress_differ'] or st['rec_viol'] or nclean < 10 or cm or fills_clean < 3 or ask_clean < 1 or not any(c['shared']) or not any(c['ended']) \
                or ctl['cap'] < 1 or ctl['early'] < 1:
            bad.append(f'reconciliation toy: regress {st["rec_regress_differ"]}, viol {st["rec_viol"]}, clean {nclean}, clean mismatches {cm}, '
                       f'clean with fills {fills_clean} (asks {ask_clean}), shared {sum(c["shared"])}, ended {sum(c["ended"])}, controls {ctl}')
    finally:
        EVERY_REC = keep_every
    return bad


def mutation_check():
    out = {}
    for nm in MUTS:
        globals()[nm] = 1
        try:
            out[nm] = bool(selftest())
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    bad = selftest()
    if bad:
        sys.exit(f'STOP-EV SELFTEST FAILED: {bad}')
    m = mutation_check()
    if not all(m.values()):
        sys.exit(f'STOP-EV PLANTED ERROR NOT CAUGHT: {m}')
    print(f'stop-ev selftests passed; planted errors caught: {m}', flush=True)
    return m


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        all_selftests()
    elif cmd == 'day':
        day_build(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit(__doc__)
