"""B12b (2026-11-11, USER "Yes find out the reason"): why the top makers' fills move less than others' at the same
price (queue place, or a better / faster guard), and why ours move most. Read-only, descriptive. Follows B12
(pf_b12_makers.py d5f68cf2, output 2026-11-10 16:43Z) and its addendum (195aa071). Same population: maker legs of the
220 consumed 11-08 cohort windows (cut 23:00Z, 0.764 window-days), first minute, research trade time anc (one time per
taker order). G = 1 day < 5: point estimates only.

Input checks before this script (structure only; no outcome by maker group was looked at):
 - Matching is price-time (FIFO)-like: over the cohort windows, 99.6% of taker-order levels of <= 1 sh and 94.2% of
   1-5 sh fill exactly one maker leg; legs per level grow with size (p50 6 above 100 sh). Pro-rata matching would fill
   most makers at a ~300 sh level in every trade. So the place in the queue decides whom a trade that stops at the
   level fills. Recomputed below as R42.
 - Book frames are ~51 ms apart in the first minute (p90 83 ms, p99 176 ms; 60 windows).
 - Live's cancel round trip (cancel request to cancel ack received) p50 19 ms, p90 33 ms; request to ack p50 34 ms
   (core_orders_v6).

Part A, queue reach. The top 5 (T) against ranks >= 6 (O); at-best legs, guard ON / OFF at lag 50, trades that do NOT
go through the level (sweep 0). Per leg: s = the taker order's shares at this price on this side (all makers' legs of
that taker order there); Q = the size shown at the best in the frame strictly before anc; reach r = s / Q. Under
price-time matching a maker filled by a trade of reach r sat within the first r of the queue as shown: a maker at the
front is filled by trades of every reach, one at the back only by trades of high reach. Buckets: r <= 0.10, 0.10-0.25,
0.25-0.50, 0.50-1.00, > 1.00 (size added after the frame, or a stale frame); s <= 5, 5-20, 20-50, 50-150, > 150 sh.
Decomposition as B12 (D = MIX + WITHIN) on cells G = guard (2), GR = guard x reach (10), GS = guard x size (10,
descriptive), GRS = guard x reach x size (50, descriptive; a cell that one group lacks is excluded and its volume share
reported). X = (WITHIN_G - WITHIN_GR) / WITHIN_G: the part of the within-guard gap that the reach mix accounts for.

Part B, exits around our guard's flips. Events: on a (window, side), a grid point where the side's guard goes from ON
to OFF (fresh points, gap <= 100 ms) with the guard ON over the 1,500 ms before and OFF over the 500 ms after (clean
flips), at t_f in [ws + 1.5 s, ws + 59.5 s]. Windows by research trade time relative to t_f: B baseline [-1500, -500)
ms, L the last ON 500 ms [-500, 0), P after the flip [0, +500). Share_T(W) = top-5 shares / (top 5 + O) shares of the
at-best legs on the event's side, pooled over events; ours apart.

Part C, our at-best first-minute legs with the guard OFF at lag 50 (B12's 25 legs) against live's order paths
(core_orders v6 + basis, linked on the nine content keys through the fill-clock audit's clock_cases): our cancel
request t_cancel against the trade at anc - shift, shift 0 / 50 / 250 ms (a match precedes its report by an unknown
lag). Descriptive: the start t_f of the research OFF run, anc - t_f and t_cancel - t_f; the same for our guard-ON
at-best legs; live's pull delay t_cancel - t_f for every accepted live order resting 500 ms before a research ON -> OFF
flip of its side in the first minute.

Readings DECLARED BEFORE THE FIRST RUN (evaluated in code; h = 1 s, lag 50):
 R40 reach    'n/a (matching)' unless R42 is FIFO-LIKE. NO-GAP if WITHIN_G < +0.10 c/sh. QUEUE-EXPLAINS if X >= 0.5 and
              WITHIN_GR is not above its null p99 (N2); NOT-QUEUE if X < 0.25 and WITHIN_GR is above the null p99; else
              PARTLY-QUEUE.
 R41 place    share-weighted median reach of the Part A legs: FRONT if the top 5's <= 0.8 x O's; BACK if >= 1.25 x O's;
              else SAME-PLACE. OURS-BEHIND if ours >= 1.25 x the top 5's (n/a under 20 own legs), else
              OURS-NOT-BEHIND. 'n/a (matching)' unless FIFO-LIKE.
 R42 matching share of the cohort's taker-order levels with 0 < s <= 5 sh that fill exactly one maker leg: FIFO-LIKE
              if >= 0.7; PRO-RATA-LIKE if <= 0.3; else MIXED.
 R43 exits    n/a under 100 events or 50 top-5 legs in either window. After: rho = Share_T(P) / Share_T(B);
              TOP-LEAVES-FIRST if rho <= 0.80 and Share_T(P) - Share_T(B) is below its null p01 (N3); TOP-STAYS if
              rho >= 1.25 and it is above the null p99; else SAME-EXIT. Before: the same with L in place of P:
              TOP-LEAVES-BEFORE / TOP-STAYS-BEFORE / SAME-BEFORE.
 R44 our late fills  shift 0, n/a under 10 legs: CANCEL-IN-FLIGHT if >= 2/3 of the legs' shares traded at or after
              our cancel request; LATE-DECISION if <= 1/3 did (the rest traded before we sent it, or the order was never
              cancelled); else MIXED. Shifts 50 / 250 reported with the same rule.
 R45 main reason (top 5 against O): q = R40 QUEUE-EXPLAINS; nq = R40 NOT-QUEUE; s = R43 TOP-LEAVES-FIRST or
              TOP-LEAVES-BEFORE. QUEUE if q and not s; QUEUE+SPEED if q and s; SPEED/SIGNAL if s and not q; OWN-SIGNAL
              if nq and not s; else UNRESOLVED. If both R43 readings are n/a, ' (exits n/a)' is appended.
 Nulls, 500 draws each, seed 20261111: N2 the T / O label permuted within (window, side, GR cell) for WITHIN_GR; N3 the
 window label (B or P; B or L) permuted among an event's taker orders within the event.
What the outcomes mean (fixed in advance): QUEUE = at the same reach the top 5 are filled by trades as bad as
everyone's; they do better because more of their fills come from trades that stop near the front of the queue: their
place in it. SPEED/SIGNAL = they are out of the book sooner when our guard sees a move (or before it): faster or
earlier pulls. OWN-SIGNAL = at the same reach they are still filled by better trades, and they do not leave sooner on
our guard's flips: information our guard does not have (a better guard). LATE-DECISION = our engine had not sent the
cancel when the trade was reported (the decision is late: our signal or its latency). CANCEL-IN-FLIGHT = the cancel
had been sent and the trade beat it (the venue round trip).

Controls (any failure -> no table): K0 the rebuilt frame reproduces the committed b12.json and b12_posthoc.json
(hash-checked): T1 legs, shares and net 1 s for every group, the lag-50 decomposition D and WITHIN (1e-9), and our
at-best guard-OFF legs (25 legs, 160.0 sh); B12's controls (pf_b12_makers.controls) and its K1-K3 checks. A1 the reach
and size buckets and the taker-level sums on planted values. A2 the n-cell decomposition identity on random cells and
two planted cases (a pure reach mix: X = 1; a within-reach +0.5 skill: X = 0). A3 the null flags a planted within-cell
skill and not its absence. B1 the flip finder on planted series (a clean flip found at its time; a flicker, a stale
gap, a warm-up start and an out-of-range time refused). B2 the exit statistic on planted events (the top-5 share halved
after the flip -> TOP-LEAVES-FIRST; unchanged -> SAME-EXIT). C1 the late-fill classes on planted orders (cancel before
/ after / never; shift) and an own leg linked to two orders flagged as a status. M1 the matching statistic on planted
price-time and pro-rata fills. Refusals: no cell shared by both groups; a NaN reads as n/a, never as a label.
Limits fixed in advance: one consumed Sunday; trade times are reports (the match time is unknown, B1a); the shown
queue is a ~50 ms snapshot that includes our own orders and cannot show cancels ahead; noise in reach biases X toward
0 (toward NOT-QUEUE); the live guard is not the research guard; values are of the flow the makers received, not of an
order we can place.
Usage: predictfun/research/run_light.sh b12b_reason -- python3 -m predictfun.research.pf_b12b_reason <out_dir>
       python3 -m predictfun.research.pf_b12b_reason --selftest
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from . import pf_b12_makers as B

RP = B.RP; S1 = B.S1; FC = B.FC; KEYS = B.KEYS
LV = f'{RP}/live_vs_replay'
B12_OUT = f'{LV}/b12_20261110'
B12_SHA = {'b12.json': '9c22ec526d9fa6dad4642ffcdc90cc891cb0e66eba9f0fe6209de0673cd5e31e',
           'b12_posthoc.json': '0f14b007e4dbf716d2bdff1d0c2505d14f249ba426fd16465a6a834980f5eff4'}
CUTOFF_MS = 1794178800000.0
RUNS = ('v6', 'basis')
LAG0 = 50.0
R_EDGES = np.array([0.10, 0.25, 0.50, 1.00]); R_LAB = ['<= 0.10', '0.10-0.25', '0.25-0.50', '0.50-1.00', '> 1.00']
S_EDGES = np.array([5.0, 20.0, 50.0, 150.0]); S_LAB = ['<= 5', '5-20', '20-50', '50-150', '> 150']
EV_ON, EV_OFF = 1500.0, 500.0
WIN = {'B': (-1500.0, -500.0), 'L': (-500.0, 0.0), 'P': (0.0, 500.0)}
SHIFTS = (0.0, 50.0, 250.0)
N_NULL = 500; SEED = 20261111
MIN_EVENTS = 100; MIN_TOP_LEGS = 50; MIN_OWN = 20; MIN_LATE = 10
GR = ['top 5', '6-20', 'rest', 'O (ranks >= 6)', 'ours']
NAN = float('nan')


# ---- instruments ----
def bucket(x, edges):
    """0 for x <= edges[0], i for edges[i-1] < x <= edges[i], len(edges) above the last edge."""
    return np.searchsorted(edges, x, side='left').astype(np.int64)


def taker_level(grp, side, pt, sh):
    """Per leg: the taker order's shares at this price on this side, and its number of maker legs there."""
    d = pd.DataFrame(dict(g=grp, s=side, p=pt, x=sh))
    gb = d.groupby(['g', 's', 'p'], sort=False).x
    return gb.transform('sum').values, gb.transform('size').values


def decompose_n(cell, isT, sh, val, n):
    """D = MIX + WITHIN over the cells that both groups have (weights renormalized there); the share of each group's
    volume in cells the other lacks is reported as excluded. Refuses when no cell is shared."""
    aT = np.bincount(cell[isT], sh[isT], n); aO = np.bincount(cell[~isT], sh[~isT], n); both = (aT > 0) & (aO > 0)
    if not both.any():
        raise ValueError('no cell shared by both groups')
    sT = np.bincount(cell[isT], sh[isT] * val[isT], n); sO = np.bincount(cell[~isT], sh[~isT] * val[~isT], n)
    vT = np.divide(sT, aT, out=np.zeros(n), where=both); vO = np.divide(sO, aO, out=np.zeros(n), where=both)
    wT = np.where(both, aT, 0.0); wT = wT / wT.sum(); wO = np.where(both, aO, 0.0); wO = wO / wO.sum()
    return dict(D=float(np.sum(wT * vT) - np.sum(wO * vO)), MIX=float(np.sum((wT - wO) * vO)), WITHIN=float(np.sum(wT * (vT - vO))),
                excl_T=float(1.0 - aT[both].sum() / aT.sum()), excl_O=float(1.0 - aO[both].sum() / aO.sum()),
                wT=wT, vT=np.where(both, vT, np.nan), wO=wO, vO=np.where(both, vO, np.nan))


def within_n(cell, isT, sh, val, n):
    aT = np.bincount(cell[isT], sh[isT], n); aO = np.bincount(cell[~isT], sh[~isT], n); both = (aT > 0) & (aO > 0)
    if not both.any():
        return NAN
    sT = np.bincount(cell[isT], sh[isT] * val[isT], n); sO = np.bincount(cell[~isT], sh[~isT] * val[~isT], n)
    vT = np.divide(sT, aT, out=np.zeros(n), where=both); vO = np.divide(sO, aO, out=np.zeros(n), where=both)
    w = np.where(both, aT, 0.0)
    return float(np.sum(w / w.sum() * (vT - vO)))


def null_within_n(cell, isT, sh, val, n, block, draws, seed):
    """WITHIN with the T / O label permuted within blocks."""
    rng = np.random.default_rng(seed); o = np.argsort(block, kind='stable'); res = np.empty(draws)
    for d in range(draws):
        o2 = np.lexsort((rng.random(len(block)), block)); lab = np.empty_like(isT); lab[o] = isT[o2]
        res[d] = within_n(cell, lab, sh, val, n)
    return res


def wmedian(x, w):
    if len(x) == 0 or w.sum() <= 0:
        return NAN
    o = np.argsort(x, kind='stable'); c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def gstates(v, line):
    return np.where(v == -1.0, B.G_WARM, np.where(v > line, B.G_ON, B.G_OFF)).astype(np.int8)


def flips(tau, v, line, lo, hi, clean=True):
    """ON -> OFF flips at grid points k (tau[k] - tau[k-1] <= FRESH) with t = tau[k] in [lo, hi]. A state holds from its
    grid point to the next one, or for FRESH ms when the next is later. clean: the guard ON over [t - EV_ON, t) and OFF
    over [t, t + EV_OFF)."""
    n = len(tau)
    if n < 2:
        return np.array([])
    st = gstates(v, line); fr = np.diff(tau) <= B.FRESH
    out = []
    for k in np.nonzero((st[1:] == B.G_OFF) & (st[:-1] == B.G_ON) & fr)[0] + 1:
        t = tau[k]
        if t < lo or t > hi:
            continue
        if not clean:
            out.append(t); continue
        i = k - 1
        while i - 1 >= 0 and st[i - 1] == B.G_ON and fr[i - 1]:
            i -= 1
        j = k
        while j + 1 < n and st[j + 1] == B.G_OFF and fr[j]:
            j += 1
        off_end = tau[j + 1] if (j + 1 < n and fr[j]) else tau[j] + B.FRESH
        if tau[i] <= t - EV_ON and off_end >= t + EV_OFF:
            out.append(t)
    return np.array(out)


def off_run_start(tau, v, line, t):
    """Start of the OFF run holding at time t and what precedes it ('after ON', 'after warm-up or gap'); 'not OFF' if the
    state at t is not OFF."""
    st = gstates(v, line); j = np.searchsorted(tau, t, side='right') - 1
    if j < 0 or t - tau[j] > B.FRESH or st[j] != B.G_OFF:
        return NAN, 'not OFF'
    i = j
    while i - 1 >= 0 and st[i - 1] == B.G_OFF and tau[i] - tau[i - 1] <= B.FRESH:
        i -= 1
    if i - 1 >= 0 and st[i - 1] == B.G_ON and tau[i] - tau[i - 1] <= B.FRESH:
        return float(tau[i]), 'after ON'
    return float(tau[i]), 'after warm-up or gap'


def exit_delta(lab, uT, uTO):
    """Share_T(window 1) - Share_T(window 0) over units (taker orders) with their T and T + O shares."""
    a = uTO[lab == 1].sum(); b = uTO[lab == 0].sum()
    if a <= 0 or b <= 0:
        return NAN, NAN, NAN
    s1 = uT[lab == 1].sum() / a; s0 = uT[lab == 0].sum() / b
    return s1 - s0, s1, s0


def null_exit(ev, lab, uT, uTO, draws, seed):
    rng = np.random.default_rng(seed); o = np.argsort(ev, kind='stable'); res = np.empty(draws)
    for d in range(draws):
        o2 = np.lexsort((rng.random(len(ev)), ev)); l2 = np.empty_like(lab); l2[o] = lab[o2]
        res[d] = exit_delta(l2, uT, uTO)[0]
    return res


def late_class(anc, t_cancel, shift):
    """1 if the trade (anc - shift) is at or after our cancel request, 0 if before it or the order was never cancelled."""
    return np.where(np.isfinite(t_cancel) & (t_cancel <= anc - shift), 1, 0)


def link_orders(legs, cc):
    """Own legs -> live order cid through clock_cases on the nine keys; a leg matching two different cids is a status."""
    j = legs[KEYS + ['lid']].merge(cc[KEYS + ['cid']], on=KEYS, how='left')
    nu = j.groupby('lid').cid.nunique().reindex(legs.lid).values; fi = j.groupby('lid').cid.first().reindex(legs.lid).values
    st = np.where(nu == 1, 'linked', np.where(nu == 0, 'no order', 'ambiguous order'))
    return np.where(nu == 1, fi, -1).astype(np.int64), st


def matching_stat(s_lvl, n_lvl):
    m = (s_lvl > 0) & (s_lvl <= 5.0)
    return float(np.mean(n_lvl[m] == 1)) if m.any() else NAN


# ---- labels ----
def fin(*x):
    return all(isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v) for v in x)


def label_r42(x):
    return 'n/a' if not fin(x) else ('FIFO-LIKE' if x >= 0.7 else ('PRO-RATA-LIKE' if x <= 0.3 else 'MIXED'))


def label_r40(Wg, Wgr, p99, fifo):
    if not fifo:
        return 'n/a (matching)'
    if not fin(Wg, Wgr, p99):
        return 'n/a'
    if Wg < 0.10:
        return 'NO-GAP'
    X = (Wg - Wgr) / Wg; above = Wgr > p99
    if X >= 0.5 and not above:
        return 'QUEUE-EXPLAINS'
    if X < 0.25 and above:
        return 'NOT-QUEUE'
    return 'PARTLY-QUEUE'


def label_r41(mT, mO, mU, nU, fifo):
    if not fifo:
        return ['n/a (matching)', 'n/a (matching)']
    a = 'n/a' if not fin(mT, mO) else ('FRONT' if mT <= 0.8 * mO else ('BACK' if mT >= 1.25 * mO else 'SAME-PLACE'))
    b = 'n/a' if (nU < MIN_OWN or not fin(mU, mT)) else ('OURS-BEHIND' if mU >= 1.25 * mT else 'OURS-NOT-BEHIND')
    return [a, b]


def label_r43(s1, s0, delta, p01, p99, n_ev, nT1, nT0, names):
    if n_ev < MIN_EVENTS or nT1 < MIN_TOP_LEGS or nT0 < MIN_TOP_LEGS or not fin(s1, s0, delta, p01, p99) or s0 <= 0:
        return 'n/a'
    rho = s1 / s0
    if rho <= 0.80 and delta < p01:
        return names[0]
    if rho >= 1.25 and delta > p99:
        return names[1]
    return names[2]


def label_r44(sh, cls, n):
    if n < MIN_LATE or sh.sum() <= 0:
        return 'n/a'
    f = sh[cls == 1].sum() / sh.sum()
    return 'CANCEL-IN-FLIGHT' if f >= 2.0 / 3.0 else ('LATE-DECISION' if f <= 1.0 / 3.0 else 'MIXED')


def label_r45(r40, ra, rb):
    q = r40 == 'QUEUE-EXPLAINS'; nq = r40 == 'NOT-QUEUE'; s = (ra == 'TOP-LEAVES-FIRST') or (rb == 'TOP-LEAVES-BEFORE')
    lab = 'QUEUE' if (q and not s) else 'QUEUE+SPEED' if (q and s) else 'SPEED/SIGNAL' if s else 'OWN-SIGNAL' if nq else 'UNRESOLVED'
    return lab + (' (exits n/a)' if (ra == 'n/a' and rb == 'n/a') else '')


# ---- controls ----
def controls():
    rep = {}; ok = True

    def chk(name, cond):
        nonlocal ok
        rep[name] = bool(cond); ok = ok and bool(cond)
    ok0, rep0 = B.controls(); chk('B12_controls', ok0)
    # A1 buckets and taker-level sums
    a1 = list(bucket(np.array([0.05, 0.10, 0.11, 0.25, 0.30, 0.50, 0.75, 1.00, 1.50]), R_EDGES)) == [0, 0, 1, 1, 2, 2, 3, 3, 4]
    a1 = a1 and list(bucket(np.array([3.0, 5.0, 6.0, 20.0, 21.0, 50.0, 100.0, 150.0, 151.0]), S_EDGES)) == [0, 0, 1, 1, 2, 2, 3, 3, 4]
    s_, n_ = taker_level(np.array([1, 1, 1, 2, 2, 3]), np.array([0, 0, 0, 0, 1, 0]), np.array([50, 50, 49, 50, 60, 50]), np.array([2.0, 3, 4, 5, 6, 7]))
    chk('A1_buckets_levels', a1 and list(s_) == [5, 5, 4, 5, 6, 7] and list(n_) == [2, 2, 1, 1, 1, 1])
    # A2 decomposition identity and planted cases (cells = 5 * off + reach bucket)
    rng = np.random.default_rng(3)
    cell = rng.integers(0, 10, 5000); isT = rng.random(5000) < 0.3; sh = rng.random(5000) * 10; val = rng.normal(0, 2, 5000) + cell * 0.2
    d = decompose_n(cell, isT, sh, val, 10); a2 = abs(d['MIX'] + d['WITHIN'] - d['D']) < 1e-12 and d['excl_T'] == 0 and d['excl_O'] == 0
    vr = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
    rT = np.r_[np.zeros(40, int), np.ones(30, int), np.full(15, 2), np.full(10, 3), np.full(5, 4)]
    rO = np.r_[np.zeros(10, int), np.ones(15, int), np.full(25, 2), np.full(30, 3), np.full(20, 4)]
    off = np.r_[np.zeros(50, int), np.ones(50, int)]               # the same guard mix for both groups
    cA = np.r_[5 * off + rT, 5 * off + rO]; tA = np.r_[np.ones(100, bool), np.zeros(100, bool)]; vA = vr[cA % 5] + 0.3 * (cA // 5)
    WgA = decompose_n(cA // 5, tA, np.ones(200), vA, 2)['WITHIN']; WgrA = decompose_n(cA, tA, np.ones(200), vA, 10)['WITHIN']
    cB = np.r_[5 * off + rO, 5 * off + rO]; vB = vr[cB % 5] + np.where(tA, 0.5, 0.0)
    WgB = decompose_n(cB // 5, tA, np.ones(200), vB, 2)['WITHIN']; WgrB = decompose_n(cB, tA, np.ones(200), vB, 10)['WITHIN']
    a2 = a2 and WgA > 0.10 and abs(WgrA) < 1e-12 and abs((WgA - WgrA) / WgA - 1.0) < 1e-12
    a2 = a2 and abs(WgB - 0.5) < 1e-12 and abs(WgrB - 0.5) < 1e-12
    a2 = a2 and label_r40(WgA, WgrA, 0.05, True) == 'QUEUE-EXPLAINS' and label_r40(WgB, WgrB, 0.05, True) == 'NOT-QUEUE'
    a2 = a2 and label_r40(WgB, WgrB, 0.05, False) == 'n/a (matching)' and label_r40(0.05, 0.0, 0.01, True) == 'NO-GAP'
    try:
        decompose_n(np.array([0, 0, 1, 1]), np.array([True, True, False, False]), np.ones(4), np.zeros(4), 2); a2 = False
    except ValueError:
        pass
    chk('A2_decomposition', a2)
    # A3 the null flags a planted within-cell skill and not its absence
    nb = 40; m = 60; blk = np.repeat(np.arange(nb), m); cl = blk % 10; isT3 = rng.random(nb * m) < 0.4; sh3 = 1.0 + rng.random(nb * m)
    noise = rng.normal(0, 1, nb * m); flags = []
    for plant in (1.0, 0.0):
        v3 = noise + np.where(isT3, plant, 0.0); obs = within_n(cl, isT3, sh3, v3, 10)
        nul = null_within_n(cl, isT3, sh3, v3, 10, blk, 200, 7); flags.append(bool(obs > np.percentile(nul, 99)))
    chk('A3_null', flags == [True, False])
    # B1 flip finder
    tau = np.arange(0.0, 5001.0, 50.0)
    v1 = np.where(tau < 2000, 0.06, np.where(tau < 2700, 0.03, np.where((tau >= 3000) & (tau < 3100), 0.03, 0.06)))
    f1 = flips(tau, v1, B.LINE_DECL, 0.0, 1e9)
    f1o = flips(tau, v1, B.LINE_DECL, 2100.0, 1e9)                  # the clean flip out of range
    tau2 = np.r_[np.arange(0.0, 1001.0, 50.0), np.arange(1200.0, 4001.0, 50.0)]   # a 200 ms gap inside the ON run
    v2 = np.where(tau2 < 2400, 0.06, 0.03)
    v3_ = np.where(tau < 1000, -1.0, np.where(tau < 2400, 0.06, 0.03))   # warm-up placeholder before a 1,400 ms ON run
    fa = flips(tau, v1, B.LINE_DECL, 0.0, 1e9, clean=False)
    os_, ost = off_run_start(tau, v1, B.LINE_DECL, 2300.0)
    chk('B1_flips', list(f1) == [2000.0] and len(f1o) == 0 and len(flips(tau2, v2, B.LINE_DECL, 0.0, 1e9)) == 0
        and len(flips(tau, v3_, B.LINE_DECL, 0.0, 1e9)) == 0 and list(fa) == [2000.0, 3000.0]
        and os_ == 2000.0 and ost == 'after ON' and off_run_start(tau, v1, B.LINE_DECL, 1500.0)[1] == 'not OFF'
        and off_run_start(tau, v3_, B.LINE_DECL, 500.0)[1] == 'not OFF')
    # B2 exit statistic: the top-5 share halved after the flip, or unchanged
    ev = np.repeat(np.arange(200), 20); lab = np.tile(np.r_[np.zeros(10, int), np.ones(10, int)], 200)
    uTO = 10.0 + rng.random(4000)
    res = []
    for sP in (0.15, 0.30):
        uT = uTO * np.where(lab == 1, sP, 0.30) * (1.0 + 0.05 * rng.normal(size=4000))
        dl, s1, s0 = exit_delta(lab, uT, uTO); nul = null_exit(ev, lab, uT, uTO, 200, 11)
        res.append(label_r43(s1, s0, dl, np.percentile(nul, 1), np.percentile(nul, 99), 200, 2000, 2000,
                             ('TOP-LEAVES-FIRST', 'TOP-STAYS', 'SAME-EXIT')))
    chk('B2_exits', res == ['TOP-LEAVES-FIRST', 'SAME-EXIT'] and label_r43(0.1, 0.3, -0.2, -0.01, 0.01, 50, 900, 900, ('a', 'b', 'c')) == 'n/a')
    # C1 late classes, labels and the order link
    lc0 = late_class(np.array([1000.0, 1000.0, 1000.0]), np.array([900.0, 1100.0, np.nan]), 0.0)
    lc2 = late_class(np.array([1000.0, 1000.0, 1000.0]), np.array([900.0, 1100.0, np.nan]), 250.0)
    c1 = list(lc0) == [1, 0, 0] and list(lc2) == [0, 0, 0]
    c1 = c1 and label_r44(np.ones(12), np.r_[np.ones(4), np.zeros(8)].astype(int), 12) == 'LATE-DECISION'
    c1 = c1 and label_r44(np.ones(12), np.r_[np.ones(8), np.zeros(4)].astype(int), 12) == 'CANCEL-IN-FLIGHT'
    c1 = c1 and label_r44(np.ones(12), np.r_[np.ones(6), np.zeros(6)].astype(int), 12) == 'MIXED' and label_r44(np.ones(5), np.ones(5, int), 5) == 'n/a'
    kk = {k: [1, 1, 2] for k in KEYS}; legs = pd.DataFrame(kk); legs['lid'] = [0, 1, 2]
    cc = pd.DataFrame({k: [1, 1, 2] for k in KEYS}); cc['cid'] = [7, 8, 9]
    cid, st = link_orders(legs, cc)
    c1 = c1 and list(st) == ['ambiguous order', 'ambiguous order', 'linked'] and list(cid) == [-1, -1, 9]
    chk('C1_late_fills', c1)
    # M1 matching statistic: price-time (one leg for small trades) against pro-rata (every maker at the level)
    sF = np.r_[np.full(80, 3.0), np.full(20, 40.0)]; nF = np.r_[np.ones(76), np.full(4, 2.0), np.full(20, 3.0)]
    sP_ = np.full(100, 3.0); nP_ = np.full(100, 8.0)
    chk('M1_matching', label_r42(matching_stat(sF, nF)) == 'FIFO-LIKE' and label_r42(matching_stat(sP_, nP_)) == 'PRO-RATA-LIKE'
        and label_r42(NAN) == 'n/a')
    # NaN -> n/a and the main-reason rule
    chk('NA_and_R45', label_r41(NAN, 0.5, 0.5, 30, True) == ['n/a', 'n/a'] and label_r41(0.3, 0.5, 0.9, 30, True) == ['FRONT', 'OURS-BEHIND']
        and label_r45('QUEUE-EXPLAINS', 'SAME-EXIT', 'SAME-BEFORE') == 'QUEUE'
        and label_r45('QUEUE-EXPLAINS', 'TOP-LEAVES-FIRST', 'SAME-BEFORE') == 'QUEUE+SPEED'
        and label_r45('PARTLY-QUEUE', 'SAME-EXIT', 'TOP-LEAVES-BEFORE') == 'SPEED/SIGNAL'
        and label_r45('NOT-QUEUE', 'SAME-EXIT', 'SAME-BEFORE') == 'OWN-SIGNAL'
        and label_r45('PARTLY-QUEUE', 'SAME-EXIT', 'SAME-BEFORE') == 'UNRESOLVED'
        and label_r45('QUEUE-EXPLAINS', 'n/a', 'n/a') == 'QUEUE (exits n/a)')
    rep['B12_controls_detail'] = rep0
    return ok, rep


# ---- data ----
def build():
    """B12's frame (pf_b12_makers.main, data part, unchanged) plus the taker order, its size at the level, the shown size
    at the best, the nine keys and the guard series."""
    sc = json.load(open(f'{RP}/scope.json')); W0 = {int(w['market']): w['ws'] * 1000.0 for w in sc['windows']}; days = len(W0) / 288.0
    LINE = float(json.load(open(f'{RP}/research/step4/minute_lines_t60k.json'))['lines']['90'][0])
    k3 = dict(line=LINE, ok=abs(LINE - B.LINE_DECL) < 1e-4)
    L = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=KEYS + ['signer', 't_rel', 't_ms', 'how', 't_e2'])
    tk = pd.read_parquet(f'{S1}/legs_tk.parquet', columns=['tk']).tk.values
    if len(tk) != len(L):
        sys.exit('REFUSED: legs_tk not aligned with legs_timed2')
    grp = np.where(tk >= 0, tk, -1 - np.arange(len(L)))
    ws_t = np.where(L.how.values > 0, L.t_ms.values.astype(float), np.nan)
    e2_t = np.where((L.how.values == 0) & np.isfinite(L.t_e2.values), L.t_e2.values, np.nan)
    a_ = pd.Series(ws_t).groupby(grp, sort=False).transform('min').values
    b_ = pd.Series(e2_t).groupby(grp, sort=False).transform('min').values
    L['anc'] = np.where(np.isfinite(a_), a_, b_); L['grp'] = grp; L['tk_ok'] = tk >= 0
    L['sweep'] = B.sweep_flag(grp, L.side.values, L.ptick.values, tk >= 0)
    L['s_lvl'], L['n_lvl'] = taker_level(grp, L.side.values, L.ptick.values, L.shares.values)
    L = L[L.mid.isin(W0)].copy(); L['ws'] = L.mid.map(W0)
    lv = L[L.tk_ok].drop_duplicates(['grp', 'side', 'ptick'])
    r42 = dict(stat=matching_stat(lv.s_lvl.values, lv.n_lvl.values), levels=int(((lv.s_lvl > 0) & (lv.s_lvl <= 5)).sum()))
    r42['label'] = label_r42(r42['stat'])
    first = (L.exec_s * 1000.0 >= L.ws) & (L.exec_s * 1000.0 < L.ws + 60000.0)
    unt = L[~np.isfinite(L.anc.values) & first]
    T0 = L[np.isfinite(L.anc.values)]; in60 = T0[(T0.anc >= T0.ws) & (T0.anc < T0.ws + 60000.0)]
    k1 = dict(untimed_legs=len(unt), untimed_shares=float(unt.shares.sum()), legs_0_60=len(in60), shares_0_60=float(in60.shares.sum()))
    k1['ok'] = all(abs(k1[k] - v) <= 1e-6 for k, v in B.QR_STAT.items())
    if hashlib.sha256(open(f'{FC}/clock_cases.parquet', 'rb').read()).hexdigest() != B.FC_SHA['clock_cases.parquet']:
        sys.exit('REFUSED: clock_cases.parquet hash differs from the fill-clock audit receipt')
    own = pd.read_parquet(f'{FC}/clock_cases.parquet')
    k2 = B.own_id(L, own); k2f = B.own_id(L, own, shuffle=B.SEED)
    k2['falsifier_shuffled_fails'] = not k2f['ok']
    if not (k1['ok'] and k2['ok'] and k2['falsifier_shuffled_fails'] and k3['ok']):
        sys.exit(f'CONTROL FAILED -> no table: K1 {k1}; K2 {k2}; K2 shuffled {k2f}; K3 {k3}')
    OWN = k2['own_id']
    vol = L[(L.t_rel >= 0) & (L.t_rel < 240) & (L.signer != OWN)].groupby('signer').shares.sum().sort_values(ascending=False, kind='stable')
    rk = pd.Series(np.arange(1, len(vol) + 1), index=vol.index)
    r_ = L.signer.map(rk).fillna(1e9).values
    L['g'] = np.where(L.signer.values == OWN, 'ours', np.where(r_ <= 5, 'top 5', np.where(r_ <= 20, '6-20', 'rest')))
    F = in60.copy(); F['g'] = L.g.reindex(F.index).values
    F = F.sort_values(['mid', 'side', 'anc'], kind='stable')
    cols = ['mid', 't_ms'] + [f'{s}{i}_{x}' for s in 'ba' for i in range(1, 6) for x in 'ps']
    parts = []
    for fn in sorted(glob.glob(f'{S1}/books/part_*.parquet')):
        d = pd.read_parquet(fn, columns=cols); parts.append(d[d.mid.isin(W0)])
    Bk = pd.concat(parts, ignore_index=True); del parts
    Bk.sort_values(['mid', 't_ms'], inplace=True, kind='stable'); BI = Bk.groupby('mid', sort=False).indices
    PR = pd.read_parquet(f'{RP}/research/step5/20261108/preds_t60k.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
    PR = PR[PR.mid.isin(W0)].sort_values(['mid', 'tau'], kind='stable')
    G = {m: (g.tau.values.astype(float), g.v_bid.values.astype(float), g.v_ask.values.astype(float)) for m, g in PR.groupby('mid')}
    del PR
    n = len(F); depth = np.full(n, -9, np.int64); status = np.full(n, 'no frame before', dtype=object)
    gst = {lag: np.full(n, B.G_NONE, np.int8) for lag in B.LAGS}; m0 = np.full(n, np.nan); mh = {h: np.full(n, np.nan) for h in B.HS}
    Qb = np.full(n, np.nan)
    Fm = F.reset_index(drop=True); FI = Fm.groupby(['mid', 'side'], sort=False).indices
    for m in sorted(W0, key=W0.get):
        if m not in BI:
            continue
        b = Bk.iloc[BI[m]]; tf = b.t_ms.values.astype(float)
        PP = b[[c for c in cols if c.endswith('_p')]].values
        B.check_frames(tf, PP)
        bb = b.b1_p.values.astype('int64'); ba = b.a1_p.values.astype('int64')
        mff = pd.Series(np.where((bb >= 0) & (ba > bb), (ba + bb) / 2.0, np.nan)).ffill().values
        g_ = G.get(m)
        for side in (0, 1):
            if (m, side) not in FI:
                continue
            ii = FI[(m, side)]; ta = Fm.anc.values[ii]; pt = Fm.ptick.values[ii].astype('int64')
            Pk = b[[f"{'b' if side == 0 else 'a'}{i}_p" for i in range(1, 6)]].values.astype('int64')
            Sk = b[[f"{'b' if side == 0 else 'a'}{i}_s" for i in range(1, 6)]].values.astype(float)
            st = B.level_status(Pk, Sk, side); rs, cz = B.run_start(st)
            okc, dep, lb, cr, _ = B.classify(tf, Pk, st, rs, cz, ta, pt, side)
            j0 = np.searchsorted(tf, ta, side='left') - 1; jj = np.clip(j0, 0, None); best = Pk[jj, 0]
            s_ = np.where(j0 < 0, 'no frame before', np.where(best < 0, 'side empty', np.where(dep < 0, 'inside the spread', 'ok')))
            depth[ii] = np.where(s_ == 'ok', dep, -9); status[ii] = s_
            Qb[ii] = np.where((s_ == 'ok') & (dep == 0), Sk[jj, 0], np.nan)
            m0[ii] = np.where(j0 >= 0, mff[jj], np.nan)
            for h in B.HS:
                j1 = np.searchsorted(tf, ta + h, side='right') - 1; mh[h][ii] = np.where(j1 >= 0, mff[np.clip(j1, 0, None)], np.nan)
            if g_ is not None:
                tau, vb, va = g_
                for lag in B.LAGS:
                    gst[lag][ii] = B.guard_state(tau, (vb, va)[side], ta - lag, LINE)[0]
    del Bk
    R = Fm[KEYS + ['signer', 'g', 'sweep', 'anc', 'grp', 's_lvl', 'n_lvl', 'ws']].copy()
    sg = np.where(R.side.values == 0, 1.0, -1.0); p = R.ptick.values.astype(float)
    R['v1'] = sg * (mh[B.HS[0]] - p); R['v5'] = sg * (mh[B.HS[1]] - p); R['spr'] = sg * (m0 - p)
    R['depth'] = depth; R['status'] = status; R['Q'] = Qb
    for lag in B.LAGS:
        R[f'g{int(lag)}'] = gst[lag]
    R['val_ok'] = np.isfinite(R.v1.values) & np.isfinite(R.v5.values) & np.isfinite(R.spr.values)
    R['lid'] = np.arange(len(R))
    return dict(R=R, G=G, W0=W0, days=days, LINE=LINE, OWN=OWN, own=own, r42=r42, K=dict(K1=k1, K2=k2, K2_shuffled=k2f, K3=k3))


def sel(df, g):
    if g == 'O (ranks >= 6)':
        return df[df.g.isin(['6-20', 'rest'])]
    return df[df.g == g]


def reconcile(R, days):
    """K0: the rebuilt frame against the committed B12 output."""
    for f, h in B12_SHA.items():
        if hashlib.sha256(open(f'{B12_OUT}/{f}', 'rb').read()).hexdigest() != h:
            return dict(ok=False, why=f'{f} hash differs')
    b12 = json.load(open(f'{B12_OUT}/b12.json')); ph = json.load(open(f'{B12_OUT}/b12_posthoc.json'))
    V = R[R.val_ok]; rep = dict(T1={}); ok = True
    for g in ('top 5', '6-20', 'rest', 'O (ranks >= 6)', 'ours'):
        x = sel(V, g); sh = x.shares.values; t = b12['T1'][g]
        e = dict(legs=int(len(x)) == t['legs'], shares=abs(sh.sum() / days - t['sh_day']) < 1e-6, net1=abs(B.wmean(sh, x.v1.values) - t['net1']) < 1e-9)
        rep['T1'][g] = e; ok = ok and all(e.values())
    A = V[(V.status == 'ok') & (V.depth == 0)]
    D = A[A.g.isin(['top 5', '6-20', 'rest']) & A.g50.isin([B.G_ON, B.G_OFF]) & (A.sweep >= 0)]
    cell = (2 * (D.g50.values == B.G_OFF) + D.sweep.values).astype(np.int64)
    d = B.decompose(cell, D.g.values == 'top 5', D.shares.values, D.v1.values); ref = b12['decomposition']['50']
    rep['dec50'] = dict(D=abs(d['D'] - ref['D']) < 1e-9, WITHIN=abs(d['WITHIN'] - ref['WITHIN']) < 1e-9, legs=int(len(D)) == ref['legs'])
    u = A[(A.g == 'ours') & (A.g50 == B.G_OFF)]; po = ph['P3']['ours']['OFF']
    rep['own_off'] = dict(legs=int(len(u)), shares=float(u.shares.sum()), ok=int(len(u)) == po['legs'] and abs(u.shares.sum() - po['sh']) < 1e-9)
    ok = ok and all(rep['dec50'].values()) and rep['own_off']['ok']
    rep['ok'] = bool(ok)
    return rep


def part_a(R, fifo):
    V = R[R.val_ok]; A = V[(V.status == 'ok') & (V.depth == 0) & V.g50.isin([B.G_ON, B.G_OFF]) & (V.sweep == 0)].copy()
    A['r'] = A.s_lvl.values / A.Q.values; A['rb'] = bucket(A.r.values, R_EDGES); A['sb'] = bucket(A.s_lvl.values, S_EDGES)
    A['off'] = (A.g50.values == B.G_OFF).astype(np.int64)
    out = dict(population='at best, guard ON / OFF at lag 50, not swept, value known', groups={})
    for g in GR:
        x = sel(A, g); sh = x.shares.values; tot = sh.sum()
        row = dict(legs=int(len(x)), shares=float(tot), median_reach=wmedian(x.r.values, sh), share_reach_le_025=float(sh[x.r.values <= 0.25].sum() / tot) if tot else NAN,
                   median_size=wmedian(x.s_lvl.values, sh), net1=B.wmean(sh, x.v1.values), move1=B.wmean(sh, x.v1.values - x.spr.values), by_reach={})
        for i, lab in enumerate(R_LAB):
            mk = x.rb.values == i
            row['by_reach'][lab] = dict(share=float(sh[mk].sum() / tot) if tot else NAN, legs=int(mk.sum()), net1=B.wmean(sh[mk], x.v1.values[mk]),
                                        move1=B.wmean(sh[mk], x.v1.values[mk] - x.spr.values[mk]))
        out['groups'][g] = row
    D = A[A.g.isin(['top 5', '6-20', 'rest'])]; isT = D.g.values == 'top 5'; sh = D.shares.values; v = D.v1.values
    cells = dict(G=(D.off.values, 2), GR=(5 * D.off.values + D.rb.values, 10), GS=(5 * D.off.values + D.sb.values, 10),
                 GRS=(25 * D.off.values + 5 * D.rb.values + D.sb.values, 50))
    dec = {}
    for k, (c, nc) in cells.items():
        d = decompose_n(c.astype(np.int64), isT, sh, v, nc)
        dec[k] = dict(D=d['D'], MIX=d['MIX'], WITHIN=d['WITHIN'], excl_T=d['excl_T'], excl_O=d['excl_O'])
    Wg = dec['G']['WITHIN']
    for k in ('GR', 'GS', 'GRS'):
        dec[k]['X'] = (Wg - dec[k]['WITHIN']) / Wg if Wg != 0 else NAN
    c = cells['GR'][0].astype(np.int64)
    blk = pd.factorize(pd.Series(list(zip(D.mid.values, D.side.values, c))))[0]
    nul = null_within_n(c, isT, sh, v, 10, blk, N_NULL, SEED)
    obs = within_n(c, isT, sh, v, 10)
    out['N2'] = dict(observed=obs, p50=float(np.percentile(nul, 50)), p99=float(np.percentile(nul, 99)), draws=N_NULL, seed=SEED, frac_ge=float((nul >= obs).mean()))
    out['decomposition'] = dec; out['legs_TO'] = int(len(D))
    out['R40'] = label_r40(Wg, dec['GR']['WITHIN'], out['N2']['p99'], fifo)
    gT = out['groups']['top 5']; gO = out['groups']['O (ranks >= 6)']; gU = out['groups']['ours']
    out['R41'] = label_r41(gT['median_reach'], gO['median_reach'], gU['median_reach'], gU['legs'], fifo)
    return out


def part_b(R, G, W0, LINE):
    V = R[R.val_ok & (R.status == 'ok') & (R.depth == 0)]
    idx = V.groupby(['mid', 'side'], sort=False).indices
    rows = []; n_ev = 0; n_flip_all = 0
    for m, ws in W0.items():
        if m not in G:
            continue
        tau, vb, va = G[m]
        for side in (0, 1):
            v = (vb, va)[side]
            n_flip_all += len(flips(tau, v, LINE, ws + EV_ON, ws + 60000.0 - EV_OFF, clean=False))
            ev = flips(tau, v, LINE, ws + EV_ON, ws + 60000.0 - EV_OFF)
            if (m, side) not in idx:
                n_ev += len(ev); continue
            ii = idx[(m, side)]; ta = V.anc.values[ii]
            for t in ev:
                e = n_ev; n_ev += 1
                for w, (a, b) in WIN.items():
                    mk = (ta >= t + a) & (ta < t + b)
                    for k in np.nonzero(mk)[0]:
                        rows.append((e, w, ii[k]))
    E = pd.DataFrame(rows, columns=['ev', 'win', 'i'])
    X = V.iloc[E.i.values][['g', 'shares', 'grp', 'v1', 'spr']].reset_index(drop=True); X['ev'] = E.ev.values; X['win'] = E.win.values
    X['T'] = np.where(X.g.values == 'top 5', X.shares.values, 0.0); X['TO'] = np.where(X.g.values != 'ours', X.shares.values, 0.0)
    out = dict(events=n_ev, flips_all=n_flip_all, windows={})
    for w in WIN:
        x = X[X.win == w]; to = x.TO.sum()
        row = dict(share_T=float(x['T'].sum() / to) if to > 0 else NAN, legs_T=int((x.g == 'top 5').sum()), legs_O=int(x.g.isin(['6-20', 'rest']).sum()),
                   own_legs=int((x.g == 'ours').sum()), own_shares=float(x.shares[x.g == 'ours'].sum()), shares_TO=float(to), by_group={})
        for g in GR:
            y = sel(x, g); sh = y.shares.values
            row['by_group'][g] = dict(legs=int(len(y)), shares=float(sh.sum()), net1=B.wmean(sh, y.v1.values), move1=B.wmean(sh, y.v1.values - y.spr.values))
        out['windows'][w] = row
    for name, w1, names in (('after', 'P', ('TOP-LEAVES-FIRST', 'TOP-STAYS', 'SAME-EXIT')), ('before', 'L', ('TOP-LEAVES-BEFORE', 'TOP-STAYS-BEFORE', 'SAME-BEFORE'))):
        x = X[X.win.isin(['B', w1])]
        u = x.groupby(['ev', 'grp'], sort=False).agg(T=('T', 'sum'), TO=('TO', 'sum'), win=('win', 'first')).reset_index()
        lab = (u.win.values == w1).astype(np.int64)
        dl, s1, s0 = exit_delta(lab, u['T'].values, u.TO.values)
        nul = null_exit(u.ev.values, lab, u['T'].values, u.TO.values, N_NULL, SEED + (1 if name == 'before' else 0))
        p01, p99 = float(np.nanpercentile(nul, 1)), float(np.nanpercentile(nul, 99))
        out[name] = dict(window=w1, share_T=s1, share_T_B=s0, rho=s1 / s0 if fin(s1, s0) and s0 > 0 else NAN, delta=dl, null_p01=p01, null_p50=float(np.nanpercentile(nul, 50)),
                         null_p99=p99, units=int(len(u)),
                         label=label_r43(s1, s0, dl, p01, p99, n_ev, out['windows'][w1]['legs_T'], out['windows']['B']['legs_T'], names))
    return out


def part_c(R, G, W0, LINE, own):
    V = R[R.val_ok & (R.status == 'ok') & (R.depth == 0) & (R.g == 'ours')]
    core = []
    for r in RUNS:
        K = pd.read_parquet(f'{LV}/core_orders_{r}.parquet')
        a = K[K.t_Accepted.notna() & K.refused.isna() & K.market.isin(list(W0))].copy(); a['run'] = r; core.append(a)
    core = pd.concat(core, ignore_index=True); core['side'] = (core.lane % 2).astype(int)
    if len(core) and float(core.t_req.max()) >= CUTOFF_MS:
        sys.exit('REFUSED: a live order decided at or after the cutoff')
    if core.cid.duplicated().any():
        sys.exit('REFUSED: live order ids (cid) not unique across runs')
    CI = core.set_index('cid')
    out = {}
    for state, gv in (('OFF', B.G_OFF), ('ON', B.G_ON)):
        u = V[V.g50 == gv].copy(); cid, lst = link_orders(u, own); u['cid'] = cid; u['link'] = lst
        lk = u[u.link == 'linked'].copy()
        o = CI.reindex(lk.cid.values)
        lk['t_req'] = o.t_req.values; lk['t_acc'] = o.t_Accepted.values; lk['t_cancel'] = o.t_cancel.values; lk['t_done'] = o.t_CancelDone.values
        lk['o_side'] = o.side.values; lk['o_tick'] = o.tick.values
        side_ok = bool(((lk.o_side.values == lk.side.values) & (lk.o_tick.values == lk.ptick.values)).all())
        tf_ = []; tfs = []
        for r in lk.itertuples():
            tau, vb, va = G[r.mid]; t0, s0 = off_run_start(tau, (vb, va)[r.side], LINE, r.anc - LAG0); tf_.append(t0); tfs.append(s0)
        lk['t_f'] = tf_; lk['tf_status'] = tfs
        sh = lk.shares.values
        row = dict(legs=int(len(u)), shares=float(u.shares.sum()), link_status=pd.Series(lst).value_counts().to_dict(), side_price_ok=side_ok,
                   never_cancelled_legs=int(np.isnan(lk.t_cancel.values).sum()), classes={})
        for s in SHIFTS:
            cls = late_class(lk.anc.values, lk.t_cancel.values, s)
            row['classes'][str(int(s))] = dict(after_cancel_share=float(sh[cls == 1].sum() / sh.sum()) if sh.sum() else NAN,
                                               after_cancel_legs=int(cls.sum()), label=label_r44(sh, cls, len(lk)))
        q = lambda x: [float(v) for v in np.nanpercentile(x, [10, 25, 50, 75, 90])] if np.isfinite(x).any() else []
        row['cancel_minus_trade_ms_p10_25_50_75_90'] = q(lk.t_cancel.values - lk.anc.values)
        row['trade_minus_ack_ms_p10_25_50_75_90'] = q(lk.anc.values - lk.t_acc.values)
        if state == 'OFF':
            row['tf_status'] = pd.Series(tfs).value_counts().to_dict()
            row['trade_minus_flip_ms_p10_25_50_75_90'] = q(lk.anc.values - lk.t_f.values)
            row['cancel_minus_flip_ms_p10_25_50_75_90'] = q(lk.t_cancel.values - lk.t_f.values)
            row['legs_detail'] = [dict(anc_minus_ws=float(r.anc - r.ws), shares=float(r.shares), trade_minus_flip=float(r.anc - r.t_f),
                                       cancel_minus_trade=float(r.t_cancel - r.anc), trade_minus_ack=float(r.anc - r.t_acc), v1=float(r.v1), swept=int(r.sweep))
                                  for r in lk.sort_values('anc').itertuples()]
        out[state] = row
    out['R44'] = out['OFF']['classes']['0']['label']
    # live's pull delay after a research ON -> OFF flip, for every live order resting 500 ms before it (first minute)
    dl = []; nst = dict(orders_flips=0, never_cancelled=0)
    CS = core.groupby(['market', 'side'], sort=False).indices
    for m, ws in W0.items():
        if m not in G:
            continue
        tau, vb, va = G[m]
        for side in (0, 1):
            if (m, side) not in CS:
                continue
            c = core.iloc[CS[(m, side)]]; ta = c.t_Accepted.values; tc = c.t_cancel.values
            for t in flips(tau, (vb, va)[side], LINE, ws, ws + 60000.0, clean=False):
                live = (ta <= t - 500.0) & (~(tc <= t - 500.0))
                nst['orders_flips'] += int(live.sum()); nst['never_cancelled'] += int((live & np.isnan(tc)).sum())
                dl.extend(list(tc[live & np.isfinite(tc)] - t))
    dl = np.array(dl)
    out['pull_delay'] = dict(n=int(len(dl)), status=nst, p10_25_50_75_90=[float(v) for v in np.percentile(dl, [10, 25, 50, 75, 90])] if len(dl) else [],
                             share_before_flip=float(np.mean(dl < 0)) if len(dl) else NAN, share_le_50=float(np.mean((dl >= 0) & (dl <= 50))) if len(dl) else NAN,
                             share_gt_250=float(np.mean(dl > 250)) if len(dl) else NAN)
    return out


def f2(x):
    return 'n/a' if not fin(x) else f'{x:+.2f}'


def write_txt(out, res):
    a = res['A']; b = res['B']; c = res['C']; rd = res['readings']
    L = [f"B12b why the top makers' fills move less, and ours most. {res['as_of_note']}", f"Controls: {json.dumps({k: v for k, v in res['controls'].items() if k != 'B12_controls_detail'})}",
         f"K0 reconcile with b12.json: {res['K0']['ok']}; own OFF legs {res['K0']['own_off']}", f"R42 matching: {res['r42']}", '',
         f"A queue reach ({a['population']}): group | legs | sh | median reach | share reach <= 0.25 | median size | net 1 s | move 1 s"]
    for g, x in a['groups'].items():
        L.append(f"{g} | {x['legs']} | {x['shares']:,.1f} | {x['median_reach']:.3f} | {x['share_reach_le_025'] * 100:.1f}% | {x['median_size']:.1f} | {f2(x['net1'])} | {f2(x['move1'])}")
    L.append('A by reach: group | bucket share / legs / net 1 s / move 1 s')
    for g, x in a['groups'].items():
        L.append(g + ': ' + '; '.join(f"{k} {v['share'] * 100:.1f}% {v['legs']} {f2(v['net1'])} {f2(v['move1'])}" for k, v in x['by_reach'].items()))
    L.append('A decomposition top 5 against O (c/sh at 1 s): cells | D | MIX | WITHIN | X | excluded T / O')
    for k, d in a['decomposition'].items():
        L.append(f"{k} | {f2(d['D'])} | {f2(d['MIX'])} | {f2(d['WITHIN'])} | {d.get('X', float('nan')):.2f} | {d['excl_T'] * 100:.1f}% / {d['excl_O'] * 100:.1f}%")
    n2 = a['N2']
    L += [f"N2 null (GR): WITHIN observed {f2(n2['observed'])}, p50 {f2(n2['p50'])}, p99 {f2(n2['p99'])}, share >= observed {n2['frac_ge']:.3f}", '',
          f"B exits: clean ON -> OFF flips (events) {b['events']}; all ON -> OFF flips in range {b['flips_all']}. window | Share_T | legs top 5 / O / ours | own sh | net 1 s top 5 / O / ours"]
    for w, x in b['windows'].items():
        bg = x['by_group']
        L.append(f"{w} {WIN[w]} | {x['share_T'] * 100:.1f}% | {x['legs_T']} / {x['legs_O']} / {x['own_legs']} | {x['own_shares']:.1f} | "
                 f"{f2(bg['top 5']['net1'])} / {f2(bg['O (ranks >= 6)']['net1'])} / {f2(bg['ours']['net1'])}")
    for k in ('after', 'before'):
        x = b[k]
        L.append(f"N3 {k} ({x['window']} vs B): Share_T {x['share_T'] * 100:.1f}% vs {x['share_T_B'] * 100:.1f}%, rho {x['rho']:.2f}, delta {x['delta'] * 100:+.1f} pp, "
                 f"null p01 {x['null_p01'] * 100:+.1f} p50 {x['null_p50'] * 100:+.1f} p99 {x['null_p99'] * 100:+.1f} pp; {x['units']} taker orders -> {x['label']}")
    L += ['', 'C our at-best first-minute legs by guard state (lag 50) against live order paths']
    for s in ('OFF', 'ON'):
        x = c[s]
        L.append(f"{s}: legs {x['legs']}, sh {x['shares']:.1f}, link {x['link_status']}, side/price ok {x['side_price_ok']}, never cancelled {x['never_cancelled_legs']}; "
                 + '; '.join(f"shift {k}: after cancel {v['after_cancel_share'] * 100:.0f}% ({v['after_cancel_legs']} legs) {v['label']}" for k, v in x['classes'].items()))
        L.append(f"   cancel - trade ms p10/25/50/75/90 {[round(v) for v in x['cancel_minus_trade_ms_p10_25_50_75_90']]}; trade - ack {[round(v) for v in x['trade_minus_ack_ms_p10_25_50_75_90']]}")
        if s == 'OFF':
            L.append(f"   OFF run start {x['tf_status']}; trade - flip {[round(v) for v in x['trade_minus_flip_ms_p10_25_50_75_90']]}; cancel - flip {[round(v) for v in x['cancel_minus_flip_ms_p10_25_50_75_90']]}")
            for r in x['legs_detail']:
                L.append(f"   leg t {r['anc_minus_ws'] / 1000:.1f} s, {r['shares']:.1f} sh, trade - flip {r['trade_minus_flip']:.0f}, cancel - trade {r['cancel_minus_trade']:.0f}, "
                         f"trade - ack {r['trade_minus_ack']:.0f} ms, swept {r['swept']}, v1 {r['v1']:+.2f}")
    pdl = c['pull_delay']
    L += [f"Live pull delay after a research ON -> OFF flip (orders live 500 ms before it, first minute): n {pdl['n']}, status {pdl['status']}, "
          f"p10/25/50/75/90 {[round(v) for v in pdl['p10_25_50_75_90']]} ms, before the flip {pdl['share_before_flip'] * 100:.1f}%, 0-50 ms {pdl['share_le_50'] * 100:.1f}%, "
          f"> 250 ms {pdl['share_gt_250'] * 100:.1f}%", '', 'Readings: ' + json.dumps(rd, default=float)]
    open(f'{out}/b12b.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    if '--selftest' in sys.argv:
        ok, rep = controls(); print(json.dumps(rep, indent=1)); print('selftest', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    ok, ctl = controls()
    if not ok:
        sys.exit(f'CONTROL FAILED -> no table: {ctl}')
    X = build(); R = X['R']
    k0 = reconcile(R, X['days'])
    if not k0['ok']:
        sys.exit(f'CONTROL FAILED (K0 reconcile with B12) -> no table: {k0}')
    fifo = X['r42']['label'] == 'FIFO-LIKE'
    a = part_a(R, fifo); b = part_b(R, X['G'], X['W0'], X['LINE']); c = part_c(R, X['G'], X['W0'], X['LINE'], X['own'])
    rd = dict(R40=a['R40'], R41=a['R41'], R42=X['r42']['label'], R43_after=b['after']['label'], R43_before=b['before']['label'], R44=c['R44'],
              R44_shifts={k: v['label'] for k, v in c['OFF']['classes'].items()}, R45=label_r45(a['R40'], b['after']['label'], b['before']['label']))
    res = dict(as_of_note='point estimates, one consumed partial day (220 windows of 2026-11-08, 0.764 window-days)', controls=ctl, K=X['K'], K0=k0,
               r42=X['r42'], own_id=X['OWN'], A=a, B=b, C=c, readings=rd)
    json.dump(res, open(f'{out}/b12b.json', 'w'), indent=1, default=float)
    write_txt(out, res)


if __name__ == '__main__':
    main()
