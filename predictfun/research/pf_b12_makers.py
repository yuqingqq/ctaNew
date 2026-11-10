"""B12 (2026-11-10, strategy review item 9): what the profitable makers do. Read-only, descriptive.

Question. The review (3.1, P) finds that the top 5 makers net +0.43 c/sh at 1 s in the first minute, while an
always-present 5-sh order of ours loses in every position (6.5). Do their fills sit in states our guard already marks
(then the gap is speed or queue), or not (then it is signal, B7)? Public tape only; this does not use our fill model.

Population. Maker legs of the 220 consumed 11-08 cohort windows (live-samewindow-20261108, cut 23:00Z, 0.764
window-days; G = 1 day < 5: point estimates only). Inputs: legs_timed2 + legs_tk, top-5 book frames, the frozen v3
guard series (step5 preds_t60k) and its minute-0 lines; no live capture, no protected day.
Identity. signer = the maker order's address (input check 2026-11-10: on chain maker == signer on all 19,671 orders of
the 10-23 14:00-14:15 slice; the REST signer is among the transaction's chain makers on 11,492 of 11,492 legs, 2.9% with
a wrong transaction). Our account = the one id that holds live's own legs (K2). Ranks: shares in t_rel 0-240 s over the
220 windows, ours excluded (as EXPLORATION §34): top 5, 6-20, rest; O = ranks >= 6.
Per leg (first minute: research trade time anc in [ws, ws + 60 s); anc = one time per taker order as
pf_review_queue_reach: earliest WS-timed leg, else earliest E2 estimate):
  value  sg (mid(anc + h) - p), h = 1, 5 s (research mid, last frame at or before); spread = sg (mid(anc-) - p) with
         mid(anc-) from the last frame strictly before anc; move = value - spread. c/sh, share-weighted.
  guard  the maker side's v (v_bid for bids, v_ask for asks) at anc - lag, lag 0 / 50 / 250 ms, from the latest grid
         point <= 100 ms old. ON if v > PM90 of minute 0 (0.0492 c), else OFF. v == -1 exactly is the warm-up
         placeholder (no prediction yet; v3 does not quote): status 'warm-up'. No grid point: 'no guard value'.
         Primary lag 50.
  depth  ticks behind the maker side's best in the frame strictly before anc (0 = at the best). Statuses: no frame
         before, side empty, inside the spread (the price is better than that frame's best: posted and hit before a
         frame showed it).
  level  at-best legs: born as best (the run of resting size at the price began with it as the best) or fall-back
         (reached when better levels emptied); level age = anc - first frame of the run; 'since first sight' and
         'censored' as in pf_maker_behavior.py, whose level_status / run_start / classify are copied here with its seven
         selftest cases.
  sweep  SWEPT if the same taker order also filled this maker side at a price behind this leg's (it went through the
         level); NOT SWEPT otherwise (includes a trade that exactly emptied the level). No taker order id: status
         'sweep unknown'.
Decomposition (at-best legs, guard ON / OFF, sweep known; T = top 5 against O): cells c = guard x sweep; w_g(c) volume
shares, v_g(c) c/sh at 1 s. D = sum w_T v_T - sum w_O v_O = MIX + WITHIN; MIX = sum (w_T - w_O) v_O; WITHIN =
sum w_T (v_T - v_O). MIX_G and MIX_S: MIX over the guard margin alone and over the sweep margin alone.

Readings DECLARED BEFORE THE FIRST RUN (evaluated in code; lag 50 and h = 1 s unless stated):
 R30 edge       TOP-EARNS if the top 5's net over all depths > 0 and top 5 minus O >= +0.30 c/sh; else
                TOP-NOT-DISTINCT.
 R31 source     at the best; NO-MATERIAL-GAP if D < +0.10 c/sh; else every label that applies of GUARD-MIX
                (MIX_G / D >= 0.5), SWEEP-MIX (MIX_S / D >= 0.5), WITHIN-STATE (WITHIN / D >= 0.5); none applies: SPLIT.
 R32 guard off  top 5 at the best with the guard OFF: OFF-ADVERSE-FOR-TOP if c/sh < 0, else OFF-NOT-ADVERSE; and
                AVOID-OFF if their OFF share of guard-known at-best volume <= O's minus 10 pp, else NO-AVOIDANCE.
 R33 depth      DEPTH-FED if the top 5's 1 s dollars from legs behind the best exceed those at the best and are > 0;
                AT-BEST-FED if the at-best dollars are >= those behind and > 0; else NEITHER-POSITIVE.
 R34 speed      (top 5's share of at-best volume on levels under 1 s old) / (their share of all age-classified at-best
                volume): SPEED if >= 1.25, SLOW if <= 0.80, else PROPORTIONAL.
 R35 birth      the same ratio for born-as-best at-best volume: BORN-HEAVY if >= 1.25, FALLBACK-HEAVY if <= 0.80, else
                PROPORTIONAL.
 R36 clock      GUARD-ROBUST if R31 and R32 give the same labels at lags 0 and 250 as at 50; else
                GUARD-CLOCK-SENSITIVE.
 N1 null        WITHIN under 500 permutations of the T / O label among the R31 legs within (window, side, cell), seed
                20261110: WITHIN-REAL if the observed WITHIN > 0 and above the null p99; else WITHIN-NOT-SHOWN.
What the outcomes mean (fixed in advance): GUARD-MIX = they earn by being filled when our guard is ON: our guard already
sees their states and the gap is getting those fills (speed, queue). SWEEP-MIX = they are filled less by trades that go
through the level: queue position or faster pulls. WITHIN-STATE with WITHIN-REAL = they do better in the same guard
state and trade type: information or pricing our guard lacks (B7). DEPTH-FED = their edge is behind the best (B5).

Controls (any failure -> no table): the seven level cases; guard lookup (fresh / stale / warm-up / before the first
point); sweep flag; MIX + WITHIN = D on random cells and three planted cases (mix only, within only, guard margin only);
the null flags a planted +1 c/sh within-cell skill and not its absence; refusals (unsorted frames, non-integer prices,
misaligned legs_tk, an empty decomposition cell); a NaN reads as n/a, never as a label. On the data: K1 the first-minute
tape statuses equal pf_review_queue_reach's (753 legs / 9,131.4 sh untimed, 94,783 / 1,074,957.4 sh timed); K2 one
signer id holds the fill-clock audit's 99 own legs (clock_cases.parquet, hash checked) and exactly live's 849.867 sh,
and K2 fails with the signer column shuffled; K3 the minute-0 PM90 line read from the file equals 0.0492 c.
Limits fixed in advance: one consumed Sunday; trade times are the replay's (the fill-clock audit and B1a find them off
in both directions), so level ages under ~250 ms and the guard state at lag 0 are uncertain; the book contains our own
orders; the values are of the flow the makers received, not of an order we can place.
Usage: predictfun/research/run_light.sh b12_makers -- python3 -m predictfun.research.pf_b12_makers <out_dir>
       python3 -m predictfun.research.pf_b12_makers --selftest
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

RP = '/home/yuqing/pmmm-data/review-exports/live-samewindow-20261108'
S1 = f'{RP}/research/step1/20261108'
FC = '/home/yuqing/ctaNew/data/predict_fun/research/fill_clock_20261109'
FC_SHA = {'clock_cases.parquet': '2f1a580bb6bb3689bf39b52ddbf5a7a443ea004e0d281274510f1193eb108a19'}
KEYS = ['mid', 'exec_s', 'first_recv_ns', 'side', 'ptick', 'shares', 'price_executed', 'taker_index', 'taker_quote']
QR_STAT = dict(untimed_legs=753, untimed_shares=9131.366331840098, legs_0_60=94783, shares_0_60=1074957.3571880765)
LIVE_SH = 849.867319
OWN_LEGS = 99
LINE_DECL = 0.0492
LAGS = (0.0, 50.0, 250.0)
LAG0 = 50.0
FRESH = 100.0
HS = (1000.0, 5000.0)
N_NULL = 500
SEED = 20261110
GROUPS = ['top 5', '6-20', 'rest']
AGE_EDGES = np.array([100.0, 1000.0, 10000.0, 60000.0]); AGE_LAB = ['<0.1 s', '0.1-1 s', '1-10 s', '10-60 s', '>60 s']
G_ON, G_OFF, G_WARM, G_NONE = 1, 0, -1, -2
CELL_LAB = ['ON, not swept', 'ON, swept', 'OFF, not swept', 'OFF, swept']


# ---- level state: copied from pf_maker_behavior.py (2026-11-03) with its selftest cases ----
def level_status(P, S, side):
    """status[k, p], p = 0..100: 1 resting size > 0; 0 inside the shown range and empty; -1 beyond the 5 shown levels."""
    n = len(P); st = np.zeros((n, 101), np.int8); pr = np.arange(101)[None, :]; last = P[:, 4][:, None]
    st[(last >= 0) & ((pr < last) if side == 0 else (pr > last))] = -1
    for i in range(5):
        ok = (P[:, i] >= 0) & (S[:, i] > 0)
        st[np.nonzero(ok)[0], P[ok, i]] = 1
    return st


def run_start(st):
    """First frame of the current run of resting size at each (frame, price), and whether a frame with the price beyond the
    shown levels lies after the last empty frame (age censored)."""
    idx = np.arange(st.shape[0])[:, None]
    la = np.maximum.accumulate(np.where(st == 0, idx, -1), axis=0)
    lu = np.maximum.accumulate(np.where(st == -1, idx, -1), axis=0)
    return la + 1, lu > la


def classify(tf, P, st, rs, cz, ta, p, side):
    """Per fill: usable flag, depth, age label, created-as-best (1/0, -1 n/a) from the frame strictly before each fill time ta.
    Not usable: no frame before, the side empty, or the fill inside the spread (negative depth)."""
    j0 = np.searchsorted(tf, ta, side='left') - 1; ok = j0 >= 0; j = np.clip(j0, 0, None)
    best = P[j, 0]
    depth = np.where(best >= 0, (best - p) if side == 0 else (p - best), -1)
    s = st[j, p]; start = rs[j, p]; cens = cz[j, p]
    age = ta - tf[np.clip(start, 0, len(tf) - 1)]
    lab = np.array(AGE_LAB, dtype=object)[np.searchsorted(AGE_EDGES, age, side='right')]
    lab = np.where(s != 1, 'not resting', np.where(cens, 'censored', np.where(start == 0, 'since first sight', lab)))
    created = np.where((s == 1) & (depth == 0), (P[np.clip(start, 0, len(tf) - 1), 0] == p).astype(int), -1)
    return ok & (best >= 0) & (depth >= 0), depth, lab, created, start


def check_frames(tf, PP):
    """Refuse frames that are not time-sorted or whose prices are not integer cents in [-1, 100] (level_status indexes by
    price)."""
    if len(tf) and not np.all(np.diff(tf) >= 0):
        raise ValueError('frames not sorted by time')
    if not (np.all(PP == np.round(PP)) and PP.min() >= -1 and PP.max() <= 100):
        raise ValueError('book prices are not integer cents in [-1, 100]')


# ---- guard, sweep, decomposition, null ----
def guard_state(tau, v, t, line):
    """State of the guard at times t: ON / OFF from the latest grid point <= FRESH ms old; v == -1 exactly is the warm-up
    placeholder; no grid point -> none. Returns (state, value)."""
    j = np.searchsorted(tau, t, side='right') - 1; jj = np.clip(j, 0, None)
    ok = (j >= 0) & (t - tau[jj] <= FRESH) if len(tau) else np.zeros(len(t), bool)
    x = np.where(ok, v[jj], np.nan) if len(tau) else np.full(len(t), np.nan)
    stt = np.where(~ok, G_NONE, np.where(x == -1.0, G_WARM, np.where(x > line, G_ON, G_OFF)))
    return stt.astype(np.int8), x


def sweep_flag(grp, side, pt, tk_ok):
    """1 if the same taker order also filled this maker side at a price behind this leg's (bids: lower, asks: higher);
    0 if not; -1 if the leg has no taker order id."""
    d = pd.DataFrame(dict(g=grp, s=side, p=pt))
    mn = d.groupby(['g', 's']).p.transform('min').values; mx = d.groupby(['g', 's']).p.transform('max').values
    sw = np.where(side == 0, mn < pt, mx > pt).astype(np.int8)
    return np.where(tk_ok, sw, -1).astype(np.int8)


def shares_vals(cell, sh, val, n):
    w = np.bincount(cell, sh, n); v = np.bincount(cell, sh * val, n) / np.where(w > 0, w, np.nan)
    return w / w.sum(), v


def decompose(cell, isT, sh, val):
    """D = MIX + WITHIN over the 4 guard x sweep cells (cell = 2 * off + swept), and MIX over each margin alone. Refuses an
    empty cell in either group (the reference value would be undefined)."""
    out = {}
    for name, c, n in (('cells', cell, 4), ('guard', cell // 2, 2), ('sweep', cell % 2, 2)):
        wT, vT = shares_vals(c[isT], sh[isT], val[isT], n); wO, vO = shares_vals(c[~isT], sh[~isT], val[~isT], n)
        if not (np.all(np.isfinite(vT)) and np.all(np.isfinite(vO))):
            raise ValueError(f'empty decomposition cell ({name})')
        out[name] = dict(wT=wT, vT=vT, wO=wO, vO=vO, mix=float(np.sum((wT - wO) * vO)), within=float(np.sum(wT * (vT - vO))))
    c = out['cells']; D = float(np.sum(c['wT'] * c['vT']) - np.sum(c['wO'] * c['vO']))
    return dict(D=D, MIX=c['mix'], WITHIN=c['within'], MIX_G=out['guard']['mix'], MIX_S=out['sweep']['mix'], parts=out)


def within_stat(cell, isT, sh, val):
    wT, vT = shares_vals(cell[isT], sh[isT], val[isT], 4); wO, vO = shares_vals(cell[~isT], sh[~isT], val[~isT], 4)
    return float(np.nansum(wT * (vT - vO)))


def null_within(cell, isT, sh, val, block, n, seed):
    """WITHIN with the T / O label permuted within blocks, n draws."""
    rng = np.random.default_rng(seed); o = np.argsort(block, kind='stable'); res = np.empty(n)
    for d in range(n):
        o2 = np.lexsort((rng.random(len(block)), block)); lab = np.empty_like(isT); lab[o] = isT[o2]
        res[d] = within_stat(cell, lab, sh, val)
    return res


def labels_r31(dec):
    D = dec['D']
    if not np.isfinite(D):
        return ['n/a']
    if D < 0.10:
        return ['NO-MATERIAL-GAP']
    lab = [k for k, x in (('GUARD-MIX', dec['MIX_G']), ('SWEEP-MIX', dec['MIX_S']), ('WITHIN-STATE', dec['WITHIN'])) if x / D >= 0.5]
    return lab or ['SPLIT']


def labels_r32(vT_off, offT, offO):
    if not (np.isfinite(vT_off) and np.isfinite(offT) and np.isfinite(offO)):
        return ['n/a']
    return ['OFF-ADVERSE-FOR-TOP' if vT_off < 0 else 'OFF-NOT-ADVERSE', 'AVOID-OFF' if offT <= offO - 0.10 else 'NO-AVOIDANCE']


def ratio_label(r, hi, lo):
    if not np.isfinite(r):
        return 'n/a'
    return hi if r >= 1.25 else (lo if r <= 0.80 else 'PROPORTIONAL')


# ---- controls ----
def controls():
    rep = {}; ok = True

    def chk(name, cond):
        nonlocal ok
        rep[name] = bool(cond); ok = ok and bool(cond)
    # C1 the seven level cases of pf_maker_behavior.selftest
    P = np.array([[51, 50, 49, 48, 47], [51, 50, 49, 48, 47], [50, 49, 48, 47, 46], [49, 48, 47, 46, 45], [50, 49, 48, 47, 46], [50, 49, 48, 47, 46]])
    S = np.ones((6, 5)); tf = np.array([0., 100., 200., 300., 400., 500.])
    st = level_status(P, S, 0); rs, cz = run_start(st)
    c1 = st[3, 50] == 0 and st[0, 50] == 1 and st[3, 44] == -1 and st[0, 52] == 0
    ok_, dep, lab, cr, _ = classify(tf, P, st, rs, cz, np.array([450.0, 550.0, 250.0]), np.array([50, 50, 50]), 0)
    c1 = c1 and lab[0] == '<0.1 s' and cr[0] == 1 and dep[0] == 0 and lab[1] == '0.1-1 s' and lab[2] == 'since first sight' and cr[2] == 0
    P2 = np.array([[50, 49, 48, 47, 46], [50, 49, 48, 47, 47], [50, 49, 48, 47, 46]]); st2 = level_status(P2, np.ones((3, 5)), 0); rs2, cz2 = run_start(st2)
    _, _, lab2, _, _ = classify(np.array([0., 100., 200.]), P2, st2, rs2, cz2, np.array([250.0]), np.array([46]), 0)
    c1 = c1 and st2[1, 46] == -1 and lab2[0] == 'censored'
    A = np.array([[52, 53, 54, 55, 56]]); sa = level_status(A, np.ones((1, 5)), 1)
    c1 = c1 and sa[0, 57] == -1 and sa[0, 51] == 0 and sa[0, 54] == 1
    _, _, lab3, _, _ = classify(tf, P, st, rs, cz, np.array([350.0]), np.array([50]), 0)
    ok4, _, _, _, _ = classify(tf, P, st, rs, cz, np.array([350.0]), np.array([50]), 0)
    chk('C1_level_cases', c1 and lab3[0] == 'not resting' and not ok4[0])
    # C2 guard lookup: fresh ON, warm-up placeholder, stale, before the first point, OFF
    tau = np.array([0.0, 50.0, 100.0, 150.0]); v = np.array([-1.0, 0.03, 0.06, 0.07])
    s_, x_ = guard_state(tau, v, np.array([160.0, 40.0, 400.0, -5.0, 90.0]), LINE_DECL)
    chk('C2_guard_lookup', list(s_) == [G_ON, G_WARM, G_NONE, G_NONE, G_OFF] and x_[0] == 0.07)
    # C3 sweep flag
    sw = sweep_flag(np.array([7, 7, 7, 8, 9, 9, 10]), np.array([0, 0, 0, 1, 1, 1, 0]), np.array([50, 50, 49, 60, 61, 62, 40]),
                    np.array([True] * 6 + [False]))
    chk('C3_sweep_flag', list(sw) == [1, 1, 0, 0, 1, 0, -1])
    # C4 decomposition identity and planted cases
    rng = np.random.default_rng(1)
    cell = rng.integers(0, 4, 4000); isT = rng.random(4000) < 0.3; sh = rng.random(4000) * 10; val = rng.normal(0, 2, 4000) + cell * 0.3
    d = decompose(cell, isT, sh, val)
    c4 = abs(d['MIX'] + d['WITHIN'] - d['D']) < 1e-12
    # (a) the same value per cell for both groups, different mixes -> WITHIN 0, MIX = D
    vcell = np.array([1.0, -2.0, 0.5, -1.5]); cellA = np.r_[np.zeros(60, int), np.ones(20, int), np.full(10, 2), np.full(10, 3),
                                                             np.zeros(25, int), np.ones(25, int), np.full(25, 2), np.full(25, 3)]
    isTA = np.r_[np.ones(100, bool), np.zeros(100, bool)]; dA = decompose(cellA, isTA, np.ones(200), vcell[cellA])
    c4 = c4 and abs(dA['WITHIN']) < 1e-12 and abs(dA['MIX'] - dA['D']) < 1e-12 and dA['D'] != 0
    # (b) the same mix, T better by +0.7 in every cell -> MIX 0, WITHIN = D = 0.7
    cellB = np.r_[cellA[100:], cellA[100:]]; isTB = np.r_[np.ones(100, bool), np.zeros(100, bool)]
    dB = decompose(cellB, isTB, np.ones(200), vcell[cellB] + np.where(isTB, 0.7, 0.0))
    c4 = c4 and abs(dB['MIX']) < 1e-12 and abs(dB['WITHIN'] - 0.7) < 1e-12
    # (c) the guard mix differs, the sweep split within each guard state is the same, values depend on the guard only
    #     -> MIX_G = MIX = D and MIX_S = 0
    cellC = np.r_[np.zeros(30, int), np.ones(30, int), np.full(20, 2), np.full(20, 3), np.zeros(10, int), np.ones(10, int), np.full(40, 2), np.full(40, 3)]
    vg = np.array([1.0, 1.0, -1.0, -1.0]); dC = decompose(cellC, isTB, np.ones(200), vg[cellC])
    c4 = c4 and abs(dC['MIX_G'] - dC['D']) < 1e-12 and abs(dC['MIX_S']) < 1e-12 and abs(dC['WITHIN']) < 1e-12
    chk('C4_decomposition', c4)
    # C5 the null flags a planted within-cell skill and not its absence
    nb = 40; m = 50; blk = np.repeat(np.arange(nb), m); cl = blk % 4; isT5 = rng.random(nb * m) < 0.4; sh5 = 1.0 + rng.random(nb * m)
    noise = rng.normal(0, 1, nb * m); flags = []
    for plant in (1.0, 0.0):
        val5 = noise + np.where(isT5, plant, 0.0); obs = within_stat(cl, isT5, sh5, val5)
        nul = null_within(cl, isT5, sh5, val5, blk, 200, 7); flags.append(obs > 0 and obs > np.percentile(nul, 99))
    chk('C5_null', flags == [True, False])
    # C6 refusals
    ref = []
    for tf_, pp in ((np.array([0.0, 2.0, 1.0]), np.zeros((3, 5))), (np.array([0.0, 1.0]), np.full((2, 5), 50.5))):
        try:
            check_frames(tf_, pp); ref.append(False)
        except ValueError:
            ref.append(True)
    try:
        decompose(np.array([0, 1, 2, 0, 1, 2]), np.array([True, True, True, False, False, False]), np.ones(6), np.zeros(6)); ref.append(False)
    except ValueError:
        ref.append(True)
    chk('C6_refusals', all(ref) and len(ref) == 3)
    chk('C7_nan_is_na', labels_r31(dict(D=float('nan'))) == ['n/a'] and labels_r32(float('nan'), 0.2, 0.4) == ['n/a']
        and ratio_label(float('nan'), 'SPEED', 'SLOW') == 'n/a' and ratio_label(1.3, 'SPEED', 'SLOW') == 'SPEED'
        and ratio_label(0.8, 'SPEED', 'SLOW') == 'SLOW' and ratio_label(1.0, 'SPEED', 'SLOW') == 'PROPORTIONAL'
        and labels_r31(dict(D=0.05)) == ['NO-MATERIAL-GAP'] and labels_r31(dict(D=1.0, MIX_G=0.6, MIX_S=0.1, WITHIN=0.3)) == ['GUARD-MIX']
        and labels_r31(dict(D=1.0, MIX_G=0.3, MIX_S=0.3, WITHIN=0.4)) == ['SPLIT']
        and labels_r32(-0.1, 0.2, 0.35) == ['OFF-ADVERSE-FOR-TOP', 'AVOID-OFF'] and labels_r32(0.0, 0.3, 0.35) == ['OFF-NOT-ADVERSE', 'NO-AVOIDANCE'])
    return ok, rep


# ---- data ----
def own_id(L, own, shuffle=None):
    """K2: the signer id holding the fill-clock audit's own legs; checks the count and the shares."""
    sg = L.signer.values.copy()
    if shuffle is not None:
        sg = np.random.default_rng(shuffle).permutation(sg)
    j = own[KEYS].reset_index().merge(L[KEYS].assign(sg=sg), on=KEYS, how='left')
    ids = j.sg.value_counts(); oid = int(ids.index[0])
    legs_with = int(j[j.sg == oid]['index'].nunique()); x = sg == oid
    rep = dict(own_id=oid, own_legs=len(own), legs_with_id=legs_with, id_legs=int(x.sum()), id_shares=float(L.shares.values[x].sum()))
    rep['ok'] = bool(legs_with == OWN_LEGS == len(own) and rep['id_legs'] == OWN_LEGS and abs(rep['id_shares'] - LIVE_SH) < 1e-3)
    return rep


def wmean(sh, x):
    s = sh.sum()
    return float((sh * x).sum() / s) if s > 0 else float('nan')


def main():
    if '--selftest' in sys.argv:
        ok, rep = controls(); print(json.dumps(rep, indent=1)); print('selftest', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    ok, rep_c = controls()
    if not ok:
        sys.exit(f'CONTROL FAILED -> no table: {rep_c}')
    sc = json.load(open(f'{RP}/scope.json')); W0 = {int(w['market']): w['ws'] * 1000.0 for w in sc['windows']}; days = len(W0) / 288.0
    LINE = float(json.load(open(f'{RP}/research/step4/minute_lines_t60k.json'))['lines']['90'][0])
    k3 = dict(line=LINE, ok=abs(LINE - LINE_DECL) < 1e-4)

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
    L['sweep'] = sweep_flag(grp, L.side.values, L.ptick.values, tk >= 0)
    L = L[L.mid.isin(W0)].copy(); L['ws'] = L.mid.map(W0)
    first = (L.exec_s * 1000.0 >= L.ws) & (L.exec_s * 1000.0 < L.ws + 60000.0)
    unt = L[~np.isfinite(L.anc.values) & first]
    T0 = L[np.isfinite(L.anc.values)]; in60 = T0[(T0.anc >= T0.ws) & (T0.anc < T0.ws + 60000.0)]
    k1 = dict(untimed_legs=len(unt), untimed_shares=float(unt.shares.sum()), legs_0_60=len(in60), shares_0_60=float(in60.shares.sum()))
    k1['ok'] = all(abs(k1[k] - v) <= 1e-6 for k, v in QR_STAT.items())

    if hashlib.sha256(open(f'{FC}/clock_cases.parquet', 'rb').read()).hexdigest() != FC_SHA['clock_cases.parquet']:
        sys.exit('REFUSED: clock_cases.parquet hash differs from the fill-clock audit receipt')
    own = pd.read_parquet(f'{FC}/clock_cases.parquet')
    k2 = own_id(L, own); k2f = own_id(L, own, shuffle=SEED)
    k2['falsifier_shuffled_fails'] = not k2f['ok']
    if not (k1['ok'] and k2['ok'] and k2['falsifier_shuffled_fails'] and k3['ok']):
        sys.exit(f'CONTROL FAILED -> no table: K1 {k1}; K2 {k2}; K2 shuffled {k2f}; K3 {k3}')
    OWN = k2['own_id']

    vol = L[(L.t_rel >= 0) & (L.t_rel < 240) & (L.signer != OWN)].groupby('signer').shares.sum().sort_values(ascending=False, kind='stable')
    rk = pd.Series(np.arange(1, len(vol) + 1), index=vol.index)
    r_ = L.signer.map(rk).fillna(1e9).values
    L['g'] = np.where(L.signer.values == OWN, 'ours', np.where(r_ <= 5, 'top 5', np.where(r_ <= 20, '6-20', 'rest')))
    top5 = [int(x) for x in vol.index[:5]]

    F = in60.copy(); F['g'] = L.g.reindex(F.index).values
    F = F.sort_values(['mid', 'side', 'anc'], kind='stable')

    cols = ['mid', 't_ms'] + [f'{s}{i}_{x}' for s in 'ba' for i in range(1, 6) for x in 'ps']
    parts = []
    for fn in sorted(glob.glob(f'{S1}/books/part_*.parquet')):
        d = pd.read_parquet(fn, columns=cols); parts.append(d[d.mid.isin(W0)])
    B = pd.concat(parts, ignore_index=True); del parts
    B.sort_values(['mid', 't_ms'], inplace=True, kind='stable'); BI = B.groupby('mid', sort=False).indices
    PR = pd.read_parquet(f'{RP}/research/step5/20261108/preds_t60k.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
    PR = PR[PR.mid.isin(W0)].sort_values(['mid', 'tau'], kind='stable')
    G = {m: (g.tau.values.astype(float), g.v_bid.values.astype(float), g.v_ask.values.astype(float)) for m, g in PR.groupby('mid')}
    del PR

    n = len(F); depth = np.full(n, -9, np.int64); lab = np.full(n, 'no frame before', dtype=object); created = np.full(n, -1, np.int64)
    status = np.full(n, 'no frame before', dtype=object); gst = {lag: np.full(n, G_NONE, np.int8) for lag in LAGS}
    m0 = np.full(n, np.nan); mh = {h: np.full(n, np.nan) for h in HS}
    FI = F.reset_index(drop=True).groupby(['mid', 'side'], sort=False).indices
    Fm = F.reset_index(drop=True)
    stat = dict(windows=len(W0), no_frames=0, no_guard_series=0)
    for m in sorted(W0, key=W0.get):
        if m not in BI:
            stat['no_frames'] += 1; continue
        b = B.iloc[BI[m]]; tf = b.t_ms.values.astype(float)
        PP = b[[c for c in cols if c.endswith('_p')]].values
        check_frames(tf, PP)
        bb = b.b1_p.values.astype('int64'); ba = b.a1_p.values.astype('int64')
        mff = pd.Series(np.where((bb >= 0) & (ba > bb), (ba + bb) / 2.0, np.nan)).ffill().values
        g_ = G.get(m)
        if g_ is None:
            stat['no_guard_series'] += 1
        for side in (0, 1):
            if (m, side) not in FI:
                continue
            ii = FI[(m, side)]; ta = Fm.anc.values[ii]; pt = Fm.ptick.values[ii].astype('int64')
            Pk = b[[f"{'b' if side == 0 else 'a'}{i}_p" for i in range(1, 6)]].values.astype('int64')
            Sk = b[[f"{'b' if side == 0 else 'a'}{i}_s" for i in range(1, 6)]].values.astype(float)
            st = level_status(Pk, Sk, side); rs, cz = run_start(st)
            okc, dep, lb, cr, _ = classify(tf, Pk, st, rs, cz, ta, pt, side)
            j0 = np.searchsorted(tf, ta, side='left') - 1; best = Pk[np.clip(j0, 0, None), 0]
            s_ = np.where(j0 < 0, 'no frame before', np.where(best < 0, 'side empty', np.where(dep < 0, 'inside the spread', 'ok')))
            depth[ii] = np.where(s_ == 'ok', dep, -9); lab[ii] = np.where(s_ == 'ok', lb, s_); created[ii] = np.where(s_ == 'ok', cr, -1)
            status[ii] = s_
            m0[ii] = np.where(j0 >= 0, mff[np.clip(j0, 0, None)], np.nan)
            for h in HS:
                j1 = np.searchsorted(tf, ta + h, side='right') - 1; mh[h][ii] = np.where(j1 >= 0, mff[np.clip(j1, 0, None)], np.nan)
            if g_ is not None:
                tau, vb, va = g_
                for lag in LAGS:
                    gst[lag][ii] = guard_state(tau, (vb, va)[side], ta - lag, LINE)[0]
    del B
    R = Fm[['mid', 'side', 'ptick', 'shares', 'signer', 'g', 'sweep', 'anc']].copy()
    sg = np.where(R.side.values == 0, 1.0, -1.0); p = R.ptick.values.astype(float)
    R['v1'] = sg * (mh[HS[0]] - p); R['v5'] = sg * (mh[HS[1]] - p); R['spr'] = sg * (m0 - p)
    R['depth'] = depth; R['lab'] = lab; R['created'] = created; R['status'] = status
    for lag in LAGS:
        R[f'g{int(lag)}'] = gst[lag]
    R['val_ok'] = np.isfinite(R.v1.values) & np.isfinite(R.v5.values) & np.isfinite(R.spr.values)
    report(out, R, dict(K1=k1, K2=k2, K2_shuffled=k2f, K3=k3, C=rep_c), stat, days, top5, OWN, len(vol))


def report(out, R, ctl, stat, days, top5, OWN, n_makers):
    res = dict(as_of_note='point estimates, one consumed partial day (220 windows of 2026-11-08, 0.764 window-days)', controls=ctl,
               stat=stat, own_id=OWN, ranked_makers=n_makers, top5_ids=top5)
    gr = ['top 5', '6-20', 'rest', 'O (ranks >= 6)', 'all others', 'ours']

    def sel(df, g):
        if g == 'O (ranks >= 6)':
            return df[df.g.isin(['6-20', 'rest'])]
        if g == 'all others':
            return df[df.g != 'ours']
        return df[df.g == g]
    V = R[R.val_ok]
    res['status_value'] = {g: dict(legs=int(len(sel(R, g))), shares=float(sel(R, g).shares.sum()),
                                   no_mid_legs=int((~sel(R, g).val_ok).sum()), no_mid_shares=float(sel(R, g).shares[~sel(R, g).val_ok].sum())) for g in gr}
    # T1 maker table, all depths
    t1 = {}
    for g in gr:
        x = sel(V, g); sh = x.shares.values
        t1[g] = dict(makers=int(x.signer.nunique()), legs=int(len(x)), sh_day=float(sh.sum() / days), spread=wmean(sh, x.spr.values),
                     move1=wmean(sh, x.v1.values - x.spr.values), net1=wmean(sh, x.v1.values), net5=wmean(sh, x.v5.values),
                     usd_day1=float((sh * x.v1.values).sum() / 100.0 / days))
    res['T1'] = t1
    # T2 depth / status per group
    def dcls(x):
        return np.where(x.status.values != 'ok', x.status.values, np.where(x.depth.values == 0, 'at best', np.where(x.depth.values == 1, '1 tick behind', '2+ behind')))
    t2 = {}
    for g in gr:
        x = sel(V, g); c = dcls(x); sh = x.shares.values; tot = sh.sum(); t2[g] = {}
        for k in ('at best', '1 tick behind', '2+ behind', 'inside the spread', 'side empty', 'no frame before'):
            mk = c == k
            t2[g][k] = dict(share=float(sh[mk].sum() / tot) if tot else float('nan'), net1=wmean(sh[mk], x.v1.values[mk]),
                            net5=wmean(sh[mk], x.v5.values[mk]), usd_day1=float((sh[mk] * x.v1.values[mk]).sum() / 100.0 / days))
    res['T2'] = t2
    # at-best legs
    A = V[(V.status == 'ok') & (V.depth == 0)]
    # T3 guard x sweep cells, and the decomposition at each lag
    t3 = {}; dec = {}; r31 = {}; r32 = {}
    for lag in LAGS:
        gcol = f'g{int(lag)}'; t3[lag] = {}
        for g in gr:
            x = sel(A, g); sh = x.shares.values; tot = sh.sum(); gs = x[gcol].values; sw = x.sweep.values; t3[lag][g] = {}
            for gn, gv in (('ON', G_ON), ('OFF', G_OFF), ('warm-up', G_WARM), ('no guard value', G_NONE)):
                for sn, sv in (('not swept', 0), ('swept', 1), ('sweep unknown', -1)):
                    mk = (gs == gv) & (sw == sv)
                    if mk.any():
                        t3[lag][g][f'{gn}, {sn}'] = dict(share=float(sh[mk].sum() / tot), net1=wmean(sh[mk], x.v1.values[mk]))
            kn = np.isin(gs, [G_ON, G_OFF])
            t3[lag][g]['OFF share of guard-known'] = float(sh[kn & (gs == G_OFF)].sum() / sh[kn].sum()) if kn.any() else float('nan')
            t3[lag][g]['net1 OFF'] = wmean(sh[gs == G_OFF], x.v1.values[gs == G_OFF])
            t3[lag][g]['net1 ON'] = wmean(sh[gs == G_ON], x.v1.values[gs == G_ON])
        D = A[A.g.isin(['top 5', '6-20', 'rest']) & A[gcol].isin([G_ON, G_OFF]) & (A.sweep >= 0)]
        cell = (2 * (D[gcol].values == G_OFF) + D.sweep.values).astype(np.int64); isT = (D.g.values == 'top 5')
        d = decompose(cell, isT, D.shares.values, D.v1.values)
        dec[lag] = dict(D=d['D'], MIX=d['MIX'], WITHIN=d['WITHIN'], MIX_G=d['MIX_G'], MIX_S=d['MIX_S'],
                        legs=int(len(D)), shares=float(D.shares.sum()),
                        cells={CELL_LAB[i]: dict(wT=float(d['parts']['cells']['wT'][i]), vT=float(d['parts']['cells']['vT'][i]),
                                                 wO=float(d['parts']['cells']['wO'][i]), vO=float(d['parts']['cells']['vO'][i])) for i in range(4)})
        r31[lag] = labels_r31(d)
        r32[lag] = labels_r32(t3[lag]['top 5']['net1 OFF'], t3[lag]['top 5']['OFF share of guard-known'],
                              t3[lag]['O (ranks >= 6)']['OFF share of guard-known'])
        if lag == LAG0:
            blk = pd.factorize(pd.Series(list(zip(D.mid.values, D.side.values, cell))))[0]
            nul = null_within(cell, isT, D.shares.values, D.v1.values, blk, N_NULL, SEED)
            obs = within_stat(cell, isT, D.shares.values, D.v1.values)
            res['N1'] = dict(observed=obs, p50=float(np.percentile(nul, 50)), p99=float(np.percentile(nul, 99)), draws=N_NULL, seed=SEED,
                             frac_ge=float((nul >= obs).mean()), reading='WITHIN-REAL' if (obs > 0 and obs > np.percentile(nul, 99)) else 'WITHIN-NOT-SHOWN')
    res['T3'] = {str(int(k)): v for k, v in t3.items()}; res['decomposition'] = {str(int(k)): v for k, v in dec.items()}
    # T4 level state at the best
    t4 = {}
    for g in gr:
        x = sel(A, g); sh = x.shares.values; tot = sh.sum(); t4[g] = {}
        for lb_ in AGE_LAB + ['since first sight', 'censored']:
            for cn, cv in (('born as best', 1), ('fall-back', 0)):
                mk = (x.lab.values == lb_) & (x.created.values == cv)
                if mk.any():
                    t4[g][f'{lb_}, {cn}'] = dict(share=float(sh[mk].sum() / tot), net1=wmean(sh[mk], x.v1.values[mk]))
    res['T4'] = t4

    def share_of(mask_fn):
        o = A[A.g != 'ours']; mk = mask_fn(o); s_all = o.shares.values[mk].sum()
        return float(o.shares.values[mk & (o.g.values == 'top 5')].sum() / s_all) if s_all > 0 else float('nan')
    aged = lambda o: np.isin(o.lab.values, AGE_LAB)
    young = lambda o: np.isin(o.lab.values, AGE_LAB[:2])
    crk = lambda o: o.created.values >= 0
    born = lambda o: o.created.values == 1
    r34_ratio = share_of(young) / share_of(aged); r35_ratio = share_of(born) / share_of(crk)
    # T5 each top-5 maker
    t5 = {}
    for s in top5:
        x = V[V.signer == s]; a = A[A.signer == s]; sh = x.shares.values; ash = a.shares.values; g50 = a[f'g{int(LAG0)}'].values
        kn = np.isin(g50, [G_ON, G_OFF])
        t5[str(s)] = dict(sh_day=float(sh.sum() / days), net1=wmean(sh, x.v1.values), net5=wmean(sh, x.v5.values),
                          at_best_share=float(ash.sum() / sh.sum()) if sh.sum() else float('nan'),
                          behind_share=float(x.shares.values[(x.status.values == 'ok') & (x.depth.values > 0)].sum() / sh.sum()),
                          inside_spread_share=float(x.shares.values[x.status.values == 'inside the spread'].sum() / sh.sum()),
                          on_share_at_best=float(ash[kn & (g50 == G_ON)].sum() / ash[kn].sum()) if kn.any() else float('nan'),
                          swept_share_at_best=float(ash[a.sweep.values == 1].sum() / ash.sum()) if ash.sum() else float('nan'),
                          born_share_at_best=float(ash[a.created.values == 1].sum() / ash[a.created.values >= 0].sum()) if (a.created.values >= 0).any() else float('nan'),
                          young_share_at_best=float(ash[np.isin(a.lab.values, AGE_LAB[:2])].sum() / ash[np.isin(a.lab.values, AGE_LAB)].sum()) if np.isin(a.lab.values, AGE_LAB).any() else float('nan'))
    res['T5'] = t5
    # readings
    top = t1['top 5']; O = t1['O (ranks >= 6)']
    r30 = 'n/a' if not (np.isfinite(top['net1']) and np.isfinite(O['net1'])) else 'TOP-EARNS' if (top['net1'] > 0 and top['net1'] - O['net1'] >= 0.30) else 'TOP-NOT-DISTINCT'
    tb = t2['top 5']; at_usd = tb['at best']['usd_day1']; bh_usd = tb['1 tick behind']['usd_day1'] + tb['2+ behind']['usd_day1']
    r33 = 'DEPTH-FED' if (bh_usd > at_usd and bh_usd > 0) else ('AT-BEST-FED' if (at_usd >= bh_usd and at_usd > 0) else 'NEITHER-POSITIVE')
    r36 = 'GUARD-ROBUST' if all(r31[l] == r31[LAG0] and r32[l] == r32[LAG0] for l in LAGS) else 'GUARD-CLOCK-SENSITIVE'
    res['readings'] = dict(R30=dict(label=r30, top5_net1=top['net1'], O_net1=O['net1'], diff=top['net1'] - O['net1']),
                           R31={str(int(l)): r31[l] for l in LAGS}, R32={str(int(l)): r32[l] for l in LAGS},
                           R33=dict(label=r33, at_best_usd_day=at_usd, behind_usd_day=bh_usd),
                           R34=dict(label=ratio_label(r34_ratio, 'SPEED', 'SLOW'), ratio=r34_ratio, top5_share_young=share_of(young), top5_share_aged=share_of(aged)),
                           R35=dict(label=ratio_label(r35_ratio, 'BORN-HEAVY', 'FALLBACK-HEAVY'), ratio=r35_ratio, top5_share_born=share_of(born), top5_share_classified=share_of(crk)),
                           R36=r36, N1=res['N1']['reading'])
    json.dump(res, open(f'{out}/b12.json', 'w'), indent=1, default=float)
    write_txt(out, res, gr)
    print(json.dumps(res['readings'], indent=1, default=float))


def f2(x):
    return 'n/a' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:+.2f}'


def write_txt(out, res, gr):
    L = [f"B12 what the profitable makers do. {res['as_of_note']}. Own id {res['own_id']}; {res['ranked_makers']} ranked makers; top 5 ids {res['top5_ids']}.",
         f"Controls: {json.dumps(res['controls']['C'])}; K1 {res['controls']['K1']['ok']}; K2 {res['controls']['K2']['ok']} (shuffled fails "
         f"{res['controls']['K2']['falsifier_shuffled_fails']}); K3 {res['controls']['K3']['ok']}. Stat {res['stat']}", '',
         'T1 first minute, all depths (c/sh; sh/day = shares / 0.764)', 'group | makers | legs | sh/day | spread | move 1 s | net 1 s | net 5 s | $/day 1 s']
    for g in gr:
        x = res['T1'][g]
        L.append(f"{g} | {x['makers']} | {x['legs']} | {x['sh_day']:,.0f} | {f2(x['spread'])} | {f2(x['move1'])} | {f2(x['net1'])} | {f2(x['net5'])} | {x['usd_day1']:+,.1f}")
    L += ['', 'Status (no mid before or after): ' + '; '.join(f"{g} {v['no_mid_legs']} legs / {v['no_mid_shares']:.1f} sh" for g, v in res['status_value'].items()), '',
          'T2 depth: share of the group\'s volume / net 1 s / net 5 s / $/day 1 s']
    for g in gr:
        L.append(g + ': ' + '; '.join(f"{k} {v['share'] * 100:.1f}% {f2(v['net1'])} {f2(v['net5'])} {v['usd_day1']:+,.1f}" for k, v in res['T2'][g].items()))
    for lag, t in res['T3'].items():
        L += ['', f'T3 at the best, guard at anc - {lag} ms x sweep: share / net 1 s']
        for g in gr:
            x = t[g]
            L.append(g + ': ' + '; '.join(f"{k} {v['share'] * 100:.1f}% {f2(v['net1'])}" for k, v in x.items() if isinstance(v, dict))
                     + f" | OFF share of guard-known {x['OFF share of guard-known'] * 100:.1f}%, net1 ON {f2(x['net1 ON'])}, OFF {f2(x['net1 OFF'])}")
    L += ['', 'Decomposition (at the best, top 5 against ranks >= 6, c/sh at 1 s): lag | legs | D | MIX | MIX_G | MIX_S | WITHIN']
    for lag, d in res['decomposition'].items():
        L.append(f"{lag} ms | {d['legs']} | {f2(d['D'])} | {f2(d['MIX'])} | {f2(d['MIX_G'])} | {f2(d['MIX_S'])} | {f2(d['WITHIN'])}")
        L.append('   cells: ' + '; '.join(f"{k} wT {v['wT'] * 100:.1f}% vT {f2(v['vT'])} wO {v['wO'] * 100:.1f}% vO {f2(v['vO'])}" for k, v in d['cells'].items()))
    n1 = res['N1']
    L += [f"N1 null: WITHIN observed {f2(n1['observed'])}, null p50 {f2(n1['p50'])}, p99 {f2(n1['p99'])}, {n1['draws']} draws, share of draws >= observed {n1['frac_ge']:.3f}", '',
          'T4 at the best, level age x born / fall-back: share / net 1 s']
    for g in gr:
        L.append(g + ': ' + '; '.join(f"{k} {v['share'] * 100:.1f}% {f2(v['net1'])}" for k, v in res['T4'][g].items()))
    L += ['', 'T5 each top-5 maker: sh/day | net 1 s | net 5 s | at best | behind | inside spread | ON at best (lag 50) | swept | born | < 1 s']
    for s, x in res['T5'].items():
        L.append(f"{s} | {x['sh_day']:,.0f} | {f2(x['net1'])} | {f2(x['net5'])} | {x['at_best_share'] * 100:.1f}% | {x['behind_share'] * 100:.1f}% | "
                 f"{x['inside_spread_share'] * 100:.1f}% | {x['on_share_at_best'] * 100:.1f}% | {x['swept_share_at_best'] * 100:.1f}% | "
                 f"{x['born_share_at_best'] * 100:.1f}% | {x['young_share_at_best'] * 100:.1f}%")
    L += ['', 'Readings: ' + json.dumps(res['readings'], default=float)]
    open(f'{out}/b12.txt', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
