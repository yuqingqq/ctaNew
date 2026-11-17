"""KEEP OR PULL THE SAME RESTING ORDER AT THE ACTUAL CANCEL LATENCY (2026-11-17, session aea5bfb9). USER: "My next priority would be to
repair that ledger, then compare keeping versus pulling the same resting order at actual cancellation latency, measuring avoided losses
against foregone gains." The latest results review (handoffs/20261117_latest_results_review.md, next priority): "use identical neutral
resting-order states to separate useful queue retention from avoidable adverse fills: the same side, price, remaining size, horizon and
actual cancellation latency, with keep and pull branches. Report foregone gains and avoided ordinary and severe losses, alongside value at
matched participation, by day. An ex-post oracle can bound loss avoidable at that latency, clearly marked non-deployable; a declared causal
score must then earn improvement over matched controls." Declared before any run. A measurement: no model is fitted, nothing is promoted.
CORRECTED BEFORE ANY RUN (2026-11-17, the ledger / keep-pull review, handoffs/20261117_ledger_keep_pull_review.md): the null now samples
 from the live risk set and matches EXECUTED pulls (a shuffle of raw triggers lost pulls on orders already gone - false K2 passes); live
 decisions start at the first step after the client knows the order is live (the +50 ms step was a pending-order withdrawal); K1's failure
 reads NOT-SHOWN and K4 is a paired zero- versus actual-latency comparison in numbers; a missing fair input is handled at its decision
 (fallback KEEP) instead of excluding the order for a missing input later in its life. AND AGAIN (2026-11-17 15:07Z, review of b6eaa7c2,
 handoffs/20261117_committed_fix_review.md): an early null pick can use up the orders a later step needs, so a trajectory draw can be
 short of the score's pulls - such a comparison is now NOT-EVALUABLE, and a decision-level matched null is declared as the fallback.

ORDERS (neutral: the queue place test's JOIN population): every counted causal state of the queue place grid (every 100th 50 ms decision
 step of every window-side; t >= ws + 2 s, t + 5 s <= ws + 298 s; known two-sided book, spread >= 1) posts one 30 sh order at the known best
 at t, live at t + 58 ms, held to END = t + 5 s on the actual tape and book through the validated branch kernel (queue M, unknown depth 0,
 structural drift / cross cancels, nothing else). This KEEP branch is v1's JOIN row. Exclusions are statuses: the ledger's path statuses,
 price_out_of_range and refused_post_only (the order never rests). Every eligible order is in the score population; a decision whose
 fair input is missing is counted and falls back to KEEP at that step.
DECISIONS (LIVE): at d_j = t + 50 j ms for j = J0..98, J0 = 2 (+100 ms: the first step after the client knows the order is live at +58 ms,
 C13), while the CLIENT knows the order is live (a full fill is known 100 ms after it; a structural cancel is the client's own decision,
 known when it is made), KEEP or PULL. A pull decided at d_j removes the order at r_j = d_j + L, L = 37 ms (the measured median cancel
 latency, pf_clock_rl.py). Both branches share everything before r_j: side, price, remaining size, queue place, structural cancels, END
 and its mark. The +50 ms step (before the order is live; the kernel's pending cancel lands at max(d + L, +95 ms)) is a PENDING WITHDRAWAL
 of the post, reported on its own row, never in the live comparisons.
VALUES (c per order, marked at END with the valid mid at t + 5 s, as v1): V_keep, and V_pull(j) = the value of the fills before r_j.
 X_j = V_keep - V_pull(j) is the value of keeping from d_j: a pull there forgoes the gain G = max(X_j, 0) and avoids the loss
 L = max(-X_j, 0), split into ordinary L1 = min(L, 30 c) and severe excess L2 = max(L - 30 c, 0) (pf_gain_loss SEVERE = 30 c an action);
 a severe event is L > 30 c. Exact fills: each order is also evaluated with its removal placed between consecutive trades that can fill it
 (the same order with an earlier removal; the kernel unchanged), which gives its fill at every trade time; any removal time is then a sum
 over fills. Fill clocks M50 (primary), M250 and M0. Per fill: young (the order's first 100 ms after it is live; mostly phantom on live
 orders, B1a) and the trade's anchor source (WS-timed / E2-estimated / imputed; mixed when trades at one time differ).
POLICIES (at most one pull per order: every order has one outcome):
 KEEP      never pull (the reference).
 PULL-ALL  pull every order at its first live decision (what is at stake).
 ORACLE    EX POST, NOT DEPLOYABLE: the live pull step (or keep) with the order's best value, knowing its future fills and END mark. With one
           price and one END mark every share of an order has the same value, so this is perfect END-sign selection at the first live
           decision - an upper bound within this setup, not ideal detection timing later in the order's life. Printed at L = 0 / 25 / 37 /
           100 / 250 / 500 ms (25 = the measured lower bound, 37 the median), and ORACLE-PENDING (also allowed the pending withdrawal).
 DELAYS    KEEP's loss split by what stops the oracle: the posting window (fills before a withdrawal sent at +50 ms could land, +95 ms),
           the pending-to-live gap (+95 to +100 ms, and orders the client knows are gone before +100 ms), the cancel latency at the first
           live decision (the same decisions valued at L = 0 and at L = 37 ms) and what remains avoidable at 37 ms.
 SCORE A   the declared causal score (no fit): the order's fair edge e_j = sg (p_bn(d_j) - p), p_bn = step 5's Binance-implied fair price
           with the relay basis rebuilt (pf_stop_ev_minutes_inputs.fair_fix), at d_j on the program's decision clock; pull at the first
           live decision with e_j < theta; theta = 0 primary, +1 / -1 c printed. A missing input never triggers (fallback KEEP).
 SCORE B   printed: the fair move since posting, sg (p_bn(d_j) - p_bn(t)) <= -theta, theta = 0.5 / 1 / 2 c (never triggers when the
           posting-time fair price is missing).
 NULLS     two matched nulls per score policy (500 draws for the declared one, 200 for each printed one), strata (day, market, side,
           contract minute), only information known at each step (no future fills or lifetime):
           TRAJ    at every live step, in every stratum, as many pulls as the score executed there, drawn from the orders the client knows
                   are live and this draw has not pulled yet (one outcome per order). An early pick can use up the orders a later step
                   needs: the shortfall is counted, and a comparison with ANY unmatched draw is NOT-EVALUABLE - no matched claim; the draws
                   are kept, never re-drawn or forced to match.
           STATES  for every (stratum, live step), the same count of decision states drawn from the orders the client knows are live at
                   that step on the neutral KEEP path, each valued alone - matched exactly by construction (the score's own pulls are in
                   that risk set); an order may be drawn at two steps of one draw.
           Declared hierarchy: K2 / K3 read against TRAJ when every draw matches, else against STATES (the reading then ends in -STATES).
READINGS (JOIN, M50, L = 37 ms, all fills; computed by report(); the day rule is >= 4 of 5 days):
 K1 ORACLE-BOUND       the live oracle's avoided loss / KEEP's loss (sum of max(-V_keep, 0)) >= 0.5 pooled and by the day rule
                       -> AVOIDABLE-AT-LATENCY, else NOT-SHOWN (a failure does not identify the cause; the DELAYS rows do).
 K2 SCORE-EARNS        (A, theta 0) its improvement over KEEP (sum of V_policy - V_keep) above the matched null's 95th percentile pooled and
                       above its mean by the day rule -> EARNS-OVER-MATCHED (-STATES), else NOT-SHOWN; the null by the declared hierarchy.
                       The same predicate is computed at M250, at M0 and without young fills; the reading says which hold (C1, C2).
 K3 SCORE-RANKS-VALUE  (A, theta 0) the value of the removed fills per removed filled share below the null's 5th percentile pooled and
                       below the null's mean by the day rule -> RANKS-VALUE (economic ranking beyond fill prediction, at matched
                       participation), else NOT-SHOWN.
 K4 LATENCY            numbers, no label: the improvement of the same decisions valued at L = 0 and at L = 37 ms and their gap, for the
                       live oracle and for A (theta 0), pooled and by day. Significance flags at two latencies would not establish a causal
                       latency effect, and a score's failure is not an information limit.
 Printed: for every policy the foregone gains, avoided L1 / L2, severe events, net, pulls, removed filled shares and live time lost, by
 day; K2 / K3 for the five other score policies (multiplicity: 6 score policies - a pass outside the declared one is a candidate, never a
 reading); the oracle's latency curve; young and trade-source splits; structural cancels by cause and side (P47); every exclusion count.
CONTROLS (any failure -> no output): (1) KEEP reproduces v1's JOIN rows bit for bit (value, filled shares, M250 value; first-100-ms value
 / shares to 1e-7 / 1e-9) and the order set equals v1's saved JOIN states; (2) per-trade fills sum to the keep branch's shares and value on every order
 and clock, cumulative fills never fall, every fill is on the order's side at or through its price; (3) the kernel run with the order
 removed at a sampled r_j equals the per-trade sum before r_j (every clock); (4) the trade index rebuilt for the source split equals the
 path's trades exactly; (5) the structural-cancel mirror leaves no kernel fill at or after its cancel landing; (6) decision invariance
 (C13): on sampled orders the fair inputs after a cut are replaced - no decision before the cut may change; (7) the null strata are exactly
 (day, market, side, minute), every null pull is taken from its own stratum's risk set (an eligible live decision), and null pulls +
 unmatched = the score's executed pulls on every draw; (8) the END mark is the same at every clock; (9) the kernel wrapper's clocks are 58 / 37 / 95 ms.
SELFTEST (synthetic tape only, no market data): the queue place ledger's synthetic markets plus scripted market 108 (an adverse sweep after
 a fair-price drop, a pending-cancel fill, a fill exactly at a pull's landing time, mixed trade sources); v1 run on the same tape for
 control (1); planted fills, pull values, structural causes (incl. P47), refusals and score decisions as scripted; every pull value of the
 planted orders equal to the kernel's own; the reading predicates on constructed inputs; the review's counterexamples (raw-trigger
 shuffle against orders already gone: the null must match the 500 executed pulls and K2 must fail; equal eligibility: 500 pulls on every
 draw; fills at +70 / +250 ms: the live oracle saves 10 c an order at 0 / 25 / 37 ms, K1 NOT-SHOWN, the shortfall in the posting window,
 not the cancel latency; a missing fair input late or at the trigger step: the order stays in, the trigger moves to the next valid step);
 planted errors (a dropped fill, the pull landing without latency, a score reading the next step, a null drawn from another stratum) and
 known-bad inputs (v1 rows changed where only the
 hash can see it, a v1 value moved with its hash updated, existing output, undeclared day) each refused BY ITS OWN control (message checked). MUTATION CHECK:
 pf_keep_pull_mutants.py plants code errors in a copy one at a time - each must make the selftest fail, the unchanged copy must pass.
LIMITS: consumed days Oct23-27 (development only, no validation claim); replay fill model (queue M unvalidated on live; young fills mostly
 phantom, B1a); one order at a time and no re-post after a pull (a pull ends that order's exposure only); one common END (not per-fill
 markouts); the kernel's one-sided-book cancel asymmetry (P47, counted); the scores are declared, not fitted - their failure would not show
 that no causal score can rank keep / pull.
Usage (run_light, PF_SLOT=claude): pf_keep_pull.py selftest | day <day> [limit] | report; driver run_keep_pull.sh
"""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_queue_place_test as QP                          # noqa: E402  v1 (frozen): build_path, evaluate, order
import pf_queue_place_ledger as QL                        # noqa: E402  grid statuses, post-only mirror, synthetic tape, v1 on a source

K, C, pd, I = QL.K, QL.C, QL.pd, QL.I
OUT = Path(I.K.IO) / 'keep_pull_20261117'
V1 = QP.OUT
DAYS = QP.DAYS
H = QP.H_MAIN                                             # the order's END: t + 5 s
STEP, NSTEP = 50., 98                                     # decision steps d_j = t + 50 j, j = 1..98 (the last pull lands before END)
PLACE, CANCEL, PENDING, NOTICE = 58., 37., 95., 100.      # live after posting; cancel landing; pending-cancel landing; own fill known
HOLD = 3.                                                 # the kernel's structural drift limit (ticks behind the best, + rung)
LATS = (0., 25., 37., 100., 250., 500.)
CLOCKS = (50., 250., 0.)                                  # trade shift (M50 primary)
CTAG = {50.: '', 250.: '_m250', 0.: '_m0'}
SIZE, SEVERE, YOUNG = 30., 30., 100.
THETA_A, THETA_B = (0., 1., -1.), (.5, 1., 2.)
J0 = int(np.ceil(PLACE / STEP))                          # first live decision step (+100 ms): the client knows the order is live at +58 ms
NULL_DRAWS, NULL_DRAWS_PRINTED, SEED = 500, 200, 20261117
SRC = ('ws', 'e2', 'imputed', 'mixed')
STATUS = ('eligible', 'refused_post_only', 'price_out_of_range') + tuple(s for s in QL.PATH_ST if s != 'eligible')
CAUSE = ('', 'own_side_empty', 'behind', 'cross_opp_empty', 'cross')
MUT_DROP_FILL = MUT_NO_LATENCY = MUT_LOOKAHEAD = MUT_NULL_CROSS = 0      # planted errors (selftest only)


def landing(j, lat=CANCEL):
    """Cancel landing (ms after posting) of a pull decided at step j (d = 50 j ms): the kernel's rule (pending before the order is live)."""
    d = STEP * np.asarray(j, float)
    if MUT_NO_LATENCY:
        return d
    return np.where(d < PLACE, np.maximum(d + lat, PENDING), d + lat)


def trade_index(W, t, end, shift):
    """QP.build_path's tape selection and order, keeping each trade's (side, index) - must equal the path's trades exactly (control 4)."""
    tape = []
    for side in (0, 1):
        z = W[side]; tt = np.asarray(z['anc']) - shift - t; ix = np.flatnonzero((tt > 0) & (tt <= end - t))
        tape.extend((float(tt[i]), side, float(z['ptl'][i]), float(z['shl'][i]), int(i)) for i in ix)
    tape.sort(key=lambda x: (x[0], x[1], x[4]))
    return np.asarray([x[:4] for x in tape], float).reshape(-1, 4), [(x[1], x[4]) for x in tape]


def fills(p, side, price):
    """Exact per-trade fills of the order QP.order(side, 0, price) on path p to END, from the kernel: the same order is also evaluated with
    its removal between consecutive trades that can fill it. Returns (keep value, keep shares, fill times, fill shares, rows that can fill)."""
    tr = p['trades']; thru = tr[:, 2] < price if side == 0 else tr[:, 2] > price
    can = np.flatnonzero((tr[:, 1] == side) & ((tr[:, 2] == price) | thru)); tt = np.unique(tr[can, 0])
    cut = np.r_[(tt[:-1] + tt[1:]) / 2., tt[-1:] + 1.]
    o = np.tile(np.asarray(QP.order(side, 0, price), float), (len(cut) + 1, 1)); o[:len(cut), 6] = cut      # the last row keeps
    v = np.zeros(len(o)); q = np.zeros(len(o))
    for a in range(0, len(o), 72):
        v[a:a + 72], q[a:a + 72] = QP.evaluate(p, o[a:a + 72], H)
    fq = np.diff(np.r_[0., q[:-1]])
    if MUT_DROP_FILL and (fq > 0).any():
        fq[np.flatnonzero(fq > 0)[0]] = 0.
    return float(v[-1]), float(q[-1]), tt, fq, can


def struct_cancel(p, side, price):
    """The kernel's structural cancel of the order (rung 0) on its 50 ms grid, from the path's known book -> (decision ms, landing ms,
    cause), or (nan, nan, '') if it never fires. Grid 0 is skipped for a new order; one-sided books follow the kernel (P47)."""
    f = p['frames']; known = 0
    for g in np.arange(STEP, H, STEP):
        while known + 1 < len(f) and f[known + 1, 1] <= g:
            known += 1
        best, opp = f[known, 2 + side * 10], f[known, 2 + (1 - side) * 10]
        behind = best - price if side == 0 else price - best
        cross = price >= opp if side == 0 else price <= opp
        if best < 0 or behind > HOLD or cross:
            cause = 'own_side_empty' if best < 0 else 'behind' if behind > HOLD else 'cross_opp_empty' if opp < 0 else 'cross'
            return float(g), float(landing(g / STEP)), cause
    return np.nan, np.nan, ''


def first(mask):
    """First decision step (1..98) where a row of the (n, 98) mask is true; 99 if never."""
    return np.where(mask.any(1), mask.argmax(1) + 1, NSTEP + 1)


def score_steps(f, side, price):
    """Declared causal scores -> {policy: first LIVE trigger step (J0..98, 99 = never)}. f: (n, 99) fair price known at d_0 (posting) ..
    d_98, NaN where missing: a missing input never triggers (the declared fallback is KEEP at that step; B never triggers without its
    posting-time fair price)."""
    sg = np.where(np.asarray(side) == 0, 1., -1.)[:, None]; x = f[:, 1:]
    if MUT_LOOKAHEAD:
        x = np.c_[f[:, 2:], f[:, -1:]]
    live = np.arange(1, NSTEP + 1)[None, :] >= J0
    a = sg * (x - np.asarray(price, float)[:, None]); b = sg * (x - f[:, :1]); out = {}
    for th in THETA_A:
        out[f'A{th:+g}'] = first((a < th) & live)
    for th in THETA_B:
        out[f'B{th:g}'] = first((b <= -th) & live)
    return out


def invariant(f, side, price, rng):
    """Control 6 (C13): replace every fair input after a random cut by an extreme value (each direction); no decision at or before the cut may
    change. Returns the number of (order, policy, direction) cases that changed."""
    base = score_steps(f, side, price); n = len(f); cut = rng.integers(1, NSTEP + 1, n); bad = 0
    for sign in (-1., 1.):
        g = f.copy(); later = np.arange(NSTEP + 1)[None, :] > cut[:, None]; g[later] = (np.asarray(price, float)[:, None] + sign * 1000.).repeat(NSTEP + 1, 1)[later]
        alt = score_steps(g, side, price)
        bad += sum(int((np.minimum(base[k], cut + 1) != np.minimum(alt[k], cut + 1)).sum()) for k in base)
    return bad


class Tape(QL.Tape):
    """The production source (v1's market list, imputation offset and loader) plus each trade's anchor source and the fair price."""
    def fair(self, day):
        reg = QP.JD.metadata(day, True)[1]
        fx, _ = I.fair_fix(day, {int(r.mid): float(r.ws) * 1000. for r in reg.itertuples()})
        return {int(m): (np.asarray(v[0], float), np.asarray(v[1], float)) for m, v in fx.items()}

    def sources(self, data, m):
        """Per side: (source code, anchor) of every trade in window_sides' order (its lexsort of the loader's legs)."""
        out = {}
        for s in (0, 1):
            if (m, s) in data['LI']:
                l = data['LT'].iloc[data['LI'][(m, s)]]; o_ = np.lexsort(((-l.ptick.values if s == 0 else l.ptick.values), l.anc.values))
                out[s] = (np.where(l.ws_timed.values, 0, np.where(l.imputed.values, 2, 1))[o_].astype(np.int8), l.anc.values[o_].astype(float))
            else:
                out[s] = (np.zeros(0, np.int8), np.zeros(0))
        return out


def window(W, day, codes, fair, rows, frows, st, inv):
    """One window: every grid step -> its status; every counted state -> one JOIN order with its status, fills at three clocks, the
    structural-cancel mirror and the fair path (controls 2-5 and 8 on every order)."""
    m = int(W[0]['m']); ws = float(W[0]['ws']); tg = np.asarray(W[0]['tg'], float); kn = np.asarray(W[0]['jk'])
    for s in (0, 1):
        C.require(np.array_equal(codes[s][1], np.asarray(W[s]['anc'], float)), 'trade sources not aligned with the window tape')
    tau, pv = fair.get(m, (np.zeros(0), np.zeros(0)))
    for s in (0, 1):
        sg = 1. if s == 0 else -1.; P5 = np.asarray(W[s]['P5'], float); O5 = np.asarray(W[1 - s]['P5'], float)
        for k in range(0, len(tg), QP.STATE_EVERY):
            t = float(tg[k]); jj = int(kn[k])
            best = float(P5[jj, 0]) if jj >= 0 else np.nan; opp = float(O5[jj, 0]) if jj >= 0 else np.nan; spread = (opp - best) if s == 0 else (best - opp)
            why = ('before_start_margin' if t < ws + 2000. else 'expiry_censored' if t + H > ws + 298000. else 'no_known_book' if jj < 0
                   else 'one_sided_book' if best < 0 or opp < 0 else 'locked_or_crossed' if spread < 1 else 'counted')
            st['grid_' + why] = st.get('grid_' + why, 0) + 1
            if why != 'counted':
                continue
            oid = st['orders']; st['orders'] += 1
            rec = dict(day=day, oid=oid, m=m, side=s, t=t, ws=ws, phase=min(5, int((t - ws) // 60000) + 1), p=best, spread=spread)
            pth, pst = QP.build_path(W, t, t + H) if 1 <= best <= 99 else (None, 'price_out_of_range')
            if pth is None:
                rows.append(dict(rec, status=pst)); continue
            rej = QL.postonly(pth, s, best); g, land, cause = struct_cancel(pth, s, best)
            rec.update(status='refused_post_only' if rej else 'eligible', mark=pth['mark'], g_struct=g, land_struct=land, cause=cause)
            for c in CLOCKS:
                tag = CTAG[c]; pc = pth if c == 50. else QP.build_path(W, t, t + H, shift=c)[0]
                C.require(pc is not None and pc['mark'] == pth['mark'] and np.array_equal(pc['frames'], pth['frames']), 'paths differ beyond the trade shift')
                trs, ix = trade_index(W, t, t + H, c); C.require(np.array_equal(trs, pc['trades']), 'trade index differs from the path')
                vk, qk, tt, fq, can = fills(pc, s, best); val = fq * sg * (pc['mark'] - best)
                C.require(abs(fq.sum() - qk) <= 1e-9 and abs(val.sum() - vk) <= 1e-7 and (fq >= -1e-12).all(), 'per-trade fills do not sum to the keep branch')
                C.require(not (fq[tt >= land] > 1e-12).any(), 'a fill at or after the structural cancel landed')
                C.require(not rej or (qk == 0 and vk == 0), 'post-only mirror: a refused order filled')
                if oid % 37 == 0:                                                  # control 3: the kernel itself with a sampled pull
                    j = 1 + (oid * 7 + int(c)) % NSTEP; r = float(landing(j)); o = QP.order(s, 0, best); o[6] = r
                    v1_, q1_ = QP.evaluate(pc, [o], H); st['kernel_checks'] += 1
                    C.require(abs(v1_[0] - val[tt < r].sum()) <= 1e-7 and abs(q1_[0] - fq[tt < r].sum()) <= 1e-9, 'kernel pull differs from the per-trade sum')
                cum = np.cumsum(fq); full = np.flatnonzero(cum >= SIZE - 1e-9)
                rec.update({'v' + tag: vk, 'q' + tag: qk, 't_full' + tag: float(tt[full[0]]) if len(full) else np.nan})
                if c == 50.:
                    rec.update(vy=float(val[tt < PLACE + YOUNG].sum()), qy=float(fq[tt < PLACE + YOUNG].sum()))
                for x, q_ in zip(tt, fq):
                    if q_ > 1e-12:
                        cs = {int(codes[s][0][ix[r_][1]]) for r_ in can if trs[r_, 0] == x}
                        frows.append((oid, c, float(x), float(q_), cs.pop() if len(cs) == 1 else 3))
            want = t + STEP * np.arange(NSTEP + 1); fj = np.full(NSTEP + 1, np.nan)
            if len(tau):
                i_ = np.clip(np.searchsorted(tau, want), 0, len(tau) - 1); okj = (tau[i_] == want) & np.isfinite(pv[i_]); fj[okj] = pv[i_][okj]
            rec['fair_missing'] = int(np.isnan(fj).sum())                     # steps 0..98 without a causal fair price (decision-level fallback)
            rec.update({f'f{j:02d}': float(fj[j]) for j in range(NSTEP + 1)})
            if np.isfinite(fj).any() and oid % 11 == 0:
                inv.append((fj, s, best))
            rows.append(rec)


def reconcile(O, v1, day, markets=None):
    """Control 1: KEEP == v1's JOIN rows (value, shares and M250 value bit for bit; first-100-ms value / shares to 1e-7 / 1e-9), the same
    states, and v1's counted-state counter (full days only). Returns a summary; any mismatch raises."""
    d = v1 / day; r = json.loads((d / 'receipt.json').read_text())
    C.require(C.sha(d / 'rows.parquet') == r['outputs']['rows.parquet'], 'v1 rows changed since its receipt')
    A = pd.read_parquet(d / 'rows.parquet'); A = A[(A.kind == 'state') & (A.arm == 'JOIN')]
    if markets is not None:
        A = A[A.m.isin(markets)]
    else:
        C.require(int(r['stat']['states']) == len(O), f"counted states {len(O)} != v1's {r['stat']['states']}")
    B = O[O.status.isin(('eligible', 'refused_post_only'))]; key = ['m', 'side', 't']
    x = A.merge(B, on=key, how='outer', suffixes=('_1', ''), indicator=True)
    C.require((x._merge == 'both').all(), f"JOIN states differ from v1: {int((x._merge == 'left_only').sum())} only in v1, {int((x._merge == 'right_only').sum())} only here")
    same = lambda a, b: bool(((x[a].values == x[b].values) | (np.isnan(x[a].values) & np.isnan(x[b].values))).all())
    C.require(same('v_1', 'v') and same('q_1', 'q') and same('v_m250_1', 'v_m250'), 'KEEP differs from v1 (value, shares or M250 value)')
    C.require(np.allclose(x.vy_1, x.vy, rtol=0, atol=1e-7) and np.allclose(x.qy_1, x.qy, rtol=0, atol=1e-9), 'first-100-ms fills differ from v1')
    C.require(np.allclose(x.p_1, x.p, rtol=0, atol=0) and np.allclose(x.spread_1, x.spread, rtol=0, atol=0), 'price or spread differs from v1')
    return dict(v1_join_rows=int(len(A)), matched=int(len(x)), v1_counted=int(r['stat']['states']), counted=int(len(O)))


def provenance():
    def git(*a):
        return subprocess.run(('git', '-C', str(HERE)) + a, capture_output=True, text=True).stdout.strip()
    files = [Path(__file__).resolve(), Path(QP.__file__).resolve(), Path(QL.__file__).resolve(), HERE / 'pf_flow_control_kernel.cpp', HERE / 'pf_flow_control_kernel.py']
    return dict(commit=git('rev-parse', 'HEAD'), source={p.name: C.sha(p) for p in files}, uncommitted=git('status', '--porcelain', '--', *map(str, files)))


def day_run(day, limit=None, src=None, v1=None, out=None):
    """One day: every window -> orders.parquet (one row per counted state) + fills.parquet (one row per fill and clock); controls 1-6, 8, 9."""
    C.require(day in DAYS, 'undeclared day'); src = src or Tape(); v1 = Path(v1 or V1); out = Path(out or OUT); start = time.time()
    C.require('float(horizon),58.,37.,95.,3.' in Path(K.__file__).read_text(), 'kernel clocks are not 58 / 37 / 95 ms, hold 3')
    d = out / (f'smoke_{day}' if limit else day); C.require(not d.exists(), 'day output exists')
    markets = src.markets(day); markets = markets[:limit] if limit else markets; offset = src.offset(day); fair = src.fair(day)
    rows, frows, inv = [], [], []; st = dict(markets=len(markets), windows=0, orders=0, kernel_checks=0)
    for begin in range(0, len(markets), 24):
        data = src.load(day, markets[begin:begin + 24], offset)
        for W in src.pairs(data):
            st['windows'] += 1; window(W, day, src.sources(data, int(W[0]['m'])), fair, rows, frows, st, inv)
        del data; C.guard('keep-pull day')
        print(json.dumps(dict(stage='day', day=day, markets=min(begin + 24, len(markets)), orders=st['orders'], seconds=round(time.time() - start), rss=round(I.K.rss_mib()))), flush=True)
    O = pd.DataFrame(rows); Fl = pd.DataFrame(frows, columns=['oid', 'clock', 't', 'q', 'src'])
    C.require(len(O) == st['orders'] and O.status.isin(STATUS).all() and O.oid.is_unique, 'order table')
    if inv:                                                                    # control 6 (C13)
        f = np.stack([x[0] for x in inv]); st['invariance_checked'] = len(inv)
        st['invariance_changed'] = invariant(f, np.array([x[1] for x in inv]), np.array([x[2] for x in inv]), np.random.default_rng(SEED + int(day)))
        C.require(st['invariance_changed'] == 0, f"a decision changed when only later fair inputs changed ({st['invariance_changed']})")
    C.require(st['kernel_checks'] > 0 or not (O.status == 'eligible').any(), 'no kernel pull check ran')
    rec = reconcile(O, v1, day, markets if limit else None)
    d.mkdir(parents=True); O.to_parquet(d / 'orders.parquet', index=False); Fl.to_parquet(d / 'fills.parquet', index=False)
    C.dump(d / 'receipt.json', dict(asof=C.stamp(), day=day, limit=limit, stat=st, reconciliation=rec, provenance=provenance(),
                                    status=O.status.value_counts().to_dict(),
                                    fair_missing=dict(orders_with_any=int((O.fair_missing > 0).sum()), steps=int(O.fair_missing.sum())) if 'fair_missing' in O else {},
                                    outputs={x: C.sha(d / x) for x in ('orders.parquet', 'fills.parquet')}, seconds=time.time() - start))
    print(json.dumps(dict(day=day, stat=st, reconciliation=rec, seconds=round(time.time() - start))), flush=True)
    return d


POLICIES = ('A+0', 'A+1', 'A-1', 'B0.5', 'B1', 'B2')      # score policies (score_steps names); A+0 is the declared one
DECLARED = 'A+0'
DRAWN = ((50., False, CANCEL), (250., False, CANCEL), (0., False, CANCEL), (50., True, CANCEL), (50., False, 0.))   # variants with nulls


def vkey(c, noyoung, lat):
    return f"M{int(c)}{'_noyoung' if noyoung else ''}_L{int(lat)}"


def load(out=None, days=DAYS):
    """Every day's orders and fills; each receipt's output hashes must match its files."""
    out = Path(out or OUT); O, Fl = [], []
    for day in days:
        d = out / day; r = json.loads((d / 'receipt.json').read_text())
        C.require(all(C.sha(d / x) == h for x, h in r['outputs'].items()), f'{day}: outputs changed since the receipt')
        O.append(pd.read_parquet(d / 'orders.parquet')); f = pd.read_parquet(d / 'fills.parquet'); f['day'] = day; Fl.append(f)
    return pd.concat(O, ignore_index=True), pd.concat(Fl, ignore_index=True)


def variant(E, Fl, c, noyoung, lat):
    """Per order of E at one fill clock, young rule and latency: V_keep, Q_keep, and for j = 1..98 the value / shares of the fills before
    r_j (Mv / Mq); the client's last decision step (endj: decisions at j < endj) and the order's venue end of life."""
    tag = CTAG[c]; n = len(E); ix = pd.Series(np.arange(n), index=pd.MultiIndex.from_arrays([E.day.values, E.oid.values]))
    f = Fl[Fl.clock == c]; ri = ix.reindex(pd.MultiIndex.from_arrays([f.day.values, f.oid.values])).values; ok = ~np.isnan(ri)
    if noyoung:
        ok &= f.t.values >= PLACE + YOUNG
    f = f[ok]; ri = ri[ok].astype(int); sg = np.where(E.side.values == 0, 1., -1.)
    val = f.q.values * sg[ri] * (E.mark.values[ri] - E.p.values[ri])
    jmin = np.searchsorted(landing(np.arange(1, NSTEP + 1), lat), f.t.values, side='right')
    Dv = np.zeros((n, NSTEP + 1)); Dq = np.zeros((n, NSTEP + 1)); np.add.at(Dv, (ri, jmin), val); np.add.at(Dq, (ri, jmin), f.q.values)
    np.cumsum(Dv, 1, out=Dv); np.cumsum(Dq, 1, out=Dq)
    if not noyoung:
        C.require(np.allclose(Dv[:, -1], E['v' + tag].values, rtol=0, atol=1e-7) and np.allclose(Dq[:, -1], E['q' + tag].values, rtol=0, atol=1e-9), 'fills differ from the keep values')
    lim = np.fmin(E.g_struct.values, E['t_full' + tag].values + NOTICE); d = STEP * np.arange(1, NSTEP + 1)
    endj = np.where(np.isnan(lim), NSTEP + 1, np.searchsorted(d, lim, side='left') + 1)
    life = np.fmin(np.fmin(E['t_full' + tag].values, E.land_struct.values), H); life = np.where(np.isnan(life), H, life)
    return dict(Vk=Dv[:, -1], Qk=Dq[:, -1], Mv=Dv[:, :NSTEP], Mq=Dq[:, :NSTEP], endj=endj, life=life, lat=lat)


def _sums(V, ii, steps, dayi, nd):
    """Per-day sums for pulls of orders ii at decision steps `steps`, each valued against KEEP: improvement, foregone gains, avoided ordinary
    / severe-excess losses, severe events, removed value X and shares R, live time lost."""
    jj = steps - 1
    X = V['Vk'][ii] - V['Mv'][ii, jj]; R = V['Qk'][ii] - V['Mq'][ii, jj]; L = np.maximum(-X, 0.)
    lost = np.maximum(V['life'][ii] - landing(steps, V['lat']), 0.)

    def by(w):
        return np.bincount(dayi[ii], weights=w, minlength=nd)
    return dict(n=by(np.ones(len(ii))), imp=by(-X), G=by(np.maximum(X, 0.)), L1=by(np.minimum(L, SEVERE)), L2=by(np.maximum(L - SEVERE, 0.)),
                sev=by((L > SEVERE).astype(float)), X=by(X), R=by(R), lost=by(lost))


def outcome(V, P, dayi, nd, j0=J0):
    """A policy's per-day sums: pull steps P (j0..98 executed while the client knows the order is live; 99 = keep), one outcome per order."""
    ii = np.flatnonzero((P >= j0) & (P <= NSTEP) & (P < V['endj']))
    return _sums(V, ii, P[ii], dayi, nd)


def outcome_states(V, oi, oj, dayi, nd):
    """The decision-level null's per-day sums: decision states (order oi, live step oj), each valued alone against KEEP."""
    C.require(bool(((oj >= J0) & (oj <= NSTEP) & (oj < V['endj'][oi])).all()), 'a matched decision state is not a live decision')
    return _sums(V, oi, oj, dayi, nd)


def oracle(V, j0=J0):
    """EX POST, NOT DEPLOYABLE: per order the pull step (j0..98, while the client knows the order is live) with the best value, or keep (99)."""
    cols = np.arange(1, NSTEP + 1)[None, :]; M = np.where((cols >= j0) & (cols < V['endj'][:, None]), V['Mv'], -np.inf)
    return np.where(M.max(1) > V['Vk'] + 1e-9, M.argmax(1) + 1, NSTEP + 1)


def hierarchy(evaluable):
    """The declared choice of matched null for K2 / K3: TRAJ when every trajectory draw matches, else STATES."""
    return 'traj' if evaluable else 'states'


def _strata(cell, endj, P):
    """Orders by stratum in a padded (S, W) matrix (-1 = empty slot) with their endj (0 = empty), stratum ids, the executed pulls of P
    (J0 <= P < endj) counted per (stratum, step), and the steps / strata where P pulls."""
    n = len(P); act = (P >= J0) & (P <= NSTEP) & (P < endj)
    o = np.argsort(cell, kind='stable'); starts = np.r_[0, np.flatnonzero(np.diff(cell[o])) + 1]; size = np.diff(np.r_[starts, n])
    S, Wd = len(starts), int(size.max()); row = np.repeat(np.arange(S), size); col = np.arange(n) - np.repeat(starts, size)
    M = np.full((S, Wd), -1); M[row, col] = o; Em = np.zeros((S, Wd), int); Em[row, col] = endj[o]; sid = np.empty(n, int); sid[o] = row
    cnt = np.zeros((S, NSTEP + 2), int); np.add.at(cnt, (sid[act], P[act]), 1)
    steps = [j for j in range(J0, NSTEP + 1) if cnt[:, j].any()]
    return M, Em, sid, cnt, int(act.sum()), steps, {j: np.flatnonzero(cnt[:, j]) for j in steps}


def null_picks(endj, P, cell, seed, draws):
    """TRAJECTORY null (control 7): at every live step, in every stratum, as many pulls as P executed there, drawn at random from the
    stratum's orders the client knows are live at that step and this draw has not pulled yet (one outcome per order). An early pick can use up
    the orders a later step needs: the shortfall is counted as unmatched and such a draw cannot support a matched reading (report()).
    Returns [(pull steps, unmatched)]."""
    n = len(P); M, Em, sid, cnt, need, steps, at = _strata(cell, endj, P); S, Wd = M.shape
    rng = np.random.default_rng(seed); out = []
    for _ in range(draws):
        key = rng.random((S, Wd)); key[M < 0] = 2.; k = np.argsort(key, 1); Ms = np.take_along_axis(M, k, 1); Es = np.take_along_axis(Em, k, 1)
        picked = np.zeros((S, Wd), bool); PN = np.full(n, NSTEP + 1); um = 0
        for j in steps:
            r = at[j]; src = (r + 1) % S if MUT_NULL_CROSS else r
            elig = (Es[src] > j) & ~picked[src]; sel = elig & (np.cumsum(elig, 1) <= cnt[r, j][:, None]); picked[src] |= sel; who = Ms[src][sel]
            C.require(bool((sid[who] == np.repeat(r, sel.sum(1))).all()), 'a null pull left its stratum')
            PN[who] = j; um += int((cnt[r, j] - sel.sum(1)).sum())
        C.require(int((PN <= NSTEP).sum()) + um == need, 'null pulls + unmatched differ from the executed pulls')
        ok = PN <= NSTEP; c2 = np.zeros_like(cnt); np.add.at(c2, (sid[ok], PN[ok]), 1)
        C.require(bool((c2 <= cnt).all() and (um > 0 or (c2 == cnt).all())), 'null pulls per (stratum, step) differ from the executed pulls')
        out.append((PN, um))
    return out


def state_picks(endj, P, cell, seed, draws):
    """DECISION-LEVEL null: for every (stratum, live step) where P executed n pulls, n distinct orders drawn at random from the stratum's
    orders the client knows are live at that step on the neutral KEEP path - other steps' draws do not use them up; each state is valued alone.
    Matched exactly by construction: the score's own pulls are in that risk set. Returns [(orders, steps)]."""
    M, Em, sid, cnt, need, steps, at = _strata(cell, endj, P); S, Wd = M.shape; rng = np.random.default_rng(seed); out = []
    for _ in range(draws):
        oi, oj = [np.zeros(0, int)], [np.zeros(0, int)]
        for j in steps:
            r = at[j]; elig = Em[r] > j; k_ = cnt[r, j]
            C.require(bool((elig.sum(1) >= k_).all()), 'a matched decision state is missing from its risk set')
            k = np.argsort(np.where(elig, rng.random((len(r), Wd)), 2.), 1); who = np.take_along_axis(M[r], k, 1)[np.arange(Wd)[None, :] < k_[:, None]]
            C.require(bool((sid[who] == np.repeat(r, k_)).all() and (endj[who] > j).all()), 'a matched decision state left its stratum or is not live')
            oi.append(who); oj.append(np.full(len(who), j))
        oi, oj = np.concatenate(oi), np.concatenate(oj); c2 = np.zeros_like(cnt); np.add.at(c2, (sid[oi], oj), 1)
        C.require(bool((c2 == cnt).all()) and len(oi) == need, 'decision-level null not matched per (stratum, step)')
        out.append((oi, oj))
    return out


def k2(res, nul):
    """K2's predicate: improvement over KEEP above the null's 95th percentile pooled and above the null's mean on >= 4 of 5 days."""
    a = np.array([x['imp'].sum() for x in nul]); dm = np.mean([x['imp'] for x in nul], 0); days = int((res['imp'] > dm).sum())
    return bool(res['imp'].sum() > np.percentile(a, 95) and days >= 4), dict(imp=float(res['imp'].sum()), null_mean=float(a.mean()),
                                                                               null_p95=float(np.percentile(a, 95)), days_above=days)


def k1(orc, loss):
    """K1's predicate: the oracle avoids >= 50% of KEEP's loss pooled and on >= 4 of 5 days (per-day arrays)."""
    return bool(orc.sum() / max(loss.sum(), 1e-9) >= .5 and int((orc / np.maximum(loss, 1e-9) >= .5).sum()) >= 4)


def k3(res, nul):
    """K3's predicate: value of the removed fills per removed share below the null's 5th percentile pooled and below its mean on >= 4 days."""
    def ratio(x, y):
        return np.where(y > 0, x / np.where(y > 0, y, 1.), np.nan)
    r = float(ratio(res['X'].sum(), res['R'].sum())); a = np.array([float(ratio(x['X'].sum(), x['R'].sum())) for x in nul])
    dm = np.nanmean([ratio(x['X'], x['R']) for x in nul], 0); days = int((ratio(res['X'], res['R']) < dm).sum())
    ok = bool(np.isfinite(r) and np.isfinite(a).sum() >= len(nul) // 2 and r < np.nanpercentile(a, 5) and days >= 4)
    return ok, dict(per_share=r, null_mean=float(np.nanmean(a)), null_p5=float(np.nanpercentile(a, 5)), days_below=days)


def tot(res):
    return {k: float(v.sum()) for k, v in res.items()}


def byday(res):
    return {k: v.tolist() for k, v in res.items()}


def report(out=None):
    """Readings K1-K3, K4 in numbers, and the printed tables -> keep_pull.txt / keep_pull.json in the output directory."""
    out = Path(out or OUT); start = time.time(); O, Fl = load(out); nd = len(DAYS); dc = {d: i for i, d in enumerate(DAYS)}
    E = O[O.status == 'eligible'].reset_index(drop=True); dayi = E.day.map(dc).values
    cell = E.groupby(['day', 'm', 'side', 'phase'], sort=True).ngroup().values
    cc = pd.DataFrame(dict(c=cell, d=dayi, m=E.m.values, s=E.side.values, p=E.phase.values))
    C.require(bool((cc.groupby('c')[['d', 'm', 's', 'p']].nunique() == 1).all().all() and (cc.groupby(['d', 'm', 's', 'p']).c.nunique() == 1).all()),
              'null strata are not (day, market, side, minute)')
    F = E[[f'f{j:02d}' for j in range(NSTEP + 1)]].values; T = score_steps(F, E.side.values, E.p.values)
    loss_keep = lambda V: np.bincount(dayi, weights=np.maximum(-V['Vk'], 0.), minlength=nd)
    K37, K0 = vkey(50., False, CANCEL), vkey(50., False, 0.)
    J = dict(asof=C.stamp(), population=dict(status=O.groupby('day').status.value_counts().unstack(fill_value=0).to_dict('index'), eligible=int(len(E)),
                                            fair_missing_orders=int((E.fair_missing > 0).sum()), fair_missing_steps=int(E.fair_missing.sum()),
                                            fills=Fl.groupby(['day', 'clock']).size().unstack(fill_value=0).to_dict('index')))
    L = ['KEEP OR PULL THE SAME RESTING ORDER AT THE ACTUAL CANCEL LATENCY (pf_keep_pull.py; replay, consumed days Oct23-27, development only)',
         'JOIN orders (30 sh at the known best, live at +58 ms, END +5 s, valued at the END mid); LIVE pulls decided every 50 ms from +100 ms land after L ms; c per order', '',
         f"population: {len(O):,} counted states, {len(E):,} eligible resting orders (all in the score population); {int((E.fair_missing > 0).sum()):,} orders miss a causal fair price "
         f"at some step ({int(E.fair_missing.sum()):,} steps; such a decision falls back to KEEP)"]
    for day in DAYS:
        L.append(f"  {day}: " + ', '.join(f'{k} {v:,}' for k, v in O[O.day == day].status.value_counts().items()))
    plan = [(50., False, 0.), (50., False, CANCEL)] + [(50., False, x) for x in LATS if x not in (0., CANCEL)] + [(50., True, CANCEL), (250., False, CANCEL), (0., False, CANCEL)]
    R = {}; D = {}; picks = {}
    for c, noyoung, lat in plan:
        V = variant(E, Fl, c, noyoung, lat); key = vkey(c, noyoung, lat)
        orc = outcome(V, oracle(V), dayi, nd); pa = outcome(V, np.full(len(E), J0), dayi, nd); pend = outcome(V, oracle(V, 1), dayi, nd, 1)
        R[key] = dict(keep_value=float(V['Vk'].sum()), keep_loss=float(loss_keep(V).sum()), keep_loss_by_day=loss_keep(V).tolist(), pull_all=tot(pa),
                      pull_all_by_day=byday(pa), oracle=tot(orc), oracle_by_day=byday(orc), oracle_pending=tot(pend))
        if key == K0:
            O0 = oracle(V); R[key]['oracle_pending_by_day'] = byday(pend)                 # the zero-latency oracle's decisions (valued at 37 ms below)
        if key == K37:
            P0 = V; cols = np.arange(1, NSTEP + 1)[None, :]; livem = (cols >= J0) & (cols < V['endj'][:, None])
            R[key]['oracle_0ms_decisions'] = byday(outcome(V, O0, dayi, nd))               # the same decisions as the 0 ms oracle, valued at 37 ms
            J['population'].update(live_decisions=int(livem.sum()), live_decisions_fair_missing=int((np.isnan(F[:, 1:]) & livem).sum()))
        if (c, noyoung, lat) in DRAWN:
            if c not in picks:                                                         # one clock at a time; reused by its variants
                picks = {c: (null_picks(V['endj'], T[DECLARED], cell, SEED, NULL_DRAWS), state_picks(V['endj'], T[DECLARED], cell, SEED + 100, NULL_DRAWS))}
            D[key] = {}
            for i, pol in enumerate(POLICIES if key == K37 else (DECLARED,)):
                pk, sk = picks[c] if pol == DECLARED else (null_picks(V['endj'], T[pol], cell, SEED + i, NULL_DRAWS_PRINTED),
                                                           state_picks(V['endj'], T[pol], cell, SEED + 100 + i, NULL_DRAWS_PRINTED))
                res = outcome(V, T[pol], dayi, nd); nul = [outcome(V, PN, dayi, nd) for PN, _ in pk]; nus = [outcome_states(V, oi, oj, dayi, nd) for oi, oj in sk]
                C.require(all(int(round(x['n'].sum())) == int((PN <= NSTEP).sum()) for x, (PN, _) in zip(nul, pk)), 'a null pull was not an eligible live decision')
                C.require(all(np.array_equal(x['n'], res['n']) for x in nus), 'decision-level null not matched on executed pulls by day')
                um = [u for _, u in pk]; ev = all(u == 0 for u in um); D[key][pol] = d_ = dict(policy=tot(res), policy_by_day=byday(res), draws=len(pk))
                for nm, nn in (('traj', nul), ('states', nus)):
                    ok2, s2 = k2(res, nn); ok3, s3 = k3(res, nn)
                    d_[nm] = dict(k2=ok2, k3=ok3, k2_stats=s2, k3_stats=s3, null_mean={k: float(np.mean([x[k].sum() for x in nn])) for k in res},
                                  null_by_day_mean={k: np.mean([x[k] for x in nn], 0).tolist() for k in res})
                d_['traj'].update(evaluable=ev, unmatched_mean=float(np.mean(um)), draws_with_unmatched=int(sum(u > 0 for u in um)))
                if not ev:                                                                    # an unmatched draw cannot support a matched reading
                    d_['traj'].update(k2=None, k3=None)
                use = d_[hierarchy(ev)]; d_['estimand'] = 'trajectory' if hierarchy(ev) == 'traj' else 'decision states'
                d_.update(k2=use['k2'], k3=use['k3'], k2_stats=use['k2_stats'], k3_stats=use['k3_stats'], null_by_day_mean=use['null_by_day_mean'])
        del V
    p = R[K37]; k1d = np.array(p['oracle_by_day']['imp']) / np.maximum(np.array(p['keep_loss_by_day']), 1e-9)
    K1 = k1(np.array(p['oracle_by_day']['imp']), np.array(p['keep_loss_by_day']))
    prim = D[K37][DECLARED]; K2 = prim['k2']; K3 = prim['k3']; holds = {k: D[k][DECLARED]['k2'] for k in D}
    o0 = np.array(R[K0]['oracle_by_day']['imp']); o0at37 = np.array(p['oracle_0ms_decisions']['imp']); o37 = np.array(p['oracle_by_day']['imp'])
    s0 = np.array(D[K0][DECLARED]['policy_by_day']['imp']); s37 = np.array(prim['policy_by_day']['imp']); pend0 = float(np.sum(R[K0]['oracle_pending_by_day']['imp']))
    X_ = p['keep_loss']; dl = dict(posting_window=X_ - pend0, pending_to_live=pend0 - float(o0.sum()), cancel_latency=float(o0.sum() - o0at37.sum()),
                                   avoidable_at_37ms_same_decisions=float(o0at37.sum()))
    C.require(abs(sum(dl.values()) - X_) <= 1e-6 * max(1., abs(X_)), 'delay rows do not add up to KEEP loss')
    K4 = dict(oracle_0ms=float(o0.sum()), oracle_same_decisions_37ms=float(o0at37.sum()), oracle_gap_by_day=(o0 - o0at37).tolist(), oracle_rechosen_37ms=float(o37.sum()),
              score_0ms=float(s0.sum()), score_37ms=float(s37.sum()), score_gap_by_day=(s0 - s37).tolist(), delays=dl)
    tag = '' if prim['estimand'] == 'trajectory' else '-STATES'
    read = {'K1 ORACLE-BOUND': 'AVOIDABLE-AT-LATENCY' if K1 else 'NOT-SHOWN', 'K2 SCORE-EARNS': f'EARNS-OVER-MATCHED{tag}' if K2 else 'NOT-SHOWN',
            'K3 SCORE-RANKS-VALUE': f'RANKS-VALUE{tag}' if K3 else 'NOT-SHOWN', 'estimand': prim['estimand'],
            'trajectory': 'EVALUABLE' if prim['traj']['evaluable'] else f"NOT-EVALUABLE-UNMATCHED ({prim['traj']['draws_with_unmatched']} / {prim['draws']} draws, mean {prim['traj']['unmatched_mean']:.1f} pulls)"}
    cand = [pol for pol in POLICIES if pol != DECLARED and (D[K37][pol]['k2'] or D[K37][pol]['k3'])]
    L += ['', 'PART 1 - what is at stake (every eligible order, M50, L 37 ms): sums in c over all orders',
          f"  KEEP value {p['keep_value']:,.0f} (loss on losing orders {X_:,.0f})",
          '  arm                 | improvement | foregone G | avoided L1 | avoided L2 (severe excess) | severe events | removed shares | live time lost (s)']
    for nm, x in (('PULL-ALL (live)', p['pull_all']), ('ORACLE (live)', p['oracle']), ('ORACLE-PENDING', p['oracle_pending'])):
        L.append(f"  {nm:<19} | {x['imp']:11,.0f} | {x['G']:10,.0f} | {x['L1']:10,.0f} | {x['L2']:26,.0f} | {x['sev']:13,.0f} | {x['R']:14,.0f} | {x['lost'] / 1000:,.0f}")
    L.append('  ORACLE (EX POST, NOT DEPLOYABLE; live decisions) by latency: ' + ' | '.join(f"L {int(x)} ms {R[vkey(50., False, x)]['oracle']['imp']:,.0f} "
                                                                                       f"({R[vkey(50., False, x)]['oracle']['imp'] / max(R[vkey(50., False, x)]['keep_loss'], 1e-9):.0%})" for x in LATS))
    L.append(f"  DELAYS of KEEP's {X_:,.0f} c loss: posting window (before a +50 ms withdrawal lands, +95 ms) {dl['posting_window']:,.0f} | pending-to-live gap "
             f"(+95 to +100 ms; orders gone before +100 ms) {dl['pending_to_live']:,.0f} | cancel latency (the zero-latency oracle's decisions valued at 0 vs 37 ms) "
             f"{dl['cancel_latency']:,.0f} | avoidable at 37 ms by those decisions {dl['avoidable_at_37ms_same_decisions']:,.0f} (re-chosen at 37 ms {o37.sum():,.0f})")
    L.append('  K1 by day (live oracle avoided / KEEP loss): ' + ', '.join(f'{d} {v:.0%}' for d, v in zip(DAYS, k1d)))
    for k in (vkey(250., False, CANCEL), vkey(0., False, CANCEL), vkey(50., True, CANCEL)):
        L.append(f"  {k}: KEEP {R[k]['keep_value']:,.0f}, loss {R[k]['keep_loss']:,.0f}, ORACLE {R[k]['oracle']['imp']:,.0f}, PULL-ALL {R[k]['pull_all']['imp']:,.0f}")
    L += ['', 'PART 2 - declared causal scores against matched nulls (500 draws for the declared policy, 200 for the others). TRAJ = one pull per order, executed pulls matched',
          '  by (day, market, side, minute) and step, early picks can use up later ones (any unmatched draw -> NOT-EVALUABLE); STATES = the same counts of live decision states per',
          '  (stratum, step), each valued alone (matched by construction). The reading uses TRAJ when every draw matches, else STATES (declared before the run).',
          '  variant          policy null   | pulls (null, unmatched) | removed sh (null) | improvement | null mean | null p95 | days > null | K2  | per removed share | null p5 | days < null | K3']
    for k in D:
        for pol in POLICIES if k == K37 else (DECLARED,):
            x = D[k][pol]
            for nm in ('traj', 'states'):
                y = x[nm]; a, b = y['k2_stats'], y['k3_stats']; um = f"{y['unmatched_mean']:5,.1f}" if nm == 'traj' else '    -'
                f2 = 'n/e' if y['k2'] is None else 'yes' if y['k2'] else 'no'; f3 = 'n/e' if y['k3'] is None else 'yes' if y['k3'] else 'no'
                L.append(f"  {k:<16} {pol:>6} {nm.upper():<6} | {x['policy']['n']:6,.0f} ({y['null_mean']['n']:6,.0f}, {um}) | {x['policy']['R']:7,.0f} ({y['null_mean']['R']:7,.0f}) | "
                         f"{a['imp']:11,.0f} | {a['null_mean']:9,.0f} | {a['null_p95']:8,.0f} | {a['days_above']:11d} | {f2:3} | {b['per_share']:17.3f} | {b['null_p5']:7.3f} | {b['days_below']:11d} | {f3}")
    x = D[K37][DECLARED]
    L += ['', f'PART 3 - by day (M50, L 37 ms): improvement over KEEP = avoided L1 + L2 - foregone G; PULL-ALL and the live ORACLE on every eligible order, '
          f"{DECLARED} and the mean of its reading's null ({x['estimand']})", '  day      | KEEP loss | PULL-ALL imp (G / L1 / L2)       | ORACLE imp (L1 / L2, severe) | ' + f'{DECLARED} imp (G / L1 / L2, severe, pulls)          | null imp (pulls)']
    for i, day in enumerate(DAYS):
        a, o, y, nb = p['pull_all_by_day'], p['oracle_by_day'], x['policy_by_day'], x['null_by_day_mean']
        L.append(f"  {day} | {p['keep_loss_by_day'][i]:9,.0f} | {a['imp'][i]:8,.0f} ({a['G'][i]:6,.0f} / {a['L1'][i]:6,.0f} / {a['L2'][i]:6,.0f}) | "
                 f"{o['imp'][i]:8,.0f} ({o['L1'][i]:6,.0f} / {o['L2'][i]:6,.0f}, {o['sev'][i]:4,.0f}) | {y['imp'][i]:7,.0f} ({y['G'][i]:6,.0f} / {y['L1'][i]:6,.0f} / {y['L2'][i]:6,.0f}, "
                 f"{y['sev'][i]:4,.0f}, {y['n'][i]:5,.0f}) | {nb['imp'][i]:7,.0f} ({nb['n'][i]:5,.0f})")
    sg = np.where(E.side.values == 0, 1., -1.); ix = pd.Series(np.arange(len(E)), index=pd.MultiIndex.from_arrays([E.day.values, E.oid.values]))
    f = Fl[Fl.clock == 50.]; ri = ix.reindex(pd.MultiIndex.from_arrays([f.day.values, f.oid.values])).values; f = f[~np.isnan(ri)]; ri = ri[~np.isnan(ri)].astype(int)
    val = f.q.values * sg[ri] * (E.mark.values[ri] - E.p.values[ri]); qq = f.q.values; age = f.t.values - PLACE
    PD = T[DECLARED]; pulled = (PD >= J0) & (PD <= NSTEP) & (PD < P0['endj'])
    rem = pulled[ri] & (f.t.values >= landing(np.minimum(PD, NSTEP), CANCEL)[ri]); SP = {}
    L += ['', f'PART 4 - splits (M50, L 37 ms): KEEP fills by trade source and by age after live; what {DECLARED} removes']
    for code, nm in enumerate(SRC):
        m_ = f.src.values == code; SP[nm] = dict(keep_c=float(val[m_].sum()), keep_sh=float(qq[m_].sum()), removed_c=float(val[m_ & rem].sum()), removed_sh=float(qq[m_ & rem].sum()))
        L.append(f"  {nm:<8}: KEEP {SP[nm]['keep_c']:10,.0f} c on {SP[nm]['keep_sh']:10,.0f} sh | removed by {DECLARED} {SP[nm]['removed_c']:9,.0f} c on {SP[nm]['removed_sh']:9,.0f} sh")
    for lo, hi in ((0., YOUNG), (YOUNG, 500.), (500., np.inf)):
        m_ = (age >= lo) & (age < hi); nm = f'age_{int(lo)}_{"END" if hi == np.inf else int(hi)}'
        SP[nm] = dict(keep_c=float(val[m_].sum()), keep_sh=float(qq[m_].sum()), removed_c=float(val[m_ & rem].sum()), removed_sh=float(qq[m_ & rem].sum()))
        L.append(f"  {nm:<12}: KEEP {SP[nm]['keep_c']:10,.0f} c on {SP[nm]['keep_sh']:10,.0f} sh | removed {SP[nm]['removed_c']:9,.0f} c")
    cs = {f'{"bid" if s_ == 0 else "ask"}:{c_ or "none"}': int(n_) for (s_, c_), n_ in E.groupby(['side', 'cause']).size().items()}; SP['structural'] = cs
    L.append('  structural-cancel triggers of eligible orders (the kernel rule on the known book) by side and cause (P47: an empty ask side cancels a bid; an empty bid side never cancels an ask): '
             + ', '.join(f'{k} {v:,}' for k, v in sorted(cs.items())))
    hs = ', '.join(k for k, v in holds.items() if v) or 'none'
    L += ['', 'READINGS (declared before the run; computed)',
          f"  K1 ORACLE-BOUND (EX POST, NOT DEPLOYABLE; live decisions): {read['K1 ORACLE-BOUND']} - the oracle avoids {o37.sum():,.0f} of KEEP's {X_:,.0f} c of loss "
          f"({o37.sum() / max(X_, 1e-9):.0%}); >= 50% on {int((k1d >= .5).sum())} / {nd} days",
          f"  matched null used for K2 / K3: {read['estimand']} (trajectory comparison {read['trajectory']})",
          f"  K2 SCORE-EARNS ({DECLARED}): {read['K2 SCORE-EARNS']} at M50 / 37 ms; the predicate (each variant's own reading null) holds at: {hs}",
          f"  K3 SCORE-RANKS-VALUE ({DECLARED}): {read['K3 SCORE-RANKS-VALUE']}",
          f"  K4 LATENCY (numbers, same decisions at 0 and 37 ms): live oracle {o0.sum():,.0f} vs {o0at37.sum():,.0f} (gap by day " + ', '.join(f'{v:,.0f}' for v in o0 - o0at37)
          + f"); {DECLARED} {s0.sum():,.0f} vs {s37.sum():,.0f} (gap by day " + ', '.join(f'{v:,.0f}' for v in s0 - s37) + ')',
          f"  live decisions {J['population']['live_decisions']:,}, of which without a causal fair price {J['population']['live_decisions_fair_missing']:,} (fallback KEEP)",
          f"  multiplicity: {len(POLICIES)} score policies; passes outside {DECLARED} (candidates only, never readings): {', '.join(cand) or 'none'}"]
    J.update(R=R, D=D, K4_latency=K4, splits=SP, readings=read, k2_holds=holds, candidates=cand, seconds=time.time() - start)
    (out / 'keep_pull.txt').write_text('\n'.join(L) + '\n'); C.dump(out / 'keep_pull.json', J)
    print('\n'.join(L), flush=True)
    return read

# --- synthetic tape (selftest only): the queue place ledger's markets plus 108, a scripted adverse sweep after a fair-price drop
def _syn_108(ws):
    """108 on a 50 / 52 book: at 50 500 ms the fair price drops to 47; at 51 000 a sweep of 20 sh through the bid (E2-timed), then 50 sh at 50
    (WS-timed) 120 ms later while the book is 47 / 49 - the 50 000 bid fills 20 + 10 and loses 2 c a share at END, and its structural cross
    cancel (decided 1 050 ms after posting) lands 37 ms later, after the second fill at M50. At 150 500 two trades at one time with different
    sources fill the 150 000 bid (source: mixed). Fair price missing at 100 000-100 200 ms (the 100 000 orders: fair_missing). At 200 010 the
    best bid jumps to 54 (the 200 000 bid at 50 is 4 ticks behind: structural cancel decided at 50 ms, landing at 95 ms by the pending rule)
    and 150 sh trade at 50 at 90 ms (M50) - filled before the landing. At 250 237 a trade fills the 250 000 bid exactly at the landing time of
    a pull decided at step 3 (187 ms: not filled in that branch); the fair price drops 1.4 c at 250 050 (one step after posting). At 270 000
    the fair price drops to 47 from 270 550 but is missing exactly at 270 500 (the first trigger step) and at 274 000-274 100: the order stays
    in the score population and triggers one step later."""
    B = QL._Book(ws, 50, 52); B.book(51000, 60000, bid=47, ask=49); B.frame(200010, 54, 56); B.book(200200, 201000, bid=54, ask=56)
    for at, s, px, q in ((51000, 0, 48, 20), (51120, 0, 50, 50), (150500, 0, 50, 50), (150500, 0, 49, 50), (200140, 0, 50, 150), (250237, 0, 50, 150)):
        B.trade(float(at), s, px, q)
    return B, {(ws + 51000., 48.): 1, (ws + 51120., 50.): 0, (ws + 150500., 50.): 0, (ws + 150500., 49.): 1, (ws + 200140., 50.): 0, (ws + 250237., 50.): 2}


class Syn(QL.Syn):
    """The ledger's synthetic source plus market 108, each trade's anchor source and a fair price on the decision grid."""
    def markets(self, day):
        return [108, 102, 101, 103, 104, 105, 106, 107]

    def day(self, day):
        if day not in self.days:
            d = QL._syn_day(day); B, code108 = _syn_108(d['ws']); d['W'][108] = B.sides(108); d['codes'] = {}; d['fair'] = {}
            for m, W in d['W'].items():
                d['codes'][m] = {s: (np.array([code108[(a, p)] if m == 108 else i % 3 for i, (a, p) in enumerate(zip(W[s]['anc'], W[s]['ptl']))], np.int8),
                                     np.asarray(W[s]['anc'], float)) for s in (0, 1)}
                tg = np.asarray(W[0]['tg'], float); jk = np.asarray(W[0]['jk']); b = np.where(jk >= 0, np.asarray(W[0]['P5'], float)[jk, 0], -1.)
                a = np.where(jk >= 0, np.asarray(W[1]['P5'], float)[jk, 0], -1.); rel = tg - d['ws']
                f = np.where((b >= 0) & (a > b), (a + b) / 2. + (0. if m == 108 else .6 * np.sin(rel / 7000.)), np.nan)
                if m == 108:
                    f = np.where((rel >= 50500.) & (rel < 60000.), 47., f); f = np.where((rel >= 100000.) & (rel < 100200.), np.nan, f)
                    f = np.where((rel >= 250050.) & (rel < 260000.), 49.6, f); f = np.where((rel >= 270550.) & (rel < 280000.), 47., f)
                    f = np.where(((rel >= 270500.) & (rel < 270550.)) | ((rel >= 274000.) & (rel < 274100.)), np.nan, f)
                d['fair'][m] = (tg, f)
            self.days[day] = d
        return self.days[day]

    def fair(self, day):
        return self.day(day)['fair']

    def sources(self, data, m):
        return self.day(data['day'])['codes'][m]


def _refused(fn):
    """The refusal message of fn, or None if it ran through."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
    except (ValueError, SystemExit) as e:
        return str(e)
    return None


def _planted(name, fn):
    """Run fn with one planted error switched on -> its refusal message (it must be refused by its own control)."""
    globals()[name] = 1
    try:
        return _refused(fn)
    finally:
        globals()[name] = 0


def _predicates():
    """The reading predicates on constructed inputs (each case flips exactly one condition)."""
    bad = []; x = np.linspace(-5., 2.9, NULL_DRAWS); nul = [dict(imp=np.full(5, v / 5.)) for v in x]
    for imp, want in (([1., 1., 1., 1., -1.], True), ([2., 2., 2., -1., -1.], False), ([.5, .5, .5, .5, .4], False)):
        if k2(dict(imp=np.array(imp)), nul)[0] != want:
            bad.append(f'k2 {imp} != {want}')
    x = np.linspace(-4., 4., NULL_DRAWS); nul = [dict(X=np.full(5, v), R=np.ones(5)) for v in x]
    for X, want in (([-5.] * 5, True), ([-10., -10., -10., 1., 1.], False), ([-3.] * 5, False)):
        if k3(dict(X=np.array(X), R=np.ones(5)), nul)[0] != want:
            bad.append(f'k3 {X} != {want}')
    for orc, want in (([6., 6., 6., 6., 3.], True), ([5., 5., 5., 5., 0.], False), ([10., 10., 10., 1., 1.], False)):
        if k1(np.array(orc), np.full(5, 10.)) != want:
            bad.append(f'k1 {orc} != {want}')
    return bad


def _readings(res, pk, sk, V, dayi, nd):
    """The declared hierarchy on one policy: (estimand, K2, K3) - TRAJ when every trajectory draw matches, else STATES."""
    if hierarchy(all(u == 0 for _, u in pk)) == 'traj':
        nul = [outcome(V, PN, dayi, nd) for PN, _ in pk]; return 'trajectory', k2(res, nul)[0], k3(res, nul)[0]
    nus = [outcome_states(V, oi, oj, dayi, nd) for oi, oj in sk]; return 'decision states', k2(res, nus)[0], k3(res, nus)[0]


def _exhaustion_cases():
    """The committed-fix review's multi-step exhaustion case (3cef4993): 5 days x 20 markets x 12 orders, six leave at the second live step,
    every removed share worth -1 c, the score pulls the six short-lived at +100 ms and the rest at +150 ms. The trajectory null runs short and
    must be NOT-EVALUABLE (K2 passed there before); the STATES null matches every pull and shows no ranking. Positive controls: equal
    eligibility matches every trajectory draw (no ranking); a score that pulls exactly the losers beats both nulls."""
    bad = []; day = np.repeat(np.arange(5), 240); cell = day * 20 + np.tile(np.repeat(np.arange(20), 12), 5); short = np.tile(np.r_[np.ones(6, bool), np.zeros(6, bool)], 100)
    n = len(day); P = np.where(short, J0, J0 + 1)
    E = pd.DataFrame(dict(day=day.astype(str), oid=np.arange(n), side=np.zeros(n, int), mark=np.full(n, 49.), p=np.full(n, 50.), v=np.full(n, -1.), q=np.ones(n),
                          g_struct=np.where(short, 150., np.nan), land_struct=np.where(short, 187., np.nan), t_full=np.full(n, np.nan)))
    Fl = pd.DataFrame(dict(day=day.astype(str), oid=np.arange(n), clock=np.full(n, 50.), t=np.where(short, 170., 300.), q=np.ones(n)))
    V = variant(E, Fl, 50., False, CANCEL); res = outcome(V, P, day, 5); pk = null_picks(V['endj'], P, cell, SEED, NULL_DRAWS); sk = state_picks(V['endj'], P, cell, SEED + 100, NULL_DRAWS)
    est, f2, f3 = _readings(res, pk, sk, V, day, 5); nus = [outcome_states(V, oi, oj, day, 5) for oi, oj in sk]
    if not (res['n'].sum() == n and all(u > 0 for _, u in pk) and est == 'decision states' and not f2 and not f3 and all(x['n'].sum() == n for x in nus)):
        bad.append(f"exhaustion case: score pulls {res['n'].sum()}, unmatched draws {sum(u > 0 for _, u in pk)}, reading {est} K2 {f2} K3 {f3}")
    V2 = variant(E.assign(g_struct=np.nan, land_struct=np.nan), Fl.assign(t=300.), 50., False, CANCEL); r2 = outcome(V2, P, day, 5)
    est, f2, f3 = _readings(r2, null_picks(V2['endj'], P, cell, SEED, 200), state_picks(V2['endj'], P, cell, SEED + 100, 200), V2, day, 5)
    if (est, f2, f3) != ('trajectory', False, False):
        bad.append(f'exhaustion positive control (equal eligibility): {est} K2 {f2} K3 {f3}')
    win = np.tile(np.r_[np.ones(6, bool), np.zeros(6, bool)], 100)                    # half the orders end 1 c above their price, half 1 c below
    E3 = E.assign(g_struct=np.nan, land_struct=np.nan, mark=np.where(win, 51., 49.), v=np.where(win, 1., -1.)); V3 = variant(E3, Fl.assign(t=300.), 50., False, CANCEL)
    P3 = np.where(win, NSTEP + 1, J0); r3 = outcome(V3, P3, day, 5)
    ranks = [_readings(r3, null_picks(V3['endj'], P3, cell, SEED, 200), state_picks(V3['endj'], P3, cell, SEED + 100, 200), V3, day, 5)]
    ranks.append(('decision states', k2(r3, [outcome_states(V3, oi, oj, day, 5) for oi, oj in state_picks(V3['endj'], P3, cell, SEED + 100, 200)])[0],
                  k3(r3, [outcome_states(V3, oi, oj, day, 5) for oi, oj in state_picks(V3['endj'], P3, cell, SEED + 100, 200)])[0]))
    if ranks != [('trajectory', True, True), ('decision states', True, True)]:
        bad.append(f'ranking positive control (the score pulls exactly the losers): {ranks}')
    return bad


def _review_cases():
    """The ledger / keep-pull review's counterexamples (handoffs/20261117_ledger_keep_pull_review.md) and their positive controls."""
    bad = []; n_day = 200; n = 5 * n_day; day = np.repeat(np.arange(5), n_day)
    act = np.tile(np.r_[np.ones(n_day // 2, bool), np.zeros(n_day // 2, bool)], 5); P = np.where(act, J0, NSTEP + 1)
    V = dict(Vk=np.full(n, -1.), Qk=np.ones(n), Mv=np.repeat(np.where(act, 0., -1.)[:, None], NSTEP, 1), Mq=np.repeat(np.where(act, 0., 1.)[:, None], NSTEP, 1),
             endj=np.where(act, NSTEP + 1, 1), life=np.where(act, 4999., 70.), lat=CANCEL)
    res = outcome(V, P, day, 5); pk = null_picks(V['endj'], P, day, SEED, NULL_DRAWS); nul = [outcome(V, PN, day, 5) for PN, _ in pk]
    if not (res['n'].sum() == 500 and all(x['n'].sum() == 500 for x in nul) and all(u == 0 for _, u in pk) and not k2(res, nul)[0] and not k3(res, nul)[0]):
        bad.append(f"review null counterexample: score pulls {res['n'].sum()}, null pulls {sorted({x['n'].sum() for x in nul})}, K2 {k2(res, nul)}, K3 {k3(res, nul)[0]}")
    pk = null_picks(np.full(n, NSTEP + 1), P, day, SEED, 200)                                       # positive control: equal eligibility
    if not all(int((PN <= NSTEP).sum()) == 500 and u == 0 for PN, u in pk) or len({tuple(np.flatnonzero(PN <= NSTEP)[:5]) for PN, _ in pk}) < 2:
        bad.append('review null positive control: equal eligibility must give 500 random pulls on every draw')
    pk = null_picks(np.array([3, NSTEP + 1]), np.array([J0, 5]), np.zeros(2, int), SEED, 200)      # the risk set can run out: unmatched is reported
    um = [u for _, u in pk]
    if not (set(um) == {0, 1} and all(int((PN <= NSTEP).sum()) + u == 2 for PN, u in pk)):
        bad.append(f'unmatched null pulls not reported: {sorted(set(um))}')
    E5 = pd.DataFrame(dict(day=[str(i) for i in range(5)], oid=np.arange(5), side=np.zeros(5, int), mark=np.full(5, 49.), p=np.full(5, 50.), v=np.full(5, -30.),
                           q=np.full(5, 30.), g_struct=np.full(5, np.nan), land_struct=np.full(5, np.nan), t_full=np.full(5, 250.)))
    F5 = pd.DataFrame([dict(day=str(i), oid=i, clock=50., t=t, q=q) for i in range(5) for t, q in ((70., 20.), (250., 10.))]); dd = np.arange(5)
    got = {}
    for lat in (0., 25., 37.):
        V5 = variant(E5, F5, 50., False, lat); got[lat] = (tot(outcome(V5, oracle(V5), dd, 5))['imp'], tot(outcome(V5, oracle(V5, 1), dd, 5, 1))['imp'])
    V0 = variant(E5, F5, 50., False, 0.); V5 = variant(E5, F5, 50., False, CANCEL); o0 = outcome(V0, oracle(V0), dd, 5)['imp']; o37 = outcome(V5, oracle(V0), dd, 5)['imp']
    if got != {0.: (50., 50.), 25.: (50., 50.), 37.: (50., 50.)} or k1(o37, np.full(5, 30.)) or float((o0 - o37).sum()) != 0.:
        bad.append(f'review oracle counterexample: live / pending oracle by latency {got}, K1 {k1(o37, np.full(5, 30.))}, paired cancel-latency gap {(o0 - o37).sum()} (must be 0: the loss sits in the posting window)')
    bad += _exhaustion_cases()
    F6 = pd.DataFrame([dict(day=str(i), oid=i, clock=50., t=t, q=q) for i in range(5) for t, q in ((70., 20.), (120., 10.))])      # the cancel window holds a loss
    V0 = variant(E5, F6, 50., False, 0.); V6 = variant(E5, F6, 50., False, CANCEL); g0 = outcome(V0, oracle(V0), dd, 5)['imp']; g37 = outcome(V6, oracle(V0), dd, 5)['imp']
    if float(g0.sum()) != 50. or float(g37.sum()) != 0. or oracle(V6).tolist() != [NSTEP + 1] * 5:
        bad.append(f'paired latency positive control: the zero-latency oracle saves {g0.sum()} (50 expected), the same decisions at 37 ms {g37.sum()} (0 expected), the 37 ms oracle {oracle(V6).tolist()} (keeps)')
    return bad


def selftest():
    """Synthetic tape only: v1 run on the same tape (control 1), planted fills / pull values / structural causes / score decisions checked
    by hand, the report end to end, planted errors and known-bad inputs refused."""
    start = time.time(); bad = []; src = Syn()
    with tempfile.TemporaryDirectory(dir=str(I.K.IO)) as tmp:
        tmp = Path(tmp)
        with QL._v1_on(src, tmp):
            for d in DAYS:
                QP.day_run(d)
        with contextlib.redirect_stdout(io.StringIO()):
            for d in DAYS:
                day_run(d, src=src, v1=tmp / 'v1', out=tmp / 'kp')
        O, Fl = load(tmp / 'kp'); d0 = DAYS[0]; ws = src.day(d0)['ws']

        def order(m, s, t):
            x = O[(O.day == d0) & (O.m == m) & (O.side == s) & (O.t == ws + t)]
            C.require(len(x) == 1, f'planted order {m} {s} {t} missing'); return x.iloc[0]

        def fl(o, c):
            x = Fl[(Fl.day == d0) & (Fl.oid == o.oid) & (Fl.clock == c)]
            return [(float(a), float(b), int(e)) for a, b, e in zip(x.t, x.q, x.src)]
        b108, a108, b25, a25 = order(108, 0, 50000.), order(108, 1, 50000.), order(102, 0, 25000.), order(102, 1, 25000.)
        exp = {'108 bid fills M50': (fl(b108, 50.), [(950., 20., 1), (1070., 10., 0)]), '108 bid fills M250': (fl(b108, 250.), [(750., 20., 1), (870., 10., 0)]),
               '108 bid fills M0 (the 1 120 trade comes after the cancel landed)': (fl(b108, 0.), [(1000., 20., 1)]),
               '108 bid keep values': ((b108.v, b108.v_m250, b108.v_m0), (-60., -60., -40.)),
               '108 bid structural cancel': ((b108.g_struct, b108.land_struct, b108.cause), (1050., 1087., 'cross')),
               '108 ask no fill, no cancel': ((a108.v, a108.q, a108.cause), (0., 0., '')),
               '108 fair path': ((b108.fair_missing, b108.f00, b108.f09, b108.f10), (0, 51., 51., 47.)),
               '108 100 000 fair missing at steps 0-3 only': ((order(108, 0, 100000.).fair_missing, bool(np.isnan(order(108, 0, 100000.).f03)), order(108, 0, 100000.).f04), (4, True, 51.)),
               '108 270 000 fair missing at the trigger step and late': ((order(108, 0, 270000.).fair_missing, bool(np.isnan(order(108, 0, 270000.).f10)), order(108, 0, 270000.).f11), (3, True, 47.)),
               '108 150 000 mixed source': (fl(order(108, 0, 150000.), 50.), [(450., 30., 3)]),
               '102 25 000 bid young fill': ((fl(b25, 50.), b25.v, b25.vy, b25.v_m250), ([(150., 30., 0)], 30., 30., 0.)),
               '102 25 000 ask fills by clock': ((fl(a25, 50.), fl(a25, 0.), fl(a25, 250.)), ([(2250., 30., 0)], [(2300., 30., 0)], [(2050., 30., 0)])),
               '102 40 000 one-sided bid side (P47)': ((order(102, 0, 40000.).cause, order(102, 1, 40000.).cause), ('own_side_empty', '')),
               '102 110 000 one-sided ask side': ((order(102, 0, 110000.).cause, order(102, 1, 110000.).cause), ('cross_opp_empty', 'own_side_empty')),
               '102 refused post-only': ((order(102, 0, 175000.).status, order(102, 1, 185000.).status), ('refused_post_only', 'refused_post_only')),
               '108 200 000 pending-cancel fill': ((fl(order(108, 0, 200000.), 50.), fl(order(108, 0, 200000.), 0.)), ([(90., 30., 0)], [])),
               '108 200 000 structural cancel': ((order(108, 0, 200000.).g_struct, order(108, 0, 200000.).land_struct, order(108, 0, 200000.).cause), (50., 95., 'behind')),
               '108 250 000 fill at a landing time': (fl(order(108, 0, 250000.), 50.), [(187., 30., 2)])}
        for k, (got, want) in exp.items():
            if got != want:
                bad.append(f'{k}: {got} != {want}')
        b250, a250 = order(108, 0, 250000.), order(108, 1, 250000.); sc = [b108, a108, b250, a250, order(108, 0, 270000.), order(108, 1, 270000.), order(108, 0, 100000.)]
        T = score_steps(np.array([[o[f'f{j:02d}'] for j in range(NSTEP + 1)] for o in sc]), np.array([o.side for o in sc]), np.array([o.p for o in sc]))
        want = {k: (10, NSTEP + 1, J0 if k in ('A+0', 'A+1', 'B0.5', 'B1') else NSTEP + 1, NSTEP + 1, 11, NSTEP + 1, NSTEP + 1) for k in POLICIES}
        if {k: tuple(v.tolist()) for k, v in T.items()} != want:
            bad.append(f'planted score decisions {T}')
        E = O[O.status == 'eligible'].reset_index(drop=True); V = variant(E, Fl, 50., False, CANCEL)
        i108 = int(np.flatnonzero((E.day == d0).values & (E.oid == b108.oid).values)[0]); i25 = int(np.flatnonzero((E.day == d0).values & (E.oid == b25.oid).values)[0])
        i250 = int(np.flatnonzero((E.day == d0).values & (E.oid == b250.oid).values)[0])
        mv = V['Mv'][i108]; want = np.r_[np.zeros(18), -40., -40., np.full(NSTEP - 20, -60.)]
        if not (np.allclose(mv, want) and V['endj'][i108] == 21 and V['Vk'][i108] == -60. and np.allclose(V['Mv'][i25][:4], [0., 0., 30., 30.])
                and V['endj'][i25] == 5 and np.allclose(V['Mv'][i250][:5], [0., 0., 0., 30., 30.])):
            bad.append(f"planted pull values: 108 {mv[:22].tolist()} endj {V['endj'][i108]}; 102 {V['Mv'][i25][:4].tolist()} endj {V['endj'][i25]}; 250 {V['Mv'][i250][:5].tolist()}")
        V100 = variant(E, Fl, 50., False, 100.); Vy = variant(E, Fl, 50., True, CANCEL)
        if not (np.allclose(V100['Mv'][i108], np.r_[np.zeros(17), -40., -40., np.full(NSTEP - 19, -60.)]) and Vy['Vk'][i25] == 0. and Vy['Vk'][i108] == -60.):
            bad.append(f"planted pull values at L 100 ms {V100['Mv'][i108][:21].tolist()} / without young fills {Vy['Vk'][i25]}, {Vy['Vk'][i108]}")
        for o_, i_ in ((b108, i108), (b25, i25), (b250, i250)):                # every pull value against the kernel itself (M50, L 37 ms)
            pth = QP.build_path(src.day(d0)['W'][int(o_.m)], o_.t, o_.t + H)[0]; ko = np.tile(np.asarray(QP.order(int(o_.side), 0, o_.p), float), (NSTEP, 1))
            ko[:, 6] = landing(np.arange(1, NSTEP + 1)); kv = np.r_[QP.evaluate(pth, ko[:72], H)[0], QP.evaluate(pth, ko[72:], H)[0]]
            if not np.allclose(kv, V['Mv'][i_], rtol=0, atol=1e-7):
                bad.append(f'pull values differ from the kernel: {o_.m} {o_.t - ws}')
        for pj, e in ((10, (60., 0., 30., 30., 1., 30.)), (18, (60., 0., 30., 30., 1., 30.)), (19, (20., 0., 20., 0., 0., 10.))):
            P = np.full(len(E), NSTEP + 1); P[i108] = pj; r = tot(outcome(V, P, np.zeros(len(E), int), 1))
            if (r['imp'], r['G'], r['L1'], r['L2'], r['sev'], r['R']) != e:
                bad.append(f'planted decomposition at step {pj}: {r}')
        Pa = np.full(len(E), NSTEP + 1); Pa[i25] = J0; r = tot(outcome(V, Pa, np.zeros(len(E), int), 1))
        if (r['imp'], r['G'], r['L1']) != (-30., 30., 0.):
            bad.append(f'planted foregone gain {r}')
        P1 = np.full(len(E), NSTEP + 1); P1[i108] = 1; z = np.zeros(len(E), int)
        if tot(outcome(V, P1, z, 1))['n'] != 0. or (tot(outcome(V, P1, z, 1, 1))['n'], tot(outcome(V, P1, z, 1, 1))['imp']) != (1., 60.):
            bad.append('the +50 ms pending withdrawal counted as a live decision, or not valued on its own row')
        if oracle(V)[i108] != J0 or oracle(V, 1)[i108] != 1:
            bad.append('the live oracle does not pull the 108 bid at the first live step (or the pending oracle at the pending step)')
        with contextlib.redirect_stdout(io.StringIO()):
            read = report(tmp / 'kp')
        J = json.loads((tmp / 'kp' / 'keep_pull.json').read_text())
        if (set(read) != {'K1 ORACLE-BOUND', 'K2 SCORE-EARNS', 'K3 SCORE-RANKS-VALUE', 'estimand', 'trajectory'} or set(J['D']) != {'M50_L37', 'M250_L37', 'M0_L37', 'M50_noyoung_L37', 'M50_L0'}
                or 'K4_latency' not in J or J['population']['eligible'] != len(E)):
            bad.append(f'report readings / variants / population {read} {sorted(J["D"])}')
        bad += _predicates() + _review_cases()
        f50 = Fl[Fl.clock == 50.].merge(E[['day', 'oid', 'side', 'mark', 'p']].assign(ri=np.arange(len(E))), on=['day', 'oid'])     # the cancel-latency row from the fills
        lose = (V['Vk'] < 0) & (V['endj'] > J0); w = f50[lose[f50.ri.values] & (f50.t.values >= STEP * J0) & (f50.t.values < STEP * J0 + CANCEL)]
        want_cl = -float((w.q * np.where(w.side == 0, 1., -1.) * (w.mark - w.p)).sum())
        if abs(J['K4_latency']['delays']['cancel_latency'] - want_cl) > 1e-6:
            bad.append(f"cancel-latency row {J['K4_latency']['delays']['cancel_latency']} != the losing orders' fills in [+100, +137) ms {want_cl}")
        for k_, x_ in J['D'].items():                                                   # the report applies the declared hierarchy
            y_ = x_[DECLARED]
            if y_['estimand'] != ('trajectory' if y_['traj']['evaluable'] else 'decision states') or (not y_['traj']['evaluable'] and y_['traj']['k2'] is not None):
                bad.append(f"{k_}: reading against {y_['estimand']} with the trajectory comparison evaluable = {y_['traj']['evaluable']} (k2 {y_['traj']['k2']})")
        # planted errors: each must be refused
        pl = dict(drop_fill=(_planted('MUT_DROP_FILL', lambda: day_run(d0, 2, src, tmp / 'v1', tmp / 'p1')), 'per-trade fills do not sum to the keep branch'),
                  no_latency=(_planted('MUT_NO_LATENCY', lambda: day_run(d0, 2, src, tmp / 'v1', tmp / 'p2')), 'a fill at or after the structural cancel landed'),
                  lookahead=(_planted('MUT_LOOKAHEAD', lambda: day_run(d0, 3, src, tmp / 'v1', tmp / 'p3')), 'a decision changed when only later fair inputs changed'),
                  null_cross=(_planted('MUT_NULL_CROSS', lambda: report(tmp / 'kp')), 'a null pull left its stratum'))
        clean = _refused(lambda: day_run(d0, 2, src, tmp / 'v1', tmp / 'p0')) is None
        A0 = pd.read_parquet(tmp / 'v1' / d0 / 'rows.parquet'); rp = tmp / 'v1' / d0 / 'receipt.json'; r0 = rp.read_text()
        A = A0.copy(); A.loc[A.arm == 'REST1', 'v'] += 1e-6; A.to_parquet(tmp / 'v1' / d0 / 'rows.parquet', index=False)       # only the hash can see this
        kb = dict(v1_rows_changed=(_refused(lambda: day_run(d0, 2, src, tmp / 'v1', tmp / 'k1')), 'v1 rows changed since its receipt'))
        A = A0.copy(); A.loc[(A.arm == 'JOIN') & (A.q > 0), 'v'] += 1e-6; A.to_parquet(tmp / 'v1' / d0 / 'rows.parquet', index=False)
        rj = json.loads(r0); rj['outputs']['rows.parquet'] = C.sha(tmp / 'v1' / d0 / 'rows.parquet'); rp.write_text(json.dumps(rj))
        kb['v1_value_moved_hash_updated'] = (_refused(lambda: day_run(d0, 2, src, tmp / 'v1', tmp / 'k2')), 'KEEP differs from v1')   # only the value comparison
        kb.update(day_output_exists=(_refused(lambda: day_run(d0, None, src, tmp / 'v1', tmp / 'kp')), 'day output exists'),
                  undeclared_day=(_refused(lambda: day_run('20261028', 2, src, tmp / 'v1', tmp / 'k3')), 'undeclared day'))
        msg = {k: v[0] for k, v in {**pl, **kb}.items()}; pl = {k: bool(m and w in m) for k, (m, w) in pl.items()}; kb = {k: bool(m and w in m) for k, (m, w) in kb.items()}
        if not (clean and all(pl.values()) and all(kb.values())):
            bad.append(f'clean smoke passes {clean}; planted errors refused by their own control {pl}; known-bad inputs refused by their own control {kb}; messages {msg}')
        st = {d: json.loads((tmp / 'kp' / d / 'receipt.json').read_text())['stat'] for d in DAYS}
    if bad:
        sys.exit('KEEP-PULL SELFTEST FAILED -> no output:\n  ' + '\n  '.join(bad))
    print('keep-pull selftests passed (synthetic tape only; v1 reproduced on it bit for bit; planted fills, latency, structural causes, scores, '
          'decomposition and oracle as scripted; planted errors and known-bad inputs each refused by its own control): '
          + json.dumps(dict(orders=int(len(O)), eligible=int(len(E)), fills=int(len(Fl)), status=O.status.value_counts().to_dict(),
                            kernel_checks=sum(s['kernel_checks'] for s in st.values()), invariance_checked=sum(s.get('invariance_checked', 0) for s in st.values()),
                            planted=pl, known_bad=kb, readings=read, seconds=round(time.time() - start))), flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[:1] == ['selftest']:
        selftest()
    elif a[:1] == ['day'] and len(a) in (2, 3):
        selftest(); day_run(a[1], int(a[2]) if len(a) == 3 else None)
    elif a == ['report']:
        report()
    else:
        sys.exit('usage: pf_keep_pull.py selftest | day <day> [limit] | report')
