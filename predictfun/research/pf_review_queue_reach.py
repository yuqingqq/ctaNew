"""Review diagnostic D-F (2026-11-10, strategy review): which trades reach a touch order that rests through a hold band.

Descriptive only; same consumed cohort as D-B..D-E (220 windows of 2026-11-08, cut 23:00Z; one partial Sunday ->
point estimates, no intervals). Reads derived cohort outputs only.

D-D / D-E valued whole-market FLOW while a two-line state is on (pro rata over every maker at the touch). This replays
ONE 5-share touch order per side under the replay engine's clock and fill rule, to see what such an order is reached by.

Single-lane replay (fixed before the table was seen):
 decisions  on the guard's 50 ms grid in [ws, ws+60 s). Book as known: kf = accumulate-max(max(t_ms+13.2,
            recv_ns/1e6-shift)+d_dec) (pf_clock_rl, clock_rl_d2.json). Order live +40 ms; cancel effective +22 ms (sent
            while the create is pending: max(t+22, t_send+95)); own fill known +100 ms. As run_forward_v3.sh.
 order      state ON and v > enter line and no order -> join the known best on that side, 5 sh (JOIN, not MICRO).
            Post-only at activation (refusal = status). Queue ahead = shown size at activation, raised to the next
            frame's if larger and <= 100 ms later (QACT max).
 keep/pull  state OFF or known book invalid -> cancel. Known best != our price: v > enter -> re-price; in the hold
            band -> stay if the best is exactly 1 tick better than ours, else cancel (>= 2 ticks behind, or alone in
            front). In the hold band: no new order, no top-up, no re-price.
 arms       v3 (exit = enter), buffer B3 (0.01 c, 100 ms), d=0.08, exit PM50 (H2), exit -0.10 c (H1), never.
 queue      M = pf_step4 fill model v2 (frame cap + trades at our price + a sweep through our level clears it);
            L = M without the frame cap (cancels ahead never credited); U = queue ahead 0 (front of the line).
 trades     every maker leg (legs_timed2 + legs_tk), one time per taker order: earliest WS-timed leg, else earliest
            E2 estimate, minus SHIFT in {0, 50, 250} ms (sensitivity to the unknown match-to-report lag; not fitted).
            No WS and no E2 time -> status 'untimed' (excluded, counted). Primary config: M, shift 50.
 value      per fill: sg*(mid(anchor+h) - price), h = 1, 5, 60 s (unshifted anchor in every config); settlement.
Not modelled: Binance / book guards, CAP10 / RO60, the 25-sh size-up, the rung, the MICRO price target, the throttle.
The fill rule itself is unvalidated until B1a (live confirmed 77.9% of replay fills on orders > 500 ms old, 2.3% < 50 ms).

Readings DECLARED BEFORE THE TABLE WAS SEEN (evaluated in code):
 R11 (H2 and H1 vs v3; M at shifts 0 / 50 / 250): RESIDENCE-PAYS if every delta USD/day at 1 s and at 5 s > 0;
     RESIDENCE-COSTS if every one < 0; else MIXED.
 R12 (H1, M, shift 50, fills on orders >= 500 ms old; n/a under 50 fills): SWEEP-FED if through-share >= 50% and
     c/sh at 1 s < 0; QUEUE-FED if through-share < 50% and c/sh at 1 s >= 0; else MIXED.
 R13 QUEUE-DECISIVE if H1's c/sh at 1 s has opposite signs under L and U (shift 50); else QUEUE-ROBUST.
 R14 (fills while in the hold band, H2 and H1, M, shift 50; n/a under 50 fills in either): HOLD-FILLS-POSITIVE if
     c/sh at 1 s and at 5 s > 0 in both; HOLD-FILLS-ADVERSE if both < 0 in both; else MIXED.

Self-test (falsifier; any failure -> no table): synthetic books with known answers through the same sim() (queue
consumption, frame cap, through-sweep, cancel latency, activation, hold-band rules, post-only refusal); unsorted frames
or trades must be refused; v3's book target against known cases and a scalar transcription of the engine's lines; five
planted errors (cancel 30 ms, place 10 ms, own fill known 400 ms, size 4, MICRO sizes swapped) must each fail at least
one synthetic case. On real data, two controls against the touch table (pf_live_vs_replay, the live bot's tape):
 (i)  identity: touch trades matched to a leg (same market, side, price, shares; times within 150 ms) must number
      >= 5,000, carry the same 1 s value in >= 80% of rows, and differ by <= 0.05 c/sh on the share-weighted mean
      (a sign error or a 1 s clock error breaks this);
 (ii) the 1 s value of gate-on at-best timed legs (state read 200 ms before) must match the touch table's gate-on
      value (state read 250 ms before its receipt time) within 0.15 c/sh.
 Control history: the first run of (ii) used WS-timed legs only and FAILED (+0.731 vs +0.422, 2026-11-10T07:47Z).
 Cause: legs the research tape could not time from the WS (E2-estimated) are the adverse ones (matched touch rows:
 -0.498 c/sh at 1 s against +0.054 for WS-timed), so WS-timed legs are a benign subset, not the touch table's
 population. (ii) now uses every timed leg, as the replay itself does; (i) was added so sign and clock are checked row by row.

Post hoc (added after the declared table was seen, 2026-11-10T07:50Z; NOT a declared reading, a cut for the next
test's design): (a) the primary config's fills split by the queue ahead of the order when it went live; (b) per arm
against v3, the net added shares and the value of the difference per added share ((S_a*m_a - S_v3*m_v3) / (S_a - S_v3)),
whose negative is the per-share maker rebate at which the arm equals v3; and how many of the 18 cells (d=0.08 / H2 / H1 x
1 s / 5 s x shifts 0 / 50 / 250, rule M) beat v3 if every filled share earned 0.20 or 0.25 c (the formula rate of the
25% taker-fee rebate at 40 c / 50 c; UNVERIFIED for this account, not in any P&L here).

Robustness configuration V (added 2026-11-10T08:24Z, after the JOIN table and both post hoc cuts were seen and BEFORE
any V table): the side's price is v3's book target instead of the known best: the MICRO target, G4 (target behind the
touch -> no quote) and G3 (target at the touch with the next level >= 2 ticks behind, or absent -> no quote),
transcribed from the `desired` closure of pf_step4_ladder.py (not importable; the MICRO lines agree with pmmmrust
pf-strategy ladder.rs). At a 1-tick spread this quotes only the side whose best level is at least as large as the other
side's; in a wider spread the target can be inside it. A withheld side cancels the resting order in every arm, in the
band too (the band extends the guard line only). Same arms, clocks, queue rules and trade shifts (configs vM 0 / 50 /
250, vL 50, vU 50). Readings R11v, R13v, R14v: defined as R11, R13, R14, on the V configs. Still not modelled: the
Binance guards (G1, G2), CAP10 / RO60, the size-up, the rung, the throttle. The transcription is checked against
hand-computed cases only, not against the engine's own decisions (they are not stored per decision).

Usage: predictfun/research/run_light.sh review_queue_reach -- python3 predictfun/research/pf_review_queue_reach.py <out.json>
"""
import bisect
import glob
import json
import math
import sys

import numpy as np
import pandas as pd

from pf_review_gate_bands import capped
from pf_review_gate_hysteresis import OUT, RP

S1 = f'{RP}/research/step1/20261108'
INF = float('inf')
PF_MIN = 13.2                                             # pf_clock_rl.PF_MIN
L_PLACE, L_CANCEL, L_PENDCANCEL, L_FILL = 40.0, 22.0, 95.0, 100.0   # run_forward_v3.sh
SIZE = 5.0
CFG = [('M', 0.0), ('M', 50.0), ('M', 250.0), ('L', 50.0), ('U', 50.0),
       ('vM', 0.0), ('vM', 50.0), ('vM', 250.0), ('vL', 50.0), ('vU', 50.0)]   # v = v3's book target (MICRO + G4 + G3)
MICRO_SWAP = 0.0                                          # planted-error switch for the self-test only
PRIM = ('M', 50.0)
AGES = [('< 100 ms', 0.0, 100.0), ('100-500 ms', 100.0, 500.0), ('0.5-2 s', 500.0, 2000.0), ('>= 2 s', 2000.0, INF)]


def wmean(x, w):
    k = np.isfinite(x) & (w > 0)
    return float((x[k] * w[k]).sum() / w[k].sum()) if k.any() else float('nan')


def v3_targets(b1, a1, bs, as_, b2, a2):
    """v3's book rules on known frames: MICRO target, G4 (target behind the touch -> none), G3 (target at the touch and
    the next level >= 2 ticks behind, or absent -> none). Returns bid / ask target ticks; -1 = no quote."""
    if MICRO_SWAP:
        bs, as_ = as_, bs
    b1f = b1.astype(float); a1f = a1.astype(float)
    micro = (b1f * as_ + a1f * bs) / np.maximum(bs + as_, 1e-12); half = (a1f - b1f) / 2.0
    qb = np.floor(micro - half + 1e-9); qa = np.ceil(micro + half - 1e-9)
    okb = (qb >= 1) & (qb < a1f) & (qb >= b1f) & ~((qb == b1f) & ((b2 < 0) | (b1 - b2 >= 2)))
    oka = (qa > b1f) & (qa <= 99) & (qa <= a1f) & ~((qa == a1f) & ((a2 < 0) | (a2 - a1 >= 2)))
    return np.where(okb, qb, -1).astype('int64'), np.where(oka, qa, -1).astype('int64')


def sim(side, tf, opp, P5, S5, cache, tau, v, s, enter, best, tt, tp, tq, qrule, t_end):
    """One 5-share touch order on one side of one window.
    Returns fills (trade index, price, shares, through, order age ms, phase) and orders (live from, gone at, queue
    ahead at activation, refused)."""
    for a in (tf, tau, tt):
        if len(a) > 1 and (np.diff(a) < 0).any():
            raise ValueError('REFUSED: frames, grid or trades not time-sorted')
    tfl = tf.tolist(); nf = len(tfl); ttl = tt.tolist(); nt = len(ttl); taul = tau.tolist()
    fills = []; orders = []; live = []; cur = None; it = 0

    def lv(pt):                                           # shown size at pt per frame; nan beyond the 5 levels shown
        if pt not in cache:
            last = P5[:, 4]
            cache[pt] = np.where((last >= 0) & ((pt < last) if side == 0 else (pt > last)), np.nan, (S5 * (P5 == pt)).sum(1))
        return cache[pt]

    def place(pt, t):
        ta = t + L_PLACE; fi = bisect.bisect_right(tfl, ta) - 1
        o = dict(pt=pt, ts=t, ta=ta, tr=INF, tc=INF, rem=SIZE, ahead=0.0, ci=fi, tfill=INF, a0=np.nan, ref=False)
        if fi < 0 or ((0 <= opp[fi] <= pt) if side == 0 else (opp[fi] >= pt)):
            o['ref'] = True; o['tr'] = ta                 # post-only refusal: never live
        else:
            a = lv(pt)[fi]; a = float(a) if np.isfinite(a) else 0.0
            if fi + 1 < nf and tfl[fi + 1] - ta <= 100.0:
                d2 = lv(pt)[fi + 1]
                if np.isfinite(d2) and d2 > a:
                    a = float(d2)
            o['a0'] = a; o['ahead'] = 0.0 if qrule == 'U' else a
            live.append(o)
        orders.append(o)
        return o

    def cancel(o, t):
        o['tc'] = min(o['tc'], t)
        o['tr'] = min(o['tr'], max(t + L_CANCEL, o['ts'] + L_PENDCANCEL) if t < o['ta'] else t + L_CANCEL)

    def apply(i):
        t = ttl[i]; p = tp[i]; left = tq[i]
        for o in live:
            if o['rem'] <= 1e-12 or t < o['ta'] or t >= o['tr']:
                continue
            if qrule == 'M':                              # shown size at our price caps the queue ahead (frames before the trade)
                j = bisect.bisect_left(tfl, t) - 1
                if j > o['ci']:
                    seg = lv(o['pt'])[o['ci'] + 1:j + 1]; seg = seg[np.isfinite(seg)]
                    if len(seg) and seg.min() < o['ahead']:
                        o['ahead'] = float(seg.min())
                    o['ci'] = j
            thr = (p < o['pt']) if side == 0 else (p > o['pt'])
            if p == o['pt']:
                use = min(left, o['ahead']); o['ahead'] -= use; q = min(left - use, o['rem'])
            elif thr:
                q = min(left, o['rem']); o['ahead'] = 0.0
            else:
                q = 0.0
            if q > 1e-12:
                left -= q; o['rem'] -= q
                g = bisect.bisect_right(taul, t) - 1
                ph = 'pend' if t >= o['tc'] else ('above' if g >= 0 and v[g] > enter else 'hold')
                fills.append((i, o['pt'], q, thr, t - o['ta'], ph, o['a0']))
                if o['rem'] <= 1e-12:
                    o['tfill'] = t

    for k, t in enumerate(taul):
        while it < nt and ttl[it] <= t:                   # trades at t come before the decision at t
            apply(it); it += 1
        if cur is not None and ((cur['ref'] and cur['ta'] <= t) or (cur['rem'] <= 1e-12 and cur['tfill'] + L_FILL <= t)):
            cur = None                                    # refusal / own fill known
        b = best[k]; ab = v[k] > enter
        if cur is not None:
            if not s[k] or b < 0:
                kill = True
            elif b != cur['pt']:
                kill = ab or ((b - cur['pt']) if side == 0 else (cur['pt'] - b)) != 1
            else:
                kill = False
            if kill:
                cancel(cur, t); cur = None
        if cur is None and s[k] and ab and b >= 0:
            cur = place(int(b), t)
        if live and (live[0]['tr'] <= t or live[0]['rem'] <= 1e-12):
            live[:] = [o for o in live if o['rem'] > 1e-12 and o['tr'] > t]
    while it < nt and ttl[it] <= t_end:
        apply(it); it += 1
    if cur is not None:
        cancel(cur, t_end)
    while it < nt and ttl[it] <= t_end + 200.0:
        apply(it); it += 1
    return fills, [(o['ta'], min(o['tr'], o['tfill'], t_end + 200.0), o['a0'], o['ref']) for o in orders]


def _run(fr, tr, qrule, v=None, s=None, t_end=600.0, rev=False):
    """Synthetic bid-side case. fr: (t, bid, bid size, ask, ask size); tr: (t, price, shares); a frame is known at its time."""
    tf = np.array([f[0] for f in fr], float); n = len(fr)
    PB = np.full((n, 5), -1); SB = np.zeros((n, 5)); PB[:, 0] = [f[1] for f in fr]; SB[:, 0] = [f[2] for f in fr]
    bb = PB[:, 0].copy(); ba = np.array([f[3] for f in fr])
    tau = np.arange(0.0, t_end, 50.0)
    v = np.ones(len(tau)) if v is None else np.asarray(v, float); s = np.ones(len(tau), bool) if s is None else np.asarray(s, bool)
    ki = np.searchsorted(np.sort(tf), tau, side='right') - 1
    best = np.where((ki >= 0) & (bb[ki] >= 0) & (ba[ki] > bb[ki]), bb[ki], -1)
    tt = np.array([x[0] for x in tr], float)
    F, O = sim(0, tf[::-1] if rev else tf, ba, PB, SB, {}, tau, v.tolist(), s.tolist(), 0.0, best.tolist(), tt,
               [x[1] for x in tr], [float(x[2]) for x in tr], qrule, t_end)
    return [(float(tt[i]), int(pt), float(q), bool(th), ph) for i, pt, q, th, _, ph, _a in F], O


def selftest():
    """Synthetic cases with known answers; returns the list of failed case names."""
    bad = []; B0 = (-100, 50, 100, 51, 100); n = 12; hold = [1.0] + [-0.05] * (n - 1)
    fr = [B0, (250, 50, 30, 51, 100)]; tr = [(200, 50, 60), (300, 50, 32), (400, 49, 10)]
    exp = {'M': [(300.0, 50, 2.0, False), (400.0, 50, 3.0, True)], 'L': [(400.0, 50, 5.0, True)],
           'U': [(200.0, 50, 5.0, False), (400.0, 50, 5.0, True)]}
    for q in 'MLU':                                       # queue consumption, frame cap, through-sweep, re-place after a known fill
        if [x[:4] for x in _run(fr, tr, q)[0]] != exp[q]:
            bad.append(f'T1-{q}')
    r = _run([B0], [(20, 50, 1), (521, 50, 1), (523, 50, 1)], 'U', s=np.arange(n) < 10)[0]
    if r != [(521.0, 50, 1.0, False, 'pend')]:            # no fill before activation; cancel effective 22 ms after the decision
        bad.append('T2')
    r = _run([B0], [(200, 50, 5), (400, 50, 5)], 'U', v=hold)[0]
    if r != [(200.0, 50, 5.0, False, 'hold')]:            # hold band: fill kept, no new order afterwards
        bad.append('T3')
    r = _run([B0, (100, 51, 20, 52, 100)], [(150, 51, 10), (160, 50, 10)], 'U', v=hold)[0]
    if r != [(160.0, 50, 5.0, False, 'hold')]:            # hold band: best one tick better -> stay
        bad.append('T4')
    if _run([B0, (100, 52, 20, 53, 100)], [(130, 50, 10)], 'U', v=hold)[0] != []:   # hold band: two ticks behind -> cancel
        bad.append('T5a')
    r = _run([B0, (100, 52, 20, 53, 100)], [(110, 50, 1), (150, 52, 3)], 'U')[0]
    if r != [(110.0, 50, 1.0, False, 'pend'), (150.0, 52, 3.0, False, 'above')]:    # above the line: re-price
        bad.append('T5b')
    r, O = _run([B0, (20, 49, 100, 50, 100)], [(60, 50, 10), (100, 49, 2)], 'U')
    if [x[:3] for x in r] != [(100.0, 49, 2.0)] or sum(o[3] for o in O) != 1:       # post-only refusal, then the new best
        bad.append('T6')
    A = lambda *x: np.array(x)
    KN = [((50, 51, 100., 50., 49, 52), (50, -1)), ((50, 51, 50., 100., 49, 52), (-1, 51)), ((50, 51, 80., 80., 49, 52), (50, 51)),
          ((50, 52, 300., 100., 49, 53), (50, -1)), ((50, 53, 300., 20., 49, 54), (51, -1)), ((50, 51, 100., 50., 48, 52), (-1, -1)),
          ((50, 51, 50., 100., 49, -1), (-1, -1)), ((50, 51, 80., 80., 49, 53), (50, -1))]
    for (b1, a1, bs, as_, b2, a2), want in KN:           # MICRO side choice, improve inside a wide spread, G4, G3
        tb, ta = v3_targets(A(b1), A(a1), A(bs), A(as_), A(b2), A(a2))
        if (int(tb[0]), int(ta[0])) != want:
            bad.append('T7'); break
    rng = np.random.default_rng(7); n_ = 2000
    b1 = rng.integers(2, 90, n_); a1 = b1 + rng.integers(1, 4, n_); bs = rng.integers(1, 500, n_).astype(float); as_ = rng.integers(1, 500, n_).astype(float)
    b2 = b1 - rng.integers(1, 4, n_); a2 = a1 + rng.integers(1, 4, n_); a2[::17] = -1; b2[::19] = -1
    tb, ta = v3_targets(b1, a1, bs, as_, b2, a2)
    for i in range(n_):                                   # scalar transcription of pf_step4_ladder.desired + G4 + G3
        micro = (b1[i] * as_[i] + a1[i] * bs[i]) / max(bs[i] + as_[i], 1e-12); half = (a1[i] - b1[i]) / 2.0
        qb = math.floor(micro - half + 1e-9); qa = math.ceil(micro + half - 1e-9)
        q = [qb if 1 <= qb < a1[i] else None, qa if b1[i] < qa <= 99 else None]
        if q[0] is not None and (q[0] < b1[i] or (q[0] == b1[i] and (b2[i] < 0 or abs(b1[i] - b2[i]) >= 2))):
            q[0] = None
        if q[1] is not None and (q[1] > a1[i] or (q[1] == a1[i] and (a2[i] < 0 or abs(a1[i] - a2[i]) >= 2))):
            q[1] = None
        if (tb[i], ta[i]) != (-1 if q[0] is None else q[0], -1 if q[1] is None else q[1]):
            bad.append('T8'); break
    for name, kw in (('bad-frames', dict(fr=fr, tr=tr, rev=True)), ('bad-trades', dict(fr=fr, tr=tr[::-1]))):
        try:                                              # known-bad input must be refused
            _run(kw['fr'], kw['tr'], 'M', rev=kw.get('rev', False)); bad.append(name)
        except ValueError:
            pass
    return bad


MUT = (('L_CANCEL', 30.0), ('L_PLACE', 10.0), ('L_FILL', 400.0), ('SIZE', 4.0), ('MICRO_SWAP', 1.0))


def mutation_check():
    """Falsifier for the synthetic cases themselves: each planted clock / size error must fail at least one of them."""
    g = globals(); out = {}
    for name, val in MUT:
        old = g[name]; g[name] = val
        try:
            out[name] = selftest()
        finally:
            g[name] = old
    return out


def met(f, o, nws, days):
    """Per-arm summary of a fill table and an order table."""
    w = f.q.values; lv_ = o[~o.ref]; life = (lv_.gone - lv_.ta).clip(lower=0).values; sh = float(w.sum())
    return dict(orders=int(len(lv_)), refused=int(o.ref.sum()), orders_per_side_min=len(lv_) / nws,
                life_p50_ms=float(np.median(life)) if len(life) else float('nan'),
                life_p90_ms=float(np.quantile(life, 0.9)) if len(life) else float('nan'),
                rest_pct=100.0 * float(life.sum()) / (nws * 60000.0), ahead_p50=float(lv_.a0.median()) if len(lv_) else float('nan'),
                fills=int(len(f)), shares=sh, sh_per_window=sh / (nws / 2.0),
                thr_pct=100.0 * float(w[f.thr.values].sum()) / sh if sh else float('nan'),
                e2_pct=100.0 * float(w[f.e2.values].sum()) / sh if sh else float('nan'),
                mo1=wmean(f.mo1.values, w), mo5=wmean(f.mo5.values, w), mo60=wmean(f.mo60.values, w), settle=wmean(f.st.values, w),
                usd1=float(np.nansum(w * f.mo1.values)) / 100.0 / days, usd5=float(np.nansum(w * f.mo5.values)) / 100.0 / days)


def main():
    bad = selftest()
    if bad:
        sys.exit(f'SELFTEST FAILED -> no table: {bad}')
    mut = mutation_check()
    if not all(mut.values()) or selftest():
        sys.exit(f'SELFTEST FAILED -> no table: a planted error passed the synthetic cases, or the constants were not restored: {mut}')
    sc = json.load(open(f'{RP}/scope.json')); W0 = {int(w['market']): w['ws'] * 1000.0 for w in sc['windows']}
    days = len(W0) / 288.0
    LN = json.load(open(f'{RP}/research/step4/minute_lines_t60k.json'))['lines']; P90, P50 = LN['90'][0], LN['50'][0]
    ARMS = {'v3 (exit = enter)': (P90, INF), 'buffer B3 (0.01 c, 100 ms)': (P90 - 0.01, 100.0), 'd=0.08': (P90 - 0.08, INF),
            'exit PM50 (H2)': (P50, INF), 'exit -0.10 (H1)': (-0.10, INF), 'never': (-INF, INF)}
    V3, H2, H1 = 'v3 (exit = enter)', 'exit PM50 (H2)', 'exit -0.10 (H1)'
    ck = json.load(open(f'{S1}/clock_rl_d2.json')); SH, DD = ck['shift_ms'], ck['d_dec']
    reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'outcome'])
    YUP = {int(m): float(o == 'Up') for m, o in zip(reg.mid, reg.outcome) if o in ('Up', 'Down')}
    P = pd.read_parquet(f'{RP}/research/step5/20261108/preds_t60k.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
    G = {int(m): (g.tau.values.astype(float), g.v_bid.values.astype(float), g.v_ask.values.astype(float)) for m, g in P.groupby('mid')}
    del P
    cols = ['mid', 't_ms', 'recv_ns'] + [f'{s}{i}_{x}' for s in 'ba' for i in range(1, 6) for x in 'ps']
    parts = []
    for fn in sorted(glob.glob(f'{S1}/books/part_*.parquet')):
        d = pd.read_parquet(fn, columns=cols); d = d[d.mid.isin(W0)]
        rel = d.t_ms.values - d.mid.map(W0).values
        parts.append(d[(rel >= -5000.0) & (rel <= 125000.0)])
    B = pd.concat(parts, ignore_index=True); del parts
    BI = B.groupby('mid', sort=False).indices

    LT = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=['mid', 'exec_s', 'side', 'ptick', 'shares', 't_ms', 'how', 't_e2'])
    tk = pd.read_parquet(f'{S1}/legs_tk.parquet', columns=['tk']).tk.values
    if len(tk) != len(LT):
        sys.exit('REFUSED: legs_tk not aligned with legs_timed2')
    grp = np.where(tk >= 0, tk, -1 - np.arange(len(LT)))   # one time per taker order (as the engine, PF_TXKEY=tk PF_TXTIME=ws)
    ws_t = np.where(LT.how.values > 0, LT.t_ms.values.astype(float), np.nan)
    e2_t = np.where((LT.how.values == 0) & np.isfinite(LT.t_e2.values), LT.t_e2.values, np.nan)
    a_ = pd.Series(ws_t).groupby(grp, sort=False).transform('min').values
    b_ = pd.Series(e2_t).groupby(grp, sort=False).transform('min').values
    LT['anc'] = np.where(np.isfinite(a_), a_, b_); LT['e2'] = ~np.isfinite(a_) & np.isfinite(b_)
    LT = LT[LT.mid.isin(W0)].copy(); LT['ws'] = LT.mid.map(W0)
    unt = LT[~np.isfinite(LT.anc.values) & (LT.exec_s * 1000.0 >= LT.ws) & (LT.exec_s * 1000.0 < LT.ws + 60000.0)]
    LT = LT[np.isfinite(LT.anc.values)]
    in60 = LT[(LT.anc >= LT.ws) & (LT.anc < LT.ws + 60000.0)]
    stat = dict(untimed_legs=int(len(unt)), untimed_shares=float(unt.shares.sum()), legs_0_60=int(len(in60)),
                shares_0_60=float(in60.shares.sum()), e2_share_pct=100.0 * float(in60.shares[in60.e2].sum()) / float(in60.shares.sum()),
                windows=len(W0), no_frames=0, no_outcome=0)
    LT = LT[(LT.anc >= LT.ws - 2000.0) & (LT.anc <= LT.ws + 61000.0)]
    LI = LT.groupby(['mid', 'side'], sort=False).indices

    rows = []; ords = []; chk = []; nws = 0
    for m in sorted(W0, key=W0.get):
        if m not in BI or m not in G:
            stat['no_frames'] += 1; continue
        if m not in YUP:
            stat['no_outcome'] += 1; continue
        ws = W0[m]; te = ws + 60000.0
        b = B.iloc[BI[m]].sort_values('t_ms', kind='stable')
        tf = b.t_ms.values.astype(float)
        kf = np.maximum.accumulate(np.maximum(tf + PF_MIN, b.recv_ns.values / 1e6 - SH) + DD)
        bb = b.b1_p.values.astype('int64'); ba = b.a1_p.values.astype('int64')
        mff = pd.Series(np.where((bb >= 0) & (ba > bb), (ba + bb) / 2.0, np.nan)).ffill().values

        def mid_at(t):
            i = np.searchsorted(tf, t, side='right') - 1
            return np.where(i >= 0, mff[np.maximum(i, 0)], np.nan)
        lev = {0: (b[[f'b{i}_p' for i in range(1, 6)]].values.astype('int64'), b[[f'b{i}_s' for i in range(1, 6)]].values.astype(float)),
               1: (b[[f'a{i}_p' for i in range(1, 6)]].values.astype('int64'), b[[f'a{i}_s' for i in range(1, 6)]].values.astype(float))}
        tau, vb, va = G[m]; k = (tau >= ws) & (tau < te); tg = tau[k]
        ki = np.searchsorted(kf, tg, side='right') - 1; okk = (ki >= 0) & (bb[ki] >= 0) & (ba[ki] > bb[ki])
        best = {0: np.where(okk, bb[ki], -1).tolist(), 1: np.where(okk, ba[ki], -1).tolist()}
        tb_, ta_ = v3_targets(bb[ki], ba[ki], lev[0][1][ki, 0], lev[1][1][ki, 0], lev[0][0][ki, 1], lev[1][0][ki, 1])
        bestV = {0: np.where(okk, tb_, -1).tolist(), 1: np.where(okk, ta_, -1).tolist()}
        for side in (0, 1):
            nws += 1; sg = 1.0 if side == 0 else -1.0; v = (vb, va)[side]; vk = v[k].tolist()
            if (m, side) in LI:
                l = LT.iloc[LI[(m, side)]]
                o_ = np.lexsort(((-l.ptick.values if side == 0 else l.ptick.values), l.anc.values))   # time, then the maker side's best price first
                anc = l.anc.values[o_]; pt = l.ptick.values[o_].astype('int64'); sh = l.shares.values[o_].astype(float); e2 = l.e2.values[o_]
            else:
                anc = np.zeros(0); pt = np.zeros(0, 'int64'); sh = np.zeros(0); e2 = np.zeros(0, bool)
            kk = (anc >= ws) & (anc < te)                   # instrument controls: every timed leg of the first 60 s
            if kk.any():
                t_ = anc[kk]; j = np.searchsorted(tf, t_, side='left') - 1
                atb = (j >= 0) & (pt[kk] == (bb, ba)[side][np.maximum(j, 0)])
                gi = np.searchsorted(tau, t_ - 200.0, side='right') - 1
                chk.append(pd.DataFrame(dict(mid=m, side=side, pk=pt[kk], sk=np.round(sh[kk] * 1000.0).astype('int64'), sh=sh[kk], anc=t_,
                                             e2=e2[kk], atb=atb, on=(gi >= 0) & (v[np.maximum(gi, 0)] > P90), mo=sg * (mid_at(t_ + 1000.0) - pt[kk]))))
            st = {a: capped(tau, v, P90, lo, cap)[0][k].tolist() for a, (lo, cap) in ARMS.items()}
            cache = {}; ptl = pt.tolist(); shl = sh.tolist()
            for q, shf in CFG:
                tt = anc - shf
                for a in ARMS:
                    F, O = sim(side, tf, (ba, bb)[side], lev[side][0], lev[side][1], cache, tg, vk, st[a], P90,
                               (bestV if q[0] == 'v' else best)[side], tt, ptl, shl, q[-1], te)
                    if F:
                        ix = np.array([f[0] for f in F]); p_ = np.array([f[1] for f in F], float); t0 = anc[ix]
                        m1, m5, m60 = (sg * (mid_at(t0 + h) - p_) for h in (1000.0, 5000.0, 60000.0))
                        for n_, f in enumerate(F):
                            rows.append((q, shf, a, m, side, f[2], bool(f[3]), f[4], f[5], bool(e2[ix[n_]]), m1[n_], m5[n_], m60[n_],
                                         sg * (100.0 * YUP[m] - f[1]), f[6]))
                    ords.extend((q, shf, a, o[0], o[1], o[2], o[3]) for o in O)
    FD = pd.DataFrame(rows, columns=['q', 'shf', 'arm', 'mid', 'side', 'q_sh', 'thr', 'age', 'ph', 'e2', 'mo1', 'mo5', 'mo60', 'st', 'a0']).rename(columns={'q': 'rule', 'q_sh': 'q'})
    OD = pd.DataFrame(ords, columns=['rule', 'shf', 'arm', 'ta', 'gone', 'a0', 'ref'])

    # instrument check on real data (positive control for sign and clocks)
    R = pd.read_parquet(f'{OUT}/touch_trade_reasons.parquet', columns=['mid', 't_in', 'up', 'ptick', 'shares', 'status', 'mo1'])
    T = R[(R.status != 'untimed') & (R.t_in < 60)].copy(); T['mid'] = T.mid.astype('int64'); T['W'] = T.mid.map(W0) + T.t_in * 1000.0
    on = np.zeros(len(T), bool)
    for m, ix in T.groupby('mid').indices.items():
        tau, vb, va = G[m]; d = T.iloc[ix]; gi = np.searchsorted(tau, d.W.values - 250.0, side='right') - 1; gz = np.maximum(gi, 0)
        on[ix] = (gi >= 0) & (np.where(d.up.values, vb[gz], va[gz]) > P90)
    ref = wmean(T.mo1.values[on], T.shares.values[on])
    LG = pd.concat(chk, ignore_index=True); g = LG[LG.atb & LG.on]; g2 = g[~g.e2]
    mine = wmean(g.mo.values, g.sh.values); mine_ws = wmean(g2.mo.values, g2.sh.values)
    T['side'] = np.where(T.up, 0, 1); T['pk'] = T.ptick.round().astype('int64'); T['sk'] = (T.shares * 1000.0).round().astype('int64')
    mm = pd.merge_asof(T.sort_values('W'), LG.sort_values('anc')[['mid', 'side', 'pk', 'sk', 'anc', 'mo']], left_on='W', right_on='anc',
                       by=['mid', 'side', 'pk', 'sk'], direction='nearest', tolerance=150.0)
    km = (mm.anc.notna() & np.isfinite(mm.mo) & np.isfinite(mm.mo1)).values; dif = (mm.mo - mm.mo1).values
    n_id = int(km.sum()); eq = float((np.abs(dif[km]) < 1e-6).mean()) if n_id else float('nan'); dmean = wmean(dif[km], mm.shares.values[km])
    ctl = dict(identity_rows=n_id, identity_equal_share=eq, identity_mean_diff=dmean,
               identity_ok=bool(n_id >= 5000 and eq >= 0.80 and abs(dmean) <= 0.05),
               touch_table_gate_on_mo1=ref, legs_gate_on_mo1=mine, legs_gate_on_mo1_ws_only=mine_ws, legs_shares=float(g.sh.sum()), tol=0.15,
               gate_ok=bool(np.isfinite(mine) and ref > 0.2 and abs(mine - ref) <= 0.15))
    ctl['ok'] = ctl['identity_ok'] and ctl['gate_ok']
    print(f'D-F queue reach | as of {pd.Timestamp.now("UTC"):%Y-%m-%dT%H:%M:%SZ} | windows {len(W0)} ({nws} window-sides, {days:.3f} window-days) | cohort 2026-11-08, consumed')
    print(f'self-test: synthetic PASS. Control (i) identity: {n_id:,} touch trades matched to a leg within 150 ms; same 1 s value in {100 * eq:.1f}%, '
          f'share-weighted mean difference {dmean:+.4f} c/sh -> {"PASS" if ctl["identity_ok"] else "FAIL"}')
    print(f'control (ii): gate-on at-best timed legs {mine:+.3f} c/sh at 1 s on {g.sh.sum():,.0f} sh (WS-timed only {mine_ws:+.3f}) vs touch table {ref:+.3f} '
          f'(tolerance 0.15) -> {"PASS" if ctl["gate_ok"] else "FAIL"}')
    print('planted errors caught by the synthetic cases: ' + '; '.join(f'{n_} {dict(MUT)[n_]:.0f} -> {b_}' for n_, b_ in mut.items()))
    if not ctl['ok']:
        sys.exit('SELFTEST FAILED -> no table: real-data control')
    print(f'statuses: untimed legs in 0-60 s (excluded) {stat["untimed_legs"]:,} / {stat["untimed_shares"]:,.0f} sh; timed legs in 0-60 s '
          f'{stat["legs_0_60"]:,} / {stat["shares_0_60"]:,.0f} sh (E2-timed {stat["e2_share_pct"]:.1f}% of shares); windows without frames '
          f'{stat["no_frames"]}, without outcome {stat["no_outcome"]}')

    M = {}
    for (q, shf) in CFG:
        for a in ARMS:
            f = FD[(FD.rule == q) & (FD.shf == shf) & (FD.arm == a)]; o = OD[(OD.rule == q) & (OD.shf == shf) & (OD.arm == a)]
            M[(q, shf, a)] = met(f, o, nws, days)
    fmt = lambda x, n=3: 'n/a' if x is None or not np.isfinite(x) else f'{x:+.{n}f}'
    print(f'\nPrimary config (queue rule {PRIM[0]}, trade shift {PRIM[1]:.0f} ms). One 5-sh touch order per side, first 60 s.')
    print(f'{"arm":28s} {"ord/side-min":>12s} {"life p50/p90 ms":>16s} {"rest %":>7s} {"ahead p50":>9s} {"refused":>7s} {"fills":>6s} {"sh":>8s} '
          f'{"sh/win":>7s} {"thr %":>6s} {"E2 %":>5s} {"c/sh 1s":>8s} {"5s":>7s} {"60s":>7s} {"settle":>7s} {"USD/d 1s":>9s} {"5s":>8s}')
    for a in ARMS:
        x = M[PRIM + (a,)]
        print(f'{a:28s} {x["orders_per_side_min"]:12.2f} {x["life_p50_ms"]:7.0f}/{x["life_p90_ms"]:8.0f} {x["rest_pct"]:7.2f} {x["ahead_p50"]:9.1f} '
              f'{x["refused"]:7d} {x["fills"]:6d} {x["shares"]:8.1f} {x["sh_per_window"]:7.2f} {x["thr_pct"]:6.1f} {x["e2_pct"]:5.1f} {fmt(x["mo1"]):>8s} '
              f'{fmt(x["mo5"]):>7s} {fmt(x["mo60"]):>7s} {fmt(x["settle"]):>7s} {fmt(x["usd1"], 2):>9s} {fmt(x["usd5"], 2):>8s}')
    SP = {}
    fp = FD[(FD.rule == PRIM[0]) & (FD.shf == PRIM[1])]
    print('\nPrimary config, fills by order age and by phase at the fill: shares | through % | c/sh at 1 s | 5 s')
    for a in ARMS:
        f = fp[fp.arm == a]; cells = []
        for name, lo, hi in AGES:
            g = f[(f.age >= lo) & (f.age < hi)]; w = g.q.values
            SP[(a, name)] = dict(fills=int(len(g)), shares=float(w.sum()), thr_pct=100.0 * float(w[g.thr.values].sum()) / w.sum() if w.sum() else float('nan'),
                                 mo1=wmean(g.mo1.values, w), mo5=wmean(g.mo5.values, w))
        for name in ('above', 'hold', 'pend'):
            g = f[f.ph == name]; w = g.q.values
            SP[(a, name)] = dict(fills=int(len(g)), shares=float(w.sum()), thr_pct=100.0 * float(w[g.thr.values].sum()) / w.sum() if w.sum() else float('nan'),
                                 mo1=wmean(g.mo1.values, w), mo5=wmean(g.mo5.values, w))
        print(f'  {a}')
        for name in [x[0] for x in AGES] + ['above', 'hold', 'pend']:
            x = SP[(a, name)]
            lab = {'above': 'phase: above the enter line', 'hold': 'phase: hold band', 'pend': 'phase: cancel pending'}.get(name, f'age {name}')
            print(f'    {lab:30s} {x["shares"]:8.1f} sh ({x["fills"]:5d} fills) | {fmt(x["thr_pct"], 1):>6s} | {fmt(x["mo1"]):>7s} | {fmt(x["mo5"]):>7s}')
    QB = [('0-25 sh', 0.0, 25.0), ('25-100 sh', 25.0, 100.0), ('100-300 sh', 100.0, 300.0), ('> 300 sh', 300.0, INF)]; PH = {}
    print('\nPOST HOC (not declared). Primary config, fills by queue ahead when the order went live: shares (fills) | through % | c/sh 1 s | 5 s || hold-band fills: shares (fills) | c/sh 1 s | 5 s')
    for a in ARMS:
        f = fp[fp.arm == a]; print(f'  {a}')
        for name, lo, hi in QB:
            g = f[(f.a0 >= lo) & (f.a0 < hi)]; w = g.q.values; h = g[g.ph == 'hold']; wh = h.q.values
            PH[(a, name)] = dict(fills=int(len(g)), shares=float(w.sum()), thr_pct=100.0 * float(w[g.thr.values].sum()) / w.sum() if w.sum() else float('nan'),
                                 mo1=wmean(g.mo1.values, w), mo5=wmean(g.mo5.values, w), hold_fills=int(len(h)), hold_shares=float(wh.sum()),
                                 hold_mo1=wmean(h.mo1.values, wh), hold_mo5=wmean(h.mo5.values, wh))
            x = PH[(a, name)]
            print(f'    {name:12s} {x["shares"]:8.1f} ({x["fills"]:4d}) | {fmt(x["thr_pct"], 1):>6s} | {fmt(x["mo1"]):>7s} | {fmt(x["mo5"]):>7s} || '
                  f'{x["hold_shares"]:8.1f} ({x["hold_fills"]:4d}) | {fmt(x["hold_mo1"]):>7s} | {fmt(x["hold_mo5"]):>7s}')
    oq = OD[(OD.rule == PRIM[0]) & (OD.shf == PRIM[1]) & ~OD.ref]
    print('  orders by queue ahead when live (share of orders, %): ' + '; '.join(
        f'{a}: ' + ' / '.join(f'{100.0 * ((oq.a0[oq.arm == a] >= lo) & (oq.a0[oq.arm == a] < hi)).mean():.1f}' for _, lo, hi in QB) for a in (V3, H1)))
    MG = {}
    print('\nPOST HOC (not declared). Rule M, each arm against v3: net added shares | value of the difference per added share, c/sh at 1 s | 5 s  (minus this = the break-even rebate)')
    for shf in (0.0, 50.0, 250.0):
        b0 = M[('M', shf, V3)]; print(f'  shift {shf:.0f} ms')
        for a in ARMS:
            if a == V3:
                continue
            x = M[('M', shf, a)]; ds = x['shares'] - b0['shares']
            MG[(shf, a)] = dict(added_shares=ds, mo1=(x['shares'] * x['mo1'] - b0['shares'] * b0['mo1']) / ds if ds > 0 else float('nan'),
                                mo5=(x['shares'] * x['mo5'] - b0['shares'] * b0['mo5']) / ds if ds > 0 else float('nan'))
            print(f'    {a:28s} {ds:8.1f} | {fmt(MG[(shf, a)]["mo1"]):>7s} | {fmt(MG[(shf, a)]["mo5"]):>7s}')
    RB = {}
    for r_ in (0.20, 0.25):
        w_ = [(a, shf, u) for a in ('d=0.08', H2, H1) for shf in (0.0, 50.0, 250.0) for u in ('mo1', 'mo5')
              if M[('M', shf, a)]['shares'] * (M[('M', shf, a)][u] + r_) > M[('M', shf, V3)]['shares'] * (M[('M', shf, V3)][u] + r_)]
        RB[f'{r_:.2f}'] = [f'{a}|{shf:.0f}|{u}' for a, shf, u in w_]
        print(f'  with {r_:.2f} c/sh on every filled share (UNVERIFIED rebate): {len(w_)} of 18 cells beat v3' + (': ' + '; '.join(RB[f"{r_:.2f}"]) if w_ else ''))
    PV = ('vM', 50.0); SV = {}
    print(f'\nRobustness V (v3 book target: MICRO + G4 + G3; declared after the JOIN table, before this one). Queue rule M, trade shift {PV[1]:.0f} ms.')
    print(f'{"arm":28s} {"ord/side-min":>12s} {"life p50/p90 ms":>16s} {"rest %":>7s} {"ahead p50":>9s} {"refused":>7s} {"fills":>6s} {"sh":>8s} '
          f'{"sh/win":>7s} {"thr %":>6s} {"E2 %":>5s} {"c/sh 1s":>8s} {"5s":>7s} {"60s":>7s} {"settle":>7s} {"USD/d 1s":>9s} {"5s":>8s}')
    fv = FD[(FD.rule == PV[0]) & (FD.shf == PV[1])]
    for a in ARMS:
        x = M[PV + (a,)]
        print(f'{a:28s} {x["orders_per_side_min"]:12.2f} {x["life_p50_ms"]:7.0f}/{x["life_p90_ms"]:8.0f} {x["rest_pct"]:7.2f} {x["ahead_p50"]:9.1f} '
              f'{x["refused"]:7d} {x["fills"]:6d} {x["shares"]:8.1f} {x["sh_per_window"]:7.2f} {x["thr_pct"]:6.1f} {x["e2_pct"]:5.1f} {fmt(x["mo1"]):>8s} '
              f'{fmt(x["mo5"]):>7s} {fmt(x["mo60"]):>7s} {fmt(x["settle"]):>7s} {fmt(x["usd1"], 2):>9s} {fmt(x["usd5"], 2):>8s}')
    print('  V, fills by phase at the fill: shares (fills) | through % | c/sh at 1 s | 5 s')
    for a in ARMS:
        f = fv[fv.arm == a]
        for name in ('above', 'hold', 'pend'):
            g = f[f.ph == name]; w = g.q.values
            SV[(a, name)] = dict(fills=int(len(g)), shares=float(w.sum()), thr_pct=100.0 * float(w[g.thr.values].sum()) / w.sum() if w.sum() else float('nan'),
                                 mo1=wmean(g.mo1.values, w), mo5=wmean(g.mo5.values, w))
        print(f'    {a:28s} ' + ' || '.join(f'{n_}: {SV[(a, n_)]["shares"]:7.1f} ({SV[(a, n_)]["fills"]:4d}) | {fmt(SV[(a, n_)]["thr_pct"], 1):>6s} | '
                                              f'{fmt(SV[(a, n_)]["mo1"]):>7s} | {fmt(SV[(a, n_)]["mo5"]):>7s}' for n_ in ('above', 'hold', 'pend')))
    print('\nBracket (all configs): shares | through % | c/sh at 1 s | 5 s | USD/day at 1 s | 5 s')
    for (q, shf) in CFG:
        print(f'  rule {q}, shift {shf:.0f} ms')
        for a in ARMS:
            x = M[(q, shf, a)]
            print(f'    {a:28s} {x["shares"]:8.1f} | {fmt(x["thr_pct"], 1):>6s} | {fmt(x["mo1"]):>7s} | {fmt(x["mo5"]):>7s} | {fmt(x["usd1"], 2):>8s} | {fmt(x["usd5"], 2):>8s}')

    # declared readings, evaluated
    dl = [M[('M', s_, a)][u] - M[('M', s_, V3)][u] for a in (H2, H1) for s_ in (0.0, 50.0, 250.0) for u in ('usd1', 'usd5')]
    R11 = 'RESIDENCE-PAYS' if all(x > 0 for x in dl) else 'RESIDENCE-COSTS' if all(x < 0 for x in dl) else 'MIXED'
    old = fp[(fp.arm == H1) & (fp.age >= 500.0)]; wo = old.q.values
    o_thr = 100.0 * float(wo[old.thr.values].sum()) / wo.sum() if wo.sum() else float('nan'); o_mo1 = wmean(old.mo1.values, wo)
    R12 = 'n/a' if len(old) < 50 else 'SWEEP-FED' if o_thr >= 50.0 and o_mo1 < 0 else 'QUEUE-FED' if o_thr < 50.0 and o_mo1 >= 0 else 'MIXED'
    l1, u1 = M[('L', 50.0, H1)]['mo1'], M[('U', 50.0, H1)]['mo1']
    R13 = 'QUEUE-DECISIVE' if np.isfinite(l1) and np.isfinite(u1) and l1 * u1 < 0 else 'QUEUE-ROBUST'
    hh = [SP[(a, 'hold')] for a in (H2, H1)]
    R14 = ('n/a' if min(x['fills'] for x in hh) < 50 else 'HOLD-FILLS-POSITIVE' if all(x['mo1'] > 0 and x['mo5'] > 0 for x in hh)
           else 'HOLD-FILLS-ADVERSE' if all(x['mo1'] < 0 and x['mo5'] < 0 for x in hh) else 'MIXED')
    brk = all(M[('L', 50.0, a)]['shares'] <= M[('M', 50.0, a)]['shares'] + 1e-9 <= M[('U', 50.0, a)]['shares'] + 2e-9 for a in ARMS)
    print(f'\nR11 = {R11}  (delta USD/day vs v3, H2 then H1, shifts 0/50/250, 1 s and 5 s: ' + ', '.join(f'{x:+.2f}' for x in dl) + ')')
    print(f'R12 = {R12}  (H1 fills on orders >= 500 ms old: {len(old)} fills, {wo.sum():.1f} sh, through {o_thr:.1f}%, {o_mo1:+.3f} c/sh at 1 s)')
    print(f'R13 = {R13}  (H1 c/sh at 1 s: rule L {l1:+.3f}, rule U {u1:+.3f})')
    print(f'R14 = {R14}  (hold-band fills: H2 {hh[0]["fills"]} fills {fmt(hh[0]["mo1"])} / {fmt(hh[0]["mo5"])}; H1 {hh[1]["fills"]} fills {fmt(hh[1]["mo1"])} / {fmt(hh[1]["mo5"])})')
    print(f'bracket ordered (filled shares L <= M <= U at shift 50, every arm): {brk}')
    dv = [M[('vM', s_, a)][u] - M[('vM', s_, V3)][u] for a in (H2, H1) for s_ in (0.0, 50.0, 250.0) for u in ('usd1', 'usd5')]
    R11v = 'RESIDENCE-PAYS' if all(x > 0 for x in dv) else 'RESIDENCE-COSTS' if all(x < 0 for x in dv) else 'MIXED'
    lv1, uv1 = M[('vL', 50.0, H1)]['mo1'], M[('vU', 50.0, H1)]['mo1']
    R13v = 'QUEUE-DECISIVE' if np.isfinite(lv1) and np.isfinite(uv1) and lv1 * uv1 < 0 else 'QUEUE-ROBUST'
    hv = [SV[(a, 'hold')] for a in (H2, H1)]
    R14v = ('n/a' if min(x['fills'] for x in hv) < 50 else 'HOLD-FILLS-POSITIVE' if all(x['mo1'] > 0 and x['mo5'] > 0 for x in hv)
            else 'HOLD-FILLS-ADVERSE' if all(x['mo1'] < 0 and x['mo5'] < 0 for x in hv) else 'MIXED')
    print(f'R11v = {R11v}  (V configs; delta USD/day vs v3, H2 then H1, shifts 0/50/250, 1 s and 5 s: ' + ', '.join(f'{x:+.2f}' for x in dv) + ')')
    print(f'R13v = {R13v}  (V configs; H1 c/sh at 1 s: rule L {lv1:+.3f}, rule U {uv1:+.3f})')
    print(f'R14v = {R14v}  (V configs; hold-band fills: H2 {hv[0]["fills"]} fills {fmt(hv[0]["mo1"])} / {fmt(hv[0]["mo5"])}; H1 {hv[1]["fills"]} fills {fmt(hv[1]["mo1"])} / {fmt(hv[1]["mo5"])})')
    json.dump(dict(as_of=f'{pd.Timestamp.now("UTC"):%Y-%m-%dT%H:%M:%SZ}', windows=len(W0), window_sides=nws, days=days, lines=dict(P90=P90, P50=P50),
                   control=ctl, planted_errors=mut, statuses=stat, arms={a: [str(lo), str(cap)] for a, (lo, cap) in ARMS.items()},
                   metrics={f'{q}|{shf:.0f}|{a}': v_ for (q, shf, a), v_ in M.items()}, splits={f'{a}|{n_}': v_ for (a, n_), v_ in SP.items()}, post_hoc_queue_ahead={f'{a}|{n_}': v_ for (a, n_), v_ in PH.items()}, post_hoc_marginal={f'{shf:.0f}|{a}': v_ for (shf, a), v_ in MG.items()}, post_hoc_rebate_cells=RB,
                   readings=dict(R11=R11, R11_deltas=dl, R12=R12, R12_fills=int(len(old)), R12_through_pct=o_thr, R12_mo1=o_mo1, R13=R13,
                                 R13_L=l1, R13_U=u1, R14=R14, bracket_ordered=brk,
                                 R11v=R11v, R11v_deltas=dv, R13v=R13v, R13v_L=lv1, R13v_U=uv1, R14v=R14v),
                   splits_v={f'{a}|{n_}': v_ for (a, n_), v_ in SV.items()}), open(sys.argv[1], 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
