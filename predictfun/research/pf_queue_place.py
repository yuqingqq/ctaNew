"""QUEUE-PLACE (2026-11-13, session ce68ebe0; USER: "How do we become front in the queue, joining existing early doesnot make much sense,
as we will also be behind them, or can you check if we join early, then replace to be back in the queue, or is it because we are already
back from when we join and lose the position from the beginning"). BEST-GAP (3db520ce) found the gap at the best is queue place: our
at-best fills are 37% trades that stop at our level against the top 5's 80%. Where does the queue ahead of our filled orders come from?

Simulator: the guard ladder of DEEP-PULL / BEST-GAP (LDG: S4's rest lane, 30 sh kept through partial fills, rungs d = 0..5 ticks behind
the known best, every rung pulled on the guard trigger v <= -0.10 and re-placed once clear for 250 ms). sim4 below is that rest lane
copied with bookkeeping (each order's price, decision / live / cancel / fill times and why it was cancelled; per fill the queue ahead just
before the trade); control (i) pins it to pf_kp_s4.sim_arm fill for fill. 9 consumed replay days, first 60 s, fill clocks M 50 (primary) and
M 0. Population: our fills AT THE BEST (depth 0 in the frame before the trade), split by the trade's timing (C12): exact = complete WS-timed
taker order (ORDERS' map, as BEST-GAP), other = the rest; 'aged' = order >= 100 ms old at the fill.

Per at-best fill of our order o (live from ta at price pt), on the simulator's clock (trade time tt):
  run, tb    the run of frames in which pt is the side's best that contains the frame before the trade; tb = its first frame's stamp.
             Kind of its start (QR's runs): improve (a better price appeared), fallback (the level in front went away), invalid, censored
             (the run starts at the first frame: a status, not decomposed). A fill whose frame does not show pt as the best on the
             simulator's clock is a status (not_best_sim), not decomposed.
  chain      o's predecessors in the same lane at the same price, walked back while the previous order ended (our cancel decision or its
             full fill) at or after tb, or within 1 s before the next one was decided (a quick re-place); t_ch = the go-live time of the
             chain's first order (o's own ta if no chain).
  g(s)       the queue ahead at the fill trade of a ghost order live from s at pt (pf_qr_queue_race.ghost: shown size in the frame <= s,
             used up by trades at pt, cleared by a trade through it, under M capped by the shown size in the frames before each trade
             and before the fill trade); our other orders ignored.
  A          = g(ta) started at o's queue ahead at activation (with the simulator's raise to the next frame): the simulator's queue ahead
             just before the fill (control ii).
  components, summing to A:
   POST chains (t_ch >= tb: our first order in the chain went live after the level became the best):
     U     = g(tb)                                  already at the level when it became the best: no entry after the birth gets ahead
     LAT   = g(min(tb + 97, t_ch)) - g(tb)          joined within 97 ms of the birth (our median feed + order path)
     LATE  = g(t_ch) - g(min(tb + 97, t_ch))        joined after we could have been there, before our first order went live
   PRE chains (t_ch < tb: our first order was at that price before it became the best, i.e. it rested behind and the level came to it):
     PREA  = g(t_ch)                                ahead of our first order at the deeper level (there even earlier than us)
   both:
     REQ   = g(ta) - g(t_ch)                        lost by our own re-places: joined after our first order, ahead of the one that filled
     RAISE = A - g(ta)                              the simulator's raise to the next frame (<= 100 ms)
  KEPT = g(t_ch): the queue ahead at this trade had our first order been kept (its fills and the pulls' protection ignored).
  REQ by link (rung 0): each link's g(next ta) - g(previous ta), attributed to why the previous order ended (trig = the guard pull,
  filled = used up by fills, drift = the best moved > HOLD_K + d ticks away, cross, nobook, end).
  Origin class of the fill: PRE-KEPT (o itself went live before tb), PRE-REPLACED (chain started before tb, o after), POST-FIRST (no
  chain), POST-REPLACED.
The top makers (descriptive, the same population as BEST-GAP's real rows: exact complete WS-timed legs at the best, first minute): their
orders are seen only through fills. Per leg and for our exact at-best fills the same way, on the trade's WS clock: the run kind and the
run's age at the trade, and PROOF = the order (their certain order hash; our order) had a fill at that price more than 250 ms before the
run began (proven present before the level became the best; a lower bound). Ours also print the truth (t_ch < tb) beside the proof.

Readings DECLARED BEFORE THE FIRST RUN (evaluated in code; population: rung 0 (BEST-GAP's BASE), M 50, exact trades, aged fills, at the
best, decomposed; sums weighted by fill shares; day rule >= 7 of 9 days):
 R480 WHERE-LOST: S = sum(q * REQ) / sum(q * A), the share of the queue ahead at our fills caused by our own re-places. S >= 0.50 ->
      REQUOTE (we join early enough, then lose the place by re-placing); S <= 0.25 -> BEHIND-FROM-ENTRY (we are behind from the moment
      our first order goes live); else MIXED. STABLE if the per-day label equals the pooled one on >= 7 of 9 days, else UNSTABLE.
 R481 ENTRY-PART: the largest of U, LAT, LATE, PREA (pooled sums) -> CROWDED-AT-BIRTH / LATENCY / LATE-ENTRY / AHEAD-AT-DEEPER-LEVEL;
      STABLE if the same on >= 7 of 9 days.
 R482 FIRST-THERE (all six rungs, same filters): c/sh at 1 s of fills from PRE chains (our order was at the price before it became the
      best) minus POST chains: > 0 on >= 7 of 9 days -> BEING-THERE-FIRST-PAYS; < 0 on >= 7 -> DOES-NOT-PAY; else NOT-SHOWN. A day with
      < 30 fills in either class does not count as passing.
 R483 THEIRS-VS-OURS (descriptive, no label): fallback share, median run age at the fill, PROOF share; top 5, field, our rung 0 and all
      rungs (with our truth).
 Printed: the components in sh and shares; KEPT against A; REQ by why the previous order ended; origin classes with share of shares,
 c/sh and STOP share; by rung; M 0; other (E2-timed) trades; statuses.
What the outcomes mean, fixed in advance: REQUOTE = keeping our order would put us near the front (the pull costs the place; whether
keeping pays is KEEP-PLACE's question, which found the pulls needed); BEHIND-FROM-ENTRY = our first order already joins behind most of the
queue, so the fix is where / when we enter (CROWDED-AT-BIRTH: only being at the price before it becomes the best helps; LATENCY: speed;
LATE-ENTRY: enter at the birth; AHEAD-AT-DEEPER-LEVEL: others rested at that price before us).
Controls (any failure -> no output): (i) sim4 equals pf_kp_s4.sim_arm(rest, guard, offset d, 30 sh) fill for fill and order for order on
every window-side, rung and clock; (ii) A equals the simulator's queue ahead before the fill for every decomposed fill not shared with
another order of ours on the same trade (shared ones skipped, counted); (iii) the rung-0 population (M 50, exact, aged, at the best)
equals BEST-GAP's BASE fills in count, shares and c/sh (checked in the report); (iv) synthetic cases with known answers (an improve
level with crowd, latency, late and re-place parts; a fallback level reached from behind; chains) and planted errors (chain across prices,
PRE read as POST, sim4 without the offset) that must each fail a case; memory self-guard.
Limits fixed in advance: replay rule M (B1a): every queue number is the simulator's model of the queue, not an observed position; consumed
days, exploratory; the six rungs are independent orders (two rungs at one price each assume they are alone); the top makers' orders are
seen only through fills, so PROOF is a lower bound for them and their entry time is unknown; the 97 ms split is fixed (QR).
Name: first committed as pf_queue_origin.py (bad57131), which overwrote the 11-06 descriptive script of that name; that file is restored and
this study renamed QUEUE-PLACE before any run (nothing else changed).
Usage (run_light, PF_SLOT=claude): pf_queue_place.py selftest | day <day> | report; driver run_queue_place.sh
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
import pf_qr_queue_race as QR                             # noqa: E402  (ghost, runs)

DAYS9 = K.DAYS9
OUTQ = f'{K.IO}/queue_place_20261113'
ORD = f'{K.IO}/orders_20261113'
BGD = f'{K.IO}/best_gap_20261113'
OLDKEY = ['mid', 'exec_s', 'first_recv_ns', 'taker_index', 'taker_quote', 'price_executed']
CFGS = [('M', 50.0), ('M', 0.0)]
RUNGS = range(6)
SIZE = 30.0
D_OURS = 97.0; REPLACE_MS = 1000.0; PROOF_MS = 250.0; MIN_FILLS = 30
INF = float('inf'); NAN = float('nan')
KINDS = ('censored', 'improve', 'fallback', 'invalid')
COMP = ('U', 'LAT', 'LATE', 'PREA', 'REQ', 'RAISE')
LABEL481 = {'U': 'CROWDED-AT-BIRTH', 'LAT': 'LATENCY', 'LATE': 'LATE-ENTRY', 'PREA': 'AHEAD-AT-DEEPER-LEVEL'}
MUT_CHAIN_ANYPRICE = MUT_PRE_AS_POST = MUT_SIM_NOOFFSET = 0   # planted errors (self-test only)
MEM_CAP_MIB = 2048.0


# ---------------------------------------------------------------- simulator: pf_kp_s4.sim2's rest lane + bookkeeping
def sim4(side, tf, opp, P5, S5, cache, tau, v, enter, best, bopp, tt, tp, tq, qrule, t_end, aux, offset, size):
    """sim2('rest') with KP switches off, REST_OFFSET = offset, SIZE = size. Returns sim2's fills and order tuples, the order dicts
    and per fill (order id, queue ahead just before the trade, shared with an earlier own order on the trade)."""
    off = 0 if MUT_SIM_NOOFFSET else int(offset)
    for a in (tf, tau, tt):
        if len(a) > 1 and (np.diff(a) < 0).any():
            raise ValueError('REFUSED: frames, grid or trades not time-sorted')
    tfl = tf.tolist(); nf = len(tfl); ttl = tt.tolist(); nt = len(ttl); taul = tau.tolist()
    fills = []; fx = []; orders = []; live = []; cur = None; it = 0; lastpull = -INF

    def lv(pt):
        if pt not in cache:
            last = P5[:, 4]
            cache[pt] = np.where((last >= 0) & ((pt < last) if side == 0 else (pt > last)), np.nan, (S5 * (P5 == pt)).sum(1))
        return cache[pt]

    def place(pt, t):
        ta = t + K.L_PLACE; fi = bisect.bisect_right(tfl, ta) - 1
        o = dict(id=len(orders), pt=pt, ts=t, ta=ta, tr=INF, tc=INF, rem=size, ahead=0.0, ci=fi, tfill=INF, a0=np.nan, ref=False, why='')
        if fi < 0 or ((0 <= opp[fi] <= pt) if side == 0 else (opp[fi] >= pt)):
            o['ref'] = True; o['tr'] = ta
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

    def cancel(o, t, why):
        if o['tc'] == INF:
            o['why'] = why
        o['tc'] = min(o['tc'], t)
        o['tr'] = min(o['tr'], max(t + K.L_CANCEL, o['ts'] + K.L_PENDCANCEL) if t < o['ta'] else t + K.L_CANCEL)

    def apply(i):
        t = ttl[i]; p = tp[i]; left = tq[i]; full = tq[i]
        for o in live:
            if o['rem'] <= 1e-12 or t < o['ta'] or t >= o['tr']:
                continue
            if qrule == 'M':
                j = bisect.bisect_left(tfl, t) - 1
                if j > o['ci']:
                    seg = lv(o['pt'])[o['ci'] + 1:j + 1]; seg = seg[np.isfinite(seg)]
                    if len(seg) and seg.min() < o['ahead']:
                        o['ahead'] = float(seg.min())
                    o['ci'] = j
            thr = (p < o['pt']) if side == 0 else (p > o['pt'])
            ahead_b = o['ahead']; shared = left < full - 1e-12
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
                fx.append((o['id'], float(ahead_b), shared))
                if o['rem'] <= 1e-12:
                    o['tfill'] = t

    for k, t in enumerate(taul):
        while it < nt and ttl[it] <= t:
            apply(it); it += 1
        if cur is not None and ((cur['ref'] and cur['ta'] <= t) or (cur['rem'] <= 1e-12 and cur['tfill'] + K.L_FILL <= t)):
            cur = None
        b = best[k]; trig = bool(aux[k])
        if cur is not None:
            why = ''
            if trig:
                why = 'trig'
            elif b < 0:
                why = 'nobook'
            else:
                beh = (b - cur['pt']) if side == 0 else (cur['pt'] - b)
                if beh > K.HOLD_K + off:
                    why = 'drift'
                elif not ((cur['pt'] < bopp[k]) if side == 0 else (cur['pt'] > bopp[k])):
                    why = 'cross'
            if why:
                cancel(cur, t, why); cur = None
        if trig:
            lastpull = t
        if cur is None and b >= 0 and not trig and t - lastpull >= K.REST_COOL:
            if not off:
                cur = place(int(b), t)
            elif 1 <= (int(b) - off if side == 0 else int(b) + off) <= 99:
                cur = place(int(b) - off if side == 0 else int(b) + off, t)
        if live and (live[0]['tr'] <= t or live[0]['rem'] <= 1e-12):
            live[:] = [o for o in live if o['rem'] > 1e-12 and o['tr'] > t]
    while it < nt and ttl[it] <= t_end:
        apply(it); it += 1
    if cur is not None:
        cancel(cur, t_end, 'end')
    while it < nt and ttl[it] <= t_end + 200.0:
        apply(it); it += 1
    return fills, [(o['ta'], min(o['tr'], o['tfill'], t_end + 200.0), o['a0'], o['ref']) for o in orders], orders, fx


# ---------------------------------------------------------------- chains and the decomposition
def chain(O, idx, tb):
    """Walk back over o's predecessors at the same price (see the header). Returns (first index, links [(index of the earlier order,
    why it ended)])."""
    c = idx; links = []
    while c > 0:
        p_ = O[c - 1]
        if p_['ref'] or (p_['pt'] != O[c]['pt'] and not MUT_CHAIN_ANYPRICE):
            break
        end = min(p_['tc'], p_['tfill'])
        if end == INF or not (end >= tb or O[c]['ts'] - end <= REPLACE_MS):
            break
        links.append((c - 1, 'filled' if p_['tfill'] <= p_['tc'] else p_['why']))
        c -= 1
    return c, links


def decompose(o, c0, tb, i, side, qrule, tfl, lvp, ttl, tpl, tql):
    """Components at the fill trade i (see the header); returns dict incl. A, KEPT, pre."""
    te = ttl[i]
    g = lambda s, a=None: QR.ghost(o['pt'], side, s, te, qrule, tfl, lvp, ttl, tpl, tql, a_start=a, i_end=i, se=True)
    A = g(o['ta'], o['a0']); gta = g(o['ta']); tch = c0['ta']; gch = g(tch)
    pre = tch < tb and not MUT_PRE_AS_POST
    r = dict(A=A, KEPT=gch, REQ=gta - gch, RAISE=A - gta, pre=pre, tch=tch)
    if pre:
        r.update(U=0.0, LAT=0.0, LATE=0.0, PREA=gch)
    else:
        gu = g(tb); g1 = g(min(tb + D_OURS, tch))
        r.update(U=gu, LAT=g1 - gu, LATE=gch - g1, PREA=0.0)
    return r


def run_at(tfl, b0, valid, rs, kind, t, pt):
    """The run of frames with pt as the best containing the frame strictly before t: (tb, kind name) or None."""
    j = bisect.bisect_left(tfl, t) - 1
    if j < 0 or not valid[j] or b0[j] != pt:
        return None
    j0 = int(rs[j])
    return tfl[j0], ('censored' if j0 == 0 else KINDS[int(kind[j0])])


# ---------------------------------------------------------------- self-tests
def _book(fr):
    tf = np.array([f[0] for f in fr], float); n = len(fr)
    PB = np.full((n, 5), -1); SB = np.zeros((n, 5))
    for r_, f in enumerate(fr):
        for j, (p, z) in enumerate(f[1][:5]):
            PB[r_, j] = p; SB[r_, j] = z
    return tf, PB, SB, np.array([f[2] for f in fr])


def selftest():
    bad = []
    # 1. improve level: born at 100 with 40 sh, +20 at 150 (< 197), +30 at 400, our first order live at 500, cancelled at 700 (guard),
    #    +25 at 750, re-placed live at 1000, filled by a 200-sh trade at 1200 -> U 40, LAT 20, LATE 30, REQ 25, RAISE 0, A 115
    fr = [(0, [(49, 10)], 52), (100, [(50, 40), (49, 10)], 52), (150, [(50, 60), (49, 10)], 52), (400, [(50, 90), (49, 10)], 52),
          (750, [(50, 115), (49, 10)], 52)]
    tf, PB, SB, ba = _book(fr); tfl = tf.tolist(); lvp = QR.lvarr(PB, SB, 50, 0)
    O = [dict(id=0, pt=50, ts=460.0, ta=500.0, tc=700.0, tfill=INF, ref=False, why='trig', a0=90.0),
         dict(id=1, pt=50, ts=960.0, ta=1000.0, tc=INF, tfill=INF, ref=False, why='', a0=115.0)]
    b0, valid, rs, kind = QR.runs(tf, PB, ba, 0)
    tb, kd = run_at(tfl, b0, valid, rs, kind, 1200.0, 50)
    c, links = chain(O, 1, tb)
    r = decompose(O[1], O[c], tb, 0, 0, 'M', tfl, lvp, [1200.0], [50], [200.0])
    want = dict(U=40.0, LAT=20.0, LATE=30.0, PREA=0.0, REQ=25.0, RAISE=0.0, A=115.0, KEPT=90.0)
    if (tb, kd) != (100.0, 'improve') or c != 0 or links != [(0, 'trig')] or any(abs(r[k_] - v_) > 1e-9 for k_, v_ in want.items()) or r['pre']:
        bad.append(f'improve case: {tb} {kd} {c} {links} {r}')
    # 2. fallback: 51 (30 sh) in front of 50 (20 sh); our order at 50 live at 20 (20 ahead); 51 gone at 100 -> 50 is the best (35 sh);
    #    filled at 1200 -> PRE, PREA 20, REQ 0, A 20 (U would be 35)
    fr = [(0, [(51, 30), (50, 20)], 53), (100, [(50, 35)], 53)]
    tf, PB, SB, ba = _book(fr); tfl = tf.tolist(); lvp = QR.lvarr(PB, SB, 50, 0)
    O = [dict(id=0, pt=50, ts=-20.0, ta=20.0, tc=INF, tfill=INF, ref=False, why='', a0=20.0)]
    b0, valid, rs, kind = QR.runs(tf, PB, ba, 0)
    tb, kd = run_at(tfl, b0, valid, rs, kind, 1200.0, 50)
    c, links = chain(O, 0, tb)
    r = decompose(O[0], O[c], tb, 0, 0, 'M', tfl, lvp, [1200.0], [50], [100.0])
    if (tb, kd) != (100.0, 'fallback') or not r['pre'] or abs(r['PREA'] - 20.0) > 1e-9 or abs(r['A'] - 20.0) > 1e-9 or r['U'] != 0.0:
        bad.append(f'fallback case: {tb} {kd} {r}')
    # 3. chains: a predecessor at another price, a predecessor that ended long before the run, a filled predecessor
    O = [dict(id=0, pt=50, ts=0.0, ta=40.0, tc=200.0, tfill=INF, ref=False, why='trig'),
         dict(id=1, pt=51, ts=300.0, ta=340.0, tc=500.0, tfill=INF, ref=False, why='drift'),
         dict(id=2, pt=50, ts=600.0, ta=640.0, tc=INF, tfill=900.0, ref=False, why=''),
         dict(id=3, pt=50, ts=1000.0, ta=1040.0, tc=INF, tfill=INF, ref=False, why='')]
    if chain(O, 3, 700.0) != (2, [(2, 'filled')]):
        bad.append(f'chain stops at another price: {chain(O, 3, 700.0)}')
    O2 = [dict(id=0, pt=50, ts=0.0, ta=40.0, tc=200.0, tfill=INF, ref=False, why='trig'),
          dict(id=1, pt=50, ts=5000.0, ta=5040.0, tc=INF, tfill=INF, ref=False, why='')]
    if chain(O2, 1, 3000.0) != (1, []) or chain(O2, 1, 100.0) != (0, [(0, 'trig')]):
        bad.append('chain gap rule')
    # 4. sim4 equals sim2's rest lane at offset 0 and 1 on a synthetic book (and differs between them)
    fr = [(-100, [(50, 100), (49, 40)], 52), (310, [(50, 50), (49, 40)], 52), (700, [(51, 10), (50, 50), (49, 40)], 53)]
    tf, PB, SB, ba = _book(fr)
    tau = np.arange(0.0, 1500.0, 50.0); n = len(tau); best = [50] * 14 + [51] * (n - 14); bo = [52] * 14 + [53] * (n - 14)
    tt = np.array([200.0, 500.0, 600.0, 800.0, 1100.0]); tp = [49, 50, 50, 51, 49]; tq = [120.0, 5.0, 25.0, 60.0, 80.0]
    aux = [False] * n; aux[8] = True
    outs = []
    for off in (0, 1):
        s0, o0 = K.SIZE, K.REST_OFFSET
        try:
            K.SIZE = SIZE; K.REST_OFFSET = off
            F2, O2_ = K.sim2('rest', 0, tf, ba, PB, SB, {}, tau, [1.0] * n, [True] * n, -INF, best, bo, tt, tp, tq, 'M', 1500.0, aux=aux)
        finally:
            K.SIZE, K.REST_OFFSET = s0, o0
        F4, O4, _, _ = sim4(0, tf, ba, PB, SB, {}, tau, [1.0] * n, -INF, best, bo, tt, tp, tq, 'M', 1500.0, aux, off, SIZE)
        if F2 != F4 or O2_ != O4:
            bad.append(f'sim4 differs from sim2 at offset {off}: {F2} / {F4}')
        outs.append(F4)
    if outs[0] == outs[1]:
        bad.append('falsifier: offsets 0 and 1 give the same fills')
    return bad


def mutation_check():
    global MUT_CHAIN_ANYPRICE, MUT_PRE_AS_POST, MUT_SIM_NOOFFSET
    out = {}
    for nm in ('MUT_CHAIN_ANYPRICE', 'MUT_PRE_AS_POST', 'MUT_SIM_NOOFFSET'):
        globals()[nm] = 1
        try:
            out[nm] = bool(selftest())
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    b = selftest(); m = mutation_check()
    if b or not all(m.values()):
        sys.exit(f'QUEUE-PLACE SELFTEST FAILED -> no output: {b} planted errors caught {m}')
    return m


# ---------------------------------------------------------------- one day
def day_run(day):
    import pf_gap_top_makers as GP
    import pf_depth_timing as DT
    mut = all_selftests()
    T0 = time.time(); D = K.load_day(day); st = D['stat']; S1d = f'{K.IO}/step1/{day}'
    full = pd.read_parquet(f'{S1d}/legs_timed2.parquet', columns=['mid', 'signer', 'shares', 't_rel', 'side', 'ptick'] + OLDKEY[1:] + ['how', 't_ms'])
    inw = full.mid.isin(list(D['W0'])).values
    G = GP.rank_groups(full.signer.values[inw], full.shares.values[inw].astype(float), full.t_rel.values[inw].astype(float))
    MP = pd.read_parquet(f'{ORD}/map_{day}.parquet').set_index('row')
    F = full[inw].join(MP, how='left')
    F['comp'] = F.groupby(OLDKEY, sort=False).tko_ok.transform('all').values
    wsT = np.where((F.how.values > 0) & F.tko_ok.values, F.t_ms.values.astype(float), np.nan)
    F['tws'] = pd.Series(wsT, index=F.index).groupby(F.tko.values).transform('min').values
    T = F[F.comp.values & np.isfinite(F.tws.values)]; TP = T.groupby('tko').ptick.agg(list).to_dict()
    LT = D['LT']; hk = MP.hash_ok.reindex(LT.index).eq(True).values
    OH = pd.Series(LT.anc.values[hk]).groupby(MP.ohash.reindex(LT.index).values[hk]).min().to_dict()   # an order hash's first fill time
    exact = set(T.index.tolist())
    rows = []; real = []; links = []
    st.update(window_sides=0, sim_differ=0, ghost_checked=0, ghost_differ=0, ghost_shared=0, real=0, fills_all=0)
    stat_f = {}
    for w in K.window_sides(D):
        st['window_sides'] += 1; m, side, sg = w['m'], w['side'], w['sg']
        tf, mff = w['tf_mff']; tfl = tf.tolist(); P5, S5, opp = w['P5'], w['S5'], w['opp']
        b0, valid, rs, kind = QR.runs(tf, P5, opp, side)

        def mid_at(t):
            j = int(np.searchsorted(tf, t, side='right')) - 1
            return mff[j] if j >= 0 else np.nan
        if (m, side) not in D['LI']:
            continue
        l = D['LT'].iloc[D['LI'][(m, side)]]
        o_ = np.lexsort(((-l.ptick.values if side == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
        for r_ in lrow:                                   # real legs at the best (BEST-GAP's rows), with the run and the proof
            if r_ not in exact or F.tko.at[r_] not in TP:
                continue
            t = float(F.tws.at[r_]); p_ = int(F.ptick.at[r_])
            if not (w['ws'] <= t < w['ws'] + 60000.0) or GP.depth_of(tfl, P5, opp, side, t, p_) != 0:
                continue
            ru = run_at(tfl, b0, valid, rs, kind, t, p_)
            tb, kd = ru if ru else (NAN, 'none')
            ok_h = bool(F.hash_ok.at[r_]); fh = OH.get(F.ohash.at[r_], NAN) if ok_h else NAN
            g_ = G.get(int(F.signer.at[r_]), 'rest')
            real.append((int(day), 'top 5' if g_ == 'top 5' else 'field', DT.trade_type(side, p_, TP[F.tko.at[r_]]), float(F.shares.at[r_]),
                         sg * (mid_at(t + 1000.0) - p_), kd, t - tb, bool(ok_h and fh < tb - PROOF_MS), ok_h))
            st['real'] += 1
        vg = (np.asarray(w['vkl'], float) <= K.GUARD_PULL).tolist()
        cache = w.setdefault('cache', {})
        for q, shf in CFGS:
            tt = w['anc'] - shf; ttl = tt.tolist(); cfg = f'{q}{int(shf)}'
            for d in RUNGS:
                Fo, Ot, O, fx = sim4(side, tf, opp, P5, S5, cache, w['tg'], w['vkl'], -INF, w['best'], w['bopp'], tt, w['ptl'], w['shl'], q, w['te'], vg, d, SIZE)
                Fk, Ok, _ = K.sim_arm(w, q, shf, 'rest', aux=vg, offset=d, size=SIZE)
                st['sim_differ'] += int(Fk != Fo or Ok != Ot)
                st['fills_all'] += len(Fo)
                ff = {}                                   # per order: its first fill's leg anchor
                for f, x in zip(Fo, fx):
                    ff[x[0]] = min(ff.get(x[0], INF), float(w['anc'][f[0]]))
                for f, x in zip(Fo, fx):
                    i, pt, qq, thr, age, ph = f[0], int(f[1]), f[2], f[3], f[4], f[5]
                    oid, ahead_b, shared = x; o = O[oid]; r_ = int(lrow[i])
                    ex = r_ in exact and F.tko.at[r_] in TP
                    t = float(F.tws.at[r_]) if ex else float(w['anc'][i])
                    if not (w['ws'] <= t < w['ws'] + 60000.0) or GP.depth_of(tfl, P5, opp, side, t, pt) != 0:
                        continue
                    typ = 'THROUGH' if thr else (DT.trade_type(side, pt, TP[F.tko.at[r_]]) if ex else 'n/a')
                    ru = run_at(tfl, b0, valid, rs, kind, t, pt)
                    tbw, kdw = ru if ru else (NAN, 'none')
                    proof = bool(ff[oid] < tbw - PROOF_MS) if ru else False
                    base = (int(day), cfg, d, 'exact' if ex else 'other', float(qq), sg * (mid_at(t + 1000.0) - pt), age >= 100.0, ph == 'pend', typ,
                            kdw, t - tbw, proof, float(age))
                    rs_ = run_at(tfl, b0, valid, rs, kind, ttl[i], pt)
                    stt = 'not_best_sim' if rs_ is None else ('censored' if rs_[1] == 'censored' else 'ok')
                    stat_f[f'{cfg}|{stt}'] = stat_f.get(f'{cfg}|{stt}', 0) + 1
                    if stt != 'ok':
                        rows.append(base + (stt, False, False, 0) + (NAN,) * 9 + ('',)); continue
                    tb = rs_[0]
                    c0, lk = chain(O, o['id'], tb); lvp = cache.get(pt)
                    if lvp is None:
                        lvp = QR.lvarr(P5, S5, pt, side); cache[pt] = lvp
                    dc = decompose(o, O[c0], tb, i, side, q, tfl, lvp, ttl, w['ptl'], w['shl'])
                    if shared:
                        st['ghost_shared'] += 1
                    else:
                        st['ghost_checked'] += 1; st['ghost_differ'] += int(abs(dc['A'] - ahead_b) > 1e-9)
                    rows.append(base + (stt, dc['pre'], o['ta'] < tb, len(lk)) + tuple(dc[k_] for k_ in COMP) + (dc['A'], dc['KEPT'], dc['tch'] - tb) +
                                (lk[0][1] if lk else '',))
                    if d == 0 and lk:                     # REQ by link, attributed to why the earlier order ended
                        gp = lambda s: QR.ghost(pt, side, s, ttl[i], q, tfl, lvp, ttl, w['ptl'], w['shl'], i_end=i, se=True)
                        nxt = o['ta']
                        for ci, why in lk:
                            g_hi = gp(nxt); g_lo = gp(O[ci]['ta'])
                            links.append((int(day), cfg, 'exact' if ex else 'other', age >= 100.0, why, float(qq), g_hi - g_lo))
                            nxt = O[ci]['ta']
        w.pop('cache', None)
        if st['window_sides'] % 80 == 0:
            r2 = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r2:.0f} MiB', flush=True)
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    if st['sim_differ'] or st['ghost_differ'] or not st['ghost_checked']:
        sys.exit(f'QUEUE-PLACE CONTROL FAILED on {day} -> no result: {st}')
    st['status_fills'] = stat_f; st['planted_caught'] = mut
    os.makedirs(OUTQ, exist_ok=True)
    cols = ['day', 'cfg', 'rung', 'src', 'q', 'v1', 'aged', 'pend', 'type', 'kind_ws', 'age_ws', 'proof', 'oage', 'status', 'pre', 'opre', 'nch'] + \
        list(COMP) + ['A', 'KEPT', 'delay', 'why_last']
    pd.DataFrame(rows, columns=cols).to_parquet(f'{OUTQ}/fills_{day}.parquet', index=False)
    pd.DataFrame(real, columns=['day', 'grp', 'type', 'q', 'v1', 'kind_ws', 'age_ws', 'proof', 'hash_ok']).to_parquet(f'{OUTQ}/real_{day}.parquet', index=False)
    pd.DataFrame(links, columns=['day', 'cfg', 'src', 'aged', 'why', 'q', 'req']).to_parquet(f'{OUTQ}/links_{day}.parquet', index=False)
    json.dump(dict(stat=st), open(f'{OUTQ}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: {st} in {time.time() - T0:.0f} s', flush=True)


# ---------------------------------------------------------------- report
def report():
    A = pd.concat([pd.read_parquet(f'{OUTQ}/fills_{d}.parquet') for d in DAYS9], ignore_index=True)
    R_ = pd.concat([pd.read_parquet(f'{OUTQ}/real_{d}.parquet') for d in DAYS9], ignore_index=True)
    LK = pd.concat([pd.read_parquet(f'{OUTQ}/links_{d}.parquet') for d in DAYS9], ignore_index=True)
    stats = {d: json.load(open(f'{OUTQ}/day_{d}.json'))['stat'] for d in DAYS9}
    A = A[np.isfinite(A.v1)]; R_ = R_[np.isfinite(R_.v1)]
    cps = lambda x: float((x.q * x.v1).sum() / x.q.sum()) if x.q.sum() else NAN
    wsum = lambda x, c: float((x.q * x[c]).sum())
    # control iii: rung 0, M 50, exact, aged, at the best = BEST-GAP's BASE
    bg = pd.concat([pd.read_parquet(f'{BGD}/fills_{d}.parquet') for d in DAYS9], ignore_index=True)
    bg = bg[(bg.arm == 'BASE') & (bg.cfg == 'M50') & bg.aged & np.isfinite(bg.v1)]
    r0 = A[(A.rung == 0) & (A.cfg == 'M50') & (A.src == 'exact') & A.aged]
    c3 = dict(n=(len(r0), len(bg)), q=(float(r0.q.sum()), float(bg.q.sum())), cps=(cps(r0), cps(bg)))
    if len(r0) != len(bg) or abs(r0.q.sum() - bg.q.sum()) > 1e-6 or abs(cps(r0) - cps(bg)) > 1e-9:
        sys.exit(f'QUEUE-PLACE REPORT CONTROL (iii) FAILED -> no table: {c3}')
    P = r0[r0.status == 'ok']
    ok = A[A.status == 'ok']; idg = np.where(ok.pre, ok.PREA, ok.U + ok.LAT + ok.LATE) + ok.REQ + ok.RAISE - ok.A
    if not np.isfinite(ok[list(COMP) + ['A', 'KEPT']].values).all() or np.abs(idg.values).max() > 1e-6 or (ok.pre & ((ok.U != 0) | (ok.LAT != 0) | (ok.LATE != 0))).any():
        sys.exit('QUEUE-PLACE REPORT CONTROL FAILED -> no table: components not finite or not summing to A')
    rb = pd.concat([pd.read_parquet(f'{BGD}/fills_{d}.parquet') for d in DAYS9], ignore_index=True); rb = rb[(rb.arm == 'real') & np.isfinite(rb.v1)]
    for g_ in ('top 5', 'field'):
        x_, y_ = R_[R_.grp == g_], rb[rb.who == g_]
        if len(x_) != len(y_) or abs(x_.q.sum() - y_.q.sum()) > 1e-6 or abs(cps(x_) - cps(y_)) > 1e-9:
            sys.exit(f'QUEUE-PLACE REPORT CONTROL (iii, real {g_}) FAILED -> no table: {len(x_)} {len(y_)} {cps(x_)} {cps(y_)}')
        c3[f'real {g_}'] = (len(x_), float(x_.q.sum()), cps(x_))

    def lab480(x):
        s = wsum(x, 'REQ') / wsum(x, 'A') if wsum(x, 'A') else NAN
        return s, ('REQUOTE' if s >= 0.5 else ('BEHIND-FROM-ENTRY' if s <= 0.25 else 'MIXED'))

    def lab481(x):
        v = {c: wsum(x, c) for c in ('U', 'LAT', 'LATE', 'PREA')}
        return v, LABEL481[max(v, key=v.get)]
    s480, l480 = lab480(P); per480 = {d: lab480(P[P.day == int(d)]) for d in DAYS9}
    n480 = sum(per480[d][1] == l480 for d in DAYS9)
    v481, l481 = lab481(P); per481 = {d: lab481(P[P.day == int(d)])[1] for d in DAYS9}; n481 = sum(per481[d] == l481 for d in DAYS9)
    R = dict(control_iii=c3, R480=dict(share_req=s480, label=l480, stable=f'{"STABLE" if n480 >= 7 else "UNSTABLE"} ({n480} of 9)',
                                     per_day={d: per480[d][0] for d in DAYS9}),
             R481=dict(sums=v481, label=l481, stable=f'{"STABLE" if n481 >= 7 else "UNSTABLE"} ({n481} of 9)', per_day=per481))
    allr = A[(A.cfg == 'M50') & (A.src == 'exact') & A.aged & (A.status == 'ok')]
    pos = neg = 0; per482 = {}
    for d in DAYS9:
        x = allr[allr.day == int(d)]; a_, b_ = x[x.pre], x[~x.pre]
        df = cps(a_) - cps(b_) if len(a_) and len(b_) else NAN; per482[d] = dict(diff=df, n_pre=len(a_), n_post=len(b_))
        if len(a_) >= MIN_FILLS and len(b_) >= MIN_FILLS:
            pos += int(df > 0); neg += int(df < 0)
    R['R482'] = dict(pre=cps(allr[allr.pre]), post=cps(allr[~allr.pre]), label='BEING-THERE-FIRST-PAYS' if pos >= 7 else ('DOES-NOT-PAY' if neg >= 7 else 'NOT-SHOWN'),
                     days_pos=pos, days_neg=neg, per_day=per482)
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    L = [f'QUEUE-PLACE, 9 consumed replay days, first 60 s, guard ladder (30 sh, rungs 0-5); as of {asof}; controls passed',
         f'control iii: rung 0 M 50 exact aged at the best = BEST-GAP BASE: {len(r0)} fills, {r0.q.sum():.0f} sh, {cps(r0):+.3f} c/sh', '',
         'statuses (at-best fills, all rungs): ' + json.dumps({k_: int(sum(stats[d].get('status_fills', {}).get(k_, 0) for d in DAYS9))
                                                              for k_ in sorted({k for d in DAYS9 for k in stats[d].get('status_fills', {})})}),
         f'ghost control: {sum(stats[d]["ghost_checked"] for d in DAYS9)} checked, {sum(stats[d]["ghost_shared"] for d in DAYS9)} shared (skipped), 0 differ;'
         f' sim4 = sim_arm on {sum(stats[d]["window_sides"] for d in DAYS9)} window-sides x 6 rungs x 2 clocks', '']

    def comp_table(x, nm):
        a = wsum(x, 'A')
        segs = ' | '.join(f'{c} {wsum(x, c) / x.q.sum():6.1f} ({wsum(x, c) / a:4.0%})' for c in COMP) if a else ''
        return f'  {nm:<34} fills {len(x):>7} sh {x.q.sum():>9.0f} | A {a / max(x.q.sum(), 1e-9):6.1f} sh | {segs} | KEPT {wsum(x, "KEPT") / max(x.q.sum(), 1e-9):6.1f}'
    L.append('queue ahead at the fill (share-weighted mean sh per filled share; share of A): U at birth | LAT <= 97 ms | LATE | PREA ahead at'
             ' the deeper level | REQ our re-places | RAISE; KEPT = had our first order been kept')
    for nm, x in (('rung 0 M50 exact aged (primary)', P), ('rung 0 M0 exact aged', A[(A.rung == 0) & (A.cfg == 'M0') & (A.src == 'exact') & A.aged & (A.status == 'ok')]),
                  ('rung 0 M50 other (E2) aged', A[(A.rung == 0) & (A.cfg == 'M50') & (A.src == 'other') & A.aged & (A.status == 'ok')]),
                  ('rungs 1-5 M50 exact aged', allr[allr.rung > 0]), ('all rungs M50 exact aged', allr)):
        L.append(comp_table(x, nm))
    L += ['', f'R480 WHERE-LOST: REQ share {s480:.1%} -> {l480}, {R["R480"]["stable"]}; per day ' +
          ' '.join(f'{per480[d][0]:.0%}' for d in DAYS9),
          f'R481 ENTRY-PART: {l481}, {R["R481"]["stable"]}; sums ' + ', '.join(f'{k_} {v_ / P.q.sum():.1f}' for k_, v_ in v481.items()), '']
    L.append('origin of our at-best fills (M 50, exact, aged): share of shares | c/sh | STOP share | mean A | mean KEPT')
    for nm, x in (('rung 0', P), ('rungs 1-5', allr[allr.rung > 0]), ('all rungs', allr)):
        for cl, msk in (('PRE-KEPT', x.pre & x.opre), ('PRE-REPLACED', x.pre & ~x.opre), ('POST-FIRST', ~x.pre & (x.nch == 0)), ('POST-REPLACED', ~x.pre & (x.nch > 0))):
            y = x[msk]
            if not len(y):
                continue
            L.append(f'  {nm:<9} {cl:<14} {y.q.sum() / x.q.sum():5.1%} | {cps(y):+.3f} | {y.q[y.type == "STOP"].sum() / y.q.sum():4.0%} |'
                     f' {wsum(y, "A") / y.q.sum():6.1f} | {wsum(y, "KEPT") / y.q.sum():6.1f}')
    L.append(f'R482 FIRST-THERE (all rungs): PRE {R["R482"]["pre"]:+.3f} vs POST {R["R482"]["post"]:+.3f} c/sh -> {R["R482"]["label"]} '
             f'(+ {pos} / - {neg} of 9 qualifying days)')
    L.append('  per day PRE - POST (n PRE / n POST): ' + ' '.join(f'{per482[d]["diff"]:+.2f} ({per482[d]["n_pre"]}/{per482[d]["n_post"]})' for d in DAYS9))
    L += ['', 'entry delay of POST chains (our first order live after the birth), rung 0 M 50 exact aged, share of shares: ' +
          ' | '.join(f'{nm} {P.q[(~P.pre) & (P.delay >= lo) & (P.delay < hi)].sum() / max(P.q[~P.pre].sum(), 1e-9):.0%}'
                     for nm, lo, hi in (('< 97 ms', -INF, 97.0), ('97-500 ms', 97.0, 500.0), ('0.5-2 s', 500.0, 2000.0), ('>= 2 s', 2000.0, INF))),
          f'  median entry delay {float(np.median(P.delay[~P.pre])) if (~P.pre).any() else NAN:.0f} ms; chain length (re-places before the filling order)'
          f' share-weighted mean {wsum(P, "nch") / P.q.sum():.2f}', '']
    lk = LK[(LK.cfg == 'M50') & (LK.src == 'exact') & LK.aged]
    tot = float((lk.q * lk.req).sum())
    L.append('REQ (rung 0, M 50, exact, aged) by why the earlier order ended: ' +
             ', '.join(f'{wy} {float((g_.q * g_.req).sum()) / P.q.sum():.1f} sh ({float((g_.q * g_.req).sum()) / tot if tot else NAN:.0%})'
                       for wy, g_ in lk.groupby('why')))
    R['req_by_why'] = {wy: float((g_.q * g_.req).sum()) / P.q.sum() for wy, g_ in lk.groupby('why')}
    L += ['', 'R483 theirs vs ours (exact at-best fills, WS clock; shares): fallback / improve share | run age at the fill median (p25 / p75) |'
          ' PROOF (an earlier fill > 250 ms before the run) | truth (ours: chain before the birth) | c/sh | STOP']
    th = {}
    for nm, x, tr in (('top 5', R_[R_.grp == 'top 5'], None), ('field', R_[R_.grp == 'field'], None),
                      ('ours rung 0', A[(A.rung == 0) & (A.cfg == 'M50') & (A.src == 'exact') & A.aged], 'pre'),
                      ('ours all rungs', A[(A.cfg == 'M50') & (A.src == 'exact') & A.aged], 'pre')):
        qs = x.q.sum(); ag = x.age_ws[np.isfinite(x.age_ws)]
        th[nm] = dict(fallback=float(x.q[x.kind_ws == 'fallback'].sum() / qs), improve=float(x.q[x.kind_ws == 'improve'].sum() / qs),
                      age_med=float(np.median(ag)) if len(ag) else NAN, proof=float(x.q[x.proof].sum() / qs),
                      truth=float(x.q[x[tr].astype(bool) & (x.status == 'ok')].sum() / x.q[x.status == 'ok'].sum()) if tr else NAN, cps=cps(x), stop=float(x.q[x.type == 'STOP'].sum() / qs))
        L.append(f'  {nm:<15} {th[nm]["fallback"]:4.0%} / {th[nm]["improve"]:4.0%} | {th[nm]["age_med"]:7.0f} ms'
                 f' ({np.percentile(ag, 25) if len(ag) else NAN:.0f} / {np.percentile(ag, 75) if len(ag) else NAN:.0f}) | {th[nm]["proof"]:4.0%} |'
                 f' {th[nm]["truth"]:4.0%} | {th[nm]["cps"]:+.3f} | {th[nm]["stop"]:4.0%}')
    R['R483'] = th
    L += ['', 'by rung (M 50, exact, aged, at the best, decomposed): share of at-best shares | c/sh | STOP | mean A | PRE share']
    for d in RUNGS:
        y = allr[allr.rung == d]
        if len(y):
            L.append(f'  rung {d}: {y.q.sum() / allr.q.sum():5.1%} | {cps(y):+.3f} | {y.q[y.type == "STOP"].sum() / y.q.sum():4.0%} | '
                     f'{wsum(y, "A") / y.q.sum():6.1f} | {y.q[y.pre].sum() / y.q.sum():4.0%}')
    os.makedirs(OUTQ, exist_ok=True)
    json.dump(R, open(f'{OUTQ}/queue_place.json', 'w'), indent=1, default=float)
    open(f'{OUTQ}/queue_place.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        print(f'selftest passed; planted errors caught: {all_selftests()}')
    elif cmd == 'day' and len(sys.argv) > 2:
        day_run(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit(__doc__)
