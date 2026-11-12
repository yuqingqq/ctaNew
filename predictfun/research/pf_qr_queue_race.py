"""QR, the queue race (2026-11-12, session ce68ebe0). USER: "it seems we quote at bad timings, not sure if it is because we
requote slower than others, or we requote and we lose the position. And yes, if we lose the latency game, we need to check how
fast we need to be".

Exploratory, on consumed days: the 9 replay days (10-23..27, 10-31, 11-01..03; first 60 s of each window) and, for part C, the
live engine's orders in the 220 consumed 11-08 windows (B13c's data). Simulator = keep-or-pull S4's sim2 (pf_kp_s4), copied here
as sim3 with bookkeeping added (which order, how its queue ahead was removed) and one new lane; controls pin sim3 to sim2.

PART A, where our queue ahead comes from (replay arms J-v3, Z-guard, Z-none at M 0 / M 50 / L 50; primary J-v3 at M 50).
 For each of our orders that is at the best of its side when it goes live (frame <= live time ta, venue stamps):
  birth tb  = stamp of the first frame of the current run of frames in which this price is the best of our side. Kinds: improve
              (the frame before had a worse best), fallback (a better best: the level was second and the best went away), invalid
              (the frame before had no valid book), censored (the run starts at the first frame: a status, not decomposed).
  chain     = our earlier orders at the same price on the same side, each cancelled by us (not filled) and followed by the next one
              within 1 s of the cancel (re-places); t_chain = max(live time of the chain's first order, tb).
  g(s)      = the queue ahead, at ta, of a ghost order live from s at the same price, by the simulator's rule: the shown size in
              the frame <= s, used up by trades at our price, cleared by a trade through it; under M also capped by the smallest
              shown size in the frames before each trade and at ta. g(ta) = this order's queue ahead without the raise rule.
  components, summing to the simulator's queue ahead at activation a0:
   U     = g(tb)                                  already there at the birth: no speed gets ahead of it
   LAT   = g(min(tb + 97, t_chain)) - g(tb)       joined before our order could be there (97 ms = our median: 53 ms feed + 44 ms)
   LATE  = g(t_chain) - g(min(tb + 97, t_chain))  joined after we could have been there, before our first order went live
   REQ   = g(ta) - g(t_chain)                     lost by cancelling and re-placing within the chain
   RAISE = a0 - g(ta)                             the simulator's raise to the next frame (<= 100 ms)
 Fill classes (how the front was reached), per fill: E = queue ahead at activation < 20 sh; else T = a trade through our price;
 else A = >= 2/3 of the queue ahead was removed by the shown-size cap (others cancelling; rule M only); else Q = mostly traded
 through. Fills are also split by the level's age at activation (< 100 ms, 100-500 ms, 0.5-2 s, >= 2 s) and first / re-place.
PART B, how fast (9 replay days; M 0 / M 50 / M 250 / L 50 / U 50).
 FB(d)-x: at every frame where the best of our side improves (stamp tb in the window, both frames valid, <= 5 s apart), our order
 is live at tb + d at the new best, decided from the birth frame only, if guard x's state was on at the last grid step (x = none:
 always; g90: v3's line PM90 and state, as J-v3). An order of ours at another price is cancelled at that moment. Exits at the 50 ms
 grid on the current clock (the known best, 53 ms stale): kept while the known best is our price or one tick better than ours;
 cancelled when it is >= 2 ticks better or worse than ours (we are alone in front), the book is invalid or the state is off. Until
 the current clock has seen the birth frame only the last two apply (grace). d in {0, 15, 30, 55, 97, 150} ms: 0 = live at the
 birth stamp (a bound, not a speed: still behind the shares that made the birth), 15 = the venue's smallest feed delay (13.2 ms)
 and nothing else, 30, 55 = that feed delay with our order path (40 ms), 97 = our median, 150. B-x = the review's lane B (entry at
 the grid step on the current clock), reported with its realised delay from the birth. RJ30-x = joins at random times (as many as
 the window-side's births, one seeded draw) at the best of the frame then, live 30 ms later, with FB's exits: speed without births.
 Sensitivity (printed, no reading): FB(d) at M 50 without the raise to the next frame (the raise counts the next snapshot's
 joiners as ahead of us, which weighs most on the fastest entries); the gap from a birth frame to the next frame is printed.
PART C, the live engine (11-08, B13c's data): part A's decomposition for every accepted live order at the best when accepted
 (frame strictly before t_Accepted, as B13c), chains by B13c's re-place match within each run, queue rules L (B13c: trades only)
 and M (cap), trade times unshifted (as B13c).

Values: per fill sg * (mid(anchor + h) - price), h = 1 s / 5 s (S4's vals); $/day = sum(shares x c/sh) / 100 / days.

Readings DECLARED BEFORE THE FIRST RUN (evaluated in code; 'days' = the 9 replay days; a day with < 30 fills in a cell does not
count as passing):
 R280 CAUSE (J-v3, M 50): the largest of U, LAT, LATE, REQ in the pooled sums over decomposed orders -> CROWDED-AT-BIRTH / LATENCY /
      LATE-ENTRY / REQUOTE; STABLE if the same one is the largest on >= 7 of 9 days, else UNSTABLE. Z-guard, Z-none, M 0 and L 50
      printed.
 R281 CAUSE-LIVE (part C, rule L and rule M; one day, point estimates): the same label for each rule.
 R282 EARLY (Z-none and J-v3, each): class E's c/sh at 1 s > 0 on >= 7 of 9 days at M 0 and at M 50 -> EARLY-PAYS; else
      EARLY-DOES-NOT-PAY.
 R283 ATTRITION (Z-none, M 0 and M 50): class A's c/sh at 1 s below class Q's on >= 7 of 9 days at both -> ATTRITION-WORSE; above
      on >= 7 at both -> ATTRITION-BETTER; else MIXED.
 R284 QUEUE-BY-SPEED (FB-none, M 50): median queue ahead at activation of FB(30) <= half of FB(97)'s -> SPEED-BUYS-QUEUE; else
      LITTLE-QUEUE-FROM-SPEED.
 R285 SPEED-PAYS (FB-g90 and FB-none, each): FB(30) minus FB(97) in c/sh at 1 s > 0 on >= 7 of 9 days at M 0 and at M 50 ->
      SPEED-PAYS; < 0 on >= 7 at both -> SPEED-HURTS; else MIXED.
 R286 SPEED-NEEDED (FB-g90; FB-none printed): POS = the largest d whose pooled c/sh at 1 s is > 0 at M 0 and at M 50 and > 0 on
      >= 7 of 9 days at both (NONE if no d); V3 = the largest d whose c/sh at 1 s is above J-v3's on >= 7 of 9 days at M 0 and at
      M 50 (NONE if no d). d = 0 is a bound, not a speed. M 250 / L 50 / U 50 printed beside.
What the outcomes mean, fixed in advance: R280 LATENCY = others reach the level before our order can; REQUOTE = our own re-places
put us behind; LATE-ENTRY = we join later than our speed allows (the guard, the grid, our decisions); CROWDED-AT-BIRTH = the queue
was there when the level became the best (a level that falls back carries the queue formed while it was second; only an order
already resting there is ahead of it). R285 SPEED-PAYS with R286 POS = a speed at which joining new levels pays in this replay;
NONE = no speed in the sweep makes it pay with these exits.

Controls (any failure -> no table): (i) sim3 equals pf_kp_s4.sim2 fill for fill and order for order for J-v3, Z-guard, Z-none
(M 0 / M 50 / L 50) and lane B none / g90 (5 configs) on every window-side; (ii) FB with lane B's own entry steps as its schedule
equals lane B on every window-side and config; (iii) the regression can fail: with L_CANCEL 30 sim3 must differ from sim2 on the
first day; (iv) for every decomposed order g(ta) equals the simulator's unraised queue ahead; for every fill the removed parts plus
the queue ahead before the fill equal a0 (M, L; no cap credit under L), and the ghost carried from ta to the fill trade equals the
simulator's queue ahead there (fills where another order of ours filled first on the same trade are skipped, counted); (v)
synthetic self-tests with known answers (ghost carry and caps, birth kinds, FB births, chains, FB live time, grace, fill classes,
lane B schedule = lane B, sim3 = sim2) and planted errors (ghost without the final cap, chains across prices, no grace, L_PLACE
10) that must each fail a case. Statuses (refused, not at the best at activation, censored births, events before the first grid
step, births per window-side) are printed with every table.
Limits fixed in advance: consumed days, exploratory (a pass makes a candidate, not a result); rule M is not validated (B1a); one
5-sh order per lane, our order is not in the public book; FB speeds up the entry only (exits and the guard keep the current clock;
B21b found a faster order path barely changes exits); frames are the venue's snapshots, so the queue at tb + d has their
resolution; part C is one day.
Usage: research/run_qr.sh (selftest, the 9 days, live, report), every job through run_light.sh in the claude slot.
       python3 predictfun/research/pf_qr_queue_race.py selftest | day <yyyymmdd> | live | report
"""
import bisect
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402

INF = float('inf'); NAN = float('nan')
L_PLACE, L_CANCEL, L_PENDCANCEL, L_FILL = K.L_PLACE, K.L_CANCEL, K.L_PENDCANCEL, K.L_FILL
SIZE = K.SIZE
DAYS9 = K.DAYS9; IO = K.IO; OUTQ = f'{IO}/qr_20261112'
CFG_A = [('M', 0.0), ('M', 50.0), ('L', 50.0)]
CFG_B = [('M', 0.0), ('M', 50.0), ('M', 250.0), ('L', 50.0), ('U', 50.0)]
PRIM = ('M', 50.0)
ARMS_A = ('J-v3', 'Z-guard', 'Z-none')
D_SWEEP = (0.0, 15.0, 30.0, 55.0, 97.0, 150.0)
D_OURS = 97.0; D_FAST = 30.0
REPLACE_MS = 1000.0; E_SH = 20.0; A_SHARE = 2.0 / 3.0; BIRTH_GAP_MS = 5000.0; MIN_FILLS = 30
COMP = ('U', 'LAT', 'LATE', 'REQ', 'RAISE')
CAUSE_LABEL = {'U': 'CROWDED-AT-BIRTH', 'LAT': 'LATENCY', 'LATE': 'LATE-ENTRY', 'REQ': 'REQUOTE'}
KINDS = ('censored', 'improve', 'fallback', 'invalid')
AGEB = [('< 100 ms', 0.0, 100.0), ('100-500 ms', 100.0, 500.0), ('0.5-2 s', 500.0, 2000.0), ('>= 2 s', 2000.0, INF)]
CLS = ('E', 'T', 'A', 'Q')
MUT_GHOST_NOCAP = MUT_CHAIN_ANYPRICE = MUT_NO_GRACE = 0   # planted errors (self-test only)
MEM_CAP_MIB = 2048.0


# ---------------------------------------------------------------- simulator (pf_kp_s4.sim2 + bookkeeping + lane fbirth)
def sim3(lane, side, tf, opp, P5, S5, cache, tau, v, s, enter, best, bopp, tt, tp, tq, qrule, t_end, aux=None, sched=None, raise_ms=100.0):
    """sim2's lanes join / rest / birth (S4's hooks inert) and lane fbirth (entries from `sched` = [(decision time, price, time the
    current clock knows the birth frame)], FB's exits). Returns fills (sim2's tuples), orders (dicts), fill bookkeeping
    (order id, queue ahead before the fill, removed by caps / by trades / by a sweep before it, another own order filled first on
    the trade) and statuses."""
    if lane not in ('join', 'rest', 'birth', 'fbirth'):
        raise ValueError(f'REFUSED: unknown lane {lane}')
    for a in (tf, tau, tt):
        if len(a) > 1 and (np.diff(a) < 0).any():
            raise ValueError('REFUSED: frames, grid or trades not time-sorted')
    tfl = tf.tolist(); nf = len(tfl); ttl = tt.tolist(); nt = len(ttl); taul = tau.tolist()
    fills = []; fx = []; orders = []; live = []; cur = None; it = 0; lastpull = -INF
    pb = -1; kb = -1; pborn = -1; tprev = None; st = dict(ev_before_grid=0)

    def lv(pt):
        if pt not in cache:
            last = P5[:, 4]
            cache[pt] = np.where((last >= 0) & ((pt < last) if side == 0 else (pt > last)), np.nan, (S5 * (P5 == pt)).sum(1))
        return cache[pt]

    def place(pt, t, gr=-INF):
        ta = t + L_PLACE; fi = bisect.bisect_right(tfl, ta) - 1
        o = dict(id=len(orders), pt=pt, ts=t, ta=ta, tr=INF, tc=INF, rem=SIZE, ahead=0.0, ci=fi, tfill=INF, a0=np.nan, a0n=np.nan,
                 ref=False, capr=0.0, trdr=0.0, thrr=0.0, gr=gr)
        if fi < 0 or ((0 <= opp[fi] <= pt) if side == 0 else (opp[fi] >= pt)):
            o['ref'] = True; o['tr'] = ta
        else:
            a = lv(pt)[fi]; a = float(a) if np.isfinite(a) else 0.0; o['a0n'] = a
            if raise_ms is not None and fi + 1 < nf and tfl[fi + 1] - ta <= raise_ms:
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
        t = ttl[i]; p = tp[i]; left = tq[i]; full = tq[i]
        for o in live:
            if o['rem'] <= 1e-12 or t < o['ta'] or t >= o['tr']:
                continue
            if qrule == 'M':
                j = bisect.bisect_left(tfl, t) - 1
                if j > o['ci']:
                    seg = lv(o['pt'])[o['ci'] + 1:j + 1]; seg = seg[np.isfinite(seg)]
                    if len(seg) and seg.min() < o['ahead']:
                        o['capr'] += o['ahead'] - float(seg.min()); o['ahead'] = float(seg.min())
                    o['ci'] = j
            thr = (p < o['pt']) if side == 0 else (p > o['pt'])
            ahead_b = o['ahead']; c0, t0, h0 = o['capr'], o['trdr'], o['thrr']; shared = left < full - 1e-12
            if p == o['pt']:
                use = min(left, o['ahead']); o['ahead'] -= use; o['trdr'] += use; q = min(left - use, o['rem'])
            elif thr:
                q = min(left, o['rem']); o['thrr'] += o['ahead']; o['ahead'] = 0.0
            else:
                q = 0.0
            if q > 1e-12:
                left -= q; o['rem'] -= q
                g = bisect.bisect_right(taul, t) - 1
                ph = 'pend' if t >= o['tc'] else ('above' if g >= 0 and v[g] > enter else 'hold')
                fills.append((i, o['pt'], q, thr, t - o['ta'], ph, o['a0']))
                fx.append((o['id'], ahead_b, c0, t0, h0, shared))
                if o['rem'] <= 1e-12:
                    o['tfill'] = t

    items = [(t, 0, k) for k, t in enumerate(taul)]
    if lane == 'fbirth':
        items += [(float(e[0]), 1, j) for j, e in enumerate(sched)]
        items.sort()
    for t, typ, k in items:
        while it < nt and ttl[it] <= t:
            apply(it); it += 1
        if cur is not None and ((cur['ref'] and cur['ta'] <= t) or (cur['rem'] <= 1e-12 and cur['tfill'] + L_FILL <= t)):
            cur = None
        if typ == 1:                                      # fbirth entry
            pt_, gr_ = sched[k][1], sched[k][2]; g = bisect.bisect_right(taul, t) - 1
            if g < 0:
                st['ev_before_grid'] += 1; continue
            if bool(s[g]) and v[g] > enter:
                if cur is not None and cur['pt'] != pt_:
                    cancel(cur, t); cur = None
                if cur is None:
                    cur = place(int(pt_), t, gr_)
            continue
        b = best[k]; ab = v[k] > enter; on = bool(s[k])
        if lane == 'join':
            if cur is not None:
                if not on or b < 0:
                    kill = True
                elif b != cur['pt']:
                    kill = ab or ((b - cur['pt']) if side == 0 else (cur['pt'] - b)) != 1
                else:
                    kill = False
                if kill:
                    cancel(cur, t); cur = None
            if cur is None and on and ab and b >= 0:
                cur = place(int(b), t)
        elif lane == 'birth':
            gap = tprev is not None and t - tprev > 60.0
            imp = pb >= 0 and b >= 0 and not gap and ((b > pb) if side == 0 else (b < pb))
            if imp:
                kb = k; pborn = b
            elif b != pborn or gap:
                kb = -1
            pb = b
            trig = kb >= 0 and k - kb == 0
            if cur is not None:
                if not on or b < 0:
                    kill = True
                else:
                    d = (b - cur['pt']) if side == 0 else (cur['pt'] - b)
                    kill = (trig and ab) if d == 1 else (d != 0)
                if kill:
                    cancel(cur, t); cur = None
            if cur is None and on and ab and trig:
                cur = place(int(b), t)
        elif lane == 'rest':
            trig = bool(aux[k])
            if cur is not None:
                if trig or b < 0:
                    kill = True
                else:
                    beh = (b - cur['pt']) if side == 0 else (cur['pt'] - b)
                    kill = beh > K.HOLD_K or not ((cur['pt'] < bopp[k]) if side == 0 else (cur['pt'] > bopp[k]))
                if kill:
                    cancel(cur, t); cur = None
            if trig:
                lastpull = t
            if cur is None and b >= 0 and not trig and t - lastpull >= K.REST_COOL:
                cur = place(int(b), t)
        else:                                             # fbirth exits
            if cur is not None:
                if not on or b < 0:
                    kill = True
                elif t < cur['gr'] and not MUT_NO_GRACE:
                    kill = False                          # grace: the current clock has not seen the birth frame yet
                else:
                    d = (b - cur['pt']) if side == 0 else (cur['pt'] - b)
                    kill = d not in (0, 1)
                if kill:
                    cancel(cur, t); cur = None
        tprev = t
        if live and (live[0]['tr'] <= t or live[0]['rem'] <= 1e-12):
            live[:] = [o for o in live if o['rem'] > 1e-12 and o['tr'] > t]
    while it < nt and ttl[it] <= t_end:
        apply(it); it += 1
    if cur is not None:
        cancel(cur, t_end)
    while it < nt and ttl[it] <= t_end + 200.0:
        apply(it); it += 1
    return fills, orders, fx, st


def otuples(O, t_end):
    return [(o['ta'], min(o['tr'], o['tfill'], t_end + 200.0), o['a0'], o['ref']) for o in O]


def b_schedule(tau, best, side):
    """Lane B's own entry steps as an fbirth schedule (control ii)."""
    ev = []; pb = -1; kb = -1; pborn = -1; tprev = None
    for k, t in enumerate(tau.tolist()):
        b = best[k]; gap = tprev is not None and t - tprev > 60.0
        imp = pb >= 0 and b >= 0 and not gap and ((b > pb) if side == 0 else (b < pb))
        if imp:
            kb = k; pborn = b
        elif b != pborn or gap:
            kb = -1
        pb = b; tprev = t
        if kb >= 0 and k - kb == 0:
            ev.append((t, b, t))
    return ev


# ---------------------------------------------------------------- book runs, births, ghosts, chains
def runs(tf, P5, opp, side):
    """Per frame: best of our side, valid book, start of the current run of frames with this best, kind of that start."""
    b0 = P5[:, 0].astype('int64'); n = len(b0)
    valid = (b0 >= 0) & (opp >= 0) & ((opp > b0) if side == 0 else (b0 > opp))
    same = np.zeros(n, bool)
    if n > 1:
        same[1:] = valid[1:] & valid[:-1] & (b0[1:] == b0[:-1])
    idx = np.arange(n); rs = np.maximum.accumulate(np.where(~same, idx, 0)) if n else idx
    kind = np.zeros(n, 'int64')                           # 0 censored, 1 improve, 2 fallback, 3 invalid (of a run start)
    if n > 1:
        better = (b0[1:] > b0[:-1]) if side == 0 else (b0[1:] < b0[:-1])
        kind[1:] = np.where(~valid[:-1], 3, np.where(better, 1, 2))
    return b0, valid, rs, kind


def fb_births(tf, b0, valid, ws, te, side):
    """Frames where the best of our side improves: both frames valid, <= 5 s apart, stamp in [ws, te)."""
    if len(b0) < 2:
        return np.zeros(0, 'int64')
    i = np.arange(1, len(b0)); better = (b0[1:] > b0[:-1]) if side == 0 else (b0[1:] < b0[:-1])
    return i[valid[1:] & valid[:-1] & better & (np.diff(tf) <= BIRTH_GAP_MS) & (tf[1:] >= ws) & (tf[1:] < te)]


def lvarr(P5, S5, pt, side):
    last = P5[:, 4]
    return np.where((last >= 0) & ((pt < last) if side == 0 else (pt > last)), np.nan, (S5 * (P5 == pt)).sum(1))


def ghost(pt, side, s, te_, qrule, tfl, lvp, ttl, tpl, tql, a_start=None, i_end=None, ss=False, se=False):
    """Queue ahead at te_ of an order live from s at pt (sim2's queue rule, our other orders ignored). ss / se = the start / end
    frame is the last one strictly before s / te_ (part C, as B13c; se for the fill check, as the cap at a trade), else the last
    one <= s / <= te_. a_start overrides the start size; i_end overrides the last trade (exclusive)."""
    j = (bisect.bisect_left(tfl, s) if ss else bisect.bisect_right(tfl, s)) - 1
    if j < 0:
        return NAN
    if a_start is None:
        a = lvp[j]; a = float(a) if np.isfinite(a) else 0.0
    else:
        a = float(a_start)
    ci = j
    i0 = bisect.bisect_left(ttl, s); i1 = bisect.bisect_left(ttl, te_) if i_end is None else i_end
    for i in range(i0, i1):
        p = tpl[i]
        if p == pt or ((p < pt) if side == 0 else (p > pt)):
            if qrule == 'M':
                jj = bisect.bisect_left(tfl, ttl[i]) - 1
                if jj > ci:
                    seg = lvp[ci + 1:jj + 1]; seg = seg[np.isfinite(seg)]
                    if len(seg):
                        a = min(a, float(seg.min()))
                    ci = jj
            a = a - min(tql[i], a) if p == pt else 0.0
    if qrule == 'M' and not MUT_GHOST_NOCAP:
        jj = (bisect.bisect_left(tfl, te_) if se else bisect.bisect_right(tfl, te_)) - 1
        if jj > ci:
            seg = lvp[ci + 1:jj + 1]; seg = seg[np.isfinite(seg)]
            if len(seg):
                a = min(a, float(seg.min()))
    return a


def chain_start(O, idx, ts_key='ts'):
    """Walk back over re-places: the previous order at the same price, cancelled by us, not filled, not refused, and this one
    decided within 1 s of that cancel. Returns (index of the chain's first order, chain position)."""
    c = idx; n = 0
    while c > 0:
        p_ = O[c - 1]
        if p_['ref'] or p_['tc'] == INF or p_['tfill'] < INF:
            break
        if p_['pt'] != O[c]['pt'] and not MUT_CHAIN_ANYPRICE:
            break
        if not (0.0 <= O[c][ts_key] - p_['tc'] <= REPLACE_MS):
            break
        c -= 1; n += 1
    return c, n


def live_chain(prev, tA, i):
    """Part C: walk back over re-places (B13c's match). A predecessor counts only if it was accepted strictly earlier: a cancel
    sent before its ack can make two orders each other's re-place (fix 2026-11-12, after the first live run looped)."""
    c = i; n = 0; seen = {i}
    while prev[c] >= 0 and tA[prev[c]] < tA[c] and prev[c] not in seen:
        c = int(prev[c]); n += 1; seen.add(c)
    return c, n


def decompose(ta, pt, a0, a0n, tb, tch, side, qrule, tfl, lvp, ttl, tpl, tql, strict=False):
    s1 = min(tb + D_OURS, tch); kw = dict(ss=strict, se=strict)
    gU = ghost(pt, side, tb, ta, qrule, tfl, lvp, ttl, tpl, tql, **kw)
    g1 = ghost(pt, side, s1, ta, qrule, tfl, lvp, ttl, tpl, tql, **kw)
    gc = ghost(pt, side, tch, ta, qrule, tfl, lvp, ttl, tpl, tql, **kw)
    gl = ghost(pt, side, ta, ta, qrule, tfl, lvp, ttl, tpl, tql, **kw)
    return dict(U=gU, LAT=g1 - gU, LATE=gc - g1, REQ=gl - gc, RAISE=a0 - gl), gl


def fclass(a0, thr, capr):
    if a0 < E_SH:
        return 'E'
    if thr:
        return 'T'
    return 'A' if capr >= A_SHARE * a0 else 'Q'


def agebin(x):
    for nm, lo, hi in AGEB:
        if lo <= x < hi:
            return nm
    return AGEB[-1][0]


# ---------------------------------------------------------------- synthetic self-tests
def syn(fr, tr, v=None, t_end=600.0, kdelay=0.0, enter=0.0):
    tf = np.array([f[0] for f in fr], float); n = len(fr)
    PB = np.full((n, 5), -1, 'int64'); SB = np.zeros((n, 5))
    for i, f in enumerate(fr):
        for j, (p, z) in enumerate(f[1][:5]):
            PB[i, j] = p; SB[i, j] = z
    bb = PB[:, 0].copy(); ba = np.array([f[2] for f in fr], 'int64')
    tau = np.arange(0.0, t_end, 50.0)
    vv = np.ones(len(tau)) if v is None else np.r_[np.asarray(v, float), np.full(max(0, len(tau) - len(v)), v[-1])][:len(tau)]
    stt = K.capped(tau, vv, enter, enter, INF)[0]
    kf = tf + kdelay
    ki = np.searchsorted(kf, tau, side='right') - 1
    ok = (ki >= 0) & (bb[np.maximum(ki, 0)] >= 0) & (ba[np.maximum(ki, 0)] > bb[np.maximum(ki, 0)])
    best = np.where(ok, bb[np.maximum(ki, 0)], -1); bo = np.where(ok, ba[np.maximum(ki, 0)], -1)
    tt = np.array([x[0] for x in tr], float)
    return dict(side=0, tf=tf, opp=ba, P5=PB, S5=SB, tau=tau, v=vv.tolist(), s=stt.tolist(), enter=enter, best=best.tolist(),
                bopp=bo.tolist(), tt=tt, tp=[int(x[1]) for x in tr], tq=[float(x[2]) for x in tr], t_end=t_end, kf=kf)


def run3(X, lane, qrule, aux=None, sched=None):
    return sim3(lane, X['side'], X['tf'], X['opp'], X['P5'], X['S5'], {}, X['tau'], X['v'], X['s'], X['enter'], X['best'], X['bopp'],
                X['tt'], X['tp'], X['tq'], qrule, X['t_end'], aux=aux, sched=sched)


def run2(X, lane, qrule, aux=None):
    return K.sim2(lane, X['side'], X['tf'], X['opp'], X['P5'], X['S5'], {}, X['tau'], X['v'], X['s'], X['enter'], X['best'], X['bopp'],
                  X['tt'], X['tp'], X['tq'], qrule, X['t_end'], aux=aux)


def selftest():
    bad = []

    def chk(name, cond):
        if not cond:
            bad.append(name)
    # S1 / S6: sim3 = sim2 for join / rest / birth, and lane B's schedule through fbirth = lane B
    fr = [(0, [(50, 10), (49, 5)], 60), (100, [(51, 4), (50, 12)], 60), (180, [(51, 9), (50, 12)], 60), (260, [(50, 12)], 60),
          (330, [(52, 3), (50, 12)], 60), (420, [(52, 7)], 60), (500, [(51, 6)], 60)]
    tr = [(120, 51, 2), (200, 51, 12), (250, 50, 3), (300, 49, 6), (360, 52, 4), (450, 52, 9), (520, 51, 20)]
    for kd in (0.0, 30.0):
        X = syn(fr, tr, v=[1, 1, 1, -1, 1, 1, 1, 1, 1, 1], kdelay=kd)
        aux = [False] * len(X['tau']); aux[5] = True
        for q in ('M', 'L', 'U'):
            for lane, ax in (('join', None), ('rest', aux), ('birth', None)):
                F2, O2 = run2(X, lane, q, ax); F3, O3, _, _ = run3(X, lane, q, ax)
                chk(f'S1 sim3=sim2 {lane} {q} k{kd:.0f}', F2 == F3 and O2 == otuples(O3, X['t_end']))
            F2, O2 = run2(X, 'birth', q); F3, O3, _, _ = run3(X, 'fbirth', q, sched=b_schedule(X['tau'], X['best'], 0))
            chk(f'S6 schedule=laneB {q} k{kd:.0f}', F2 == F3 and O2 == otuples(O3, X['t_end']) and len(F2) > 0)
    # S2: ghost carry and caps
    fr2 = [(0, [(50, 10)], 60), (100, [(50, 30)], 60), (200, [(50, 25)], 60), (250, [(50, 15)], 60)]
    X = syn(fr2, [(150, 50, 8)]); tfl = X['tf'].tolist(); lvp = lvarr(X['P5'], X['S5'], 50, 0)
    gh = lambda s, te, q, trs=None: ghost(50, 0, s, te, q, tfl, lvp, [x[0] for x in (trs or [(150, 50, 8)])],
                                          [x[1] for x in (trs or [(150, 50, 8)])], [float(x[2]) for x in (trs or [(150, 50, 8)])])
    chk('S2 ghost M 0->240', gh(0, 240, 'M') == 2.0)
    chk('S2 ghost M 100->240', gh(100, 240, 'M') == 22.0)
    chk('S2 ghost M 100->260 final cap', gh(100, 260, 'M') == 15.0)
    chk('S2 ghost L 100->260', gh(100, 260, 'L') == 22.0)
    chk('S2 ghost L 0->260', gh(0, 260, 'L') == 2.0)
    chk('S2 ghost through', gh(100, 260, 'M', [(150, 50, 8), (180, 49, 1)]) == 0.0)
    chk('S2 ghost at ta = shown', gh(210, 210, 'M') == 25.0)
    # S3: runs and FB births
    fr3 = [(0, [(50, 1)], 60), (100, [(51, 1)], 60), (200, [(51, 1)], 60), (300, [(50, 1)], 60), (400, [], 60), (500, [(50, 1)], 60),
           (600, [(52, 1)], 60)]
    X = syn(fr3, [(1, 50, 1)])
    b0, valid, rs, kind = runs(X['tf'], X['P5'], X['opp'], 0)
    chk('S3 run starts', rs.tolist() == [0, 1, 1, 3, 4, 5, 6])
    chk('S3 kinds', [int(kind[j]) for j in (1, 3, 5, 6)] == [1, 2, 3, 1] and not valid[4])
    chk('S3 FB births', fb_births(X['tf'], b0, valid, 0.0, 1000.0, 0).tolist() == [1, 6])
    # S4: chains
    O = [dict(pt=50, ts=-30.0, ta=10.0, tc=100.0, tfill=INF, ref=False), dict(pt=50, ts=150.0, ta=190.0, tc=300.0, tfill=INF, ref=False),
         dict(pt=51, ts=350.0, ta=390.0, tc=500.0, tfill=INF, ref=False), dict(pt=51, ts=1600.0, ta=1640.0, tc=INF, tfill=INF, ref=False)]
    chk('S4 chains', [chain_start(O, i)[1] for i in range(4)] == [0, 1, 0, 0])
    pv = np.array([-1, 0, 3, 2]); tAc = np.array([10.0, 50.0, 150.0, 120.0])   # 2 and 3: each other's re-place (cancel before ack)
    chk('S4 live chains end (cycle)', [live_chain(pv, tAc, i) for i in range(4)] == [(0, 0), (0, 1), (3, 1), (3, 0)])
    # S5: FB live time and grace
    fr5 = [(0, [(50, 10)], 60), (100, [(51, 5)], 60), (300, [(51, 8)], 60)]
    X = syn(fr5, [(5, 49, 1)], kdelay=60.0)
    _, O5, _, _ = run3(X, 'fbirth', 'M', sched=[(100.0 + 30.0 - L_PLACE, 51, float(X['kf'][1]))])
    chk('S5 FB live at tb + d', len(O5) == 1 and O5[0]['ta'] == 130.0 and O5[0]['a0'] == 5.0)
    chk('S5 FB grace keeps the order', len(O5) == 1 and O5[0]['tc'] == X['t_end'])
    # S7: fill classes and the removal identity
    for s2, cls in ((18, 'Q'), (6, 'A')):
        X = syn([(0, [(50, 30)], 60), (100, [(50, s2)], 60)], [(150, 50, 40)])
        F, O7, fx, _ = run3(X, 'join', 'M')
        if not F:
            chk(f'S7 fill {cls}', False); continue
        oid, ab, c0, t0, h0, _sh = fx[0]; o = O7[oid]
        chk(f'S7 identity {cls}', abs(c0 + t0 + h0 + ab - o['a0']) < 1e-9 and o['a0'] == 30.0)
        chk(f'S7 class {cls}', fclass(o['a0'], F[0][3], c0) == cls)
        gfx = ghost(50, 0, o['ta'], X['tt'][F[0][0]], 'M', X['tf'].tolist(), lvarr(X['P5'], X['S5'], 50, 0), X['tt'].tolist(), X['tp'],
                    X['tq'], a_start=o['a0'], i_end=F[0][0], se=True)
        chk(f'S7 ghost = carry {cls}', gfx == ab)
    return bad


def mutation_check():
    global L_PLACE, MUT_GHOST_NOCAP, MUT_CHAIN_ANYPRICE, MUT_NO_GRACE
    out = {}
    for nm in ('MUT_GHOST_NOCAP', 'MUT_CHAIN_ANYPRICE', 'MUT_NO_GRACE', 'L_PLACE'):
        old = globals()[nm]; globals()[nm] = 10.0 if nm == 'L_PLACE' else 1
        try:
            out[nm] = len(selftest()) > 0
        finally:
            globals()[nm] = old
    return out


def guard_selftest():
    bad = selftest(); mut = mutation_check()
    if bad or not all(mut.values()):
        sys.exit(f'SELFTEST FAILED -> no output: failing {bad}; planted errors caught {mut}')
    return mut


# ---------------------------------------------------------------- one replay day
def newacc():
    return dict(orders=0, ref=0, fills=0, q=0.0, v1=0.0, v5=0.0, nan1=0)


def addv(a, r, O=None):
    a['fills'] += int(len(r['q'])); a['q'] += float(r['q'].sum())
    a['v1'] += float(np.nansum(r['q'] * r['m1'])); a['v5'] += float(np.nansum(r['q'] * r['m5'])); a['nan1'] += int(np.isnan(r['m1']).sum())
    if O is not None:
        a['orders'] += len(O); a['ref'] += sum(1 for o in O if o['ref'])


def run_day(day):
    global L_CANCEL
    mut = guard_selftest()
    os.makedirs(OUTQ, exist_ok=True)
    D = K.load_day(day); st = D['stat']; T0 = time.time()
    st.update(window_sides=0, births=0, rj_events=0, ev_before_grid=0, reg_differ=0, sched_differ=0, ghost_ta_differ=0, ident_differ=0,
              fillghost_differ=0, fillghost_checked=0, fillghost_shared=0, lcancel_differ=0)
    ACC = {}; DEC = {}; CLA = {}; AGE = {}; ARR = {}
    rng = np.random.default_rng(int(day))
    for w in K.window_sides(D):
        st['window_sides'] += 1
        side = w['side']; tf = w['tf']; tfl = tf.tolist(); cache = w.setdefault('cache', {})
        b0, valid, rs, kind = runs(tf, w['P5'], w['opp'], side)
        bi = fb_births(tf, b0, valid, w['ws'], w['te'], side); st['births'] += int(len(bi))
        u = np.sort(rng.uniform(w['ws'], w['te'], len(bi))); ju = np.searchsorted(tf, u, side='right') - 1
        rj = [(float(x + D_FAST - L_PLACE), int(b0[j]), float(w['kf'][j])) for x, j in zip(u, ju) if j >= 0 and valid[j]]
        st['rj_events'] += len(rj)
        ARR.setdefault('bgap', []).extend((tf[i + 1] - tf[i]) if i + 1 < len(tf) else NAN for i in bi)
        for q, shf in CFG_B:
            tt = w['anc'] - shf; ttl = tt.tolist(); key = f'{q}|{shf:.0f}'

            def sims(lane, st_, enter, aux=None, sched=None, reg=True, raise_ms=100.0):
                a = (side, tf, w['opp'], w['P5'], w['S5'], cache, w['tg'], w['vkl'], st_, enter, w['best'], w['bopp'], tt, w['ptl'], w['shl'], q, w['te'])
                F3, O3, fx, s3 = sim3(lane, *a, aux=aux, sched=sched, raise_ms=raise_ms); st['ev_before_grid'] += s3['ev_before_grid']
                if reg:
                    F2, O2 = K.sim2('birth' if lane == 'fbirth' else lane, *a, aux=aux)
                    if F2 != F3 or O2 != otuples(O3, w['te']):
                        st['reg_differ' if lane != 'fbirth' else 'sched_differ'] += 1
                return F3, O3, fx
            if (q, shf) in CFG_A:
                for arm in ARMS_A:
                    if arm == 'J-v3':
                        F, O, fx = sims('join', w['stv3'], w['P90'])
                    else:
                        aux = (np.asarray(w['vkl']) <= K.GUARD_PULL).tolist() if arm == 'Z-guard' else [False] * w['n']
                        F, O, fx = sims('rest', w['stn'], -INF, aux=aux)
                    if day == DAYS9[0] and st['window_sides'] <= 40 and arm == 'J-v3' and (q, shf) == PRIM:   # control iii
                        L_CANCEL = 30.0
                        try:
                            Fm, Om, _, _ = sim3('join', side, tf, w['opp'], w['P5'], w['S5'], cache, w['tg'], w['vkl'], w['stv3'], w['P90'], w['best'],
                                                w['bopp'], tt, w['ptl'], w['shl'], q, w['te'])
                        finally:
                            L_CANCEL = K.L_CANCEL
                        st['lcancel_differ'] += int(Fm != F or otuples(Om, w['te']) != otuples(O, w['te']))
                    r = K.vals(w, F, tt); a_ = ACC.setdefault(f'{arm}|{key}', newacc()); addv(a_, r, O)
                    dec = DEC.setdefault(f'{arm}|{key}', dict(n=0, a0=0.0, **{c: 0.0 for c in COMP}, status={}, kinds={k_: 0 for k_ in KINDS},
                                                             replaced=0, chain_len=0, comp_kind={k_: {c: 0.0 for c in COMP} for k_ in KINDS[1:]}))
                    info = {}
                    for idx, o in enumerate(O):
                        if o['ref']:
                            dec['status']['refused'] = dec['status'].get('refused', 0) + 1; continue
                        jl = bisect.bisect_right(tfl, o['ta']) - 1
                        if jl < 0 or not valid[jl] or b0[jl] != o['pt']:
                            dec['status']['not_best_at_live'] = dec['status'].get('not_best_at_live', 0) + 1; continue
                        j0 = int(rs[jl]); kd = KINDS[0] if j0 == 0 else KINDS[int(kind[j0])]
                        dec['kinds'][kd] += 1
                        if j0 == 0:
                            dec['status']['censored'] = dec['status'].get('censored', 0) + 1; continue
                        tb = tfl[j0]; c, npos = chain_start(O, idx); tch = max(O[c]['ta'], tb)
                        lvp = cache[o['pt']]
                        cm, gl = decompose(o['ta'], o['pt'], o['a0'], o['a0n'], tb, tch, side, q, tfl, lvp, ttl, w['ptl'], w['shl'])
                        st['ghost_ta_differ'] += int(abs(gl - o['a0n']) > 1e-9)
                        dec['n'] += 1; dec['a0'] += o['a0']; dec['replaced'] += int(npos > 0); dec['chain_len'] += npos
                        for cc in COMP:
                            dec[cc] += cm[cc]; dec['comp_kind'][kd][cc] += cm[cc]
                        info[o['id']] = (o['ta'] - tb, npos, kd)
                        if (q, shf) == PRIM:
                            ARR.setdefault(f'age|{arm}', []).append(o['ta'] - tb); ARR.setdefault(f'start|{arm}', []).append(tch - tb if npos == 0 else NAN)
                            ARR.setdefault(f'a0|{arm}', []).append(o['a0'])
                    for f, x, m1, qq in zip(F, fx, r['m1'], r['q']):
                        oid, ab, c0, t0, h0, shared = x; o = O[oid]
                        if q in ('M', 'L'):
                            st['ident_differ'] += int(abs(c0 + t0 + h0 + ab - o['a0']) > 1e-6 or (q == 'L' and c0 > 0))
                            if f[1] == o['pt'] and not f[3]:
                                if shared:
                                    st['fillghost_shared'] += 1
                                else:
                                    gf = ghost(o['pt'], side, o['ta'], ttl[f[0]], q, tfl, cache[o['pt']], ttl, w['ptl'], w['shl'], a_start=o['a0'],
                                               i_end=f[0], se=True)
                                    st['fillghost_checked'] += 1; st['fillghost_differ'] += int(abs(gf - ab) > 1e-9)
                        cl = fclass(o['a0'], f[3], c0)
                        e = CLA.setdefault(f'{arm}|{key}|{cl}', dict(fills=0, q=0.0, v1=0.0)); e['fills'] += 1; e['q'] += qq
                        e['v1'] += float(qq * m1) if np.isfinite(m1) else 0.0
                        if oid in info:
                            age, npos, kd = info[oid]
                            for lab in (f'age:{agebin(age)}', 'chain:first' if npos == 0 else 'chain:replace', f'kind:{kd}'):
                                e = AGE.setdefault(f'{arm}|{key}|{lab}', dict(fills=0, q=0.0, v1=0.0)); e['fills'] += 1; e['q'] += qq
                                e['v1'] += float(qq * m1) if np.isfinite(m1) else 0.0
            for gx in ('none', 'g90'):
                st_, enter = (w['stn'], -INF) if gx == 'none' else (w['stv3'], w['P90'])
                F, O, fx = sims('fbirth', st_, enter, sched=b_schedule(w['tg'], w['best'], side))
                r = K.vals(w, F, tt); addv(ACC.setdefault(f'B-{gx}|{key}', newacc()), r, O)
                if (q, shf) == PRIM:
                    for o in O:
                        if not o['ref']:
                            jl = bisect.bisect_right(tfl, o['ta']) - 1
                            if jl >= 0 and valid[jl] and b0[jl] == o['pt'] and rs[jl] > 0:
                                ARR.setdefault(f'bdelay|B-{gx}', []).append(o['ta'] - tfl[int(rs[jl])])
                            ARR.setdefault(f'a0|B-{gx}', []).append(o['a0'])
                for d in D_SWEEP:
                    sch = [(float(tf[i] + d - L_PLACE), int(b0[i]), float(w['kf'][i])) for i in bi]
                    F, O, fx = sims('fbirth', st_, enter, sched=sch, reg=False)
                    r = K.vals(w, F, tt); addv(ACC.setdefault(f'FB{d:.0f}-{gx}|{key}', newacc()), r, O)
                    if (q, shf) == PRIM:
                        ARR.setdefault(f'a0|FB{d:.0f}-{gx}', []).extend(o['a0'] for o in O if not o['ref'])
                        F, O, fx = sims('fbirth', st_, enter, sched=sch, reg=False, raise_ms=None)   # sensitivity: no raise
                        r = K.vals(w, F, tt); addv(ACC.setdefault(f'FB{d:.0f}nr-{gx}|{key}', newacc()), r, O)
                        ARR.setdefault(f'a0|FB{d:.0f}nr-{gx}', []).extend(o['a0'] for o in O if not o['ref'])
                F, O, fx = sims('fbirth', st_, enter, sched=rj, reg=False)
                r = K.vals(w, F, tt); addv(ACC.setdefault(f'RJ30-{gx}|{key}', newacc()), r, O)
                if (q, shf) == PRIM:
                    ARR.setdefault(f'a0|RJ30-{gx}', []).extend(o['a0'] for o in O if not o['ref'])
        w.pop('cache', None)
        if st['window_sides'] % 40 == 0:
            r_ = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r_:.0f} MiB', flush=True)
            if r_ > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r_:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    bad = {k_: st[k_] for k_ in ('reg_differ', 'sched_differ', 'ghost_ta_differ', 'ident_differ', 'fillghost_differ') if st[k_]}
    if day == DAYS9[0] and st['lcancel_differ'] == 0:
        bad['lcancel_never_differs'] = 0
    if bad:
        sys.exit(f'QR CONTROL FAILED on {day} -> no result: {bad}')
    out = dict(day=day, days=len(D['W0']) / 288.0, stat=st, planted=mut, acc=ACC, dec=DEC, cla=CLA, age=AGE)
    json.dump(out, open(f'{OUTQ}/day_{day}.json', 'w'), default=float)
    np.savez_compressed(f'{OUTQ}/arr_{day}.npz', **{k_: np.asarray(v_, float) for k_, v_ in ARR.items()})
    print(f'{day}: done in {time.time() - T0:.0f} s; controls passed {json.dumps({k_: st[k_] for k_ in st if "differ" in k_ or "fillghost" in k_})}', flush=True)


# ---------------------------------------------------------------- part C: the live engine on 11-08
def run_live():
    mut = guard_selftest()
    os.makedirs(OUTQ, exist_ok=True)
    sys.path.insert(0, ROOT)
    import pandas as pd
    from predictfun.research import pf_b12b_reason as BB, pf_b12d_bursts as DB, pf_b13c_queue_pulls as BC
    ok, ctl = BC.controls()
    if not ok:
        sys.exit(f'CONTROL FAILED (B13c controls) -> no table: {ctl}')
    X = BB.build(); R = X['R']; W0 = X['W0']
    k0 = BB.reconcile(R, X['days'])
    if not k0['ok']:
        sys.exit(f'CONTROL FAILED (K0) -> no table: {k0}')
    core = []
    for rn in ('v6', 'basis'):
        Kd = pd.read_parquet(f'{BB.LV}/core_orders_{rn}.parquet')
        core.append(Kd[Kd.t_Accepted.notna() & Kd.refused.isna() & Kd.market.isin(list(W0))].assign(run=rn))
    core = pd.concat(core, ignore_index=True)
    if abs(float(core.filled.sum()) - BC.LIVE_SH) > 1e-3:
        sys.exit(f'CONTROL FAILED (K-check live shares) -> no table: {float(core.filled.sum())}')
    core['side'] = (core.lane % 2).astype(int); core['ws'] = core.market.map(W0).astype(float)
    FR = DB.frames(W0)
    Tl = R[np.isfinite(R.anc.values)][['mid', 'side', 'ptick', 'anc', 'shares']].sort_values('anc', kind='stable')
    TR = {k_: (g.anc.values.astype(float).tolist(), g.ptick.values.astype('int64').tolist(), g.shares.values.astype(float).tolist())
          for k_, g in Tl.groupby(['mid', 'side'])}
    res = dict(as_of_note='point estimates, one consumed partial day (220 windows of 2026-11-08), first minute by t_Accepted', planted=mut, K0=k0['ok'])
    st = dict(orders=0, outside_first_minute=0, no_frames=0, not_best_at_live=0, censored=0, ghost_ta_differ=0)
    DECL = {q: dict(n=0, a0=0.0, **{c: 0.0 for c in COMP}, replaced=0, kinds={k_: 0 for k_ in KINDS}) for q in ('L', 'M')}
    delays = []; ages = []
    RBC = {}
    for rn, cr in core.groupby('run', sort=False):
        cr = cr.sort_values(['market', 't_Accepted'], kind='stable').reset_index(drop=True)
        rp = BC.replace_match(cr); prev = np.full(len(cr), -1, 'int64')
        for i, j in enumerate(rp):
            if j >= 0 and (prev[j] < 0 or cr.t_cancel.values[i] > cr.t_cancel.values[prev[j]]):
                prev[j] = i
        tA = cr.t_Accepted.values.astype(float); tick = cr.tick.values.astype('int64'); sd = cr.side.values; mk = cr.market.values; ws = cr.ws.values
        for i in range(len(cr)):
            st['orders'] += 1
            if st['orders'] % 5000 == 0:
                print(f'live: {st["orders"]} orders, rss {K.rss_mib():.0f} MiB', flush=True)
            if not (ws[i] <= tA[i] < ws[i] + 60000.0):
                st['outside_first_minute'] += 1; continue
            m = int(mk[i]); s_ = int(sd[i])
            if m not in FR:
                st['no_frames'] += 1; continue
            tf, S = FR[m]; Pk, Sk = S[s_]; opp = S[1 - s_][0][:, 0]; tfl = tf.tolist()
            if (m, s_) not in RBC:
                RBC[(m, s_)] = runs(tf, Pk, opp, s_)
            b0, valid, rs, kind = RBC[(m, s_)]
            jl = bisect.bisect_left(tfl, tA[i]) - 1
            if jl < 0 or not valid[jl] or b0[jl] != tick[i]:
                st['not_best_at_live'] += 1; continue
            j0 = int(rs[jl])
            if j0 == 0:
                st['censored'] += 1; continue
            kd = KINDS[int(kind[j0])]; tb = tfl[j0]
            c, npos = live_chain(prev, tA, i)
            tch = max(tA[c], tb)
            lvp = lvarr(Pk, Sk, int(tick[i]), s_)
            a0 = BC.queue_at(tf, Pk, Sk, tA[i], int(tick[i]))
            ttl, tpl, tql = TR.get((m, s_), ([], [], []))
            for q in ('L', 'M'):
                cm, gl = decompose(tA[i], int(tick[i]), a0, NAN, tb, tch, s_, q, tfl, lvp, ttl, tpl, tql, strict=True)
                st['ghost_ta_differ'] += int(abs(gl - DB.size_at(Pk, Sk, jl, int(tick[i]))) > 1e-9)
                dd = DECL[q]; dd['n'] += 1; dd['a0'] += a0; dd['replaced'] += int(npos > 0); dd['kinds'][kd] += 1
                for cc in COMP:
                    dd[cc] += cm[cc]
            ages.append(tA[i] - tb)
            if npos == 0:
                delays.append(tA[i] - tb)
    if st['ghost_ta_differ']:
        sys.exit(f'CONTROL FAILED (live g(ta) differs from the shown size) -> no table: {st}')
    res.update(stat=st, dec=DECL, age_p10_50_90=[float(x) for x in np.percentile(ages, [10, 50, 90])] if ages else [],
               first_delay_p10_50_90=[float(x) for x in np.percentile(delays, [10, 50, 90])] if delays else [],
               first_delay_share_le_97=float(np.mean(np.asarray(delays) <= D_OURS)) if delays else NAN)
    json.dump(res, open(f'{OUTQ}/live.json', 'w'), indent=1, default=float)
    print(json.dumps(res, default=float)[:2000], flush=True)


# ---------------------------------------------------------------- report and readings
def cps(a):
    return a['v1'] / a['q'] if a['q'] > 0 else NAN


def cause_label(d):
    tot = {c: d[c] for c in ('U', 'LAT', 'LATE', 'REQ')}
    k_ = max(tot, key=tot.get)
    return CAUSE_LABEL[k_], k_


def report():
    DY = {}
    for day in DAYS9:
        fn = f'{OUTQ}/day_{day}.json'
        if not os.path.exists(fn):
            sys.exit(f'REFUSED: {fn} missing')
        DY[day] = json.load(open(fn))
    AR = {day: dict(np.load(f'{OUTQ}/arr_{day}.npz')) for day in DAYS9}
    LIVE = json.load(open(f'{OUTQ}/live.json')) if os.path.exists(f'{OUTQ}/live.json') else None
    ndays = sum(DY[d]['days'] for d in DAYS9)
    L = []; RES = dict(as_of=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), days=DAYS9, day_equiv=ndays)
    P = lambda s='': L.append(s)

    def pooled(kind, key):
        out = None
        for d in DAYS9:
            x = DY[d][kind].get(key)
            if x is None:
                continue
            if out is None:
                out = json.loads(json.dumps(x))
            else:
                for k_, v_ in x.items():
                    if isinstance(v_, (int, float)):
                        out[k_] += v_
                    elif isinstance(v_, dict):
                        for k2, v2 in v_.items():
                            if isinstance(v2, dict):
                                for k3, v3 in v2.items():
                                    out[k_][k2] = out[k_].get(k2, {}); out[k_][k2][k3] = out[k_][k2].get(k3, 0.0) + v3
                            else:
                                out[k_][k2] = out[k_].get(k2, 0) + v2
        return out
    st = {k_: sum(DY[d]['stat'].get(k_, 0) for d in DAYS9) for k_ in DY[DAYS9[0]]['stat'] if isinstance(DY[DAYS9[0]]['stat'][k_], (int, float))}
    RES['stat'] = st
    P(f'QR queue race - 9 replay days ({ndays:.2f} day-equivalents), first 60 s; as of {RES["as_of"]}')
    P(f'controls: sim3=sim2 differ {st["reg_differ"]}, schedule=laneB differ {st["sched_differ"]}, g(ta) differ {st["ghost_ta_differ"]}, '
      f'identity differ {st["ident_differ"]}, fill ghost differ {st["fillghost_differ"]} of {st["fillghost_checked"]} (shared skipped '
      f'{st["fillghost_shared"]}), L_CANCEL 30 differs on {DY[DAYS9[0]]["stat"]["lcancel_differ"]} window-sides of day 1')
    P(f'statuses: window-sides {st["window_sides"]}, births {st["births"]}, random joins {st["rj_events"]}, events before the first grid step '
      f'{st["ev_before_grid"]}')
    # ---- part A
    P(); P('PART A - where the queue ahead at activation comes from (mean sh per decomposed order; share of a0)')
    P(f'{"arm|cfg":<16} {"orders":>8} {"a0":>7} ' + ' '.join(f'{c:>14}' for c in COMP) + '  replaced  statuses')
    RES['A'] = {}
    for arm in ARMS_A:
        for q, shf in CFG_A:
            key = f'{arm}|{q}|{shf:.0f}'; d = pooled('dec', key)
            if not d or d['n'] == 0:
                continue
            RES['A'][key] = d
            P(f'{key:<16} {d["n"]:>8} {d["a0"] / d["n"]:>7.1f} ' + ' '.join(f'{d[c] / d["n"]:>6.1f} ({d[c] / d["a0"]:>5.1%})' for c in COMP)
              + f'  {d["replaced"] / d["n"]:>7.1%}  {d["status"]}')
    P('birth kinds of decomposed orders (J-v3, M 50): ' + json.dumps(RES['A'].get('J-v3|M|50', {}).get('kinds', {})))
    d = RES['A'].get('J-v3|M|50')
    if d:
        P('J-v3 M 50 components by birth kind (mean sh): ' + '; '.join(
            f'{kd}: ' + ', '.join(f'{c} {d["comp_kind"][kd][c] / max(1, d["kinds"][kd]):.1f}' for c in COMP) for kd in KINDS[1:]))
    for arm in ARMS_A:
        a = np.concatenate([AR[dd].get(f'age|{arm}', np.zeros(0)) for dd in DAYS9]); s0 = np.concatenate([AR[dd].get(f'start|{arm}', np.zeros(0)) for dd in DAYS9])
        s0 = s0[np.isfinite(s0)]
        if len(a):
            P(f'{arm} M 50: level age at activation p10/50/90 {np.percentile(a, [10, 50, 90]).round(0).tolist()} ms; first orders of chains: '
              f'activation - birth p10/50/90 {np.percentile(s0, [10, 50, 90]).round(0).tolist() if len(s0) else []} ms, share <= 97 ms '
              f'{np.mean(s0 <= D_OURS) if len(s0) else NAN:.1%}')
    # R280
    d = RES['A']['J-v3|M|50']; lab, kc = cause_label(d)
    perday = []
    for day in DAYS9:
        x = DY[day]['dec'].get('J-v3|M|50')
        perday.append(cause_label(x)[1] if x and x['n'] else None)
    r280 = f'{lab} ' + ('STABLE' if sum(1 for x in perday if x == kc) >= 7 else 'UNSTABLE')
    RES['R280'] = dict(label=r280, per_day=perday)
    P(f'R280 CAUSE (J-v3, M 50): {r280}; largest per day {perday}')
    # fill classes
    P(); P('fill classes (c/sh at 1 s, fills): E = queue ahead at activation < 20 sh, T = through-trade, A = mostly removed by others cancelling, Q = mostly traded through')
    RES['classes'] = {}
    for arm in ARMS_A:
        for q, shf in CFG_A:
            row = []
            for cl in CLS:
                e = pooled('cla', f'{arm}|{q}|{shf:.0f}|{cl}')
                if e:
                    RES['classes'][f'{arm}|{q}|{shf:.0f}|{cl}'] = e; row.append(f'{cl} {cps(e):+.2f} ({e["fills"]})')
            P(f'  {arm}|{q}|{shf:.0f}: ' + ', '.join(row))
    P('by level age at activation / first vs re-placed / birth kind (c/sh at 1 s, fills), M 50:')
    for arm in ARMS_A:
        row = []
        for lab_ in [f'age:{a[0]}' for a in AGEB] + ['chain:first', 'chain:replace'] + [f'kind:{k_}' for k_ in KINDS[1:]]:
            e = pooled('age', f'{arm}|M|50|{lab_}')
            if e:
                row.append(f'{lab_} {cps(e):+.2f} ({e["fills"]})')
        P(f'  {arm}: ' + ', '.join(row))

    def dayc(kind, key):
        out = []
        for day in DAYS9:
            e = DY[day][kind].get(key)
            out.append(cps(e) if e and e['fills'] >= MIN_FILLS else NAN)
        return np.array(out)
    RES['R282'] = {}
    for arm in ('Z-none', 'J-v3'):
        okc = [int(np.sum(dayc('cla', f'{arm}|M|{s_:.0f}|E') > 0)) for s_ in (0.0, 50.0)]
        RES['R282'][arm] = dict(label='EARLY-PAYS' if min(okc) >= 7 else 'EARLY-DOES-NOT-PAY', days_pos_M0_M50=okc)
        P(f'R282 EARLY ({arm}): {RES["R282"][arm]["label"]}; days with class E > 0 at M 0 / M 50: {okc}')
    worse = []; better = []
    for s_ in (0.0, 50.0):
        a_ = dayc('cla', f'Z-none|M|{s_:.0f}|A'); q_ = dayc('cla', f'Z-none|M|{s_:.0f}|Q')
        worse.append(int(np.sum(a_ < q_))); better.append(int(np.sum(a_ > q_)))
    r283 = 'ATTRITION-WORSE' if min(worse) >= 7 else ('ATTRITION-BETTER' if min(better) >= 7 else 'MIXED')
    RES['R283'] = dict(label=r283, days_A_below_Q=worse, days_A_above_Q=better)
    P(f'R283 ATTRITION (Z-none): {r283}; days A < Q at M 0 / M 50 {worse}, A > Q {better}')
    # ---- part B
    P(); P('PART B - joining new levels at speed d (FB), lane B (current clock), random joins at 30 ms (RJ30), J-v3')
    names = [f'FB{d:.0f}' for d in D_SWEEP] + ['B', 'RJ30']
    for gx in ('g90', 'none'):
        P(f'guard {gx}: c/sh at 1 s (fills) per config; $/day at 1 s; median queue ahead at activation (M 50)')
        P(f'{"arm":<12} ' + ' '.join(f'{q + str(int(s_)):>16}' for q, s_ in CFG_B) + f' {"$/day M50":>10} {"a0 p50":>7} {"orders/ws":>9} {"refused":>8}')
        for nm in names + (['J-v3'] if gx == 'g90' else []):
            arm = nm if nm == 'J-v3' else f'{nm}-{gx}'
            cells = []
            for q, s_ in CFG_B:
                a = pooled('acc', f'{arm}|{q}|{s_:.0f}')
                cells.append(f'{cps(a):+.2f} ({a["fills"]:>6})' if a else f'{"":>16}')
            a = pooled('acc', f'{arm}|M|50')
            a0s = np.concatenate([AR[dd].get(f'a0|{arm}', np.zeros(0)) for dd in DAYS9])
            P(f'{arm:<12} ' + ' '.join(f'{c:>16}' for c in cells) + f' {a["v1"] / 100.0 / ndays:>10.2f} {np.median(a0s) if len(a0s) else NAN:>7.0f} '
              f'{a["orders"] / st["window_sides"]:>9.2f} {a["ref"] / max(1, a["orders"]):>8.1%}')
            RES.setdefault('B', {})[arm] = dict(cells={f'{q}|{s_:.0f}': pooled('acc', f'{arm}|{q}|{s_:.0f}') for q, s_ in CFG_B},
                                               a0_p50=float(np.median(a0s)) if len(a0s) else NAN)
        bd = np.concatenate([AR[dd].get(f'bdelay|B-{gx}', np.zeros(0)) for dd in DAYS9])
        if len(bd):
            P(f'lane B-{gx}: realised activation - birth p10/50/90 {np.percentile(bd, [10, 50, 90]).round(0).tolist()} ms')
    P('sensitivity, no raise to the next frame (M 50; c/sh at 1 s, fills, median queue ahead): ' + '; '.join(
        f'FB{d:.0f}nr-{gx} {cps(pooled("acc", f"FB{d:.0f}nr-{gx}|M|50")):+.2f} ({pooled("acc", f"FB{d:.0f}nr-{gx}|M|50")["fills"]}, '
        f'{np.median(np.concatenate([AR[dd].get(f"a0|FB{d:.0f}nr-{gx}", np.zeros(0)) for dd in DAYS9])):.0f})' for gx in ('g90', 'none') for d in D_SWEEP))
    bg = np.concatenate([AR[dd].get('bgap', np.zeros(0)) for dd in DAYS9]); bg = bg[np.isfinite(bg)]
    P(f'frame after a birth frame: gap p10/50/90 {np.percentile(bg, [10, 50, 90]).round(1).tolist() if len(bg) else []} ms')
    a30 = RES['B']['FB30-none']['a0_p50']; a97 = RES['B']['FB97-none']['a0_p50']
    RES['R284'] = dict(label='SPEED-BUYS-QUEUE' if a30 <= 0.5 * a97 else 'LITTLE-QUEUE-FROM-SPEED', a0_p50_FB30=a30, a0_p50_FB97=a97)
    P(f'R284 QUEUE-BY-SPEED: {RES["R284"]["label"]} (median queue ahead FB30 {a30:.0f} vs FB97 {a97:.0f} sh)')
    RES['R285'] = {}; RES['R286'] = {}
    for gx in ('g90', 'none'):
        pos_ = []; neg_ = []
        for s_ in (0.0, 50.0):
            dlt = dayc('acc', f'FB30-{gx}|M|{s_:.0f}') - dayc('acc', f'FB97-{gx}|M|{s_:.0f}')
            pos_.append(int(np.sum(dlt > 0))); neg_.append(int(np.sum(dlt < 0)))
        lab = 'SPEED-PAYS' if min(pos_) >= 7 else ('SPEED-HURTS' if min(neg_) >= 7 else 'MIXED')
        RES['R285'][gx] = dict(label=lab, days_pos=pos_, days_neg=neg_)
        P(f'R285 SPEED-PAYS (FB-{gx}): {lab}; days FB30 > FB97 at M 0 / M 50 {pos_}, < {neg_}')
        POS = []; V3 = []
        for dd_ in D_SWEEP:
            arm = f'FB{dd_:.0f}-{gx}'
            okp = all(cps(pooled('acc', f'{arm}|M|{s_:.0f}')) > 0 and np.sum(dayc('acc', f'{arm}|M|{s_:.0f}') > 0) >= 7 for s_ in (0.0, 50.0))
            okv = all(np.sum(dayc('acc', f'{arm}|M|{s_:.0f}') > dayc('acc', f'J-v3|M|{s_:.0f}')) >= 7 for s_ in (0.0, 50.0))
            if okp:
                POS.append(dd_)
            if okv:
                V3.append(dd_)
        RES['R286'][gx] = dict(POS=max(POS) if POS else 'NONE', V3=max(V3) if V3 else 'NONE', pos_set=POS, v3_set=V3)
        P(f'R286 SPEED-NEEDED (FB-{gx}): POS {RES["R286"][gx]["POS"]} (d passing: {POS}); V3 {RES["R286"][gx]["V3"]} (d passing: {V3}); d = 0 is a bound')
    # ---- part C
    P(); P('PART C - the live engine, 11-08 (one day, point estimates)')
    if LIVE:
        RES['C'] = LIVE; RES['R281'] = {}
        P(f'statuses: {LIVE["stat"]}')
        for q in ('L', 'M'):
            d = LIVE['dec'][q]
            if d['n'] == 0:
                continue
            lab, _ = cause_label(d); RES['R281'][q] = lab
            P(f'rule {q}: orders {d["n"]}, a0 {d["a0"] / d["n"]:.1f} sh; ' + ', '.join(f'{c} {d[c] / d["n"]:.1f} ({d[c] / d["a0"]:.1%})' for c in COMP)
              + f'; re-placed {d["replaced"] / d["n"]:.1%}; kinds {d["kinds"]}; R281 CAUSE-LIVE {lab}')
        P(f'level age at accept p10/50/90 {LIVE["age_p10_50_90"]} ms; first orders: accept - birth p10/50/90 {LIVE["first_delay_p10_50_90"]} ms, '
          f'share <= 97 ms {LIVE["first_delay_share_le_97"]:.1%}')
    else:
        P('live.json missing: part C not run')
    json.dump(RES, open(f'{OUTQ}/qr.json', 'w'), indent=1, default=float)
    open(f'{OUTQ}/qr.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        bad = selftest(); mut = mutation_check(); print('failing:', bad, 'planted errors caught:', mut)
        sys.exit(0 if not bad and all(mut.values()) else 1)
    elif cmd == 'day':
        run_day(sys.argv[2])
    elif cmd == 'live':
        run_live()
    elif cmd == 'report':
        report()
    else:
        sys.exit(__doc__)
