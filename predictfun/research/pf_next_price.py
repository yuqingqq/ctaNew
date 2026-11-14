"""NEXT-PRICE (2026-11-14, session ce68ebe0; USER: "For example, when best bid is 50c, their orders are 49c, and always there? Can we
also put our orders early there, It's we are late when we first there or we late because we repost").

At the moment the next price becomes our side's best (the best bid at 50c goes and 49c becomes the best: a FALLBACK birth), where were
we, and who held the front of that price?

Simulator: the guard ladder (LDG) exactly as QUEUE-PLACE (pf_queue_place.sim4: S4's rest lane, 30 sh kept through partial fills, rungs
d = 0..5 ticks behind the known best; rung d is placed at the known best - d, kept while the best stays within HOLD_K + d ticks of it,
pulled on the guard trigger v <= -0.10 and re-placed once the trigger has been clear 250 ms; each rung an independent order). 9 consumed
replay days (10-23..27, 10-31, 11-01..03), first 60 s of each window, fill clocks M 50 (primary), M 0, M 250.

Births: QR.runs on the venue frames (a run = consecutive valid frames with the same best on our side): the run's first frame (stamp tb),
its price pt and kind (fallback: the level in front went away; improve: a better price appeared); the run ends at the next run's first
frame (or the lane's end). Population: births with tb in the lane's minute [ws, te). Statuses: no decision step at or before tb
(pre_grid); the guard value masked at tb (v = -1 until the window's start price is known, FINDINGS P16) - both outside the readings.

Our status at a birth, over the orders of all six rungs on that side at price pt (an order is in the book from its go-live time until its
cancel lands or it is used up; 'requested' = our cancel decision):
  LIVE    an order of ours at pt went live by tb, and no cancel was requested and it was not used up by tb (the oldest such order counts)
  PULLED  none LIVE; the latest order of ours at pt whose cancel was requested (or which was used up) in the 5 s before tb was pulled by
          the guard (PENDING beside: the cancel requested before tb lands after it, so it is still in the book at the birth, leaving)
  MOVED   ... cancelled because the best had moved more than HOLD_K + d ticks away from it (the ladder re-centred)
  USED    ... used up by fills
  OTHER   ... cross / nobook / end
  NEVER   no order of ours at pt went live and ended in the 5 s before tb (or is LIVE); split by the guard at tb (on / off)
  then: the delay from tb to our next order at pt going live (any rung) while the run lasts (NONE if the run ends first).
Queue ahead (rule M's arithmetic, the order's own fills and trades ignored): the size shown at pt when the order went live (with the
simulator's raise to the next frame within 100 ms) capped by the smallest size shown at pt in the frames after its activation frame up to
and including the birth frame. At tb for LIVE orders; at tb for PULLED orders had they been kept; at activation for our next order after
tb. The size shown at pt in the birth frame beside.
The top makers (real fills, descriptive; their orders are seen only through fills): the first taker order with a leg at pt in
[tb - 250 ms, run end), ordered by the legs' anchors; if it is not exactly timed (complete WS-timed taker order in ORDERS' map, as
QUEUE-PLACE / BEST-GAP) the birth has the status first_not_exact. Per first trade: the top 5's volume at pt and all volume at pt;
CROSSING = the same taker order also filled a better price for this side (the orders it met at pt were resting there before the birth);
SMALL = <= 30 sh at pt (only the front of the queue is filled). Baseline: the top 5's share of all exactly timed at-best maker volume in
the same windows' first minute (BEST-GAP's real rows).
Readings DECLARED BEFORE THE FIRST RUN (fallback births, M 50, masked / pre_grid excluded; day rule >= 7 of 9):
 R570 WHY-NOT-THERE: among births where we were not LIVE, the share PULLED + MOVED (we had an order at that price and cancelled it before
      the price became the best: "late because we repost") against NEVER ("late when we first get there"): (PULLED + MOVED) >= 0.5 ->
      LATE-BY-REPOST; NEVER >= 0.5 -> LATE-FROM-FIRST-ARRIVAL; else MIXED. STABLE if the per-day label equals the pooled one on >= 7 of 9
      days, else UNSTABLE.
 R571 EARLY-IS-AHEAD: the median queue ahead at tb of LIVE orders, and of PULLED orders had they been kept, each <= half the median queue
      ahead of our next order after tb at its activation, pooled and on >= 7 of 9 days (a day lacking a class does not count as passing)
      -> EARLY-IS-AHEAD; else NOT-SHOWN.
 R572 FRONT-AT-BIRTH (theirs): the top 5's volume share at pt in SMALL exactly timed first trades, over their baseline share: >= 1.2
      pooled and > 1 on >= 7 of 9 days -> THEY-HOLD-THE-FRONT; else NOT-SHOWN. Printed: the share of births whose small first trade fills
      a top-5 maker; all first trades; CROSSING.
Printed, not readings: improve births; M 0 / M 250; by fallback size (1 tick / >= 2); the guard at tb; PENDING; rungs; time since the pull;
how long LIVE orders stay after tb and what ends them; delay to our next order; statuses.
Controls (any failure -> no output): (i) sim4 equals pf_kp_s4.sim_arm fill for fill and order for order on every window-side, rung and
clock (QUEUE-PLACE's control (i)), and QUEUE-PLACE's self-tests pass; (ii) synthetic cases with known answers for the status (LIVE,
PULLED, PENDING, MOVED, USED, outside the 5 s lookback, another price), the queue arithmetic, the births and the first trade (not exact,
crossing, small, outside the window), with planted errors (the cancel landing read as the request; the lookback ignored; the price
ignored; crossing read on the wrong side) that must each fail a case; (iii) memory self-guard (2 GiB).
Limits: replay rule M (B1a): every queue number is the simulator's model; our order is not in the public book; the rungs are independent
orders; the top makers' entry times are not observable (CROSSING proves presence before the birth, not for how long); first trade only;
consumed days (C8); the 5 s lookback and the 30 sh line are fixed here.
Usage (run_light, PF_SLOT=claude): pf_next_price.py selftest | day <day> | report; driver run_next_price.sh
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
import pf_qr_queue_race as QR                             # noqa: E402  (runs, lvarr)
import pf_queue_place as QP                               # noqa: E402  (sim4, ORD, OLDKEY, self-tests)

DAYS9 = K.DAYS9
OUTD = f'{K.IO}/next_price_20261114'
CFGS = [('M', 50.0), ('M', 0.0), ('M', 250.0)]
RUNGS = range(6)
SIZE = 30.0
LOOKBACK = 5000.0; PRE_TRADE = 250.0; SMALL = 30.0; MASK_V = -1.0
INF = float('inf'); NAN = float('nan')
KNAME = {1: 'improve', 2: 'fallback'}
STATUSES = ('LIVE', 'PULLED', 'MOVED', 'USED', 'OTHER', 'NEVER')
MUT_LANDING_AS_REQ = MUT_NO_LOOKBACK = MUT_ANY_PRICE = MUT_CROSS_SIDE = 0     # planted errors (self-test only)
MEM_CAP_MIB = 2048.0


# ---------------------------------------------------------------- per birth
def births(tfl, b0, valid, rs, kind, ws, te):
    """Run starts of kind improve / fallback with stamp in [ws, te): (frame, tb, pt, kind, run end, ticks moved)."""
    n = len(tfl); starts = [j for j in range(1, n) if rs[j] == j]; out = []
    for a, j in enumerate(starts):
        if int(kind[j]) not in KNAME or not valid[j] or not (ws <= tfl[j] < te):
            continue
        tend = tfl[starts[a + 1]] if a + 1 < len(starts) else te
        out.append((j, tfl[j], int(b0[j]), KNAME[int(kind[j])], min(tend, te), abs(int(b0[j]) - int(b0[j - 1]))))
    return out


def status_at(orders, pt, tb):
    """Our status at price pt at the birth tb (header). orders: sim4 order dicts of all rungs on the side (with 'rung').
    Returns (status, the order or None, pending)."""
    cand = [o for o in orders if not o['ref'] and (MUT_ANY_PRICE or o['pt'] == pt) and o['ta'] <= tb]
    if MUT_LANDING_AS_REQ:
        req = lambda o: min(o['tr'], o['tfill'])
    else:
        req = lambda o: min(o['tc'], o['tfill'])
    keep = [o for o in cand if req(o) > tb]
    if keep:
        return 'LIVE', min(keep, key=lambda o: (o['ta'], o['rung'])), False
    ended = [(req(o), o) for o in cand if MUT_NO_LOOKBACK or req(o) > tb - LOOKBACK]
    if ended:
        _, o = max(ended, key=lambda x: (x[0], x[1]['rung'], x[1]['id']))
        why = 'filled' if o['tfill'] <= o['tc'] else o['why']
        return {'trig': 'PULLED', 'drift': 'MOVED', 'filled': 'USED'}.get(why, 'OTHER'), o, bool(min(o['tr'], o['tfill']) > tb)
    return 'NEVER', None, False


def ahead_at(o, t, lvp, tfl):
    """Rule M's queue ahead of o at t (its own fills and trades ignored): a0 capped by the smallest size shown at its price in the frames
    after its activation frame up to and including the last frame at or before t."""
    a = float(o['a0']) if np.isfinite(o['a0']) else 0.0
    fi = bisect.bisect_right(tfl, o['ta']) - 1; j = bisect.bisect_right(tfl, t) - 1
    if j > fi:
        seg = lvp[fi + 1:j + 1]; seg = seg[np.isfinite(seg)]
        if len(seg):
            a = min(a, float(seg.min()))
    return a


def next_after(orders, pt, tb, tend):
    nx = [o for o in orders if not o['ref'] and o['pt'] == pt and tb < o['ta'] < tend]
    return min(nx, key=lambda o: (o['ta'], o['rung'])) if nx else None


def first_trade(L, pt, side, tb, tend, TP):
    """L: the window-side's legs sorted by anchor (arrays anc, p, q, top, exact, tko). The first taker order with a leg at pt in
    [tb - PRE_TRADE, tend)."""
    idx = np.flatnonzero((L['p'] == pt) & (L['anc'] >= tb - PRE_TRADE) & (L['anc'] < tend))
    if not len(idx):
        return dict(st='none')
    i0 = int(idx[0])
    if not L['exact'][i0]:
        return dict(st='first_not_exact')
    tk = L['tko'][i0]; sel = idx[L['tko'][idx] == tk]
    q = L['q'][sel]; tp_ = L['top'][sel]
    bid_side = (side == 0) != bool(MUT_CROSS_SIDE)
    crossing = any((x > pt) if bid_side else (x < pt) for x in TP[tk])
    return dict(st='ok', dt=float(L['anc'][i0] - tb), crossing=bool(crossing), small=bool(q.sum() <= SMALL), q=float(q.sum()),
                qtop=float(q[tp_].sum()))


# ---------------------------------------------------------------- self-tests
def _o(rung, oid, pt, ta, tc=INF, tr=INF, tfill=INF, why='', a0=10.0):
    return dict(rung=rung, id=oid, pt=pt, ta=ta, tc=tc, tr=tr, tfill=tfill, why=why, ref=False, a0=a0)


def selftest():
    bad = []

    def chk(nm, c):
        if not c:
            bad.append(nm)
    tb = 10000.0
    cases = [('live', [_o(1, 0, 49, 100.0)], 'LIVE', False),
             ('pulled', [_o(1, 0, 49, 100.0, tc=9600.0, tr=9622.0, why='trig')], 'PULLED', False),
             ('pending', [_o(1, 0, 49, 100.0, tc=9990.0, tr=10012.0, why='trig')], 'PULLED', True),
             ('moved', [_o(1, 0, 49, 100.0, tc=9000.0, tr=9022.0, why='drift')], 'MOVED', False),
             ('used', [_o(1, 0, 49, 100.0, tfill=9500.0)], 'USED', False),
             ('old', [_o(1, 0, 49, 100.0, tc=4000.0, tr=4022.0, why='trig')], 'NEVER', False),
             ('other price', [_o(2, 0, 48, 100.0)], 'NEVER', False),
             ('later', [_o(1, 0, 49, 10050.0)], 'NEVER', False),
             ('latest wins', [_o(1, 0, 49, 100.0, tc=8000.0, tr=8022.0, why='drift'), _o(0, 0, 49, 8500.0, tc=9900.0, tr=9922.0, why='trig')],
              'PULLED', False)]
    for nm, os_, want, wp in cases:
        s_, _, p_ = status_at(os_, 49, tb)
        chk(f'status {nm}: {s_} {p_}', s_ == want and p_ == wp)
    tfl = [0.0, 100.0, 200.0, 300.0, 400.0]; lvp = np.array([10.0, 12.0, 8.0, 9.0, 3.0])
    o = _o(1, 0, 49, 50.0, a0=10.0)
    chk('ahead to 300', ahead_at(o, 300.0, lvp, tfl) == 8.0)
    chk('ahead to 150', ahead_at(o, 150.0, lvp, tfl) == 10.0)
    chk('ahead to 400', ahead_at(o, 400.0, lvp, tfl) == 3.0)
    b0 = np.array([50, 50, 49, 49, 51, 51]); valid = np.ones(6, bool); rs = np.array([0, 0, 2, 2, 4, 4]); kind = np.array([0, 0, 2, 2, 1, 1])
    B = births([0.0, 10.0, 20.0, 30.0, 40.0, 50.0], b0, valid, rs, kind, 0.0, 100.0)
    chk(f'births {B}', [(x[0], x[2], x[3], x[4], x[5]) for x in B] == [(2, 49, 'fallback', 40.0, 1), (4, 51, 'improve', 100.0, 2)])
    L = dict(anc=np.array([900.0, 1000.0, 1000.0, 1200.0, 5000.0]), p=np.array([49, 49, 49, 49, 49]), q=np.array([5.0, 10.0, 15.0, 40.0, 5.0]),
             top=np.array([False, True, False, True, False]), exact=np.array([True, True, True, False, True]), tko=np.array([7, 1, 1, 2, 3]))
    TP = {1: [50, 49], 2: [49], 3: [49], 7: [49]}
    ft = first_trade(L, 49, 0, 1200.0, 4000.0, TP)
    chk(f'first trade {ft}', ft.get('st') == 'ok' and ft['crossing'] and ft['small'] and ft['q'] == 25.0 and ft['qtop'] == 10.0)
    ft2 = first_trade(L, 49, 0, 1400.0, 4000.0, TP)
    chk(f'first not exact {ft2}', ft2.get('st') == 'first_not_exact')
    ft3 = first_trade(L, 49, 0, 6000.0, 9000.0, TP)
    chk(f'no trade {ft3}', ft3.get('st') == 'none')
    ft4 = first_trade(L, 49, 1, 1200.0, 4000.0, TP)
    chk(f'ask side not crossing {ft4}', ft4.get('st') == 'ok' and not ft4['crossing'])
    chk('label570 repost', label570({'LIVE': 5, 'PULLED': 6, 'NEVER': 3, 'MOVED': 1})[2] == 'LATE-BY-REPOST')
    chk('label570 never', label570({'LIVE': 5, 'PULLED': 2, 'NEVER': 7})[2] == 'LATE-FROM-FIRST-ARRIVAL')
    chk('label570 mixed', label570({'PULLED': 4, 'NEVER': 4, 'USED': 3})[2] == 'MIXED')
    return bad


def mutation_check():
    global MUT_LANDING_AS_REQ, MUT_NO_LOOKBACK, MUT_ANY_PRICE, MUT_CROSS_SIDE
    out = {}
    for nm in ('MUT_LANDING_AS_REQ', 'MUT_NO_LOOKBACK', 'MUT_ANY_PRICE', 'MUT_CROSS_SIDE'):
        globals()[nm] = 1
        try:
            out[nm] = len(selftest()) > 0
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    bad = selftest()
    if bad:
        sys.exit(f'NEXT-PRICE SELFTEST FAILED: {bad}')
    m = mutation_check()
    if not all(m.values()):
        sys.exit(f'NEXT-PRICE PLANTED ERROR NOT CAUGHT: {m}')
    QP.all_selftests()
    print(f'selftests passed; planted errors caught: {m}', flush=True)
    return m


def label570(n):
    """n: counts by status. Among births where we were not LIVE: (PULLED + MOVED) share, NEVER share, label."""
    nl = sum(v for k, v in n.items() if k != 'LIVE')
    if nl == 0:
        return NAN, NAN, 'NOT-SHOWN'
    rp = (n.get('PULLED', 0) + n.get('MOVED', 0)) / nl; nv = n.get('NEVER', 0) / nl
    return rp, nv, ('LATE-BY-REPOST' if rp >= 0.5 else ('LATE-FROM-FIRST-ARRIVAL' if nv >= 0.5 else 'MIXED'))


# ---------------------------------------------------------------- one day
def day_run(day):
    import pf_gap_top_makers as GP
    mut = all_selftests()
    T0 = time.time(); D = K.load_day(day); st = D['stat']; S1d = f'{K.IO}/step1/{day}'
    full = pd.read_parquet(f'{S1d}/legs_timed2.parquet', columns=['mid', 'signer', 'shares', 't_rel', 'side', 'ptick'] + QP.OLDKEY[1:] + ['how', 't_ms'])
    inw = full.mid.isin(list(D['W0'])).values
    G = GP.rank_groups(full.signer.values[inw], full.shares.values[inw].astype(float), full.t_rel.values[inw].astype(float))
    MP = pd.read_parquet(f'{QP.ORD}/map_{day}.parquet').set_index('row')
    F = full[inw].join(MP, how='left')
    F['comp'] = F.groupby(QP.OLDKEY, sort=False).tko_ok.transform('all').values
    wsT = np.where((F.how.values > 0) & F.tko_ok.values, F.t_ms.values.astype(float), np.nan)
    F['tws'] = pd.Series(wsT, index=F.index).groupby(F.tko.values).transform('min').values
    T = F[F.comp.values & np.isfinite(F.tws.values)]; TP = T.groupby('tko').ptick.agg(list).to_dict()
    exact = set(T.index.tolist())
    st.update(window_sides=0, sim_differ=0, births=0, base_q=0.0, base_qtop=0.0)
    ours = []; brow = []
    for w in K.window_sides(D):
        st['window_sides'] += 1; m, side = w['m'], w['side']
        tf, _ = w['tf_mff']; tfl = tf.tolist(); P5, S5, opp = w['P5'], w['S5'], w['opp']; tgl = w['tg'].tolist()
        b0, valid, rs, kind = QR.runs(tf, P5, opp, side)
        B = births(tfl, b0, valid, rs, kind, w['ws'], w['te']); st['births'] += len(B)
        if (m, side) in D['LI']:
            l = D['LT'].iloc[D['LI'][(m, side)]]
            o_ = np.lexsort(((-l.ptick.values if side == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
            Fi = F.reindex(lrow); sig = Fi.signer.values
            L = dict(anc=l.anc.values[o_].astype(float), p=l.ptick.values[o_].astype('int64'), q=l.shares.values[o_].astype(float),
                     top=np.array([bool(np.isfinite(s_) and G.get(int(s_), 'rest') == 'top 5') for s_ in sig], bool),
                     exact=np.array([r_ in exact for r_ in lrow], bool), tko=Fi.tko.values)
            for i, r_ in enumerate(lrow):                 # baseline: exactly timed at-best maker volume in the minute (BEST-GAP's real rows)
                if not L['exact'][i]:
                    continue
                t = float(F.tws.at[r_]); p_ = int(L['p'][i])
                if w['ws'] <= t < w['ws'] + 60000.0 and GP.depth_of(tfl, P5, opp, side, t, p_) == 0:
                    st['base_q'] += float(L['q'][i]); st['base_qtop'] += float(L['q'][i]) * bool(L['top'][i])
        else:
            L = dict(anc=np.zeros(0), p=np.zeros(0, 'int64'), q=np.zeros(0), top=np.zeros(0, bool), exact=np.zeros(0, bool), tko=np.zeros(0))
        for (j0, tb, pt, kd, tend, ticks) in B:
            ft = first_trade(L, pt, side, tb, tend, TP)
            brow.append((int(day), m, side, kd, tb - w['ws'], pt, ticks, tend - tb, ft['st'], ft.get('dt', NAN), ft.get('crossing', False),
                         ft.get('small', False), ft.get('q', NAN), ft.get('qtop', NAN)))
        vg = (np.asarray(w['vkl'], float) <= K.GUARD_PULL).tolist()
        cache = w.setdefault('cache', {})
        for q, shf in CFGS:
            tt = w['anc'] - shf; cfg = f'{q}{int(shf)}'; O = []
            for d in RUNGS:
                Fo, Ot, Od, _ = QP.sim4(side, tf, opp, P5, S5, cache, w['tg'], w['vkl'], -INF, w['best'], w['bopp'], tt, w['ptl'], w['shl'], q,
                                        w['te'], vg, d, SIZE)
                Fk, Ok, _ = K.sim_arm(w, q, shf, 'rest', aux=vg, offset=d, size=SIZE)
                st['sim_differ'] += int(Fk != Fo or Ok != Ot)
                for o in Od:
                    o['rung'] = d
                O.extend(Od)
            for (j0, tb, pt, kd, tend, ticks) in B:
                k = bisect.bisect_right(tgl, tb) - 1; vv = float(w['vkl'][k]) if k >= 0 else NAN
                gs = 'pre_grid' if k < 0 else ('masked' if vv == MASK_V else ('on' if vv <= K.GUARD_PULL else 'off'))
                s_, o, pend = status_at(O, pt, tb)
                lvp = cache.get(pt)
                if lvp is None:
                    lvp = QR.lvarr(P5, S5, pt, side); cache[pt] = lvp
                shown = float(lvp[j0]) if np.isfinite(lvp[j0]) else NAN
                ah = ahead_at(o, tb, lvp, tfl) if (o is not None and s_ in ('LIVE', 'PULLED')) else NAN
                age = tb - o['ta'] if o is not None else NAN
                since = tb - min(o['tc'], o['tfill']) if (o is not None and s_ != 'LIVE') else NAN
                stay = why_end = None
                if s_ == 'LIVE':
                    stay = min(o['tc'], o['tfill']) - tb; why_end = 'filled' if o['tfill'] <= o['tc'] else o['why']
                nx = next_after(O, pt, tb, tend)
                ours.append((int(day), cfg, m, side, kd, tb - w['ws'], pt, ticks, gs, s_, bool(pend), o['rung'] if o is not None else -1, age,
                             since, ah, shown, stay if stay is not None else NAN, why_end or '', (nx['ta'] - tb) if nx else NAN,
                             float(nx['a0']) if nx else NAN))
        w.pop('cache', None)
        if st['window_sides'] % 80 == 0:
            r2 = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r2:.0f} MiB', flush=True)
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    if st['sim_differ'] or not st['window_sides']:
        sys.exit(f'NEXT-PRICE CONTROL FAILED on {day} -> no result: {st}')
    st['planted_caught'] = mut
    os.makedirs(OUTD, exist_ok=True)
    pd.DataFrame(ours, columns=['day', 'cfg', 'm', 'side', 'kind', 'tb_rel', 'pt', 'ticks', 'guard', 'status', 'pending', 'rung', 'age', 'since',
                                'ahead', 'shown', 'stay', 'why_end', 'delay', 'next_a0']).to_parquet(f'{OUTD}/ours_{day}.parquet', index=False)
    pd.DataFrame(brow, columns=['day', 'm', 'side', 'kind', 'tb_rel', 'pt', 'ticks', 'run_ms', 'ft', 'dt', 'crossing', 'small', 'q', 'qtop']
                 ).to_parquet(f'{OUTD}/births_{day}.parquet', index=False)
    json.dump(dict(stat=st), open(f'{OUTD}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: {st} in {time.time() - T0:.0f} s', flush=True)


# ---------------------------------------------------------------- report
def report():
    stats = {d: json.load(open(f'{OUTD}/day_{d}.json'))['stat'] for d in DAYS9}
    O = pd.concat([pd.read_parquet(f'{OUTD}/ours_{d}.parquet') for d in DAYS9], ignore_index=True)
    B = pd.concat([pd.read_parquet(f'{OUTD}/births_{d}.parquet') for d in DAYS9], ignore_index=True)
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    med = lambda x: float(np.nanmedian(x)) if np.isfinite(np.asarray(x, float)).any() else NAN
    q3 = lambda x: tuple(float(v) for v in np.nanpercentile(x, [25, 50, 75])) if np.isfinite(np.asarray(x, float)).any() else (NAN, NAN, NAN)
    R = {}
    L = [f'NEXT-PRICE, 9 consumed replay days, first 60 s, guard ladder (LDG, 6 rungs x 30 sh); as of {asof}; controls passed; births '
         f'{len(B)} (fallback {int((B.kind == "fallback").sum())}, improve {int((B.kind == "improve").sum())}); window-sides '
         f'{sum(s["window_sides"] for s in stats.values())}', '']
    L.append('OUR STATUS WHEN A PRICE BECOMES OUR SIDE\'S BEST (share of births; guard masked / no decision step excluded)')
    for cfg in ('M50', 'M0', 'M250'):
        for kd in ('fallback', 'improve'):
            X = O[(O.cfg == cfg) & (O.kind == kd)]; ex = X.guard.isin(['masked', 'pre_grid']); Y = X[~ex]
            n = Y.status.value_counts().to_dict()
            parts = ' | '.join(f'{s_} {n.get(s_, 0) / max(len(Y), 1):5.1%}' for s_ in STATUSES)
            pend = float(Y[Y.status == 'PULLED'].pending.mean()) if (Y.status == 'PULLED').any() else NAN
            nvg = Y[Y.status == 'NEVER'].guard.value_counts(normalize=True).to_dict()
            L.append(f'  {cfg:<4} {kd:<8} n {len(Y):>6} (excluded {int(ex.sum())}): {parts}; PULLED still pending at the birth {pend:.0%};'
                     f' NEVER with the guard on {nvg.get("on", 0):.0%}')
    X = O[(O.cfg == 'M50') & (O.kind == 'fallback') & ~O.guard.isin(['masked', 'pre_grid'])]
    rp, nv, lab = label570(X.status.value_counts().to_dict())
    per = {}
    for d in DAYS9:
        y = X[X.day == int(d)]; per[d] = label570(y.status.value_counts().to_dict())
    same = sum(per[d][2] == lab for d in DAYS9)
    R['R570'] = dict(repost=rp, never=nv, label=lab, stable=('STABLE' if same >= 7 else 'UNSTABLE'), days_same=same,
                     per_day={d: list(per[d]) for d in DAYS9})
    L.append(f'  R570 (fallback, M 50): of the births where we were not LIVE, PULLED + MOVED {rp:.1%}, NEVER {nv:.1%} -> {lab}, '
             f'{R["R570"]["stable"]} ({same} of 9 days the same)')
    L.append('    per day (repost / never): ' + ', '.join(f'{d[4:]} {per[d][0]:.0%} / {per[d][1]:.0%}' for d in DAYS9))
    for tk_, nm in ((1, '1 tick'), (2, '>= 2 ticks')):
        y = X[(X.ticks == 1) if tk_ == 1 else (X.ticks >= 2)]; n_ = y.status.value_counts(normalize=True).to_dict()
        L.append(f'    fallback by {nm:<10} n {len(y):>6}: ' + ' | '.join(f'{s_} {n_.get(s_, 0):5.1%}' for s_ in STATUSES))
    lv = X[X.status == 'LIVE']; pl = X[X.status == 'PULLED']
    L.append(f'    rungs of LIVE orders: ' + ', '.join(f'{r} {v:.0%}' for r, v in lv.rung.value_counts(normalize=True).sort_index().items()) +
             '; of PULLED: ' + ', '.join(f'{r} {v:.0%}' for r, v in pl.rung.value_counts(normalize=True).sort_index().items()))
    a_ = q3(pl.since.values)
    L.append(f'    PULLED: our cancel was decided {a_[1]:.0f} ms before the birth (p25 {a_[0]:.0f} / p75 {a_[2]:.0f})')
    if len(lv):
        L.append(f'    LIVE orders after the birth: still there 250 ms later {float((lv.stay > 250).mean()):.0%}, 1 s later {float((lv.stay > 1000).mean()):.0%};'
                 f' ended within 1 s by the guard {float(((lv.stay <= 1000) & (lv.why_end == "trig")).mean()):.0%}, by fills '
                 f'{float(((lv.stay <= 1000) & (lv.why_end == "filled")).mean()):.0%}; age at the birth median {med(lv.age.values):.0f} ms')
    nl = X[X.status != 'LIVE']; dl = q3(nl.delay.values)
    L.append(f'    not LIVE: our next order at that price goes live {dl[1]:.0f} ms after the birth (p25 {dl[0]:.0f} / p75 {dl[2]:.0f});'
             f' none while the level lasts {float(nl.delay.isna().mean()):.0%}')
    L.append('')
    L.append('QUEUE AHEAD AT THE BIRTH (rule M arithmetic, sh; median (p25 / p75)) - fallback, M 50')
    nxt = X.next_a0.values; kept = pl.ahead.values; lva = lv.ahead.values
    for nm, v_ in (('LIVE orders, at the birth', lva), ('PULLED orders had they been kept', kept), ('our next order after the birth, at activation', nxt),
                   ('size shown at the price in the birth frame', X.shown.values)):
        a_ = q3(v_); L.append(f'  {nm:<46} {a_[1]:7.1f} ({a_[0]:.1f} / {a_[2]:.1f})  n {int(np.isfinite(np.asarray(v_, float)).sum())}')
    ok_p = med(lva) <= 0.5 * med(nxt) and med(kept) <= 0.5 * med(nxt) if np.isfinite(med(nxt)) else False
    dd = 0
    for d in DAYS9:
        y = X[X.day == int(d)]; a1 = med(y[y.status == 'LIVE'].ahead.values); a2 = med(y[y.status == 'PULLED'].ahead.values); a3 = med(y.next_a0.values)
        dd += int(np.isfinite(a1) and np.isfinite(a2) and np.isfinite(a3) and a1 <= 0.5 * a3 and a2 <= 0.5 * a3)
    R['R571'] = dict(live=med(lva), kept=med(kept), next=med(nxt), days=dd, label='EARLY-IS-AHEAD' if (ok_p and dd >= 7) else 'NOT-SHOWN')
    L.append(f'  R571: LIVE {med(lva):.1f} / kept {med(kept):.1f} vs next {med(nxt):.1f} sh; both <= half on {dd} of 9 days -> {R["R571"]["label"]}')
    L.append('')
    L.append('THE TOP MAKERS AT THE SAME BIRTHS (real fills; first taker order with a leg at the new best)')
    bq = sum(s['base_q'] for s in stats.values()); bqt = sum(s['base_qtop'] for s in stats.values()); base = bqt / bq if bq else NAN
    L.append(f'  baseline: top 5 share of all exactly timed at-best maker volume in the minute {base:.1%}')
    for kd in ('fallback', 'improve'):
        Y = B[B.kind == kd]; vc = Y.ft.value_counts().to_dict(); ok = Y[Y.ft == 'ok']
        L.append(f'  {kd:<8} births {len(Y)}: first trade exact {vc.get("ok", 0)}, not exact {vc.get("first_not_exact", 0)}, none {vc.get("none", 0)}')
        for nm, z in (('all first trades', ok), ('SMALL (<= 30 sh at the price)', ok[ok.small]), ('CROSSING', ok[ok.crossing])):
            if len(z):
                L.append(f'    {nm:<30} n {len(z):>6}: a top-5 maker filled {float((z.qtop > 0).mean()):5.1%}; top-5 volume share '
                         f'{float(z.qtop.sum() / z.q.sum()):5.1%} (x{float(z.qtop.sum() / z.q.sum()) / base:.2f} their baseline); median'
                         f' {float(z.dt.median()):.0f} ms after the birth')
    Y = B[(B.kind == 'fallback') & (B.ft == 'ok') & B.small]
    ratio = float(Y.qtop.sum() / Y.q.sum()) / base if len(Y) and base > 0 else NAN
    up = 0; rr = {}
    for d in DAYS9:
        y = Y[Y.day == int(d)]; s0 = stats[d]
        b_ = s0['base_qtop'] / s0['base_q'] if s0['base_q'] else NAN
        rr[d] = float(y.qtop.sum() / y.q.sum()) / b_ if len(y) and y.q.sum() > 0 and b_ > 0 else NAN
        up += int(np.isfinite(rr[d]) and rr[d] > 1)
    R['R572'] = dict(ratio=ratio, days_up=up, per_day=rr, label='THEY-HOLD-THE-FRONT' if (np.isfinite(ratio) and ratio >= 1.2 and up >= 7) else 'NOT-SHOWN')
    L.append(f'  R572: small first trades at the new best (fallback): top-5 volume share x{ratio:.2f} their baseline, above it on {up} of 9 days -> '
             f'{R["R572"]["label"]}')
    L.append('    per day: ' + ', '.join(f'{d[4:]} x{rr[d]:.2f}' for d in DAYS9))
    L.append('')
    L.append('READINGS: ' + json.dumps({k: R[k]['label'] for k in ('R570', 'R571', 'R572')}))
    txt = '\n'.join(L) + '\n'
    open(f'{OUTD}/next_price.txt', 'w').write(txt)
    json.dump(dict(asof=asof, readings=R, stats=stats), open(f'{OUTD}/next_price.json', 'w'), indent=1, default=float)
    print(txt)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        all_selftests()
    elif cmd == 'day':
        day_run(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit('usage: pf_next_price.py selftest | day <day> | report')
