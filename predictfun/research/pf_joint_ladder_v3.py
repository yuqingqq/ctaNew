"""JOINT-LADDER v3 (2026-11-14, session ce68ebe0). Repairs v2 (pf_joint_ladder_v2.py, 3040edf2; its run was stopped and its partial outputs
deleted) after the USER's review: "it releases capacity using a rejection that has not happened yet ... Whether it will be rejected at 40 ms
changes whether another order is submitted at time zero. Both runs have identical information available at submission ... The fix is to
reserve capacity from submission until the acceptance/rejection becomes known, keeping client-known state separate from venue execution state.
Add a test requiring identical decisions when only future events change."

CLIENT-KNOWN STATE (all of the lane's decisions use it; the venue state only executes): an order of ours counts as LIVE to the client from its
submission until (a) its rejection is known (at its activation time ta: the lane's model of when a post-only refusal is seen), (b) its full
fill is known (every fill is known L_NOTE = L_FILL = 100 ms after it), or (c) its cancel is confirmed (the cancel's landing + L_NOTE). On that
basis:
  - the hard cap: side exposure = known position in the side's direction + sum over our LIVE orders of (size - their fills known by t) + the
    new size <= CAP = K x size (v2 dropped a submitted order the moment the venue would refuse it, 40 ms in the future: the reported bug);
  - "our order rests" (the trigger's entry line and the decision callback): LIVE, activated, no cancel requested (v1 / v2 used the venue's
    remaining size, so a fill not yet known changed decisions up to 100 ms early);
  - "a price we hold" (no two of our orders at one price): any LIVE order at that price (v1 / v2: venue remaining and cancel landing).
The soft cap already used the known position. Fill execution, queues and refusals stay the venue's (the S4 rest lane's model).
A 'compat' mode keeps v1's venue-state decisions only to show the refactor reproduces v1 (regression control).
DECISION LOG: every place / cancel / cap refusal / held-price wait with its time, side, rung and price.
Arms (as v1 / v2): LDG, FB25, OFB2 (the 2 s oracle: it reads the future by design and is excluded from the invariance falsifier). Caps: hard 2 /
5 / 10 x size, soft 2 x size, none. 9 consumed days, first 60 s, M 0 / 50 / 250. Values and inventory statistics as v2.
Readings DECLARED BEFORE THE FIRST RUN (WS $/day w/o young fills at 1 s; day rule >= 7 of 9):
 R613 KEEP-UNDER-HARD-CAP (FB25 vs LDG and OFB2 vs LDG at hard 2 / 5 / 10 x size): higher at M 0 AND M 50 on >= 7 days -> PAYS; lower at both on
      >= 7 -> DOES-NOT-PAY; else NOT-SHOWN; M 250 and the sign printed.
 R615 CLIENT-VS-VENUE (printed, no label): LDG / FB25 at soft 2 x size and none, client-known decisions minus v1's venue-state decisions, per
      clock: what v1's decision leak was worth.
Controls (any failure -> no output): (i) DECISION INVARIANCE on real data (10-23, 10-31, 11-03; cuts at 20 s and 40 s into each window; LDG and
FB25; hard 2 x size and none; M 50): every venue event after the cut is changed (frames stamped after it: prices one tick up on both sides,
sizes x 2 + 7; trades after it doubled; each step's known best recomputed from the changed frames) and the decision logs up to the cut must be
identical; (ii) synthetic invariance cases - the USER's (a 30-sh order submitted at 0 refused or accepted at 40 ms under a 30-sh hard cap:
the decisions at 0 must not differ), a fill not yet known (decisions before its notice must not differ), a fill not yet known at a price
another rung wants - with planted errors (the refusal releases capacity at submission; "rests" on the venue's remaining; the held price on
the venue's state) that must each break a case; (iii) conservation / accounting checks on every run (v2's: leg, order, live, priority,
collision, position, cap); (iv) regression: compat mode soft-2 and no-cap fills equal v1's row for row (every arm, clock, day); compat mode
with INDEP equals pf_kp_s4.sim_arm per rung (v1's parity; 3 days); v2's synthetic cases; memory self-guard.
Limits: replay rule M (B1a); the lane's model of when a refusal is known (at ta) and the 100 ms notice for fills and cancel confirmations are
fixed choices; upstream inputs (the guard, P) are taken as known at their steps; consumed days (C8); the guard ladder, not the frozen v3 (C11).
Usage (run_light, PF_SLOT=claude): pf_joint_ladder_v3.py selftest | invariance <day> | parity <day> | eval <day> | report; driver
run_joint_ladder_v3.sh
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
import pf_qr_queue_race as QR                             # noqa: E402
import pf_timing_split as TS                              # noqa: E402
import pf_pull_depth as PD                                # noqa: E402
import pf_future_best as FB                               # noqa: E402
import pf_joint_ladder as JL1                             # noqa: E402  (v1: synthetic contexts, its saved fills for the regression control)

DAYS9 = K.DAYS9
OUTD = f'{K.IO}/joint_ladder_v3_20261114'
SIZE = 30.0; NR = 6
L_NOTE = K.L_FILL
CAPS = (('hard', 2), ('hard', 5), ('hard', 10), ('soft', 2), ('none', None))
COMPAT_CAPS = (('soft', 2), ('none', None))
ARMS = ('LDG', 'FB25', 'OFB2')
CFGS = [('M', 0.0), ('M', 50.0), ('M', 250.0)]
AGEB = FB.AGEB
VIOL = ('leg', 'order', 'live', 'priority', 'collision', 'position', 'cap')
INV_DAYS = ('20261023', '20261031', '20261103'); INV_CUTS = (20000.0, 40000.0)
INF = float('inf'); NAN = float('nan')
MUT_REF_RELEASE = MUT_REST_TRUE = MUT_HELD_TRUE = 0       # planted errors (self-test only): v2's decision leaks
MUT_LEG_LEAK = MUT_PRIORITY = 0                           # planted errors (self-test only): the conservation checks must fire
INDEP = 0                                                 # control only (compat mode): each order the whole leg, no held-price wait
MEM_CAP_MIB = 2048.0


def cap_name(mode, k):
    return 'none' if mode == 'none' else f'{mode}{k}'


# ---------------------------------------------------------------- the joint lane
def sim_joint(W, q, shf, cap, aux, decide=None, idle_only=0, size=SIZE, nr=NR, mode='soft', state='client'):
    """Both sides of one window. mode: 'soft' / 'hard' / 'none' (cap); state: 'client' (decisions on client-known state, the header) or
    'compat' (v1's venue-state decisions; regression control only). Returns per (s, d) fills / orders (sim2 tuple form), the decision log,
    stats incl. conservation violations."""
    if mode == 'none':
        cap = None
    client = state == 'client'
    taul = W[0]['tg'].tolist(); tfl = W[0]['tf'].tolist(); nf = len(tfl); t_end = W[0]['te']
    TL = []
    for s in (0, 1):
        tt = (W[s]['anc'] - shf).tolist()
        TL += [(tt[i], s, i) for i in range(len(tt))]
    TL.sort(key=lambda x: (x[0], x[1], x[2]))
    ST = {}
    for s in (0, 1):
        w = W[s]
        ST[s] = dict(P5=w['P5'], S5=w['S5'], opp=w['opp'], best=w['best'], bopp=w['bopp'], v=w['vkl'], ptl=w['ptl'], shl=w['shl'],
                     cache=w.setdefault('cache', {}), live=[], cl=[], cur=[None] * nr, lastpull=[-INF] * nr, fills=[[] for _ in range(nr)],
                     orders=[[] for _ in range(nr)], aux=aux[s], dec=[(decide.get((s, d)) if decide else None) for d in range(nr)])
    stat = dict(pulls=[0, 0], cap_pulls=[0, 0], cap_steps=0, skip=[0, 0], inter=[0, 0], maxpos=0.0, hard_block=[0, 0],
                viol={v_: 0 for v_ in VIOL})
    pos = [0.0]; KN = []; fl = []; log = []

    def kfill(o, t):
        return sum(q_ for (tq_, q_) in o['fl'] if tq_ + L_NOTE <= t)

    def live_c(o, t):                                     # LIVE to the client at t (header)
        if o['ts'] > t:
            return False
        if o['ref'] and o['ta'] <= t:
            return False
        if o['ref'] and MUT_REF_RELEASE:
            return False
        if kfill(o, t) >= size - 1e-9:
            return False
        if o['tc'] < INF and o['tr'] + L_NOTE <= t:
            return False
        return True

    def lv(s, pt):
        c = ST[s]['cache']
        if pt not in c:
            P5, S5 = ST[s]['P5'], ST[s]['S5']; last = P5[:, 4]
            c[pt] = np.where((last >= 0) & ((pt < last) if s == 0 else (pt > last)), np.nan, (S5 * (P5 == pt)).sum(1))
        return c[pt]

    def place(s, d, pt, t):
        ta = t + K.L_PLACE; fi = bisect.bisect_right(tfl, ta) - 1; opp = ST[s]['opp']
        o = dict(pt=pt, ts=t, ta=ta, tr=INF, tc=INF, rem=size, ahead=0.0, ci=fi, tfill=INF, a0=np.nan, ref=False, rung=d, fl=[], side=s)
        if fi < 0 or ((0 <= opp[fi] <= pt) if s == 0 else (opp[fi] >= pt)):
            o['ref'] = True; o['tr'] = ta                 # the venue's refusal at ta (the client learns it at ta)
        else:
            a = lv(s, pt)[fi]; a = float(a) if np.isfinite(a) else 0.0
            if fi + 1 < nf and tfl[fi + 1] - ta <= 100.0:
                d2 = lv(s, pt)[fi + 1]
                if np.isfinite(d2) and d2 > a:
                    a = float(d2)
            o['a0'] = a; o['ahead'] = 0.0 if q == 'U' else a
            if not INDEP and any(x['pt'] == pt and x['rem'] > 1e-12 and x['tr'] > ta for x in ST[s]['live']):
                stat['viol']['collision'] += 1            # venue: two of our orders resting at one price at once
            ST[s]['live'].append(o)
        ST[s]['orders'][d].append(o); ST[s]['cl'].append(o)
        log.append((t, s, d, 'place', pt))
        return o

    def cancel(o, t, why):
        if o['tc'] == INF:
            log.append((t, o['side'], o['rung'], f'cancel:{why}', o['pt']))
        o['tc'] = min(o['tc'], t)
        o['tr'] = min(o['tr'], max(t + K.L_CANCEL, o['ts'] + K.L_PENDCANCEL) if t < o['ta'] else t + K.L_CANCEL)

    def exposure(s, t, kpos):
        e = kpos if s == 0 else -kpos
        for o in ST[s]['cl']:
            if live_c(o, t):
                e += size - kfill(o, t)
        return e

    def apply(t, s, i):
        S_ = ST[s]; p = S_['ptl'][i]; left = S_['shl'][i]; full = left; got = 0.0
        el = [o for o in S_['live'] if o['rem'] > 1e-12 and o['ta'] <= t < o['tr']]
        el.sort(key=lambda o: ((-o['pt']) if s == 0 else o['pt'], o['ta']))
        reach = [o for o in el if o['pt'] == p or ((p < o['pt']) if s == 0 else (p > o['pt']))]
        if len(reach) >= 2:
            stat['inter'][s] += 1
        if MUT_PRIORITY:
            el = el[::-1]
        filled = []
        for o in el:
            if q == 'M':
                j = bisect.bisect_left(tfl, t) - 1
                if j > o['ci']:
                    seg = lv(s, o['pt'])[o['ci'] + 1:j + 1]; seg = seg[np.isfinite(seg)]
                    if len(seg) and seg.min() < o['ahead']:
                        o['ahead'] = float(seg.min())
                    o['ci'] = j
            thr = (p < o['pt']) if s == 0 else (p > o['pt'])
            lft = full if (INDEP or MUT_LEG_LEAK) else left
            if p == o['pt']:
                use = min(lft, o['ahead']); o['ahead'] -= use; qq = min(lft - use, o['rem'])
            elif thr:
                qq = min(lft, o['rem']); o['ahead'] = 0.0
            else:
                qq = 0.0
            if qq > 1e-12:
                if not (o['ta'] <= t < o['tr']):
                    stat['viol']['live'] += 1
                left -= qq; o['rem'] -= qq; got += qq; filled.append(o)
                g = bisect.bisect_right(taul, t) - 1
                ph = 'pend' if t >= o['tc'] else ('above' if g >= 0 and S_['v'][g] > -INF else 'hold')
                S_['fills'][o['rung']].append((i, o['pt'], qq, thr, t - o['ta'], ph, o['a0']))
                o['fl'].append((t, qq))
                sq = qq if s == 0 else -qq
                pos[0] += sq; KN.append((t + L_NOTE, sq)); fl.append((s, o['rung'], t, o['pt'], qq))
                stat['maxpos'] = max(stat['maxpos'], abs(pos[0]))
                if cap is not None and mode == 'hard' and abs(pos[0]) > cap + 1e-9:
                    stat['viol']['cap'] += 1
                if o['rem'] <= 1e-12:
                    o['tfill'] = t
        if not INDEP and got > full + 1e-9:
            stat['viol']['leg'] += 1
        if not INDEP:
            for o in filled:
                for a in reach:
                    if a is not o and ((a['pt'] > o['pt']) if s == 0 else (a['pt'] < o['pt'])) and a['rem'] > 1e-12:
                        stat['viol']['priority'] += 1

    it = 0; ik = 0; kpos = 0.0
    for k, t in enumerate(taul):
        while it < len(TL) and TL[it][0] <= t:
            apply(*TL[it]); it += 1
        while ik < len(KN) and KN[ik][0] <= t:
            kpos += KN[ik][1]; ik += 1
        capside = -1
        if mode == 'soft' and cap is not None and abs(kpos) >= cap - 1e-9:
            capside = 0 if kpos > 0 else 1
            stat['cap_steps'] += 1
        for s in (0, 1):
            S_ = ST[s]; b = S_['best'][k]
            for d in range(nr):
                cur = S_['cur'][d]
                if cur is not None:
                    if client:
                        done = (cur['ref'] and cur['ta'] <= t) or kfill(cur, t) >= size - 1e-9
                    else:
                        done = (cur['ref'] and cur['ta'] <= t) or (cur['rem'] <= 1e-12 and cur['tfill'] + K.L_FILL <= t)
                    if done:
                        cur = None
                trig = bool(S_['aux'][k])
                if client and not MUT_REST_TRUE:
                    rest_ = cur is not None and live_c(cur, t) and cur['ta'] <= t and cur['tc'] > t
                else:
                    rest_ = cur is not None and not cur['ref'] and cur['ta'] <= t and cur['rem'] > 1e-12
                if idle_only and rest_:
                    trig = False
                if S_['dec'][d] is not None and rest_ and not trig:
                    trig = bool(S_['dec'][d](k, t, cur))
                if capside == s:
                    if rest_ and not trig:
                        stat['cap_pulls'][s] += 1
                    trig = True
                if trig and rest_:
                    stat['pulls'][s] += 1
                if cur is not None:
                    why = ''
                    if trig:
                        why = 'trig'
                    elif b < 0:
                        why = 'nobook'
                    else:
                        beh = (b - cur['pt']) if s == 0 else (cur['pt'] - b)
                        if beh > K.HOLD_K + d:
                            why = 'drift'
                        elif not ((cur['pt'] < S_['bopp'][k]) if s == 0 else (cur['pt'] > S_['bopp'][k])):
                            why = 'cross'
                    if why:
                        cancel(cur, t, why); cur = None
                if trig:
                    S_['lastpull'][d] = t
                if cur is None and b >= 0 and not trig and t - S_['lastpull'][d] >= K.REST_COOL:
                    tgt = int(b) if d == 0 else ((int(b) - d) if s == 0 else (int(b) + d))
                    if d == 0 or 1 <= tgt <= 99:
                        if client and not MUT_HELD_TRUE:
                            held = (not INDEP) and any(o['pt'] == tgt and live_c(o, t) for o in S_['cl'])
                        else:
                            held = (not INDEP) and any(o['pt'] == tgt and o['rem'] > 1e-12 and o['tr'] > t for o in S_['live'])
                        if held:
                            stat['skip'][s] += 1; log.append((t, s, d, 'held', tgt))
                        elif mode == 'hard' and cap is not None and exposure(s, t, kpos) + size > cap + 1e-9:
                            stat['hard_block'][s] += 1; log.append((t, s, d, 'cap', tgt))
                        else:
                            cur = place(s, d, tgt, t)
                S_['cur'][d] = cur
            lv_ = S_['live']
            if lv_ and (lv_[0]['tr'] <= t or lv_[0]['rem'] <= 1e-12):
                S_['live'] = [o for o in lv_ if o['rem'] > 1e-12 and o['tr'] > t]
            if len(S_['cl']) > 24:
                S_['cl'] = [o for o in S_['cl'] if live_c(o, t)]
    while it < len(TL) and TL[it][0] <= t_end:
        apply(*TL[it]); it += 1
    for s in (0, 1):
        for d in range(nr):
            if ST[s]['cur'][d] is not None:
                cancel(ST[s]['cur'][d], t_end, 'end')
    while it < len(TL) and TL[it][0] <= t_end + 200.0:
        apply(*TL[it]); it += 1
    for s in (0, 1):
        for d in range(nr):
            for o in ST[s]['orders'][d]:
                tot = sum(qq for (_, qq) in o['fl'])
                if tot > size + 1e-9 or abs((size - tot) - o['rem']) > 1e-9:
                    stat['viol']['order'] += 1
    if abs(pos[0] - sum((x[4] if x[0] == 0 else -x[4]) for x in fl)) > 1e-6:
        stat['viol']['position'] += 1
    return dict(fills={(s, d): ST[s]['fills'][d] for s in (0, 1) for d in range(nr)},
                orders={(s, d): [(o['ta'], min(o['tr'], o['tfill'], t_end + 200.0), o['a0'], o['ref']) for o in ST[s]['orders'][d]]
                        for s in (0, 1) for d in range(nr)},
                cancels={(s, d): [o['tc'] for o in ST[s]['orders'][d]] for s in (0, 1) for d in range(nr)},
                log=log, stat=stat, endpos=pos[0], fl=fl)


def log_until(r, T):
    return [x for x in r['log'] if x[0] <= T]


# ---------------------------------------------------------------- self-tests
def selftest():
    bad = []
    _ctx = JL1._ctx
    z20 = {0: [False] * 20, 1: [False] * 20}
    fb = [(0.0, [(50, 100), (49, 100), (48, 100)], 52)]; fa = [(0.0, [(52, 100), (53, 100)], 50)]
    # v2's cases on the client lane
    W = {0: _ctx(0, fb, [(500.0, 47, 40.0)]), 1: _ctx(1, fa, [])}
    r = sim_joint(W, 'U', 0.0, None, z20, nr=2, mode='none')
    if (sum(x[2] for x in r['fills'][(0, 0)]), sum(x[2] for x in r['fills'][(0, 1)])) != (30.0, 10.0) or any(r['stat']['viol'].values()):
        bad.append(f'shared sweep {r["stat"]}')
    W = {0: _ctx(0, fb, [(500.0, 47, 200.0)]), 1: _ctx(1, fa, [])}
    r = sim_joint(W, 'U', 0.0, 60.0, z20, nr=3, mode='hard')
    tot = sum(x[2] for d in range(3) for x in r['fills'][(0, d)])
    if not (len(r['orders'][(0, 2)]) == 0 and tot <= 60.0 + 1e-9 and r['stat']['maxpos'] <= 60.0 + 1e-9 and not any(r['stat']['viol'].values())):
        bad.append(f'hard cap sweep {r["orders"][(0, 2)]} {tot} {r["stat"]["maxpos"]}')
    W = {0: _ctx(0, fb, [(500.0, 50, 30.0), (700.0, 49, 30.0)]), 1: _ctx(1, fa, [])}
    r = sim_joint(W, 'U', 0.0, 30.0, z20, nr=2, mode='hard')
    if not (len(r['orders'][(0, 1)]) == 0 and r['stat']['maxpos'] <= 30.0 + 1e-9 and not any(r['stat']['viol'].values())):
        bad.append(f'hard cap unseen fill {r["orders"][(0, 1)]} {r["stat"]["maxpos"]}')
    # INVARIANCE (I1, the USER's case): a 30-sh hard cap; rung 0 submits 30 sh at 0 ms (live at 40 ms). R: the book moves at 30 ms so the venue
    # refuses it at 40 ms; A: it is accepted. Decisions up to 29 ms must be identical (rung 1 refused by the cap at 0 ms in both).
    fbA = [(0.0, [(50, 100), (49, 100)], 52)]; faA = [(0.0, [(52, 100)], 50)]
    fbR = fbA + [(30.0, [(49, 100), (48, 100)], 50)]; faR = faA + [(30.0, [(50, 100)], 49)]
    rA = sim_joint({0: _ctx(0, fbA, []), 1: _ctx(1, faA, [])}, 'U', 0.0, 30.0, z20, nr=2, mode='hard')
    rR = sim_joint({0: _ctx(0, fbR, []), 1: _ctx(1, faR, [])}, 'U', 0.0, 30.0, z20, nr=2, mode='hard')
    if log_until(rA, 29.0) != log_until(rR, 29.0):
        bad.append(f'I1 refusal leaks into the decision at 0 ms: {log_until(rA, 29.0)} vs {log_until(rR, 29.0)}')
    if not any(x[3] == 'cap' and x[1] == 0 and x[2] == 1 and x[0] == 0.0 for x in rA['log']):
        bad.append('I1: rung 1 must be refused by the cap at 0 ms')
    # I2: a fill not yet known. Rung 0 (30 sh at 50) fully filled at 500 ms (F) or not (N); the guard is on 500-700 ms with a keep callback and
    # the guard as the entry line. Decisions before the notice (600 ms) must be identical.
    vg = [False] * 10 + [True] * 5 + [False] * 5
    keep = {(0, 0): (lambda k, t, cur: False)}
    rF = sim_joint({0: _ctx(0, fbA, [(500.0, 50, 100.0)]), 1: _ctx(1, faA, [])}, 'U', 0.0, None, {0: vg, 1: [False] * 20}, decide=keep, idle_only=1,
                   nr=1, mode='none')
    rN = sim_joint({0: _ctx(0, fbA, []), 1: _ctx(1, faA, [])}, 'U', 0.0, None, {0: vg, 1: [False] * 20}, decide=keep, idle_only=1, nr=1, mode='none')
    if log_until(rF, 599.0) != log_until(rN, 599.0):
        bad.append(f'I2 an unseen fill changes decisions: {log_until(rF, 599.0)} vs {log_until(rN, 599.0)}')
    # I3: a fill not yet known at a price another rung wants. Rung 1 pulled by its callback at 100 ms; the best moves to 51 at 300 ms, so rung 1
    # wants 50, held by rung 0; rung 0 fully filled at 320 ms (F) or not (N). Decisions before the notice (420 ms) must be identical.
    fb3 = [(0.0, [(50, 100), (49, 100)], 52), (300.0, [(51, 100), (50, 100), (49, 100)], 53)]
    fa3 = [(0.0, [(52, 100)], 50), (300.0, [(53, 100)], 51)]
    pull1 = {(0, 1): (lambda k, t, cur: k == 2)}
    rF = sim_joint({0: _ctx(0, fb3, [(320.0, 50, 100.0)]), 1: _ctx(1, fa3, [])}, 'U', 0.0, None, z20, decide=pull1, nr=2, mode='none')
    rN = sim_joint({0: _ctx(0, fb3, []), 1: _ctx(1, fa3, [])}, 'U', 0.0, None, z20, decide=pull1, nr=2, mode='none')
    if log_until(rF, 419.0) != log_until(rN, 419.0):
        bad.append(f'I3 an unseen fill frees a held price: {log_until(rF, 419.0)} vs {log_until(rN, 419.0)}')
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_REF_RELEASE', 'MUT_REST_TRUE', 'MUT_HELD_TRUE', 'MUT_LEG_LEAK', 'MUT_PRIORITY'):
        globals()[nm] = 1
        try:
            out[nm] = bool(selftest())
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    FB.all_selftests()
    b = selftest(); m = mutation_check()
    if b or not all(m.values()):
        sys.exit(f'JOINT-LADDER v3 SELFTEST FAILED -> no output: {b} planted errors caught {m}')
    print(f'joint-ladder v3 selftests passed; planted errors caught: {m}', flush=True)
    return m


# ---------------------------------------------------------------- window inputs
def window_ctx(W, PO, POI, n):
    ctx = {}
    for s in (0, 1):
        w_ = W[s]; tf, mff = w_['tf_mff']; tfl = tf.tolist(); P5, opp = w_['P5'], w_['opp']; m = W[0]['m']
        Pm = np.full((n, len(PD.JS)), np.nan)
        if (m, s) in POI:
            ii = POI[(m, s)]; Pm[PO.k.values[ii]] = PO[[f'P{j}' for j in PD.JS]].values[ii]
        vg = (np.asarray(w_['vkl'], float) <= K.GUARD_PULL).tolist()
        b0, valid, _, _ = QR.runs(tf, P5, opp, s)
        mv2 = np.full(n, np.nan)
        for k in range(n):
            if vg[k]:
                jk = int(w_['jk'][k])
                if jk >= 0 and valid[jk] and w_['best'][k] >= 0:
                    mv2[k] = PD.move_ticks(b0, valid, tfl, s, jk, float(w_['tg'][k]), 2000.0)
        ctx[s] = dict(Pm=Pm, vg=vg, mv2=mv2, tf=tf, mff=mff, tfl=tfl, sg=w_['sg'])
    return ctx


def arm_dec(arm, W, ctx):
    if arm == 'LDG':
        return None, 0
    return ({(s, d): (FB.fb_dec(ctx[s]['vg'], W[s]['best'], s, ctx[s]['Pm'], d, tau=0.25) if arm == 'FB25' else
                      FB.fb_dec(ctx[s]['vg'], W[s]['best'], s, ctx[s]['Pm'], d, mv=ctx[s]['mv2'])) for s in (0, 1) for d in range(NR)}, 1)


def perturb_after(W, T, shf):
    """Copies of the two contexts with every venue event after T changed (control i): frames stamped after T one tick up on both sides and sizes
    x 2 + 7; trades whose simulator time is after T doubled; each step's known best / other best recomputed from the changed frames by the
    known-time clock (steps at or before T cannot change)."""
    tf = W[0]['tf']; kf = np.asarray(W[0]['kf'], float); fut = tf > T
    P5b = W[0]['P5'].copy(); S5b = W[0]['S5'].copy(); P5a = W[1]['P5'].copy(); S5a = W[1]['S5'].copy()
    for P_, S_ in ((P5b, S5b), (P5a, S5a)):
        m_ = fut[:, None] & (P_ >= 0)
        P_[m_] += 1; S_[m_] = S_[m_] * 2.0 + 7.0
    bb = P5b[:, 0]; ba = P5a[:, 0]
    tg = W[0]['tg']; ki = np.searchsorted(kf, tg, side='right') - 1
    okk = (ki >= 0) & (bb[np.clip(ki, 0, None)] >= 0) & (ba[np.clip(ki, 0, None)] > bb[np.clip(ki, 0, None)])
    kb = np.where(okk, bb[np.clip(ki, 0, None)], -1).tolist(); ka = np.where(okk, ba[np.clip(ki, 0, None)], -1).tolist()
    out = {}
    for s in (0, 1):
        w2 = dict(W[s]); w2['cache'] = {}
        w2['P5'] = P5b if s == 0 else P5a; w2['S5'] = S5b if s == 0 else S5a; w2['opp'] = ba if s == 0 else bb
        w2['best'] = kb if s == 0 else ka; w2['bopp'] = ka if s == 0 else kb
        sh = np.asarray(W[s]['shl'], float); tt = W[s]['anc'] - shf
        w2['shl'] = np.where(tt > T, sh * 2.0, sh).tolist()
        out[s] = w2
    return out


# ---------------------------------------------------------------- controls on real data
def invariance_day(day):
    """Control (i): decision logs up to each cut identical when every venue event after it changes."""
    all_selftests()
    T0 = time.time(); D = K.load_day(day)
    PO = pd.read_parquet(f'{PD.OUTD}/pred_{day}.parquet'); POI = PO.groupby(['m', 'side']).indices
    st = dict(windows=0, checks=0, differ=0, decisions_compared=0); pend = {}; q, shf = 'M', 50.0
    for w in K.window_sides(D):
        pend.setdefault(w['m'], {})[w['side']] = w
        if len(pend[w['m']]) < 2:
            continue
        W = pend.pop(w['m']); st['windows'] += 1; n = W[0]['n']; ctx = window_ctx(W, PO, POI, n)
        for cut in INV_CUTS:
            T = W[0]['ws'] + cut; Wp = perturb_after(W, T, shf)
            for arm in ('LDG', 'FB25'):
                dec, idle = arm_dec(arm, W, ctx)
                for mode, ck in (('hard', 2), ('none', None)):
                    capv = None if ck is None else ck * SIZE
                    r0 = sim_joint(W, q, shf, capv, {0: ctx[0]['vg'], 1: ctx[1]['vg']}, decide=dec, idle_only=idle, mode=mode)
                    r1 = sim_joint(Wp, q, shf, capv, {0: ctx[0]['vg'], 1: ctx[1]['vg']}, decide=dec, idle_only=idle, mode=mode)
                    a, b = log_until(r0, T), log_until(r1, T)
                    st['checks'] += 1; st['decisions_compared'] += len(a); st['differ'] += int(a != b)
        for s in (0, 1):
            W[s].pop('cache', None)
    os.makedirs(OUTD, exist_ok=True)
    json.dump(st, open(f'{OUTD}/invariance_{day}.json', 'w'), indent=1)
    if st['differ'] or not st['checks']:
        sys.exit(f'JOINT-LADDER v3 DECISION-INVARIANCE FAILED on {day}: {st}')
    print(f'invariance {day}: {st} in {time.time() - T0:.0f} s', flush=True)


def parity_day(day):
    """Control (iv): compat mode with INDEP and no cap equals pf_kp_s4.sim_arm per rung (v1's parity)."""
    global INDEP
    all_selftests()
    T0 = time.time(); D = K.load_day(day)
    PO = pd.read_parquet(f'{PD.OUTD}/pred_{day}.parquet'); POI = PO.groupby(['m', 'side']).indices
    st = dict(windows=0, checked=0, differ=0); pend = {}
    INDEP = 1
    try:
        for w in K.window_sides(D):
            pend.setdefault(w['m'], {})[w['side']] = w
            if len(pend[w['m']]) < 2:
                continue
            W = pend.pop(w['m']); st['windows'] += 1; n = W[0]['n']; ctx = window_ctx(W, PO, POI, n)
            for q, shf in CFGS:
                for arm in ARMS:
                    dec, idle = arm_dec(arm, W, ctx)
                    r = sim_joint(W, q, shf, None, {0: ctx[0]['vg'], 1: ctx[1]['vg']}, decide=dec, idle_only=idle, mode='none', state='compat')
                    for s in (0, 1):
                        st['checked'] += 1; bad = 0
                        for d in range(NR):
                            F_, O_, _ = K.sim_arm(W[s], q, shf, 'rest', aux=ctx[s]['vg'], decide=(dec[(s, d)] if dec else None), idle_only=idle,
                                                  offset=d, size=SIZE)
                            bad += int(F_ != r['fills'][(s, d)] or O_ != r['orders'][(s, d)])
                        st['differ'] += int(bad > 0)
            for s in (0, 1):
                W[s].pop('cache', None)
    finally:
        INDEP = 0
    os.makedirs(OUTD, exist_ok=True)
    json.dump(st, open(f'{OUTD}/parity_{day}.json', 'w'), indent=1)
    if st['differ'] or not st['checked']:
        sys.exit(f'JOINT-LADDER v3 PARITY FAILED on {day}: {st}')
    print(f'parity {day}: {st} in {time.time() - T0:.0f} s', flush=True)


# ---------------------------------------------------------------- one day
def eval_day(day):
    import pf_gap_top_makers as GP
    mut = all_selftests()
    T0 = time.time(); D = K.load_day(day); st = D['stat']; S1d = f'{K.IO}/step1/{day}'
    tkf = pd.read_parquet(f'{S1d}/legs_tk.parquet', columns=['tk']).tk.values
    LT = D['LT']; ix = LT.index.values
    grp = np.where(tkf[ix] >= 0, tkf[ix], -1 - ix)
    WS = pd.Series(TS.ws_groups(LT.how.values, grp), index=LT.index)
    PO = pd.read_parquet(f'{PD.OUTD}/pred_{day}.parquet'); POI = PO.groupby(['m', 'side']).indices
    st.update(windows=0, runs=0, viol={v_: 0 for v_ in VIOL}, maxexp_hard={})
    fills = []; inv = []; pend = {}
    runs = [(st_, mo, ck) for (mo, ck) in CAPS for st_ in ('client',)] + [('compat', mo, ck) for (mo, ck) in COMPAT_CAPS]
    for w in K.window_sides(D):
        pend.setdefault(w['m'], {})[w['side']] = w
        if len(pend[w['m']]) < 2:
            continue
        W = pend.pop(w['m']); st['windows'] += 1; m = W[0]['m']; n = W[0]['n']
        if not np.array_equal(W[0]['tg'], W[1]['tg']) or not np.array_equal(W[0]['tf'], W[1]['tf']):
            sys.exit(f'REFUSED: {day} window {m}: the two sides do not share the grid / frames')
        ctx = window_ctx(W, PO, POI, n)
        src = {}
        for s in (0, 1):
            if (m, s) in D['LI']:
                l = D['LT'].iloc[D['LI'][(m, s)]]
                o_ = np.lexsort(((-l.ptick.values if s == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
            else:
                lrow = np.zeros(0, 'int64')
            src[s] = WS.reindex(lrow).values.astype(bool) if len(lrow) else np.zeros(0, bool)
        tend = W[0]['te']; jte = int(np.searchsorted(ctx[0]['tf'], tend, side='right')) - 1; mid_end = ctx[0]['mff'][max(jte, 0)]
        yup = 100.0 * W[0]['yup']
        for q, shf in CFGS:
            cfg = f'{q}{int(shf)}'
            for arm in ARMS:
                dec, idle = arm_dec(arm, W, ctx)
                for state, mode, ck in runs:
                    r = sim_joint(W, q, shf, None if ck is None else ck * SIZE, {0: ctx[0]['vg'], 1: ctx[1]['vg']}, decide=dec, idle_only=idle,
                                  mode=mode, state=state)
                    st['runs'] += 1
                    for v_ in VIOL:
                        st['viol'][v_] += r['stat']['viol'][v_]
                    cn = cap_name(mode, ck) + ('' if state == 'client' else '_compat')
                    if mode == 'hard':
                        st['maxexp_hard'][cn] = max(st['maxexp_hard'].get(cn, 0.0), r['stat']['maxpos'])
                    inv.append((int(day), cfg, arm, cn, m, r['stat']['maxpos'], r['endpos'], r['stat']['cap_steps'], sum(r['stat']['cap_pulls']),
                                sum(r['stat']['pulls']), sum(r['stat']['skip']), sum(r['stat']['inter']), sum(r['stat']['hard_block'])))
                    for s in (0, 1):
                        c_ = ctx[s]; sg = c_['sg']; tf, mff, tfl = c_['tf'], c_['mff'], c_['tfl']; anc = W[s]['anc']
                        val = lambda t, h, pt: sg * (mff[max(int(np.searchsorted(tf, t + h, side='right')) - 1, 0)] - pt)
                        for d in range(NR):
                            for f in r['fills'][(s, d)]:
                                i, pt, qq, thr, age, ph = f[0], int(f[1]), f[2], f[3], f[4], f[5]
                                t = float(anc[i])
                                if not (W[s]['ws'] <= t < W[s]['ws'] + 61000.0):
                                    continue
                                d_ = GP.depth_of(tfl, W[s]['P5'], W[s]['opp'], s, t, pt)
                                fills.append((int(day), cfg, arm, cn, s, d, 'WS' if src[s][i] else 'E2', -9 if d_ is None else d_, bool(thr),
                                              age < 100.0, ph == 'pend', float(qq), val(t, 1000.0, pt), val(t, 5000.0, pt),
                                              sg * (mid_end - pt), sg * (yup - pt), float(age)))
        for s in (0, 1):
            W[s].pop('cache', None)
        if st['windows'] % 40 == 0:
            r2 = K.rss_mib(); print(f'{day}: {st["windows"]} windows, {time.time() - T0:.0f} s, rss {r2:.0f} MiB', flush=True)
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    F = pd.DataFrame(fills, columns=['day', 'cfg', 'arm', 'capk', 'side', 'rung', 'src', 'depth', 'thr', 'young', 'pend', 'q', 'v1', 'v5', 'vend', 'vset', 'age'])
    R1 = pd.read_parquet(f'{JL1.OUTD}/fills_{day}.parquet'); reg_bad = {}
    for cn, ck1 in (('soft2_compat', '2'), ('none_compat', 'none')):
        for arm in ARMS:
            for cfg in ('M0', 'M50', 'M250'):
                X = F[(F.capk == cn) & (F.arm == arm) & (F.cfg == cfg)].drop(columns=['capk']).reset_index(drop=True)
                Y = R1[(R1.capk == ck1) & (R1.arm == arm) & (R1.cfg == cfg)].drop(columns=['capk']).reset_index(drop=True)
                if not FB.same_rows(X, Y):
                    reg_bad[f'{cn} {arm} {cfg}'] = (len(X), len(Y))
    st['regression_bad'] = reg_bad
    for cn, mx in st['maxexp_hard'].items():
        if mx > int(cn.split('_')[0][4:]) * SIZE + 1e-9:
            st['viol']['cap'] += 1
    if any(st['viol'].values()) or reg_bad:
        sys.exit(f'JOINT-LADDER v3 CONTROL FAILED on {day} (conservation / accounting / regression) -> no output: viol {st["viol"]}, regression {reg_bad}')
    st['planted_caught'] = mut
    os.makedirs(OUTD, exist_ok=True)
    F.to_parquet(f'{OUTD}/fills_{day}.parquet', index=False)
    pd.DataFrame(inv, columns=['day', 'cfg', 'arm', 'capk', 'm', 'maxpos', 'endpos', 'cap_steps', 'cap_pulls', 'pulls', 'skip', 'inter', 'hard_block']
                 ).to_parquet(f'{OUTD}/inv_{day}.parquet', index=False)
    json.dump(dict(stat=st, days=len(D['W0']) / 288.0), open(f'{OUTD}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: windows {st["windows"]}, runs {st["runs"]}, violations {st["viol"]}, max hard exposure {st["maxexp_hard"]} in {time.time() - T0:.0f} s', flush=True)


# ---------------------------------------------------------------- report
def report():
    stats = {d: json.load(open(f'{OUTD}/day_{d}.json')) for d in DAYS9}
    par = {d: json.load(open(f'{OUTD}/parity_{d}.json')) for d in INV_DAYS}
    inv_ = {d: json.load(open(f'{OUTD}/invariance_{d}.json')) for d in INV_DAYS}
    A = pd.concat([pd.read_parquet(f'{OUTD}/fills_{d}.parquet') for d in DAYS9], ignore_index=True)
    I = pd.concat([pd.read_parquet(f'{OUTD}/inv_{d}.parquet') for d in DAYS9], ignore_index=True)
    nbad = int((~np.isfinite(A.v1)).sum()); A = A[np.isfinite(A.v1)]
    dd = {d: stats[d]['days'] for d in DAYS9}; ndays = sum(dd.values())
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    CN = [cap_name(mo, k) for mo, k in CAPS]

    def cell(arm, cfg, cn, srcs=('WS',), d=None, h='v1', aged=True):
        x = A[(A.arm == arm) & (A.cfg == cfg) & (A.capk == cn) & A.src.isin(srcs)]
        if aged:
            x = x[~x.young]
        if d:
            x = x[x.day == int(d)]
        return float((x.q * x[h]).sum()) / 100.0 / (dd[d] if d else ndays)
    runs = sum(stats[d]['stat']['runs'] for d in DAYS9)
    mx = {}
    for d in DAYS9:
        for cn, v in stats[d]['stat']['maxexp_hard'].items():
            mx[cn] = max(mx.get(cn, 0.0), v)
    R = {}
    L = [f'JOINT-LADDER v3, 9 consumed replay days, first 60 s, both sides of the guard ladder, shared position, decisions on client-known state, caps '
         f'hard / soft = K x {SIZE:.0f} sh; as of {asof}; controls passed: decision invariance {sum(v["checks"] for v in inv_.values())} checks '
         f'({sum(v["decisions_compared"] for v in inv_.values())} decisions) on 3 days with every venue event after the cut changed, 0 differ; conservation '
         f'/ accounting on all {runs} runs, 0 violations; compat soft-2 / no-cap = v1 row for row; compat independent-mode parity '
         f'{sum(p["checked"] for p in par.values())} checks, 0 differ; max true exposure under the hard caps {mx} sh; fills without a 1 s value {nbad}', '',
         'PART 1: $/day w/o young: WS 1 s | E2 1 s | all trades 1 s | WS 5 s | all fills marked at 60 s | at settlement || pulls a day | placements refused'
         ' by the hard cap a day | held-price waits a day | max |position| per window median / p95 / max sh']
    for cfg in ('M0', 'M50', 'M250'):
        for arm in ARMS:
            for cn in CN:
                x = I[(I.arm == arm) & (I.cfg == cfg) & (I.capk == cn)]
                L.append(f'  {cfg:<4} {arm:<4} {cn:<6} {cell(arm, cfg, cn):+8.1f} | {cell(arm, cfg, cn, ("E2",)):+8.1f} | '
                         f'{cell(arm, cfg, cn, ("WS", "E2")):+8.1f} | {cell(arm, cfg, cn, h="v5"):+8.1f} | {cell(arm, cfg, cn, ("WS", "E2"), h="vend", aged=False):+8.1f} | '
                         f'{cell(arm, cfg, cn, ("WS", "E2"), h="vset", aged=False):+9.1f} || {x.pulls.sum() / ndays:7.0f} | {x.hard_block.sum() / ndays:7.0f} | '
                         f'{x.skip.sum() / ndays:6.0f} | {x.maxpos.median():4.0f} / {x.maxpos.quantile(0.95):4.0f} / {x.maxpos.max():4.0f}')
        L.append('')
    L.append('READINGS (R613: higher at M 0 AND M 50 on >= 7 of 9 days)')
    for a, b in (('FB25', 'LDG'), ('OFB2', 'LDG')):
        for cn in ('hard2', 'hard5', 'hard10'):
            up = {c: sum(cell(a, c, cn, d=d) > cell(b, c, cn, d=d) for d in DAYS9) for c in ('M0', 'M50', 'M250')}
            dn = {c: sum(cell(a, c, cn, d=d) < cell(b, c, cn, d=d) for d in DAYS9) for c in ('M0', 'M50', 'M250')}
            lab = 'PAYS' if min(up['M0'], up['M50']) >= 7 else ('DOES-NOT-PAY' if min(dn['M0'], dn['M50']) >= 7 else 'NOT-SHOWN')
            diff = {c: cell(a, c, cn) - cell(b, c, cn) for c in ('M0', 'M50', 'M250')}
            R[f'R613 {a} vs {b} {cn}'] = dict(label=lab, up=up, down=dn, diff=diff)
            L.append(f'  R613 {a} vs {b}, {cn:<6}: {lab:<13} (M 0 {diff["M0"]:+7.1f}, {up["M0"]}/9; M 50 {diff["M50"]:+7.1f}, {up["M50"]}/9;'
                     f' M 250 {diff["M250"]:+8.1f}, {up["M250"]}/9)')
    L.append('')
    L.append('R615 CLIENT vs VENUE-STATE decisions (printed): client-known minus v1 (compat), WS $/day w/o young 1 s (M 0 / M 50 / M 250)')
    for arm in ARMS:
        for cn in ('soft2', 'none'):
            L.append(f'  {arm:<4} {cn:<6}: ' + ' / '.join(f'{cell(arm, c, cn) - cell(arm, c, cn + "_compat"):+7.1f}' for c in ('M0', 'M50', 'M250')))
    L += ['', '  per day, WS $/day w/o young, M 50, hard2: ' + ' | '.join(ARMS)]
    for d in DAYS9:
        L.append(f'    {d} ' + ' | '.join(f'{cell(a_, "M50", "hard2", d=d):+7.1f}' for a_ in ARMS))
    L += ['', 'C1 / C12: $/day at 1 s by order age, M 50, hard2: WS (< 100 | 100-500 | >= 500 ms) || E2 (same)']
    for arm in ARMS:
        x = A[(A.arm == arm) & (A.cfg == 'M50') & (A.capk == 'hard2')]; parts = []
        for s_ in ('WS', 'E2'):
            y = x[x.src == s_]
            parts.append(' | '.join(f'{float((y.q * y.v1)[(y.age >= lo) & (y.age < hi)].sum()) / 100.0 / ndays:+8.1f}' for _, lo, hi in AGEB))
        L.append(f'  {arm:<4} {parts[0]} || {parts[1]}')
    L.append('')
    L.append('READINGS: ' + json.dumps({k_: v_['label'] for k_, v_ in R.items()}))
    os.makedirs(OUTD, exist_ok=True)
    json.dump(R, open(f'{OUTD}/joint_ladder_v3.json', 'w'), indent=1, default=float)
    open(f'{OUTD}/joint_ladder_v3.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        all_selftests()
    elif cmd == 'invariance':
        invariance_day(sys.argv[2])
    elif cmd == 'parity':
        parity_day(sys.argv[2])
    elif cmd == 'eval':
        eval_day(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit('usage: pf_joint_ladder_v3.py selftest | invariance <day> | parity <day> | eval <day> | report')
