"""JOINT-LADDER (2026-11-14, session ce68ebe0; USER: "The economic test is incomplete. The current replay evaluates ladder levels independently.
Preserving several orders needs evaluation under shared inventory limits, including adverse fills and cancellation/re-entry costs." / "We
should have larger cap when we increase the size").

One simulation holds the guard ladder of BOTH sides of a window with a shared position (FUTURE-BEST / TARGET-TEST simulate each rung of each
side alone):
  - per side, rungs d = 0..5 of SIZE sh: rung d placed at the known best - d (asks + d), kept while the known best stays within HOLD_K + d
    ticks of it and it does not cross the other side's known best, re-placed once its triggers have been clear for REST_COOL - S4's rest lane
    (pf_kp_s4.sim2 'rest'), line for line, per rung;
  - our orders never share a price on one side: a placement at a price we already hold there (live or pending) waits (counted);
  - the trade legs of both sides in time order; within a leg our orders on that side best price first, then go-live time, each leg's volume
    shared among them; queue rule M / U, cancel latency, post-only refusal, own fills known L_FILL later - S4's;
  - the position: + bid fills, - ask fills; the KNOWN position counts fills known by t (fill time + L_FILL);
  - CAP = CAP_K x SIZE (scales with the order size, USER): while |known position| >= CAP, every order on the side that would increase
    |position| is pulled and that side's entries are blocked (v3's soft cap: the side is withheld); its rungs then wait REST_COOL;
  - decisions: each side's guard (v <= -0.10) is each rung's trigger; with a decision callback (FUTURE-BEST's fb_dec) the guard blocks entries
    and the callback decides each resting order (as FUTURE-BEST).
Arms (30 sh, rungs 0-5, both sides): LDG (pull every rung on the guard), FB25 (FUTURE-BEST's keep rule with PULL-DEPTH's P), OFB2 (the same
rule with the realized move to 2 s: perfect foresight, TARGET-TEST's best oracle), each at CAP_K = 2 (v3's soft cap) / 5 (v3's hard cap) / 10 /
none. 9 consumed replay days (10-23..27, 10-31, 11-01..03), first 60 s, fill clocks M 0 / M 50 / M 250.
Values: per fill c/sh x shares at 1 s / 5 s from the trade's anchor (FUTURE-BEST's), by order age and trade timing (C1, C12); every fill also
marked to the Predict mid at the lane's end (60 s) and to settlement (100 x Up), i.e. the window's position held to then; inventory: max
|position| and end position per window, steps at the cap, cap pulls.
Readings DECLARED BEFORE THE FIRST RUN (WS $/day w/o young fills (order >= 100 ms) at 1 s; day rule >= 7 of 9):
 R610 PARITY (control, any failure -> no output): with no cap, every window-side without an interaction (a placement waiting on a price we
      hold, or a trade leg that could fill two of our orders on that side) equals the per-rung lanes (pf_kp_s4.sim_arm with the same trigger and
      callback) fill for fill and order for order, every arm and clock. Interaction counts printed.
 R611 KEEP-UNDER-CAP (FB25 vs LDG and OFB2 vs LDG, each at CAP_K = 2, 5, 10 and none): higher at M 0 AND M 50 on >= 7 days -> PAYS; lower at both
      on >= 7 -> DOES-NOT-PAY; else NOT-SHOWN; M 250 and whether the sign holds printed.
 R612 CAP-EFFECT (printed, no label): each arm's $/day at each CAP_K minus at no cap, per clock; FB25 - LDG at each CAP_K.
 Printed: the 60 s and settlement marks, all trades / E2, age split, inventory, pulls and cap pulls a day, per-day table.
Controls: R610; S4's sim2 self-test via FUTURE-BEST's / PULL-DEPTH's self-tests; synthetic joint cases (a sweep leg reaching two of our bids:
volume shared in price priority; the cap withholds only the side that would increase |position|, from L_FILL after the fill; a placement
waiting on a price we hold) with planted errors (each order given the whole leg; the cap on the wrong side; the cap on the unlagged position)
that must each fail a case; memory self-guard (2 GiB).
Limits: replay rule M (B1a), inflated volumes (C4), consumed days (C8); the guard ladder, not the frozen v3 (C11); LDG / FB25 decisions are
unchanged here (the ORDER-HARM policy comes next); settlement marks rest on one binary outcome per window (noisy).
Usage (run_light, PF_SLOT=claude): pf_joint_ladder.py selftest | eval <day> | report; driver run_joint_ladder.sh
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

DAYS9 = K.DAYS9
OUTD = f'{K.IO}/joint_ladder_20261114'
SIZE = 30.0; NR = 6
CAPKS = (2, 5, 10, None)
ARMS = ('LDG', 'FB25', 'OFB2')
CFGS = [('M', 0.0), ('M', 50.0), ('M', 250.0)]
AGEB = FB.AGEB
INF = float('inf'); NAN = float('nan')
MUT_LEG_ALL = MUT_CAP_SIDE = MUT_CAP_NOLAG = 0            # planted errors (self-test only)
MEM_CAP_MIB = 2048.0


# ---------------------------------------------------------------- the joint lane
def sim_joint(W, q, shf, cap, aux, decide=None, idle_only=0, size=SIZE, nr=NR):
    """W: {0: bid context, 1: ask context} of one window (pf_kp_s4.window_sides), trades shifted by shf, queue rule q; cap in shares or None;
    aux: {s: per-step trigger}; decide: {(s, d): callback(k, t, cur)} or None. Returns per (s, d) fills / orders in sim2's tuple form and stats."""
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
                     cache=w.setdefault('cache', {}), live=[], cur=[None] * nr, lastpull=[-INF] * nr, fills=[[] for _ in range(nr)],
                     orders=[[] for _ in range(nr)], aux=aux[s], dec=[(decide.get((s, d)) if decide else None) for d in range(nr)])
    stat = dict(pulls=[0, 0], cap_pulls=[0, 0], cap_steps=0, skip=[0, 0], inter=[0, 0], maxpos=0.0)
    pos = [0.0]; KN = []; fl = []                         # KN: (time known, signed shares); fl: (side, rung, time, price, shares)

    def lv(s, pt):
        c = ST[s]['cache']
        if pt not in c:
            P5, S5 = ST[s]['P5'], ST[s]['S5']; last = P5[:, 4]
            c[pt] = np.where((last >= 0) & ((pt < last) if s == 0 else (pt > last)), np.nan, (S5 * (P5 == pt)).sum(1))
        return c[pt]

    def place(s, d, pt, t):
        ta = t + K.L_PLACE; fi = bisect.bisect_right(tfl, ta) - 1; opp = ST[s]['opp']
        o = dict(pt=pt, ts=t, ta=ta, tr=INF, tc=INF, rem=size, ahead=0.0, ci=fi, tfill=INF, a0=np.nan, ref=False, rung=d)
        if fi < 0 or ((0 <= opp[fi] <= pt) if s == 0 else (opp[fi] >= pt)):
            o['ref'] = True; o['tr'] = ta
        else:
            a = lv(s, pt)[fi]; a = float(a) if np.isfinite(a) else 0.0
            if fi + 1 < nf and tfl[fi + 1] - ta <= 100.0:
                d2 = lv(s, pt)[fi + 1]
                if np.isfinite(d2) and d2 > a:
                    a = float(d2)
            o['a0'] = a; o['ahead'] = 0.0 if q == 'U' else a
            ST[s]['live'].append(o)
        ST[s]['orders'][d].append(o)
        return o

    def cancel(o, t):
        o['tc'] = min(o['tc'], t)
        o['tr'] = min(o['tr'], max(t + K.L_CANCEL, o['ts'] + K.L_PENDCANCEL) if t < o['ta'] else t + K.L_CANCEL)

    def apply(t, s, i):
        S_ = ST[s]; p = S_['ptl'][i]; left = S_['shl'][i]; full = left
        el = [o for o in S_['live'] if o['rem'] > 1e-12 and o['ta'] <= t < o['tr']]
        el.sort(key=lambda o: ((-o['pt']) if s == 0 else o['pt'], o['ta']))
        reach = [o for o in el if o['pt'] == p or ((p < o['pt']) if s == 0 else (p > o['pt']))]
        if len(reach) >= 2:
            stat['inter'][s] += 1
        for o in el:
            if q == 'M':
                j = bisect.bisect_left(tfl, t) - 1
                if j > o['ci']:
                    seg = lv(s, o['pt'])[o['ci'] + 1:j + 1]; seg = seg[np.isfinite(seg)]
                    if len(seg) and seg.min() < o['ahead']:
                        o['ahead'] = float(seg.min())
                    o['ci'] = j
            thr = (p < o['pt']) if s == 0 else (p > o['pt'])
            lft = full if MUT_LEG_ALL else left
            if p == o['pt']:
                use = min(lft, o['ahead']); o['ahead'] -= use; qq = min(lft - use, o['rem'])
            elif thr:
                qq = min(lft, o['rem']); o['ahead'] = 0.0
            else:
                qq = 0.0
            if qq > 1e-12:
                left -= qq; o['rem'] -= qq
                g = bisect.bisect_right(taul, t) - 1
                ph = 'pend' if t >= o['tc'] else ('above' if g >= 0 and S_['v'][g] > -INF else 'hold')
                S_['fills'][o['rung']].append((i, o['pt'], qq, thr, t - o['ta'], ph, o['a0']))
                sq = qq if s == 0 else -qq
                pos[0] += sq; KN.append((t + (0.0 if MUT_CAP_NOLAG else K.L_FILL), sq)); fl.append((s, o['rung'], t, o['pt'], qq))
                stat['maxpos'] = max(stat['maxpos'], abs(pos[0]))
                if o['rem'] <= 1e-12:
                    o['tfill'] = t

    it = 0; ik = 0; kpos = 0.0
    for k, t in enumerate(taul):
        while it < len(TL) and TL[it][0] <= t:
            apply(*TL[it]); it += 1
        while ik < len(KN) and KN[ik][0] <= t:
            kpos += KN[ik][1]; ik += 1
        capside = -1
        if cap is not None and abs(kpos) >= cap - 1e-9:
            capside = 0 if kpos > 0 else 1
            if MUT_CAP_SIDE:
                capside = 1 - capside
            stat['cap_steps'] += 1
        for s in (0, 1):
            S_ = ST[s]; b = S_['best'][k]
            for d in range(nr):
                cur = S_['cur'][d]
                if cur is not None and ((cur['ref'] and cur['ta'] <= t) or (cur['rem'] <= 1e-12 and cur['tfill'] + K.L_FILL <= t)):
                    cur = None
                trig = bool(S_['aux'][k])
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
                    if trig or b < 0:
                        kill = True
                    else:
                        beh = (b - cur['pt']) if s == 0 else (cur['pt'] - b)
                        kill = beh > K.HOLD_K + d or not ((cur['pt'] < S_['bopp'][k]) if s == 0 else (cur['pt'] > S_['bopp'][k]))
                    if kill:
                        cancel(cur, t); cur = None
                if trig:
                    S_['lastpull'][d] = t
                if cur is None and b >= 0 and not trig and t - S_['lastpull'][d] >= K.REST_COOL:
                    tgt = int(b) if d == 0 else ((int(b) - d) if s == 0 else (int(b) + d))
                    if d == 0 or 1 <= tgt <= 99:
                        if any(o['pt'] == tgt and o['rem'] > 1e-12 and o['tr'] > t for o in S_['live']):
                            stat['skip'][s] += 1
                        else:
                            cur = place(s, d, tgt, t)
                S_['cur'][d] = cur
            lv_ = S_['live']
            if lv_ and (lv_[0]['tr'] <= t or lv_[0]['rem'] <= 1e-12):
                S_['live'] = [o for o in lv_ if o['rem'] > 1e-12 and o['tr'] > t]
    while it < len(TL) and TL[it][0] <= t_end:
        apply(*TL[it]); it += 1
    for s in (0, 1):
        for d in range(nr):
            if ST[s]['cur'][d] is not None:
                cancel(ST[s]['cur'][d], t_end)
    while it < len(TL) and TL[it][0] <= t_end + 200.0:
        apply(*TL[it]); it += 1
    out = dict(fills={(s, d): ST[s]['fills'][d] for s in (0, 1) for d in range(nr)},
               orders={(s, d): [(o['ta'], min(o['tr'], o['tfill'], t_end + 200.0), o['a0'], o['ref']) for o in ST[s]['orders'][d]]
                       for s in (0, 1) for d in range(nr)},
               cancels={(s, d): [o['tc'] for o in ST[s]['orders'][d]] for s in (0, 1) for d in range(nr)},
               stat=stat, endpos=pos[0], fl=fl)
    return out


# ---------------------------------------------------------------- self-tests (synthetic windows)
def _ctx(side, fr, legs, t_end=1000.0, v=None):
    """A minimal window-side context. fr: [(t, [(price, size) x <= 5] best first on this side, the other side's best)]; legs: [(t, price, sh)].
    Every frame is known at its time."""
    tf = np.array([f[0] for f in fr], float); n = len(fr)
    P5 = np.full((n, 5), -1, dtype='int64'); S5 = np.zeros((n, 5))
    for r_, f in enumerate(fr):
        for j, (p, z) in enumerate(f[1][:5]):
            P5[r_, j] = p; S5[r_, j] = z
    opp = np.array([f[2] for f in fr], dtype='int64')
    tg = np.arange(0.0, t_end, 50.0); ki = np.searchsorted(tf, tg, side='right') - 1
    best = [int(P5[i, 0]) if i >= 0 else -1 for i in ki]; bopp = [int(opp[i]) if i >= 0 else -1 for i in ki]
    vk = [0.5] * len(tg) if v is None else v
    return dict(side=side, tf=tf, kf=tf.copy(), jk=ki, tg=tg, n=len(tg), te=t_end, P5=P5, S5=S5, opp=opp, best=best, bopp=bopp, vkl=vk,
                stn=[True] * len(tg), X=np.zeros((len(tg), 37)), anc=np.array([x[0] for x in legs], float), ptl=[x[1] for x in legs],
                shl=[float(x[2]) for x in legs], ws=0.0, sg=1.0 if side == 0 else -1.0)


def selftest():
    bad = []
    fb = [(0.0, [(50, 100), (49, 100), (48, 100)], 52)]; fa = [(0.0, [(52, 100), (53, 100)], 50)]
    # (a) a sweep leg at 47 reaches our bids at 50 and 49 (rungs 0 and 1, 30 sh each, queue U): 40 sh shared in price priority
    W = {0: _ctx(0, fb, [(500.0, 47, 40.0)]), 1: _ctx(1, fa, [])}
    r = sim_joint(W, 'U', 0.0, None, {0: [False] * 20, 1: [False] * 20}, nr=2)
    f0 = sum(x[2] for x in r['fills'][(0, 0)]); f1 = sum(x[2] for x in r['fills'][(0, 1)])
    if (f0, f1) != (30.0, 10.0) or r['stat']['inter'][0] != 1:
        bad.append(f'sweep shared {f0} {f1} inter {r["stat"]["inter"]}')
    # (b) the cap: 60 sh of bid fills (rungs 0 and 1 swept at 500 ms) -> |known position| 60 >= 2 x 30 once the fills are known (500 ms +
    # L_FILL, at the next grid step): bids withheld; the asks rest until the window's end (cancelled there, at 1000 ms)
    W = {0: _ctx(0, fb, [(500.0, 47, 60.0)]), 1: _ctx(1, fa, [])}
    r = sim_joint(W, 'U', 0.0, 60.0, {0: [False] * 20, 1: [False] * 20}, nr=3)
    c2 = r['cancels'][(0, 2)]; ca = r['cancels'][(1, 0)]; tcap = float(np.ceil((500.0 + K.L_FILL) / 50.0) * 50.0)
    if not (len(c2) == 1 and c2[0] == tcap and ca == [1000.0] and r['stat']['cap_pulls'][0] >= 1):
        bad.append(f'cap: bid rung 2 cancel {c2}, ask rung 0 cancel {ca}, cap pulls {r["stat"]["cap_pulls"]}')
    # (c) a placement waiting on a price we hold: rung 1 pulled by its callback at 100 ms; the best moves to 51 at 300 ms while rung 0 holds 50
    fb2 = [(0.0, [(50, 100), (49, 100)], 52), (300.0, [(51, 100), (50, 100), (49, 100)], 53)]; fa2 = [(0.0, [(52, 100)], 50), (300.0, [(53, 100)], 51)]
    W = {0: _ctx(0, fb2, []), 1: _ctx(1, fa2, [])}
    r = sim_joint(W, 'U', 0.0, None, {0: [False] * 20, 1: [False] * 20}, decide={(0, 1): lambda k, t, cur: k == 2}, nr=2)
    if r['stat']['skip'][0] < 1:
        bad.append(f'collision skip {r["stat"]["skip"]}')
    pts = [x for x in r['orders'][(0, 1)]]
    if len(pts) != 1:
        bad.append(f'rung 1 re-placed onto a price we hold: {pts}')
    # (d) parity with the per-rung lane on a synthetic window without interactions (rung 0 alone, queue M, a trade at 50)
    W = {0: _ctx(0, fb, [(400.0, 50, 150.0)]), 1: _ctx(1, fa, [])}
    r = sim_joint(W, 'M', 0.0, None, {0: [False] * 20, 1: [False] * 20}, nr=1)
    F_, O_, _ = K.sim_arm(W[0], 'M', 0.0, 'rest', aux=[False] * 20, offset=0, size=SIZE)
    if r['fills'][(0, 0)] != F_ or r['orders'][(0, 0)] != O_:
        bad.append(f'parity: joint {r["fills"][(0, 0)]} {r["orders"][(0, 0)]} vs lane {F_} {O_}')
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_LEG_ALL', 'MUT_CAP_SIDE', 'MUT_CAP_NOLAG'):
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
        sys.exit(f'JOINT-LADDER SELFTEST FAILED -> no output: {b} planted errors caught {m}')
    print(f'joint-ladder selftests passed; planted errors caught: {m}', flush=True)
    return m


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
    st.update(windows=0, parity_checked=0, parity_bad=0, parity_skipped=0)
    fills = []; inv = []
    pend = {}
    for w in K.window_sides(D):
        pend.setdefault(w['m'], {})[w['side']] = w
        if len(pend[w['m']]) < 2:
            continue
        W = pend.pop(w['m']); st['windows'] += 1; m = W[0]['m']; n = W[0]['n']
        if not np.array_equal(W[0]['tg'], W[1]['tg']) or not np.array_equal(W[0]['tf'], W[1]['tf']):
            sys.exit(f'REFUSED: {day} window {m}: the two sides do not share the grid / frames')
        ctx = {}
        for s in (0, 1):
            w_ = W[s]; tf, mff = w_['tf_mff']; tfl = tf.tolist(); P5, opp = w_['P5'], w_['opp']
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
            if (m, s) in D['LI']:
                l = D['LT'].iloc[D['LI'][(m, s)]]
                o_ = np.lexsort(((-l.ptick.values if s == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
            else:
                lrow = np.zeros(0, 'int64')
            src = WS.reindex(lrow).values.astype(bool) if len(lrow) else np.zeros(0, bool)
            ctx[s] = dict(Pm=Pm, vg=vg, mv2=mv2, src=src, tf=tf, mff=mff, tfl=tfl, sg=w_['sg'])
        tend = W[0]['te']; jte = int(np.searchsorted(ctx[0]['tf'], tend, side='right')) - 1; mid_end = ctx[0]['mff'][max(jte, 0)]
        yup = 100.0 * W[0]['yup']
        for q, shf in CFGS:
            cfg = f'{q}{int(shf)}'
            for arm in ARMS:
                dec = None; idle = 0
                if arm != 'LDG':
                    idle = 1
                    dec = {(s, d): (FB.fb_dec(ctx[s]['vg'], W[s]['best'], s, ctx[s]['Pm'], d, tau=0.25) if arm == 'FB25' else
                                    FB.fb_dec(ctx[s]['vg'], W[s]['best'], s, ctx[s]['Pm'], d, mv=ctx[s]['mv2'])) for s in (0, 1) for d in range(NR)}
                for capk in CAPKS:
                    r = sim_joint(W, q, shf, None if capk is None else capk * SIZE, {0: ctx[0]['vg'], 1: ctx[1]['vg']}, decide=dec, idle_only=idle)
                    if capk is None:                      # R610: parity with the per-rung lanes where nothing interacted
                        for s in (0, 1):
                            if r['stat']['inter'][s] or r['stat']['skip'][s]:
                                st['parity_skipped'] += 1; continue
                            st['parity_checked'] += 1
                            for d in range(NR):
                                F_, O_, _ = K.sim_arm(W[s], q, shf, 'rest', aux=ctx[s]['vg'], decide=(dec[(s, d)] if dec else None), idle_only=idle,
                                                      offset=d, size=SIZE)
                                if F_ != r['fills'][(s, d)] or O_ != r['orders'][(s, d)]:
                                    st['parity_bad'] += 1
                    ck = 'none' if capk is None else str(capk)
                    inv.append((int(day), cfg, arm, ck, m, r['stat']['maxpos'], r['endpos'], r['stat']['cap_steps'], sum(r['stat']['cap_pulls']),
                                sum(r['stat']['pulls']), sum(r['stat']['skip']), sum(r['stat']['inter'])))
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
                                fills.append((int(day), cfg, arm, ck, s, d, 'WS' if c_['src'][i] else 'E2', -9 if d_ is None else d_, bool(thr),
                                              age < 100.0, ph == 'pend', float(qq), val(t, 1000.0, pt), val(t, 5000.0, pt),
                                              sg * (mid_end - pt), sg * (yup - pt), float(age)))
            if (K.L_PLACE, K.L_CANCEL, K.L_PENDCANCEL) != (PD.BG.CLK0['L_PLACE'], PD.BG.CLK0['L_CANCEL'], PD.BG.CLK0['L_PENDCANCEL']):
                sys.exit('REFUSED: clock constants changed')
        for s in (0, 1):
            W[s].pop('cache', None)
        if st['windows'] % 40 == 0:
            r2 = K.rss_mib(); print(f'{day}: {st["windows"]} windows, {time.time() - T0:.0f} s, rss {r2:.0f} MiB', flush=True)
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    if st['parity_bad'] or not st['parity_checked']:
        sys.exit(f'JOINT-LADDER CONTROL FAILED on {day} (R610 parity) -> no output: {st}')
    st['planted_caught'] = mut
    os.makedirs(OUTD, exist_ok=True)
    pd.DataFrame(fills, columns=['day', 'cfg', 'arm', 'capk', 'side', 'rung', 'src', 'depth', 'thr', 'young', 'pend', 'q', 'v1', 'v5', 'vend', 'vset', 'age']
                 ).to_parquet(f'{OUTD}/fills_{day}.parquet', index=False)
    pd.DataFrame(inv, columns=['day', 'cfg', 'arm', 'capk', 'm', 'maxpos', 'endpos', 'cap_steps', 'cap_pulls', 'pulls', 'skip', 'inter']
                 ).to_parquet(f'{OUTD}/inv_{day}.parquet', index=False)
    json.dump(dict(stat=st, days=len(D['W0']) / 288.0), open(f'{OUTD}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: {json.dumps({k_: v_ for k_, v_ in st.items() if not isinstance(v_, dict)}, default=float)} in {time.time() - T0:.0f} s', flush=True)


# ---------------------------------------------------------------- report
def report():
    stats = {d: json.load(open(f'{OUTD}/day_{d}.json')) for d in DAYS9}
    A = pd.concat([pd.read_parquet(f'{OUTD}/fills_{d}.parquet') for d in DAYS9], ignore_index=True)
    I = pd.concat([pd.read_parquet(f'{OUTD}/inv_{d}.parquet') for d in DAYS9], ignore_index=True)
    nbad = int((~np.isfinite(A.v1)).sum()); A = A[np.isfinite(A.v1)]
    dd = {d: stats[d]['days'] for d in DAYS9}; ndays = sum(dd.values())
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    CK = ['2', '5', '10', 'none']

    def cell(arm, cfg, ck, srcs=('WS',), d=None, h='v1', aged=True):
        x = A[(A.arm == arm) & (A.cfg == cfg) & (A.capk == ck) & A.src.isin(srcs)]
        if aged:
            x = x[~x.young]
        if d:
            x = x[x.day == int(d)]
        return float((x.q * x[h]).sum()) / 100.0 / (dd[d] if d else ndays)
    pc = {k_: sum(stats[d]['stat'][k_] for d in DAYS9) for k_ in ('windows', 'parity_checked', 'parity_skipped', 'parity_bad')}
    R = {}
    L = [f'JOINT-LADDER, 9 consumed replay days, first 60 s, both sides of the guard ladder in one lane, shared position, cap = K x {SIZE:.0f} sh; '
         f'as of {asof}; controls passed (R610 parity: {pc["parity_checked"]} window-sides equal the per-rung lanes, {pc["parity_skipped"]} skipped for '
         f'an interaction, 0 differ); windows {pc["windows"]}; fills without a 1 s value {nbad}', '',
         'PART 1: $/day w/o young: WS 1 s | E2 1 s | all trades 1 s | WS 5 s | all fills marked at 60 s | at settlement || pulls a day | cap pulls a day |'
         ' steps at the cap | max |position| per window median / p95 sh | interactions a day']
    for cfg in ('M0', 'M50', 'M250'):
        for arm in ARMS:
            for ck in CK:
                x = I[(I.arm == arm) & (I.cfg == cfg) & (I.capk == ck)]; nst = len(x) * 1200.0
                L.append(f'  {cfg:<4} {arm:<4} cap {ck:<4} {cell(arm, cfg, ck):+8.1f} | {cell(arm, cfg, ck, ("E2",)):+8.1f} | '
                         f'{cell(arm, cfg, ck, ("WS", "E2")):+8.1f} | {cell(arm, cfg, ck, h="v5"):+8.1f} | {cell(arm, cfg, ck, ("WS", "E2"), h="vend", aged=False):+8.1f} | '
                         f'{cell(arm, cfg, ck, ("WS", "E2"), h="vset", aged=False):+9.1f} || {x.pulls.sum() / ndays:7.0f} | {x.cap_pulls.sum() / ndays:6.0f} | '
                         f'{x.cap_steps.sum() / max(nst, 1):5.1%} | {x.maxpos.median():4.0f} / {x.maxpos.quantile(0.95):4.0f} | {x.inter.sum() / ndays:5.0f}')
        L.append('')
    L.append('READINGS (R611: higher at M 0 AND M 50 on >= 7 of 9 days)')
    for a, b in (('FB25', 'LDG'), ('OFB2', 'LDG')):
        for ck in CK:
            up = {c: sum(cell(a, c, ck, d=d) > cell(b, c, ck, d=d) for d in DAYS9) for c in ('M0', 'M50', 'M250')}
            dn = {c: sum(cell(a, c, ck, d=d) < cell(b, c, ck, d=d) for d in DAYS9) for c in ('M0', 'M50', 'M250')}
            lab = 'PAYS' if min(up['M0'], up['M50']) >= 7 else ('DOES-NOT-PAY' if min(dn['M0'], dn['M50']) >= 7 else 'NOT-SHOWN')
            diff = {c: cell(a, c, ck) - cell(b, c, ck) for c in ('M0', 'M50', 'M250')}
            R[f'R611 {a} vs {b} cap {ck}'] = dict(label=lab, up=up, down=dn, diff=diff)
            L.append(f'  R611 {a} vs {b}, cap {ck:<4}: {lab:<13} (M 0 {diff["M0"]:+7.1f}, {up["M0"]}/9; M 50 {diff["M50"]:+7.1f}, {up["M50"]}/9;'
                     f' M 250 {diff["M250"]:+8.1f}, {up["M250"]}/9)')
    L.append('')
    L.append('R612 CAP EFFECT (printed): $/day at the cap minus with no cap, WS w/o young 1 s (M 0 / M 50 / M 250)')
    for arm in ARMS:
        for ck in CK[:-1]:
            L.append(f'  {arm:<4} cap {ck:<3}: ' + ' / '.join(f'{cell(arm, c, ck) - cell(arm, c, "none"):+7.1f}' for c in ('M0', 'M50', 'M250')))
    L.append('')
    L.append('  per day, WS $/day w/o young, M 50, cap 2 / none: ' + ' | '.join(ARMS))
    for d in DAYS9:
        L.append(f'    {d} ' + ' | '.join(f'{cell(a_, "M50", "2", d=d):+7.1f} / {cell(a_, "M50", "none", d=d):+7.1f}' for a_ in ARMS))
    L += ['', 'C1 / C12: $/day at 1 s by order age, M 50, cap 2: WS (< 100 | 100-500 | >= 500 ms) || E2 (same)']
    for arm in ARMS:
        x = A[(A.arm == arm) & (A.cfg == 'M50') & (A.capk == '2')]; parts = []
        for s_ in ('WS', 'E2'):
            y = x[x.src == s_]
            parts.append(' | '.join(f'{float((y.q * y.v1)[(y.age >= lo) & (y.age < hi)].sum()) / 100.0 / ndays:+8.1f}' for _, lo, hi in AGEB))
        L.append(f'  {arm:<4} {parts[0]} || {parts[1]}')
    L += ['', 'By rung, M 50, cap 2 / none: WS $/day w/o young (shares a day)']
    for arm in ARMS:
        parts = []
        for dg in range(NR):
            for ck in ('2', 'none'):
                x = A[(A.arm == arm) & (A.cfg == 'M50') & (A.capk == ck) & (A.rung == dg) & (A.src == 'WS') & ~A.young]
                parts.append(f'r{dg} {ck}: {float((x.q * x.v1).sum()) / 100.0 / ndays:+6.1f} ({x.q.sum() / ndays:5.0f})')
        L.append(f'  {arm:<4} ' + '; '.join(parts))
    L.append('')
    L.append('READINGS: ' + json.dumps({k_: v_['label'] for k_, v_ in R.items()}))
    os.makedirs(OUTD, exist_ok=True)
    json.dump(R, open(f'{OUTD}/joint_ladder.json', 'w'), indent=1, default=float)
    open(f'{OUTD}/joint_ladder.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        all_selftests()
    elif cmd == 'eval':
        eval_day(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit('usage: pf_joint_ladder.py selftest | eval <day> | report')
