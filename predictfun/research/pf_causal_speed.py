"""CS, the causal speed test (2026-11-12, session ce68ebe0). USER: "Yes" to "rerun the speed test with honest timing".

Replaces the VOID fast-entry arms of QR (d = 0 / 15 / 30 ms) and QR2 (F30, Y+F30, QR2-fast): their decision was scheduled at
birth + d - 40 ms, before the level existed (FINDINGS.md C6; independent review 2026-11-12 10:42Z, 9053263d). Here every clock is
explicit and causal. A new best level of our side is born at the venue stamp tb of the frame where the best improves (QR's births:
both frames valid, <= 5 s apart, tb in the window). Our system can know it no earlier than tb + f; the decision is made then, using
the birth frame only (no later frame is observable by tb + f); the order goes live p ms after the decision.
 f (observation, incl. 2 ms decision): 'cur' = the simulator's own known time of the birth frame (kf, median ~53 ms after tb);
   30 ms; 15.2 ms (the smallest feed delay ever measured, 13.2 ms, + 2 ms).
 p (placement): 40 ms (today), 20 ms, 10 ms. A faster path also shortens the pending-cancel window: a cancel sent while the order is
   pending takes effect at max(t + 22, create + 55 + p) (95 ms at p = 40, as now).
Arms (9 + controls), all = v3 (pf_kp_s4.sim2 'join', the frozen v3 lane) plus these birth entries; v3's own grid orders are unchanged:
 E(f,p): at tb + f, if v3's state is on and v > PM90 at the last grid step <= tb + f: an order of ours at another price is cancelled
   and an order at the birth price is sent, live at tb + f + p. Exits are v3's on the current clock, with QR's grace (no price exit
   until the current clock has seen the birth frame).
 RJ(15.2,10): the control - as many joins as births, at uniform random times u in the window (one seeded draw), at the best of the
   last frame stamped <= u - 15.2 ms, decided at u, live at u + 10. Speed without the birth.
 J-v3: v3 alone (control: equals sim2 'join').
Configs M 0 / M 50 / M 250 / L 50. Values as QR2 (1 s markout; $/day = sum(shares x c/sh) / 100 / days).
FINDINGS checklist applied in the tables: C1 every arm's $/day split by the order's age at the fill (< 100 / 100-500 / >= 500 ms),
without fills on orders < 100 ms old ('w/o young', the primary metric) and B1a-weighted (B1a's precision by age and config); C2 M 0 /
M 50 / M 250; C3 cancel-pending share; C5 L 50 printed; C6 the causal control below; C8 the 9 consumed replay days, arms fixed here
in advance (no parameter chosen on these days beyond QR's birth rule and v3's lines).

Readings DECLARED BEFORE THE FIRST RUN (evaluated in code; day rule = >= 7 of 9 days):
 R320 NOW (E(cur,40) vs J-v3): w/o-young $/day at 1 s above J-v3 by the day rule at M 0 and at M 50 -> GAIN; below at both -> LOSS;
      else NOT-SHOWN. Event-driven joining at today's speed.
 R321 SPEED (each E(f,p) vs J-v3, R320's rule) -> GAIN / LOSS / NOT-SHOWN per arm; SPEED-NEEDED = the GAIN arm with the largest
      median live delay (tb to live), NONE if no arm gains.
 R322 WEIGHTED (each E(f,p) vs J-v3): B1a-weighted $/day by the day rule at M 0 and M 50 -> GAIN / LOSS / NOT-SHOWN per arm.
 R323 AT-250 (each E(f,p) vs J-v3): w/o-young $/day above J-v3 at M 250 by the day rule -> HOLDS-AT-250, else NOT-AT-250.
 R324 BIRTH (E(15.2,10) vs RJ(15.2,10)): R320's rule -> BIRTH-MATTERS / RANDOM-BETTER / NOT-SHOWN.
What the outcomes mean, fixed in advance: GAIN = with honest clocks, joining new levels at that speed adds money to v3 in this replay
beyond fills on orders under 100 ms old; SPEED-NEEDED = the slowest such speed; NONE = no tested speed adds money once young fills
are set aside. Any GAIN is a replay candidate: young fills still need the live capture (C10), and the days are consumed.

Controls (any failure -> no table): (i) with no birth entries the lane equals pf_kp_s4.sim2 'join' on every window-side and config;
(ii) with QR2's old schedule (decision at tb + 30 - 40, p = 40) the lane equals QR2's committed F30 day totals (the only change is the
clock); (iii) CAUSAL: on the first 40 window-sides of day 1 at M 50, for every arm, the book and trades are cut at t_c = 5 ms before a
birth whose guard state is on (a random time if none), frames and trades at or after t_c removed and the known best recomputed;
every order sent and every cancel before t_c, and every fill more than 100 ms before t_c (the fill model's raise looks 100 ms past
activation), must be identical to the uncut run; the planted acausal schedule (QR2's) must differ on at least one window-side; (iv) synthetic cases: live time = tb + f + p, the decision never precedes tb + f, the pending-cancel window, the random
control's frame at u - 15.2; planted errors (the decision at tb + f - p; placement ignored) must each fail a case. Statuses printed.
Usage: research/run_cs.sh; python3 predictfun/research/pf_causal_speed.py selftest | day <yyyymmdd> | report
"""
import bisect
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402
import pf_qr_queue_race as Q                              # noqa: E402
import pf_qr2_v3_early as Q2                              # noqa: E402

INF = float('inf'); NAN = float('nan')
L_PLACE, L_CANCEL, L_PENDCANCEL, L_FILL = K.L_PLACE, K.L_CANCEL, K.L_PENDCANCEL, K.L_FILL
SIZE = K.SIZE
DAYS9 = K.DAYS9; OUTC = f'{K.IO}/cs_20261112'; QR2D = f'{K.IO}/qr2_20261112'
CFG = [('M', 0.0), ('M', 50.0), ('M', 250.0), ('L', 50.0)]; PRIM = ('M', 50.0)
F_MIN = K.PF_MIN + 2.0                                    # 15.2 ms
FS = ('cur', 30.0, F_MIN); PS = (40.0, 20.0, 10.0)
ARMS = [f'E({f if f == "cur" else f"{f:g}"},{p:g})' for f in FS for p in PS]
BINS = (('< 100 ms', 0.0, 100.0), ('100-500 ms', 100.0, 500.0), ('>= 500 ms', 500.0, INF))
B1A = {'M|0': (0.08288286541654095, 0.5476125580415678, 0.46785435948727594), 'M|50': (0.12337820978184168, 0.2858571490112695, 0.47665862368157524),
       'M|250': (0.006996168670787272, 0.07718840565404715, 0.21964816762389805), 'L|50': (0.12662915131929803, 0.2959812057268478, 0.4629535648558688)}
MUT_EARLY = MUT_NOPLACE = 0                               # planted errors (self-test only)
N_CAUSAL = 40


def sim5(side, tf, opp, P5, S5, cache, tau, v, s, enter, best, bopp, tt, tp, tq, qrule, t_end, sched=None):
    """v3's join lane (pf_qr2_v3_early.sim4 with every switch off) plus birth entries sched = [(decision time, price, time the
    current clock knows the frame, placement latency)]."""
    for a in (tf, tau, tt):
        if len(a) > 1 and (np.diff(a) < 0).any():
            raise ValueError('REFUSED: frames, grid or trades not time-sorted')
    tfl = tf.tolist(); nf = len(tfl); ttl = tt.tolist(); nt = len(ttl); taul = tau.tolist()
    fills = []; orders = []; live = []; cur = None; it = 0

    def lv(pt):
        if pt not in cache:
            last = P5[:, 4]
            cache[pt] = np.where((last >= 0) & ((pt < last) if side == 0 else (pt > last)), np.nan, (S5 * (P5 == pt)).sum(1))
        return cache[pt]

    def place(pt, t, gr=-INF, lp=L_PLACE):
        lp = L_PLACE if MUT_NOPLACE else lp
        ta = t + lp; fi = bisect.bisect_right(tfl, ta) - 1
        o = dict(id=len(orders), pt=pt, ts=t, ta=ta, tr=INF, tc=INF, rem=SIZE, ahead=0.0, ci=fi, tfill=INF, a0=np.nan, ref=False, gr=gr,
                 pend=L_PENDCANCEL - L_PLACE + lp)
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

    def cancel(o, t):
        o['tc'] = min(o['tc'], t)
        o['tr'] = min(o['tr'], max(t + L_CANCEL, o['ts'] + o['pend']) if t < o['ta'] else t + L_CANCEL)

    def apply(i):
        t = ttl[i]; p = tp[i]; left = tq[i]
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

    items = [(t, 0, k) for k, t in enumerate(taul)]
    if sched:
        items += [(float(e[0]), 1, j) for j, e in enumerate(sched)]
        items.sort()
    for t, typ, k in items:
        while it < nt and ttl[it] <= t:
            apply(it); it += 1
        if cur is not None and ((cur['ref'] and cur['ta'] <= t) or (cur['rem'] <= 1e-12 and cur['tfill'] + L_FILL <= t)):
            cur = None
        if typ == 1:
            pt_, gr_, lp_ = sched[k][1], sched[k][2], sched[k][3]; g = bisect.bisect_right(taul, t) - 1
            if g >= 0 and bool(s[g]) and v[g] > enter:
                if cur is not None and cur['pt'] != pt_:
                    cancel(cur, t); cur = None
                if cur is None:
                    cur = place(int(pt_), t, gr_, lp_); cur['tb'] = sched[k][4] if len(sched[k]) > 4 else NAN
            continue
        b = best[k]; ab = v[k] > enter; on = bool(s[k])
        if cur is not None:
            if b < 0 or not on:
                kill = True
            elif t < cur['gr']:
                kill = False
            elif b != cur['pt']:
                kill = ab or ((b - cur['pt']) if side == 0 else (cur['pt'] - b)) != 1
            else:
                kill = False
            if kill:
                cancel(cur, t); cur = None
        if cur is None and on and ab and b >= 0:
            cur = place(int(b), t)
        if live and (live[0]['tr'] <= t or live[0]['rem'] <= 1e-12):
            live[:] = [o for o in live if o['rem'] > 1e-12 and o['tr'] > t]
    while it < nt and ttl[it] <= t_end:
        apply(it); it += 1
    if cur is not None:
        cancel(cur, t_end)
    while it < nt and ttl[it] <= t_end + 200.0:
        apply(it); it += 1
    return fills, orders


def schedule(tf, kf, b0, bi, f, p):
    """Causal birth entries: decision at tb + f (f = 'cur': the current clock's known time of the frame), live p ms later."""
    out = []
    for i in bi:
        tb = float(tf[i]); td = float(kf[i]) if f == 'cur' else tb + f
        if MUT_EARLY:
            td = td - p                                   # planted: the old acausal construction
        out.append((td, int(b0[i]), float(kf[i]), float(p), tb))
    return out


def rj_schedule(tf, kf, b0, valid, n, ws, te, rng, f=F_MIN, p=10.0):
    """Random-time control: decided at u, at the best of the last frame stamped <= u - f, live at u + p."""
    if n == 0:
        return []
    u = np.sort(rng.uniform(ws, te, n)); j = np.searchsorted(tf, u - f, side='right') - 1
    return [(float(x), int(b0[jj]), float(kf[jj]), float(p), float(x)) for x, jj in zip(u, j) if jj >= 0 and valid[jj]]


def truncate(w, t_c, shf):
    """The window-side with frames and trades at or after t_c removed and the known best recomputed (control iii)."""
    keep = w['tf'] < t_c; kf = w['kf'][keep]; P5 = w['P5'][keep]; S5 = w['S5'][keep]; opp = w['opp'][keep]; tf = w['tf'][keep]
    b0 = P5[:, 0]; side = w['side']
    valid = (b0 >= 0) & (opp >= 0) & ((opp > b0) if side == 0 else (b0 > opp))
    ki = np.searchsorted(kf, w['tg'], side='right') - 1; okk = (ki >= 0) & valid[np.maximum(ki, 0)] if len(kf) else np.zeros(len(w['tg']), bool)
    best = np.where(okk, b0[np.maximum(ki, 0)] if len(kf) else -1, -1).tolist(); bopp = np.where(okk, opp[np.maximum(ki, 0)] if len(kf) else -1, -1).tolist()
    tt = w['anc'] - shf; kt = tt < t_c
    return dict(tf=tf, kf=kf, P5=P5, S5=S5, opp=opp, best=best, bopp=bopp, tt=tt[kt], ptl=[x for x, k_ in zip(w['ptl'], kt) if k_],
                shl=[x for x, k_ in zip(w['shl'], kt) if k_])


def prefix(F, O, t_c, tt):
    return ([(round(o['ts'], 6), o['pt'], round(o['ta'], 6)) for o in O if o['ts'] < t_c], sorted(round(o['tc'], 6) for o in O if o['tc'] < t_c),
            [(round(tt[f[0]], 6), f[1], round(f[2], 9)) for f in F if tt[f[0]] < t_c - 100.0])


# ---------------------------------------------------------------- self-tests
def selftest():
    bad = []

    def chk(n, c):
        if not c:
            bad.append(n)
    X = Q.syn([(0, [(50, 10)], 60), (100, [(51, 5)], 60), (300, [(51, 8)], 60)], [(5, 49, 1)], kdelay=60.0)
    b0, valid, rs, kind = Q.runs(X['tf'], X['P5'], X['opp'], 0); bi = Q.fb_births(X['tf'], b0, valid, 0.0, 600.0, 0)
    for f, p, want_td in ((F_MIN, 10.0, 100.0 + F_MIN), (30.0, 20.0, 130.0), ('cur', 40.0, 160.0)):
        sch = schedule(X['tf'], X['kf'], b0, bi, f, p)
        _, O = sim5(X['side'], X['tf'], X['opp'], X['P5'], X['S5'], {}, X['tau'], X['v'], X['s'], X['enter'], X['best'], X['bopp'], X['tt'], X['tp'], X['tq'], 'M', X['t_end'], sched=sch)
        e = [o for o in O if o['pt'] == 51]
        chk(f'T1 decision at tb + f ({f}, {p})', len(sch) == 1 and abs(sch[0][0] - want_td) < 1e-9 and sch[0][0] >= 100.0)
        chk(f'T2 live at tb + f + p ({f}, {p})', len(e) >= 1 and abs(e[0]['ta'] - (want_td + p)) < 1e-9)
    _, O = sim5(X['side'], X['tf'], X['opp'], X['P5'], X['S5'], {}, X['tau'], X['v'], X['s'], X['enter'], X['best'], X['bopp'], X['tt'], X['tp'], X['tq'], 'M', X['t_end'],
                sched=[(120.0, 51, 160.0, 10.0)])
    o = [x for x in O if x['pt'] == 51][0]
    chk('T3 pending-cancel window scales with p', abs(o['pend'] - (L_PENDCANCEL - L_PLACE + 10.0)) < 1e-9)
    tf = np.array([0.0, 100.0, 200.0]); kf = tf + 50.0; b0r = np.array([50, 51, 52]); vr = np.array([True, True, True])
    rj = rj_schedule(tf, kf, b0r, vr, 1, 210.0, 211.0, np.random.default_rng(0))
    chk('T4 random control uses the frame <= u - 15.2', len(rj) == 1 and rj[0][1] == 51 and rj[0][3] == 10.0)
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_EARLY', 'MUT_NOPLACE'):
        old = globals()[nm]; globals()[nm] = 1
        try:
            out[nm] = len(selftest()) > 0
        finally:
            globals()[nm] = old
    return out


def guard_selftest():
    bad = selftest(); mut = mutation_check()
    if bad or not all(mut.values()):
        sys.exit(f'SELFTEST FAILED -> no output: {bad}; planted errors caught {mut}')
    return mut


def agebin(a):
    for nm, lo, hi in BINS:
        if a < hi:
            return nm
    return BINS[-1][0]


# ---------------------------------------------------------------- one replay day
def run_day(day):
    global MUT_EARLY
    mut = guard_selftest(); os.makedirs(OUTC, exist_ok=True)
    D = K.load_day(day); st = D['stat']; T0 = time.time()
    st.update(window_sides=0, births=0, rj=0, reg_differ=0, qr2_old_differ=0, causal_checked=0, causal_differ=0, causal_planted_differ=0)
    q2 = json.load(open(f'{QR2D}/day_{day}.json'))['acc']; OLD = {}
    A = {}; DL = {}
    rng = np.random.default_rng(int(day) + 3); rngc = np.random.default_rng(int(day) + 4)

    def add(key, F, r):
        e = A.setdefault(key, dict(fills=0, q=0.0, v1=0.0, pend_q=0.0, pend_v1=0.0, **{f'q|{b[0]}': 0.0 for b in BINS}, **{f'v1|{b[0]}': 0.0 for b in BINS}))
        for f, qq, m1 in zip(F, r['q'], r['m1']):
            vv = float(qq * m1) if np.isfinite(m1) else 0.0; bn = agebin(f[4])
            e['fills'] += 1; e['q'] += float(qq); e['v1'] += vv; e[f'q|{bn}'] += float(qq); e[f'v1|{bn}'] += vv
            if f[5] == 'pend':
                e['pend_q'] += float(qq); e['pend_v1'] += vv
    for w in K.window_sides(D):
        st['window_sides'] += 1
        side = w['side']; tf = w['tf']; tfl = tf.tolist(); cache = w.setdefault('cache', {})
        b0, valid, rs, kind = Q.runs(tf, w['P5'], w['opp'], side)
        bi = Q.fb_births(tf, b0, valid, w['ws'], w['te'], side); st['births'] += int(len(bi))
        SCH = {arm: schedule(tf, w['kf'], b0, bi, f, p) for arm, (f, p) in zip(ARMS, [(f, p) for f in FS for p in PS])}
        SCH['RJ(15.2,10)'] = rj_schedule(tf, w['kf'], b0, valid, len(bi), w['ws'], w['te'], rng); st['rj'] += len(SCH['RJ(15.2,10)'])
        old = [(float(tf[i] + Q2.D_F - Q2.L_PLACE), int(b0[i]), float(w['kf'][i]), L_PLACE) for i in bi]
        for q_, shf in CFG:
            tt = w['anc'] - shf; key = f'{q_}|{shf:.0f}'
            a = (side, tf, w['opp'], w['P5'], w['S5'], cache, w['tg'], w['vkl'], w['stv3'], w['P90'], w['best'], w['bopp'], tt, w['ptl'], w['shl'], q_, w['te'])
            F0, O0 = sim5(*a); F2, O2 = K.sim2('join', *a)
            st['reg_differ'] += int(F0 != F2 or Q.otuples(O0, w['te']) != O2)
            add(f'J-v3|{key}', F0, K.vals(w, F0, tt))
            if key in ('M|0', 'M|50', 'M|250'):
                Fo, Oo = sim5(*a, sched=old); r = K.vals(w, Fo, tt); e = OLD.setdefault(key, dict(fills=0, q=0.0, v1=0.0))
                e['fills'] += len(Fo); e['q'] += float(r['q'].sum()); e['v1'] += float(np.nansum(r['q'] * r['m1']))
            for arm, sch in SCH.items():
                F, O = sim5(*a, sched=sch); add(f'{arm}|{key}', F, K.vals(w, F, tt))
                if (q_, shf) == PRIM:
                    DL.setdefault(arm, []).extend(o['ta'] - o['tb'] for o in O if 'tb' in o and np.isfinite(o['tb']) and not o['ref'])
            if day == DAYS9[0] and st['window_sides'] <= N_CAUSAL and (q_, shf) == PRIM:     # control iii
                cand = [float(tf[i]) for i in bi if w['ws'] + 1000.0 < tf[i] < w['te'] - 1000.0 and (lambda g_: g_ >= 0 and bool(w['stv3'][g_]) and w['vkl'][g_] > w['P90'])(
                    bisect.bisect_right(w['tg'].tolist(), float(tf[i]) - 10.0) - 1)]
                t_c = (float(rngc.choice(cand)) - 5.0) if cand else float(rngc.uniform(w['ws'] + 1000.0, w['te'] - 1000.0))
                st['causal_at_birth'] = st.get('causal_at_birth', 0) + int(bool(cand)); T = truncate(w, t_c, shf)
                b0t, vt, rst, _ = Q.runs(T['tf'], T['P5'], T['opp'], side); bit = Q.fb_births(T['tf'], b0t, vt, w['ws'], w['te'], side)
                at = (side, T['tf'], T['opp'], T['P5'], T['S5'], {}, w['tg'], w['vkl'], w['stv3'], w['P90'], T['best'], T['bopp'], T['tt'], T['ptl'], T['shl'], q_, w['te'])
                ttl_full = tt.tolist(); ttl_cut = T['tt'].tolist()
                for arm, (f, p) in list(zip(ARMS, [(f, p) for f in FS for p in PS])) + [('J-v3', (None, None))]:
                    if arm == 'J-v3':
                        Fa, Oa = sim5(*a); Fb, Ob = sim5(*at)
                    else:
                        Fa, Oa = sim5(*a, sched=SCH[arm]); Fb, Ob = sim5(*at, sched=schedule(T['tf'], T['kf'], b0t, bit, f, p))
                    st['causal_checked'] += 1; st['causal_differ'] += int(prefix(Fa, Oa, t_c, ttl_full) != prefix(Fb, Ob, t_c, ttl_cut))
                MUT_EARLY = 1
                try:
                    Fa, Oa = sim5(*a, sched=schedule(tf, w['kf'], b0, bi, 30.0, 40.0)); Fb, Ob = sim5(*at, sched=schedule(T['tf'], T['kf'], b0t, bit, 30.0, 40.0))
                finally:
                    MUT_EARLY = 0
                st['causal_planted_differ'] += int(prefix(Fa, Oa, t_c, ttl_full) != prefix(Fb, Ob, t_c, ttl_cut))
        w.pop('cache', None)
        if st['window_sides'] % 80 == 0:
            r_ = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r_:.0f} MiB', flush=True)
            if r_ > 2048.0:
                sys.exit(f'REFUSED: resident memory {r_:.0f} MiB')
    bad = {}
    if st['reg_differ']:
        bad['reg_differ'] = st['reg_differ']
    for key, e in OLD.items():
        ref = q2[f'F30|{key}']
        if ref['fills'] != e['fills'] or abs(ref['q'] - e['q']) > 1e-6 or abs(ref['v1'] - e['v1']) > 1e-6:
            bad[f'qr2_old_schedule {key}'] = (ref['fills'], e['fills'])
    if day == DAYS9[0] and (st['causal_differ'] or st['causal_planted_differ'] == 0):
        bad['causal'] = (st['causal_differ'], st['causal_planted_differ'])
    if bad:
        sys.exit(f'CS CONTROL FAILED on {day} -> no result: {bad}')
    json.dump(dict(day=day, days=len(D['W0']) / 288.0, stat=st, planted=mut, acc=A, delay_p50={k_: float(np.median(v_)) if v_ else NAN for k_, v_ in DL.items()},
                   delay_n={k_: len(v_) for k_, v_ in DL.items()}), open(f'{OUTC}/day_{day}.json', 'w'), default=float)
    print(f'{day}: done in {time.time() - T0:.0f} s; controls {json.dumps({k_: st[k_] for k_ in ("reg_differ", "causal_checked", "causal_differ", "causal_planted_differ")})}', flush=True)


# ---------------------------------------------------------------- report
def report():
    DY = {d: json.load(open(f'{OUTC}/day_{d}.json')) for d in DAYS9}
    nd = sum(DY[d]['days'] for d in DAYS9); L = []; P = L.append
    RES = dict(as_of=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), days=DAYS9, b1a_precision=B1A)
    st = {k_: sum(DY[d]['stat'].get(k_, 0) for d in DAYS9) for k_ in ('window_sides', 'births', 'rj', 'reg_differ', 'causal_checked', 'causal_differ', 'causal_planted_differ')}
    RES['stat'] = st
    ALL = ['J-v3'] + ARMS + ['RJ(15.2,10)']
    g = lambda d, arm, key: DY[d]['acc'][f'{arm}|{key}']
    oldv = lambda e: e[f'v1|100-500 ms'] + e[f'v1|>= 500 ms']
    wtv = lambda e, key: sum(e[f'v1|{b[0]}'] * p for b, p in zip(BINS, B1A[key]))
    dayv = lambda arm, key, fn: np.array([fn(g(d, arm, key)) / 100.0 / DY[d]['days'] for d in DAYS9])

    def pool(arm, key):
        e = {}
        for d in DAYS9:
            for k_, v_ in g(d, arm, key).items():
                e[k_] = e.get(k_, 0.0) + v_
        return e
    dl = {arm: float(np.nanmedian([DY[d]['delay_p50'].get(arm, NAN) for d in DAYS9])) for arm in ARMS + ['RJ(15.2,10)']}
    RES['delay_p50'] = dl
    P(f'CS causal speed - 9 replay days ({nd:.2f} day-equivalents), first 60 s; as of {RES["as_of"]}')
    P(f'controls: v3 lane = sim2 join differ {st["reg_differ"]}; old schedule = QR2 F30 (checked per day); causal prefix checks {st["causal_checked"]}, '
      f'differ {st["causal_differ"]}; planted acausal schedule differs on {st["causal_planted_differ"]} window-sides; births {st["births"]}, random joins {st["rj"]}')
    for q_, shf in CFG:
        key = f'{q_}|{shf:.0f}'
        P(''); P(f'{key}: $/day at 1 s - all | < 100 ms | 100-500 ms | >= 500 ms | w/o young | B1a-weighted | pending share of shares | c/sh all, w/o young | live delay p50')
        for arm in ALL:
            e = pool(arm, key); RES[f'{arm}|{key}'] = e
            wo_q = e['q|100-500 ms'] + e['q|>= 500 ms']
            P(f'  {arm:<12} {e["v1"] / 100 / nd:>7.2f} | {e["v1|< 100 ms"] / 100 / nd:>6.2f} | {e["v1|100-500 ms"] / 100 / nd:>6.2f} | {e["v1|>= 500 ms"] / 100 / nd:>6.2f} | '
              f'{oldv(e) / 100 / nd:>6.2f} | {wtv(e, key) / 100 / nd:>6.2f} | {e["pend_q"] / max(e["q"], 1e-9):>5.1%} | {e["v1"] / max(e["q"], 1e-9):+.2f}, '
              f'{oldv(e) / max(wo_q, 1e-9):+.2f} | {dl.get(arm, NAN):>5.0f} ms')

    def rule(a, b, fn, keys=('M|0', 'M|50')):
        up = [int(np.sum(dayv(a, k_, fn) > dayv(b, k_, fn))) for k_ in keys]; dn = [int(np.sum(dayv(a, k_, fn) < dayv(b, k_, fn))) for k_ in keys]
        return ('GAIN' if min(up) >= 7 else ('LOSS' if min(dn) >= 7 else 'NOT-SHOWN')), up, dn
    P('')
    lab, up, dn = rule('E(cur,40)', 'J-v3', oldv); RES['R320'] = dict(label=lab, days_up=up, days_down=dn)
    P(f'R320 NOW (E(cur,40) vs J-v3, w/o young): {lab}; days above at M 0 / M 50 {up}, below {dn}')
    RES['R321'] = {}; RES['R322'] = {}; RES['R323'] = {}; gains = []
    for arm in ARMS:
        lab, up, dn = rule(arm, 'J-v3', oldv); RES['R321'][arm] = dict(label=lab, days_up=up, days_down=dn,
                                                                        diff_usd=[float((dayv(arm, k_, oldv) - dayv('J-v3', k_, oldv)).mean()) for k_ in ('M|0', 'M|50')])
        if lab == 'GAIN':
            gains.append((dl[arm], arm))
        upw = [int(np.sum(np.array([wtv(g(d, arm, k_), k_) / DY[d]['days'] for d in DAYS9]) > np.array([wtv(g(d, 'J-v3', k_), k_) / DY[d]['days'] for d in DAYS9]))) for k_ in ('M|0', 'M|50')]
        dnw = [int(np.sum(np.array([wtv(g(d, arm, k_), k_) / DY[d]['days'] for d in DAYS9]) < np.array([wtv(g(d, 'J-v3', k_), k_) / DY[d]['days'] for d in DAYS9]))) for k_ in ('M|0', 'M|50')]
        RES['R322'][arm] = dict(label='GAIN' if min(upw) >= 7 else ('LOSS' if min(dnw) >= 7 else 'NOT-SHOWN'), days_up=upw, days_down=dnw)
        n250 = int(np.sum(dayv(arm, 'M|250', oldv) > dayv('J-v3', 'M|250', oldv)))
        RES['R323'][arm] = dict(label='HOLDS-AT-250' if n250 >= 7 else 'NOT-AT-250', days_above=n250)
        P(f'R321 {arm:<11} (live p50 {dl[arm]:>4.0f} ms): {lab:<9} days above {up}, below {dn}, mean diff w/o young M 0 / M 50 '
          + ' / '.join(f'{x:+.2f}' for x in RES['R321'][arm]['diff_usd']) + f' $/day | R322 weighted {RES["R322"][arm]["label"]} {upw} | R323 {RES["R323"][arm]["label"]} ({n250} of 9)')
    RES['SPEED_NEEDED'] = max(gains)[1] if gains else 'NONE'
    P(f'SPEED-NEEDED: {RES["SPEED_NEEDED"]}' + (f' (live p50 {max(gains)[0]:.0f} ms)' if gains else ''))
    lab, up, dn = rule('E(15.2,10)', 'RJ(15.2,10)', oldv)
    RES['R324'] = dict(label={'GAIN': 'BIRTH-MATTERS', 'LOSS': 'RANDOM-BETTER'}.get(lab, lab), days_up=up, days_down=dn)
    P(f'R324 BIRTH (E(15.2,10) vs RJ(15.2,10), w/o young): {RES["R324"]["label"]}; days above {up}, below {dn}')
    json.dump(RES, open(f'{OUTC}/cs.json', 'w'), indent=1, default=float)
    open(f'{OUTC}/cs.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        bad = selftest(); mut = mutation_check(); print('failing:', bad, 'planted errors caught:', mut)
        sys.exit(0 if not bad and all(mut.values()) else 1)
    elif cmd == 'day':
        run_day(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit(__doc__)
