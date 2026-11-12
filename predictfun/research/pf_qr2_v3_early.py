"""QR2, v3 joining earlier (2026-11-12, session ce68ebe0). USER: "You mean we can improve by joining earlier now" / "Yea we do the
test to check the performance first, then check what's the remaining gap to top makers".

QR (a3f93fff) found that the queue ahead of v3's orders mostly joins after we could have been there (41%), that v3's fills on
levels >= 0.5 s old and on re-placed orders earn about 0 c/sh, and that v3 already joins every new best level while its guard is on
(~110 ms after the birth). Its late joins come from the guard turning on at an old level and from re-joining after a fill or a guard
flicker. QR2 changes those, one at a time and together, on the 9 consumed replay days (first 60 s, one 5-sh order per side).
POST HOC: the 500 ms age cut and the 2-step debounce were chosen after QR's tables on these same days; the -0.10 c exit is the
review's 90/10 line (11-08). A pass makes a candidate for a freeze and a forward test, nothing more.

Arms (lane v3x = v3's join lane, pf_kp_s4.sim2 'join', with three switches; all off = v3 exactly):
 J-v3     v3.
 Y        entries at grid steps only while the known level is young: the stamp of the first frame of the current run of frames with
          this best (frames known at the step, venue clock) is <= 500 ms before the step; a level the best since the frames' start
          counts as old.
 K        flicker hold: an order is cancelled for the state being off only after it has been off at 2 consecutive grid steps (100 ms).
 K2       90/10 exit: an order is cancelled for the guard only when v <= -0.10 c (entries still need v above PM90).
 F30      also join every new best level (venue frame where the best of our side improves, as QR's FB) live 30 ms after its birth
          stamp, if v3's state is on and v > PM90 at the last grid step; an order at another price is cancelled at that moment;
          QR's grace (no price exit until the current clock has seen the birth frame).
 Y+F30, QR2-now = Y+K (no speed change), QR2-fast = Y+K+F30.
 The price rules are v3's for every arm (cancel when the known best moves and v > PM90 or it is not exactly one tick better).
 Null for Y: 200 draws per window-side at M 0 and M 50 that skip v3's grid entry steps at random with probability q = the share of
 J-v3's own entry steps that Y would block there (matched on the expected share of entries removed; realised entries printed).
Configs M 0 / M 50 / M 250 / L 50 / U 50 (primary M 50). Values as QR (1 s markout, $/day = sum(shares x c/sh) / 100 / days).
QR's queue decomposition is printed for J-v3, QR2-now and QR2-fast at M 50.

Readings DECLARED BEFORE THE FIRST RUN (evaluated in code; day rule = >= 7 of 9 days):
 R290 NOW (QR2-now vs J-v3): $/day at 1 s above J-v3 by the day rule at M 0 and at M 50, and QR2-now's pooled c/sh at 1 s > 0 at
      both -> IMPROVES-NOW; below by the day rule at both -> WORSE-NOW; else NOT-SHOWN.
 R291 FAST (QR2-fast vs QR2-now): the same rule -> SPEED-ADDS / SPEED-COSTS / NOT-SHOWN.
 R292 PARTS (Y, K, K2, F30, Y+F30, each vs J-v3): R290's rule -> IMPROVES / WORSE / NOT-SHOWN per arm.
 R293 AGE-NULL (Y): pooled $/day at 1 s above the p95 of the 200 random-skip draws at M 0 and at M 50 -> AGE-SELECTS; else NOT-AGE.
 R294 AT-250 (QR2-now and QR2-fast vs J-v3 at M 250, day rule): HOLDS-AT-250 / NOT-AT-250 (printed with every claim).
What the outcomes mean, fixed in advance: IMPROVES-NOW = v3 earns more in this replay by not joining old levels and not cancelling on
a one-step flicker, at today's speed; SPEED-ADDS = a 30 ms entry at new levels adds to that; NOT-AGE = the age cut's gain is not
beyond skipping entries at random. Every positive result is replay-only (rule M unvalidated; young orders are where B1a found the
replay's fills mostly phantom) and on consumed days.

Controls (any failure -> no table): (i) v3x with every switch off equals pf_kp_s4.sim2 'join' fill for fill and order for order on
every window-side and config; (ii) off-parity through the override paths: an explicit all-true entry mask with a 1-step debounce
and an empty schedule, and the age mask computed with an infinite cut and censored runs counted young, each equal J-v3; (iii) the regression can fail: L_CANCEL 30
differs on day 1; (iv) synthetic cases with known answers (debounce keeps through one off step, the -0.10 exit, the entry mask, the
fast entry's live time and grace, the age mask) and planted errors (debounce ignored, mask ignored, no grace, L_PLACE 10) that must
each fail a case. Statuses printed.
Usage: research/run_qr2.sh; python3 predictfun/research/pf_qr2_v3_early.py selftest | day <yyyymmdd> | report
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

INF = float('inf'); NAN = float('nan')
L_PLACE, L_CANCEL, L_PENDCANCEL, L_FILL = K.L_PLACE, K.L_CANCEL, K.L_PENDCANCEL, K.L_FILL
SIZE = K.SIZE
DAYS9 = K.DAYS9; OUT2 = f'{K.IO}/qr2_20261112'
CFG = [('M', 0.0), ('M', 50.0), ('M', 250.0), ('L', 50.0), ('U', 50.0)]
PRIM = ('M', 50.0); NULL_CFG = [('M', 0.0), ('M', 50.0)]; NULL_DRAWS = 200
Y_MAX = 500.0; DEB = 2; EXIT_K2 = -0.10; D_F = 30.0
ARMS = {'J-v3': dict(), 'Y': dict(y=1), 'K': dict(deb=DEB), 'K2': dict(ex=EXIT_K2), 'F30': dict(f=1), 'Y+F30': dict(y=1, f=1),
        'QR2-now': dict(y=1, deb=DEB), 'QR2-fast': dict(y=1, deb=DEB, f=1)}
PARITY = ('OFF-explicit', 'Y-inf')
MUT_DEB_IGNORED = MUT_ALLOW_IGNORED = MUT_NO_GRACE = 0    # planted errors (self-test only)
MEM_CAP_MIB = 2048.0


def sim4(side, tf, opp, P5, S5, cache, tau, v, s, enter, best, bopp, tt, tp, tq, qrule, t_end, allow=None, deb=1, exit_line=None,
         sched=None):
    """v3's join lane (pf_kp_s4.sim2 'join') with an entry mask, a flicker debounce, an exit line and fast entries. Returns fills
    (sim2's tuples) and orders (dicts)."""
    for a in (tf, tau, tt):
        if len(a) > 1 and (np.diff(a) < 0).any():
            raise ValueError('REFUSED: frames, grid or trades not time-sorted')
    tfl = tf.tolist(); nf = len(tfl); ttl = tt.tolist(); nt = len(ttl); taul = tau.tolist()
    fills = []; orders = []; live = []; cur = None; it = 0; offrun = 0

    def lv(pt):
        if pt not in cache:
            last = P5[:, 4]
            cache[pt] = np.where((last >= 0) & ((pt < last) if side == 0 else (pt > last)), np.nan, (S5 * (P5 == pt)).sum(1))
        return cache[pt]

    def place(pt, t, gr=-INF):
        ta = t + L_PLACE; fi = bisect.bisect_right(tfl, ta) - 1
        o = dict(id=len(orders), pt=pt, ts=t, ta=ta, tr=INF, tc=INF, rem=SIZE, ahead=0.0, ci=fi, tfill=INF, a0=np.nan, a0n=np.nan, ref=False, gr=gr)
        if fi < 0 or ((0 <= opp[fi] <= pt) if side == 0 else (opp[fi] >= pt)):
            o['ref'] = True; o['tr'] = ta
        else:
            a = lv(pt)[fi]; a = float(a) if np.isfinite(a) else 0.0; o['a0n'] = a
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
        if typ == 1:                                      # fast entry at a new level
            pt_, gr_ = sched[k][1], sched[k][2]; g = bisect.bisect_right(taul, t) - 1
            if g >= 0 and bool(s[g]) and v[g] > enter:
                if cur is not None and cur['pt'] != pt_:
                    cancel(cur, t); cur = None
                if cur is None:
                    cur = place(int(pt_), t, gr_)
            continue
        b = best[k]; ab = v[k] > enter; on = bool(s[k])
        offrun = 0 if on else offrun + 1
        if cur is not None:
            off_kill = (not (v[k] > exit_line)) if exit_line is not None else (offrun >= (1 if MUT_DEB_IGNORED else deb))
            if b < 0 or off_kill:
                kill = True
            elif t < cur['gr'] and not MUT_NO_GRACE:
                kill = False
            elif b != cur['pt']:
                kill = ab or ((b - cur['pt']) if side == 0 else (cur['pt'] - b)) != 1
            else:
                kill = False
            if kill:
                cancel(cur, t); cur = None
        if cur is None and on and ab and b >= 0 and (allow is None or allow[k] or MUT_ALLOW_IGNORED):
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


def young_mask(tau, jk, tf, rs, ymax, censored_old=True):
    """Per grid step: the known level (frame jk[k]) became the best <= ymax ms before the step (venue stamps); censored runs (the
    best since the frames' start) count as old unless censored_old is False (the parity control)."""
    out = []
    for t, j in zip(tau.tolist(), jk.tolist()):
        out.append(bool(j >= 0 and (rs[j] > 0 or not censored_old) and t - tf[rs[j]] <= ymax))
    return out


# ---------------------------------------------------------------- self-tests
def run4(X, qrule='M', **kw):
    return sim4(X['side'], X['tf'], X['opp'], X['P5'], X['S5'], {}, X['tau'], X['v'], X['s'], X['enter'], X['best'], X['bopp'], X['tt'],
                X['tp'], X['tq'], qrule, X['t_end'], **kw)


def selftest():
    bad = []

    def chk(name, cond):
        if not cond:
            bad.append(name)
    fr = [(0, [(50, 10), (49, 5)], 60), (100, [(51, 4), (50, 12)], 60), (180, [(51, 9), (50, 12)], 60), (260, [(50, 12)], 60),
          (330, [(52, 3), (50, 12)], 60), (420, [(52, 7)], 60), (500, [(51, 6)], 60)]
    tr = [(120, 51, 2), (200, 51, 12), (250, 50, 3), (300, 49, 6), (360, 52, 4), (450, 52, 9), (520, 51, 20)]
    for kd in (0.0, 30.0):
        X = Q.syn(fr, tr, v=[1, 1, 1, -1, 1, 1, 1, 1, 1, 1], kdelay=kd)
        for q in ('M', 'L', 'U'):
            F2, O2 = Q.run2(X, 'join', q); F4, O4 = run4(X, q)
            chk(f'T1 v3x off = sim2 join {q} k{kd:.0f}', F2 == F4 and O2 == Q.otuples(O4, X['t_end']) and len(F2) > 0)
            F4b, O4b = run4(X, q, allow=[True] * len(X['tau']), deb=1, sched=[])
            chk(f'T1 off-explicit {q} k{kd:.0f}', F2 == F4b and O2 == Q.otuples(O4b, X['t_end']))
    flat = [(0, [(50, 10)], 60), (600, [(50, 10)], 60)]
    X = Q.syn(flat, [(5, 49, 1)], v=[1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1], enter=0.6)
    _, O1 = run4(X); _, O2 = run4(X, deb=2)
    chk('T2 v3 re-places on a one-step flicker', len(O1) == 2)
    chk('T2 debounce keeps through it', len(O2) == 1)
    X = Q.syn(flat, [(5, 49, 1)], v=[1, 1, 1, 0.0, 1, 1, 1, 1, 1, 1, 1, 1], enter=0.6)
    _, O3 = run4(X, exit_line=EXIT_K2); _, O3b = run4(X)
    chk('T3 exit -0.10 keeps at v = 0', len(O3) == 1 and len(O3b) == 2)
    X = Q.syn(flat, [(5, 49, 1)], v=[1, 1, 1, -0.5, 1, 1, 1, 1, 1, 1, 1, 1], enter=0.6)
    _, O3c = run4(X, exit_line=EXIT_K2)
    chk('T3 exit -0.10 cancels at v = -0.5', len(O3c) == 2 and O3c[0]['tc'] == 150.0)
    X = Q.syn(flat, [(5, 49, 1)])
    _, O4 = run4(X, allow=[k >= 4 for k in range(len(X['tau']))])
    chk('T4 entry mask', len(O4) >= 1 and O4[0]['ts'] == 200.0)
    X = Q.syn([(0, [(50, 10)], 60), (100, [(51, 5)], 60), (300, [(51, 8)], 60)], [(5, 49, 1)], kdelay=60.0)
    _, O5 = run4(X, sched=[(100.0 + D_F - L_PLACE, 51, float(X['kf'][1]))])
    chk('T5 fast entry live at tb + 30 and kept through the grace', O5[0]['pt'] == 51 and O5[0]['ta'] == 130.0 and O5[0]['tc'] == X['t_end'])
    X = Q.syn([(0, [(50, 1)], 60), (100, [(51, 1)], 60), (700, [(51, 1)], 60)], [(5, 49, 1)], t_end=1000.0)
    b0, valid, rs, kind = Q.runs(X['tf'], X['P5'], X['opp'], 0)
    jk = np.searchsorted(X['kf'], X['tau'], side='right') - 1
    ym = young_mask(X['tau'], jk, X['tf'].tolist(), rs, Y_MAX)
    chk('T6 age mask', ym[0] is False and ym[2] is True and ym[12] is True and ym[13] is False and ym[19] is False)
    chk('T6 infinite cut = censored only', young_mask(X['tau'], jk, X['tf'].tolist(), rs, INF)[0] is False and all(young_mask(X['tau'], jk, X['tf'].tolist(), rs, INF)[2:]))
    chk('T6 parity mask all true', all(young_mask(X['tau'], jk, X['tf'].tolist(), rs, INF, censored_old=False)))
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_DEB_IGNORED', 'MUT_ALLOW_IGNORED', 'MUT_NO_GRACE', 'L_PLACE'):
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
def run_day(day):
    global L_CANCEL
    mut = guard_selftest(); os.makedirs(OUT2, exist_ok=True)
    D = K.load_day(day); st = D['stat']; T0 = time.time()
    st.update(window_sides=0, births=0, reg_differ=0, parity_differ=0, lcancel_differ=0)
    ACC = {}; NUL = {f'{q}|{s_:.0f}': np.zeros((NULL_DRAWS, 4)) for q, s_ in NULL_CFG}; DEC = {}
    rng = np.random.default_rng(int(day) + 2)
    for w in K.window_sides(D):
        st['window_sides'] += 1
        side = w['side']; tf = w['tf']; tfl = tf.tolist(); cache = w.setdefault('cache', {})
        b0, valid, rs, kind = Q.runs(tf, w['P5'], w['opp'], side)
        young = young_mask(w['tg'], w['jk'], tfl, rs, Y_MAX); yinf = young_mask(w['tg'], w['jk'], tfl, rs, INF, censored_old=False)
        bi = Q.fb_births(tf, b0, valid, w['ws'], w['te'], side); st['births'] += int(len(bi))
        sch = [(float(tf[i] + D_F - L_PLACE), int(b0[i]), float(w['kf'][i])) for i in bi]
        for q, shf in CFG:
            tt = w['anc'] - shf; key = f'{q}|{shf:.0f}'
            a = (side, tf, w['opp'], w['P5'], w['S5'], cache, w['tg'], w['vkl'], w['stv3'], w['P90'], w['best'], w['bopp'], tt, w['ptl'], w['shl'], q, w['te'])
            F2, O2 = K.sim2('join', *a)
            res = {}
            for arm, sw in ARMS.items():
                F, O = sim4(*a, allow=young if sw.get('y') else None, deb=sw.get('deb', 1), exit_line=sw.get('ex'), sched=sch if sw.get('f') else None)
                res[arm] = (F, O)
                r = K.vals(w, F, tt); e = ACC.setdefault(f'{arm}|{key}', dict(orders=0, ref=0, fills=0, q=0.0, v1=0.0, v5=0.0))
                e['orders'] += len(O); e['ref'] += sum(1 for o in O if o['ref']); e['fills'] += len(F); e['q'] += float(r['q'].sum())
                e['v1'] += float(np.nansum(r['q'] * r['m1'])); e['v5'] += float(np.nansum(r['q'] * r['m5']))
            st['reg_differ'] += int(res['J-v3'][0] != F2 or Q.otuples(res['J-v3'][1], w['te']) != O2)
            Fo, Oo = sim4(*a, allow=[True] * w['n'], deb=1, sched=[])
            st['parity_differ'] += int(Fo != F2 or Q.otuples(Oo, w['te']) != O2)
            Fy, Oy = sim4(*a, allow=yinf)
            st['yinf_differ'] = st.get('yinf_differ', 0) + int(Fy != F2 or Q.otuples(Oy, w['te']) != O2)
            if day == DAYS9[0] and st['window_sides'] <= 40 and (q, shf) == PRIM:
                L_CANCEL = 30.0
                try:
                    Fm, Om = sim4(*a)
                finally:
                    L_CANCEL = K.L_CANCEL
                st['lcancel_differ'] += int(Fm != F2 or Q.otuples(Om, w['te']) != O2)
            if (q, shf) in NULL_CFG:
                ent = [o['ts'] for o in res['J-v3'][1]]; kix = np.searchsorted(w['tg'], ent)
                blocked = [not young[min(k_, w['n'] - 1)] for k_ in kix]
                qb = float(np.mean(blocked)) if blocked else 0.0
                for dr in range(NULL_DRAWS):
                    msk = (rng.random(w['n']) >= qb).tolist()
                    F, O = sim4(*a, allow=msk); r = K.vals(w, F, tt)
                    NUL[key][dr] += (float(r['q'].sum()), float(np.nansum(r['q'] * r['m1'])), len(F), len(O))
            if (q, shf) == PRIM:
                ttl = tt.tolist()
                for arm in ('J-v3', 'QR2-now', 'QR2-fast'):
                    O = res[arm][1]; d = DEC.setdefault(arm, dict(n=0, a0=0.0, **{c: 0.0 for c in Q.COMP}, not_best=0, censored=0, refused=0))
                    for idx, o in enumerate(O):
                        if o['ref']:
                            d['refused'] += 1; continue
                        jl = bisect.bisect_right(tfl, o['ta']) - 1
                        if jl < 0 or not valid[jl] or b0[jl] != o['pt']:
                            d['not_best'] += 1; continue
                        j0 = int(rs[jl])
                        if j0 == 0:
                            d['censored'] += 1; continue
                        tb = tfl[j0]; c, npos = Q.chain_start(O, idx); tch = max(O[c]['ta'], tb)
                        cm, gl = Q.decompose(o['ta'], o['pt'], o['a0'], o['a0n'], tb, tch, side, q, tfl, cache[o['pt']], ttl, w['ptl'], w['shl'])
                        d['n'] += 1; d['a0'] += o['a0']
                        for cc in Q.COMP:
                            d[cc] += cm[cc]
        w.pop('cache', None)
        if st['window_sides'] % 40 == 0:
            r_ = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r_:.0f} MiB', flush=True)
            if r_ > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r_:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    bad = {k_: st[k_] for k_ in ('reg_differ', 'parity_differ', 'yinf_differ') if st.get(k_)}
    if day == DAYS9[0] and st['lcancel_differ'] == 0:
        bad['lcancel_never_differs'] = 0
    if bad:
        sys.exit(f'QR2 CONTROL FAILED on {day} -> no result: {bad}')
    json.dump(dict(day=day, days=len(D['W0']) / 288.0, stat=st, planted=mut, acc=ACC, null={k_: v_.tolist() for k_, v_ in NUL.items()}, dec=DEC),
              open(f'{OUT2}/day_{day}.json', 'w'), default=float)
    print(f'{day}: done in {time.time() - T0:.0f} s; controls {json.dumps({k_: st.get(k_) for k_ in ("reg_differ", "parity_differ", "yinf_differ", "lcancel_differ")})}', flush=True)


# ---------------------------------------------------------------- report
def report():
    DY = {}
    for day in DAYS9:
        fn = f'{OUT2}/day_{day}.json'
        if not os.path.exists(fn):
            sys.exit(f'REFUSED: {fn} missing')
        DY[day] = json.load(open(fn))
    nd = sum(DY[d]['days'] for d in DAYS9); L = []; P = L.append
    RES = dict(as_of=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), days=DAYS9, day_equiv=nd)
    st = {k_: sum(DY[d]['stat'].get(k_, 0) for d in DAYS9) for k_ in ('window_sides', 'births', 'reg_differ', 'parity_differ', 'yinf_differ')}
    RES['stat'] = st

    def pool(arm, key):
        e = dict(orders=0, ref=0, fills=0, q=0.0, v1=0.0, v5=0.0)
        for d in DAYS9:
            for k_, v_ in DY[d]['acc'][f'{arm}|{key}'].items():
                e[k_] += v_
        return e

    def dayv(arm, key):
        return np.array([DY[d]['acc'][f'{arm}|{key}']['v1'] / 100.0 / DY[d]['days'] for d in DAYS9])
    cps = lambda e: e['v1'] / e['q'] if e['q'] > 0 else NAN
    P(f'QR2 v3 joining earlier - 9 replay days ({nd:.2f} day-equivalents), first 60 s; as of {RES["as_of"]}')
    P(f'controls: v3x off = sim2 join differ {st["reg_differ"]}; off-explicit differ {st["parity_differ"]}; Y-inf differ {st["yinf_differ"]}; '
      f'L_CANCEL 30 differs on {DY[DAYS9[0]]["stat"]["lcancel_differ"]} window-sides of day 1; window-sides {st["window_sides"]}, births {st["births"]}')
    P(''); P(f'{"arm":<10} ' + ' '.join(f'{q + str(int(s_)) + " c/sh":>12} {"$/day":>8}' for q, s_ in CFG) + f' {"fills/day M50":>13} {"orders/ws":>9}')
    RES['arms'] = {}
    for arm in ARMS:
        cells = []
        for q, s_ in CFG:
            e = pool(arm, f'{q}|{s_:.0f}'); RES['arms'][f'{arm}|{q}|{s_:.0f}'] = e
            cells.append(f'{cps(e):>+12.2f} {e["v1"] / 100.0 / nd:>8.2f}')
        e = pool(arm, 'M|50')
        P(f'{arm:<10} ' + ' '.join(cells) + f' {e["fills"] / nd:>13.0f} {e["orders"] / st["window_sides"]:>9.2f}')

    def rule(a, b):
        up = [int(np.sum(dayv(a, k_) > dayv(b, k_))) for k_ in ('M|0', 'M|50')]
        dn = [int(np.sum(dayv(a, k_) < dayv(b, k_))) for k_ in ('M|0', 'M|50')]
        pos = all(cps(pool(a, k_)) > 0 for k_ in ('M|0', 'M|50'))
        lab = 'IMPROVES' if min(up) >= 7 and pos else ('WORSE' if min(dn) >= 7 else 'NOT-SHOWN')
        return lab, up, dn
    P('')
    lab, up, dn = rule('QR2-now', 'J-v3'); RES['R290'] = dict(label={'IMPROVES': 'IMPROVES-NOW', 'WORSE': 'WORSE-NOW'}.get(lab, lab), days_up=up, days_down=dn)
    P(f'R290 NOW (QR2-now vs J-v3): {RES["R290"]["label"]}; days above at M 0 / M 50 {up}, below {dn}')
    lab, up, dn = rule('QR2-fast', 'QR2-now'); RES['R291'] = dict(label={'IMPROVES': 'SPEED-ADDS', 'WORSE': 'SPEED-COSTS'}.get(lab, lab), days_up=up, days_down=dn)
    P(f'R291 FAST (QR2-fast vs QR2-now): {RES["R291"]["label"]}; days above {up}, below {dn}')
    RES['R292'] = {}
    for arm in ('Y', 'K', 'K2', 'F30', 'Y+F30'):
        lab, up, dn = rule(arm, 'J-v3'); RES['R292'][arm] = dict(label=lab, days_up=up, days_down=dn)
        P(f'R292 PART {arm} vs J-v3: {lab}; days above {up}, below {dn}')
    RES['R293'] = {}; ok = []
    for q, s_ in NULL_CFG:
        k_ = f'{q}|{s_:.0f}'; N = np.sum([np.array(DY[d]['null'][k_]) for d in DAYS9], axis=0)
        usd = N[:, 1] / 100.0 / nd; p95 = float(np.percentile(usd, 95)); y = pool('Y', k_)['v1'] / 100.0 / nd
        ok.append(y > p95)
        RES['R293'][k_] = dict(Y_usd=y, null_p95=p95, null_mean=float(usd.mean()), null_fills_mean=float(N[:, 2].mean() / nd), null_orders_mean=float(N[:, 3].mean() / nd),
                               Y_fills=pool('Y', k_)['fills'] / nd, Y_orders=pool('Y', k_)['orders'] / nd)
        P(f'R293 null {k_}: Y {y:.2f} $/day vs random-skip mean {usd.mean():.2f}, p95 {p95:.2f}; orders/day Y {pool("Y", k_)["orders"] / nd:.0f} vs null {N[:, 3].mean() / nd:.0f}; '
          f'fills/day Y {pool("Y", k_)["fills"] / nd:.0f} vs null {N[:, 2].mean() / nd:.0f}')
    RES['R293']['label'] = 'AGE-SELECTS' if all(ok) else 'NOT-AGE'
    P(f'R293 AGE-NULL: {RES["R293"]["label"]}')
    RES['R294'] = {}
    for arm in ('QR2-now', 'QR2-fast'):
        n_ = int(np.sum(dayv(arm, 'M|250') > dayv('J-v3', 'M|250')))
        RES['R294'][arm] = dict(label='HOLDS-AT-250' if n_ >= 7 else 'NOT-AT-250', days_above=n_)
        P(f'R294 AT-250 ({arm}): {RES["R294"][arm]["label"]} ({n_} of 9 days above J-v3 at M 250)')
    P(''); P('queue ahead at go-live, QR decomposition at M 50 (mean sh; share of a0):')
    for arm in ('J-v3', 'QR2-now', 'QR2-fast'):
        d = {}
        for dd in DAYS9:
            for k_, v_ in DY[dd]['dec'][arm].items():
                d[k_] = d.get(k_, 0) + v_
        RES.setdefault('dec', {})[arm] = d
        P(f'  {arm:<9} orders {d["n"]}, a0 {d["a0"] / d["n"]:.0f}: ' + ', '.join(f'{c} {d[c] / d["n"]:.0f} ({d[c] / d["a0"]:.0%})' for c in Q.COMP)
          + f'; not at the best {d["not_best"]}, censored {d["censored"]}, refused {d["refused"]}')
    P(''); P('$/day at 1 s per day (M 50): ' + '; '.join(f'{arm} ' + ' '.join(f'{x:.1f}' for x in dayv(arm, 'M|50')) for arm in ('J-v3', 'QR2-now', 'QR2-fast')))
    json.dump(RES, open(f'{OUT2}/qr2.json', 'w'), indent=1, default=float)
    open(f'{OUT2}/qr2.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


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
