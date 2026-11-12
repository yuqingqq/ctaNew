"""GAP, the remaining gap to the top makers (2026-11-12, session ce68ebe0). USER: "Yea we do the test to check the performance
first, then check what's the remaining gap to top makers". After QR2 (a3ff70fe).

Same 9 consumed replay days, first 60 s of each window. Theirs: the real maker legs on the public tape at the best of their side
(price = the maker side's best in the frame strictly before the leg's anchor, B12's depth 0); legs behind the best and inside the
spread are counted as statuses. Makers ranked per day by shares in t_rel 0-240 s over the day's windows (B12's rule): top 5,
6-20, rest. Signer ids are assigned per day by the step-1 build, so ranks are per day and our own account cannot be identified on
these days (if it traded it is among the ranked makers; limit). Ours: the simulated 5-sh order per side of QR2's arms J-v3, Y and
Y+F30 at M 0 / M 50 / M 250 (re-run with QR2's code; control ii pins them to QR2's committed day totals). Values for both: sg *
(mid(anchor + h) - price), h = 1 s / 5 s, with S4's vals (the same mid series and anchor). $/day = sum(shares x c/sh) / 100 / days.
Guard state at a leg: v3's state (v > PM90 and the state on) at the last grid step <= anchor - 50 ms (B12's lag 50), a step more
than 100 ms old = 'no guard value'. Taker order = legs_tk's id (legs without one are their own order).

Readings DECLARED BEFORE THE FIRST RUN (evaluated in code; day rule = >= 7 of 9 days):
 R300 PER-SHARE: the top 5's at-best c/sh at 1 s minus ours (J-v3; Y+F30 printed by the same rule), at M 0 and at M 50: > 0 by the
      day rule at both -> TOP-HIGHER; < 0 at both -> OURS-HIGHER; else MIXED.
 R301 SCALE (printed, no label): the top 5's at-best shares and $/day (in all and per maker) against ours.
 R302 WHEN: the share of the top 5's at-best $/day at 1 s earned on legs while v3's guard is OFF: > 50% pooled and on >= 7 days ->
      OFF-DOMINANT; < 50% pooled and on >= 7 days -> ON-DOMINANT; else MIXED. Their c/sh with the guard ON / OFF printed.
 R303 SAME-FLOW (J-v3, M 50): the share of our filled shares on taker orders that also filled a top-5 maker on our side at our
      price: >= 50% pooled and on >= 7 days -> SAME-FLOW; < 50% pooled and on >= 7 days -> DIFFERENT-FLOW; else MIXED. Printed: our
      c/sh on shared / not-shared fills, and the top 5's c/sh on legs of taker orders that also filled us / not.
What the outcomes mean, fixed in advance: OURS-HIGHER with a large scale gap = per share we are not behind in this replay, the gap
is presence and volume (they quote through what our guard avoids); OFF-DOMINANT = most of their money is made while our guard says
stay out; SAME-FLOW = we are filled by the trades that fill them, so the per-share gap on that flow is queue place and price, not
selection. All of ours is simulated (rule M unvalidated, B1a); all of theirs is real.

Controls (any failure -> no table): (i) the legs aligned to each window-side (with signer and taker order) have exactly the
simulator's anchors; (ii) our re-run arms reproduce QR2's committed day totals (fills, shares, value at 1 s) for J-v3, Y and Y+F30
at every config; (iii) synthetic cases: the at-best classifier (at the best / behind / inside, the frame strictly before), the
shared-flow matcher (same taker order and price -> shared; another order or price -> not), the per-day ranking, each with a planted
error that must fail (frame <= anchor, matching on price only, ranking on first-minute shares). Statuses printed.
Usage: research/run_gap.sh; python3 predictfun/research/pf_gap_top_makers.py selftest | day <yyyymmdd> | report
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
import pf_qr_queue_race as Q                              # noqa: E402
import pf_qr2_v3_early as Q2                              # noqa: E402

INF = float('inf'); NAN = float('nan')
DAYS9 = K.DAYS9; OUTG = f'{K.IO}/gap_20261112'; QR2D = f'{K.IO}/qr2_20261112'
CFG = [('M', 0.0), ('M', 50.0), ('M', 250.0)]; PRIM = ('M', 50.0)
OURS = {'J-v3': dict(), 'Y': dict(y=1), 'Y+F30': dict(y=1, f=1)}
GROUPS = ('top 5', '6-20', 'rest')
LAG = 50.0; MAX_AGE = 100.0
MUT_DEPTH_LE = MUT_PRICE_ONLY = MUT_RANK_60 = 0           # planted errors (self-test only)
MEM_CAP_MIB = 2048.0


def depth_of(tfl, P5, opp, side, anc, p):
    """Ticks behind the maker side's best in the frame strictly before anc (0 = at the best, < 0 = inside the spread);
    None if no valid frame."""
    j = (bisect.bisect_right(tfl, anc) if MUT_DEPTH_LE else bisect.bisect_left(tfl, anc)) - 1
    if j < 0:
        return None
    b = P5[j, 0]; o = opp[j]
    if b < 0 or o < 0 or not ((o > b) if side == 0 else (b > o)):
        return None
    return int((b - p) if side == 0 else (p - b))


def shared_flags(our_legs, tko, ptl, sig_top, our_pt):
    """For each of our fills (leg index): does its taker order also fill a top-5 maker on this side at our price?"""
    top_by = {}
    for j, (t_, p_, s_) in enumerate(zip(tko, ptl, sig_top)):
        if s_:
            top_by.setdefault((p_,) if MUT_PRICE_ONLY else (t_, p_), True)
    return [bool(top_by.get((our_pt[n],) if MUT_PRICE_ONLY else (tko[i], our_pt[n]))) for n, i in enumerate(our_legs)]


def rank_groups(sig, shares, t_rel):
    """Per day: makers ranked by shares in t_rel 0-240 s (B12); returns signer -> group."""
    m = (t_rel >= 0) & (t_rel < (60 if MUT_RANK_60 else 240))
    vol = pd.Series(shares[m]).groupby(sig[m]).sum().sort_values(ascending=False, kind='stable')
    return {int(s_): ('top 5' if r < 5 else ('6-20' if r < 20 else 'rest')) for r, s_ in enumerate(vol.index)}


def selftest():
    bad = []

    def chk(n, c):
        if not c:
            bad.append(n)
    tfl = [0.0, 100.0, 200.0]; P5 = np.array([[50, 49, 48, 47, 46], [51, 50, 49, 48, 47], [50, 49, 48, 47, 46]]); opp = np.array([60, 60, 60])
    chk('D1 at the best (frame before)', depth_of(tfl, P5, opp, 0, 150.0, 51) == 0)
    chk('D2 behind', depth_of(tfl, P5, opp, 0, 150.0, 50) == 1)
    chk('D3 strictly before: a frame at the anchor is not used', depth_of(tfl, P5, opp, 0, 200.0, 51) == 0)
    chk('D4 inside the spread', depth_of(tfl, P5, opp, 0, 50.0, 51) == -1)
    tko = [7, 7, 8, 9]; ptl = [50, 50, 50, 51]; top = [True, False, False, True]
    chk('S1 shared flow', shared_flags([1, 2, 3], tko, ptl, top, [50, 50, 50]) == [True, False, False])
    sig = np.array([1, 1, 2, 3, 3, 3, 4]); sh = np.array([5.0, 5.0, 30.0, 1.0, 1.0, 1.0, 20.0]); tr = np.array([10, 200, 10, 10, 10, 10, 100])
    g = rank_groups(sig, sh, tr)
    chk('R1 ranking on 0-240 s', g[2] == 'top 5' and len(g) == 4 and list(pd.Series(sh).groupby(sig).sum().sort_values(ascending=False).index[:2]) == [2, 4])
    chk('R2 a leg at 100 s counts (0-240 s, not the first minute)', g.get(4) == 'top 5')
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_DEPTH_LE', 'MUT_PRICE_ONLY', 'MUT_RANK_60'):
        old = globals()[nm]; globals()[nm] = 1
        try:
            out[nm] = len(selftest()) > 0
        finally:
            globals()[nm] = old
    return out


def guard_selftest():
    bad = selftest(); mut = mutation_check(); bad2 = Q2.selftest()
    if bad or bad2 or not all(mut.values()):
        sys.exit(f'SELFTEST FAILED -> no output: {bad} {bad2}; planted errors caught {mut}')
    return mut


def run_day(day):
    mut = guard_selftest(); os.makedirs(OUTG, exist_ok=True)
    D = K.load_day(day); st = D['stat']; T0 = time.time()
    S1d = f'{K.IO}/step1/{day}'
    full = pd.read_parquet(f'{S1d}/legs_timed2.parquet', columns=['mid', 'signer', 'shares', 't_rel'])
    tkf = pd.read_parquet(f'{S1d}/legs_tk.parquet', columns=['tk']).tk.values
    if len(tkf) != len(full):
        sys.exit('REFUSED: legs_tk not aligned with legs_timed2')
    inw = full.mid.isin(list(D['W0'])).values
    G = rank_groups(full.signer.values[inw], full.shares.values[inw].astype(float), full.t_rel.values[inw].astype(float))
    SIG = full.signer.values.astype('int64'); TK = np.where(tkf >= 0, tkf, -1 - np.arange(len(tkf)))
    del full
    q2 = json.load(open(f'{QR2D}/day_{day}.json'))['acc']
    st.update(window_sides=0, align_differ=0, legs_0_60=0, at_best=0, behind=0, inside=0, no_frame=0, nan1=0, guard_on=0, guard_off=0, guard_none=0)
    TH = {(g_, gs): dict(legs=0, q=0.0, v1=0.0, v5=0.0) for g_ in GROUPS for gs in ('ALL', 'ON', 'OFF', 'none')}
    TH.update({('top 5', f'shared|{arm}'): dict(legs=0, q=0.0, v1=0.0, v5=0.0) for arm in OURS})
    TH.update({('top 5', f'notshared|{arm}'): dict(legs=0, q=0.0, v1=0.0, v5=0.0) for arm in OURS})
    OU = {}
    for w in K.window_sides(D):
        st['window_sides'] += 1
        side = w['side']; tf = w['tf']; tfl = tf.tolist(); cache = w.setdefault('cache', {})
        if (w['m'], side) in D['LI']:
            l = D['LT'].iloc[D['LI'][(w['m'], side)]]
            o_ = np.lexsort(((-l.ptick.values if side == 0 else l.ptick.values), l.anc.values)); ix = l.index.values[o_]
            st['align_differ'] += int(not np.array_equal(l.anc.values[o_], w['anc']))
        else:
            ix = np.zeros(0, 'int64')
        sig = SIG[ix]; tko = TK[ix]; grp = [G.get(int(s_), 'rest') for s_ in sig]
        legs = [i for i in range(len(ix)) if w['ws'] <= w['anc'][i] < w['te']]
        st['legs_0_60'] += len(legs)
        rl = K.vals(w, [(i, w['ptl'][i], w['shl'][i]) for i in legs], w['anc'])
        top_at = [False] * len(ix); legrows = []; tgl = w['tg'].tolist()
        for n, i in enumerate(legs):
            dp = depth_of(tfl, w['P5'], w['opp'], side, w['anc'][i], w['ptl'][i])
            if dp is None:
                st['no_frame'] += 1; continue
            if dp < 0:
                st['inside'] += 1; continue
            if dp > 0:
                st['behind'] += 1; continue
            st['at_best'] += 1
            g_ = bisect.bisect_right(tgl, w['anc'][i] - LAG) - 1
            if g_ < 0 or w['anc'][i] - LAG - w['tg'][g_] > MAX_AGE:
                gs = 'none'
            else:
                gs = 'ON' if (w['stv3'][g_] and w['vkl'][g_] > w['P90']) else 'OFF'
            st[f'guard_{gs.lower()}'] += 1
            m1, m5, q = rl['m1'][n], rl['m5'][n], rl['q'][n]
            if not np.isfinite(m1):
                st['nan1'] += 1; continue
            for key in ((grp[i], 'ALL'), (grp[i], gs)):
                e = TH[key]; e['legs'] += 1; e['q'] += q; e['v1'] += q * m1; e['v5'] += q * (m5 if np.isfinite(m5) else 0.0)
            if grp[i] == 'top 5':
                top_at[i] = True; legrows.append((i, q, m1, m5))
        b0, valid, rs, kind = Q.runs(tf, w['P5'], w['opp'], side)
        young = Q2.young_mask(w['tg'], w['jk'], tfl, rs, Q2.Y_MAX)
        bi = Q.fb_births(tf, b0, valid, w['ws'], w['te'], side)
        sch = [(float(tf[i] + Q2.D_F - Q2.L_PLACE), int(b0[i]), float(w['kf'][i])) for i in bi]
        for q_, shf in CFG:
            tt = w['anc'] - shf; key = f'{q_}|{shf:.0f}'
            a = (side, tf, w['opp'], w['P5'], w['S5'], cache, w['tg'], w['vkl'], w['stv3'], w['P90'], w['best'], w['bopp'], tt, w['ptl'], w['shl'], q_, w['te'])
            for arm, sw in OURS.items():
                F, O = Q2.sim4(*a, allow=young if sw.get('y') else None, sched=sch if sw.get('f') else None)
                r = K.vals(w, F, tt); e = OU.setdefault(f'{arm}|{key}', dict(orders=0, ref=0, fills=0, q=0.0, v1=0.0, v5=0.0, sh_q=0.0, sh_v1=0.0, ns_q=0.0, ns_v1=0.0))
                e['orders'] += len(O); e['ref'] += sum(1 for o in O if o['ref']); e['fills'] += len(F); e['q'] += float(r['q'].sum())
                e['v1'] += float(np.nansum(r['q'] * r['m1'])); e['v5'] += float(np.nansum(r['q'] * r['m5']))
                if (q_, shf) == PRIM and F:
                    fl = shared_flags([f[0] for f in F], tko.tolist(), w['ptl'], top_at, [f[1] for f in F])
                    for f, s_, qq, mm in zip(F, fl, r['q'], r['m1']):
                        vv = qq * mm if np.isfinite(mm) else 0.0
                        if s_:
                            e['sh_q'] += qq; e['sh_v1'] += vv
                        else:
                            e['ns_q'] += qq; e['ns_v1'] += vv
                    ours_t = {(tko[f[0]], f[1]) for f in F}
                    for i, qq, m1, m5 in legrows:
                        k2 = ('top 5', f'{"shared" if (tko[i], w["ptl"][i]) in ours_t else "notshared"}|{arm}')
                        e2 = TH[k2]; e2['legs'] += 1; e2['q'] += qq; e2['v1'] += qq * m1
                elif (q_, shf) == PRIM:
                    for i, qq, m1, m5 in legrows:
                        e2 = TH[('top 5', f'notshared|{arm}')]; e2['legs'] += 1; e2['q'] += qq; e2['v1'] += qq * m1
        w.pop('cache', None)
        if st['window_sides'] % 80 == 0:
            r_ = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r_:.0f} MiB', flush=True)
            if r_ > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r_:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    bad = {}
    if st['align_differ']:
        bad['align_differ'] = st['align_differ']
    for k_, e in OU.items():
        ref = q2[k_]
        if ref['fills'] != e['fills'] or abs(ref['q'] - e['q']) > 1e-6 or abs(ref['v1'] - e['v1']) > 1e-6:
            bad[f'qr2_differs {k_}'] = (ref['fills'], e['fills'])
    if bad:
        sys.exit(f'GAP CONTROL FAILED on {day} -> no result: {bad}')
    json.dump(dict(day=day, days=len(D['W0']) / 288.0, stat=st, planted=mut, ranked_makers=len(G),
                   theirs={f'{g_}|{gs}': v_ for (g_, gs), v_ in TH.items()}, ours=OU), open(f'{OUTG}/day_{day}.json', 'w'), default=float)
    print(f'{day}: done in {time.time() - T0:.0f} s; legs at the best {st["at_best"]}, aligned, QR2 reproduced', flush=True)


def report():
    DY = {}
    for day in DAYS9:
        fn = f'{OUTG}/day_{day}.json'
        if not os.path.exists(fn):
            sys.exit(f'REFUSED: {fn} missing')
        DY[day] = json.load(open(fn))
    nd = sum(DY[d]['days'] for d in DAYS9); L = []; P = L.append
    RES = dict(as_of=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), days=DAYS9, day_equiv=nd)
    cps = lambda e: e['v1'] / e['q'] if e['q'] > 0 else NAN
    th = lambda d, k_: DY[d]['theirs'][k_]; ou = lambda d, k_: DY[d]['ours'][k_]

    def pool(fn, k_):
        e = {}
        for d in DAYS9:
            for a, b in fn(d, k_).items():
                e[a] = e.get(a, 0.0) + b
        return e
    st = {k_: sum(DY[d]['stat'].get(k_, 0) for d in DAYS9) for k_ in ('window_sides', 'legs_0_60', 'at_best', 'behind', 'inside', 'no_frame', 'nan1', 'guard_on', 'guard_off', 'guard_none')}
    RES['stat'] = st
    P(f'GAP to the top makers - 9 replay days ({nd:.2f} day-equivalents), first 60 s; as of {RES["as_of"]}')
    P(f'controls passed per day (legs aligned, QR2 reproduced); statuses {st}')
    P(''); P('theirs (real at-best maker legs): legs/day, shares/day, c/sh 1 s, c/sh 5 s, $/day 1 s; per maker for the top 5')
    for g_ in GROUPS:
        e = pool(th, f'{g_}|ALL')
        P(f'  {g_:<6} legs {e["legs"] / nd:>8.0f}  shares {e["q"] / nd:>9.0f}  c/sh {cps(e):+.2f} / {e["v5"] / e["q"]:+.2f}  $/day {e["v1"] / 100 / nd:>8.2f}'
          + (f'  (per maker {e["q"] / nd / 5:.0f} sh, {e["v1"] / 100 / nd / 5:.2f} $/day)' if g_ == 'top 5' else ''))
        RES[f'theirs|{g_}'] = e
    for gs in ('ON', 'OFF', 'none'):
        e = pool(th, f'top 5|{gs}'); RES[f'theirs|top 5|{gs}'] = e
        P(f'  top 5 while v3 guard {gs:<4}: shares/day {e["q"] / nd:>8.0f}, c/sh {cps(e):+.2f}, $/day {e["v1"] / 100 / nd:.2f}')
    P(''); P('ours (simulated 5-sh order per side): c/sh 1 s, $/day 1 s, shares/day')
    for arm in OURS:
        P(f'  {arm:<6} ' + '; '.join(f'{q_}{int(s_)} {cps(pool(ou, f"{arm}|{q_}|{s_:.0f}")):+.2f} c/sh {pool(ou, f"{arm}|{q_}|{s_:.0f}")["v1"] / 100 / nd:.2f} $/day '
                                       f'{pool(ou, f"{arm}|{q_}|{s_:.0f}")["q"] / nd:.0f} sh' for q_, s_ in CFG))
    dtop = lambda d: cps(th(d, 'top 5|ALL'))
    RES['R300'] = {}
    for arm in ('J-v3', 'Y+F30'):
        up = [int(sum(dtop(d) > cps(ou(d, f'{arm}|{k_}')) for d in DAYS9)) for k_ in ('M|0', 'M|50')]
        dn = [int(sum(dtop(d) < cps(ou(d, f'{arm}|{k_}')) for d in DAYS9)) for k_ in ('M|0', 'M|50')]
        lab = 'TOP-HIGHER' if min(up) >= 7 else ('OURS-HIGHER' if min(dn) >= 7 else 'MIXED')
        RES['R300'][arm] = dict(label=lab, days_top_higher=up, days_ours_higher=dn)
        P(f'R300 PER-SHARE (top 5 vs {arm}): {lab}; days top higher at M 0 / M 50 {up}, ours higher {dn}')
    e5 = pool(th, 'top 5|ALL')
    for arm in ('J-v3', 'Y+F30'):
        o = pool(ou, f'{arm}|M|50')
        P(f'R301 SCALE (vs {arm}, M 50): top 5 trade {e5["q"] / max(o["q"], 1e-9):.0f}x our shares and earn {e5["v1"] / max(o["v1"], 1e-9):.1f}x our $ '
          f'(per maker {e5["q"] / 5 / max(o["q"], 1e-9):.0f}x / {e5["v1"] / 5 / max(o["v1"], 1e-9):.1f}x)')
    off = [th(d, 'top 5|OFF')['v1'] / th(d, 'top 5|ALL')['v1'] if th(d, 'top 5|ALL')['v1'] > 0 else NAN for d in DAYS9]
    po = pool(th, 'top 5|OFF')['v1'] / e5['v1'] if e5['v1'] > 0 else NAN
    lab = 'OFF-DOMINANT' if po > 0.5 and sum(x > 0.5 for x in off) >= 7 else ('ON-DOMINANT' if po < 0.5 and sum(x < 0.5 for x in off) >= 7 else 'MIXED')
    RES['R302'] = dict(label=lab, pooled_off_share=po, per_day=off)
    P(f'R302 WHEN: {lab}; share of the top 5\'s at-best $ earned while v3\'s guard is OFF: pooled {po:.1%}, per day {[round(x, 2) for x in off]}')
    shs = [ou(d, 'J-v3|M|50')['sh_q'] / max(ou(d, 'J-v3|M|50')['sh_q'] + ou(d, 'J-v3|M|50')['ns_q'], 1e-9) for d in DAYS9]
    o = pool(ou, 'J-v3|M|50'); ps = o['sh_q'] / max(o['sh_q'] + o['ns_q'], 1e-9)
    lab = 'SAME-FLOW' if ps >= 0.5 and sum(x >= 0.5 for x in shs) >= 7 else ('DIFFERENT-FLOW' if ps < 0.5 and sum(x < 0.5 for x in shs) >= 7 else 'MIXED')
    RES['R303'] = dict(label=lab, pooled_shared=ps, per_day=shs)
    P(f'R303 SAME-FLOW (J-v3, M 50): {lab}; share of our shares on taker orders that also filled a top-5 maker at our price: pooled {ps:.1%}, per day {[round(x, 2) for x in shs]}')
    for arm in OURS:
        o = pool(ou, f'{arm}|M|50'); a_ = pool(th, f'top 5|shared|{arm}'); b_ = pool(th, f'top 5|notshared|{arm}')
        P(f'  {arm}: ours on shared {o["sh_v1"] / max(o["sh_q"], 1e-9):+.2f} c/sh ({o["sh_q"] / nd:.0f} sh/day), not shared {o["ns_v1"] / max(o["ns_q"], 1e-9):+.2f} ({o["ns_q"] / nd:.0f}); '
          f'top 5 on taker orders that also filled us {cps(a_):+.2f} ({a_["q"] / nd:.0f} sh/day), others {cps(b_):+.2f} ({b_["q"] / nd:.0f})')
        RES[f'flow|{arm}'] = dict(ours=o, top_shared=a_, top_notshared=b_)
    P(''); P('per day c/sh 1 s: top 5 ' + ' '.join(f'{dtop(d):+.2f}' for d in DAYS9) + ' | J-v3 M 50 ' + ' '.join(f'{cps(ou(d, "J-v3|M|50")):+.2f}' for d in DAYS9))
    json.dump(RES, open(f'{OUTG}/gap.json', 'w'), indent=1, default=float)
    open(f'{OUTG}/gap.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


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
