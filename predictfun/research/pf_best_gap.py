"""BEST-GAP (2026-11-13, session ce68ebe0; USER: "The gap at the best price, is it mostly the timing issue, or latency, or size? We have been
working on this, still not sure about the cause"): on exactly-timed trades at the best, our guard ladder's rung-0 fills earn about +0.05 c/sh
and the top makers' real fills +0.38 (TIMING-SPLIT, ad088c80). How much of that gap does each lever close?

Population: 9 consumed replay days, first 60 s, fills AT THE BEST (depth 0 before the trade) on trades whose taker order is exact, complete
and WS-timed (ORDERS' map; as DEPTH-TIMING); value sg (mid(t + 1 s) - price), the same function for real and ours; 'aged' = our order >= 100
ms old at the fill (real legs have no age: all of theirs).
Ours: S4's rest lane, rung 0 (one order per side at the best), kept through fills, the guard trigger v <= -0.10 (LDG's rung 0). Arms:
  BASE      30 sh, today's clocks (feed >= 13.2 ms after the venue stamp via the receipt clock, decision 2 ms, place 40 ms, cancel 22 ms)
  S5, S100  size 5 / 100 sh                                                                   -> SIZE lever
  FAST      30 sh, order path 5 ms (place, cancel, pending cancel) and the book known at stamp + 13.2 + 2 ms (fastest feed)
  ZERO      30 sh, order path 0 ms and the book known at its stamp (bound; the guard value keeps its live clock)  -> LATENCY lever
  BASE at fill rule U (nothing ahead of us)                                                    -> QUEUE lever (bound)
  TIMING lever: BASE's c/sh within each P tercile (DEPTH-TIMING's lines) re-weighted to the top 5's tercile mix
Readings DECLARED BEFORE THE FIRST RUN (fill clock M 50, aged fills; gap G = top 5 c/sh - BASE c/sh; each lever's closure = its change in
BASE's c/sh as a share of G):
 R460 SIZE: max(S100, S5) - BASE.   R461 LATENCY: FAST - BASE (ZERO - BASE printed).   R462 QUEUE: BASE at U - BASE at M.
 R463 TIMING: re-weighted - BASE.
 R464 MAIN CAUSE: the lever with the largest closure is MAIN if it closes >= 50% of G; if two or more close >= 30% each -> MIXED (named);
      if none closes >= 30% -> NONE-OF-THESE. Day consistency printed (each lever's sign by day). Also M 0 printed.
 Descriptive: STOP / THROUGH share and c/sh by type (ours and the top 5's), pending share, all-age values.
Controls (any failure -> no output): FRONT-PULL-SIZE's self-tests; the clock override restores the constants and the window context (checked
after every arm); the BASE arm equals DEEP-PULL's LDG rung 0 lane fill for fill; C1 at rung 0; memory self-guard.
Limits: replay rule M (B1a) for M; U is a bound; FAST / ZERO change our order path and book view but not the guard value's clock (its
inputs keep their live delay) nor the 50 ms decision grid; consumed days; the levers are not additive.
Usage (run_light, PF_SLOT=claude): pf_best_gap.py day <day> | report; driver run_best_gap.sh
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402
import pf_front_pull as FP                                # noqa: E402
import pf_front_pull_size as FS                           # noqa: E402

DAYS9 = K.DAYS9
OUTB = f'{K.IO}/best_gap_20261113'
ORD = f'{K.IO}/orders_20261113'
OLDKEY = ['mid', 'exec_s', 'first_recv_ns', 'taker_index', 'taker_quote', 'price_executed']
TERC = (0.09040728596733005, 0.4236075378593026)
CFGS = [('M', 50.0), ('M', 0.0), ('U', 50.0)]
CLK0 = dict(L_PLACE=K.L_PLACE, L_CANCEL=K.L_CANCEL, L_PENDCANCEL=K.L_PENDCANCEL)
MEM_CAP_MIB = 2048.0


def fast_context(w, mode):
    """A copy of the window context with the book known earlier: FAST = stamp + PF_MIN + DD, ZERO = stamp."""
    tf = w['tf']; P5 = w['P5']; opp = w['opp']; side = w['side']
    kf = np.maximum.accumulate(tf + (K.PF_MIN + 2.0 if mode == 'FAST' else 0.0))
    ki = np.searchsorted(kf, w['tg'], side='right') - 1
    own = P5[:, 0]
    bb, ba = (own, opp) if side == 0 else (opp, own)
    ok = (ki >= 0) & (bb[np.clip(ki, 0, None)] >= 0) & (ba[np.clip(ki, 0, None)] > bb[np.clip(ki, 0, None)])
    best = np.where(ok, own[np.clip(ki, 0, None)], -1); bopp = np.where(ok, opp[np.clip(ki, 0, None)], -1)
    w2 = dict(w); w2.update(kf=kf, jk=ki, best=best.tolist(), bopp=bopp.tolist(), cache={})
    return w2


def run_lane(w, q, shf, vg, size, mode):
    """Rung 0, guard trigger; mode BASE / FAST / ZERO (order path 40-22-95 / 5-5-5 / 0-0-0 ms)."""
    if mode == 'BASE':
        return K.sim_arm(w, q, shf, 'rest', aux=vg, size=size)
    path = 5.0 if mode == 'FAST' else 0.0
    try:
        K.L_PLACE = path; K.L_CANCEL = path; K.L_PENDCANCEL = path
        return K.sim_arm(fast_context(w, mode), q, shf, 'rest', aux=vg, size=size)
    finally:
        K.L_PLACE, K.L_CANCEL, K.L_PENDCANCEL = CLK0['L_PLACE'], CLK0['L_CANCEL'], CLK0['L_PENDCANCEL']


def selftest():
    bad = []
    w = dict(tf=np.array([0.0, 50.0, 100.0]), P5=np.array([[50, -1, -1, -1, -1], [51, -1, -1, -1, -1], [51, -1, -1, -1, -1]]), opp=np.array([53, 53, 53]),
             side=0, tg=np.array([20.0, 60.0, 120.0]), kf=np.array([60.0, 110.0, 160.0]), jk=np.array([-1, 0, 1]), best=[-1, 50, 51], bopp=[-1, 53, 53])
    z = fast_context(w, 'ZERO')
    if z['best'] != [50, 51, 51] or w['best'] != [-1, 50, 51]:
        bad.append(f'ZERO context {z["best"]} / original {w["best"]}')
    f = fast_context(w, 'FAST')
    if f['best'] != [50, 50, 51]:                       # known at stamp + 15.2 ms: the 50 ms frame is not yet known at 60 ms
        bad.append(f'FAST context {f["best"]}')
    return bad


def day_run(day):
    import lightgbm as lgb
    import pf_gap_top_makers as GP
    import pf_depth_timing as DT
    FS.all_selftests(); b = selftest()
    if b:
        sys.exit(f'SELFTEST FAILED: {b}')
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
    mdl = lgb.Booster(model_file=f'{FP.OUTF}/models/av_{day}.txt')
    rows = []; st.update(window_sides=0, c1_differ=0, base_differ=0, clock_restored=1, real=0, ours=0)
    nb = len(K.G32)
    for w in K.window_sides(D):
        st['window_sides'] += 1; m, side, sg, n = w['m'], w['side'], w['sg'], w['n']
        tf, mff = w['tf_mff']; tfl = tf.tolist(); P5, opp = w['P5'], w['opp']

        def mid_at(t):
            j = int(np.searchsorted(tf, t, side='right')) - 1
            return mff[j] if j >= 0 else np.nan
        Xw = np.column_stack([w['X'], np.full(n, float(side))]).astype(np.float32)
        P = mdl.predict(Xw) if n else np.zeros(0); vk = w['X'][:, nb] if n else np.zeros(0)
        P = np.where(~np.isfinite(vk) | (vk == -1.0) | (np.asarray(w['best']) < 0) | (w['jk'] < 0), np.nan, P)
        terc = lambda t: (lambda k: 'n/a' if k < 0 or not (P[k] == P[k]) else ('low' if P[k] <= TERC[0] else ('mid' if P[k] <= TERC[1] else 'high')))(
            int(np.searchsorted(w['tg'], t - 24.0, side='right')) - 1)
        if (m, side) not in D['LI']:
            continue
        l = D['LT'].iloc[D['LI'][(m, side)]]
        o_ = np.lexsort(((-l.ptick.values if side == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
        for i, r_ in enumerate(lrow):                    # real legs at the best, exact complete WS-timed taker orders, first minute
            if r_ not in F.index or not (bool(F.comp.at[r_]) and np.isfinite(F.tws.at[r_]) and F.tko.at[r_] in TP):
                continue
            t = float(F.tws.at[r_]); p_ = int(F.ptick.at[r_])
            if not (w['ws'] <= t < w['ws'] + 60000.0) or GP.depth_of(tfl, P5, opp, side, t, p_) != 0:
                continue
            g_ = G.get(int(F.signer.at[r_]), 'rest')
            rows.append((int(day), 'real', 'n/a', 'top 5' if g_ == 'top 5' else 'field', DT.trade_type(side, p_, TP[F.tko.at[r_]]), float(F.shares.at[r_]),
                         sg * (mid_at(t + 1000.0) - p_), terc(t), True, False))
            st['real'] += 1
        vg = (np.asarray(w['vkl'], float) <= K.GUARD_PULL).tolist()
        for q, shf in CFGS:
            arms = [('BASE', 30.0, 'BASE'), ('S5', 5.0, 'BASE'), ('S100', 100.0, 'BASE')]
            if q == 'M':
                arms += [('FAST', 30.0, 'FAST'), ('ZERO', 30.0, 'ZERO')]
            for arm, size, mode in arms:
                Fo, Oo, _ = run_lane(w, q, shf, vg, size, mode)
                if (K.L_PLACE, K.L_CANCEL, K.L_PENDCANCEL) != (CLK0['L_PLACE'], CLK0['L_CANCEL'], CLK0['L_PENDCANCEL']):
                    st['clock_restored'] = 0
                if arm == 'BASE':
                    Fd, Od, _, _ = FS.lane(w, q, shf, 'guard', 30.0, 0, P, np.asarray(w['vkl'], float), vg, 0.0, 0.0)
                    st['base_differ'] += int(Fd != Fo or Od != Oo)
                    if q == 'M' and shf == 50.0:
                        F0, O0, _ = K.sim_arm(w, q, shf, 'rest', size=30.0)
                        Fk, Ok, _ = K.sim_arm(w, q, shf, 'rest', decide=K.decide_record, size=30.0)
                        st['c1_differ'] += int(Fk != F0 or Ok != O0)
                for f in Fo:
                    i, pt, qq, thr, age, ph = f[0], f[1], f[2], f[3], f[4], f[5]
                    r_ = int(lrow[i])
                    if r_ not in F.index or not (bool(F.comp.at[r_]) and np.isfinite(F.tws.at[r_]) and F.tko.at[r_] in TP):
                        continue
                    t = float(F.tws.at[r_])
                    if not (w['ws'] <= t < w['ws'] + 60000.0) or GP.depth_of(tfl, P5, opp, side, t, int(pt)) != 0:
                        continue
                    typ = 'THROUGH' if thr else DT.trade_type(side, int(pt), TP[F.tko.at[r_]])
                    rows.append((int(day), arm, f'{q}{int(shf)}', 'ours', typ, float(qq), sg * (mid_at(t + 1000.0) - pt), terc(t), age >= 100.0, ph == 'pend'))
                    st['ours'] += 1
        if st['window_sides'] % 80 == 0:
            r2 = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r2:.0f} MiB', flush=True)
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    if st['c1_differ'] or st['base_differ'] or not st['clock_restored']:
        sys.exit(f'BEST-GAP CONTROL FAILED: {st}')
    os.makedirs(OUTB, exist_ok=True)
    pd.DataFrame(rows, columns=['day', 'arm', 'cfg', 'who', 'type', 'q', 'v1', 'pt', 'aged', 'pend']).to_parquet(f'{OUTB}/fills_{day}.parquet', index=False)
    json.dump(dict(stat=st), open(f'{OUTB}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: {st} in {time.time() - T0:.0f} s', flush=True)


def report():
    A = pd.concat([pd.read_parquet(f'{OUTB}/fills_{d}.parquet') for d in DAYS9], ignore_index=True)
    A = A[np.isfinite(A.v1)]
    cps = lambda x: float((x.q * x.v1).sum() / x.q.sum()) if x.q.sum() else float('nan')
    top = A[A.who == 'top 5']; fld = A[A.who == 'field']
    sel = lambda arm, cfg, aged=True: A[(A.arm == arm) & (A.cfg == cfg) & (A.aged if aged else True)]
    tmix = {tp: float(top.q[top.pt == tp].sum() / top.q[top.pt != 'n/a'].sum()) for tp in ('low', 'mid', 'high')}

    def reweight(x):
        return sum(tmix[tp] * cps(x[x.pt == tp]) for tp in ('low', 'mid', 'high') if x[x.pt == tp].q.sum() > 0)

    def levers(cfg, day=None):
        f = (lambda x: x[x.day == day]) if day else (lambda x: x)
        b = cps(f(sel('BASE', cfg))); g = cps(f(top)) - b
        lv = dict(SIZE=max(cps(f(sel('S100', cfg))), cps(f(sel('S5', cfg)))) - b, QUEUE=cps(f(sel('BASE', 'U50'))) - cps(f(sel('BASE', 'M50'))),
                  TIMING=reweight(f(sel('BASE', cfg))) - b)
        if cfg == 'M50':
            lv['LATENCY'] = cps(f(sel('FAST', cfg))) - b; lv['LATENCY_ZERO'] = cps(f(sel('ZERO', cfg))) - b
        return b, g, lv
    b, g, lv = levers('M50')
    share = {k: v / g if g else float('nan') for k, v in lv.items()}
    main = [k for k in ('SIZE', 'LATENCY', 'QUEUE', 'TIMING') if share[k] >= 0.5]
    big = [k for k in ('SIZE', 'LATENCY', 'QUEUE', 'TIMING') if share[k] >= 0.3]
    R = dict(gap=g, base=b, top5=cps(top), field=cps(fld), levers=lv, closure_share=share, top5_tercile_mix=tmix)
    R['R464'] = (f'MAIN: {max(main, key=lambda k: share[k])}' if main else (f'MIXED: {", ".join(big)}' if len(big) >= 2 else
                 (f'PARTLY: {big[0]}' if big else 'NONE-OF-THESE')))
    per = {}
    for d in DAYS9:
        _, gd, ld = levers('M50', int(d)); per[d] = dict(gap=gd, **ld)
    R['per_day'] = per
    b0, g0, lv0 = levers('M0'); R['M0'] = dict(base=b0, gap=g0, levers=lv0)
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    L = [f'BEST-GAP, 9 consumed replay days, first 60 s, fills at the best on exact complete WS-timed trades; as of {asof}; controls passed',
         f'top 5 {cps(top):+.3f} c/sh, field {cps(fld):+.3f}, ours BASE (aged, M 50) {b:+.3f}; gap G {g:+.3f}',
         'levers (change in BASE c/sh, share of G): ' + ', '.join(f'{k} {v:+.3f} ({share[k]:.0%})' for k, v in lv.items()),
         f'R464: {R["R464"]}', f'M 0: base {b0:+.3f}, gap {g0:+.3f}, levers ' + ', '.join(f'{k} {v:+.3f}' for k, v in lv0.items()), '',
         'c/sh and share of shares by type (STOP = the level held, THROUGH = it was taken); aged fills unless "all"; pending share']
    for nm, x in (('top 5', top), ('field', fld)) + tuple((f'{a} {c}', sel(a, c)) for a in ('BASE', 'S5', 'S100', 'FAST', 'ZERO') for c in ('M50', 'M0', 'U50')):
        if not x.q.sum():
            continue
        segs = [f'{ty} {cps(x[x.type == ty]):+.2f} ({x.q[x.type == ty].sum() / x.q.sum():.0%})' for ty in ('STOP', 'THROUGH')]
        segs.append('P low/mid/high ' + ' / '.join(f'{x.q[x.pt == tp].sum() / max(x.q[x.pt != "n/a"].sum(), 1e-9):.0%}' for tp in ('low', 'mid', 'high')))
        segs.append(f'pending {x.q[x.pend].sum() / x.q.sum():.0%}')
        L.append(f'  {nm:<10} all {cps(x):+.3f} sh {x.q.sum():>9.0f} | ' + ' | '.join(segs))
    for a in ('BASE', 'FAST', 'ZERO'):
        x = sel(a, 'M50', aged=False)
        L.append(f'  {a} M50 incl. young: {cps(x):+.3f} (young share {x.q[~x.aged].sum() / max(x.q.sum(), 1e-9):.0%})')
    L += ['', 'per day (M 50): gap | SIZE | LATENCY | QUEUE | TIMING']
    for d in DAYS9:
        p_ = per[d]; L.append(f"  {d} {p_['gap']:+.2f} | {p_['SIZE']:+.2f} | {p_['LATENCY']:+.2f} | {p_['QUEUE']:+.2f} | {p_['TIMING']:+.2f}")
    os.makedirs(OUTB, exist_ok=True)
    json.dump(R, open(f'{OUTB}/best_gap.json', 'w'), indent=1, default=float)
    open(f'{OUTB}/best_gap.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'day' and len(sys.argv) > 2:
        day_run(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit(__doc__)
