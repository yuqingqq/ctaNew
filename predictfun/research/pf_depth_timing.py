"""DEPTH-TIMING (2026-11-13, session ce68ebe0; USER: "If deep ladder doesnot work, why top makers do so? Check if it still the timing issue, I
don't think it is live vs replay issue, top makers still make money in replay and their performance is much better"): why do the top
makers' real fills behind the best pay while our simulated rungs there lose (FRONT-PULL-SIZE: LZ-none-30 / LFPQ-30)?

Same days, same tape, same valuation: 9 consumed replay days, first 60 s, WS-timed trades whose taker order is exact and complete (ORDERS'
map, 9f8ec338). For every fill, real (a maker leg) or ours (S4's rest lane, rungs d = 1..5 ticks behind the best at 30 sh, never pulled =
LZ-none-30's rungs, and pulled at the best by FPQ = LFPQ-30's rungs; fill rule M 50):
  depth     ticks behind the side's best in the frame strictly before the trade (GP.depth_of; 0 = at the best)
  type      STOP: the trade's taker order stopped at this price (the level held); THROUGH: it filled a worse price too (the level was taken).
            Our fills inherit the type of the real taker order that filled them (at our price, or through it)
  value     sg (mid(t + 1 s) - price) (and 5 s), c/sh; t = the trade's WS anchor
  P         FRONT-PULL's per-day model (fitted without the day) at the last 50 ms step <= t - 24 ms
  before    sg (mid(t) - mid(t - 1 s)): how far the price had already moved toward the order before the fill
Groups: top 5 / field (ranks 6+) by daily volume (B12), real legs; ours = the simulated rungs (all rungs pooled; by depth printed).
Readings DECLARED BEFORE THE FIRST RUN (evaluated in code; depth >= 1 unless stated; c/sh at 1 s):
 R420 GAP: top 5 minus ours, decomposed over type: MIX = sum_type (w5 - wo) x v_ours, WITHIN = sum_type w5 x (v5 - v_ours). MIX >= 70% of the
      gap -> WHICH-TRADES (the deep orders are reached by different trades: queue place / when they posted); WITHIN >= 70% -> WITHIN-TYPE
      (the same kind of trade pays them more: when they are there); else BOTH. The gap's sign on each day printed (day rule >= 7 of 9).
 R421 STOP-SHARE: the share of deep-fill shares from STOP trades, top 5 against ours: >= +20 points -> MORE-STOPS; else NOT-SHOWN.
 R422 TIMING: the share of deep-fill shares at P in its bottom tercile (terciles over all real deep fills): top 5 against ours, >= +10 points
      -> PRESENT-WHEN-SAFE; <= -10 -> NOT-WHEN-SAFE; else NOT-SHOWN. Value by P tercile and type printed.
 Printed: the same tables at depth 0; by depth 1 / 2 / 3+; the field beside the top 5; 'before' by group and type.
Controls (any failure -> no output): ORDERS' map coverage as built; a synthetic case for the type of our fills (a fill at our price in a
taker order that went deeper is THROUGH; one at the stop is STOP; a through-fill is THROUGH); C1 for the rungs (never-pull callback =
plain lane at offset d); P aligned with FRONT-PULL's fit rows; value function identical for both (one function).
Limits: replay rule M for our fills (B1a); real legs include orders that queued anywhere at their level; consumed days, design after
FRONT-PULL-SIZE (exploratory).
Usage (run_light, PF_SLOT=claude): pf_depth_timing.py day <day> | report; driver run_depth_timing.sh
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
OUTD = f'{K.IO}/depth_timing_20261113'
ORD = f'{K.IO}/orders_20261113'
RUNG_D = range(1, 6)
OLDKEY = ['mid', 'exec_s', 'first_recv_ns', 'taker_index', 'taker_quote', 'price_executed']
MEM_CAP_MIB = 2048.0


def trade_type(side, price, prices_of_order):
    """STOP if the taker order's worst price is this one, THROUGH if it filled a worse price (bids: lower; asks: higher)."""
    worst = min(prices_of_order) if side == 0 else max(prices_of_order)
    return 'STOP' if worst == price else 'THROUGH'


def selftest():
    bad = []
    if trade_type(0, 50, [51, 50, 49]) != 'THROUGH' or trade_type(0, 49, [51, 50, 49]) != 'STOP' or trade_type(1, 52, [51, 52]) != 'STOP':
        bad.append('trade type')
    if trade_type(1, 51, [51, 52]) != 'THROUGH':
        bad.append('ask through')
    return bad


def day_run(day):
    import lightgbm as lgb
    import pf_gap_top_makers as GP
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
    ok = F.comp.values & np.isfinite(F.tws.values)
    T = F[ok]                                             # legs of exact, complete, WS-timed taker orders
    TP = T.groupby('tko').ptick.agg(list).to_dict()       # each taker order's prices
    leg2tko = {}                                          # (mid, side, trade time, price) -> taker orders, for our fills (they ride the real legs)
    LT = D['LT']
    tko_of_row = F.tko.reindex(LT.index).values; comp_of_row = (F.comp.reindex(LT.index).values.astype(bool) & np.isfinite(F.tws.reindex(LT.index).values))
    mdl = lgb.Booster(model_file=f'{FP.OUTF}/models/av_{day}.txt')
    fit = json.load(open(f'{FP.OUTF}/fit.json'))['days'][day]['lines']; l50, l90 = fit['0.50'], fit['0.90']
    rows = []; st.update(window_sides=0, real=0, ours=0, ours_unknown_type=0, c1_differ=0)
    LI = LT.groupby(['mid', 'side'], sort=False).indices; nb = len(K.G32); q, shf = 'M', 50.0
    for w in K.window_sides(D):
        st['window_sides'] += 1; m, side = w['m'], w['side']; sg = w['sg']
        tf, mff = w['tf_mff']; tfl = tf.tolist(); P5, opp = w['P5'], w['opp']
        mid_at = lambda t: mff[max(int(np.searchsorted(tf, t, side='right')) - 1, 0)] if np.searchsorted(tf, t, side='right') > 0 else np.nan
        n = w['n']
        Xw = np.column_stack([w['X'], np.full(n, float(side))]).astype(np.float32)
        P = mdl.predict(Xw) if n else np.zeros(0); vk = w['X'][:, nb] if n else np.zeros(0)
        P = np.where(~np.isfinite(vk) | (vk == -1.0) | (np.asarray(w['best']) < 0) | (w['jk'] < 0), np.nan, P)
        Pat = lambda t: (lambda k: float(P[k]) if k >= 0 else np.nan)(int(np.searchsorted(w['tg'], t - 24.0, side='right')) - 1)

        def value(t, price):
            a, b_, c = mid_at(t + 1000.0), mid_at(t + 5000.0), mid_at(t)
            z = mid_at(t - 1000.0)
            return sg * (a - price), sg * (b_ - price), sg * (c - z)
        if (m, side) not in LI:
            continue
        x = LT.iloc[LI[(m, side)]]
        # real legs (first minute, exact complete WS-timed taker orders)
        sel = comp_of_row[LI[(m, side)]] & (x.anc.values >= w['ws']) & (x.anc.values < w['ws'] + 60000.0)
        xr = x[sel]
        for i_, s_, p_, q_ in zip(xr.index.values, xr.signer.values if 'signer' in xr else F.signer.reindex(xr.index).values, xr.ptick.values, xr.shares.values):
            tko = F.tko.at[i_]; t = float(F.tws.at[i_])
            d_ = GP.depth_of(tfl, P5, opp, side, t, int(p_))
            if d_ is None or d_ < 0:
                continue
            v1, v5, bf = value(t, float(p_))
            rows.append((int(day), 'real', G.get(int(F.signer.at[i_]), 'rest'), d_, trade_type(side, int(p_), TP[tko]), float(q_), v1, v5, bf, Pat(t), -1))
            st['real'] += 1
        # our rungs (S4 rest lane, 30 sh), never pulled and FPQ-at-the-best
        tt = w['anc'] - shf; vv = np.asarray(w['vkl'], float); vg = (vv <= K.GUARD_PULL).tolist()
        rix = LI[(m, side)]                               # w['anc'] is window_sides' leg order: rebuild the leg -> row link
        l = D['LT'].iloc[D['LI'][(m, side)]]
        o_ = np.lexsort(((-l.ptick.values if side == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
        for d in RUNG_D:
            for arm, kind in (('LZ-none', 'none'), ('LFPQ', 'fpq')):
                Fo, Oo, _, _ = FS.lane(w, q, shf, kind, FS.LAD_SIZE, d, P, vv, vg, l50, l90)
                if kind == 'none':
                    Fk, Ok, _ = K.sim_arm(w, q, shf, 'rest', decide=K.decide_record, offset=d, size=FS.LAD_SIZE)
                    st['c1_differ'] += int(Fk != Fo or Ok != Oo)
                for f in Fo:
                    i, pt, qq, thr = f[0], f[1], f[2], f[3]
                    t = float(w['anc'][i]); row = int(lrow[i])
                    if not (w['ws'] <= t < w['ws'] + 60000.0):
                        continue
                    tko = F.tko.at[row] if row in F.index else None
                    if not (row in F.index and bool(F.comp.at[row]) and np.isfinite(F.tws.at[row]) and tko in TP):
                        st['ours_unknown_type'] += 1; continue
                    typ = 'THROUGH' if thr else trade_type(side, int(pt), TP[tko])
                    d_ = GP.depth_of(tfl, P5, opp, side, float(F.tws.at[row]), int(pt))
                    if d_ is None or d_ < 0:
                        continue
                    v1, v5, bf = value(float(F.tws.at[row]), float(pt))
                    rows.append((int(day), arm, 'ours', d_, typ, float(qq), v1, v5, bf, Pat(float(F.tws.at[row])), d))
                    st['ours'] += 1
        if st['window_sides'] % 80 == 0:
            r_ = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r_:.0f} MiB', flush=True)
            if r_ > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r_:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    if st['c1_differ']:
        sys.exit(f'DEPTH-TIMING CONTROL FAILED: C1 {st["c1_differ"]}')
    os.makedirs(OUTD, exist_ok=True)
    pd.DataFrame(rows, columns=['day', 'src', 'grp', 'depth', 'type', 'q', 'v1', 'v5', 'before', 'P', 'rung']).to_parquet(f'{OUTD}/fills_{day}.parquet', index=False)
    json.dump(st, open(f'{OUTD}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: {st} in {time.time() - T0:.0f} s', flush=True)


def report():
    A = pd.concat([pd.read_parquet(f'{OUTD}/fills_{d}.parquet') for d in DAYS9], ignore_index=True)
    ST = {d: json.load(open(f'{OUTD}/day_{d}.json')) for d in DAYS9}
    A = A[np.isfinite(A.v1)]
    A['who'] = np.where(A.src == 'real', np.where(A.grp == 'top 5', 'top 5', 'field'), 'ours ' + A.src)
    wm = lambda x, c: float((x.q * x[c]).sum() / x.q.sum()) if x.q.sum() else float('nan')
    deep = A[A.depth >= 1]; real_deep = deep[deep.src == 'real']
    terc = np.nanquantile(real_deep.P, [1 / 3, 2 / 3]) if real_deep.P.notna().any() else [np.nan, np.nan]
    R = {}

    def decomp(top, ours):
        tot5 = top.q.sum(); toto = ours.q.sum(); mix = 0.0; within = 0.0; parts = {}
        for ty in ('STOP', 'THROUGH'):
            a = top[top.type == ty]; b = ours[ours.type == ty]
            w5 = a.q.sum() / tot5 if tot5 else 0.0; wo = b.q.sum() / toto if toto else 0.0
            v5, vo = wm(a, 'v1'), wm(b, 'v1')
            mix += (w5 - wo) * (vo if vo == vo else 0.0); within += w5 * ((v5 - vo) if (v5 == v5 and vo == vo) else 0.0)
            parts[ty] = dict(w_top=w5, w_ours=wo, v_top=v5, v_ours=vo)
        return mix, within, parts
    top = deep[deep.who == 'top 5']; ours = deep[deep.who == 'ours LZ-none']
    gap = wm(top, 'v1') - wm(ours, 'v1'); mix, within, parts = decomp(top, ours)
    lab = 'WHICH-TRADES' if gap and mix / gap >= 0.7 else ('WITHIN-TYPE' if gap and within / gap >= 0.7 else 'BOTH')
    perd = []
    for d in DAYS9:
        t_ = top[top.day == int(d)]; o_ = ours[ours.day == int(d)]
        perd.append(wm(t_, 'v1') - wm(o_, 'v1'))
    R['R420'] = lab; R['R420_detail'] = dict(gap=gap, mix=mix, within=within, mix_share=mix / gap if gap else float('nan'), parts=parts,
                                             days_positive=int(sum(x > 0 for x in perd)), per_day=perd)
    ss = lambda x: float(x.q[x.type == 'STOP'].sum() / x.q.sum()) if x.q.sum() else float('nan')
    R['R421'] = 'MORE-STOPS' if ss(top) - ss(ours) >= 0.20 else 'NOT-SHOWN'; R['R421_detail'] = dict(top5=ss(top), ours=ss(ours), field=ss(deep[deep.who == 'field']))
    low = lambda x: float(x.q[x.P <= terc[0]].sum() / x.q[x.P.notna()].sum()) if x.q[x.P.notna()].sum() else float('nan')
    dl = low(top) - low(ours)
    R['R422'] = 'PRESENT-WHEN-SAFE' if dl >= 0.10 else ('NOT-WHEN-SAFE' if dl <= -0.10 else 'NOT-SHOWN')
    R['R422_detail'] = dict(top5=low(top), ours=low(ours), field=low(deep[deep.who == 'field']), terciles=list(map(float, terc)))
    whos = ['top 5', 'field', 'ours LZ-none', 'ours LFPQ']
    L = [f'DEPTH-TIMING, 9 consumed replay days, first 60 s, WS-timed exact complete taker orders; as of {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; '
         'controls passed', 'statuses ' + json.dumps({k: sum(ST[d].get(k, 0) for d in DAYS9) for k in ('real', 'ours', 'ours_unknown_type', 'window_sides')}), '',
         'Readings: ' + json.dumps({k: v for k, v in R.items() if not k.endswith('_detail')}), '  ' + json.dumps({k: v for k, v in R.items() if k.endswith('_detail')}, default=float), '',
         'c/sh at 1 s (5 s) | share of shares | before (c) - by who x depth x type:']
    for dep, msk in (('depth 0', A.depth == 0), ('depth 1', A.depth == 1), ('depth 2', A.depth == 2), ('depth 3+', A.depth >= 3), ('depth >= 1', A.depth >= 1)):
        L.append(f'  {dep}')
        for who in whos:
            x = A[msk & (A.who == who)]
            if not len(x):
                continue
            seg = []
            for ty in ('STOP', 'THROUGH'):
                y = x[x.type == ty]
                seg.append(f'{ty} {wm(y, "v1"):+.2f} ({wm(y, "v5"):+.2f}) {y.q.sum() / x.q.sum():.0%} bf {wm(y, "before"):+.2f}')
            L.append(f'    {who:<13} all {wm(x, "v1"):+.2f} ({wm(x, "v5"):+.2f}) sh {x.q.sum():>10.0f} | ' + ' | '.join(seg))
    L += ['', f'depth >= 1, by P tercile (P <= {terc[0]:.3f} / <= {terc[1]:.3f} / above): share of shares and c/sh at 1 s, STOP / THROUGH']
    for who in whos:
        x = deep[(deep.who == who) & deep.P.notna()]
        segs = []
        for nm, lo, hi in (('low', -1.0, terc[0]), ('mid', terc[0], terc[1]), ('high', terc[1], 2.0)):
            y = x[(x.P > lo) & (x.P <= hi)] if nm != 'low' else x[x.P <= hi]
            segs.append(f'{nm} {y.q.sum() / x.q.sum() if x.q.sum() else float("nan"):.0%}: {wm(y[y.type == "STOP"], "v1"):+.2f} / {wm(y[y.type == "THROUGH"], "v1"):+.2f}')
        L.append(f'  {who:<13} ' + ' | '.join(segs))
    L += ['', 'R420 per day (top 5 minus ours LZ-none, depth >= 1, c/sh 1 s): ' + ' '.join(f'{x:+.2f}' for x in perd)]
    json.dump(dict(readings=R), open(f'{OUTD}/depth_timing.json', 'w'), indent=1, default=float)
    open(f'{OUTD}/depth_timing.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'day' and len(sys.argv) > 2:
        day_run(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit(__doc__)
