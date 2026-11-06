"""What warns of a bad sweep in time for us? (2026-11-03; follows pf_speed_race: most of our swept fills have no Binance move
of 0.5 bp before them, and the other makers' swept fills are far milder than ours.) Descriptive, 8 read days, live 0-240 s.

Sweeps = transactions (key as pf_fill_mix) with legs at two or more prices on one side; informed = s (m1 - m0) for the
makers, c/sh (venue mids strictly before and 1 s after); classes bad <= -2, mid -2..0, good >= 0. At-risk size = shown size
at the prices the sweep crossed, from its first to its deepest price. In time for us = visible in a frame at or before
t - 75 ms (Predict frames reach us a median 53 ms late, a cancel lands 22 ms after sending).
  pulled (lower bound) = max(0, Q(t-500) - Q(t-75) - traded there in between) / Q(t-500)
  thinned              = 1 - Q(t-75) / Q(t-500)      (pulls minus re-posts, both visible)
  Binance              = an adverse move of 0.2 / 0.5 bp that crossed at or before t - 25.5 ms within 1 s (our cancel lands in
                         time) vs only later
Our replay swept fills get the same measures at their own price (single level).
Statuses: sweeps with no frame 500 ms before, legs with a 1 s stamp only.
Usage: python3 -m predictfun.research.pf_sweep_precursor
"""
import os
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = os.environ.get('PF_SP_DAYS', '20261023,20261024,20261025,20261026,20261027,20261031,20261101,20261102').split(','); FRESH = ('20261101', '20261102')
S4 = f'{io.OUT}/step4'; OUTD = f"{S4}/sweep_precursor{os.environ.get('PF_SP_SFX', '')}"; os.makedirs(OUTD, exist_ok=True)
COLS = ['mid', 't_ms'] + [f'{s}{i}_{k}' for s in 'ba' for i in range(1, 6) for k in 'ps']
TK = ['mid', 'exec_s', 'first_recv_ns', 'taker_index', 'taker_quote', 'price_executed']
VIS, CANCEL_AT, PRE = 75.0, 25.5, 500.0
OURS = [('ours: ladder', lambda d: f"{S4}/ladder_{'preds_c3' if d in FRESH else 'preds_o3'}_lad1q/{d}/fills/K3_z5_S60_TH0.5_CAP50_BUD75_RF1000_THIN30_CD500.parquet"),
        ('ours: single quote', lambda d: f"{S4}/live_tokyoreal_custom_e2_{'preds_c3' if d in FRESH else 'preds_o3'}_lc/{d}/fills/"
                                         "MICRO_G4_G2_G6_G5_G3_G1_M025_XNA_XMIN1_XS0_XV_S60_CAP10__mid_szA1-5-25-0.parquet")]


def since_cross(K, LM, t, d, X):
    """As pf_speed_race.since_cross: ms since the Binance log-mid crossed X bp toward its value at t in direction d, 1 s back."""
    out = np.full(len(t), np.nan)
    i1 = np.searchsorted(K, t, side='right') - 1; i0 = np.maximum(np.searchsorted(K, t - 1000.0, side='right') - 1, 0)
    for n in range(len(t)):
        b = i1[n]
        if b < 0:
            continue
        a = i0[n]; seg = LM[a:b + 1]; idx = np.flatnonzero(d[n] * (seg[-1] - seg) >= X * 1e-4 - 1e-12)
        if len(idx):
            out[n] = t[n] - K[a + idx[-1] + 1]
    return out


def load_bn(day):
    ks, ms = [], []
    for h in io.hours_of(day, extra_after=1, extra_before=1):
        p_ = f'{io.BN_DIR}/{h}.csv.gz'
        if not os.path.exists(p_):
            continue
        df = pd.read_csv(p_, header=None, names=['recv', 'E', 'T', 'u', 'bid', 'bq', 'ask', 'aq'], usecols=['E', 'bid', 'ask'])
        df = df.apply(pd.to_numeric, errors='coerce').dropna(); df = df[(df.bid > 0) & (df.ask > df.bid)]
        k = df.E.values.astype(np.float64); m = (df.bid.values + df.ask.values) / 2.0; del df
        o = np.argsort(k, kind='stable'); k, m = k[o], m[o]; keep = np.r_[True, m[1:] != m[:-1]]
        ks.append(k[keep]); ms.append(m[keep])
    K = np.concatenate(ks); M = np.concatenate(ms); o = np.argsort(K, kind='stable')
    return K[o], np.log(M[o])


def at_risk(P, S, i, lo, hi):
    """Shown size at prices lo..hi in frame i (NaN if no frame)."""
    if i < 0:
        return np.nan
    return float(np.where((P[i] >= lo) & (P[i] <= hi), S[i], 0.0).sum())


SW, OF, st = [], [], dict(legs_1s_only=0, sweeps_no_pre_frame=0, ours_no_pre_frame=0)
for d in DAYS:
    S1 = f'{io.OUT}/step1/{d}'
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=COLS) for f in sorted(os.listdir(f'{S1}/books'))], ignore_index=True)
    bk.sort_values(['mid', 't_ms'], inplace=True, kind='stable')
    reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'ws']).set_index('mid')
    lt = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=TK + ['side', 'ptick', 'shares', 't_ms', 'how', 't_e2', 't_rel'])
    lt = lt[lt.mid.isin(reg.index) & (lt.t_rel >= 0) & (lt.t_rel < 240)].copy()
    timed = (lt.how.values > 0) | np.isfinite(lt.t_e2.values); st['legs_1s_only'] += int((~timed).sum()); lt = lt[timed].copy()
    lt['ta'] = np.where(lt.how.values > 0, lt.t_ms.values, lt.t_e2.values)
    T = lt.groupby(TK, sort=False).agg(side=('side', 'first'), nside=('side', 'nunique'), pmax=('ptick', 'max'), pmin=('ptick', 'min'),
                                        q=('shares', 'sum'), ta=('ta', 'min')).reset_index()
    T = T[(T.nside == 1) & (T.pmax > T.pmin)]
    ours = []
    for g_, f_ in OURS:
        f = pd.read_parquet(f_(d), columns=['mid', 't', 'side', 'pt', 'q', 'through'])
        f = f[f.through.values]
        ours.append(pd.DataFrame(dict(mid=f.mid.values, side=f.side.values, p=f.pt.values.astype(int), q=f.q.values,
                                      ta=reg.loc[f.mid.values].ws.values * 1000.0 + f.t.values * 1000.0, grp=g_)))
    O = pd.concat(ours, ignore_index=True)
    K, LM = load_bn(d)
    TB = {m: x for m, x in T.groupby('mid', sort=False)}; OB = {m: x for m, x in O.groupby('mid', sort=False)}; LB = {k: x for k, x in lt.groupby(['mid', 'side'], sort=False)}
    for mid, g in bk.groupby('mid', sort=False):
        if mid not in reg.index:
            continue
        tf = g.t_ms.values.astype(np.float64); b1 = g.b1_p.values; a1 = g.a1_p.values; vm = (b1 >= 0) & (a1 > b1); tv = tf[vm]; mv = ((b1 + a1) / 2.0)[vm]
        BK = {0: (g[[f'b{i}_p' for i in range(1, 6)]].values, g[[f'b{i}_s' for i in range(1, 6)]].values),
              1: (g[[f'a{i}_p' for i in range(1, 6)]].values, g[[f'a{i}_s' for i in range(1, 6)]].values)}
        for kind, X in (('sweep', TB.get(mid)), ('ours', OB.get(mid))):
            if X is None:
                continue
            for r in X.itertuples(index=False):
                side = int(r.side); P, S = BK[side]; ta = float(r.ta); sg = 1.0 if side == 0 else -1.0
                lo, hi = (int(r.pmin), int(r.pmax)) if kind == 'sweep' else (int(r.p), int(r.p))
                i500 = np.searchsorted(tf, ta - PRE, side='right') - 1; i75 = np.searchsorted(tf, ta - VIS, side='right') - 1; i0 = np.searchsorted(tf, ta, side='left') - 1
                if i500 < 0:
                    st['sweeps_no_pre_frame' if kind == 'sweep' else 'ours_no_pre_frame'] += 1; continue
                Q500 = at_risk(P, S, i500, lo, hi); Q75 = at_risk(P, S, i75, lo, hi); Q0 = at_risk(P, S, i0, lo, hi)
                h = LB.get((mid, side))
                V = float(h.shares.values[(h.ptick.values >= lo) & (h.ptick.values <= hi) & (h.ta.values > tf[i500]) & (h.ta.values <= tf[i75])].sum()) if h is not None else 0.0
                j0 = np.searchsorted(tv, ta, side='left') - 1; j1 = np.searchsorted(tv, ta + 1000.0, side='right') - 1
                inf = sg * (mv[j1] - mv[j0]) if (j0 >= 0 and j1 >= 0) else np.nan
                rec = dict(day=d, q=float(r.q), Q500=Q500, Q75=Q75, Q0=Q0, V=V, informed=inf, levels=hi - lo + 1, ta=ta, dd=-1 if side == 0 else 1)
                if kind == 'ours':
                    rec['grp'] = r.grp; rec['net'] = inf + sg * ((mv[j0] if j0 >= 0 else np.nan) - r.p)
                (SW if kind == 'sweep' else OF).append(rec)
    for L_ in (SW, OF):
        x = [r for r in L_ if r['day'] == d]
        if x:
            t_ = np.array([r['ta'] for r in x]); dd = np.array([r['dd'] for r in x])
            for X_, nm in ((0.2, 'bn02'), (0.5, 'bn05')):
                lag = since_cross(K, LM, t_, dd, X_)
                for r, l_ in zip(x, lag):
                    r[nm] = l_
    del bk, lt, T, O, K, LM
    io.mem_check(f'day {d}')

A = pd.DataFrame(SW); B = pd.DataFrame(OF); A.to_parquet(f'{OUTD}/sweeps.parquet', index=False); B.to_parquet(f'{OUTD}/ours.parquet', index=False)


def measures(x):
    pulled = np.maximum(x.Q500 - x.Q75 - x.V, 0) / x.Q500.where(x.Q500 > 0)
    thinned = 1 - x.Q75 / x.Q500.where(x.Q500 > 0)
    return pulled, thinned


def summary(x, w):
    pulled, thinned = measures(x); ok = pulled.notna()
    ww = w[ok]
    row = dict(n=int(len(x)), pulled_mean=float(np.average(pulled[ok], weights=ww)), pulled_ge30=float(np.average(pulled[ok] >= 0.3, weights=ww)),
               thinned_ge30=float(np.average(thinned[ok] >= 0.3, weights=ww)), empty_by_vis=float(np.average(x.Q75[ok] <= 1e-9, weights=ww)))
    for nm in ('bn02', 'bn05'):
        row[f'{nm}_in_time'] = float(np.average(x[nm].fillna(-1) >= CANCEL_AT, weights=w))
        row[f'{nm}_too_late'] = float(np.average(x[nm].notna() & (x[nm] < CANCEL_AT), weights=w))
    return row


print(f'statuses {st}')
A['cls'] = np.where(A.informed <= -2, 'bad (<= -2)', np.where(A.informed < 0, 'mid (-2..0)', 'good (>= 0)'))
print(f'market sweeps (2+ prices), 8 days, 0-240 s: {len(A)}; share-weighted by sweep shares')
print(f"{'class':14s} {'n':>7s} {'share':>6s} {'levels':>6s} | {'pulled>=':>8s} {'pulled>=30%':>11s} {'thinned>=30%':>12s} {'empty@-75ms':>11s} | {'BN .2 in time':>13s} {'too late':>8s} {'BN .5 in time':>13s} {'too late':>8s}")
out = {}
for cls, x in A.groupby('cls'):
    r = summary(x, x.q); r['share'] = float(x.q.sum() / A.q.sum()); r['levels'] = float(np.average(x.levels, weights=x.q)); out[cls] = r
    print(f"{cls:14s} {r['n']:7d} {r['share']:6.1%} {r['levels']:6.2f} | {r['pulled_mean']:8.1%} {r['pulled_ge30']:11.1%} {r['thinned_ge30']:12.1%} {r['empty_by_vis']:11.1%} | "
          f"{r['bn02_in_time']:13.1%} {r['bn02_too_late']:8.1%} {r['bn05_in_time']:13.1%} {r['bn05_too_late']:8.1%}")
print('our replay swept fills, at our own price (bad = net 1 s <= -2 c/sh)')
for gname, x in B.groupby('grp'):
    x = x[np.isfinite(x.net)]
    for cls, y in ((f'bad', x[x.net <= -2]), ('rest', x[x.net > -2])):
        if not len(y):
            continue
        r = summary(y, y.q); out[f'{gname}|{cls}'] = r
        print(f"  {gname:20s} {cls:5s} n {r['n']:6d} share {y.q.sum() / x.q.sum():5.1%} | pulled {r['pulled_mean']:5.1%}, >=30% {r['pulled_ge30']:5.1%}, thinned >=30% {r['thinned_ge30']:5.1%}, "
              f"empty at -75 ms {r['empty_by_vis']:5.1%} | Binance 0.2 bp in time {r['bn02_in_time']:5.1%} (too late {r['bn02_too_late']:4.1%}), 0.5 bp in time {r['bn05_in_time']:5.1%}")
io.write_json(f'{OUTD}/report.json', dict(statuses=st, rows=out))
