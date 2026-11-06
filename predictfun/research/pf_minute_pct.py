"""Per-minute percentiles of the per-side guard (descriptive; user 2026-11-05: "we just need some historical percentile values").
Guard value v = out-of-fold prediction - that fold's global 60% pull line (preds_t60k: real taker-order labels, start withhold), in c.
(A) the value of v at percentiles 50..97.5 within each minute of the window (0-60, 60-120, 120-180, 180-240 s), bid and ask pooled,
    8 read days; 11-03 beside. Today's rule: pull at v <= 0 (global 60th percentile), size up at v >= 0.06.
(B) realized label per percentile bucket within each minute, on the guard grid rows (every 500 ms; label = value of keeping an
    order at the known best through the next 500 ms exposure, frame-stamp timing, c / share; 0 when not swept): mean value, swept
    share, bad share (<= -2 c), mean when swept. Buckets use the minute's own percentiles from (A).
Usage: python3 -m predictfun.research.pf_minute_pct
"""
import numpy as np, pandas as pd
from . import pf_io as io

D8 = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
P = [50, 60, 70, 80, 85, 90, 93, 95, 97.5]; MIN = ((0, 60), (60, 120), (120, 180), (180, 240))


def load(d):
    p = pd.read_parquet(f'{io.OUT}/step5/{d}/preds_t60k.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
    ws = pd.read_parquet(f'{io.OUT}/step1/{d}/registry.parquet', columns=['mid', 'ws']).set_index('mid').ws
    p['rel'] = (p.tau.values - p.mid.map(ws).values * 1000.0) / 1000.0; p['day'] = d
    return p[(p.rel >= 0) & (p.rel < 240)]


A = pd.concat([load(d) for d in D8], ignore_index=True); A['day'] = A.day.astype('category'); F = load('20261103')
minute = lambda x: np.minimum((x.rel.values // 60).astype(int), 3)
A['m'] = minute(A); F['m'] = minute(F)
vals = lambda x: np.r_[x.v_bid.values, x.v_ask.values]
print('(A) guard value v (c above today\'s pull line) at each percentile, within each minute; bid + ask pooled')
print(f"{'minute':10s} " + ' '.join(f'p{q:<6g}' for q in P) + '   share v<=0  share v>=0.06')
Q = {}
for i, (a, b) in enumerate(MIN):
    v = vals(A[A.m == i]); v = v[np.isfinite(v)]; Q[i] = np.percentile(v, P)
    v3 = vals(F[F.m == i]); v3 = v3[np.isfinite(v3)]; q3 = np.percentile(v3, P)
    print(f'{a:3d}-{b:3d} s  ' + ' '.join(f'{x:+7.3f}' for x in Q[i]) + f'   {np.mean(v <= 0):9.1%}  {np.mean(v >= 0.06):9.1%}   (rows {len(v)})')
    print(f'  11-03    ' + ' '.join(f'{x:+7.3f}' for x in q3) + f'   {np.mean(v3 <= 0):9.1%}  {np.mean(v3 >= 0.06):9.1%}')
G = pd.read_parquet(f'{io.OUT}/step4/badfill_info_tx_tk/grid.parquet', columns=['day', 'mid', 'tau', 't_rel', 'v_bid', 'v_ask'])
G['day'] = pd.Categorical(G.day, categories=A.day.cat.categories)
J = G.merge(A[['day', 'mid', 'tau', 'v_bid', 'v_ask']], on=['day', 'mid', 'tau'], how='inner', suffixes=('_y', ''))
print(f'\n(B) realized value per percentile bucket within each minute (grid rows joined {len(J)} of {len(G)}; 8 read days, bid + ask pooled)')
print(f"{'minute':10s} {'bucket':>12s} {'v range (c)':>18s} {'mean c/sh':>10s} {'swept':>7s} {'bad':>6s} {'c/sh if swept':>13s} {'rows':>8s}")
EDGES = [0, 50, 60, 70, 80, 90, 95, 100]
for i, (a, b) in enumerate(MIN):
    g = J[(J.t_rel >= a) & (J.t_rel < b)]
    pv = np.r_[g.v_bid.values, g.v_ask.values]; y = np.r_[g.v_bid_y.values, g.v_ask_y.values]
    ok = np.isfinite(pv); pv, y = pv[ok], y[ok]
    allv = vals(A[A.m == i]); allv = allv[np.isfinite(allv)]; cut = np.percentile(allv, EDGES[1:-1])
    k = np.searchsorted(cut, pv, side='right')
    for j in range(len(EDGES) - 1):
        s = k == j; yy = y[s]; sw = yy != 0
        lo = -np.inf if j == 0 else cut[j - 1]; hi = np.inf if j == len(EDGES) - 2 else cut[j]
        print(f'{a:3d}-{b:3d} s  {EDGES[j]:>4d}-{EDGES[j + 1]:<4d} pct  [{lo:+7.3f},{hi:+7.3f})  {yy.mean():+10.4f} {sw.mean():7.1%} {np.mean(yy <= -2):6.2%} '
              f'{(yy[sw].mean() if sw.any() else np.nan):+13.3f} {s.sum():8d}')
    print()
