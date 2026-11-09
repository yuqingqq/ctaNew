"""Step 1d: time the untimed REST trades from book-level drops, calibrated on WS-timed trades.

Cell = (market, executedAt second S, YES side, price tick). executedAt is the true time rounded to the
second, so the trade happened in about [S-0.5 s, S+0.5 s). Candidates = book frames in [S-600, S+600) ms
where the displayed size at (side, price) DROPPED versus the previous frame (top-5 depth; ticks below the
visible depth are unknown, never counted as drops).
Estimators (scored on cells whose legs all carry a WS time, truth = median WS time of those legs):
  E1 largest drop    E2 drop closest to the cell volume    E3 earliest drop >= half the cell volume
  baseline S (the current "mid" placement)
The best estimator by median |error| is written for the untimed legs (how2 = 3) only if it beats the
baseline by a declared margin: median |err| <= 0.5 x baseline AND share within 100 ms >= 2 x baseline.
Usage: python3 -m predictfun.research.pf_step1_booktime 20261023
Writes step1/<day>/legs_timed2.parquet (t_ms2, how2: 1 WS, 3 book, 0 none) + booktime.json
"""
import sys, os, collections
from .pf_trade_tape import refuse_legacy_clock
refuse_legacy_clock()  # Superseded: executedAt is a confirmation stamp, not a rounded matching stamp.
import numpy as np, pandas as pd
from . import pf_io as io

day = sys.argv[1]; S1 = f'{io.OUT}/step1/{day}'
L = pd.read_parquet(f'{S1}/legs_timed.parquet')
cols = ['mid', 't_ms'] + [f'{s}{i}_{k}' for s in 'ba' for i in range(1, 6) for k in 'ps']
B = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=cols) for f in sorted(os.listdir(f'{S1}/books'))], ignore_index=True)
B.sort_values(['mid', 't_ms'], inplace=True, kind='stable')
cells = L.groupby(['mid', 'exec_s', 'side', 'ptick']).agg(vol=('shares', 'sum'), n=('shares', 'size'),
                                                          n_timed=('how', lambda h: int((h > 0).sum())),
                                                          truth=('t_ms', 'median')).reset_index()
cells['all_timed'] = cells.n_timed == cells.n
est = {e: np.full(len(cells), np.nan) for e in ('E1', 'E2', 'E3', 'E4')}
ncand = np.zeros(len(cells), dtype=np.int32)
cell_by_mid = {m: g.index.values for m, g in cells.groupby('mid')}
for mid, g in B.groupby('mid', sort=False):
    if mid not in cell_by_mid:
        continue
    n = len(g); t = g.t_ms.values.astype(np.float64)
    Sz = np.full((n, 2, 101), np.nan, dtype=np.float32)
    for side, pre in ((0, 'b'), (1, 'a')):
        P = np.stack([g[f'{pre}{i}_p'].values for i in range(1, 6)], 1)
        Q = np.stack([g[f'{pre}{i}_s'].values for i in range(1, 6)], 1).astype(np.float32)
        full = P[:, 4] >= 0                                  # 5 levels shown -> depth beyond level 5 unknown
        edge = np.where(full, P[:, 4], 0 if side == 0 else 100)
        ticks = np.arange(101)
        vis = (ticks[None, :] >= edge[:, None]) if side == 0 else (ticks[None, :] <= edge[:, None])
        Sz[:, side, :] = np.where(vis, 0.0, np.nan)
        for i in range(5):
            ok = P[:, i] >= 0
            Sz[np.where(ok)[0], side, P[ok, i]] = Q[ok, i]
    D = np.nan_to_num(np.maximum(Sz[:-1] - Sz[1:], 0.0), nan=0.0)   # drop at frame k+1
    D = np.concatenate([np.zeros((1, 2, 101), np.float32), D])
    del Sz
    for ci in cell_by_mid[mid]:
        S = cells.exec_s.values[ci] * 1000.0; side = cells.side.values[ci]; pt = cells.ptick.values[ci]; vol = cells.vol.values[ci]
        a = np.searchsorted(t, S - 600.0); b = np.searchsorted(t, S + 600.0)
        d = D[a:b, side, pt]; tt = t[a:b]
        m = d > 1e-6
        ncand[ci] = int(m.sum())
        if not m.any():
            continue
        d = d[m]; tt = tt[m]
        est['E1'][ci] = tt[np.argmax(d)]
        j2 = np.argmin(np.abs(d - vol))
        est['E2'][ci] = tt[j2]
        if abs(d[j2] - vol) <= max(0.1 * vol, 0.5):
            est['E4'][ci] = tt[j2]
        h = np.where(d >= 0.5 * vol)[0]
        est['E3'][ci] = tt[h[0]] if len(h) else tt[np.argmax(d)]
    io.mem_check(f'mid {mid}')
cal = cells.all_timed.values
truth = cells.truth.values
def score(x):
    e = np.abs(x[cal] - truth[cal]); e = e[np.isfinite(e)]
    return dict(n=int(len(e)), coverage=round(float(len(e) / cal.sum()), 3), med_abs_ms=round(float(np.median(e)), 1),
                p75=round(float(np.percentile(e, 75)), 1), p90=round(float(np.percentile(e, 90)), 1),
                within_100ms=round(float((e <= 100).mean()), 3), within_250ms=round(float((e <= 250).mean()), 3))
base = cells.exec_s.values * 1000.0
res = dict(day=day, n_cells=int(len(cells)), n_cal_cells=int(cal.sum()), baseline_S=score(base))
for k, v in est.items():
    res[k] = score(v)
    # baseline on the SAME covered cells, for a like-for-like margin
    cov = np.isfinite(v)
    e0 = np.abs(base[cal & cov] - truth[cal & cov]); res[k]['baseline_same_cells_med_abs_ms'] = round(float(np.median(e0)), 1)
    res[k]['baseline_same_cells_within_100ms'] = round(float((e0 <= 100).mean()), 3)
# E4 (declared 2026-11-01 04:40Z, after E2's fat tail was seen on 10-23 and before E4 was scored):
# adopt E4 if median|err| <= 0.5 x baseline (same cells) AND p90 <= 250 ms; else fall back to the E1-E3 rule.
b4 = res['E4']
if b4['n'] > 0 and b4['med_abs_ms'] <= 0.5 * b4['baseline_same_cells_med_abs_ms'] and b4['p90'] <= 250:
    best, adopt, rule = 'E4', True, 'E4: median|err| <= 0.5 x baseline AND p90 <= 250 ms'
else:
    best = min(('E1', 'E2', 'E3'), key=lambda k: res[k]['med_abs_ms'])
    bb = res[best]
    adopt = bb['med_abs_ms'] <= 0.5 * bb['baseline_same_cells_med_abs_ms'] and bb['within_100ms'] >= 2 * bb['baseline_same_cells_within_100ms']
    rule = 'median|err| <= 0.5 x baseline AND within-100ms >= 2 x baseline (same covered cells)'
res.update(best=best, adopted=bool(adopt), rule=rule)
# busy vs quiet (calibration cells): candidates in the window as a busyness proxy
for lab, msk in (('quiet_<=3_candidates', ncand <= 3), ('busy_>3_candidates', ncand > 3)):
    x = est[best]; e = np.abs(x[cal & msk] - truth[cal & msk]); e = e[np.isfinite(e)]
    res[f'{best}_{lab}'] = dict(n=int(len(e)), med_abs_ms=round(float(np.median(e)), 1) if len(e) else None)
# write per-leg times
cells['t_book'] = est[best] if adopt else np.nan
cells['t_e2'] = est['E2']; cells['t_e4'] = est['E4']
L2 = L.merge(cells[['mid', 'exec_s', 'side', 'ptick', 't_book', 't_e2', 't_e4']], on=['mid', 'exec_s', 'side', 'ptick'], how='left')
L2['t_ms2'] = np.where(L2.how > 0, L2.t_ms, L2.t_book)
L2['how2'] = np.where(L2.how > 0, 1, np.where(np.isfinite(L2.t_book), 3, 0)).astype(np.int8)
vol = L2.shares.sum()
res['volume_share'] = {f'how2={k}': round(float(L2.shares[L2.how2 == k].sum() / vol), 4) for k in (1, 3, 0)}
res['volume_share_e4_untimed'] = round(float(L2.shares[(L2.how == 0) & np.isfinite(L2.t_e4)].sum() / vol), 4)
L2.drop(columns=['t_book']).to_parquet(f'{S1}/legs_timed2.parquet', index=False)
io.write_json(f'{S1}/booktime.json', res)
for k, v in res.items():
    print(k, v)
