"""B12g (2026-11-11, USER: "after adding the pressure features, it performs better at 1 s ... but 60 s it looks worse. What top makers
performance on these"): the makers' fill value at 1 s, 5 s, 60 s and at settlement, on the public tape. Read-only, descriptive.

Population: maker legs with a research trade time (one time per taker order: earliest WS-timed leg, else earliest E2 estimate, as
B12), in the strategy's window [ws, ws + 240 s) and in the first minute, on the 9 days the B14 / B16 replays used (10-23..27, 10-31,
11-01..03; every window of the day with an outcome) and on 11-08 (the 220 consumed cohort windows of live-samewindow-20261108, as
B12; our own maker id from B12's K2, reviews/b12_20261110/b12.json). Groups per day: makers ranked by shares in 0-240 s (ours
excluded): top 5, ranks 6-20, rest; ours (11-08 only). Value per leg, c/sh, share-weighted: sg (mid(anc + h) - p) for h = 1 / 5 /
60 s (mid = (b1 + a1) / 2 of the last valid frame at or before), and at settlement sg (100 Y - p) (Y = 1 if the outcome is Up; sg =
+1 for a bid maker, -1 for an ask maker).
Beside: the B14 / B16 replays' own c/sh (100 x USD / shares; REF = the live guard b14r, qa b16a, qb b16b; J and X) at 1 s, 60 s and
settlement, the same horizons from the replay's simulated fills (fill model not validated; another population).
Readings DECLARED BEFORE THE FIRST RUN (9 replay days, 0-240 s, all depths, pooled = share-weighted over the 9 days):
 R170 top 5: EDGE-HOLDS if pooled net > 0 at 1 s, 60 s and settlement and > 0 on >= 6 of 9 days at each; EDGE-FADES if pooled net
      > 0 at 1 s and < 0 at 60 s or at settlement; else MIXED.
 R171 ranks 6-20: the same rule.
Control (any failure -> no table): K12 on 11-08, first minute, all depths, this loader reproduces B12's published net at 1 s and 5 s
for top 5 / ranks 6-20 / rest / ours (+0.36 / +0.27, +0.19 / +0.09, -0.33 / -0.46, -0.44 / -0.26 c/sh) within 0.005, and our id's
first-minute shares B12's 961 sh/day within 1 sh/day; the reading rules on synthetic inputs.
Limits: descriptive; whole-market fills, not an order we can place (pro rata is not an order); settlement values carry the binary
outcome's variance (one window's outcome moves every fill in it together), so per-day settlement numbers are noisy; the 9 days were
read by earlier research.
Usage: predictfun/research/run_light.sh b12g_horizons -- python3 -m predictfun.research.pf_b12g_horizons <out_dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102', '20261103']
RP = '/home/yuqing/pmmm-data/review-exports/live-samewindow-20261108'
B12J = '/home/yuqing/ctaNew/predictfun/reviews/b12_20261110/b12.json'
KEYS = ['mid', 'exec_s', 'side', 'ptick', 'shares']
H = (1000.0, 5000.0, 60000.0)
B12_PUB = {'top 5': (0.36, 0.27), '6-20': (0.19, 0.09), 'rest': (-0.33, -0.46), 'ours': (-0.44, -0.26)}
B12_OURS_SH_DAY = 961.0


def load(S1, windows=None):
    """Timed maker legs of one day with their trade time (anc), window start, outcome and per-horizon / settlement values."""
    L = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=KEYS + ['signer', 't_rel', 't_ms', 'how', 't_e2'])
    tk = pd.read_parquet(f'{S1}/legs_tk.parquet', columns=['tk']).tk.values
    if len(tk) != len(L):
        sys.exit(f'REFUSED: legs_tk not aligned in {S1}')
    grp = np.where(tk >= 0, tk, -1 - np.arange(len(L)))
    ws_t = np.where(L.how.values > 0, L.t_ms.values.astype(float), np.nan)
    e2_t = np.where((L.how.values == 0) & np.isfinite(L.t_e2.values), L.t_e2.values, np.nan)
    a_ = pd.Series(ws_t).groupby(grp, sort=False).transform('min').values
    b_ = pd.Series(e2_t).groupby(grp, sort=False).transform('min').values
    L['anc'] = np.where(np.isfinite(a_), a_, b_)
    reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'ws', 'outcome'])
    reg = reg[reg.outcome.isin(['Up', 'Down'])]
    if windows is not None:
        reg = reg[reg.mid.isin(windows)]
    W = dict(zip(reg.mid.values, reg.ws.values * 1000.0)); Y = dict(zip(reg.mid.values, (reg.outcome.values == 'Up').astype(float)))
    Lw = L[L.mid.isin(W)]
    vol = Lw[(Lw.t_rel >= 0) & (Lw.t_rel < 240)].groupby('signer').shares.sum()   # B12's ranking population: every leg, t_rel 0-240
    L = L[L.mid.isin(W) & np.isfinite(L.anc.values)].copy()
    L['ws'] = L.mid.map(W); L['y'] = L.mid.map(Y); L['rel'] = (L.anc - L.ws) / 1000.0
    L = L[(L.rel >= 0) & (L.rel < 240)].copy()
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=['mid', 't_ms', 'b1_p', 'a1_p']) for f in sorted(os.listdir(f'{S1}/books'))])
    bk = bk[(bk.b1_p >= 0) & (bk.a1_p > bk.b1_p) & bk.mid.isin(W)].sort_values(['mid', 't_ms'], kind='stable')
    sg = np.where(L.side.values == 0, 1.0, -1.0); p = L.ptick.values.astype(float)
    vals = {h: np.full(len(L), np.nan) for h in H}
    BK = {m: (g.t_ms.values.astype(float), ((g.b1_p.values + g.a1_p.values) / 2.0)) for m, g in bk.groupby('mid', sort=False)}
    for m, idx in L.groupby('mid', sort=False).indices.items():
        if m not in BK:
            continue
        tf, mv = BK[m]; t = L.anc.values[idx]
        for h in H:
            k = np.searchsorted(tf, t + h, side='right') - 1
            ok = k >= 0
            vals[h][idx[ok]] = sg[idx[ok]] * (mv[k[ok]] - p[idx[ok]])
    for h in H:
        L[f'v{int(h / 1000)}'] = vals[h]
    L['vs'] = sg * (100.0 * L.y.values - p)
    return L, vol


def groups(L, vol, own=None):
    vol = vol[vol.index != own].sort_values(ascending=False, kind='stable')
    rk = {s: i + 1 for i, s in enumerate(vol.index)}
    r_ = L.signer.map(rk).fillna(1e9).values
    return np.where(L.signer.values == own, 'ours', np.where(r_ <= 5, 'top 5', np.where(r_ <= 20, '6-20', 'rest')))


def wmean(x, w):
    k = np.isfinite(x) & (w > 0)
    return float((x[k] * w[k]).sum() / w[k].sum()) if k.any() else float('nan')


def table(L, g):
    out = {}
    for name in ('top 5', '6-20', 'rest', 'ours'):
        s = g == name
        if not s.any():
            continue
        w = L.shares.values[s]
        out[name] = dict(shares=float(w.sum()), v1=wmean(L.v1.values[s], w), v5=wmean(L.v5.values[s], w), v60=wmean(L.v60.values[s], w),
                         vs=wmean(L.vs.values[s], w), missing_60=int(np.isnan(L.v60.values[s]).sum()))
    return out


def label(per_day, pooled):
    hs = ('v1', 'v60', 'vs')
    if not all(np.isfinite(pooled[h]) for h in hs):
        return 'n/a'
    if all(pooled[h] > 0 for h in hs) and all(sum(d[h] > 0 for d in per_day) >= 6 for h in hs):
        return 'EDGE-HOLDS'
    if pooled['v1'] > 0 and (pooled['v60'] < 0 or pooled['vs'] < 0):
        return 'EDGE-FADES'
    return 'MIXED'


def rules_ok():
    up = [dict(v1=1.0, v60=1.0, vs=1.0)] * 9
    return (label(up, dict(v1=1.0, v60=1.0, vs=1.0)) == 'EDGE-HOLDS' and label(up, dict(v1=1.0, v60=-1.0, vs=1.0)) == 'EDGE-FADES'
            and label([dict(v1=1.0, v60=1.0, vs=-1.0)] * 9, dict(v1=1.0, v60=1.0, vs=0.5)) == 'MIXED'
            and label(up, dict(v1=float('nan'), v60=1.0, vs=1.0)) == 'n/a')


def replay_cps():
    from .pf_inv_report import name
    from .pf_ladder_v3_build import ARM
    stems = {'b14r': ('REF', 'preds_t60k'), 'b16a': ('qa', 'preds_qa60km'), 'b16b': ('qb', 'preds_qb60km'), 'b14e': ('B14 EMPH', 'preds_e60km')}
    out = {}
    for tag, (nm, stem) in stems.items():
        for tm in ('J', 'X'):
            sh = u1 = u60 = us = 0.0
            for d in DAYS:
                a = json.load(open(f'{io.OUT}/step4/ladder_tokyoreal_custom_e2_{stem}_{tag}{tm}/{d}/summary.json'))['arms'][name(ARM)]
                sh += a['shares']; u1 += a['mo1_usd']; u60 += a['mo60_usd']; us += a['gross_usd']
            out[f'{nm} {tm}'] = dict(shares=sh, v1=100 * u1 / sh, v60=100 * u60 / sh, vs=100 * us / sh)
    return out


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    if not rules_ok():
        sys.exit('REFUSED: reading rules on synthetic inputs')
    own = json.load(open(B12J))['K']['K2']['own_id']
    sc = json.load(open(f'{RP}/scope.json')); W0 = [int(w['market']) for w in sc['windows']]; days08 = len(W0) / 288.0
    L8, vol8 = load(f'{RP}/research/step1/20261108', windows=set(W0)); g8 = groups(L8, vol8, own)
    f60 = L8.rel.values < 60
    t8 = table(L8[f60], g8[f60])
    k12 = {n: (round(t8[n]['v1'], 2), round(t8[n]['v5'], 2)) for n in B12_PUB if n in t8}
    ok = all(n in t8 and abs(t8[n]['v1'] - B12_PUB[n][0]) <= 0.005 + 1e-9 and abs(t8[n]['v5'] - B12_PUB[n][1]) <= 0.005 + 1e-9 for n in B12_PUB)
    ours_sh_day = t8.get('ours', {}).get('shares', 0.0) / days08
    ok = ok and abs(ours_sh_day - B12_OURS_SH_DAY) <= 1.0
    ctl = dict(K12=k12, published=B12_PUB, ours_sh_day=ours_sh_day, ok=bool(ok))
    print('control', ctl, flush=True)
    if not ok:
        json.dump(dict(control=ctl), open(f'{out}/b12g.json', 'w'), indent=1, default=float); sys.exit('CONTROL FAILED -> no table')
    res = dict(control=ctl, day_1108=dict(first_minute=t8, window_0_240=table(L8, g8)), days={}, pooled={})
    allL = []
    for d in DAYS:
        L, vol = load(f'{io.OUT}/step1/{d}'); g = groups(L, vol)
        res['days'][d] = dict(window_0_240=table(L, g), first_minute=table(L[L.rel.values < 60], g[L.rel.values < 60]))
        L = L.assign(g=g)[['g', 'shares', 'v1', 'v5', 'v60', 'vs', 'rel']]; L['day'] = d; allL.append(L)
        print(d, {k: (round(v['v1'], 3), round(v['v60'], 3), round(v['vs'], 3)) for k, v in res['days'][d]['window_0_240'].items()}, flush=True)
        io.mem_check(d)
    A = pd.concat(allL, ignore_index=True)
    res['pooled'] = dict(window_0_240=table(A, A.g.values), first_minute=table(A[A.rel.values < 60], A.g.values[A.rel.values < 60]))
    for nm, key in (('top 5', 'R170'), ('6-20', 'R171')):
        per = [res['days'][d]['window_0_240'][nm] for d in DAYS]
        res[key] = label(per, res['pooled']['window_0_240'][nm])
    res['replay'] = replay_cps()
    json.dump(res, open(f'{out}/b12g.json', 'w'), indent=1, default=float)
    Lx = [f"B12g makers' fill value by horizon (c/sh). R170 top 5 = {res['R170']}; R171 ranks 6-20 = {res['R171']}",
          '', '9 replay days, 0-240 s, all depths (pooled; shares):']
    for nm, v in res['pooled']['window_0_240'].items():
        Lx.append(f"  {nm:6s} 1 s {v['v1']:+.3f} | 5 s {v['v5']:+.3f} | 60 s {v['v60']:+.3f} | settle {v['vs']:+.3f} | {v['shares']:,.0f} sh")
    Lx.append('9 replay days, first minute:')
    for nm, v in res['pooled']['first_minute'].items():
        Lx.append(f"  {nm:6s} 1 s {v['v1']:+.3f} | 5 s {v['v5']:+.3f} | 60 s {v['v60']:+.3f} | settle {v['vs']:+.3f} | {v['shares']:,.0f} sh")
    Lx.append('11-08 cohort, 0-240 s:')
    for nm, v in res['day_1108']['window_0_240'].items():
        Lx.append(f"  {nm:6s} 1 s {v['v1']:+.3f} | 5 s {v['v5']:+.3f} | 60 s {v['v60']:+.3f} | settle {v['vs']:+.3f} | {v['shares']:,.0f} sh")
    Lx.append('11-08 cohort, first minute (B12 control population):')
    for nm, v in res['day_1108']['first_minute'].items():
        Lx.append(f"  {nm:6s} 1 s {v['v1']:+.3f} | 5 s {v['v5']:+.3f} | 60 s {v['v60']:+.3f} | settle {v['vs']:+.3f} | {v['shares']:,.0f} sh")
    Lx.append('per day, top 5 (0-240 s): ' + ', '.join(f"{d[4:]} {res['days'][d]['window_0_240']['top 5']['v1']:+.2f}/{res['days'][d]['window_0_240']['top 5']['v60']:+.2f}/{res['days'][d]['window_0_240']['top 5']['vs']:+.2f}" for d in DAYS) + '  (1 s / 60 s / settle)')
    Lx.append('replay fills, 9 days (c/sh): ' + '; '.join(f"{k} 1 s {v['v1']:+.2f} 60 s {v['v60']:+.2f} settle {v['vs']:+.2f}" for k, v in res['replay'].items()))
    open(f'{out}/b12g.txt', 'w').write('\n'.join(Lx) + '\n'); print('\n'.join(Lx))


if __name__ == '__main__':
    main()
