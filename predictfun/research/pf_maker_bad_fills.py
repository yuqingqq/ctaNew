"""Bad fills of the other makers, same definition as ours (2026-11-04; user: "what about top makers, how many bad fills they get").
Per timed maker leg (8 read days; makers ranked per day by live-window volume, bands top 5 / 6-20 / rest): mark-out at +1 s from
the fill price, s (m1 - p), c/sh (m1 = venue mid at fill + 1 s); bad <= -2, small loss -2..0, good >= 0, in shares and USD, for
fills in the first 60 s and in 0-240 s. Ours for reference: pf_anchor_report (P60 / P240, real model and ideal anchor).
Statuses: legs with a 1 s stamp only, no mid after the fill.
Usage: python3 -m predictfun.research.pf_maker_bad_fills
"""
import os
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
acc = []; st = dict(legs_1s_only=0, no_mid=0)
for d in DAYS:
    S1 = f'{io.OUT}/step1/{d}'
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=['mid', 't_ms', 'b1_p', 'a1_p']) for f in sorted(os.listdir(f'{S1}/books'))])
    bk = bk[(bk.b1_p >= 0) & (bk.a1_p > bk.b1_p)].sort_values(['mid', 't_ms'], kind='stable')
    BY = {m: (g.t_ms.values.astype(float), (g.b1_p.values + g.a1_p.values) / 2.0) for m, g in bk.groupby('mid', sort=False)}; del bk
    reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid'])
    lt = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=['mid', 'side', 'ptick', 'shares', 't_ms', 'how', 't_e2', 'signer', 't_rel'])
    lt = lt[lt.mid.isin(reg.mid) & (lt.t_rel >= 0) & (lt.t_rel < 240)].copy()
    vol = lt.groupby('signer').shares.sum().sort_values(ascending=False); rk = pd.Series(np.arange(1, len(vol) + 1), index=vol.index)
    r_ = lt.signer.map(rk).values; lt['band'] = np.where(r_ <= 5, 'top 5', np.where(r_ <= 20, '6-20', 'rest'))
    timed = (lt.how.values > 0) | np.isfinite(lt.t_e2.values); st['legs_1s_only'] += int((~timed).sum()); lt = lt[timed].copy()
    lt['ta'] = np.where(lt.how.values > 0, lt.t_ms.values, lt.t_e2.values)
    out = []
    for mid, g in lt.groupby('mid', sort=False):
        v = BY.get(mid)
        if v is None:
            st['no_mid'] += len(g); continue
        tv, mv = v; i1 = np.searchsorted(tv, g.ta.values + 1000.0, side='right') - 1; ok = i1 >= 0
        st['no_mid'] += int((~ok).sum())
        s = np.where(g.side.values == 0, 1.0, -1.0); c = s * (mv[np.clip(i1, 0, None)] - g.ptick.values)
        out.append(pd.DataFrame(dict(band=g.band.values, w60=g.t_rel.values < 60, q=g.shares.values, c=c))[ok])
    x = pd.concat(out); x['cls'] = np.where(x.c <= -2, 'bad', np.where(x.c < 0, 'small loss', 'good')); x['usd'] = x.q * x.c / 100.0
    for part, m in (('first 60 s', x.w60), ('0-240 s', x.q > -1)):
        a = x[m].groupby(['band', 'cls'])[['q', 'usd']].sum().reset_index(); a['part'] = part; a['day'] = d; acc.append(a)
    del BY, lt
    io.mem_check(f'day {d}')
A = pd.concat(acc).groupby(['part', 'band', 'cls'])[['q', 'usd']].sum()
print(f'statuses {st}; mark-out at +1 s from the fill price; bad <= -2 c/sh, small loss -2..0, good >= 0')
print(f"{'part':10s} {'band':6s} {'shares':>10s} | {'bad %':>6s} {'bad $':>9s} | {'small loss $':>12s} | {'good $':>9s} | {'net c/sh':>8s} {'bad $ / good $':>14s}")
rep = {}
for part in ('first 60 s', '0-240 s'):
    for band in ('top 5', '6-20', 'rest'):
        t = A.loc[(part, band)]; q = t.q.sum(); g = lambda k: float(t.loc[k].usd) if k in t.index else 0.0; gq = lambda k: float(t.loc[k].q) if k in t.index else 0.0
        r = dict(shares=float(q), bad_share=gq('bad') / q, bad_usd=g('bad'), small_usd=g('small loss'), good_usd=g('good'), net_c=100 * t.usd.sum() / q)
        rep[f'{part}|{band}'] = r
        print(f"{part:10s} {band:6s} {q:10.0f} | {r['bad_share']:6.1%} {r['bad_usd']:9.0f} | {r['small_usd']:12.0f} | {r['good_usd']:9.0f} | {r['net_c']:8.2f} {-r['bad_usd'] / r['good_usd']:14.0%}")
io.write_json(f'{io.OUT}/step4/report_maker_bad_fills.json', rep)
