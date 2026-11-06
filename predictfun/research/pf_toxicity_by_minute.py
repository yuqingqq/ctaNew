"""Toxicity by minute of the window: all market maker fills (REST legs) vs our fills (2026-11-03; user: are later fills more
informed, or do we act poorly?). For each maker fill (book side s = +1 bid / -1 ask, price p, shares q; m0 = venue mid
just before, m1 = 1 s later, m60 = 60 s later):
  spread = s (m0 - p), informed = s (m1 - m0), reversal = s (m60 - m1)  [cents per share, share-weighted]
Market fills: step-1 legs in the live window (WS / book-drop time where known, else executedAt). Ours: the full-window
base arm (sequential, venue throttle, Tokyo clock, decision 2 ms).
Usage: python3 -m predictfun.research.pf_toxicity_by_minute
"""
import bisect, os
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
S4 = f'{io.OUT}/step4'; ARM = 'MICRO_G4_G2_G6_G5_G3_G1_M025_XNA_XMIN1_XS0_XV__mid_szA1-5-25-0'


def marks(BY, mids, tabs):
    m0 = np.full(len(mids), np.nan); m1 = m0.copy(); m60 = m0.copy()
    for i, (mid, ta) in enumerate(zip(mids, tabs)):
        v = BY.get(mid)
        if v is None:
            continue
        tv, mv = v
        j0 = bisect.bisect_left(tv, ta) - 1; j1 = bisect.bisect_right(tv, ta + 1000) - 1; j6 = bisect.bisect_right(tv, ta + 60000) - 1
        if j0 >= 0:
            m0[i], m1[i], m60[i] = mv[j0], mv[j1], mv[j6]
    return m0, m1, m60


acc = []
for d in DAYS:
    S1 = f'{io.OUT}/step1/{d}'
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=['mid', 't_ms', 'b1_p', 'a1_p']) for f in sorted(os.listdir(f'{S1}/books'))])
    bk = bk[(bk.b1_p >= 0) & (bk.a1_p > bk.b1_p)].sort_values(['mid', 't_ms'], kind='stable')
    BY = {m: (g.t_ms.values.astype(np.float64), (g.b1_p.values + g.a1_p.values) / 2.0) for m, g in bk.groupby('mid', sort=False)}
    del bk
    reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'ws']).set_index('mid')
    lt = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=['mid', 'exec_s', 'side', 'ptick', 'shares', 't_ms', 'how', 't_e2'])
    lt = lt[lt.mid.isin(reg.index)]
    ta = np.where(lt.how.values > 0, lt.t_ms.values, np.where(np.isfinite(lt.t_e2.values), lt.t_e2.values, lt.exec_s.values * 1000.0))
    ws = reg.loc[lt.mid.values].ws.values * 1000.0; rel = (ta - ws) / 1000.0
    keep = (rel >= 0) & (rel < 240); lt = lt[keep]; ta = ta[keep]; rel = rel[keep]
    m0, m1, m60 = marks(BY, lt.mid.values, ta)
    s = np.where(lt.side.values == 0, 1.0, -1.0); p = lt.ptick.values.astype(float); q = lt.shares.values
    acc.append(pd.DataFrame(dict(who='market', minute=(rel // 60).astype(int), q=q, spread=s * (m0 - p), informed=s * (m1 - m0), reversal=s * (m60 - m1))))
    sub = f"{S4}/live_tokyoreal_custom_e2_{'preds_c3' if d >= '20261101' else 'preds_o3'}_o3a"
    f = pd.read_parquet(f'{sub}/{d}/fills/{ARM}.parquet', columns=['mid', 't', 'side', 'pt', 'q', 'through'])
    tf = reg.loc[f.mid.values].ws.values * 1000.0 + f.t.values * 1000.0
    m0, m1, m60 = marks(BY, f.mid.values, tf); s = np.where(f.side.values == 0, 1.0, -1.0); p = f.pt.values.astype(float)
    acc.append(pd.DataFrame(dict(who='ours', minute=(f.t.values // 60).clip(max=3).astype(int), q=f.q.values, spread=s * (m0 - p),
                                 informed=s * (m1 - m0), reversal=s * (m60 - m1), sweep=f.through.values)))
    del BY
    io.mem_check(f'day {d}')
A = pd.concat(acc); A = A[np.isfinite(A.spread) & np.isfinite(A.informed) & np.isfinite(A.reversal)]
def wavg(g, c):
    return float(np.average(g[c], weights=g.q))
out = A.groupby(['who', 'minute']).apply(lambda g: pd.Series(dict(shares_k=g.q.sum() / 1e3, spread=wavg(g, 'spread'), informed=wavg(g, 'informed'),
                                                                  net_1s=wavg(g, 'spread') + wavg(g, 'informed'), reversal_60s=wavg(g, 'reversal'),
                                                                  net_60s=wavg(g, 'spread') + wavg(g, 'informed') + wavg(g, 'reversal'))))
print('cents per share, share-weighted (maker view: + = good for the maker); 8 days, live minutes 0-3')
print(out.round(2).to_string())
io.write_json(f'{S4}/report_toxicity_by_minute.json', {f'{k[0]}_{k[1]}': v for k, v in out.round(4).to_dict('index').items()})
