"""P&L decomposition of market-making arms (2026-11-03; user: the spread must cover informed traders, passive unwinds and
leftovers). For every fill (side s = +1 buy / -1 sell, price p, shares q, venue mid m0 = last valid frame strictly before
the fill, m1 = mid 1 s later, mS = mid at stop + 100 ms, y = outcome 0/100), all in USD:
  spread    s q (m0 - p) / 100            earned at the fill against the mid
  informed  s q (m1 - m0) / 100           the mid moving against us within 1 s (adverse selection)
  drift     s q (mS - m1) / 100           inventory P&L while waiting to unwind (from 1 s after the fill to the stop)
  carry     s q (y - mS) / 100            the leftover's settlement against the mid at the stop
spread + informed + drift + carry = gross settlement P&L (checked per arm). Mid lookups use the step-1 venue books.
Usage: python3 -m predictfun.research.pf_pnl_decomp
"""
import bisect, os
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
S4 = f'{io.OUT}/step4'
ST = 'MICRO+G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1+XS0+XV'
ARMS = [('full window (to 240 s)', f'{ST}/mid@A1:5:25:0', 'o3a', 240.0),
        ('first 60 s', f'{ST}+S60/mid@A1:5:25:0', 'lc', 60.0),
        ('first 60 s + cap 10', f'{ST}+S60+CAP10/mid@A1:5:25:0', 'lc', 60.0)]


def stem(n):
    return n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')


def subdir(d, tag):
    return f"{S4}/live_tokyoreal_custom_e2_{'preds_c3' if d >= '20261101' else 'preds_o3'}_{tag}"


res = {lab: dict(spread=0.0, informed=0.0, drift=0.0, carry=0.0, gross=0.0, shares=0.0) for lab, *_ in ARMS}
for d in DAYS:
    S1 = f'{io.OUT}/step1/{d}'
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=['mid', 't_ms', 'b1_p', 'a1_p']) for f in sorted(os.listdir(f'{S1}/books'))])
    bk = bk[(bk.b1_p >= 0) & (bk.a1_p > bk.b1_p)].sort_values(['mid', 't_ms'], kind='stable')
    BY = {m: (g.t_ms.values.astype(np.float64), ((g.b1_p.values + g.a1_p.values) / 2.0)) for m, g in bk.groupby('mid', sort=False)}
    del bk
    reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'ws', 'outcome']).set_index('mid')
    for lab, arm, tag, stop in ARMS:
        f = pd.read_parquet(f'{subdir(d, tag)}/{d}/fills/{stem(arm)}.parquet', columns=['mid', 't', 'side', 'pt', 'q', 'settle'])
        if not len(f):
            continue
        sg = np.where(f.side.values == 0, 1.0, -1.0); q = f.q.values; p = f.pt.values.astype(float)
        m0 = np.full(len(f), np.nan); m1 = np.full(len(f), np.nan); mS = np.full(len(f), np.nan); yv = np.full(len(f), np.nan)
        for i, (mid, t) in enumerate(zip(f.mid.values, f.t.values)):
            tv, mv = BY.get(mid, (None, None))
            if tv is None:
                continue
            ws = float(reg.loc[mid].ws) * 1000.0; ta = ws + t * 1000.0
            j0 = bisect.bisect_left(tv, ta) - 1; j1 = bisect.bisect_right(tv, ta + 1000.0) - 1; jS = bisect.bisect_right(tv, ws + stop * 1000.0 + 100.0) - 1
            m0[i] = mv[j0] if j0 >= 0 else np.nan; m1[i] = mv[j1] if j1 >= 0 else np.nan; mS[i] = mv[jS] if jS >= 0 else np.nan
            yv[i] = 100.0 if reg.loc[mid].outcome == 'Up' else 0.0
        ok = np.isfinite(m0) & np.isfinite(m1) & np.isfinite(mS)
        r = res[lab]
        r['spread'] += float(np.sum((sg * q * (m0 - p) / 100)[ok])); r['informed'] += float(np.sum((sg * q * (m1 - m0) / 100)[ok]))
        r['drift'] += float(np.sum((sg * q * (mS - m1) / 100)[ok])); r['carry'] += float(np.sum((sg * q * (yv - mS) / 100)[ok]))
        r['gross'] += float(np.sum(f.settle.values[ok])); r['shares'] += float(np.sum(q[ok]))
    del BY
    io.mem_check(f'day {d}')
print(f"{'arm (8 days, USD)':24s} {'spread':>8s} {'informed':>9s} {'drift':>8s} {'carry':>8s} | {'sum':>8s} {'gross':>8s} | per share: spread / informed / drift / carry (c)")
for lab, r in res.items():
    tot = r['spread'] + r['informed'] + r['drift'] + r['carry']; sh = max(r['shares'], 1)
    print(f"{lab:24s} {r['spread']:8.0f} {r['informed']:9.0f} {r['drift']:8.0f} {r['carry']:8.0f} | {tot:8.0f} {r['gross']:8.0f} | "
          f"{100 * r['spread'] / sh:5.2f} / {100 * r['informed'] / sh:5.2f} / {100 * r['drift'] / sh:5.2f} / {100 * r['carry'] / sh:5.2f}  ({sh:,.0f} sh)")
io.write_json(f'{S4}/report_pnl_decomp.json', res)
