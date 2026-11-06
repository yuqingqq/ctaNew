"""How often does a trade fill makers behind the best while the better prices stand? (2026-11-03; check on the replay's
price-priority assumption, found in pf_fill_mix.) 8 read days, live 0-240 s, timed legs, transaction key as pf_fill_mix.
Per transaction (one taker, its makers, all on one YES side): its best maker price vs the best on that side in the frame
strictly before (gap = ticks the taker skipped); and whether that better best still stands in the frame at or after the
trade (if it does, the better orders were not filled by this trade). Shares are of timed maker volume.
Usage: python3 -m predictfun.research.pf_skip_match
"""
import os
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
TK = ['mid', 'exec_s', 'first_recv_ns', 'taker_index', 'taker_quote', 'price_executed']
acc = []
for d in DAYS:
    S1 = f'{io.OUT}/step1/{d}'
    lt = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=TK + ['side', 'ptick', 'shares', 't_ms', 'how', 't_e2', 't_rel'])
    lt = lt[(lt.t_rel >= 0) & (lt.t_rel < 240) & ((lt.how.values > 0) | np.isfinite(lt.t_e2.values))].copy()
    lt['ta'] = np.where(lt.how.values > 0, lt.t_ms.values, lt.t_e2.values)
    g = lt.groupby(TK, sort=False)
    T = g.agg(side=('side', 'first'), nside=('side', 'nunique'), pmax=('ptick', 'max'), pmin=('ptick', 'min'), q=('shares', 'sum'), ta=('ta', 'min'), n=('ptick', 'size')).reset_index()
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=['mid', 't_ms', 'b1_p', 'a1_p']) for f in sorted(os.listdir(f'{S1}/books'))]).sort_values(['mid', 't_ms'], kind='stable')
    B = {m: x for m, x in bk.groupby('mid', sort=False)}; del bk
    rows = []
    for mid, x in T.groupby('mid', sort=False):
        b = B.get(mid)
        if b is None:
            continue
        tf = b.t_ms.values.astype(float); ta = x.ta.values.astype(float); s = x.side.values
        j0 = np.searchsorted(tf, ta, 'left') - 1; jn = np.searchsorted(tf, ta, 'left'); ok = (j0 >= 0) & (jn < len(tf)) & (x.nside.values == 1)
        j0 = np.clip(j0, 0, None); jn = np.clip(jn, 0, len(tf) - 1)
        best0 = np.where(s == 0, b.b1_p.values[j0], b.a1_p.values[j0]); bestn = np.where(s == 0, b.b1_p.values[jn], b.a1_p.values[jn])
        top = np.where(s == 0, x.pmax.values, x.pmin.values)                     # the transaction's best maker price
        gap = np.where(s == 0, best0 - top, top - best0); ok &= best0 >= 0
        stands = np.where(s == 0, bestn >= best0, (bestn >= 0) & (bestn <= best0))  # the better best still there after
        rows.append(pd.DataFrame(dict(q=x.q.values, gap=gap, stands=stands, multi=x.n.values > 1, dt0=ta - tf[j0]))[ok])
    acc.append(pd.concat(rows).assign(day=d)); del B, lt, T
    io.mem_check(f'day {d}')
A = pd.concat(acc, ignore_index=True); A = A[A.gap >= -50]; v = A.q.sum()
print(f'transactions {len(A)}, timed maker volume {v / 1e6:.1f}M sh (8 days, 0-240 s)')
for lab, m in (('took the best (gap 0)', A.gap == 0), ('better than the shown best (gap < 0: book older than the trade)', A.gap < 0),
               ('skipped 1 tick', A.gap == 1), ('skipped 2+ ticks', A.gap >= 2)):
    sub = A[m]
    print(f"  {lab:62s} {sub.q.sum() / v:6.1%} of volume" + (f" | better best still stands after: {sub.q[sub.stands].sum() / max(sub.q.sum(), 1e-9):5.1%}; "
          f"frame before median {sub.dt0.median():.0f} ms earlier; multi-maker {sub.q[sub.multi].sum() / max(sub.q.sum(), 1e-9):4.0%}" if (m & (A.gap > 0)).any() else ''))
sk = A[(A.gap > 0) & A.stands]
print(f'skip-matches (taker filled behind the best and the better best stood before and after): {sk.q.sum() / v:.1%} of maker volume; '
      f'by day ' + ', '.join(f"{d[4:]} {sk.q[sk.day == d].sum() / A.q[A.day == d].sum():.1%}" for d in DAYS))
io.write_json(f'{io.OUT}/step4/report_skip_match.json', dict(volume_sh=float(v), skip_share=float(sk.q.sum() / v),
              gap_share={str(k): float(x) for k, x in (A.groupby(A.gap.clip(-1, 3)).q.sum() / v).items()}))
