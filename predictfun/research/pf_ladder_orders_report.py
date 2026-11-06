"""Ladder order log (PF_ORDLOG=1 rerun of one lad1 arm, tag lad1q; 2026-11-03, user: "Why we are still in the back?").

Per order: queue ahead at activation (shown size at our price), life, end reason, whether it reached the front of its
queue (queue ahead 0 at its end, or a queue fill) and how it filled. Parity: the rerun's summary for the arm must equal
lad1's field for field (the log is not allowed to change the replay); a mismatch aborts.
Usage: python3 -m predictfun.research.pf_ladder_orders_report
"""
import json
import numpy as np, pandas as pd
from . import pf_io as io

READ = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031']; FRESH = ['20261101', '20261102']; ALL = READ + FRESH
S4 = f'{io.OUT}/step4'; ARM = 'K=3,z=5,S=60,TH=0.5,CAP=50,BUD=75,RF=1000,THIN=30,CD=500'; ST = ARM.replace('=', '').replace(',', '_')


def ddir(d, tag):
    return f"{S4}/ladder_{'preds_c3' if d in FRESH else 'preds_o3'}_{tag}/{d}"


for d in ALL:
    a = json.load(open(f'{ddir(d, "lad1")}/summary.json'))['arms'][ARM]; b = json.load(open(f'{ddir(d, "lad1q")}/summary.json'))['arms'][ARM]
    if a != b:
        raise SystemExit(f'PARITY FAIL {d}: ' + str({k: (a[k], b.get(k)) for k in a if a[k] != b.get(k)}))
print(f'parity: lad1q summary == lad1 summary for {ARM} on all 8 days')
O = pd.concat([pd.read_parquet(f'{ddir(d, "lad1q")}/orders/{ST}.parquet').assign(day=d) for d in ALL], ignore_index=True)
lv = O[O.live]; n_w = O.groupby('day').mid.nunique().sum()
front = (lv.ahead_end <= 1e-9) | (lv.fill_queue > 0)
filled = lv[(lv.fill_queue + lv.fill_swept) > 0]
print(f'orders {len(O)} over {n_w} windows ({len(O) / n_w:.0f} per 60 s window); live {len(lv)}, refused post-only {(~O.live).sum()}')
print(f"life of live orders (ms): p25 {lv.life_ms.quantile(.25):.0f}  median {lv.life_ms.median():.0f}  p75 {lv.life_ms.quantile(.75):.0f}  "
      f"share < 1 s {(lv.life_ms < 1000).mean():.0%}, < 5 s {(lv.life_ms < 5000).mean():.0%}")
print('end reason (live orders):', (lv.why.value_counts(normalize=True) * 100).round(1).to_dict())
print('queue ahead at activation (shares; our order is 5):')
for dd, g in lv.groupby('d'):
    fr = front[g.index]
    print(f"   {dd} tick(s) behind best: n {len(g):6d}  ahead median {g.ahead0.median():6.0f}  p25/p75 {g.ahead0.quantile(.25):.0f}/{g.ahead0.quantile(.75):.0f}  "
          f"reached the front {fr.mean():5.1%}  ahead left at end (median) {g.ahead_end.median():6.0f}")
print(f'reached the front of the queue: {front.mean():.1%} of live orders')
q = filled.fill_queue.sum(); s = filled.fill_swept.sum()
print(f'filled orders {len(filled)} ({len(filled) / len(lv):.1%} of live); shares from the queue {q:.0f}, swept {s:.0f} (swept {s / (q + s):.0%})')
print('cancelled orders, by reason: median ahead at activation -> at cancel')
for why, g in lv[~lv.why.isin(['filled'])].groupby('why'):
    print(f"   {why:8s} n {len(g):6d}  {g.ahead0.median():6.0f} -> {g.ahead_end.median():6.0f}   life median {g.life_ms.median():6.0f} ms")
io.write_json(f'{S4}/report_ladder_orders_lad1q.json', dict(arm=ARM, orders=len(O), windows=int(n_w), live=len(lv), life_ms_median=float(lv.life_ms.median()),
              why=lv.why.value_counts().to_dict(), ahead0_median_by_d={int(k): float(v) for k, v in lv.groupby('d').ahead0.median().items()},
              reached_front=float(front.mean()), queue_shares=float(q), swept_shares=float(s)))
