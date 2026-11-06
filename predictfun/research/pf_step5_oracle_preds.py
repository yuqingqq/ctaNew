"""Ideal-anchor predictions (declared in STATUS.yml 2026-11-04 01:10Z): pred_1000 = y1000, pred_250 = y250, the realised venue
mid 1 s / 250 ms after our knowledge time minus the known mid (rows_rl_d2, Tokyo clock with 2 ms decision), on the same rows as
preds_o3 / preds_c3 (alignment asserted). An upper bound for 'if we have the anchor price', never a candidate.
Usage: python3 -m predictfun.research.pf_step5_oracle_preds
"""
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
rep = {}
for d in DAYS:
    r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows_rl_d2.parquet', columns=['mid', 'tau', 'y250', 'y1000'])
    a = pd.read_parquet(f"{io.OUT}/step5/{d}/{'preds_c3' if d >= '20261101' else 'preds_o3'}.parquet", columns=['mid', 'tau'])
    assert len(a) == len(r) and (a.mid.values == r.mid.values).all() and np.allclose(a.tau.values, r.tau.values), f'{d}: row alignment'
    o = pd.DataFrame(dict(mid=r.mid.values, tau=r.tau.values, pred_250=r.y250.values.astype(np.float32), pred_1000=r.y1000.values.astype(np.float32)))
    o.to_parquet(f'{io.OUT}/step5/{d}/preds_oracle.parquet', index=False)
    rep[d] = dict(rows=len(o), valid_1000=int(o.pred_1000.notna().sum()), flagged_025=float((o.pred_1000.abs() >= 0.25).mean()))
    print(d, rep[d], flush=True); del r, a, o
io.write_json(f'{io.OUT}/step5/oracle_preds_report.json', rep)
