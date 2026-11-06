"""Matched-pull control for stage 3 (declared in STATUS.yml 2026-11-04): the current guard's input ranks the exposures instead of
the trained model. Per side, value = s * pred_1000 (bid +1, ask -1; low = pull first), shifted by its k-quantile (k = 10 / 20 / 30%)
on the training days' grid rows (leave one day out for the 8 read days; all 8 days for a fresh day), so the engine's V0 token pulls
the same share of exposures as preds_g<k>. Output step5/<day>/preds_b<k>.parquet (pred_1000 / pred_250 copied unchanged).
Usage: python3 -m predictfun.research.pf_step5_baseguard_preds [fresh_day ...]
"""
import sys
import numpy as np, pandas as pd
from . import pf_io as io

import os
READ = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
KS = tuple(int(x) for x in os.environ.get('PF_GUARD_KS', '10,20,30').split(','))
G = pd.read_parquet(f'{io.OUT}/step4/badfill_info/grid.parquet', columns=['day', 'pred_1000'])
rep = {}
for d in READ + sys.argv[1:]:
    tr = (G.day != d).values if d in READ else np.ones(len(G), bool)
    p = G.pred_1000.values[tr]; c = {k: dict(bid=float(np.quantile(p, k / 100)), ask=float(np.quantile(-p, k / 100))) for k in KS}
    src = 'preds_c3' if d >= '20261101' else 'preds_o3'
    pr = pd.read_parquet(f'{io.OUT}/step5/{d}/{src}.parquet', columns=['mid', 'tau', 'pred_1000', 'pred_250'])
    for k in KS:
        o = pr.copy(); o['v_bid'] = (pr.pred_1000 - c[k]['bid']).astype(np.float32); o['v_ask'] = (-pr.pred_1000 - c[k]['ask']).astype(np.float32)
        o.to_parquet(f'{io.OUT}/step5/{d}/preds_b{k}.parquet', index=False)
    rep[d] = c; print(d, src, c, flush=True)
io.write_json(f"{io.OUT}/step5/baseguard_preds_report{'' if KS == (10, 20, 30) else '_' + '_'.join(map(str, KS))}.json", rep)
