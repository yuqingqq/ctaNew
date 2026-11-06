"""Round O1 follow-up: where does the per-share edge of D1 + M05 come from as size grows (5 -> 25 -> 100 sh)?

Per size, over the 6 days (fills from the size armset, mid timing): shares and c/share at settlement and at +1 s,
split by fill kind (at our price / swept through), price region (YES <= 10 or >= 90 vs middle), phase (minute of the
window), and fill quantity relative to the order size (partial vs full). Descriptive.
Usage: python3 -m predictfun.research.pf_o1_size_decomp
"""
import os
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031']
D = f'{io.OUT}/step4/tokyo20_size_e2'
NAME = 'MICRO_G4_G2_G6_G5_G3_G1_M05__mid'
out = {}
for sz in (5, 25, 100):
    fs = []
    for d in DAYS:
        f = pd.read_parquet(f"{D}/{d}/fills/{NAME}{'' if sz == 5 else '_sz%d' % sz}.parquet"); f['day'] = d; fs.append(f)
    f = pd.concat(fs, ignore_index=True)
    f['region'] = np.where((f.pt <= 10) | (f.pt >= 90), 'edge <=10/>=90', 'middle 11-89')
    f['phase'] = (f.t // 60).clip(0, 4).astype(int)
    f['fill'] = np.where(f.q >= sz - 1e-9, 'full order', 'partial')
    def agg(g):
        return pd.Series(dict(shares=round(g.q.sum()), settle_c_sh=round(100 * g.settle.sum() / g.q.sum(), 3),
                              mo1_c_sh=round(100 * g.mo1.sum() / g.q.sum(), 3), usd=round(g.settle.sum(), 1)))
    res = {'all': agg(f).to_dict()}
    for k in ('through', 'region', 'phase', 'fill'):
        res[k] = f.groupby(k).apply(agg).to_dict('index')
    out[sz] = res
    print(f'\n=== size {sz} ===')
    print(' all', res['all'])
    for k in ('through', 'region', 'phase', 'fill'):
        for kk, v in res[k].items():
            print(f'  {k}={kk}: {v}')
io.write_json(f'{io.OUT}/step4/o1_size_decomp.json', {str(k): v for k, v in out.items()})
