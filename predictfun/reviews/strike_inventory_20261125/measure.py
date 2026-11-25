from collections import Counter
import json
from pathlib import Path
import resource
import sys

import numpy as np
import pandas as pd

from risk import selftest
H = Path(__file__).resolve().parent
ROOT = H.parents[2]
DATA = ROOT/'data/predict_fun/research'
V = DATA/'maker_trace_v2_20261123'
sys.path.insert(0,str(ROOT/'predictfun/research'))
import pf_mt_decision as D
import pf_mt_econ as ECON
DAYS = ['20261024','20261025','20261026','20261027']
ANCHORS = [30,60,120,150,180,240,270,295]

def features(bt, t, ws):
    x = t - 1000
    f = D.btc_feats(bt, x, ws, ws+300000)
    f['pf'] = np.where(x>=ws, f['pf'], np.nan)
    return f

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(30,31))
    selftest()
    grid = pd.read_parquet(DATA/'maker3_book_conditions_20261125/grid.parquet',
                           filters=[('k','in',[s-1 for s in ANCHORS])])
    assert not grid.duplicated(['mid','k']).any()
    panels = []
    for day in DAYS:
        g = grid[grid.day.astype(str)==day].copy()
        cens = g.loc[g.st=='boundary_censored','mid'].unique()
        reg = ECON.registry(DATA/'step1'/day/'registry.parquet',None,censored=cens)
        g = g.merge(reg[['mid','ws_ms','V']],on='mid',validate='many_to_one')
        assert len(g)==288*len(ANCHORS) and (g.g_ms==g.ws_ms+1000*g.k).all()
        bt = D.load_btc(V/'btc',day,Counter())
        f = features(bt,g.g_ms.to_numpy()+1000,g.ws_ms.to_numpy())
        g['z'],g['fair'] = f['z'],f['pf']
        g['b'] = (g.bid_c+g.ask_c)/200
        g['status'] = np.select([g.V.isna(),g.st!='ok',~np.isfinite(g.fair)],
                                ['no_outcome','no_book','no_btc'],default='ok')
        panels.append(g)
    p = pd.concat(panels)
    print(p.groupby(['day','status']).size().to_string())
    p = p[p.status=='ok'].copy()
    for arm,q in [('book',p.b),('blend',(p.b+p.fair)/2)]:
        p['brier_'+arm] = (q-p.V)**2
        p['score_'+arm] = (((p.b**2*(1-q)-(p.b-p.V).clip(lower=0)**2)**2
                            +((1-p.b)**2*q-(p.V-p.b).clip(lower=0)**2)**2)/2)
    cols=['brier_book','brier_blend','score_book','score_blend']
    table = p.groupby('day')[cols].mean()
    print(table.to_string())
    p['long_c'] = 100*(p.b-p.V).clip(lower=0)
    p['short_c'] = 100*(p.V-p.b).clip(lower=0)
    m=p[p.k==149].copy()
    m['distance']=np.select([m.z.abs()<=.5,m.z.abs()>=1.5],['near','far'],default='middle')
    print(m.groupby('distance').agg(n=('mid','size'),long_c=('long_c','mean'),short_c=('short_c','mean')).to_string())
    test=table.loc[table.index.astype(str).isin(DAYS[2:])]
    assert len(test)==2
    passed=bool((test.score_blend<test.score_book).all())
    print('forecast gate',passed)
    (H/'forecast.json').write_text(json.dumps(dict(passed=passed,days=table.to_dict('index')),indent=2)+'\n')

if __name__=='__main__':
    main()
