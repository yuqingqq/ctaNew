"""Consumed Oct 24–27. At 60/150/240 s compare abs BTC change from open
(<2, 2–5, >=5 bp). Require earlier maker fill and abs inventory >10.
First entry into [-10,10] within the next 60s, on net same-stamp fills.
Measure any strictly later passive fill until market end; same stamp ambiguous.
For near/far standardize hit probability and time (misses capped at 60s) by
day x prior abs position using the midpoint comparison's strata. Descriptive.
"""
from pathlib import Path

import numpy as np
import pandas as pd

here = Path(__file__).resolve().parent
data = here.parents[2] / 'data/predict_fun/research'
registry = pd.read_csv(here.parent/'maker3_market_coverage_20261124/markets.csv').set_index('mid')
legs = pd.read_parquet(data/'maker3_late_activity_20261124/actor_legs.parquet')
legs = legs[legs.t_rel_s.between(0,300,inclusive='left')]
reference = pd.read_csv(here/'market_rows.csv').set_index('mid')
rows = []
anchors = [60,150,240]

def first_hit(t, pos, anchor):
    idx = np.flatnonzero((t>=anchor)&(t<anchor+60)&(abs(pos)<=10+1e-8))
    return t[idx[0]] if idx.size else np.nan

assert first_hit(np.array([149,155,210]),np.array([0,9,0]),150)==155
assert np.isnan(first_hit(np.array([149,155,210]),np.array([0,-12,0]),150))

for day,day_legs in legs.groupby('day'):
    tape = pd.read_parquet(data/f'maker_trace_v2_20261123/btc/btc_{day}.parquet',columns=['mid','g_ms','status']).set_index('g_ms')
    prices = tape.mid.where(tape.status.eq('ok'))
    for mid,g in day_legs.groupby('mid'):
        start_ms = int(registry.loc[mid,'ws'])*1000
        stamp = g.groupby('settle_ms',sort=True).dq.sum()
        t = (stamp.index.to_numpy()-start_ms)/1000
        assert np.allclose(g.t_rel_s,(g.settle_ms-start_ms)/1000)
        p = stamp.cumsum().to_numpy()
        passive = g.loc[g.role.eq('maker'),'t_rel_s'].to_numpy()
        for anchor in anchors:
            i = t.searchsorted(anchor)
            pos = p[i-1] if i else 0
            if anchor==150:
                assert abs(pos-reference.loc[mid,'pos'])<1e-6
            if not (passive<anchor).any() or abs(pos)<=10+1e-8:
                continue
            gap = abs(1e4*(prices.at[start_ms+1000*anchor]/prices.at[start_ms]-1))
            hit = first_hit(t,p,anchor)
            rows.append(dict(day=day,mid=mid,anchor=anchor,gap=gap,starting=abs(pos),
                             hit=np.isfinite(hit),wait=hit-anchor if np.isfinite(hit) else 60,
                             later=np.isfinite(hit) and (passive>hit).any()))
r = pd.DataFrame(rows)
r['band'] = pd.cut(r.gap,[0,2,5,np.inf],right=False,labels=['near','middle','far'])
r['size'] = pd.cut(r.starting,[10,50,100,250,np.inf])
for anchor,g in r.groupby('anchor'):
    print('anchor',anchor,'eligible',len(g),'BTC unavailable',g.band.isna().sum())
    print(g.groupby('band',observed=True).agg(markets=('mid','size'),hit_in_60=('hit','sum'),later_passive_after_hit=('later','sum')).to_string())
    comps=[]
    for _,s in g.groupby(['day','size'],observed=True):
        a,b = s[s.band=='near'],s[s.band=='far']
        if len(a) and len(b):
            comps.append([len(a)*len(b)/(len(a)+len(b)),len(a),len(b),a.hit.mean(),b.hit.mean(),a.wait.mean(),b.wait.mean()])
    v=np.array(comps)
    print('matched near/far counts',v[:,1:3].sum(axis=0),'hit probabilities and capped mean time (s)',np.average(v[:,3:],axis=0,weights=v[:,0]))
