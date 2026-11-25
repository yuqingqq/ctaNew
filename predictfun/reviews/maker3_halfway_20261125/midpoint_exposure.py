"""Descriptive Oct 24–27. Anchor 150s. First-half passive activity required.
Sort on absolute Binance mid distance from its opening mid (<2, 2–5, >=5 bp).
Forward book range requires 61 valid two-sided seconds (also show coverage).
Zero-start YES-equivalent inventory at 150s, then 210s using both roles.
Among |position|>10 compare shrinkage and taker fills opposing the 150s sign.
Match near and far within day and starting magnitude (10–50,50–100,100–250,
250+) with harmonic overlap weights. Point estimates. No policy test.
"""
from pathlib import Path
import numpy as np
import pandas as pd

here = Path(__file__).resolve().parent
data = here.parents[2] / 'data/predict_fun/research'
base = pd.read_csv(here/'market_rows.csv').query('first > 0').copy()
registry = pd.read_csv(here.parent/'maker3_market_coverage_20261124/markets.csv')
base = base.merge(registry[['mid','ws']],validate='one_to_one',on='mid')
for day,g in base.groupby('day'):
    tape = pd.read_parquet(data/f'maker_trace_v2_20261123/btc/btc_{day}.parquet')
    px = tape.set_index('g_ms').mid.where(tape.set_index('g_ms').status.eq('ok'))
    t = g.ws.to_numpy()*1000
    base.loc[g.index,'dist'] = abs(10000*(px.reindex(t+150000).to_numpy()/px.reindex(t).to_numpy()-1))
base['band'] = pd.cut(base.dist,[0,2,5,np.inf],right=False,labels=['near','middle','far'])
book = pd.read_parquet(data/'maker3_book_conditions_20261125/grid.parquet',filters=[('k','>=',150),('k','<=',210)])
assert not book.duplicated(['mid','k']).any()
book['price'] = ((book.bid_c+book.ask_c)/2).where(book.st.eq('ok'))
b = book.groupby('mid').price.agg(['max','min','count'])
base['yes_range'] = base.mid.map((b['max']-b['min']).where(b['count']==61))
legs = pd.read_parquet(data/'maker3_late_activity_20261124/actor_legs.parquet')
def position(t):
    return legs[legs.t_rel_s.between(0,t,inclusive='left')].groupby('mid').dq.sum()
anchor = base.mid.map(position(150)).fillna(0)
assert np.allclose(anchor,base.pos)
base['open_size'] = anchor.abs()
base['next_size'] = base.mid.map(position(210)).fillna(0).abs()
base['smaller'] = base.next_size < base.open_size-1e-6
next_legs = legs[legs.t_rel_s.between(150,210,inclusive='left')]
takes = next_legs[next_legs.role.eq('taker')].copy()
takes['against'] = takes.dq*takes.mid.map(base.set_index('mid').pos)<0
base['opposing_take'] = base.mid.isin(takes.loc[takes.against,'mid'])
base['size_bin'] = pd.cut(base.open_size,[10,50,100,250,np.inf])
print('early passive',len(base),'BTC missing',base.band.isna().sum())
for band,g in base.groupby('band',observed=True):
    exposed = g[g.open_size>10]
    print(band,'n',len(g),'book n/range',g.yes_range.count(),g.yes_range.median(),
          'abs pos',g.open_size.median(),'exposed',len(exposed),'shrunk',exposed.smaller.sum(),
          'against take',exposed.opposing_take.sum())
pairs = []
for _,g in base.groupby(['day','size_bin'],observed=True):
    a,b = g[g.band=='near'],g[g.band=='far']
    if len(a) and len(b):
        pairs.append([len(a),len(b),len(a)*len(b)/(len(a)+len(b)),a.smaller.mean(),b.smaller.mean(),
                      a.opposing_take.mean(),b.opposing_take.mean()])
p = np.array(pairs)
print('overlap strata / counts',len(p),p[:,:2].sum(axis=0),'matched shrink near/far and take near/far',np.average(p[:,3:],axis=0,weights=p[:,2]))
