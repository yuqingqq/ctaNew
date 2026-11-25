"""Descriptive halfway census, Oct 24–27. Freeze before reading.

Use 150s record time, before [0,150) and after [150,300). Passive shares
measure quoting activity through executions. Both roles count for the
zero-start YES-equivalent balance. Distinguish getting back to flat from
being small throughout: prior peak >50, final abs <=10. Sensitivities
10/50/100. Condition only on a passive fill before halfway for comparisons.
Classify the public book at second 149 as extreme (mid <=10 or >=90),
interior, one-sided or unavailable. Frequencies are descriptive.
"""
import json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=ROOT/'data/predict_fun/research'
DASH=HERE.parent/'maker_dashboard_20261124'
selected={3702960,3789484,3877472}|set(map(int,json.loads((DASH/'markets/selection.json').read_text())))
legs=pd.read_parquet(DATA/'maker3_late_activity_20261124/actor_legs.parquet')
legs=legs.loc[legs.t_rel_s.between(0,300,inclusive='left')]
reference=pd.read_csv(HERE.parent/'maker3_late_activity_20261124/market_activity.csv').set_index('mid')
assert len(reference)==934 and (legs.groupby('mid').q.sum()-reference.shares).abs().max()<1e-6
grid=pd.read_parquet(DATA/'maker3_book_conditions_20261125/grid.parquet',filters=[('k','=',149)])
assert grid.mid.is_unique
grid=grid.set_index('mid')
rows=[]
for mid, group in legs.groupby('mid'):
    early=group[group.t_rel_s<150]
    late=group[group.t_rel_s>=150]
    path=early.groupby('settle_ms').dq.sum().cumsum()
    pos=early.loc[early.y_side=='bid','q'].sum()-early.loc[early.y_side=='ask','q'].sum()
    assert abs(pos-early.dq.sum())<1e-7
    book=grid.loc[mid]
    price=(book.bid_c+book.ask_c)/2
    state=('extreme' if price<=10 or price>=90 else 'interior') if book.st=='ok' else book.st
    first=early.loc[early.role=='maker','q'].sum()
    second=late.loc[late.role=='maker','q'].sum()
    assert abs(first+second-group.loc[group.role=='maker','q'].sum())<1e-7
    rows.append(dict(mid=mid,day=group.day.iloc[0],first=first,second=second,pos=pos,peak=path.abs().max() if len(path) else 0,
                     last=group.loc[group.role=='maker','t_rel_s'].max(),late_take=late.role.eq('taker').any(),book=state))
all_rows=pd.DataFrame(rows)
all_rows.to_csv(HERE/'market_rows.csv',index=False)
def counts(g):
    return dict(markets=len(g),no_later_passive=int(g.second.eq(0).sum()),any_later_passive=int(g.second.gt(0).sum()),
                no_later_trade=int((g.second.eq(0)&~g.late_take).sum()))
for name,g in [('dashboard',all_rows[all_rows.mid.isin(selected)]),('all',all_rows)]:
    early=g[g['first']>0]
    print(name,'active',len(g),'early',counts(early),'both shares',g[['first','second']].sum().tolist(),'first-majority markets',int((g['first']>g.second).sum()))
    for limit in [10,50,100]:
        flat=early.pos.abs()<=limit
        print(limit,'flat',counts(early[flat]),'other',counts(early[~flat]))
    print('returned',counts(early[(early.peak>50)&(early.pos.abs()<=10)]))
    for book,z in early.groupby('book'):
        print(book,counts(z))
    if name=='dashboard':print(g.to_string(index=False))
