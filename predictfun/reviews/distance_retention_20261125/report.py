"""Recheck the written market rows and render the paired development comparison."""
import json
import math
from pathlib import Path

import pandas as pd

from hashlib import sha256

def validate(row):
    assert math.isclose(row.settle,row.residual+row.turnover,abs_tol=1e-6)
    assert math.isclose(row.turnover,row.same_price_usd+row.through_price_usd,abs_tol=1e-6)
    assert math.isclose(row.closed,row.same_price_sh+row.through_price_sh,abs_tol=1e-6)
    assert math.isclose(row.shares,2*row.closed+row.abs_end,abs_tol=1e-6)

H=Path(__file__).resolve().parent
d=json.loads((H/'result.json').read_text())
f=pd.read_csv(H/'markets.csv')
assert sha256((H/'markets.csv').read_bytes()).hexdigest()==d['market_hash'] and len(f)==2*d['market_count']==140
for _,row in f.iterrows(): validate(row)
p=f.pivot(index='mid',columns='arm',values='settle')
delta=p.retain-p.reference
assert len(delta)==70 and math.isclose(delta.sum(),d['delta'],abs_tol=1e-6)
for arm,g in f.groupby('arm'):
    for key in ('posts','all_pulls','closed','turnover','residual','settle','through_price_usd','same_price_usd'):
        assert math.isclose(d['arms'][arm][key],g[key].sum(),abs_tol=1e-6)
a,b=d['arms']['reference'],d['arms']['retain']
def usd(n): return ('−' if n<0 else '+')+f'${abs(n):,.2f}'
def line(name,fmt,va,vb):
    return '| '+' | '.join([name,fmt(va),fmt(vb),fmt(vb-va)])+' |'
lines=['# Keep a quote when it falls behind the best price','',
       f"Frozen development reading: **{d['reading']}**. The two arms cover identical tapes and settlement outcomes for 70 complete October 26 markets. The registry also contains 124 markets with missing endpoints and 94 with stale books. The 288-market date is consumed. Only the distance-alone cancellation was disabled; all other ways of cancelling and all risk and order limits operated as before.",'',
       '| Metric | Current ladder | Keep distant quotes | Change |','|---|---:|---:|---:|']
for label,k,fmt in [
    ('Settlement','settle',usd),('Completed FIFO turnover','turnover',usd),('Residual settlement P&L','residual',usd),
    ('Completed cents per matched share','c_per_matched',lambda x:f'{x:+.3f}'),('Matched shares','closed',lambda x:f'{x:,.0f}'),
    ('Mean absolute ending inventory (shares)','mean_abs_end',lambda x:f'{x:.1f}'),
    ('Submissions','posts',lambda x:f'{x:,.0f}'),('Cancellation requests','all_pulls',lambda x:f'{x:,.0f}'),
    ('Both sides active (fraction)','both',lambda x:f'{x:.3f}'),
    ('Same-price opening cycles','same_price_usd',usd),('Through-price opening cycles','through_price_usd',usd),
    ('Same-price opening matched shares','same_price_sh',lambda x:f'{x:,.1f}'),('Through-price opening matched shares','through_price_sh',lambda x:f'{x:,.1f}')]:
    lines.append(line(label,fmt,a[k],b[k]))
for name,k in [('Same-price opening','same_price'),('Through-price opening','through_price')]:
    lines.append(line(name+' cents per matched share',lambda x:f'{x:+.3f}',100*a[k+'_usd']/a[k+'_sh'],100*b[k+'_usd']/b[k+'_sh']))
flat=f.assign(flat=f.abs_end<=10).groupby('arm').flat.sum().to_dict()
lines+=['',f"Positive settlement markets: {(p.reference>0).sum()}/70 before and {(p.retain>0).sum()}/70 after. Within 10 shares of flat: {flat['reference']}/70 before and {flat['retain']}/70 after.", '',f"Pulls per full market-minute went from {a['all_pulls']/d['complete_minutes']:.1f} to {b['all_pulls']/d['complete_minutes']:.1f}. Baseline requests solely for distance were {a['distance_only']:,.0f}; the alternative had {b['distance_only']:,.0f}, and {b['unique_retained']:,.0f} distinct generations reached the exemption at least once. A kept order could still receive a same-tick pressure or inventory request. Pressure pulls went from {a['pressure_pulls']:,.0f} to {b['pressure_pulls']:,.0f}.",
       '',f"Paired settlement improved in {d['better']} markets, worsened in {d['worse']} and tied in {d['equal']}. Excluding the single largest improvement gives {usd(d['delta_without_biggest_gain'])}. The opening trade class is observed when a fill happens. The class-level differences include changes to future orders, inventory and FIFO closes. They do not measure the causal value of capturing a specific queue position.",
       '','The disabled arm reconciled against the previous per-market settlement, positions, fills, turnover, participation and pull counts, plus both opening classes in the previous cycle attribution. The per-tick guard and accounting tests include a falsifier. The gate and candidate were committed before the replay. No other settings were varied, no untouched dates were used and no rebate was credited. External order flow remains fixed.','',
       '[Predeclared test](protocol.md) · [Market rows](markets.csv) · [Computed decision](result.json)']
(H/'report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:24]))
