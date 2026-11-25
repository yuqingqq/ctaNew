"""Independent checks of the four-arm score and inventory history."""
import hashlib
import json
import math
from pathlib import Path
import pandas as pd

H=Path(__file__).resolve().parent
d=json.loads((H/'result.json').read_text())
def sha(name):return hashlib.sha256((H/name).read_bytes()).hexdigest()
assert sha('markets.csv')==d['markets_sha'] and sha('inventory.csv')==d['inventory_sha']
df=pd.read_csv(H/'markets.csv',float_precision='round_trip')
hist=pd.read_csv(H/'inventory.csv',float_precision='round_trip')
assert len(df)==280 and df.groupby('mid').arm.nunique().eq(4).all()
assert len(hist)==210 and hist.groupby('mid').arm.nunique().eq(3).all()
def near(a,b):return math.isclose(a,b,abs_tol=1e-6)
def check(r):
    return (near(r['settle'],r['residual']+r['turnover']) and near(r['closed'],r['through_price_sh']+r['same_price_sh'])
            and near(r['shares'],2*r['closed']+r['abs_end']) and near(r['turnover'],r['through_price_usd']+r['same_price_usd']))
assert all(check(r) for r in df.to_dict('records'))
row=df.iloc[0].to_dict()
assert not check(dict(row,settle=row['settle']+1))
for arm,g in df.groupby('arm'):
    s=d['groups'][arm]
    for key in ('turnover','settle','closed','residual','distance_only','all_pulls'):
        assert near(g[key].sum(),s[key])
    assert near(g.abs_end.mean(),s['mean_abs_end']) and near(g.abs_end.quantile(.9),s['p90_abs_end'])
for arm,g in hist.groupby('arm'):
    v=df[df.arm==arm].set_index('mid').abs_end.sort_index()
    assert (abs(g.set_index('mid').abs_2s.sort_index()-v)<1e-6).all() and (g.over_live_s<=g.over_s+1e-7).all()
    assert all(near(g[k].mean(),x) for k,x in d['mean_by_market'][arm].items())
group=d['groups']; orig,urg=group['reference'],group['urgency']
assert d['pass_all']==(urg['turnover']>max(0,orig['turnover']) and urg['settle']>max(0,orig['settle'])
                       and urg['mean_abs_end']<=orig['mean_abs_end']+10)
def usd(v):return ('−' if v<0 else '+')+f'${abs(v):,.2f}'
def table(label,key,fmt):return '| '+' | '.join([label]+[fmt(group[a][key]) for a in ('reference','retain','role','urgency')])+' |'
lines=['# Limit inventory by repricing exits when over budget','',
       f"Frozen decision: **{'pass' if d['pass_all'] else 'fail'}**. On October 26, the same 70 complete markets were eligible out of 288. Two previously tested controls reproduced their per-market metrics on all 70, and the original ladder comparison is hash-bound. The other 124 markets lacked a supported endpoint and 94 had stale books. The four policies share external flow, settlement outcomes, both passive levels, pressure protection and costs.",
       '','| Metric | Original | Keep all distant | Reprice every distant closing quote | Reprice only while over budget |',
       '|---|---:|---:|---:|---:|']
for label,k,fmt in [
    ('Settlement','settle',usd),('Completed FIFO','turnover',usd),('Residual inventory P&L','residual',usd),
    ('¢ per matched share','c_per_matched',lambda x:f'{x:+.3f}'),('Matched shares','closed',lambda x:f'{x:,.0f}'),
    ('Mean absolute ending shares','mean_abs_end',lambda x:f'{x:.1f}'),('90th percentile absolute ending shares','p90_abs_end',lambda x:f'{x:.1f}'),
    ('Largest absolute ending position','max_abs_end',lambda x:f'{x:.1f}'),('Within 10 shares of flat','within_ten',lambda x:f'{x}/70'),
    ('Positive markets','positive_markets',lambda x:f'{x}/70'),('All cancellation requests','all_pulls',lambda x:f'{x:,.0f}'),
    ('Distance-only requests','distance_only',lambda x:f'{x:,.0f}'),('Both sides live','both',lambda x:f'{100*x:.1f}%'),
    ('Same-price opening cycle P&L','same_price_usd',usd),('Through-price opening cycle P&L','through_price_usd',usd)]:
    lines.append(table(label,k,fmt))
lines+=['',f"The expiry exposure ceiling is {d['gate_shares']:.2f} mean absolute shares. "+', '.join(k+'='+str(v) for k,v in d['criteria'].items())+'.','']
for base,z in d['comparisons'].items():
    lines.append(f"Settlement versus {base}: {usd(z['delta'])}; {z['better']} improved, {z['worse']} worsened, {z['equal']} equal. Omitting the largest improvement: {usd(z['without_largest_gain'])}.")
lines+=['','## Inventory path','',
        '| Seconds to expiry | Target, shares | Keep all distant | Always responsive | Urgency only |',
        '|---:|---:|---:|---:|---:|']
for t in (150,120,90,60,30,10,2):
    values=[d['mean_by_market'][a][f'abs_{t}s'] for a in ('retain','role','urgency')]
    lines.append('| '+' | '.join([str(t),f'{1000*max(0,min(1,(t-30)/120)):.0f}']+[f'{v:.1f}' for v in values])+' |')
lines+=['','The target uses client-known net holdings at the decision. The path uses realized holdings, averaged in absolute shares across markets. A decreasing target does not guarantee a passive counterparty. Ending exposures are the last simulated position two seconds before settlement, valued at the recorded payout.',
        '','| Policy | Mean seconds over budget | Of those, exit actually live | Of those, exit side under pressure |',
        '|---|---:|---:|---:|']
for a in ('retain','role','urgency'):
    v=d['mean_by_market'][a]
    lines.append(f"| {a} | {v['over_s']:.1f} | {v['over_live_s']:.1f} | {v['over_pressure_s']:.1f} |")
lines+=['','These are paired feedback paths on one previously examined day. No additional days, active taking or parameter search were used. A positive development result warrants a frozen replication; it is not validation.','',
        '[Protocol](protocol.md) · [Market rows](markets.csv) · [Inventory traces](inventory.csv) · [Criteria and hashes](result.json)']
(H/'report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
