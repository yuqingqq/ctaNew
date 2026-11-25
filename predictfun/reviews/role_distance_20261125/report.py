"""Independent recount of the three policies and the released-slot accounting."""
import hashlib
import json
import math
from pathlib import Path
import pandas as pd

H=Path(__file__).resolve().parent
d=json.loads((H/'result_v2.json').read_text())
def digest(name): return hashlib.sha256((H/name).read_bytes()).hexdigest()
assert digest('markets.csv')==d['market_hash'] and digest('replacement.csv')==d['replacement_hash']
f=pd.read_csv(H/'markets.csv')
rr=pd.read_csv(H/'replacement.csv')
def near(a,b): return math.isclose(a,b,abs_tol=1e-6)
def accounting(row):
    return all(near(a,b) for a,b in [(row['settle'],row['turnover']+row['residual']),
                 (row['turnover'],row['same_price_usd']+row['through_price_usd']),
                 (row['closed'],row['same_price_sh']+row['through_price_sh']),
                 (row['shares'],2*row['closed']+row['abs_end'])])
assert f.groupby('mid').arm.nunique().eq(3).all() and len(f)==210 and len(rr)==70
assert rr.mid.nunique()==70 and set(rr.mid)==set(f.mid)
assert all(accounting(r) for r in f.to_dict('records'))
example=f.iloc[0].to_dict()
assert not accounting(dict(example,residual=example['residual']+1))
assert (rr.unique_reposts+rr.ambiguous+rr.no_repost==rr.released).all()
assert (rr.closing_successors<=rr.accepted).all() and (rr.accepted<=rr.unique_reposts).all()
role=f.query("arm=='role'").set_index('mid').sort_index()
assert role.distance_only.eq(rr.set_index('mid').released.sort_index()).all()
arms=d['arms']
order=['reference','retain','role']
for arm,g in f.groupby('arm'):
    for k in ('settle','closed','turnover','residual','posts','all_pulls','same_price_sh','same_price_usd','through_price_sh','through_price_usd'):
        assert near(g[k].sum(),arms[arm][k])
    assert near(g.abs_end.mean(),arms[arm]['mean_abs_end'])
assert d['pass_all']==(role.settle.sum()>max(0.,arms['reference']['settle']) and
         role.turnover.sum()>max(0.,arms['reference']['turnover']) and role.abs_end.mean()<=d['ceiling'])
for k in d['trace']: assert near(d['trace'][k],rr[k].sum())

def usd(x): return ('−' if x<0 else '+')+f'${abs(x):,.2f}'
def table(name,k,fmt): return '| '+' | '.join([name]+[fmt(arms[a][k]) for a in order])+' |'
lines=['# Patient entries and responsive exits on the two-level ladder','',
       f"Frozen decision: **{'pass' if d['pass_all'] else 'fail'}** on 70 complete markets on consumed October 26. Both comparator ledgers were bound to the prior receipt for every market and reproduced in {len(d['parity'])} paired control markets (one per represented UTC opening hour). The other 218 of 288 scheduled markets had missing endpoint marks (124) or stale books (94).",
       '','| Metric | Original | Keep all distant quotes | Keep only distant adding quotes |','|---|---:|---:|---:|']
for name,k,fmt in [
 ('Settlement P&L','settle',usd),('Completed FIFO P&L','turnover',usd),('Residual settlement P&L','residual',usd),
 ('Completed cents per matched share','c_per_matched',lambda x:f'{x:+.3f}'),
 ('Matched shares','closed',lambda x:f'{x:,.0f}'),('Mean absolute ending shares','mean_abs_end',lambda x:f'{x:.1f}'),
 ('Within 10 shares of flat','within_ten',lambda x:f'{x}/70'),('Markets with positive settlement','positive_markets',lambda x:f'{x}/70'),
 ('Both sides active','both',lambda x:f'{100*x:.1f}%'),('Submissions','posts',lambda x:f'{x:,.0f}'),
 ('Cancel requests','all_pulls',lambda x:f'{x:,.0f}'),('Opening print at our quote, P&L','same_price_usd',usd),
 ('Opening print through our quote, P&L','through_price_usd',usd),('Same-price matched shares','same_price_sh',lambda x:f'{x:,.0f}'),
 ('Through-price matched shares','through_price_sh',lambda x:f'{x:,.0f}')]:
    lines.append(table(name,k,fmt))
lines+=['',f"Average absolute expiry exposure ceiling: {d['ceiling']:.2f} shares. Gate components: "+', '.join(k+'='+str(v) for k,v in d['criteria'].items())+'.','']
for a,c in d['paired'].items():
    lines.append(f"Against {a}, role-specific settlement changed by {usd(c['delta'])}, with {c['better']} markets better, {c['worse']} worse and {c['equal']} equal. With the largest gain excluded the change was {usd(c['excluding_largest_gain'])}.")
z=d['trace']
lines+=['','## What actually happened to released exit slots','',
f"There were {int(z['released']):,} distance-only exit requests, including {int(z['mixed']):,} mixed-purpose orders. {int(z['unique_reposts']):,} had a uniquely attributable later same-side submission, {int(z['ambiguous']):,} had competing vacancies, and {int(z['no_repost']):,} unambiguous vacancies had no successor by the end. Among the uniquely linked submissions, {int(z['closer']):,} offered a higher bid or lower ask and {int(z['accepted']):,} were accepted. {int(z['closing_successors']):,} successor orders executed an inventory-closing fill, closing {z['closed_sh']:,.1f} FIFO shares in total. This measures the successor order's role at execution; an intervening trade can change the inventory it actually closes.",
'','This test changes a distance-only cancellation. A closer replacement must still pass the same price, pressure, capacity, and minimum-size constraints, with retiring orders still reserved. Fill and exit changes are whole-policy feedback on a fixed external tape. The reference uses zero passive fees and rebates. The original JSON writer stopped after writing all market and replacement rows because it could not serialize a NumPy boolean. The superseding score uses the same frozen predicates on those rows with ordinary JSON booleans; both controls and the gate were checked independently. There are no new dates or validation intervals.','',
'[Frozen protocol](protocol.md) · [Market rows](markets.csv) · [Release and replacement counts](replacement.csv) · [Computed criteria and hashes](result_v2.json)']
(H/'report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:30]))
