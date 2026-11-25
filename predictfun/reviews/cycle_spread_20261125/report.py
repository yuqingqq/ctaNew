"""Generate the descriptive receipt from reconciled FIFO allocations."""
import json
from pathlib import Path

import pandas as pd

from cycles import R, validate, Counter

here=Path(__file__).resolve().parent
r=json.loads((here/'result.json').read_text())
assert r['market_hash']==R.sha(here/'markets.csv')
groups={(x['lag_ms'],x['entry']):Counter(x) for x in r['rows']}
for x in groups.values():
    validate(x)
df=pd.read_csv(here/'markets.csv')
markets=df[(df.lag_ms==0)&(df.entry=='all')]
x=groups[0,'all']
def per(group,key):
    denominator={'early':'split_sh','late':'split_sh','net':'shares'}.get(key,'valued')
    return group[key]/group[denominator]
def row(name,key):
    return f"| {name} | {per(x,key):+.3f} | {x[key]/100:+,.2f} |"
lines=['# Spread earned on the retained passive strategy',
'',
'The branch is the buffered straight-line unwind with the existing pressure guard. This diagnostic replays the original fills on the 70 complete markets of consumed October 26. All markets reproduce the saved fill quantity, matched quantity, completed and residual P&L, settlement, ending position, posts and cancellations. The full registry contains 288 markets, including 124 missing the endpoint and 94 with stale books.',
'',
'| FIFO attribution | ¢ per matched share with the relevant marks | USD |',
'|---|---:|---:|',row('At opening execution versus the preceding mid','opening'),row('Mid movement while holding','drift'),
row('At closing execution versus the preceding mid','closing'),row('Realized completed turnover, all matched shares','net'),'',
f"Of {x['shares']:,.1f} matched shares, {x['valued']:,.1f} had valid marks at both legs. Residual realized turnover excluded from the three-component sum by missing marks was ${x['missing_legs_net']/100:+,.2f} over {x['missing_legs_sh']:,.1f} shares. The two fill edges sum to {per(x,'opening')+per(x,'closing'):+.3f}¢ on those marked shares, and net turnover on that cohort was {per(x,'valued_net'):+.3f}¢. At market level, the two fill edges summed positive in {int((markets.opening+markets.closing>0).sum())}/70 and realized completed turnover was negative in {int((markets.net<0).sum())}/70.",
'',
f"With an intermediate mark on {x['split_sh']:,.1f} shares, the first second after opening (or the entire holding period if shorter) contributed {per(x,'early'):+.3f}¢ per share. The time from that mark to the close contributed {per(x,'late'):+.3f}¢. On all shares, {100*x['good_sh']/x['shares']:.1f}% completed at a gain, {100*x['flat_sh']/x['shares']:.1f}% flat and {100*x['bad_sh']/x['shares']:.1f}% at a loss.",
'',
'| Opening public trade relative to our limit | Matched shares | Net ¢ per matched share | Net USD |',
'|---|---:|---:|---:|']
for kind in ('same_price','through_price'):
    y=groups[0,kind]
    lines.append(f"| {kind.replace('_',' ')} | {y['shares']:,.1f} | {per(y,'net'):+.3f} | {y['net']/100:+,.2f} |")
z=groups[50,'all']
lines.extend(['',f"With all midpoints sampled 50 ms earlier, the opening and closing execution edges combined are {per(z,'opening')+per(z,'closing'):+.3f}¢, holding movement is {per(z,'drift'):+.3f}¢, and both marks cover {z['valued']:,.1f} shares. Fill prices and all-share completed turnover are identical.",
'','## Interpretation and limits','',
'On a long cycle the signed components are opening midpoint minus buy price, closing midpoint minus opening midpoint, and sell price minus closing midpoint. Signs reverse for a short. FIFO shares are counted once per completed cycle, including partial fills and trades crossing flat. Each mark uses the last venue frame strictly before that time, maximum age one second, and the modeled trade clock. Execution edge is a measurement against that snapshot. A displayed midpoint is not a guaranteed price for unwinding. Only completed turnover is in this decomposition; terminal inventory is accounted for separately.',
'','The opening trade type is observed on execution. Grouping by it describes where money went on this replay and cannot identify a rule that knows whether a sweep is coming. The signed holding movement combines selection on opening and closing, exposure duration and subsequent market movement. This calculation alone does not identify which action would have improved P&L.',
'','[Per-market decomposition](markets.csv) · [Totals](result.json) · [Source and synthetic controls](cycles.py)'])
(here/'report.md').write_text('\n'.join(lines)+'\n')
