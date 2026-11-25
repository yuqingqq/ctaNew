"""Reconcile and render the descriptive timing audit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

H=Path(__file__).resolve().parent
r=json.loads((H/'result.json').read_text())
a=pd.read_csv(H/'requests.csv').fillna(0)
d=pd.read_csv(H/'cycles.csv')
ref=pd.read_csv(H.parent/'smooth_unwind_20261125/markets.csv').query("arm=='linear'").set_index('mid')
def valid(x):
    return np.allclose(x.requests,x.live_remainder+x.fully_filled_before_request+x.post_pending+x.get('venue_refused',0),rtol=0,atol=1e-7)
assert valid(a) and not valid(a.assign(requests=a.requests+1))
assert np.allclose(d.net_usd,d.gains_usd+d.losses_usd)
assert np.allclose(a.groupby('mid').requests.sum().sort_index(),ref.all_pulls.sort_index())
assert np.allclose(d.groupby('mid').net_usd.sum().sort_index(),ref.turnover.sort_index())
def count(v): return f'{v:,.0f}'
def money(v): return ('−' if v<0 else '+')+f'${abs(v):,.2f}'
lines=['# Where the retained ladder pulls, and when through-price openings match','',
       'All 70 complete October 26 markets reproduce the saved baseline order, cancellation, fill-volume, turnover and settlement totals. The FIFO opening-type subtotals match the earlier cycle audit in every market. The scope is one consumed day, 70 of 288 scheduled markets. No policy change or new signal was tested.','',
       '| First request reason | Orders | With live venue remainder | Before activation (accepted) | Already fully filled | Venue refused | Shares still filled after request |',
       '|---|---:|---:|---:|---:|---:|---:|']
for why,g in a.groupby('reason'):
    v=g.sum(numeric_only=True)
    lines.append('| '+' | '.join([why]+[count(v.get(k,0)) for k in ('requests','live_remainder','post_pending','fully_filled_before_request','venue_refused','post_request_shares')])+' |')
lines+=['','Structural reasons use the engine’s primary flag in precedence order (no same-side book, distance, crossing). The crossing test compares the order price with the received opposite-side best field. Overlapping flags are retained in requests.csv. Distance means more than 3 ticks plus the birth rung behind the same-side best. The request denominator counts a generation even if matching had already exhausted it and the final fill notice was still in flight. Such a request had no remaining queue position to remove.','',
'| Through-price FIFO opening timing | Matched shares | Gross gains on closes | Gross losses on closes | Net on closes |',
'|---|---:|---:|---:|---:|']
tt=d[d.kind=='through_price'].groupby('status').sum(numeric_only=True)
for st,v in tt.iterrows():
    lines.append(f'| {st} | {count(v.shares)} | {money(v.gains_usd)} | {money(v.losses_usd)} | {money(v.net_usd)} |')
lines+=['','The timings refer to the opening fill, and the values refer to its actual subsequent FIFO closes, weighted by matched shares. An in-flight request must precede the fill strictly; trades at a decision tick execute first. The live KEEP category requires a recorded eligible KEEP early enough for a 37 ms cancel to arrive before or at the match. A fill matching at or after an actual removal is refused by the audit. With 50 ms decisions and activation at +58 ms, a post-pending request at +50 ms could first take effect at +95 ms; a live decision at +100 ms could first take effect at +137 ms. These pending-only windows are mechanical possibilities. Realized sweep direction and later closing P&L are outcomes.','',
'The tables distinguish a feasible decision from a decision that would have earned money. Removing a resting order changes subsequent fills, closes and queue position. The missing-endpoint and stale-book markets are outside this comparison.','',
'[Market request counts](requests.csv) · [FIFO timing groups](cycles.csv) · [Computed totals](result.json) · [Frozen audit](audit.py)']
(H/'report.md').write_text('\n'.join(lines)+'\n')
print((H/'report.md').read_text())
