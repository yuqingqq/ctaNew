"""Render the paired five-minute outcome and explain the exposure diagnostic."""
import json
from pathlib import Path
import pandas as pd

here=Path(__file__).resolve().parent
r=json.loads((here/'results.json').read_text())
f=pd.read_csv(here/'paired_markets.csv',float_precision='round_trip')
days=sorted(str(i) for i in f.day.unique())
p=r['pooled']
c,b,n=(p[a] for a in ['current','binance','confirmed'])
d=p['diagnostics']
out=['# Does Predict.fun confirmation improve Binance-triggered maker cancels?',
     '',f'The confirmed rule {"passed" if r["passed"] else "failed"} the frozen development screen. As of November 28, 2026 UTC, the replay covered {len(f)} eligible five-minute markets of 864 scheduled on three consumed dates. The common census excluded 465 missing endpoints and 181 stale books. All P&L is simulated USD at settlement under zero maker fees and rebates.',
     '', '| UTC | Markets | Current pressure guard | Binance-only overlay | Order-level confirmation | Change vs current |',
     '|---|---:|---:|---:|---:|---:|']
for name in days+['pooled']:
    g=r[name]
    cur,broad,sel=[g[x]['total'] for x in ('current','binance','confirmed')]
    out.append(f'| {name} | {g["current"]["n"]} | {cur:+,.2f} | {broad:+,.2f} | {sel:+,.2f} | {sel-cur:+,.2f} |')
out+=['','| Measure | Current | Binance only | Confirmed |','|---|---:|---:|---:|']
for key,label,fmt in [('sharpe','Five-minute market Sharpe', '.4f'),('worst_decile','Mean P&L in worst decile of markets, USD',',.2f'),('turnover','Completed FIFO closes, USD',',.2f'),('residual','Residual inventory at payout, USD',',.2f'),('requests','Cancel requests',',.0f'),('reposts','Reposts',',.0f'),('shares','Filled shares',',.0f'),('mark_1s','One-second markout, cents per valued share','.3f')]:
    out.append('| '+' | '.join([label]+[format(a[key],fmt) for a in [c,b,n]])+' |')
out+=['',f'Settlement changed by {n["total"]-c["total"]:+,.2f} USD versus the current policy. Completed closes contributed {n["turnover"]-c["turnover"]:+,.2f} USD and residual inventory contributed {n["residual"]-c["residual"]:+,.2f} USD. The confirmed policy beat the current policy in {d["better"]} markets, lost in {d["worse"]}, and tied in {len(f)-d["better"]-d["worse"]}. Sharpe uses mean P&L divided by its sample standard deviation across five-minute markets, unannualized. The three dates do not support a day-cluster interval.',
      '',f'The extra rule sent {d["offgrid_pulls"]:,} off-grid and {d["grid_pulls"]:,} grid pulls. At {d["offgrid_one_of_two"]:,} off-grid decisions it pulled one of two simultaneously live orders on the same side. It withheld {d["waits"]:,} proposed slots. These action counts are on the candidate path. Total cancellation differences also reflect changed fills, inventory and future orders.',
      '', '| Incumbent exposure | Shares | Shares without +1 s mark | Gross adverse USD | Gross favorable USD | Net cents per valued share |','|---|---:|---:|---:|---:|---:|']
for key in ['timely','remaining']:
    x=d[key]
    net='—' if x['net_c_per_share'] is None else f'{x["net_c_per_share"]:+.3f}'
    out.append(f'| {key} | {x["shares"]:,.1f} | {x["censored"]:,.1f} | {x["adverse"]:+,.2f} | {x["favorable"]:+,.2f} | {net} |')
out+=['','Timely exposure identifies incumbent fills when the combined conditions preceded matching by at least the modeled 37 ms cancel delay, before an incumbent request. Its value describes selection on that original path. The independently replayed policy includes all downstream fills and inventory.',
      '', 'The Binance alert uses its previous score and modeled 4 ms delivery. Each local public settlementsPending increase is linked to an original book frame and read on the normalized Predict.fun receipt clock. Side-specific net depletion and current displayed depth are required at the same quote price. A settlement increase can be reported on either side. The thin-depth condition is a proxy for queue ahead and can miss an order at the front of a deep level. The original 50 ms posting grid and 58 ms placement delay still apply. All three policies assume unchanged external flow.',
      '', 'Each incumbent market matched the archived tape identity and nine archived order or settlement metrics. Fill quantities and settlement reconciled on each paired path. The evaluator rejected a corrupted adverse-markout subtotal on each day. A separate scalar fixture reproduced the local trigger and detected substitution of venue time for receipt time. The first launch failed during module import, before processing data. A loader-only amendment preceded the completed run.',
      '', 'The frozen checks evaluate '+', '.join(k.replace('_',' ')+' '+('PASS' if v else 'FAIL') for k,v in r['gates'].items())+'.',
      '', '[Protocol](protocol.md) · [Paired market data](paired_markets.csv) · [Computed results](results.json)']
(here/'report.md').write_text('\n'.join(out)+'\n')
