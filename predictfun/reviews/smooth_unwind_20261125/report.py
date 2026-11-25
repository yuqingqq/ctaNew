"""Render the checked point estimates for the frozen arms."""
import json
from pathlib import Path

H=Path(__file__).resolve().parent
s=json.loads((H/'result.json').read_text())
fit=json.loads((H/'fit.json').read_text())
names={'stair':'Original staircase','stair_buffer':'Staircase, buffered','linear':'Straight line, buffered','adaptive':'Market-conditioned curve, buffered'}
r=s['arms']
lines=['# Smooth unwind and completed turnover', '',
       'Matched replay of 70 complete October 26 markets. The reference staircase was reproduced in every market. All arms use the same opening information, passive execution, queue, pressure signals, fees, $50 admission rule, 1,000-share hard cap and final 30-second reduce-only period. The live-order buffer lets an existing order keep its queue when exposure exceeds the time or state limit by at most 22 shares before reduce-only. New proposals always obey the stated limit.',
       '', '| Policy | FIFO-matched shares | Fills closing any shares | Completed P&L | ¢ per matched share | Residual P&L | Settlement P&L |',
       '|---|---:|---:|---:|---:|---:|---:|']
for a,name in names.items():
    v=r[a]
    lines.append(f"| {name} | {v['closed']:,.0f} | {v['close_events']:,.0f} | ${v['turnover']:+,.2f} | {v['cents_per_closed']:+.3f} | ${v['residual']:+,.2f} | ${v['settle']:+,.2f} |")
lines+=['','| Mean absolute inventory (shares) | 60 s left | 30 s left | 10 s left | Settlement | Markets within 10 shares of flat | Total cancellation requests |','|---|---:|---:|---:|---:|---:|---:|']
for a,name in names.items():
    v=r[a]
    lines.append(f"| {name} | {v['abs_60s']:.1f} | {v['abs_30s']:.1f} | {v['abs_10s']:.1f} | {v['abs_end']:.1f} | {v['within_ten']}/70 | {v['all_pulls']:,.0f} |")
lines+=['','## Paired contrasts','',
        'The declared gate requires more matched shares, higher completed P&L, higher settlement P&L, and mean absolute ending holdings no more than 10 shares above the comparator. Completed turnover uses FIFO. A fill that both closes inventory and opens a position in the opposite direction contributes only the closing portion to matched shares. Changes in FIFO pairing can move value between completed and residual P&L.']
for contrast,c in s['comparisons'].items():
    a,b=contrast.split('-')
    lines+=['',f"- {names[a]} versus {names[b]}: {r[a]['closed']-r[b]['closed']:+,.0f} matched shares, ${r[a]['turnover']-r[b]['turnover']:+,.2f} completed P&L, ${c['delta']:+,.2f} settlement, {r[a]['abs_end']-r[b]['abs_end']:+.1f} mean absolute ending shares. Markets better/worse/equal on settlement: {c['better']}/{c['worse']}/{c['equal']}. Joint gate: {'pass' if c['gate'] else 'fail'}."]
for a in ['linear','adaptive']:
    v,w=r[a],r['stair']
    lines+=['',f"Relative to the original unbuffered staircase, {names[a]} changed closed quantity by {v['closed']-w['closed']:+,.0f} shares, close events by {v['close_events']-w['close_events']:+,.0f}, settlement by ${v['settle']-w['settle']:+,.2f}, and absolute ending inventory by {v['abs_end']-w['abs_end']:+.1f} shares per market."]

lines+=['','## Market inputs and limitations','',
        f"The exit-rate fit used {len(fit['inputs'])} consumed Oct 24–25 markets. At each predetermined anchor it offered a fixed 440-share position for 15 seconds from a fresh queue on the reducing side only. Sell and buy rates were fitted separately. Of {fit['model']['0']['probes']} sell probes, {fit['model']['0']['zero']} closed zero shares; of {fit['model']['1']['probes']} buy probes, {fit['model']['1']['zero']} closed zero. The state inputs were causal BTC drift relative to remaining volatility, recent delivered reducing fills, and current pressure on the exit side. On the evaluation ramp the mean absolute deviation of the state multiplier from 1 was {r['adaptive']['mean_abs_multiplier_change']:.1%} over queried side–ticks. Only {round(fit['model']['0']['mean'][3]*fit['model']['0']['probes'])} sell probes and {round(fit['model']['1']['mean'][3]*fit['model']['1']['probes'])} buy probes began with a pressure veto on the exit side. Specific conclusions about behavior during that veto have thin support. Throughput adjusts the cap relative to the same side and time. It does not forecast per-fill profitability.",
        '', 'The fit was committed before Oct 26 scoring. An initial synthetic test caught the reference probes using a branch expecting a resting-review hook. The corrected probes and original B roll-in use their original branch; the synthetic checks passed before the fit. A separate inherited policy-selector collision stopped the first evaluation before any market was written. The superseding evaluator verifies every training definition against the committed fit source. Each matched evaluation used that same fit.',
        '', 'These are simulated outcomes on one previously used development day, as of Nov 25. The Oct 26 registry had 288 markets, including 124 missing the required endpoint and 94 with stale books. This is a conditional complete-path comparison. No pricing advantage, distribution of future days or live profitability follows from a passing implementation or a single-day gate.',
        '', '[Protocol](protocol.md) · [Market-level results](markets.csv) · [Checked totals](result.json) · [Earlier-day fit](fit.json)']
(H/'report.md').write_text('\n'.join(lines)+'\n')
