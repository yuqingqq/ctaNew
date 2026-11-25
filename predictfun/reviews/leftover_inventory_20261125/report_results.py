"""Render the one-day decision from the verified receipts."""
import json
from pathlib import Path
import pandas as pd

H=Path(__file__).resolve().parent
def read(name):
    return json.loads((H/name).read_text())

def main():
    result=read('results_v2.json')
    trace=read('trace_check.json')
    assert trace['c_fills_and_orders_exact']
    market=pd.read_csv(H/'markets_v2.csv')
    prior=pd.read_csv(H/'markets.csv')
    comp=market.merge(prior,on=['mid','arm'],suffixes=('','_old'),validate='one_to_one')
    changed={a:int(((g.settle-g.settle_old).abs()>1e-6).sum()) for a,g in comp.groupby('arm')}
    sizes=market.groupby('arm').size()
    assert len(sizes)==3 and sizes.nunique()==1
    count=int(sizes.iloc[0])
    r=result['rows']; delta=result['contrasts']
    table=['| Policy | Completed round trips | Inventory at settlement | Settlement P&L | END mid P&L | Filled shares | Both sides live | Markets positive |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    names={'A':'Fixed inventory','B':'Fixed inventory + pressure','C':'Projected leftover + pressure'}
    for a in 'ABC':
        v=r[a]
        table.append(f"| {names[a]} | {v['turnover']:+,.2f} | {v['residual']:+,.2f} | {v['settle']:+,.2f} | {v['marked']:+,.2f} | {v['shares']:,.0f} | {v['both']:.1%} | {v['winners']} / {count} |")
    cov=result['capacity_at_C_submissions']
    clock=result['capacity_on_clock']['C']
    counts=read('turnover_check.json')['counts']
    n=counts['projected_can_clear_current_holding']
    z=result['capacity_on_C_clock_by_abs_z']
    m=market[market.mid==3789484].set_index('arm')
    trace_table=['| Policy | Settlement P&L | Completed turnover | Ending YES-equivalent shares |','|---|---:|---:|---:|']
    for a,v in m.iterrows():
        trace_table.append(f'| {a} | {v.settle:+,.2f} | {v.turnover:+,.2f} | {v.end_pos:+,.1f} |')
    outcome='passed' if result['passes'] else 'failed'
    text=f"""# One-day passive inventory test

The projected-leftover controller {outcome} the declared criterion on consumed Oct 26, 2026. The criterion required positive settlement P&L and an improvement over the fixed inventory controller with the same pressure guard. Market-level data are in [markets_v2.csv](markets_v2.csv). All dollar totals sum separate simulated markets.

{chr(10).join(table)}

All three policies waited for the same opening trigger, with a median first submission {r['C']['first_delay_median']:.1f} s after opening. Each used two passive levels per side, up to 220 shares per level, and a 1,000-share hard limit. No active takes. The passive simulator used 58 ms activation, 37 ms live cancel, 100 ms fill notices, current queue at activation and no queue credit for external cancellations. Settlement uses the recorded outcome. Completed turnover is FIFO. Book marking is at the replay endpoint, opening +298 seconds. Both-sided participation is the mean per-market grid fraction.

## Where the difference came from

- Pressure contributed {delta['B-A']['net']:+,.2f} USD, improving {delta['B-A']['better']} markets and worsening {delta['B-A']['worse']}. The fixed-rule inventory settings were identical for this comparison.
- Changing the inventory policy under the same pressure rule contributed {delta['C-B']['net']:+,.2f} USD, improving {delta['C-B']['better']} markets and worsening {delta['C-B']['worse']}. The completed-turnover contribution was {r['C']['turnover']-r['B']['turnover']:+,.2f} USD and the settlement residual contribution was {r['C']['residual']-r['B']['residual']:+,.2f} USD. C changes both new-quote sizing and the old soft-limit inventory pulls.
- Median peak exposure rose from {r['B']['median_peak']:.0f} to {r['C']['median_peak']:.0f} shares. Mean absolute ending inventory rose from {r['B']['mean_abs_end']:.0f} to {r['C']['mean_abs_end']:.0f}. The largest C inventory was {r['C']['max_peak']:.3f} shares. C submitted {r['C']['posts']:,.0f} orders, issued {r['C']['all_pulls']:,.0f} cancellation requests, of which {r['C']['pressure_pulls']:,.0f} were pressure pulls.

## Can we count on the exits?

At {cov['covered']:,} of {cov['n']:,} C submissions ({cov['covered']/cov['n']:.1%}), subsequent gross opposite-side passive fills met or exceeded the estimate. For strictly positive forecasts this was {cov['covered_positive']:,} / {cov['forecast_positive']:,}. The intended lower-quartile coverage was about 75%. Each submission is a repeated decision on a common market path. Using actual subsequent exit volume in the same all-reservations-fill calculation put {cov['all_fill_stress_over_50']:,} submissions ({cov['all_fill_stress_over_50']/cov['n']:.1%}) above $50. That stress calculation assumes every reserved share fills. It is separate from realized inventory P&L.

At the regularly spaced 5 s anchors, C's volume estimate was covered {clock['covered']/clock['n']:.1%} of the time. Coverage was {z['near']['covered']/z['near']['n']:.1%} near the BTC opening price (|z| < 0.5, n={z['near']['n']:,}) and {z['far']['covered']/z['far']['n']:.1%} farther away (|z| ≥ 1.5, n={z['far']['n']:,}). The latter descriptive split also mixes different remaining times. Normalized BTC drift did not set order sizes in this test.

A post-run inventory diagnostic distinguishes selling the original position from ending flat. There were {n:,} nonflat C clock states where projected exit volume covered the entire current holding. In {counts['gross_exits_enough']:,} of those, the market actually supplied that many opposite fills. In {counts['gross_exits_enough_and_same_direction_at_end']:,} of those latter cases, the strategy still finished with inventory on the same side after subsequent trading. These repeated anchors show how gross turnover and ending exposure can diverge. They do not identify the value of a different quoting policy.

## Charted market 3789484

{chr(10).join(trace_table)}

The first order was submitted {m.loc['C','first_s']:.1f} s after opening in all three arms. An independent notice-by-notice reconstruction reproduced each arm's summary and proposals, and exactly reproduced C's fill list and order lifecycle on this market.

## Scope and reproducibility

As of the Nov 25 run, the consumed Oct 26 registry contained 288 markets. The paired comparison includes all {count} with complete books and endpoint marks. The other 124 lacked the required endpoint mark and 94 had a stale future book. The completeness filter uses future tape availability. These are point estimates for the replayable subset. The simulator assumes unchanged external trades and continuous access to complementary token buys, a $1 minimum buy and no maker fees or rebates. There is no untouched-day or capital-adjusted return estimate.

The fit was pinned from 46 markets on Oct 24–25. An explanatory timestamp join and a score-module import were repaired. Independent enumeration then exposed a floating-point discrepancy at exactly 220 shares of recent flow. The corrected fit and complete replay supersede the first pass. Settlement changed by more than one millionth of a dollar in {changed['A']} A, {changed['B']} B and {changed['C']} C markets. The primary criteria and economic parameters were fixed before either run. [Protocol](protocol.md) · [Superseding fit](fit_v2.json) · [Results and checksums](results_v2.json) · [Trace control](trace_check.json) · [Gross versus net diagnostic](turnover_check.json)
"""
    (H/'report.md').write_text(text)

if __name__=='__main__':
    main()
