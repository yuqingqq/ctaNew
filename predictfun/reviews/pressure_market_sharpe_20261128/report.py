from pathlib import Path
import json

import pandas as pd
from diagnose import HERE, DAYS, BLOCKS
from score import OUT

def load(name):
    return json.loads((HERE/name).read_text())

def cash(x):
    return f'{x:+,.2f}'

def build():
    r=load('result.json')
    b=load('baseline.json')['all_eight']
    history=load('historical_sharpe.json')
    g=pd.read_csv(HERE/'fifo_attribution.csv')
    census=pd.concat([pd.read_csv(HERE.parent/BLOCKS[d>='20261124']/f'census_{d}.csv') for d in DAYS])
    counts=census.status.value_counts()
    text=['# Five-minute market Sharpe and the cost of entering a tight book','',
          'November 28, 2026 UTC. The unit is the five-minute market. Each eligible market has equal weight, including any that settle at zero. Dollar P&L at constant allocated capital gives the same Sharpe as returns on that capital. Values are unannualized, using a zero benchmark and sample standard deviation.',
          '',f'**Decision.** The single candidate {"passed" if r["advance"] else "failed"} the prespecified development screen.',
          '','# Population and strategy','',
          f'The optimization compares {counts["complete"]} paired markets out of {len(census):,} scheduled on November 14–18 and 24–26. Status counts were '+', '.join(f'{n:,} {k}' for k,n in counts.items())+'. Missing markets have unknown P&L.',
          '', 'Both strategies use the existing two-level passive policy, up to 50 shares per quote, its 150-share inventory cap with expiry taper, the same fixed historical order flow, modeled queue and event-driven adverse-pressure cancellations. The historical clock assumes 4 ms for the Binance feed, 58 ms for placement, 37 ms for live cancellation and a 50 ms posting grid. Reported USD assume zero passive fees, rebates and funding.',
          '', 'At a one-tick received spread, the candidate posts at the touch only up to known reducing inventory left after same-side reservations. It can still add exposure at the second level. A resting order runs its original management logic. The rule was frozen before replay. Both recent date blocks had already been examined before this research choice.',
          '','# Whole-market results','',
          '| Date or block | Markets | Current USD | Candidate USD | Current 5m Sharpe | Candidate 5m Sharpe | Current / candidate worst-decile mean USD |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for name,v in list(r['by_day'].items())+list(r['stages'].items()):
        c,n=v['current'],v['candidate']
        text.append(f'| {name} | {c["n"]} | {cash(c["total"])} | {cash(n["total"])} | {c["sharpe"]:.4f} | {n["sharpe"]:.4f} | {cash(c["worst_decile"])} / {cash(n["worst_decile"])} |')
    pool=r['stages']['pooled']
    current,new=pool['current'],pool['candidate']
    lo,hi=r['sharpe_delta_day_bootstrap_95']
    text += ['',f'The candidate improved {pool["better_markets"]} markets and worsened {pool["worse_markets"]}. Excluding the best improvement date ({r["most_improved_day"]}), the P&L difference was {cash(r["delta_ex_best"])} USD. The descriptive 95% paired day-bootstrap range for the market Sharpe difference was {lo:+.4f} to {hi:+.4f} using 4,096 draws. It has only eight day clusters. The unit for the point estimate remains the individual market.',
             '',f'The gate outcomes were '+', '.join(f'{k} {v}' for k,v in r['gates'].items())+'.',
             '',f'Across all 22 previously tested dates the current strategy had market Sharpe {history["all_22"]["event"]["sharpe"]:.4f} and USD {cash(history["all_22"]["event"]["total"])} over 1,591 eligible markets. Those dates include strategy development. The 50 ms grid comparator had {history["all_22"]["grid"]["sharpe"]:.4f}. On the recent 550 markets the grid comparator had {history["recent_8"]["grid"]["sharpe"]:.4f}.',
             '','# Loss attribution','',
             f'Across the {b["losing"]["n"]} losing markets the current strategy settled {cash(b["losing"]["settle"])} USD. Completed FIFO buy/sell pairs accounted for {cash(b["losing"]["turnover"])} USD and remaining inventory {cash(b["losing"]["residual"])} USD. {b["losing"]["negative_round_trip_net"]} of these markets already had negative completed turnover. Across all markets, profitable completed shares gained {100*b["all"]["gross_gain"]/b["all"]["win_sh"]:.2f} cents per share and losing completed shares lost {-100*b["all"]["gross_loss"]/b["all"]["loss_sh"]:.2f} cents per share.',
             '', '| Strategy and opening cohort | Matched shares | Missing-mid shares | Entry edge USD | Mid move during holding USD | Exit edge USD | Missing-mid net USD | Completed USD | Mean holding s |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ('current','candidate'):
        p=g[g.arm==arm]
        for name,rows in [('all',p),('losing markets',p[p.losing_market]),('losing FIFO pairs',p[p.sign=='loss']),
                          ('won FIFO pairs',p[p.sign=='gain']),('one-tick touch',p[p.birth=='one_tick_touch']),
                          ('other opening quotes',p[p.birth=='other'])]:
            z=rows.sum(numeric_only=True)
            assert abs(z.pnl-sum(z.get(k,0.) for k in ('open_edge','hold_move','close_edge','missing_pnl')))<1e-5
            duration=f'{z.share_seconds/z.shares:.1f}' if z.shares>0 else '—'
            text.append(f'| {arm} / {name} | {z.shares:,.1f} | {z.get("missing_shares",0.):,.1f} | {cash(z.open_edge)} | {cash(z.hold_move)} | {cash(z.close_edge)} | {cash(z.get("missing_pnl",0.))} | {cash(z.pnl)} | {duration} |')
    text += ['', 'Edge uses the most recent valid received book no older than one second at each fill. Signed entry edge plus midpoint movement plus exit edge equals the realized closed profit. Cohorts are descriptive and overlap across different breakdowns. The one-tick group is defined at the opening fill’s original quote proposal. Some orders can change their inventory role before execution.',
             '', '| Pooled measure | Current | Candidate |','|---|---:|---:|']
    for k,label in [('mean','Mean USD per market'),('sd','SD USD per market'),('positive','Winning markets'),('flat','Flat markets'),
                    ('gross_gain','Gross matched gains USD'),('gross_loss','Gross matched losses USD'),('turnover','Completed FIFO USD'),('residual','Residual at payout USD'),
                    ('shares','Filled shares'),('posts','Orders submitted'),('requests','Cancel requests'),('mean_abs_end','Mean absolute settlement inventory in shares'),
                    ('mark_1s_c_per_share','One-second markout in cents per valued share')]:
        text.append(f'| {label} | {current[k]:,.3f} | {new[k]:,.3f} |')
    text += ['',f'The additional rule withheld {r["vetoed_proposals"]:,} proposed placements and reduced {r["clipped_proposals"]:,} sizes. A repeated withheld proposal can refer to the same empty slot.',
             '', 'Every eligible market reproduced the archived control table, including the two policies, fill values, and off-grid diagnostics. Both trial arms reconciled settlement against signed fills and against matched plus residual value. Each FIFO decomposition reproduced its matched ledger. Fixed external flow and simulated queues limit what this establishes about live execution.',
             '', '[Frozen protocol](protocol.md) · [All paired markets](paired_markets.csv) · [Scores and artifact hashes](result.json) · [Five-minute Sharpe for all 22 dates](historical_sharpe.json) · [FIFO attribution](fifo_attribution.csv)']
    loss_days=[(d,v['current']) for d,v in r['by_day'].items() if v['current']['total']<0]
    day_values=', '.join(f'{d} completed {cash(x["turnover"])} USD and residual {cash(x["residual"])} USD' for d,x in loss_days)
    text[-2:-2]=['', 'The negative dates break down as follows. '+day_values+'.',
                 '', f'Across all eight dates the change in gross winning turnover was {cash(new["gross_gain"]-current["gross_gain"])} USD, the change in gross losing turnover was {cash(new["gross_loss"]-current["gross_loss"])} USD and the change in residual was {cash(new["residual"]-current["residual"])} USD. These three quantities sum to the settlement difference.',
                 '', f'The baseline net settlement edge averaged {100*current["total"]/current["shares"]:.3f} cents per executed share before any unmodeled cost.']
    day_lots=pd.concat([pd.read_csv(OUT/d/'attribution.csv') for d in DAYS])
    current_lots=day_lots[day_lots.arm=='current'].groupby(['day','birth']).pnl.sum().unstack(fill_value=0.)
    assert len(current_lots)==len(DAYS) and abs(current_lots.to_numpy().sum()-current['turnover'])<1e-6
    current_lots.to_csv(HERE/'opening_cohorts_by_day.csv')
    text[-2:-2]=['', '## Implication for the next test', '',
                 f'On the original policy, pairs opened by a best-price quote submitted at a one-tick spread earned {cash(current_lots.one_tick_touch.sum())} USD, positive on {int((current_lots.one_tick_touch>0).sum())} of {len(DAYS)} dates. All other opening quotes contributed {cash(current_lots.other.sum())} USD. The per-date breakdown is a descriptive follow-up. These groups differ in market state and subsequent fills.',
                 '', 'Keep the existing best-price entry rule. The next diagnostic should separate second-level openings from best-price openings at wider spreads and measure subsequent inventory movement for each. Use the latest public trades available at the placement decision to distinguish a sweep toward the quote from ordinary flow. This tests whether the losing exposure could be recognized early enough to reduce new inventory while preserving profitable quotes and exits.']
    (HERE/'report.md').write_text('\n'.join(text)+'\n')

if __name__=='__main__':
    build()
