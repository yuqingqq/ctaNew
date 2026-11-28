from pathlib import Path
import json
from audit import HERE, DAYS

def build():
    r=json.loads((HERE/'results.json').read_text())
    a,p=r['archive'],r['pooled']
    lines=['# Did late cancellation cause the losses?', '',
           'November 28, 2026 UTC. The unchanged event-driven strategy had two negative dates in the last eight. This audit reconstructs every fill in all 124 eligible markets on those dates, including profitable markets. The other 452 scheduled markets on those two days had incomplete or stale inputs. They have unknown economics.',
           '',f'On all eight dates, {a["post_request"]:,.1f} of {a["shares"]:,.1f} simulated shares filled strictly after a cancel request ({100*a["post_request"]/a["shares"]:.2f}%). Moving pressure cancellation from the 50 ms grid to feed events had improved settlement on the same 550 markets by ${a["event_minus_grid"]:,.2f}.',
           '', '## Which legs had a cancel in flight?', '', '| Date | Pair classification | Matched shares | Gross losses USD | Gross gains USD | Net completed USD |','|---|---|---:|---:|---:|---:|']
    for name in (*DAYS,'pooled'):
        obj=r[name]
        for label,sources in [('Either leg pending',['opening_pending','closing_only_pending']),('Neither pending',['neither_pending']),('All',['all'])]:
            fields=[sum(obj[k][s] for k in sources) for s in ('shares','bad','good','net')]
            lines.append('| '+name+' | '+label+' | '+' | '.join(f'{v:,.2f}' for v in fields)+' |')
    lines+=['','A request must precede the match strictly. Fills at the decision timestamp execute first. Live cancel effect is modeled at request +37 ms. Each FIFO tranche is counted once. This split describes the existing feedback path. A changed cancellation time could affect future orders, good fills and inventory.',
            '', '## Where did the price move?', '', '| Completed cohort | Entry edge USD | First second of holding USD | Later movement until close USD | Exit edge USD | Unmarked pair P&L USD | Total USD |','|---|---:|---:|---:|---:|---:|---:|']
    for group in ('all','losing_pairs','winning_or_flat','losing_markets'):
        v=p[group]
        lines.append('| '+group+' | '+' | '.join(f'{v[s]:,.2f}' for s in ('entry','first_second','later','exit','missing_value','net'))+' |')
    lines+=['','Any pair closed within one second has zero later movement. All three marks must be valid and at most one second old. Matched shares missing at least one mark total '+f'{p["all"]["missing_shares"]:,.2f} of {p["all"]["shares"]:,.2f}. The losing-pair breakdown is a hindsight cohort.',
            '', '## How identifiable were the openings?', '', '| Opening-fill classification | Matched shares | Gross losses USD | Gross gains USD | Net completed USD |','|---|---:|---:|---:|---:|']
    for dim in ('opening','prefill','trade'):
        for k,v in p[dim].items():
            lines.append('| '+k+' | '+' | '.join(f'{v[s]:,.2f}' for s in ('shares','bad','good','net'))+' |')
    lines += ['', 'The warning is the current order side’s 250 ms Binance pressure threshold, checked on grid ticks and off-grid feed evaluations from activation up to, but excluding, the match. A fill can arrive immediately after activation; absence of a warning does not establish time for a feasible cancel. The prefill midpoint uses the strictly preceding received two-sided book within one second. Local midpoint, public execution price and an executable prediction of fair value are different measures. A through-price print crosses our quote on the realized tape.',
              '', '## One-second fill markout', '', '| Status | Filled shares | Shares with mark | Gross adverse USD | Gross favorable USD |','|---|---:|---:|---:|---:|']
    for k,v in p['fills'].items():
        lines.append('| '+k+' | '+' | '.join(f'{v[s]:,.2f}' for s in ('shares','valued','gross_bad_1s','gross_good_1s'))+' |')
    lines += ['', 'Both days reproduced every archived per-market control column. Economic differences reflect modeled matching, the historical clock, a fixed public tape and zero maker fees or rebates. The eight dates were previously examined. The two-day breakdown is descriptive.',
              '', '[Protocol](protocol.md) · [Full results](results.json) · [Earlier eight-date attribution](../pressure_market_sharpe_20261128/report.md)']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    build()
