"""Render computed full-date, extension-only and original-block results."""
from collections import Counter
from pathlib import Path
import json
import hashlib

HERE=Path(__file__).resolve().parent


def usd(x):
    return ('−' if x<0 else '+')+f'${abs(x):,.2f}'


def report():
    r=json.loads((HERE/'result.json').read_text())
    for day,digest in r['hashes'].items():
        f=HERE/f'paired_{day}.csv'
        if not f.exists():
            f=HERE.parent/'event_pressure_cancel_20261127'/f.name
        assert hashlib.sha256(f.read_bytes()).hexdigest()==digest
    p=r['blocks']['all_fourteen']
    c=Counter()
    for counts in r['coverage'].values():
        c.update(counts)
    names={'additional_nine':'Additional 9 dates','previous_five':'Previously reported 5','all_fourteen':'All 14 dates'}
    out=['# Event-driven pressure cancels across 14 historical dates','',
         f"The event-driven strategy settled at **{usd(p['event_settle'])}** on {p['markets']:,} eligible five-minute markets. The 50 ms control settled at **{usd(p['grid_settle'])}**. The paired difference was {usd(p['difference'])}, with an improvement on {p['positive_delta_days']} of {p['days']} dates.",
         '', '| Block | Eligible markets | 50 ms settlement | Event settlement | Difference | Dates improved | Event profitable dates | Delta without best improvement day |',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
    for block,name in names.items():
        s=r['blocks'][block]
        out.append('| '+' | '.join([name,str(s['markets']),usd(s['grid_settle']),usd(s['event_settle']),usd(s['difference']),f"{s['positive_delta_days']}/{s['days']}",f"{s['event_positive_days']}/{s['days']}",usd(s['delta_without_best_day'])])+' |')
    out.extend(['','| UTC date | Eligible / 288 | 50 ms settlement | Event settlement | Difference | Event completed FIFO | Event residual at payout |',
                '|---|---:|---:|---:|---:|---:|---:|'])
    for day,x in sorted(r['daily'].items()):
        out.append('| '+' | '.join([f'{day[:4]}-{day[4:6]}-{day[6:]}',str(x['markets']),*[usd(x[k]) for k in ('grid_settle','event_settle','difference','event_turnover','event_residual')]])+' |')
    out.extend(['','## Turnover, risk and queue churn','',
                '| Pooled metric | 50 ms | Event-driven |','|---|---:|---:|'])
    for suffix,title,fmt in [('turnover','Completed FIFO profit',usd),('residual','Residual P&L at payout',usd),
                           ('without_best_day','Settlement excluding best day',usd),('without_best_market','Settlement excluding best market',usd),
                           ('mean_end','Mean absolute inventory at payout, shares',lambda v:f'{v:.2f}'),
                           ('mark_c_per_sh','One-second markout, cents per eligible share',lambda v:f'{v:.3f}'),
                           ('shares','Shares filled',lambda v:f'{v:,.0f}'),('closed','Shares in completed round trips',lambda v:f'{v:,.0f}'),
                           ('posts','Submissions',lambda v:f'{v:,}'),('requests','Cancel requests',lambda v:f'{v:,}'),
                           ('reposts','Same-lane reposts following a cancel',lambda v:f'{v:,}')]:
        out.append(f"| {title} | {fmt(p['grid_'+suffix])} | {fmt(p['event_'+suffix])} |")
    out.extend(['',f"The event strategy won on {p['event_winning_markets']} of {p['markets']} markets and broke even on {p['event_zero_markets']}. Its per-market settlement quantiles are shown below.",
                '','| Min | P10 | P25 | Median | P75 | P90 | Max |','|---:|---:|---:|---:|---:|---:|---:|','| '+' | '.join(map(usd,p['event_market_quantiles']))+' |',
                '', '## Interpretation and scope',''])
    for block in ('additional_nine','all_fourteen'):
        s=r['blocks'][block]
        out.append(f"- {names[block]}. The directional screen {'passed' if s['directional_screen'] else 'failed'}. The absolute settlement screen for event cancels {'passed' if s['event_absolute_screen'] else 'failed'}.")
    out.extend(['',f"The registry accounted for {sum(c.values()):,} scheduled markets. "+', '.join(f"{n:,} {k.replace('_',' ')}" for k,n in sorted(c.items()))+'. Excluded markets have unknown economics.',
                '', 'One candidate. Both arms used the frozen BTC pressure thresholds and modeled feed arrival at exchange time plus 4 ms. Live cancel effect took 37 ms, and submissions and reposts remained on the 50 ms grid. Feed updates with an unchanged mid were eligible for pressure review. Both arms generated their own subsequent orders and fills. Public flow, queue matching and zero maker fees/rebates follow the original replay.',
                '', 'All dates have been used in research. The full sample describes historical behavior. The nine additional dates are reported separately because the five-day event result was already known before this extension. The 50 ms policy reproduced its archived ledger fields for every eligible market. Results do not measure exchange-side cancellation or actual wallet fills.',
                '', '[Protocol](protocol.md) · [Daily figures, checks, coverage and file hashes](result.json) · [Original five-date report](../event_pressure_cancel_20261127/report.md)'])
    (HERE/'report.md').write_text('\n'.join(out)+'\n')


if __name__=='__main__':
    report()
