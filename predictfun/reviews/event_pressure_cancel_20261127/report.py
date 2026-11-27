"""Write a short receipt from verified per-market and per-day results."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
r=json.loads((HERE/'result.json').read_text())
for name,digest in r['files'].items():
    assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
p=r['pooled']


def money(v):
    return ('−' if v<0 else '')+f'${abs(v):,.2f}'


def fmt(v):
    return f'{v:,.0f}'


lines=['# Checking pressure between 50 ms decisions', '',
       f"November 19–23, 2026 UTC. Both policies ran on {p['markets']} complete markets from the previously consumed five-day block. Every control reproduced its archived nominal ledger. This replay used one additional candidate.",
       '', '## Paired outcome', '',
       '| Date UTC | Markets | Grid settlement | Event settlement | Difference | Off-grid cancel requests |',
       '|---|---:|---:|---:|---:|---:|']
for label,v in [*r['daily'].items(),('Pooled',p)]:
    lines.append('| '+' | '.join([label,str(v['markets']),money(v['grid_settle']),money(v['event_settle']),money(v['difference']),fmt(v['event_event_pulls'])])+' |')
lines.extend(['',f"The event policy improved settlement on {r['positive_delta_days']} of 5 dates. Its pooled difference after dropping the largest daily improvement was {money(r['delta_without_best_day'])}. The predeclared screen for a further shadow test **{'passed' if r['next_shadow_screen'] else 'failed'}**.",
              '', '| Metric | Grid | Event |','|---|---:|---:|'])
for key,title,disp in [
        ('turnover','Completed FIFO turnover P&L',money),('residual','Residual inventory P&L at payout',money),
        ('closed','Matched shares',fmt),('mean_end','Mean absolute ending inventory, shares',lambda x:f'{x:.2f}'),
        ('posts','Submissions',fmt),('requests','Cancel requests, all reasons',fmt),
        ('reposts','Same-lane submissions following a cancel',fmt),('shares','Filled shares',fmt),
        ('post_request_shares','Shares matched after a cancel request',fmt),
        ('mark_c_per_sh','Eligible +1 s markout, cents per share',lambda x:f'{x:.3f}')]:
    lines.append(f"| {title} | {disp(p['grid_'+key])} | {disp(p['event_'+key])} |")
by_day = [x['event_settle'] for x in r['daily'].values()]
lines.extend(['',f"The event strategy itself earned a positive settlement on {sum(x>0 for x in by_day)} of five days. Removing its best day leaves {money(sum(by_day)-max(by_day))} across the other four."])
if p['event_event_pulls']:
    lines.extend(['',f"The {fmt(p['event_event_pulls'])} off-grid requests were issued a mean {p['event_lead_sum']/p['event_event_pulls']:.1f} ms before the next grid tick. A transient signal may have cleared by that tick, and later orders can diverge."])
lines.extend(['','## Fill differences at one second','',
              'Quantity is matched between arms by public trade, side and fill price. The values below include changes in later orders after the policies diverge.',
              '', '| Fill quantity present only in | Favorable shares | Favorable value | Adverse shares | Adverse value | Flat shares | Unvalued shares |',
              '|---|---:|---:|---:|---:|---:|---:|'])
for key,label in [('lost','Grid arm'),('added','Event arm')]:
    lines.append('| '+' | '.join([label,fmt(p[key+'_win_sh']),money(p[key+'_favorable']),fmt(p[key+'_loss_sh']),money(p[key+'_adverse']),fmt(p[key+'_flat_sh']),fmt(p[key+'_censored'])])+' |')
lines.extend(['','The venue can match until 37 ms after a live cancel request in both arms. Prices, including size-only update opportunities, use the frozen model of delivery at Binance event time plus 4 ms with one observable book per event millisecond. The public trade path is fixed and private fills are simulated. A wallet or live Tokyo latency measurement requires separate evidence.',
              '', 'The 1,440 scheduled markets included 872 missing endpoints, 215 stale books and one missing future book. Economics for those markets are unknown. Dates were previously used in the November holdout. The 4,096 day-level sign flips yield only 32 distinct sign patterns.',
              '', '[Frozen protocol](protocol.md) · [All metrics and per-market file hashes](result.json)'])
(HERE/'report.md').write_text('\n'.join(lines)+'\n')
