from pathlib import Path
import json

from verify import DAYS

HERE=Path(__file__).resolve().parent
receipt=json.loads((HERE/'result.json').read_text())
summary=receipt['pooled']['arms']['urgency']


def usd(v):
    return f"{'−' if v<0 else '+'}${abs(v):,.2f}"


lines=[
    '# Passive urgency controller on four later dates','',
    f"Using the controller frozen before these replays, settlement was positive on **{receipt['positive_days']}/4 dates**. "
    f"Across {summary['markets']} complete markets it earned **{usd(summary['settle'])}**: completed FIFO {usd(summary['turnover'])} "
    f"and residual inventory {usd(summary['residual'])}. The original daily criteria all passed on **{receipt['passed_days']}/4 dates**.",'',
    '| Date | Complete / scheduled | Original settlement | Urgency settlement | Urgency completed FIFO | Residual P&L | ¢ / matched share | Mean abs. ending / daily ceiling (shares) | Full gate |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---|'
]
for day in DAYS:
    cell=receipt['days'][day]
    a,b=cell['arms']['urgency'],cell['arms']['original']
    failed=', '.join(k.replace('_',' ') for k,v in cell['gate'].items() if not v)
    verdict='miss: '+failed if failed else 'pass'
    lines.append(f"| {day} | {a['markets']} / 288 | {usd(b['settle'])} | {usd(a['settle'])} | {usd(a['turnover'])} | {usd(a['residual'])} | {a['cents_per_matched']:.3f} | {a['mean_abs_end']:.2f} / {b['mean_abs_end']+10:.2f} | {verdict} |")
lines.extend(['','## Coverage and paired contrasts','',
    '| Date | Census statuses (all scheduled markets) | Urgency − original | After dropping largest improvement | Improved / worsened / equal markets |',
    '|---|---|---:|---:|---:|'])
for day in DAYS:
    cell=receipt['days'][day]
    c=', '.join(f'{k} {v}' for k,v in sorted(cell['coverage'].items()))
    same=cell['coverage']['complete']-cell['better']-cell['worse']
    lines.append(f"| {day} | {c} | {usd(cell['delta'])} | {usd(cell['delta_without_best'])} | {cell['better']} / {cell['worse']} / {same} |")
lines.extend(['','## Scope and accounting','',
    'As of November 26, 2026. All four dates are later than the October 24–25 capacity fit and the October 26–27 urgency work. They had been used in other project research. Incomplete markets have unknown economic outcomes. Only complete tapes enter the economic totals. The November 3 forward-boundary exclusion was carried from the existing census. No interval is reported. The original daily gate compares mean absolute ending inventory with the matched original ladder plus ten shares.',
    '',
    'Both arms start flat per market, act on the same recorded external flow and keep their own queues and inventory. All executions are passive, with zero modeled maker fees or rebates. Remaining holdings at the last supported tick (two seconds before expiry) receive the registry settlement payout, including ties. A cent per matched share is completed FIFO P&L divided by matched quantity. These are dollars across the replayed markets and do not define a return on capital.',
    '',
    'The lowest-ID complete October 27 control market reproduced both arms’ saved settlement, turnover, residual, matched shares, signed and absolute ending positions, total shares, submissions and cancellations. Each later tape matched its archived identity. The independent checker reconciled every fill-derived market total, shared tape, daily gate and pooled total, and rejected a deliberately corrupted residual.',
    '',
    '[Frozen evaluation protocol](protocol.md) · [Result and criteria](result.json) · [Independent check](verify.py)'
])
with (HERE/'report.md').open('x') as f:
    f.write('\n'.join(lines)+'\n')
