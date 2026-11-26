import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
result=json.loads((HERE/'result.json').read_text())


def money(amount):
    return ('−' if amount<0 else '+')+f'${abs(amount):,.2f}'


rows=['# Earlier-date replay of the retained urgency controller','',
      'These three dates were selected before scoring from the existing unreserved tapes. The capacity lookup used October 24 and 25. Applying it to October 23 uses a later fit. October 24–25 share training-day information, including direct overlap with the fit manifest. Read these as retrospective stress results, separately from the six previously reported dates.','',
      '| UTC date | Complete / scheduled | Original ladder | Urgency settlement | Completed FIFO | Residual | ¢ / matched share | Mean abs. ending / ceiling (shares) | Criteria |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---|']
for day,v in result['days'].items():
    a,b=v['arms']['urgency'],v['arms']['original']
    missing=[k.replace('_',' ') for k,ok in v['gate'].items() if not ok]
    verdict='miss: '+', '.join(missing) if missing else 'pass'
    rows.append(f"| {day} | {a['markets']} / 288 | {money(b['settle'])} | {money(a['settle'])} | {money(a['turnover'])} | {money(a['residual'])} | {a['cents_per_matched']:.2f} | {a['mean_abs_end']:.1f} / {b['mean_abs_end']+10:.1f} | {verdict} |")
rows.extend(['','## Capacity training overlap','',
             '| Fit date | Markets directly used in fit | Settlement there | Other complete markets on same date | Settlement there |',
             '|---|---:|---:|---:|---:|'])
for day in ('20261024','20261025'):
    groups=result['days'][day]['fit_overlap_breakdown']
    used,other=(groups[k]['arms']['urgency'] for k in ('True','False'))
    rows.append(f"| {day} | {used['markets']} | {money(used['settle'])} | {other['markets']} | {money(other['settle'])} |")
rows.extend(['','## Coverage and verification',''])
for d,v in result['days'].items():
    rows.append('- '+d+' — '+', '.join(f'{k}: {n}' for k,n in sorted(v['coverage'].items()))+'.')
rows.extend(['','As of November 26, 2026. Missing or stale markets have unknown hypothetical P&L. Settlement includes both completed FIFO trades and holdings through the recorded payout. Both arms were reproduced on a previously checked market for each signal archive. The resulting paired ledgers matched the source tape identities, and the independent CSV check reconciled each daily result and refused a $1 corruption. One wrapper import failed before the run and was corrected before scoring. No new strategy variant or fees. No untouched November block was opened by this test.',
             '','[Frozen protocol](protocol.md) · [Import correction](amendment.md) · [Results and market-level breakdowns](result.json)'])
with (HERE/'report.md').open('x') as f:
    f.write('\n'.join(rows)+'\n')
