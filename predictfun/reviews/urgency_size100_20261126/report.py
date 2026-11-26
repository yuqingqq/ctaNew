"""Verify receipt totals from CSV rows and report paired size economics."""
from collections import defaultdict
import csv
import json
from math import isclose
from pathlib import Path

HERE=Path(__file__).resolve().parent


def require_equal(a,b):
    if not isclose(a,b,abs_tol=1e-6,rel_tol=1e-12):
        raise ValueError(f'{a} differs from {b}')


def verify(rows,s):
    pairs=defaultdict(dict)
    for r in rows:
        arm=r['size']
        key=(r['day'],r['mid'])
        assert arm not in pairs[key]
        pairs[key][arm]=r
        require_equal(float(r['turnover'])+float(r['residual']),float(r['settle']))
        require_equal(2*float(r['closed'])+abs(float(r['end_pos'])),float(r['shares']))
    assert all(set(p)=={'100','220'} for p in pairs.values())
    for arm in ('100','220'):
        v=s['arms'][arm]
        assert len(pairs)==v['n']
        for field in ('turnover','residual','settle','closed','posts','all_pulls'):
            require_equal(sum(float(p[arm][field]) for p in pairs.values()),v[field])
        absolute=sorted(abs(float(p[arm]['end_pos'])) for p in pairs.values())
        require_equal(sum(absolute)/len(absolute),v['mean_abs_end'])
        rank=(len(absolute)-1)*.9
        i=int(rank)
        p90=absolute[i]+(rank-i)*(absolute[min(i+1,len(absolute)-1)]-absolute[i])
        require_equal(p90,v['p90_end'])
        require_equal(v['turnover']*100/v['closed'],v['cents_per_matched'])
    differences=[float(p['100']['settle'])-float(p['220']['settle']) for p in pairs.values()]
    require_equal(sum(differences),s['delta'])
    require_equal(sum(differences)-max(differences),s['without_best'])


def dollars(v):
    return ('−' if v<0 else '+')+f'${abs(v):,.2f}'


def main():
    receipt=json.loads((HERE/'result.json').read_text())
    combined=[]
    later=[]
    table=[]
    counts=defaultdict(int)
    for day,s in receipt['days'].items():
        with (HERE/f'markets_{day}.csv').open() as f:
            rows=list(csv.DictReader(f))
        assert {r['day'] for r in rows}=={day}
        verify(rows,s)
        assert sum(s['coverage'].values())==288 and s['coverage']['complete']*2==len(rows)
        for flag,count in s['coverage'].items():
            counts[flag]+=count
        combined.extend(rows)
        if day>='20261031':
            later.extend(rows)
        a,b=s['arms']['100'],s['arms']['220']
        table.append(f"| {day} | {a['n']} | {dollars(a['settle'])} | {dollars(b['settle'])} | {dollars(s['delta'])} | {a['cents_per_matched']:.2f} / {b['cents_per_matched']:.2f} | {a['mean_abs_end']:.1f} / {b['mean_abs_end']:.1f} |")
    verify(combined,receipt['all_dates'])
    verify(later,receipt['four_later_dates'])
    corrupt=[dict(combined[0],residual=str(float(combined[0]['residual'])+1)),*combined[1:]]
    try:
        verify(corrupt,receipt['all_dates'])
    except ValueError:
        pass
    else:
        raise ValueError('corrupt residual passed')
    a,b=receipt['all_dates']['arms']['100'],receipt['all_dates']['arms']['220']
    lines=['# Order-size sensitivity in the passive urgency controller','',
           f"The 100-share cap earned {dollars(a['settle'])} across {a['n']} complete markets. The matched 220-share control earned {dollars(b['settle'])}. The smaller size had positive settlement on {receipt['profitable_days']}/9 days and higher settlement than 220 on {receipt['higher_pnl_days']}/9.",'',
           '| Date | Markets | Settlement at 100 | Settlement at 220 | Difference | ¢ per matched share, 100 / 220 | Mean absolute ending shares, 100 / 220 |',
           '|---|---:|---:|---:|---:|---:|---:|',*table,'','## Aggregate economics','',
           '| Population and maximum size | Completed FIFO | Residual | Settlement | Matched shares | ¢ / matched share | Mean / p90 abs. ending shares | Cancellation requests |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,key in [('All nine dates','all_dates'),('Later four dates','four_later_dates')]:
        for unit,v in receipt[key]['arms'].items():
            lines.append(f"| {name}, {unit} | {dollars(v['turnover'])} | {dollars(v['residual'])} | {dollars(v['settle'])} | {v['closed']:,.0f} | {v['cents_per_matched']:.3f} | {v['mean_abs_end']:.2f} / {v['p90_end']:.2f} | {v['all_pulls']:,.0f} |")
    lines.extend(['','## Scope and checks','',
        'As of November 26, 2026. Nine previously examined dates: '+', '.join(f'{k} {v}' for k,v in sorted(counts.items()))+'. Each daily table uses every complete market in the frozen census. October 23 predates the October 24–25 capacity fit. October 26–27 were strategy development dates. October 31–November 3 are reported separately as a later slice previously used in other research.',
        '',
        'Both sizes use two rungs, the fixed 1,000-share inventory ceiling, the same time-dependent allowance, a 440-share same-side reservation ceiling including pending cancellations, the same frozen capacity lookup and the same queue, latency and payout accounting. Orders may be smaller than the maximum. Positive passive settlement excludes rebates and assumes no maker fee or change to public flow. The smaller size can change later orders, fills and position through policy feedback. This replay measures no return on capital.',
        '',
        'All 220-share controls are bound to saved committed outputs. The lowest-ID complete market on each date reproduced all nine checked market fields. Each candidate submission was checked against 100; a synthetic 101-share order was refused. An independent read of the paired market CSVs reconciled the daily and pooled summaries and rejected a corrupted residual. Freeze: 4ed4049f2.',
        '',
        '[Protocol](protocol.md) · [All metrics and coverage by date](result.json)'])
    with (HERE/'report.md').open('x') as f:
        f.write('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
