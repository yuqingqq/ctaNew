"""Read saved market CSVs independently and reconcile the three-size comparison."""
import csv
import io
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DAYS = ('20261023','20261024','20261025','20261026','20261027',
        '20261031','20261101','20261102','20261103')
SIZES = (220, 100, 50)
FIELDS = ('settle','turnover','residual','closed','end_pos','abs_end','shares','posts','all_pulls')
COUNTS = (70,81,72,70,58,102,88,72,76)
TOL = 1e-6


def validate(r):
    assert all(math.isfinite(v) for v in r.values())
    assert abs(r['settle']-r['turnover']-r['residual']) < TOL
    assert abs(r['turnover']-r['gross_gain']-r['gross_loss']) < TOL
    assert abs(r['closed']-sum(r[k] for k in ('win_sh','loss_sh','flat_sh'))) < TOL
    assert abs(r['shares']-2*r['closed']-r['abs_end']) < TOL
    assert abs(r['abs_end']-abs(r['end_pos'])) < TOL
    for col in ('closing_orders','filled_orders','posts','fill_events','order_pairs','all_pulls'):
        assert r[col] >= 0 and r[col]==int(r[col])
    assert r['closing_orders']<=r['filled_orders']<=min(r['posts'],r['fill_events'])
    assert r['order_pairs']>=r['closing_orders'] and r['gross_gain']>=0>=r['gross_loss']
    assert min(r[k] for k in ('win_sh','loss_sh','flat_sh'))>=0


def read():
    groups = defaultdict(list)
    by_market = {}
    for day,count in zip(DAYS,COUNTS):
        prev = HERE.parent/'urgency_size100_20261126'/f'markets_{day}.csv'
        committed = subprocess.check_output(['git','show',f'c2f5ff352:{prev.relative_to(ROOT)}'],cwd=ROOT,text=True)
        controls = {(int(r['mid']),int(r['size'])):r for r in csv.DictReader(io.StringIO(committed))}
        seen = set()
        with (HERE/f'markets_{day}.csv').open() as f:
            for rec in csv.DictReader(f):
                assert rec.pop('day')==day
                mid, n = int(rec.pop('mid')), int(rec.pop('size'))
                assert n in SIZES and (mid,n) not in seen
                seen.add((mid,n))
                r = {k:float(v) for k,v in rec.items()}
                validate(r)
                if n!=50:
                    assert all(abs(r[k]-float(controls[mid,n][k])) < TOL for k in FIELDS)
                for label in ('all', day, 'later' if day>=DAYS[5] else 'earlier'):
                    groups[label,n].append(r)
                by_market[day,mid,n] = r
        ids = {m for m,_ in controls}
        assert len(ids)==count and seen=={(m,n) for m in ids for n in SIZES}
    sample = next(iter(by_market.values()))
    for field, bad in [('settle',sample['settle']+1), ('closing_orders',sample['filled_orders']+1)]:
        try:
            validate(dict(sample,**{field:bad}))
        except AssertionError:
            continue
        raise AssertionError(f'accepted damaged {field}')
    return groups


def aggregate(rows):
    total = {k:math.fsum(r[k] for r in rows) for k in rows[0]}
    total['cents_per_matched'] = 100*total['turnover']/total['closed']
    total['mean_closed_per_pair'] = total['closed']/total['order_pairs']
    total['mean_abs_end'] = total['abs_end']/len(rows)
    total['markets'] = len(rows)
    return total


def report():
    sums = {key:aggregate(rows) for key,rows in read().items()}
    text = ['# What changes at 50 shares per level?', '',
            'Paired historical replay as of November 26, 2026. All three sizes use the same 689 complete markets. Every order follows its own queue, fill and cancellation history.', '']
    def table(header, rows):
        text.extend(['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|'])
        text.extend('| '+' | '.join(row)+' |' for row in rows)
        text.append('')
    def money(n):
        return ('−' if n<0 else '+')+f'${abs(n):,.2f}'
    def count(n):
        return f'{n:,.0f}'
    for scope, title in (('all','All nine dates'),('later','October 31 through November 3')):
        text.extend(['## '+title,''])
        table(['Max shares per level','Orders posted','Orders with fills','Fill events','Orders closing inventory','FIFO order pairs','Matched shares'],
              [[str(n)]+[count(sums[scope,n][k]) for k in ('posts','filled_orders','fill_events','closing_orders','order_pairs','closed')] for n in SIZES])
        table(['Max shares per level','Gross completed gains','Gross completed losses','Completed FIFO','¢ per matched share','Residual at payout','Settlement','Mean abs. end shares'],
              [[str(n)]+[money(sums[scope,n][k]) for k in ('gross_gain','gross_loss','turnover')]+[f"{sums[scope,n]['cents_per_matched']:.3f}"]+[money(sums[scope,n][k]) for k in ('residual','settle')]+[f"{sums[scope,n]['mean_abs_end']:.2f}"] for n in SIZES])
    text.extend(['## Settlement by date',''])
    table(['UTC date','Markets','220','100','50','50 less 100'],
          [[d,count(sums[d,50]['markets'])]+[money(sums[d,n]['settle']) for n in SIZES]+[money(sums[d,50]['settle']-sums[d,100]['settle'])] for d in DAYS])
    text.extend(['## Interpretation',''])
    for n, base in ((100,220),(50,100)):
        a,b = sums['all',n],sums['all',base]
        wins = sum(sums[d,n]['settle']>sums[d,base]['settle'] for d in DAYS)
        pos = sum(sums[d,n]['settle']>0 for d in DAYS)
        text.append(f"Moving from {base} to {n} changed gross gains by {money(a['gross_gain']-b['gross_gain'])}, gross losses by {money(a['gross_loss']-b['gross_loss'])} and residual value by {money(a['residual']-b['residual'])}. Settlement changed by {money(a['settle']-b['settle'])}. The {n}-share arm exceeded {base} on {wins}/9 dates and exceeded zero on {pos}/9.")
        text.append('')
    text.extend(['Each order is counted once per market when it receives its first fill. A single sell can close several earlier buys. A FIFO order pair counts a distinct opening/closing order-ID combination, and matched shares count the completed quantity once. Multiple executions against the same order IDs can add volume without adding a pair. Gross gains and losses apply to matched quantities at the two realized prices. Remaining inventory is valued at the binary payout.',
                 '', 'Of 2,592 scheduled markets, 1,062 lacked an endpoint, 840 had stale future books and one crossed the end of the available feed. October 23 precedes the capacity fit, October 24–25 supplied the fit, and October 26–27 were used in controller development. The four later dates were already examined. Results assume the same public order flow for each arm, zero maker fees and no rebates. No evidence here identifies live queue priority or an optimal size.',
                 '', 'The 100- and 220-share paths reproduced the nine previously saved market fields on every market. Fill and FIFO accounting reconciled for each size, including a synthetic partial-fill/reversal and corrupted-count and settlement refusals. The runner and protocol were frozen in 8831ad669.'])
    (HERE/'report.md').write_text('\n'.join(text)+'\n')
    with (HERE/'summary.csv').open('w') as f:
        w = csv.DictWriter(f,fieldnames=['scope','size',*next(iter(sums.values()))])
        w.writeheader()
        for (label,n), row in sorted(sums.items()):
            w.writerow(dict(scope=label,size=n,**row))


if __name__=='__main__':
    report()
