"""Second accounting of the daily and pooled receipts from the published paired CSVs."""
import csv
import json
from math import isclose
from pathlib import Path

HERE = Path(__file__).resolve().parent
DAYS = ('20261031','20261101','20261102','20261103')


def equal(a,b):
    if not isclose(a,b,rel_tol=1e-12,abs_tol=1e-6):
        raise ValueError(f'mismatch {a} {b}')


def verify(rows,summary):
    arms={name:{} for name in ('original','urgency')}
    for r in rows:
        key=r['day'],r['mid']
        assert key not in arms[r['arm']]
        arms[r['arm']][key]=r
        equal(float(r['settle']),float(r['turnover'])+float(r['residual']))
        equal(float(r['shares']),2*float(r['closed'])+abs(float(r['end_pos'])))
    a,b=arms['urgency'],arms['original']
    assert a.keys()==b.keys() and all(r['tape_id']==b[k]['tape_id'] for k,r in a.items())
    results={}
    for arm,ledger in arms.items():
        out=summary['arms'][arm]
        assert out['markets']==len(ledger)
        result={f:sum(float(r[f]) for r in ledger.values()) for f in ('settle','turnover','residual','closed','abs_end')}
        for f in ('settle','turnover','residual'):
            equal(result[f],out[f])
        equal(result['closed'],out['matched'])
        equal(100*result['turnover']/result['closed'],out['cents_per_matched'])
        result['mean_end']=result['abs_end']/len(ledger)
        equal(result['mean_end'],out['mean_abs_end'])
        results[arm]=result
    candidate,base=results['urgency'],results['original']
    checks=dict(positive_settlement=candidate['settle']>0,positive_turnover=candidate['turnover']>0,
                improved_settlement=candidate['settle']>base['settle'],improved_turnover=candidate['turnover']>base['turnover'],
                within_inventory_limit=candidate['mean_end']<=base['mean_end']+10)
    assert checks==summary['gate'] and all(checks.values())==summary['passed']
    diffs=[float(v['settle'])-float(b[k]['settle']) for k,v in a.items()]
    equal(sum(diffs),summary['delta'])
    equal(sum(diffs)-max(diffs),summary['delta_without_best'])


if __name__=='__main__':
    receipt=json.loads((HERE/'result.json').read_text())
    all_rows=[]
    for day in DAYS:
        with (HERE/f'markets_{day}.csv').open() as f:
            rows=list(csv.DictReader(f))
        assert {r['day'] for r in rows}=={day}
        stats=receipt['days'][day]
        assert sum(stats['coverage'].values())==288 and stats['coverage']['complete']*2==len(rows)
        verify(rows,stats)
        all_rows.extend(rows)
    verify(all_rows,receipt['pooled'])
    assert receipt['positive_days']==sum(receipt['days'][d]['arms']['urgency']['settle']>0 for d in DAYS)
    assert receipt['passed_days']==sum(receipt['days'][d]['passed'] for d in DAYS)
    all_rows[0]=dict(all_rows[0],residual=str(float(all_rows[0]['residual'])+1))
    try:
        verify(all_rows,receipt['pooled'])
    except ValueError:
        print('All market pairs, payouts, daily gates and pooled values reconcile. Altered residual refused.')
    else:
        raise AssertionError('accepted a corrupted residual')
