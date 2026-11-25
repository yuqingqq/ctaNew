"""Reconcile the published totals with market CSVs and the two disjoint inventory groups."""
import csv
import json
from math import isclose
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'inventory_urgency_20261125'


def equal(a, b):
    assert isclose(a,b,rel_tol=1e-12,abs_tol=1e-6), (a,b)


def verify(rows, result):
    for arm in ('urgency','reference','retain'):
        data = [r for r in rows if r['arm']==arm]
        assert len({r['mid'] for r in data}) == len(data) == result[arm]['markets']
        for r in data:
            equal(float(r['settle']), float(r['residual'])+float(r['turnover']))
            equal(float(r['shares']), 2*float(r['closed'])+abs(float(r['end_pos'])))
        for field in ('settle','residual','turnover','closed'):
            equal(sum(float(r[field]) for r in data),result[arm][field])
        if arm!='urgency':
            equal(result['urgency']['settle']-result[arm]['settle'],result['paired'][arm]['delta'])
    for name in ('settlement','residual','turnover','markets'):
        equal(result['inventory']['over_60'][name]+result['inventory']['within_60'][name],
              result['urgency'][name])


if __name__=='__main__':
    summary=json.loads((HERE/'result.json').read_text())
    for folder,day in [(OLD,'20261026'),(HERE,'20261027')]:
        with (folder/'markets.csv').open() as f:
            rows=list(csv.DictReader(f))
        verify(rows,summary[day])
        next(r for r in rows if r['arm']=='urgency')['residual']='1000000000'
        try:
            verify(rows,summary[day])
        except AssertionError:
            pass
        else:
            raise ValueError('corrupt residual accepted')
    print('Both days reconcile; corrupt residual refused for each day.')
