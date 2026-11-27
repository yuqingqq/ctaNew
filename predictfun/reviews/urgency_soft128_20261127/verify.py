"""Independent CSV read-back of eligible markets, tails, day panels and the decision."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'urgency_cap150_eight_dates_20261126'
DAYS = ('20261023','20261024','20261025','20261027','20261031','20261101','20261102','20261103')


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def matching(actual,expected):
    if not math.isclose(float(actual),float(expected),rel_tol=0,abs_tol=1e-6):
        raise ValueError(f'{actual} differs from {expected}')


def main():
    matching(1,1)
    try:
        matching(1,2)
    except ValueError:
        pass
    else:
        raise AssertionError('mutated amount accepted')
    result = json.loads((HERE/'result.json').read_text())
    old_hashes = json.loads((BASE/'verification.json').read_text())['sha256']
    panels = defaultdict(list)
    for day in DAYS:
        name = f'markets_{day}.csv'
        membership = []
        for policy,root,hashes in [('current',BASE,old_hashes),('soft128',HERE,result['sha256'])]:
            path = root/name
            assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[name]
            keys = set()
            for row in read(path):
                key = (row['mid'],row['arm'],row['tape_id'])
                assert key not in keys and row['day']==day
                keys.add(key)
                num = {k:float(row[k]) for k in ('settle','turnover','residual','abs_end','shares','closed','peak_position')}
                matching(num['settle'],num['turnover']+num['residual'])
                matching(num['shares'],2*num['closed']+num['abs_end'])
                assert 0<=num['abs_end']<=num['peak_position']+1e-6 and num['peak_position']<=150+1e-6
                if policy=='soft128': assert num['peak_position']<=128+1e-6
                for when in (day,'pooled'):
                    panels[(when,row['arm'],policy)].append((day,num))
            membership.append(keys)
        assert membership[0]==membership[1]
    summary = read(HERE/'comparison.csv')
    assert len(summary)==len(panels)==36
    truths = {}
    for row in summary:
        key = row['day'],row['arm'],row['policy']
        data = panels.pop(key)
        n = len(data)
        matching(row['markets'],n)
        k = (n+9)//10
        pnl = defaultdict(float)
        for day,v in data:
            pnl[day]+=v['settle']
        values = {c:[r[c] for _,r in data] for c in data[0][1]}
        expected = {c:sum(values[c]) for c in ('settle','turnover','residual','shares','closed')}
        expected.update(mean_end=sum(values['abs_end'])/n,tail_end=sum(sorted(values['abs_end'])[-k:])/k,
                        tail_loss=-sum(sorted(values['settle'])[:k])/k,positive_days=sum(x>0 for x in pnl.values()),
                        max_peak=max(values['peak_position']),mean_peak=sum(values['peak_position'])/n,
                        at_cap=sum(x>=150-1e-6 for x in values['abs_end']))
        for col,value in expected.items():
            matching(row[col],value)
        truths[key] = expected
    assert not panels
    passes = []
    for arm in ('nominal','short'):
        base = truths['pooled',arm,'current']
        new = truths['pooled',arm,'soft128']
        lower = all(new[c]<base[c] for c in ('mean_end','tail_end'))
        passes.append(lower and new['tail_loss']<=base['tail_loss'] and new['settle']>=base['settle'] and new['positive_days']>=base['positive_days'])
        assert result['verdicts'][arm]==('passes' if passes[-1] else 'risk_tradeoff' if lower else 'no_inventory_improvement')
    assert result['candidate_passes']==all(passes)
    print('Verified 36 daily/pooled panels, pair membership, source hashes, risk and payout identities and combined screen.')


if __name__=='__main__':
    main()
