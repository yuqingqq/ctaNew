"""Read-back of the paired inventory test using CSV rows and unrounded sums."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'urgency_early_strong_20261127'


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def match(x,y):
    if not math.isclose(float(x),float(y),rel_tol=0,abs_tol=1e-6):
        raise ValueError((x,y))


def run():
    match(1,1)
    try:
        match(1,2)
    except ValueError:
        pass
    else:
        raise AssertionError('corruption accepted')
    receipt=json.loads((HERE/'result.json').read_text())
    prior=json.loads((PRIOR/'result.json').read_text())
    panels=defaultdict(list)
    for day in ('20261024','20261025','20261027'):
        keys=defaultdict(list)
        for root,record in ((HERE,receipt),(PRIOR,prior)):
            path=root/f'markets_{day}.csv'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'][path.name]
            for r in read(path):
                assert r['day']==day
                policy='previous_btc' if r['policy']=='btc_time' else r['policy']
                key=(r['mid'],r['arm'],r['tape_id'])
                keys[policy].append(key)
                v={k:float(r[k]) for k in ('closed','abs_end','shares','peak_position','settle','turnover','residual','clip','wait','posts')}
                match(v['settle'],v['turnover']+v['residual'])
                match(v['shares'],2*v['closed']+v['abs_end'])
                assert 0<=v['abs_end']<=v['peak_position']+1e-6 and v['peak_position']<=150+1e-6
                if root==HERE:
                    assert 0<=int(r['far_tighter_ticks'])<=int(r['tighter_ticks'])<=int(r['active_ticks'])
                for when in ('pooled',day):
                    panels[(when,r['arm'],policy)].append((day,v))
        assert set(keys)=={'previous_btc','time_only','lower_far'}
        assert all(len(set(k))==len(k) and sorted(k)==sorted(keys['previous_btc']) for k in keys.values())
    path=HERE/'comparison.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['sha256'][path.name]
    summary=read(path)
    assert len(summary)==24
    pooled={}
    for r in summary:
        data=panels.pop((r['day'],r['arm'],r['policy']))
        vals={k:[v[k] for d,v in data] for k in data[0][1]}
        n=len(data)
        k=(n+9)//10
        days=defaultdict(float)
        for d,v in data:
            days[d]+=v['settle']
        calc={c:sum(vals[c]) for c in ('turnover','residual','settle','closed','shares')}
        calc.update(markets=n,mean_end=sum(vals['abs_end'])/n,tail_end=sum(sorted(vals['abs_end'])[-k:])/k,
                    at_cap=sum(x>=150-1e-6 for x in vals['abs_end']),mean_peak=sum(vals['peak_position'])/n,
                    max_peak=max(vals['peak_position']),positive_days=sum(v>0 for v in days.values()),
                    tail_loss=-sum(sorted(vals['settle'])[:k])/k,clips=sum(vals['clip']),waits=sum(vals['wait']),
                    proposals=sum(vals['posts'])+sum(vals['wait']))
        for c,v in calc.items():
            match(r[c],v)
        if r['day']=='pooled':
            pooled[r['arm'],r['policy']]=calc
    assert not panels
    for a in ('short','nominal'):
        new,old=(pooled[a,k] for k in ('lower_far','previous_btc'))
        risk=all(new[k]<old[k] for k in ('mean_end','tail_end'))
        passes=risk and new['settle']>=old['settle'] and new['tail_loss']<=old['tail_loss'] and new['positive_days']>=old['positive_days']
        assert receipt['tests'][a]==('passes' if passes else 'risk_tradeoff' if risk else 'no_inventory_improvement')
    assert receipt['passes']==all(v=='passes' for v in receipt['tests'].values())
    print('Verified market membership and hashes, 24 panels, settlement/quantity identities and both decisions.')


if __name__=='__main__':
    run()
