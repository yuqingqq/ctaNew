"""Independent read-back of day membership, ledgers, grid receipts and extension decision."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
PILOT=HERE.parent/'urgency_far_taper_20261127'
OLD=HERE.parent/'urgency_early_strong_20261127'
BASE=HERE.parent/'urgency_cap150_eight_dates_20261126'
EXTRA=('20261023','20261031','20261101','20261102','20261103')
REF=('20261024','20261025','20261027')
KEYS=('abs_end','peak_position','turnover','residual','settle','closed','shares')


def matching(a,b):
    if not math.isclose(float(a),float(b),abs_tol=1e-6,rel_tol=0):
        raise ValueError((a,b))


def rows(root,filename,hashes):
    p=root/filename
    assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[filename]
    with p.open() as f:
        return list(csv.DictReader(f))


def run():
    matching(1,1)
    try:
        matching(1,-1)
    except ValueError:
        pass
    else:
        raise AssertionError('mutated value accepted')
    r=json.loads((HERE/'result.json').read_text())
    hashes={d:json.loads((d/f).read_text())['sha256'] for d,f in ((HERE,'result.json'),(PILOT,'result.json'),(OLD,'result.json'),(BASE,'verification.json'))}
    stats=defaultdict(list)
    coverage=defaultdict(int)
    census=json.loads((BASE/'result.json').read_text())['coverage']
    n_ext=0
    zticks=missing=0
    for day in (*REF,*EXTRA):
        file=f'markets_{day}.csv'
        original=rows(BASE,file,hashes[BASE])
        pair=[(v,'original') for v in original]
        if day in EXTRA:
            extra=rows(HERE,file,hashes[HERE])
            pair.extend((v,v['policy']) for v in extra)
            for v in extra:
                if v['arm']=='nominal' and v['policy']=='lower_far':
                    n_ext+=1
                    a,b=int(v['active_ticks']),int(v['missing_ticks'])
                    assert 0<=b<=a and 0<=int(v['far_tighter_ticks'])<=int(v['tighter_ticks'])<=a-b
                    zticks+=a
                    missing+=b
            for k,v in census[day].items():
                coverage[k]+=v
        else:
            pair.extend((v,'lower_far') for v in rows(PILOT,file,hashes[PILOT]))
            pair.extend((v,'previous_btc') for v in rows(OLD,file,hashes[OLD]) if v['policy']=='btc_time')
        seen=defaultdict(set)
        for v,policy in pair:
            assert v['day']==day
            key=v['mid'],v['tape_id'],v['arm']
            assert key not in seen[policy]
            seen[policy].add(key)
            x={k:float(v[k]) for k in KEYS}
            matching(x['settle'],x['turnover']+x['residual'])
            matching(x['shares'],x['abs_end']+2*x['closed'])
            assert 0<=x['abs_end']<=x['peak_position']+1e-6 and x['peak_position']<=150+1e-6
            for label in (day,'extension5' if day in EXTRA else 'pilot3','all8'):
                stats[v['arm'],policy,label].append((day,x))
        assert set(seen)=={'original','lower_far','previous_btc'}
        assert seen['original']==seen['previous_btc']==seen['lower_far'] and len(original)==2*census[day]['complete']
    assert coverage['complete']==n_ext==408
    for day in (*EXTRA,'20261024'):
        filename=f'grid_{day}.json'
        assert hashlib.sha256((HERE/filename).read_bytes()).hexdigest()==hashes[HERE][filename]
        g=json.loads((HERE/filename).read_text())
        grid=HERE.parents[2]/'data/predict_fun/research/urgency_far_extension_20261127/btc'/f'btc_{day}.parquet'
        assert hashlib.sha256(grid.read_bytes()).hexdigest()==g['sha256']
        assert sum(g['status'].values())==864000 and g['control_parity']==(day=='20261024')
    p={}
    for row in rows(HERE,'comparison.csv',hashes[HERE]):
        key=row['arm'],row['policy'],row['period']
        g=stats.pop(key)
        n=len(g)
        k=(n+9)//10
        v={c:[x[c] for _,x in g] for c in KEYS}
        d=defaultdict(float)
        for day,x in g:
            d[day]+=x['settle']
        expected={c:sum(v[c]) for c in ('turnover','residual','settle','closed','shares')}
        expected.update(markets=n,mean_end=sum(v['abs_end'])/n,tail_end=sum(sorted(v['abs_end'])[-k:])/k,
                        at_cap=sum(x>=150-1e-6 for x in v['abs_end']),mean_peak=sum(v['peak_position'])/n,
                        max_peak=max(v['peak_position']),positive_days=sum(x>0 for x in d.values()),
                        tail_loss=-sum(sorted(v['settle'])[:k])/k)
        for c,y in expected.items():
            matching(row[c],y)
        p[key]=expected
    assert not stats and len(p)==66
    for label,arms in r['decisions'].items():
        for arm,verdict in arms.items():
            a,b=(p[arm,policy,label] for policy in ('lower_far','previous_btc'))
            risk=a['mean_end']<b['mean_end'] and a['tail_end']<b['tail_end']
            gain=a['settle']>=b['settle'] and a['tail_loss']<=b['tail_loss'] and a['positive_days']>=b['positive_days']
            assert verdict==('passes' if risk and gain else 'risk_tradeoff' if risk else 'no_inventory_improvement')
    assert r['extension_passes']==all(v=='passes' for v in r['decisions']['extension5'].values())
    print('Verified all 66 panels and market/tape pairs;',dict(coverage),'extension BTC fallback',missing,'of',zticks)


if __name__=='__main__':
    run()
