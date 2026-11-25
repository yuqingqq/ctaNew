"""Independent integrity checks on frozen replay records, scores and guard boundaries."""
import copy
import csv
import hashlib
import math
from types import SimpleNamespace
from collections import Counter
from unittest.mock import patch

import numpy as np
import trial as T


def same(a,b):
    return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-6)


def validate_market(record, reference):
    arms=record['arms']
    assert set(arms)=={'reference','guard'}
    for arm,r in arms.items():
        assert same(r['settle'],r['turnover']+r['residual']) and same(r['shares'],2*r['closed']+abs(r['end_pos']))
        assert 0<=r['both']<=1 and r['all_pulls']<=r['posts']
        if arm=='reference':
            assert all(same(v,float(reference[k])) for k,v in r.items())
    diag=record['diag']['reference']
    for name in ('flagged','controls'):
        rows=diag[name]
        assert len({gid for gid,*_ in rows})==len(rows)
        for gid,key,val in rows:
            assert 0<=gid<arms['reference']['posts'] and key[0] in ('live','slot')
            total=val.get('post_effect_shares',0)
            assert same(val.get('settle',0),val.get('positive_settle',0)+val.get('negative_settle',0))
            for h in (1,5,30):
                assert same(total,val.get(f'valid_sh_{h}',0)+val.get(f'censored_sh_{h}',0))
                assert same(val.get(f'value_{h}',0),val.get(f'positive_{h}',0)+val.get(f'negative_{h}',0))


def test_slot():
    class Branch:
        k=0
        E=SimpleNamespace(o=[])
    p=T.Guard.__new__(T.Guard)
    p.enabled=True; p.b=Branch(); p.dep=np.ones((2,1)); p.count=Counter()
    def ordinary(policy,v,rung,price):
        policy.proposals.append((0,1,300,1,200,.5))
        policy.opened,policy.first_side=0,1
        return 'POST',200,'post'
    with patch.object(T.S.Smooth,'slot',ordinary):
        def run(pos,orders):
            p.proposals=[]; p.opened=p.first_side=None
            return p.slot(dict(position=pos,A=sum(x['size'] for x in orders),side=1,orders=orders),0,60)
        assert run(0,[])[:2]==('WAIT',None) and p.opened is None and not p.proposals
        assert run(100,[])[:2]==('POST',100) and p.opened==0
        pending=[dict(gid=0,submit=0,size=30,known_filled=0,state='cancel_pending')]
        assert run(100,pending)[:2]==('POST',70)
        assert run(1,[])[:2]==('POST',200)
        p.dep[:]=0
        assert run(0,[])[:2]==('POST',200)
    # Exact inclusive boundary and timely delivery: exchange event alone does not enter the input.
    t=np.array([0.,200.,300.,301.])
    book=np.array([[100,20,101,30],[100,20,101,30],[100,10,101,30],[100,11,101,30]])
    assert T.signal(t,book,np.array([299.,300.,301.]))[0].tolist()==[-1,1,0]


def main():
    test_slot()
    result=T.S.read(T.HERE/'results.json')
    reference={int(r['mid']):r for r in csv.DictReader((T.S.HERE/'markets.csv').open()) if r['arm']=='linear'}
    census={r['mid']:r['tape_id'] for r in T.S.read(T.R.CENSUS)['rows'] if r['day']==T.R.EVAL and r['status']=='complete'}
    assert set(reference)==set(census)
    totals={arm:Counter() for arm in ('guard','reference')}
    parts={}; flagged=Counter(); flags=0
    for mid in sorted(census):
        path=T.OUT/f'{mid}.json'
        parts[str(mid)]=T.R.sha(path)
        record=T.S.read(path)
        assert record['mid']==mid and record['tape']==census[mid] and record['source']==T.R.sha(T.HERE/'trial.py')
        validate_market(record,reference[mid])
        for arm,row in record['arms'].items():
            totals[arm].update(row)
        for _,_,val in record['diag']['reference']['flagged']:
            flags+=1; flagged.update(val)
    assert all(same(v,result['flagged_reference_exposure'][k]) for k,v in flagged.items())
    assert flags==result['unique_first_flags'] and flags>=result['matched_n']>0
    for arm,total in totals.items():
        assert all(same(total[k], result['arms'][arm][k]) for k in ('settle','turnover','closed','residual','all_pulls'))
    assert (result['decision']=='follow_up')==(totals['guard']['settle']>0 and totals['guard']['turnover']>totals['reference']['turnover'])
    bad=copy.deepcopy(record)
    bad['arms']['guard']['settle']+=1
    try:
        validate_market(bad,reference[mid])
    except AssertionError:
        pass
    else:
        raise AssertionError('settlement corruption passed')
    T.R.write(T.HERE/'verification.json',dict(markets=len(parts),part_sha256=parts,checks_pass=True,settlement_corruption_refused=True))
    print('verified paired records, cash, share, horizon, source and policy controls',len(parts))


if __name__=='__main__':
    main()
