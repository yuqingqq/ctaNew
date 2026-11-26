"""Verify proposal examples and recompute the daily gate from the saved pairs."""
import csv
import json
from math import isclose
from statistics import mean
from types import SimpleNamespace

import replay_v2 as E


def proposal(cls, side, signed, reserved, left):
    branch=SimpleNamespace(ident=dict(opening_ms=0,start_abs_ms=0),inv=dict(Qside=440,S=500),
                           end=298000, q0=220, cap=1000, k=0, prof=dict(place=10),
                           flow=lambda *args:0)
    on={0:[False],1:[False]}
    policy=cls(branch,(on,on),{f'{side}/5':dict(markets=10,q25=1000)},'linear',{},[])
    policy.opened=-1
    t=300000-left*1000
    view=dict(side=side,position=(1-2*side)*signed,B=reserved if side==0 else 0,
              A=reserved if side==1 else 0,t=t,received=t,bid=49,ask=51)
    action,q,_=policy.slot(view,0,50)
    assert len(policy.proposals)==int(action=='POST')
    assert policy.limit(t)==E.U.S.curve(left)
    return q


def check_rows(rows, receipt):
    assert len(set((r['day'],r['mid'],r['arm']) for r in rows))==len(rows)
    for day in E.DAYS:
        pairs={}
        for r in rows:
            if r['day']==day:
                pairs.setdefault(r['mid'],{})[r['arm']]=r
                assert isclose(float(r['settle']),float(r['turnover'])+float(r['residual']),abs_tol=1e-6)
        assert len(pairs)==receipt['coverage'][day]['complete']
        daily={}
        for arm in ('ahead','urgency'):
            data=[p[arm] for p in pairs.values()]
            record=receipt[day][arm]
            pnl=sum(float(r['settle']) for r in data)
            fifo=sum(float(r['turnover']) for r in data)
            ending=mean(float(r['abs_end']) for r in data)
            over=sum(float(r['abs_60s'])>250+1e-8 for r in data)
            for a,b in [(pnl,record['settle']),(fifo,record['turnover']),(ending,record['mean_end']),(over,record['over_60'])]:
                assert isclose(a,b,abs_tol=1e-6)
            daily[arm]=(pnl,fifo,ending)
        a,b=daily['ahead'], daily['urgency']
        assert receipt[day]['passed']==(a[0]>b[0] and a[1]>0 and a[2]<b[2])
    assert receipt['passed_both_days']==all(receipt[d]['passed'] for d in E.DAYS)


if __name__=='__main__':
    for side in (0,1):
        for cls,expected in [(E.Ahead,10),(E.U.S.Smooth,135)]:
            assert proposal(cls,side,210,30,75)==expected
        assert proposal(E.Ahead,side,-350,30,75)==220
        assert proposal(E.Ahead,side,210,40,75) is None
        assert proposal(E.Ahead,side,210,30,91)==220
    with (E.HERE/'markets.csv').open() as f:
        rows=list(csv.DictReader(f))
    receipt=json.loads((E.HERE/'result.json').read_text())
    check_rows(rows,receipt)
    rows[0]['settle']=str(float(rows[0]['settle'])+1)
    try:
        check_rows(rows,receipt)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt settlement accepted')
    print('Native slot path verified on both sides. Both daily gates reconcile. Damaged settlement refused.')
