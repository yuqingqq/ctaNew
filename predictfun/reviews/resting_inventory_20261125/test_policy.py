"""Regression checks for whole-order selection, reservations and the venue cancel race."""
import numpy as np
from policy import Branch, Policy, R, excess_orders


def order(gid, size, filled=0):
    return dict(gid=gid, size=size, known_filled=filled, submit=gid*50.)


def fixture(s, enabled, trade=True):
    frame=[0.,0.,49,48,47,46,45,0,0,0,0,0,51,52,53,54,55,0,0,0,0,0]
    trades=[[70,s,48 if s==0 else 52,20]] if trade else []
    b=Branch(dict(frames=[frame],trades=trades,mark=50.),500.,{})
    flags={i:np.zeros(10,dtype=bool) for i in (0,1)}
    p=Policy(b,(flags,{side:arr.copy() for side,arr in flags.items()}),{},enabled)
    for price,q in [(48 if s==0 else 52,100),(49 if s==0 else 51,220)]:
        b.unclosed[s].append(b.E.add_order(s,price,0.,q,0))
    return b,p


def shrinking_exit_forecast(s):
    b,p=fixture(s,True,False)
    # Build a signed 350-share holding using two real passive fills before the review.
    prices=[49,48] if s==0 else [51,52]
    path=dict(frames=b.E.f, trades=[[70,s,47 if s==0 else 53,350]], mark=50.)
    b=Branch(path,500.,{})
    p=Policy(b,(p.on,p.trigger),{})
    p.opened=0
    for rung,price in enumerate(prices):
        b.unclosed[s].append(b.E.add_order(s,price,0.,175,rung))
    b.pre_tick(200)
    assert b.reserve(200)[0]==(1-2*s)*350
    gid=b.E.add_order(s,prices[0],200.,100,0)
    b.unclosed[s].append(gid)
    b.pre_tick(250)
    b.k=5
    name=R.groups(p.context(b.k,s))[0]
    p.model[name]=dict(q25=600.,markets=10)
    p.review(250,s)
    assert b.E.o[gid]['req'] is None
    p.model[name]['q25']=250.
    b.side_commands(250,s,p)
    assert b.E.o[gid]['req']==250 and b.reserve(250)[1+s]==100
    p.model[name]['q25']=300.
    b.pre_tick(300)
    b.k=6
    assert p.slot(b.side_view(300,s),1,prices[1])[0]=='WAIT'
    b.pre_tick(400)
    b.k=8
    assert b.reserve(400)[1+s]==0 and p.slot(b.side_view(400,s),1,prices[1])[:2]==('POST',50)


def main():
    keep=order(0,100)
    newer=order(1,220,120)
    assert excess_orders([keep,newer],200)==[]
    assert excess_orders([keep,newer],100)==[newer]
    assert excess_orders([order(0,80),order(1,100)],100)==[order(0,80)]
    assert excess_orders([keep],0)==[keep]
    for s in (0,1):
        shrinking_exit_forecast(s)
        b,p=fixture(s,True)
        R.LB.step(b,1,p)
        assert [(x[1],x[2]) for x in p.trims]==[(1,'post_pending')]
        assert b.reserve(50)[1+s]==320 and b.E.o[0]['req'] is None
        assert b.E.o[1]['remove']==95 and b.E.o[1]['req']==50
        b.pre_tick(100)
        assert b.E.fills[0][1:4]==(1,s,20.) and b.reserve(100)[1+s]==320
        b.k=2; p.review(100,s)
        assert len(p.trims)==1 and b.E.o[0]['req'] is None
        b.pre_tick(175)
        assert b.reserve(175)[1+s]==300 and b.reserve(175)[0]==(1-2*s)*20
        b.pre_tick(200)
        assert b.reserve(200)[1+s]==100 and b.closed[1]==('removed',200.)
        legacy,p0=fixture(s,False,False)
        R.LB.step(legacy,2,p0)
        assert legacy.E.o[1]['req'] is None  # old behavior exposes the defect
        live,p1=fixture(s,True,False)
        R.LB.step(live,2,p1)
        assert live.E.o[1]['remove']==137 and p1.trims[0][2]=='live'
        guard,p2=fixture(s,True,False)
        p2.on[s][2]=True
        R.LB.step(guard,2,p2)
        assert len(p2.trims)==0 and all(o['req']==100 for o in guard.E.o)
    print('Both sides, partial quantity, queue tie, pending and live effects, pressure precedence, legacy falsifier passed.')


if __name__=='__main__':
    main()
