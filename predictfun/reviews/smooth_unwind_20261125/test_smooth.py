"""Controls for the curve, close count, causal horizons, protection and liquidation label."""
import math
import numpy as np
from smooth import curve, close_stats, state_vector, probe, Smooth, R
from test_policy import fixture


def main():
    assert all(curve(30,r)==0 and curve(150,r)==1000 for r in (.5,1,2))
    assert curve(90)==500 and curve(90,.5)<500<curve(90,2)
    assert all(curve(t)<=curve(t+1) for t in range(300))
    assert abs(np.trapezoid([curve(t) for t in range(301)],dx=1)-210000)<1e-7
    assert state_vector(90,0,2,220,True)==[.6,2,math.log(2),1.]
    assert state_vector(90,1,2,0,False)==[.6,-2,0.,0.]
    fills=[(0,0,0,200,50,0),(1,1,1,250,50,1),(2,2,0,20,50,2)]
    assert close_stats(fills)==(220,2)
    for s in (0,1):
        b,_=fixture(s,True,False)
        n=400
        flags={i:np.zeros(n,dtype=bool) for i in (0,1)}
        trades=[[50,s,40 if s==0 else 60,220],[100,s,40 if s==0 else 60,600],[15000,s,40 if s==0 else 60,600],[15001,s,40 if s==0 else 60,600]]
        path=dict(frames=b.E.f,trades=trades,mark=50.)
        info=dict(end_rel=20000.)
        assert probe(path,info,{},flags,0,s)==440
        assert probe(dict(path,trades=[trades[0],trades[-1]]),info,{},flags,0,s)==0
        flags[s][:]=True
        assert probe(path,info,{},flags,0,s)==0
        b.ident.update(opening_ms=0.,start_abs_ms=210000.)
        model={str(i):dict(coef=[0,1,0,0],mean=[0]*4,scale=[1]*4) for i in (0,1)}
        p=Smooth(b,(flags,flags),{},'adaptive',model,np.ones(n))
        assert p.arm=='C' and p.schedule=='adaptive'
        p.active_side=s
        assert math.isclose(p.limit(0),curve(90,2 if s==0 else .5))
        b.ident['start_abs_ms']=270000.
        p.review(0,s)
        assert p.cap==0
        b.ident['start_abs_ms']=210000.
        p=Smooth(b,(flags,flags),{},'linear',model,np.ones(n))
        p.review(0,s)
        assert p.cap==522 and p.limit(0)==500
    print('Area, endpoints, side sign, buffer, reduce-only, fill conservation, zero fills, timing and FIFO close controls passed.')


if __name__=='__main__':
    main()
