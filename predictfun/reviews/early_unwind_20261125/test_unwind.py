"""Regression controls for deadline, signed room, outstanding risk and protective pressure."""
from early_unwind import EarlyUnwind, R, band
from test_policy import fixture


def main():
    for sec,expected in [(300,1000),(150.001,1000),(150,800),(120,600),(90,400),(60,200),(30.001,200),(30,0),(2,0)]:
        assert band(sec*1000)==expected
    for s in (0,1):
        b,old=fixture(s,True,False)
        b.ident.update(opening_ms=0.,start_abs_ms=270000.)
        p=EarlyUnwind(b,(old.on,old.trigger),{})
        p.opened=0
        b.k=2
        b.pre_tick(100)
        p.review(100,s)
        assert {r[1] for r in p.trims}=={0,1} and b.reserve(100)[1+s]==320
        b.pre_tick(250)
        b.k=5
        assert b.reserve(250)[1+s]==0
        v=b.side_view(250,s)
        assert p.slot(v,0,50)[0]=='WAIT'  # flat, reduce-only
        v['position']=-(1-2*s)*150
        assert p.slot(v,0,50)[:2]==('POST',150)
        v['A' if s else 'B']=100
        assert p.slot(v,0,50)[:2]==('POST',50)
        p.on[s][5]=True
        assert p.slot(v,0,50)[0]=='WAIT'
        p.on[s][5]=False
        b.ident['start_abs_ms']=240000.
        v['A' if s else 'B']=0
        v['position']=(1-2*s)*180
        # Plenty of forecast exits. The 200-share envelope independently allows only 20.
        p.model[R.groups(p.context(5,s))[0]]=dict(q25=5000.,markets=10)
        assert p.slot(v,0,50)[:2]==('POST',20)
        assert R.Policy.slot(p,v,0,50)[1]==220
    print('Clock edges, long/short, cancel reservation, flat, reducing, pressure and old-rule falsifier passed.')


if __name__=='__main__':
    main()
