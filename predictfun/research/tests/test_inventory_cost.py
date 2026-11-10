import math
import pytest
from predictfun.research import pf_inventory_cost_policy as p
from predictfun.research.pf_inventory_cost_engine import source


def order(side=0,size=5,kq=0,pt=50):
    return dict(side=side,size=size,kq=kq,pt=pt)


def test_soft_only_reducing_narrow_fresh_phase():
    assert not p.soft_flag('soft_exit',100,10,1,.045,.05,True)
    for mode,rel,net,side,v in [('late_reserved',100,10,1,.045),('soft_exit',59,10,1,.045),
        ('soft_exit',240,10,1,.045),('soft_exit',100,10,0,.045),('soft_exit',100,0,1,.045),
        ('soft_exit',100,10,1,None),('soft_exit',100,10,1,.039)]:
        assert p.soft_flag(mode,rel,net,side,v,.05,True)


def test_risk_quantity_premium_and_independent_sides():
    assert p.risk_quantity(0,0,25,[],.07,.05)==25
    a=order(0,25);b=order(1,25)
    expected=math.sqrt(625+1250)-25
    assert p.risk_quantity(0,0,25,[a,b],.07,.05)==pytest.approx(expected)
    assert p.risk_quantity(0,0,25,[a,a,b],.07,.05)==pytest.approx(expected)
    assert p.risk_quantity(10,1,10,[],.05,.05)==10
    assert p.risk_quantity(10,1,25,[],.05,.05)==20  # equal radius, not a no-flip promise
    assert p.risk_quantity(0,0,25,[],None,.05)==0


@pytest.mark.parametrize('bad',[float('nan'),-1.])
def test_invalid_quantity_refused(bad):
    with pytest.raises(ValueError):p.risk_quantity(0,0,bad,[],.07,.05)


def test_marginal_pending_reservation_and_size_retention():
    cur=order(0,25,kq=20)
    out=p.plan(0,[(50,5),(None,0)], [cur,None],[cur],'marginal_entry',20,[.07,.07],.05)
    assert out[0]==(50,25)
    pending=order(0,100)
    out=p.plan(0,[(50,25),(None,0)], [None,None],[pending],'marginal_entry',20,[.07,.07],.05)
    assert out[0][1]<25


def test_dust_and_reverse_refusal():
    out=p.plan(10,[(None,0),(98,10)],[None,None],[],'soft_exit',100,[.1,.1],.05)
    assert out[1]==(None,0)
    assert p.admission(0,1,10,[],'soft_exit',100,.1,.05,50,None,{})==0
    assert p.admission(10,1,25,[order(1,8)],'soft_exit',100,.1,.05,50,None,{})==2


def test_cost_cap_includes_unfilled_and_keeps_existing():
    state={'committed_concession':0.};meta=dict(price=51,base_price=50)
    assert p.price(-25,0,50,50,53,200,'budget_exit',state,None)==51
    q=p.admission(-25,0,25,[],'budget_exit',200,.1,.05,51,meta,state)
    p.sent('budget_exit',0,51,q,meta,state)
    assert state['committed_concession']==pytest.approx(.25)
    assert p.admission(-25,0,25,[],'budget_exit',200,.1,.05,51,meta,state)==0
    assert p.price(-25,0,50,50,53,200,'budget_exit',state,order(0,25,pt=51))==51
    assert p.price(-25,0,50,50,53,200,'budget_exit',state,None)==50
    with pytest.raises(ValueError):p.sent('budget_exit',0,51,1,meta,state)


def test_cost_never_crosses_or_improves_wrong_phase():
    st={'committed_concession':0.}
    for rel,net,b,a in [(179,-20,50,53),(240,-20,50,53),(200,20,50,53),(200,-20,50,51)]:
        assert p.price(net,0,50,b,a,rel,'budget_exit',st,None)==50


def test_audit_positive_and_bad_quote_refusal():
    frame=dict(b1_p=49,a1_p=51,b1_s=100,a1_s=100,b2_p=48,a2_p=52)
    args=(100,10,frame,[.1,.04],.05,[5,5],(False,False),[])
    r=p.feasibility(*args,[(None,0),(None,0)])
    assert r['first_reason']=='score_line'
    with pytest.raises(ValueError):p.feasibility(*args,[(None,0),(51,5)])
    r=p.feasibility(100,10,frame,[.1,.1],.05,[5,5],(False,False),[],[(None,0),(51,5)])
    assert r['first_reason']=='quoted'


def test_episode_duration_and_clock_refusal():
    e=p.Episodes();state={'first_reason':'quoted'}
    e.observe(1,60,state);e.observe(1,70,state);e.observe(1,100,{'first_reason':'stop'});e.finish()
    assert len(e.rows)==2 and e.rows[0]['decisions']==2
    assert sum(r['end']-r['start'] for r in e.rows)==240
    e.observe(2,80,state)
    with pytest.raises(ValueError):e.observe(2,79,state)


def test_generated_source_and_frozen_guard():
    s=source();compile(s,'study','exec')
    assert 'soft_flag(ENTRY_KIND' in s and 'cost_sent(ENTRY_KIND' in s
    with pytest.raises(ValueError):p.old.parent.replace(s,'missing-sentinel','bad')


def test_raw_notional_and_lot_submission_are_distinct():
    frame=dict(b1_p=50,a1_p=52,b1_s=100,a1_s=100,b2_p=49,a2_p=53)
    r=p.feasibility(100,2.084,frame,[.1,.1],.05,[5,5],(False,False),[],[(None,0),(52,2.084)])
    assert r['first_reason']=='quoted' and 'rounded_minimum' in r['flags']
    assert 'minimum_order' not in r['flags']
