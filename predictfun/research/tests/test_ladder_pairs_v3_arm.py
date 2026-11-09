"""Exact named v3 arm checks on wholly synthetic full-window inputs."""
import json
import numpy as np
import pytest
from predictfun.research.tests.test_ladder_pairs import pair_engine
from predictfun.research.tests.test_simulation_causality import engine,frame,leg,run

RULES=set('G4+G2+G6+G5+G3+G1+PM90+GS+LD2+CAP10+XNA+XMIN1+XV+RO60+S240'.split('+'))
SIZE='AM:5:25:95'

def exact_case(side=0,kind='no_trade',paired=True,frozen=False,size=SIZE):
    ns=engine('pf_step4_ladder_v3.py') if frozen else pair_engine()
    ns['LINES'].update({'90':[.0492,.0623,.0789,.0917],'95':[.0698,.0841,.1009,.1094]})
    ns['KB']=np.arange(-2000.,302001.,50.);ns['MB']=np.full(len(ns['KB']),100.);ns['BC']=np.zeros(len(ns['KB']),dtype=np.int8)
    ns['BCV']={};ns['_FLIP_CACHE']={}
    times=np.arange(0.,300001.,50.)
    entry=np.where(times<100,.055,np.where(times<200,.1,np.where(times<1000,.055,-1.)))
    reducing=np.where(times<59000,-1.,.2)
    ns['PREDVB']={1:(times,entry if side==0 else reducing)};ns['PREDVA']={1:(times,reducing if side==0 else entry)}
    f=frame(-1,50,52,100.,100.);f.update(b2_p=49,a2_p=53,b2_s=100.,a2_s=100.)
    if side:
        for i in range(1,6):
            bp,ap=f[f'b{i}_p'],f[f'a{i}_p'];f[f'b{i}_p'],f[f'a{i}_p']=(100-ap if ap>=0 else -1),(100-bp if bp>=0 else -1)
    trade_price=lambda s:45 if s==0 else 55
    if kind=='no_trade':trades=[]
    elif kind=='reduce_fill':trades=[leg(250,side,trade_price(side),5),leg(60050,1-side,trade_price(1-side),2),leg(60600,1-side,trade_price(1-side),3)]
    elif kind=='cap_fill':trades=[leg(150,side,trade_price(side),25)]
    else:raise ValueError(kind)
    trace=[]
    if paired:ns['_LZ_TRACE']=trace.append
    result=run(ns,trades,[f],[0.],RULES|({'LZ'} if paired else set()),size,policy='MICRO')
    return result,trace

@pytest.mark.parametrize('side',[0,1])
@pytest.mark.parametrize('kind',['no_trade','reduce_fill','cap_fill'])
def test_exact_v3_no_token_full_output_parity(side,kind):
    new,_=exact_case(side,kind,paired=False);old,_=exact_case(side,kind,paired=False,frozen=True)
    assert json.dumps(new,sort_keys=True)==json.dumps(old,sort_keys=True)

@pytest.mark.parametrize('side',[0,1])
@pytest.mark.parametrize('size',[5.,25.])
def test_exact_v3_fixed_size_without_resize_or_rejection(side,size):
    new,_=exact_case(side,size=size);old,_=exact_case(side,paired=False,size=size)
    assert json.dumps(new,sort_keys=True)==json.dumps(old,sort_keys=True)

@pytest.mark.parametrize('side',[0,1])
def test_exact_v3_variable_size_positive(side):
    new,trace=exact_case(side,'reduce_fill');old,_=exact_case(side,'reduce_fill',paired=False)
    assert new['live_ctr']['lz_resize']>0 and new['requests']!=old['requests']
    initial=[e for e in trace if e['kind']=='place' and e['lane']%4==side and e['t']<250]
    assert [(e['t'],e['size']) for e in initial]==[(0.,5.),(100.,20.)]
    assert not [e for e in trace if e['kind']=='cancel' and e['lane']==1-side and e['t']==60150]

@pytest.mark.parametrize('side',[0,1])
@pytest.mark.parametrize('kind',['reduce_fill','cap_fill'])
def test_exact_v3_inventory_guards_and_ro_boundary(side,kind):
    new,trace=exact_case(side,kind)
    if kind=='cap_fill':
        assert new['withheld']['CAPN']>0 and new['withheld']['LD_CAPN']>0
        assert not [e for e in trace if e['kind']=='place' and e['lane']%2==side and e['t']>=250]
    else:
        assert new['net_end']==0
        assert not [e for e in trace if e['kind']=='place' and e['t']>=60000 and e['lane']%4>=2]
        assert not [e for e in trace if e['kind']=='place' and e['t']>=60000 and e['lane']%2==side]
