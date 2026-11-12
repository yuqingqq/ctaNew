"""Synthetic engine forks and falsifiers; no market tapes are read."""
import ast
import copy
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from predictfun.research.pf_continuation_engine import source,verify_transform
from predictfun.research.pf_continuation_build import install,equal_fills
from predictfun.research.tests.test_simulation_causality import engine,leg
from predictfun.research.pf_model_spec import SETS


def scenario(cancel=22.,queue='M'):
    _,Manager=install();manager=Manager.__new__(Manager)
    ts=np.arange(0.,50001.,50.);x=np.zeros((len(ts),len(SETS['ALL'])))
    x[:,SETS['ALL'].index('bn_ntr1000')]=10
    manager.inputs=SimpleNamespace(by={1:(ts,x)})
    manager.day='20261023';manager.phase='build';manager.timing='X';manager.models={}
    manager.paired=[];manager.population=[];manager.reconciled=0;manager.windows=[]
    manager.save=lambda *args:None
    original=manager.window;hooks=[]
    def window(*args):
        h=original(*args);hooks.append(h)
        h.selected=lambda r:r['generation']==1
        return h
    manager.window=window
    frames=[]
    for t in ts:
        bid=49. if t<5400 else 50.
        f={'t_ms':t}
        for k in range(1,6):f.update({f'b{k}_p':bid+1-k,f'a{k}_p':53.+k-1,f'b{k}_s':0.,f'a{k}_s':0.})
        frames.append(f)
    ns=engine('pf_step4_ladder_v3.py');tree=ast.parse(source(queue))
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'fork_test','exec'),ns)
    ns.update(_STUDY=manager,KB=ts,MB=np.ones(len(ts))*100,BC=np.zeros(len(ts)),BCV={},_FLIP_CACHE={},L_CA=cancel)
    for k in ('PREDVB','PREDVA'):ns[k]={1:(ts,np.ones(len(ts))*.01)}
    ns['LT_BY']={1:pd.DataFrame([leg(5160,0,49,2),leg(5200,0,49,3),leg(6500,0,50,5),leg(11000,1,53,5),leg(15000,0,50,5)])}
    with patch('predictfun.research.pf_default_pull_hooks.probe_age',return_value=100.):
        result=ns['run_window'](1,frames,ts,{'XNA','LD2','S240','PMzero'},'JOIN','mid',5.,dict(realised=0.,halted=False))
    return manager,hooks,result


def test_transform_and_aliases():
    verify_transform()
    o={'size':5,'kq':0};bundle={'orders':{0:o},'known':{0:o},'notice':(100,o),'cancel_wait':{id(o):(o,122)}}
    z=copy.deepcopy(bundle);z['cancel_wait']={id(x):(x,t) for x,t in z['cancel_wait'].values()}
    assert z['orders'][0] is z['known'][0] is z['notice'][1]
    assert id(z['orders'][0]) in z['cancel_wait'] and id(o) not in z['cancel_wait']
    z['known'][0]['kq']=2;assert o['kq']==0 and z['notice'][1]['kq']==2
    bad=copy.deepcopy(bundle);assert id(bad['orders'][0]) not in bad['cancel_wait']


def test_same_engine_forks_and_current_touch():
    manager,hooks,result=scenario()
    assert manager.reconciled==1 and len(manager.paired)==1
    a=manager.paired[0]
    assert a['decision']==5150 and a['order_price']==49
    assert a['pull_first_place_delay']>=1000 and a['pull_first_place_price']==50
    assert a['keep_30000_q']>a['keep_2000_q']
    assert all(a['keep_reconciled_'+str(h)] for h in (2000,10000,30000))
    assert result['n_fills']>0


def test_impossible_cancel_latency_is_caught():
    with pytest.raises(ValueError,match='KEEP fork'):
        scenario(cancel=0.)


def test_fill_reconciliation_refuses_corruption():
    a=[(1.,0,49.,2.,'at',True,5.,False,0,0.,100.)]
    equal_fills(a,a)
    b=[tuple([1.,0,49.,3.,'at',True,5.,False,0,0.,100.])]
    with pytest.raises(ValueError):equal_fills(a,b)


def test_joint_profiles_calibration_and_refusals():
    from predictfun.research import pf_continuation_model as M
    manager,_,_=scenario();a=dict(manager.paired[0]);b=dict(a);b['id']='zero'
    for branch in ('keep','pull'):
        for h in (2000,10000,30000):
            for k in ('q','q2','same_q','gain','loss','risk','value','net2'):b[f'{branch}_{h}_{k}']=0.
        for h in (250,1000,2000,10000,30000):b[f'{branch}_first_{h}']=0.
    frame=pd.DataFrame([a,b]);event,v,prior=M.tensor(frame)
    j=np.repeat(prior[None],100,axis=0);M.support(j)
    wanted=np.array([0]*70+[int(event[0])]*30)
    fit=M.fit_calibration(j,wanted,prior);assert fit['status']=='PASS'
    assert not np.allclose(fit['weights'],1.)
    z=M.reweight(j,fit['weights']);out=M.flatten(z)
    assert (out['keep_first_30000']>=out['keep_first_2000']).all()
    table=M.tables(np.zeros((2,2),int),event,v,prior)
    np.testing.assert_allclose(table[0,0],prior)
    bad=j.copy();bad[:,:,M.IX['keep_2000_q']]=0.
    with pytest.raises(ValueError):M.support(bad)
    with pytest.raises(ValueError):M.reweight(j,[1,-1,1,1])
    with pytest.raises(ValueError):M.tensor(pd.concat([frame,frame]))
