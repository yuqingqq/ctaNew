"""Positive controls, deliberate corruptions, and independent artifact checks."""
from collections import Counter
import copy
import json
import pf_flow_control_common as F
import pf_flow_control_kernel as K
import pf_flow_control_paths as PATH
import pf_flow_control_policy as P
import pf_flow_control_lane as LANE
import pf_stop_queue_activation_lane as ORIGINAL
import pf_ev_lag_gap as CLOCK
C,np=F.C,F.np


def toy(trades,marks=(51.,51.),depth=20.):
    f=np.array([[-100.,-50.,50,49,48,47,46,depth,depth,depth,depth,depth,52,53,54,55,56,depth,depth,depth,depth,depth],
                [100.,120.,50,49,48,47,46,100,100,100,100,100,52,53,54,55,56,100,100,100,100,100]],float)
    return dict(frames=f,trades=np.asarray(trades,float).reshape(-1,4),bid=50.,marks=list(marks),
                base_weight=1.,classes=[5,0],phase=1,spread=2.,min_price=0.,max_price=100.)


def refused(fn,label):
    try:fn()
    except (ValueError,AssertionError):return True
    raise ValueError('known-bad control was accepted: '+label)


def selftest():
    checks={}
    o=np.array([[0,0,50,30,20,-100,1e30,-158,0]],float)
    p=toy([(20,0,50,25),(80,0,50,10),(150,0,50,10)])
    out,q=K.evaluate(K.pack([p],50,1000),o,[[1],[2]],0,51)
    C.require(np.allclose(q[0,:,0],[25.,5.]),'keep/cancel delay or queue replenishment')
    C.require(out[0,0,0]==25. and out[0,1,0]==5.,'shared endpoint cash valuation')
    checks['unavoidable_prefix_and_queue_preservation']=True
    n=o.copy();n[0,4:]=[0,58,1e30,0,1]
    x,qn=K.evaluate(K.pack([p],50,1000),n,[[1],[0]],0,51)
    C.require(qn[0,1,0]==0 and qn[0,0,0]==0,'new order activation/back of queue')
    checks['placement_latency_and_wait']=True
    shifted=n.copy();shifted[0,2]=60.
    early=toy([(80,0,50,25)],depth=100.)
    packed=K.pack([early],60,1000.)
    x,q0=K.evaluate(packed,shifted,[[1]],0,61)
    K.current_book(packed,[np.arange(60,55,-1),np.arange(62,67)], [np.zeros(5),np.zeros(5)])
    x,q1=K.evaluate(packed,shifted,[[1]],0,61)
    C.require(q0.sum()==0 and q1.sum()==25 and x[0,0,0]==25,'actual initial depth / translated price conditioning')
    checks['actual_initial_depth_changes_new_queue']=True
    sweep=toy([(100,0,49,40)],depth=0.)
    ladder=np.array([[0,0,50,30,0,-100,1e30,-158,0],[0,1,49,30,0,-100,1e30,-158,0]],float)
    x,q=K.evaluate(K.pack([sweep],50,1000),ladder,[[1,1]],0,51)
    C.require(np.allclose(q[0,0],[30.,10.]) and q.sum()==40.,'shared volume and price priority')
    checks['shared_volume_and_price_priority']=True
    noflow=toy([]);x,q=K.evaluate(K.pack([noflow],50,1000),o,[[1]],10,51)
    C.require(q.sum()==0 and x[0,0,0]==0,'zero flow')
    checks['no_flow']=True
    for sign in (-1,1):
        px=50 if sign==1 else 52
        order=np.array([[0 if sign==1 else 1,0,px,30,0,-100,1e30,-158,0]],float)
        price=49 if sign==1 else 53
        path=toy([(100,0 if sign==1 else 1,price,10)],marks=(48.,48.),depth=0.)
        x,q=K.evaluate(K.pack([path],50,1000),order,[[1]],0,51)
        val=q[0,0,0]*sign*(48.-px)
        C.require(q[0,0,0]>0 and np.isclose(x[0,0,0],val),'side sign')
    checks['bid_ask_value_sign']=True
    loss=toy([(100,0,49,10)],marks=(48.,48.),depth=0.)
    state=dict(orders=ladder[:1],info=[dict(side=0,rung=0,kind='cancel')],baseline=np.ones(1),candidates=[0],position=0.,existing=1)
    a,selected,_=P.choose(state,K.pack([loss],50,1000),np.ones(1),51,1000.)
    C.require(a[0]==2 and selected,'adverse path must choose cancellation')
    gain=toy([(100,0,49,10)],marks=(52.,52.),depth=0.)
    a,selected,_=P.choose(state,K.pack([gain],50,1000),np.ones(1),51,1000.)
    C.require(a[0]==1 and not selected,'good resting order should stay')
    state.update(orders=n,info=[dict(side=0,rung=0,kind='post')],baseline=np.zeros(1),existing=0)
    a,selected,_=P.choose(state,K.pack([gain],50,1000),np.ones(1),51,1000.)
    C.require(a[0]==1,'positive entry must post')
    a,selected,_=P.choose(state,K.pack([loss],50,1000),np.ones(1),51,1000.)
    C.require(a[0]==0,'negative entry must wait')
    state['position']=F.CAP
    a,selected,_=P.choose(state,K.pack([gain],50,1000),np.ones(1),51,1000.)
    C.require(a[0]==0,'inventory reservation must constrain entry')
    checks['economic_actions_and_inventory_cap']=True
    bad=copy.deepcopy(p);bad['trades'][0,3]=-1
    checks['negative_volume_refused']=refused(lambda:K.pack([bad],50,1000),'negative volume')
    badorder=o.copy();badorder[0,3]=np.nan
    checks['nonfinite_state_refused']=refused(lambda:K.evaluate(K.pack([p],50,1000),badorder,[[1]],0,51),'nonfinite state')
    ps=[toy([],marks=(51.,51.)),toy([],marks=(52.,52.))];ps[0]['classes']=[0,0];ps[1]['classes']=[6,6]
    probability=np.zeros((2,15));probability[:,0]=.2;probability[:,6]=.8
    w,d=PATH.weights(ps,probability)
    C.require(np.allclose(w,[.2,.8],atol=1e-6),'joint projection control')
    checks['forecast_changes_shared_weights']=True
    probability[0,0]=-1
    checks['invalid_probability_refused']=refused(lambda:PATH.weights(ps,probability),'negative probability')
    # The observer hook is the only change to the continuous execution lane.
    W=C.I.M.toy_lane_window([80.,150.,300.],[50,49,50],[130.,60.,20.]);nstep=W[0]['n'];aux={s:[False]*nstep for s in (0,1)}
    with CLOCK.clocks(58.,37.):
        a=ORIGINAL.sim_joint_g(copy.deepcopy(W),'M',0.,None,aux,size=30.,nr=6,mode='none',cooldown=0.)
        b=LANE.sim_joint_g(copy.deepcopy(W),'M',0.,None,aux,size=30.,nr=6,mode='none',cooldown=0.,on_step=lambda *args:None)
    for key in ['fl','stat','orders','cancels']:np.testing.assert_equal(a[key],b[key])
    checks['observer_lane_off_parity']=True
    checks['wrong_ledger_refused']=refused(lambda:np.testing.assert_equal(a['fl'],a['fl']+[(0,0,0,50,1)]),'corrupt ledger')
    checks.update(kernel_parity())
    checks.update(client_state_controls())
    return checks


def client_state_controls():
    W=C.I.M.toy_lane_window([],[],[],n=40)
    for s,w in W.items():
        w['ws']=0.;w['kf']=w['tf']+10.;w['jk']=np.searchsorted(w['kf'],w['tg'],side='right')-1
    order=dict(pt=50.,ts=0.,ta=58.,tc=1e30,tr=1e30,ref=False,rung=0,fl=[(100.,5.),(180.,7.)],rem=18.,ahead=3.)
    ST={s:dict(cl=[],cur=[None]*6,orders=[[] for _ in range(6)]) for s in (0,1)}
    ST[0]['cl']=[order];ST[0]['cur'][0]=order;ST[0]['orders'][0]=[order]
    before=P.snapshot(W,5,250.,ST,5.)
    altered=copy.deepcopy(ST);a=altered[0]['cl'][0];a['rem']=1.;a['ahead']=999.;a['fl'].append((249.,2.))
    after=P.snapshot(W,5,250.,altered,5.)
    np.testing.assert_equal(before['orders'],after['orders'])
    C.require(before['orders'][0,3]==25.,'fill notification boundary')
    C.require(before['orders'][0,4]==0.,'known partial fill must retain front queue position')
    wrong_queue=before['orders'].copy();wrong_queue[0,4]=100.
    refused(lambda:np.testing.assert_equal(before['orders'],wrong_queue),'partial fill pushed behind displayed queue')
    altered=copy.deepcopy(ST);altered[0]['cl'][0]['fl'].append((150.,2.))
    changed=P.snapshot(W,5,250.,altered,7.)
    C.require(changed['orders'][0,3]==23.,'received fill positive control')
    mutant=before['orders'].copy();mutant[0,3]=order['rem']
    refused(lambda:np.testing.assert_equal(before['orders'],mutant),'true remainder leakage')
    return dict(unreceived_fill_and_hidden_queue_ignored=True,received_fill_changes_state=True,true_remainder_leak_refused=True,
                confirmed_partial_fill_front_preserved=True,known_bad_replenished_ahead_refused=True)


def kernel_parity():
    rng=np.random.default_rng(20261116);runs=0;positive=0
    for trial in range(24):
        tt=np.sort(rng.uniform(1,1800,24));pp=rng.integers(45,55,24);qq=rng.uniform(1,100,24)
        W=C.I.M.toy_lane_window(tt,pp,qq,n=40);tf=W[0]['tf'];move=rng.integers(-3,4,len(tf));move[tf<=0]=0
        for s,w in W.items():
            w['kf']=tf+10.;w['jk']=np.searchsorted(w['kf'],w['tg'],side='right')-1
            w['P5']=w['P5']+move[:,None];w['S5']=rng.uniform(1,150,(len(tf),5))
        for s,w in W.items():
            w['opp']=W[1-s]['P5'][:,0];w['best']=w['P5'][w['jk'],0].tolist();w['bopp']=w['opp'][w['jk']].tolist()
        W[1]['anc']=tt.copy();W[1]['ptl']=(100-pp).tolist();W[1]['shl']=qq[::-1].tolist()
        aux={s:[False]*40 for s in (0,1)};gate={(s,d):lambda k:k!=0 for s in (0,1) for d in range(6)}
        with CLOCK.clocks(58.,37.):r=ORIGINAL.sim_joint_g(copy.deepcopy(W),'M',0.,None,aux,gate=gate,size=30.,nr=6,mode='none',cooldown=0.)
        j=int(np.searchsorted(tf+10,0,side='right')-1)
        frames=np.column_stack([tf[j:],tf[j:]+10,W[0]['P5'][j:],W[0]['S5'][j:],W[1]['P5'][j:],W[1]['S5'][j:]])
        tape=np.array(sorted([(t,s,p,q) for s,w in W.items() for t,p,q in zip(w['anc'],w['ptl'],w['shl'])],key=lambda z:(z[0],z[1])))
        terminal=(W[0]['P5'][np.searchsorted(tf,2000,side='right')-1,0]+W[1]['P5'][np.searchsorted(tf,2000,side='right')-1,0])/2
        path=dict(frames=frames,trades=tape,bid=50.,marks=[51.,terminal])
        orders=np.array([[s,d,50-d if s==0 else 52+d,30,0,58,1e30,0,1] for s in (0,1) for d in range(6)],float)
        out,q=K.evaluate(K.pack([path],50,2000),orders,np.ones((1,12)),0,51,2000.)
        actual=np.array([sum(z[2] for z in r['fills'][s,d]) for s in (0,1) for d in range(6)])
        C.require(np.allclose(actual,q[0,0],atol=1e-7),'compiled / canonical branch fill disagreement')
        cash=-sum((1-2*s)*p*q for s,d,t,p,q in r['fl'])
        C.require(np.isclose(out[0,0,0],cash+r['endpos']*terminal),'compiled/canonical wealth mismatch')
        runs+=1;positive+=int(actual.sum()>0)
    C.require(positive>0,'parity fixture has no fills')
    return dict(compiled_canonical_paths=runs,compiled_canonical_nonempty=positive)


def audit(directory):
    from pathlib import Path
    import pandas as pd
    out=Path(directory);r=json.loads((out/'receipt.json').read_text());F.verify()
    for name,h in r['outputs'].items():C.require(C.sha(out/name)==h,'changed result '+name)
    f=pd.read_parquet(out/'fills.parquet');markets=pd.read_parquet(out/'markets.parquet')
    for row in markets.itertuples():
        x=f[(f.m==row.m)&(f.case==row.case)]
        C.require(np.isclose(x.shares.sum(),row.shares) and np.isclose(x.terminal_pnl_usd.sum(),row.terminal_pnl_usd), 'artifact fills/summary reconciliation')
    bad=markets.copy();bad.loc[bad.index[0],'shares']+=1
    row=bad.iloc[0];x=f[(f.m==row.m)&(f.case==row.case)]
    C.require(not np.isclose(x.shares.sum(),row.shares),'summary-corruption instrument did not fire')
    return dict(asof=C.stamp(),controls=selftest(),markets=len(markets),fills=len(f),summary_corruption_detected=True)
