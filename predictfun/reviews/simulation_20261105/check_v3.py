"""Synthetic follow-up: execute source functions only, never load research data."""
import ast, bisect, collections, heapq, itertools, math, json, hashlib, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

ROOT=Path('/home/yuqing/ctaNew/predictfun/research')
def load(file):
    src=(ROOT/file).read_text(); tree=ast.parse(src)
    ns=dict(np=np,pd=pd,bisect=bisect,collections=collections,heapq=heapq,itertools=itertools,
        math=math,SIZE=5.,V2=True,SPLIT=True,L_O=22.,L_FILL=100.,L_PL=40.,L_CA=22.,L_ANS=44.,
        LEGACY_INTENT=False,QACT=False,G2V={},PRED={},PRED250={},PREDVB={},PREDVA={},
        FILLPRIO='price',OWNCROSS='on',LEGORDER='price',REJRESET='on',CK=SimpleNamespace(L_PENDCANCEL=95.),
        LINES={'zero':[0.]*4,'inf':[float('inf')]*4,'test':[0.]*4})
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),str(ROOT/file),'exec'),ns)
    ns['reg']=pd.DataFrame([dict(mid=1,ws=0.,outcome='Up')]).set_index('mid')
    return ns,hashlib.sha256(src.encode()).hexdigest()
def fr(t,b=50,a=51,bs=100.,az=100.):
    f=dict(t_ms=float(t))
    for s in 'ba':
        for i in range(1,6):
            f[f'{s}{i}_p']=-1; f[f'{s}{i}_s']=0.
    f.update(b1_p=b,b1_s=bs,a1_p=a,a1_s=az)
    return f
def leg(t,side,pt,sh):
    return dict(t_ms=float(t),side=side,ptick=pt,shares=float(sh),how=1,exec_s=0)
def run(ns,trades,F=None,kf=None,rules=None,size=10.,policy='JOIN'):
    ns['LT_BY']={1:pd.DataFrame(trades)} if trades else {}
    return ns['run_window'](1,F or [fr(-1)],kf or [0.],rules or {'XNA','G5','S1'},policy,'mid',size)
out={}
for file in ('pf_step4_live.py','pf_step4_ladder.py'):
    ns,sha=load(file); r={"sha256":sha}
    # Original priority counterexample is now fixed, and its regression switch must reveal it.
    F=[fr(-1,49,52,0.,0.),fr(10,50,52,0.,0.)]
    fixed=run(ns,[leg(60,0,48,5)],F,[0.,10.],size=5.)
    assert fixed['fills'][0][3]==50
    ns['FILLPRIO']='old'; old=run(ns,[leg(60,0,48,5)],F,[0.,10.],size=5.)
    assert old['fills'][0][3]==49; ns['FILLPRIO']='price'
    r['earlier_price_priority_fix']={'fixed_pt':50,'regression_pt':49}
    # One atomic sell: 100 historical shares at 50 then 5 at 49; our bid 10 has 100 ahead.
    # Only the remaining 5 can reach us. Input rows explicitly list better price first.
    sweep=run(ns,[leg(60,0,50,100),leg(60,0,49,5)])
    bid_q=sum(x[4] for x in sweep['fills'] if x[2]==0)
    split_time=run(ns,[leg(59,0,50,100),leg(60,0,49,5)])
    split_q=sum(x[4] for x in split_time['fills'] if x[2]==0)
    aggregate=run(ns,[leg(60,0,50,105)])
    agg_q=sum(x[4] for x in aggregate['fills'] if x[2]==0)
    ask=run(ns,[leg(60,1,51,100),leg(60,1,52,5)])
    ask_q=sum(x[4] for x in ask['fills'] if x[2]==1)
    assert bid_q==split_q==agg_q==ask_q==5,(bid_q,split_q,agg_q,ask_q)
    ns['LEGORDER']='old'; bad_sweep=run(ns,[leg(60,0,50,100),leg(60,0,49,5)]); ns['LEGORDER']='price'
    assert sum(x[4] for x in bad_sweep['fills'] if x[2]==0)==10
    r['atomic_sweep']={'observed_bid_fill':bid_q,'expected_bid_fill':5.,'better_leg_earlier_control':split_q,
        'single_aggregate_leg_control':agg_q,'mirrored_ask_control':ask_q,'fills':sweep['fills']}
    # A venue post-only refusal leaves the rejected intent installed. At the next decision,
    # recovered book state wants that identical price, but the engine never re-sends it.
    # Equal knowledge times are supported (the production caller uses maximum.accumulate).
    F=[fr(-1,50,52,0.,0.),fr(30,49,50,0.,0.),fr(60,50,52,0.,0.)]
    rejected=run(ns,[leg(160,0,49,5)],F,[0.,100.,100.],size=5.)
    Fctrl=[fr(-1,50,52,0.,0.),fr(30,49,51,0.,0.),fr(60,50,52,0.,0.)]
    accepted=run(ns,[leg(160,0,49,5)],Fctrl,[0.,100.,100.],size=5.)
    assert rejected['live_ctr'].get('postonly_refused',0)==1
    assert rejected['n_fills']==accepted['n_fills']==1
    bid_places=[q for q in rejected['requests'] if q[1]==0 and q[2]=='place']
    assert len(bid_places)>=2 and bid_places[0][0]==0. and bid_places[1][0]==100.,bid_places
    ns['REJRESET']='off'; bad_retry=run(ns,[leg(160,0,49,5)],F,[0.,100.,100.],size=5.); ns['REJRESET']='on'
    assert bad_retry['n_fills']==0
    r['postonly_retry']={'refusals':1,'bid_places':bid_places,'fills_after_recovery':1,'no_refusal_control_fills':1,'off_control_fills':0}
    # Fresh negative guard blocks; GS also blocks missing/stale coverage.
    ns['PREDVB']={1:(np.array([0.,10.]),np.array([-1.,-1.]))}
    ns['PREDVA']={1:(np.array([0.,10.]),np.array([-1.,-1.]))}
    stale=run(ns,[leg(160,0,48,5)],[fr(-1,50,52,0.,0.)],[0.],{'XNA','G5','S1','V0','GS'},5.)
    assert stale['n_fills']==0
    ns['PREDVB']={}; ns['PREDVA']={}
    missing=run(ns,[leg(60,0,48,5)],[fr(-1,50,52,0.,0.)],[0.],{'XNA','G5','S1','V0','GS'},5.)
    assert missing['n_fills']==0
    r['GS_fix']={'missing_fills':0,'stale_fills':0}
    # Prior own-cross witness: on rejects it; off is a positive control for the instrument.
    ns['PREDVB']={1:(np.array([0.,10.]),np.array([1.,1.]))}
    ns['PREDVA']={1:(np.array([0.,10.]),np.array([1.,1.]))}
    if file=='pf_step4_ladder.py':
        F=[fr(-1,49,51,0.,0.),fr(10,50,52,0.,0.)]
        crossed=run(ns,[],F,[0.,10.],{'XNA','G5','S1','IMtest'},5.)
        assert crossed['live_ctr'].get('postonly_refused_own',0)>0 and crossed['live_ctr'].get('own_cross',0)==0
        ns['OWNCROSS']='off'; bad=run(ns,[],F,[0.,10.],{'XNA','G5','S1','IMtest'},5.); ns['OWNCROSS']='on'
        assert bad['live_ctr'].get('own_cross',0)>0
        r['earlier_own_cross_fix']={'fixed_cross':0,'refusals':crossed['live_ctr']['postonly_refused_own'],'off_control_crosses':bad['live_ctr']['own_cross']}
        # A public frame stamped 85 is received at 100; our rung fills at 90 and is not
        # reported until 190. Identical public inputs must produce identical actions before 190.
        F=[fr(-100,50,52,10.,10.),fr(45,49,52,10.,10.),fr(85,49,52,10.,10.)]
        kf=[0.,60.,100.]
        ns['PREDVB']={1:(np.array([0.,60.,100.,150.]),np.array([1.,1.,1.,1.]))}
        ns['PREDVA']={1:(np.array([0.,60.,100.,150.]),np.array([-1.,-1.,-1.,-1.]))}
        rules={'XNA','G5','S1','V0','GS','LD2'}
        filled=run(ns,[leg(90,0,49,5)],F,kf,rules,5.)
        no_fill=run(ns,[],F,kf,rules,5.)
        def early_places(result):
            return [q for q in result['requests'] if 90.<q[0]<190. and q[2]=='place']
        fp=early_places(filled); cp=early_places(no_fill)
        assert filled['fills'][0][1:5]==(0.09,0,49,5.),filled['fills']
        assert fp and not cp,(fp,cp)
        ns['L_FILL']=1000.
        slower=run(ns,[leg(90,0,49,5)],F,kf,rules,5.); ns['L_FILL']=100.
        sp=early_places(slower); assert sp==fp
        r['fill_knowledge_leak']={'actual_fill_ms':90.,'configured_fill_known_ms':190.,
            'frame_causing_decision_stamped_ms':85.,'new_places_before_fill_known':fp,
            'identical_public_inputs_no_fill_control_places':cp,
            'L_FILL_1000_control_early_places':sp,'all_requests':filled['requests']}
        # Repeat with the active MICRO/AV arm's full guard stack; Binance is flat.
        ns['KB']=np.arange(-2000.,4000.,100.); ns['MB']=np.full(60,100.); ns['BC']=np.zeros(60,dtype=np.int8)
        ns['BCV']={}; ns['_FLIP_CACHE']={}
        ns['PREDVB']={1:(np.array([0.,60.,100.,150.]),np.full(4,.01))}
        for f in F:
            f.update(b2_p=f['b1_p']-1,b2_s=10.,a2_p=f['a1_p']+1,a2_s=10.)
        fullrules={'G1','G2','G3','G4','G5','G6','V0','GS','LD2','XNA','XMIN1','XV','S60'}
        ff=run(ns,[leg(90,0,49,15)],F,kf,fullrules,'AV:5:25:0.06','MICRO')
        fc=run(ns,[],F,kf,fullrules,'AV:5:25:0.06','MICRO')
        assert early_places(ff) and not early_places(fc),(ff['requests'],fc['requests'])
        r['full_stack_fill_knowledge_leak']={'new_places_before_fill_known':early_places(ff),
            'no_fill_control_places':early_places(fc),'fills':ff['fills']}
    # Unsupported split/amend input must still be refused.
    try: run(ns,[],rules={'G5','S1'})
    except SystemExit: r['known_bad_input_refused']=True
    else: raise AssertionError('unsupported split/amend accepted')
    out[file]=r
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
