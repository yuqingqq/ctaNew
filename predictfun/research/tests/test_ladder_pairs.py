"""Synthetic B4 implementation checks. No tapes, future days or P&L study."""
import ast
import copy
import json
import numpy as np
import pytest
from predictfun.research.pf_ladder_pairs import PairMapper,OFF
from predictfun.research import pf_ladder_pairs_build as b
from predictfun.research.tests.test_simulation_causality import engine,neutral_inputs,frame,leg,run

def pair_engine(trace=None):
    source=b.generate()
    assert b.DST.read_text()==source
    ns=engine('pf_step4_ladder_v3.py');ns['PairMapper']=PairMapper
    tree=ast.parse(source)
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),str(b.DST),'exec'),ns)
    ns['LINES'].update({'90':[.0492,.0623,.0789,.0917],'95':[.0698,.0841,.1009,.1094]})
    if trace is not None:ns['_LZ_TRACE']=trace.append
    return ns

def scenario(side=0,changes=((0,5),(100,25),(200,5)),trades=(),extra=(),paired=True,frames=None,known_times=None,trace=None,size='AV:5:25:0.06',frozen=False,delay=100.):
    ns=engine('pf_step4_ladder_v3.py') if frozen else pair_engine(trace)
    neutral_inputs(ns,side);ns['L_FILL']=delay
    ns['LINES'].update({'90':[.0492,.0623,.0789,.0917],'95':[.0698,.0841,.1009,.1094]})
    times=np.unique(np.r_[np.arange(0.,2101.,50.),[t for t,q in changes]])
    values=[-1. if (q:=[z for t0,z in changes if t0<=t][-1])==0 else .055 if q==5 else .1 for t in times]
    ns['PREDVB' if side==0 else 'PREDVA']={1:(times,np.array(values))}
    rules={'XNA','G5','V0','GS'}|set(extra)
    if not any(r.startswith('S') and r[1:].isdigit() for r in rules):rules.add('S1')
    if paired:rules.add('LZ')
    f=frame(-1,50,52,0.,0.);f.update(b2_p=49,a2_p=53,b2_s=0.,a2_s=0.)
    return run(ns,trades,frames or [f],known_times or [0.],rules,size)

def order(lane,qty,remaining=None,tick=50,sent=0):
    return dict(lane=lane,pt=tick,size=qty,remaining=qty if remaining is None else remaining,t_send=sent)

def wants(p,z,current=None,pending=None,intent=None,tick=50):
    return p.wants(((tick,z),OFF,OFF,OFF),current or {},pending or {},intent or {})

def test_frozen_and_generated_source_identity_falsifier():
    assert b.generate()==b.DST.read_text()
    with pytest.raises(ValueError,match='hash mismatch'):b.generate(b.SRC.read_text()+'\n')

@pytest.mark.parametrize('side',[0,1])
@pytest.mark.parametrize('extra',[set(),{'LD2','G6','CAP10'}, {'LD2','RO1','S2','XV','XMIN1'}])
def test_no_token_exact_frozen_outputs(side,extra):
    args=dict(side=side,extra=extra,trades=[leg(60,side,49 if side==0 else 53,2),leg(160,side,48 if side==0 else 54,23)],paired=False)
    assert json.dumps(scenario(**args),sort_keys=True)==json.dumps(scenario(**args,frozen=True),sort_keys=True)

@pytest.mark.parametrize('side',[0,1])
@pytest.mark.parametrize('size',[5.,25.])
@pytest.mark.parametrize('extra',[set(),{'LD2','XV','XMIN1'}])
def test_fixed_size_exact_control(side,size,extra):
    args=dict(side=side,size=size,changes=((0,25),),extra=extra,trades=[leg(60,side,49 if side==0 else 53,2),leg(400,side,48 if side==0 else 54,100)])
    assert json.dumps(scenario(**args),sort_keys=True)==json.dumps(scenario(**args,paired=False),sort_keys=True)

@pytest.mark.parametrize('side',[0,1])
def test_size_growth_keeps_front_and_variable_arm_falsifier(side):
    trace=[];args=dict(side=side,extra={'LD2','XV','XMIN1','PM90'},size='AM:5:25:95',trades=[leg(250,side,49 if side==0 else 53,5)])
    result=scenario(**args,trace=trace);original=scenario(**args,paired=False)
    assert result['live_ctr']['lz_resize']>0
    assert result['requests']!=original['requests']
    f=[x for x in result['fills'] if x[1]==.25][0]
    assert f[11]==5 and f[15]==210
    assert [x['size'] for x in trace if x['kind']=='place' and x['lane']%4==side and x['t']<250]==[5,20]

@pytest.mark.parametrize('side',[0,1])
def test_partial_member_fill_is_not_replenished(side):
    tr=[];scenario(side,changes=((0,5),(100,25)),trades=[leg(150,side,49 if side==0 else 53,2),leg(180,side,49 if side==0 else 53,3)],trace=tr)
    assert [r['size'] for r in tr if r['kind']=='place' and r['t']<500]==[5,20]
    assert [r['t'] for r in tr if r['kind']=='fill']==[250,280]

@pytest.mark.parametrize('side',[0,1])
def test_whole_pair_fill_reposts_once(side):
    tr=[];scenario(side,changes=((0,5),(100,25)),trades=[leg(150,side,49 if side==0 else 53,25)],trace=tr)
    assert [(r['t'],r['size']) for r in tr if r['kind']=='place' and r['t']<500]==[(0,5),(100,20),(250,25)]

@pytest.mark.parametrize('side',[0,1])
def test_pending_create_counts_and_replace_is_concurrent(side):
    tr=[];scenario(side,changes=((0,5),(10,25),(20,5),(30,25)),trace=tr)
    assert [(r['t'],r['size']) for r in tr if r['kind']=='place' and r['t']<200]==[(0,5),(10,20),(30,20)]
    tr=[];scenario(side,changes=((0,25),(100,5)),trace=tr)
    assert [(r['t'],r['size']) for r in tr if r['kind']=='place' and r['t']<200]==[(0,25),(100,5)]

@pytest.mark.parametrize('side',[0,1])
def test_cancel_pending_secondary_does_not_jump_old_front(side):
    tr=[];r=scenario(side,changes=((0,5),(100,25),(150,5)),trades=[leg(160,side,49 if side==0 else 53,5)],trace=tr)
    fill=r['fills'][0];assert fill[11]==5 and fill[15]==120 and not fill[12]

def test_notification_boundary_and_cancelled_late_fill():
    p=PairMapper(1.);base=order(0,5)
    assert wants(p,5)[0]==(50,5)
    p.notify(0,50,2)
    assert wants(p,5,{0:order(0,5,3)})[0]==(50,5)
    assert wants(p,25,{0:order(0,5,3)})[4]==(50,22)
    p.notify(0,50,3,ended=True)
    assert p.filled[0]==0

def test_front_can_rotate_and_pending_partner_keeps_pair_alive():
    p=PairMapper(1.);wants(p,25)
    p.notify(0,50,5)
    cur={4:order(4,20,sent=100)}
    out=wants(p,25,cur);assert out[0]==OFF and out[4]==(50,20)
    assert wants(p,30,cur)[0]==(50,10)
    p=PairMapper(1.);wants(p,25);p.notify(0,50,5)
    pending={4:dict(key=(50,20))};intent={4:(50,20)}
    assert p.active(4,50,{},pending,intent)
    assert wants(p,25,{},pending,intent)[0]==(50,20)
    assert not p.active(4,50,{},pending,{4:OFF})

def test_minimum_floor_and_subshare_remainder():
    p=PairMapper(1.)
    assert wants(p,1.999,tick=50)[0]==OFF
    assert wants(p,2.,tick=50)[0]==(50,2.)
    p=PairMapper();wants(p,5)
    out=wants(p,5,{0:order(0,5,4.5)});assert out[0]==(50,5) and out[4]==OFF and p.last_withheld==1

@pytest.mark.parametrize('qty',[float('nan'),-1,float('inf')])
def test_invalid_target_refused(qty):
    with pytest.raises(ValueError):wants(PairMapper(),qty)

def test_trace_is_read_only_and_unsupported_mode_refused():
    tr=[];a=scenario(trace=tr);b0=scenario();assert json.dumps(a,sort_keys=True)==json.dumps(b0,sort_keys=True) and tr
    ns=pair_engine();neutral_inputs(ns);ns['POLICYVIEW']='physical'
    with pytest.raises(SystemExit,match='LZ requires'):run(ns,rules={'LZ','XNA','G5','S1'})

def test_ro_quantity_shrink_keeps_partly_filled_whole_order():
    ns=pair_engine();neutral_inputs(ns);ns['L_FILL']=10.
    times=np.arange(0.,2101.,50.)
    ns['PREDVB']={1:(times,np.where(times<200,.1,-1.))}
    ns['PREDVA']={1:(times,np.where(times<900,-1.,.1))}
    tr=[];ns['_LZ_TRACE']=tr.append
    run(ns,[leg(60,0,49,20),leg(1050,1,53,2)],[frame(-1,50,52,0.,0.)],[0.],{'LZ','XNA','G5','S2','RO1','V0','GS'},'AV:5:25:0.06')
    assert not [r for r in tr if r['kind']=='cancel' and r['lane']==1 and 1060<=r['t']<1100]

@pytest.mark.parametrize('side',[0,1])
def test_rejection_releases_unanswered_slot_before_answer_timer(side):
    frames=[frame(-1,50,52,0.,0.),frame(24,51,53,0.,0.),frame(39,48,50,0.,0.)]
    for f in frames:
        f.update(b2_p=f['b1_p']-1,a2_p=f['a1_p']+1,b2_s=0.,a2_s=0.)
        if side:
            for i in range(1,6):
                bp,ap=f[f'b{i}_p'],f[f'a{i}_p']
                f[f'b{i}_p'],f[f'a{i}_p']=(100-ap if ap>=0 else -1),(100-bp if bp>=0 else -1)
    args=dict(side=side,changes=((0,25),),size=25.,extra={'LD2','XV','XMIN1'},frames=frames,known_times=[0.,25.,40.])
    paired=scenario(**args);legacy=scenario(**args,paired=False)
    assert any(t==40. and kind=='place' and lane==side for t,_,kind,lane in paired['requests'])
    assert not any(t==40. and kind=='place' and lane==side for t,_,kind,lane in legacy['requests'])
    assert paired['live_ctr']['lz_rejection_answer_release']>0
    # This known frozen-handler defect is an explicit exception to blanket
    # fixed-size output parity; suppressing the correction would fail Rust.
