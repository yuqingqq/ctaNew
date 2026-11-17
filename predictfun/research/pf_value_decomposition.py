"""Retrospective action-value decomposition; no model or policy changes."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import pf_conditional_map as B
import pf_value_bridge_actions as A
import pf_flow_control_checks as CHECK
F,C,np,pd,I,V,K=B.F,B.C,B.np,B.pd,B.I,B.V,B.K
OUT=Path(I.K.IO)/'value_decomposition_20261117_v1'
REVIEW=F.HERE.parent/'reviews/value_decomposition_20261117'
FREEZE=REVIEW/'freeze_v1.json'
SOURCES=('pf_value_decomposition.py','pf_value_decomposition_run.py','run_value_decomposition.sh')
PARTS=('edge','motion','risk')
SUBSTITUTIONS=('original','true_edge','true_motion','true_risk','true_other_wealth',
               'independent_focal_fill_move','true_endpoint_all_fills','true_focal_quantity')


def reconcile(x):
    C.require(np.allclose(x['wealth'],x['edge']+x['motion'],rtol=0,atol=1e-7),'wealth decomposition')
    C.require(np.allclose(x['edge'],x['focal_edge']+x['other_edge'],rtol=0,atol=1e-7),'edge allocation')
    C.require(np.allclose(x['motion'],x['focal_motion']+x['other_motion'],rtol=0,atol=1e-7),'motion allocation')
    C.require(np.allclose(x['value'],x['wealth']+x['risk'],rtol=0,atol=1e-7),'utility decomposition')
    C.require(np.all(np.asarray(x['affected_quantity'])>=-1e-7),'negative affected quantity')


def components(orders,qty,positions,index,kind,mid,marks):
    """Action minus keep/wait, with common decision-time mid and endpoint.

    qty is [scenario, branch, order]. Cancellation quantity is q_keep-q_cancel,
    excluding the unavoidable common exposure. Other-order redistribution is
    retained explicitly, rather than attributed to the focal order.
    """
    C.require(kind in ('post','cancel'),'invalid action kind')
    C.require(qty.ndim==3 and qty.shape[1:]==(2,len(orders)) and np.isfinite(qty).all() and
              (qty>=-1e-9).all() and np.isfinite(positions).all(),'component array contract')
    C.require(np.isin(orders[:,0],[0,1]).all() and np.isfinite(orders[:,2]).all(),'component order contract')
    delta=qty[:,1]-qty[:,0];sign=1-2*orders[:,0];move=np.asarray(marks)-mid
    edge=delta*sign*(mid-orders[:,2]);motion=delta*sign*move[:,None]
    orientation=1. if kind=='post' else -1.
    x=dict(edge=edge.sum(1),motion=motion.sum(1),focal_edge=edge[:,index],focal_motion=motion[:,index],
        other_edge=edge.sum(1)-edge[:,index],other_motion=motion.sum(1)-motion[:,index],
        affected_quantity=orientation*delta[:,index],delta_position=(delta*sign).sum(1),
        endpoint_move=move,focal_move=sign[index]*move,
        risk=-F.RISK*(positions[:,1]**2-positions[:,0]**2))
    x['wealth']=x['edge']+x['motion'];x['value']=x['wealth']+x['risk'];reconcile(x)
    return x


def summarize(x,weights):
    C.require(np.isfinite(weights).all() and (weights>=0).all() and np.isclose(weights.sum(),1.),'invalid decomposition weights')
    z={name:float(weights@v) for name,v in x.items()}
    q=z['affected_quantity'];joint=float(weights@(x['affected_quantity']*x['focal_move']))
    z.update(focal_fill_move=joint/q if q>1e-12 else np.nan,
        focal_fill_move_status='eligible' if q>1e-12 else 'zero_predicted_affected_fill',
        focal_covariance=joint-q*z['focal_move'])
    reconcile(z);return z


def estimate(state,index,kind,ctx):
    packed,w,mid,diag=ctx;act=state['baseline'].copy();act[index]=1. if kind=='post' else 2.
    out,q=K.evaluate(packed,state['orders'],np.stack([state['baseline'],act]),state['position'],mid,1000.)
    x=components(state['orders'],q,out[:,:,2],index,kind,mid,packed[-1])
    C.require(np.allclose(x['wealth'],out[:,1,0]-out[:,0,0],rtol=0,atol=1e-7),'kernel wealth differs')
    z=summarize(x,w)
    return {**{'pred_'+k:v for k,v in z.items()},'predicted':z['value'],
        'predicted_wealth':z['wealth'],'predicted_risk':z['risk'],
        'predicted_quantity':float(w@q[:,int(kind=='post'),index]),**diag}


def truth(W,row,reference,mid,price):
    """Re-run exact canonical branches and reconcile the archived label first."""
    t=float(row['decision_ms']);end=t+1000.;s=int(row['side']);d=int(row['rung']);post=row['kind']=='post'
    activation=t+58. if post else t-row['x_age_ms'];key=(s,d,activation);branches=[];prefix=[]
    mt,mm=A.AC.valid_marks(W[0],0);j=np.searchsorted(mt,end,side='right')-1
    C.require(j>=0 and end-mt[j]<=1000.,'missing common endpoint');mark=float(mm[j])
    for act in (False,True):
        z=I.truncate(W,end-W[0]['ws'])
        for side,w in z.items():
            mask=np.asarray(w['anc'])-50.<=end
            for field in ('anc','ptl','shl'):w[field]=np.asarray(w[field])[mask]
        def gate(side,rung):
            return lambda k:bool(float(z[0]['tg'][k])>=t and not (post and act and side==s and rung==d and float(z[0]['tg'][k])==t))
        gates={(side,rung):gate(side,rung) for side in (0,1) for rung in range(6)}
        dec={(s,d):(lambda k,now,o:bool(not post and act and now>=t and abs(o['ta']-activation)<.001))}
        with A.CLOCK.clocks(58.,37.):
            r=A.LANE.sim_joint_g(z,'M',50.,F.CAP,{side:[False]*z[side]['n'] for side in (0,1)},
                decide=dec,gate=gates,size=30.,nr=6,mode='hard',cooldown=0.)
        C.require(not any(r['stat']['viol'].values()),'canonical branch violation')
        C.require(Counter(x for x in r['fl'] if x[2]<=t)==Counter(x for x in reference['fl'] if x[2]<=t),'canonical historical prefix')
        prefix.append(Counter(x for x in r['fl'] if x[2]<(t+58. if post else t+37.)))
        C.require(all(ft<=end for _,_,ft,_,_ in r['fl']),'post-endpoint fill')
        orders={}
        for (side,rung),group in r['order_states'].items():
            for o in group:
                identity=(side,rung,float(o['ta']));C.require(identity not in orders,'duplicate canonical order')
                orders[identity]=(float(o['pt']),sum(q for ft,q in o['fl'] if ft>t))
        pos=sum((1-2*side)*q for side,rung,ft,p,q in r['fl'])
        cash=-sum((1-2*side)*p*q for side,rung,ft,p,q in r['fl'])
        C.require(np.isclose(pos,r['endpos'],rtol=0,atol=1e-7),'canonical position ledger')
        C.require((key in orders)==(bool(act) if post else True),'focal canonical identity')
        branches.append(dict(orders=orders,position=pos,wealth=cash+pos*mark))
    C.require(prefix[0]==prefix[1],'unavoidable branch prefix differs')
    identities=sorted(set(branches[0]['orders'])|set(branches[1]['orders']));orders=[];qty=np.zeros((1,2,len(identities)))
    for i,identity in enumerate(identities):
        prices={b['orders'][identity][0] for b in branches if identity in b['orders']};C.require(len(prices)==1,'order price identity')
        orders.append([identity[0],identity[1],prices.pop()])
        for k,b in enumerate(branches):qty[0,k,i]=b['orders'].get(identity,(0.,0.))[1]
    index=identities.index(key);C.require(orders[index][2]==price,'focal client/canonical price')
    x=components(np.asarray(orders),qty,np.array([[b['position'] for b in branches]]),index,row['kind'],mid,np.array([mark]))
    wealth=branches[1]['wealth']-branches[0]['wealth']
    C.require(np.isclose(x['wealth'][0],wealth,rtol=0,atol=1e-7),'canonical cash decomposition')
    for name,field in [('value','actual'),('wealth','actual_wealth'),('risk','actual_risk')]:
        C.require(np.isclose(x[name][0],row[field],rtol=0,atol=1e-7),'archived actual changed '+field)
    C.require(np.isclose(qty[0,int(post),index],row['actual_quantity'],rtol=0,atol=1e-7),'archived raw focal quantity changed')
    return {'actual_'+k:float(v[0]) for k,v in x.items()}


def selftest():
    orders=np.array([[0,0,50.]])
    qty=np.array([[[0.],[10.]],[[0.],[0.]]]);positions=qty.sum(2)
    x=components(orders,qty,positions,0,'post',51.,np.array([48.,54.]));z=summarize(x,np.array([.5,.5]))
    C.require(z['edge']==5. and z['motion']==-15. and z['wealth']==-10. and z['focal_covariance']==-15.,'joint fill/price positive control')
    C.require(z['affected_quantity']*(1.+z['focal_move'])==5.,'decorrelation control')
    q=np.array([[[10.],[4.]]]);x=components(orders,q,q.sum(2),0,'cancel',51.,np.array([48.]))
    C.require(x['affected_quantity'][0]==6. and x['wealth'][0]==12.,'avoidable quantity / cancellation sign')
    ladder=np.array([[0,0,50.],[0,1,49.]])
    q=np.array([[[20.,0.],[0.,20.]]]);x=components(ladder,q,q.sum(2),0,'cancel',51.,np.array([48.]))
    C.require(x['focal_motion'][0]==60. and x['other_motion'][0]==-60. and x['wealth'][0]==20.,'other order redistribution')
    empty=components(orders,np.zeros((1,2,1)),np.zeros((1,2)),0,'post',51.,np.array([48.]))
    C.require(empty['value'][0]==0. and summarize(empty,np.ones(1))['focal_fill_move_status']=='zero_predicted_affected_fill','zero fill included')
    bad=dict(x);bad['motion']=x['motion']+1.
    checks=dict(joint_fill_price_dependence_detected=True,unavoidable_quantity_excluded=True,other_order_redistribution_accounted=True,
        no_fill_included=True,corrupt_component_refused=CHECK.refused(lambda:reconcile(bad),'bad component'),
        negative_weight_refused=CHECK.refused(lambda:summarize(x,np.array([-1.])),'bad weight'),
        reversed_action_refused=CHECK.refused(lambda:components(orders,np.array([[[0.],[10.]]]),np.array([[0.,10.]]),0,'cancel',51.,np.array([48.])),'bad sign'))
    return checks


def freeze():
    B.verify();C.require(not FREEZE.exists(),'decomposition freeze exists')
    C.dump(FREEZE,dict(asof=C.stamp(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip(),
        sources={str((F.HERE/n).relative_to(F.ROOT)):C.sha(F.HERE/n) for n in SOURCES},conditional_map_freeze_sha256=C.sha(B.FREEZE),
        population='Exact archived neutral actions:418 Oct26 and443 Oct27,12 fixed markets each; all statuses/no-fill retained. Canonical paired labels re-run and reconciled by generation/time identity.',
        estimand='Action minus keep/wait at common t+1s endpoint; sum_i delta_q_i*side_i*(m0-p_i) + delta_position*(mH-m0) - lambda*(pos_action^2-pos_base^2). Split focal/other orders. Affected focal quantity excludes unavoidable prefix.',
        cases=list(B.CASES),channels=['ordinary','true_pair_Oct27_only','actual_path_client_state'],
        decomposition=list(PARTS),substitutions=list(SUBSTITUTIONS),
        null=dict(draws=200,seed=20261116,match='market/side/action-kind positive-action count; one market per two-hour block',metric='actual incremental economic value'),
        tails='Retained exposure value = action value for post, -action value for cancel. severe_loss<-30c; loss[-30,0); zero tolerance1e-9; gain(0,30]; severe_gain>30. Fixed1c/share full-size30 boundary; no loss clipping.',
        largest_errors='Archive10 largest squared-error actions per day/channel/case/kind as descriptive examples only; never a training population or parameter-selection rule.',
        error_accounting='Report per-component RMSE and signed sum(error_component*total_error)/sum(total_error^2), including cross terms. These are accounting contributions, not causal attribution. Component substitutions are future-information diagnostics only.',
        quantity_substitution='Replace focal expected quantity using actual avoidable quantity while keeping model fill-weighted movement; refuse if predicted affected quantity zero and actual positive. All other contributions unchanged.',
        multiplicity=dict(new_models=0,new_policies=0,diagnostic_designs=1,component_substitutions=len(SUBSTITUTIONS)-1),
        limits=['Consumed Oct26/27 development; no intervals, promotion or full-policy P&L',
            'Future values appear only in labelled diagnostic substitutions, never policy inputs',
            'Same estimated trade clocks, queue assumptions,1s zero continuation and neutral target',
            'Decomposition reconciles simulator economics; does not establish learnability or external realism'],output=str(OUT)))


def verify():
    B.verify();r=json.loads(FREEZE.read_text());C.require(r['conditional_map_freeze_sha256']==C.sha(B.FREEZE),'inherited mapper changed')
    for p,h in r['sources'].items():C.require(C.sha(F.ROOT/p)==h,'decomposition source changed '+p)
    return r
