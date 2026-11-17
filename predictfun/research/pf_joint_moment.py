"""Action-conditioned fill/price moments, with hidden state confined to labels."""
from pathlib import Path
import json
import subprocess
import pf_value_decomposition as D
import pf_flow_price_data as DATA
import pf_flow_price_model as FM
F,C,np,pd,I,A,V,K=D.F,D.C,D.np,D.pd,D.I,D.A,D.V,D.K
OUT=Path(I.K.IO)/'joint_moment_20261117_v2'
REVIEW=F.HERE.parent/'reviews/joint_moment_20261117/v2'
FREEZE=REVIEW/'freeze_v2.json'
EXTRA=['jm_kind','jm_rung','jm_edge','jm_depth','jm_age','jm_remaining','jm_ahead',
       'jm_scale','jm_seconds_left','jm_known_position','jm_ahead_own','jm_behind_own','jm_opposite_own',
       'jm_shown','jm_unknown_depth','jm_cum_displayed_ahead']
FEATURES=C.MARKET+EXTRA
MECHANICAL=['fside','pf_mid','pf_spread']+EXTRA
SOURCES=('pf_joint_moment.py','pf_joint_moment_data.py','pf_joint_moment_fit.py','run_joint_moment.sh')
CASES=('SOFT_JOINT','JOINT_MOMENT','MECHANICAL_MOMENT','DIRECT_WEALTH','REMAINDER_ONLY')


def features(W,k,t,state,index,market,scale):
    C.require(np.isfinite(scale) and scale>0,'invalid liquidity scale')
    o=state['orders'][index];side=int(o[0]);sign=1-2*side;j=int(W[0]['jk'][k])
    mid=(W[0]['P5'][j,0]+W[1]['P5'][j,0])/2.;best=W[side]['P5'][j,0]
    own_ahead=own_behind=opposite=0.
    for i,z in enumerate(state['orders'][:state['existing']]):
        if i==index:continue
        if z[0]!=side:opposite+=z[3]
        elif sign*(z[2]-o[2])>0 or (z[2]==o[2] and z[5]<o[5]):own_ahead+=z[3]
        else:own_behind+=z[3]
    prices=W[side]['P5'][j];sizes=W[side]['S5'][j]
    unknown=bool(prices[4]>=0 and sign*(prices[4]-o[2])>0)
    shown=float(sizes[prices==o[2]].sum())
    displayed_ahead=float(sizes[(prices>=0)&(sign*(prices-o[2])>0)].sum())
    extra=[float(state['info'][index]['kind']=='cancel'),o[1],sign*(mid-o[2]),sign*(best-o[2]),
        np.log1p(max(-o[5],0.)/50.),o[3]/30.,np.log1p(o[4]/30.),np.log1p(scale/30.),
        (W[0]['ws']+300000.-t)/1000.,sign*state['position']/180.,
        np.log1p(own_ahead/30.),np.log1p(own_behind/30.),np.log1p(opposite/30.),
        np.nan if unknown else np.log1p(shown/30.),float(unknown),np.log1p(displayed_ahead/30.)]
    x=np.r_[market,extra].astype(np.float32)
    C.require(x.shape==(len(FEATURES),) and not np.isinf(x).any(),'moment feature contract')
    return x


def label_state(W,k,t,ST,kpos,state):
    """Latent exchange state is used ONLY for labels, never in features()."""
    orders=state['orders'].copy();position=float(kpos)
    lookup={}
    for s in (0,1):
        for o in ST[s]['cl']:
            lookup[s,o['rung'],o['ta']]=o
            position+=(1-2*s)*sum(q for ft,q in o['fl'] if ft<=t<ft+100.)
    for i,info in enumerate(state['info'][:state['existing']]):
        s=int(info['side']);o=lookup[s,info['rung'],info['activation']]
        orders[i,3]=max(0.,30.-sum(q for ft,q in o['fl'] if ft<=t))
        orders[i,6]=min(1e30,o['tr']-t)
        if o['ta']<=t:
            ahead=float(o['ahead']);w=W[s];last=int(np.searchsorted(w['tf'],t,side='right'))
            begin=max(int(o['ci'])+1,0)
            if last>begin:
                p=w['P5'][begin:last];q=w['S5'][begin:last];deep=(p[:,4]>=0)&((o['pt']<p[:,4]) if s==0 else (o['pt']>p[:,4]))
                shown=(q*(p==o['pt'])).sum(1);shown=shown[~deep]
                if len(shown):ahead=min(ahead,float(shown.min()))
            orders[i,4]=ahead
    C.require(np.isfinite(orders).all() and (orders[:,3]>=0).all(),'latent label state')
    return dict(state,orders=orders,position=position)


def fast_label(W,k,t,ST,kpos,state,index,kind,day):
    latent=label_state(W,k,t,ST,kpos,state);path,status=V.PATH.extract(W,t,day)
    if path is None:return dict(status='label_unavailable:'+status)
    j=int(W[0]['jk'][k]);bid=float(W[0]['P5'][j,0]);mid=(bid+float(W[1]['P5'][j,0]))/2.
    packed=K.pack([path],bid,1000.);K.current_book(packed,[W[s]['P5'][j] for s in (0,1)],[W[s]['S5'][j] for s in (0,1)])
    result=D.estimate(latent,index,kind,(packed,np.ones(1),mid,{}));orientation=1. if kind=='post' else -1.
    return dict(y_q=result['pred_affected_quantity'],y_m=orientation*result['pred_focal_motion'],
        y_w=orientation*(result['pred_focal_edge']+result['pred_focal_motion']),y_value=result['predicted'],
        y_risk=result['pred_risk'],y_other=result['pred_other_edge']+result['pred_other_motion'],
        y_severe=float(orientation*result['predicted'] < -30.),
        y_remaining=latent['orders'][index,3]/max(state['orders'][index,3],1e-12),
        actual_endpoint_move=result['pred_endpoint_move'])


def canonical_label(W,row,reference,mid,price,remaining):
    base=dict(row,**A.truth(W,row,reference));z=D.truth(W,base,reference,mid,price)
    orientation=1. if row['kind']=='post' else -1.
    return dict(y_q=z['actual_affected_quantity'],y_m=orientation*z['actual_focal_motion'],
        y_w=orientation*(z['actual_focal_edge']+z['actual_focal_motion']),y_value=z['actual_value'],
        y_risk=z['actual_risk'],y_other=z['actual_other_edge']+z['actual_other_motion'],
        y_severe=float(orientation*z['actual_value'] < -30.),y_remaining=remaining,
        actual_endpoint_move=z['actual_endpoint_move'])


def bounded(q,m,remaining,mid,side):
    q=np.clip(q,0.,remaining);sign=1-2*np.asarray(side)
    low=np.minimum(-sign*mid,sign*(100.-mid));high=np.maximum(-sign*mid,sign*(100.-mid))
    m=np.clip(m,q*low,q*high)
    C.require(np.isfinite(q).all() and np.isfinite(m).all(),'nonfinite moment predictions')
    return q,m


def freeze():
    D.verify();C.require(not FREEZE.exists(),'moment freeze exists')
    C.dump(FREEZE,dict(asof=C.stamp(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip(),
        sources={str((F.HERE/n).relative_to(F.ROOT)):C.sha(F.HERE/n) for n in SOURCES},inherited_decomposition_sha256=C.sha(D.FREEZE),
        supersedes=dict(freeze_sha256=C.sha(REVIEW.parent/'freeze_v1.json'),reason='V1 stopped on missing/stale future path during first training day, before fitting or evaluation. V2 retains selected unavailable-label actions as explicit statuses without refill; target/features/models unchanged.'),
        training=list(F.TRAIN),evaluation=[F.CAL,F.TEST],features=FEATURES,mechanical=MECHANICAL,
        parameters=C.V.PARAMS,rounds=200,calibration='No fitted scaling, threshold, temperature or horizon search. Oct26 earlier consumed diagnostic; Oct27 main consumed diagnostic.',
        population='Capped neutral path. One deterministic probe per cancellable generation plus post opportunities; choose2 per market/side/phase/kind by existing identity hash after coverage checks, BEFORE client availability/reservation checks; unavailable selected actions keep statuses without refill, including label_unavailable path statuses. All selected training markets on Oct23-25; exact418/443 archived eval actions. No outcome sampling or fill-only fitting.',
        label_checks='Every evaluation action against its exact archived canonical components; train markets m%17==0 plus first market/day, minimum-hash selected action per kind/side, checked by full canonical paired replay. Nonfinite no-cancel clock uses finite1e30 sentinel. Preflight861 actions passed before freezing; failed preliminary sentinel adapter run retained.',
        labels='Same1s paired action target, no later posts. True-state actual-future kernel for labels only, with canonical parity gate on every eval action and deterministic training checks. Never latent inputs. Q=affected focal shares; M=Q*signed endpoint movement; W=known edge*Q+M; severe retained total value<-30c; remaining fraction is actual/known.',
        model='Pinned LGBM regressions on all selected actions for Q/30,M/30,W/30; M is a joint product target, not a product of forecasts. Full and mechanical Q/M and severe heads. Remaining-fraction regression on cancellation exposures only. Explicit finite-payoff moment bounds, no target winsorization.',
        comparison=list(CASES),integration='Prediction screen only: replace focal wealth with orientation*(edge*Q+M), retain baseline other-order and risk terms. Direct W and mechanical ablations. Remaining-only scales baseline focal wealth for cancel, posts unchanged. No joint multi-order controller rollout; shared-volume integration remains required before deployment.',
        null=dict(draws=200,seed=20261116,match='market/side/kind positive-action count',metric='realized incremental action value'),
        acceptance='Ordinary JOINT_MOMENT must improve value RMSE vs SOFT for both post/cancel on both dates, and severe retained-loss subset RMSE no worse on both dates/kinds; compared incrementally to mechanical/direct baselines. A descriptive screen, not promotion.',
        multiplicity=dict(joint_moment_candidates=1,direct_baselines=1,mechanical_baselines=1,remaining_ablation=1,new_controllers=0,hyperparameter_search=0),
        limits=['Consumed days only; no untouched validation, intervals, promotion or strategy P&L',
            'Separate Q/M regressions estimate joint moments but are not a joint scenario generator or a simultaneous multi-order allocator',
            'Pending-state head is evaluated separately; Q/M labels already include latent remainder uncertainty, no double scaling',
            'Unchanged estimated clocks, queue assumptions,1s zero continuation and neutral-to-policy transport'],output=str(OUT)))


def verify():
    D.verify();r=json.loads(FREEZE.read_text());C.require(r['inherited_decomposition_sha256']==C.sha(D.FREEZE),'decomposition changed')
    for p,h in r['sources'].items():C.require(C.sha(F.ROOT/p)==h,'moment source changed '+p)
    return r
