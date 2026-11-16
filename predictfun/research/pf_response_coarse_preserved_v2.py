"""Retrospective representation control; never usable by the candidate policy.

Give the projection correct fine response categories, then compare with the
previous true-coarse-category diagnostic on the identical neutral action IDs.
This measures finite-bank representation error separately from forecast error.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time
import pf_response_bridge_v2 as B
import pf_response_eval_v2 as E
import pf_value_bridge_actions as A
import pf_value_bridge_run as R
import pf_flow_control_checks as CHECK
import pf_flow_control_run as RUN
F,C,np,pd,I=B.F,B.C,B.np,B.pd,B.I
DECL=B.REVIEW/'coarse_preserved_freeze_v2.json'


def freeze():
    B.verify();C.require(not DECL.exists(),'diagnostic already frozen')
    C.dump(DECL,dict(asof=C.stamp(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip(),
        source_sha256=C.sha(Path(__file__)),candidate_manifest=C.sha(B.FREEZE),
        supersedes=dict(freeze='coarse_preserved_freeze.json',reason='Fix a receipt-path typo found by review before the diagnostic was run. Old source/freeze retained; no outputs were produced.'),
        design='Same Oct27 archived443 actions. Actual future tape supplies fine/coarse signatures only to this retrospective control. After the original fine projection, set weights outside the known coarse pair to zero and renormalize. Missing coarse pair or zero posterior mass remain explicit statuses. Same client order state and training donor pool. Unsupported true marginal becomes an explicit status. Compare all metrics on common identities with learned RICH_JOINT, SOFT_JOINT and original SOFT_JOINT_TRUE_CLASS and ACTUAL_FUTURE_CLIENT_STATE.',
        role='Post-hoc projection diagnostic: condition the already computed true-fine posterior on the true coarse pair. Frozen before this check, not a candidate or usable forecast.',
        null=dict(draws=200,match='market/side/kind positive-action count'),limits=B.verify()['limits']))


def verify():
    f=json.loads(DECL.read_text());B.verify()
    C.require(C.sha(Path(__file__))==f['source_sha256'] and C.sha(B.FREEZE)==f['candidate_manifest'],'diagnostic changed')


def audit_frames(frame,targets):
    C.require(not frame.uid.duplicated().any(),'duplicate diagnostic action')
    a=frame.set_index('uid').sort_index();b=targets.set_index('uid').sort_index();C.require(a.index.equals(b.index),'diagnostic identity mismatch')
    for key in ('decision_ms','m','side','rung','generation','actual','actual_quantity','actual_wealth','actual_risk'):
        C.require(np.allclose(a[key],b[key],equal_nan=True,atol=1e-10,rtol=0),'diagnostic label mismatch '+key)
    good=frame[frame.evaluation_status=='eligible']
    C.require(np.allclose(good.predicted,good.predicted_wealth+good.predicted_risk,atol=1e-8),'diagnostic value arithmetic')
    C.require((good.predicted_quantity>=-1e-7).all() and (good.predicted_quantity<=30+1e-7).all(),'diagnostic fill bound')


def require_coarse(weight,mask):
    C.require(np.isclose(weight.sum(),1.) and float(weight[~mask].sum())==0.,'coarse information erased')


def run():
    verify();start=time.time();src,oldreceipt=E.archived('actions_'+F.TEST)
    old=pd.read_parquet(src/'values.parquet');targets=old[old.case=='SOFT_JOINT'].copy()
    selected,data,fair,depth,raw=A.load(F.TEST);paths=B.PATH.load();values=[];status_counts=Counter()
    for W in I.pairs(data):
        m=int(W[0]['m']);local=targets[targets.m==m];bytime={float(t):g for t,g in local.groupby('decision_ms')}
        cache={}
        def observe(k,t,ST,kpos):
            if t not in bytime:return
            state=B.P.snapshot(W,k,t,ST,kpos);j=int(W[0]['jk'][k]);bid=float(W[0]['P5'][j,0]);spread=float(W[1]['P5'][j,0]-bid)
            phase=min(5,int((t-W[0]['ws'])//60000)+1);key=(phase,bid,spread)
            if key not in cache:
                pp=B.V.pool(paths,*key);cache[key]=(pp,B.K.pack(pp,bid,1000.),np.stack([B.signature(p,spread) for p in pp]))
            pp,packed,labels=cache[key];actual,status=B.PATH.extract(W,t,F.TEST);ctx=None
            if actual is not None:
                C.require(actual['bid']==bid and actual['spread']==spread,'future path book mismatch')
                true=B.signature(actual,spread);probs=[]
                for x,(name,d) in zip(true,B.specs()*2):probs.append(np.eye(15 if name=='joint' else B.HEADS[name][1])[x])
                missing=[not (labels[:,i]==x).any() for i,x in enumerate(true)]
                coarse=(labels[:,0]==true[0])&(labels[:,len(B.specs())]==true[len(B.specs())])
                if not coarse.any():status='true_coarse_pair_unsupported'
                elif any(missing):status='true_response_marginal_unsupported'
                else:
                    weight,diag=B.project(pp,labels,probs)
                    mass=float(weight[coarse].sum())
                    if mass<=0:status='zero_true_coarse_posterior_mass'
                    else:
                        weight=weight*coarse/mass
                        require_coarse(weight,coarse)
                        residual=[float(np.abs(np.bincount(labels[:,i],weights=weight,minlength=len(p))-p).sum()) for i,p in enumerate(probs)]
                        diag.update(coarse_pair_mass_before=mass,coarse_l1_before=diag['coarse_marginal_l1'],coarse_marginal_l1=0.,
                            marginal_l1=float(np.mean(residual)),effective_paths=float(1/(weight@weight)))
                        B.K.current_book(packed,[W[s]['P5'][j] for s in (0,1)],[W[s]['S5'][j] for s in (0,1)])
                        ctx=(packed,weight,bid+spread/2.,diag);status='eligible'
            for row in bytime[t].to_dict('records'):
                ids=[i for i,v in enumerate(state['info']) if v['kind']==row['kind'] and v['side']==row['side'] and v['rung']==row['rung'] and (row['kind']=='post' or abs(v['activation']-(t-row['x_age_ms']))<.001)]
                C.require(len(ids)==1,'diagnostic client action identity')
                keep=('uid','day','m','side','rung','generation','phase','kind','decision_ms','actual','actual_wealth','actual_risk','actual_quantity')
                result={x:row[x] for x in keep};result.update(case='COARSE_PRESERVED_TRUE_RICH',evaluation_status=status)
                if ctx is not None:result.update(A.prediction(state,ids[0],row['kind'],ctx))
                status_counts[status]+=1;values.append(result)
        replay=RUN.run_lane(W,cap=F.CAP,observer=observe)
        with A.CLOCK.clocks(58.,37.):ref=A.AC.replay_neutral(W,size=30.,shift_ms=50.,queue='M',nr=6,cap=F.CAP,mode='hard')
        for field in ('fl','stat','orders','cancels'):np.testing.assert_equal(replay[field],ref[field])
        C.guard('rich signature diagnostic');print(json.dumps(dict(m=m,seconds=round(time.time()-start))),flush=True)
    frame=pd.DataFrame(values);audit_frames(frame,targets)
    controls=dict(positive_control_actions=len(frame))
    require_coarse(np.array([1.,0.]),np.array([True,False]))
    controls['known_bad_coarse_relaxation_refused']=CHECK.refused(lambda:require_coarse(np.array([0.,1.]),np.array([True,False])),'coarse relaxation')
    bad=frame.copy();bad.loc[bad.index[0],'actual']+=1.
    controls['corrupt_label_refused']=CHECK.refused(lambda:audit_frames(bad,targets),'corrupt diagnostic value')
    bad=frame.copy();bad.loc[bad.index[0],'uid']='wrong-action'
    controls['wrong_action_refused']=CHECK.refused(lambda:audit_frames(bad,targets),'wrong action identity')
    learned_src=B.OUT/('actions_'+F.TEST);lr=json.loads((learned_src/'receipt.json').read_text())
    C.require(C.sha(learned_src/'values.parquet')==lr['outputs']['values.parquet'],'learned values changed')
    learned=pd.read_parquet(learned_src/'values.parquet');learned=learned[learned.case.isin(['RICH_JOINT','SOFT_JOINT'])]
    oracle_src=B.V.OUT/'oracle_v1';orr=json.loads((oracle_src/'receipt.json').read_text())
    C.require(C.sha(oracle_src/'values.parquet')==orr['outputs']['values.parquet'],'old diagnostic changed')
    prior=pd.read_parquet(oracle_src/'values.parquet');prior=prior[prior.case.isin(['SOFT_JOINT_TRUE_CLASS','ACTUAL_FUTURE_CLIENT_STATE'])]
    fine_src=B.OUT/'oracle';fine_receipt=json.loads((fine_src/'receipt.json').read_text())
    C.require(C.sha(fine_src/'values.parquet')==fine_receipt['outputs']['values.parquet'],'fine diagnostic changed')
    fine=pd.read_parquet(fine_src/'values.parquet')
    allrows=pd.concat([learned,prior,fine,frame],ignore_index=True);good=allrows[allrows.evaluation_status=='eligible']
    for case,g in allrows.groupby('case'):audit_frames(g,targets)
    n=good.groupby('uid').case.nunique();common=good[good.uid.isin(n[n==6].index)];metrics={}
    for (case,kind),g in common.groupby(['case','kind']):metrics[case+':'+kind]=R.action_metrics(g)
    dest=B.OUT/'oracle_coarse_preserved';C.require(not dest.exists(),'diagnostic exists');dest.mkdir()
    frame.to_parquet(dest/'values.parquet',index=False)
    C.dump(dest/'report.json',dict(asof=C.stamp(),n=len(frame),statuses=dict(status_counts),common_support_actions=common.uid.nunique(),
        metrics=metrics,controls=controls,coarse_projection=dict(eligible=int((frame.evaluation_status=='eligible').sum()),mean_true_coarse_mass_before=float(frame.coarse_pair_mass_before.mean()),mean_coarse_l1_before=float(frame.coarse_l1_before.mean()),mean_coarse_l1_after=float(frame.coarse_marginal_l1.mean())),source_manifest=C.sha(DECL),limits=json.loads(DECL.read_text())['role']))
    C.dump(dest/'receipt.json',dict(asof=C.stamp(),source_manifest=C.sha(DECL),candidate_manifest=C.sha(B.FREEZE),
        inputs={str(p):C.sha(p) for p in (src/'receipt.json',learned_src/'receipt.json',oracle_src/'receipt.json',fine_src/'receipt.json')},
        outputs={p.name:C.sha(p) for p in dest.iterdir()},seconds=time.time()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run']);args=p.parse_args()
    if args.mode=='freeze':freeze()
    else:run()
