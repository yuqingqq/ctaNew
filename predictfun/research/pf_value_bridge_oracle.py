"""Post-hoc diagnostic, never a candidate policy: separate forecast and response gaps.

Use the SAME frozen Oct27 neutral action sample. (1) Supply the true pair of
coarse public flow/price classes to each historical response mapping, preserving
its conditional prior. (2) Supply the actual paired future path while keeping
the same CLIENT-KNOWN order/position state. Future information appears only in
this retrospective diagnostic. Unsupported classes remain explicit statuses.
No predictor, threshold, candidate selection or policy replay changes.
"""
from collections import Counter
import json
from pathlib import Path
import subprocess
import time
import pf_value_bridge as V
import pf_value_bridge_actions as A
import pf_value_bridge_run as R
import pf_flow_price_data as DATA
import pf_flow_control_run as RUN
import pf_flow_control_checks as CHECK
F,C,np,pd,I=V.F,V.C,V.np,V.pd,V.I


def run():
    V.verify();start=time.time();source=V.OUT/'actions_20261027';receipt=json.loads((source/'receipt.json').read_text())
    for p,h in receipt['outputs'].items():C.require(C.sha(source/p)==h,'oracle input changed')
    existing=pd.read_parquet(source/'values.parquet');targets=existing[existing.case=='EXACT_JOINT'].copy()
    oldroot=F.OUT/'bounded_probe_v1';oldreceipt=json.loads((oldroot/'receipt.json').read_text())
    C.require(C.sha(oldroot/'probes.parquet')==oldreceipt['outputs']['probes.parquet'],'old cancellation artifact changed')
    old=pd.read_parquet(oldroot/'probes.parquet');keys=['m','side','rung','generation','decision_ms']
    old=old.set_index(keys).sort_index();current=targets[targets.kind=='cancel'].set_index(keys).sort_index()
    C.require(old.index.equals(current.index),'exact cancellation population changed')
    C.require(np.allclose(current.predicted,old.predicted_advantage_cents,atol=1e-9) and np.allclose(current.actual,old.realized_advantage_cents,atol=1e-9),'exact cancellation predictions or labels changed')
    selected,data,fair,depth,raw=A.load(F.TEST);paths=V.PATH.load();results=[];controls={}
    for W in I.pairs(data):
        m=int(W[0]['m']);local=targets[targets.m==m];bytime={float(t):g for t,g in local.groupby('decision_ms')}
        X,_=I.market_inputs(W,fair[m],depth);prob=V.P.forecasts(W,X)
        controllers={case:V.Controller(W,paths,prob,case) for case in ('EXACT_JOINT','SOFT_JOINT')}
        def observe(k,t,ST,kpos):
            if t not in bytime:return
            state=V.P.snapshot(W,k,t,ST,kpos);j=int(W[0]['jk'][k]);bid=float(W[0]['P5'][j,0]);spread=float(W[1]['P5'][j,0]-bid);mid=bid+spread/2
            true=[]
            for side,price in ((0,bid),(1,bid+spread)):
                z=DATA.public_target(W[side],t,price,mid);true.append(int(C.classes(z['flow_quantity'],z['move_cents'])[2]))
            C.require(all(0<=x<15 for x in true),'invalid true class')
            perfect,status=V.PATH.extract(W,t,F.TEST)
            if perfect is not None:
                C.require(perfect['bid']==bid and perfect['spread']==spread and perfect['classes']==true,'oracle starts at a different known book')
                packed=V.K.pack([perfect],bid,1000.);V.K.current_book(packed,[W[s]['P5'][j] for s in (0,1)],[W[s]['S5'][j] for s in (0,1)])
                perfect_context=(packed,np.ones(1),mid,dict(unsupported_mass=0.,marginal_l1=0.,effective_paths=1.))
            for row in bytime[t].to_dict('records'):
                ids=[i for i,x in enumerate(state['info']) if x['kind']==row['kind'] and x['side']==row['side'] and x['rung']==row['rung'] and (row['kind']=='post' or abs(x['activation']-(t-row['x_age_ms']))<.001)]
                C.require(len(ids)==1,'oracle action identity');index=ids[0]
                base={k:row[k] for k in ('uid','day','m','side','rung','generation','phase','kind','decision_ms','actual','actual_wealth','actual_risk','actual_quantity')}
                for case,controller in controllers.items():
                    ctx,st=controller.context(k,t)
                    if ctx is None:results.append(dict(**base,case=case+'_TRUE_CLASS',evaluation_status=st));continue
                    pp=controller.cache[(min(5,int((t-W[0]['ws'])//60000)+1),bid,spread)][0]
                    match=np.array([p['classes']==true for p in pp]);weights=np.array([p['base_weight'] for p in pp])*match
                    if not match.any():results.append(dict(**base,case=case+'_TRUE_CLASS',evaluation_status='true_joint_class_unsupported'));continue
                    weights/=weights.sum();diagnostic=dict(unsupported_mass=0.,marginal_l1=0.,effective_paths=float(1/(weights@weights)))
                    est=A.prediction(state,index,row['kind'],(ctx[0],weights,mid,diagnostic))
                    results.append(dict(**base,case=case+'_TRUE_CLASS',evaluation_status='eligible',**est))
                result=dict(**base,case='ACTUAL_FUTURE_CLIENT_STATE',evaluation_status=status)
                if perfect is not None:result.update(A.prediction(state,index,row['kind'],perfect_context));result['evaluation_status']='eligible'
                results.append(result)
        replay=RUN.run_lane(W,cap=F.CAP,observer=observe)
        with A.CLOCK.clocks(58.,37.):reference=A.AC.replay_neutral(W,size=30.,shift_ms=50.,queue='M',nr=6,cap=F.CAP,mode='hard')
        for field in ('fl','stat','orders','cancels'):np.testing.assert_equal(replay[field],reference[field])
        C.guard('bridge oracle diagnostic');print(json.dumps(dict(m=m,seconds=round(time.time()-start))),flush=True)
    df=pd.DataFrame(results);C.require(not df.duplicated(['uid','case']).any(),'duplicate oracle identity')
    for case,g in df.groupby('case'):C.require(set(g.uid)==set(targets.uid),'missing oracle action')
    C.require((df.actual==df.actual_wealth+df.actual_risk).all() or np.allclose(df.actual,df.actual_wealth+df.actual_risk),'actual decomposition')
    good=df[df.evaluation_status=='eligible'];counts=good.groupby('uid').case.nunique();common=set(counts[counts==3].index)
    metrics={};common_metrics={}
    combined=pd.concat([existing[existing.case.isin(['EXACT_JOINT','SOFT_JOINT'])],df],ignore_index=True)
    for (case,kind),g in combined[combined.evaluation_status=='eligible'].groupby(['case','kind']):
        metrics[case+':'+kind]=R.action_metrics(g)
        z=g[g.uid.isin(common)]
        if len(z):common_metrics[case+':'+kind]=R.action_metrics(z)
    C.require(len(good)>0,'no positive oracle control')
    controls['all_branch_rows_on_original_action_identities']=True
    controls['exact_old_cancellation_predictions_and_labels_match']=len(old)
    controls['known_bad_realized_value_refused']=CHECK.refused(lambda:C.require(np.isclose(good.actual.iloc[0]+1.,good.actual_wealth.iloc[0]+good.actual_risk.iloc[0],atol=1e-7),'wrong actual'),'corrupt actual')
    controls['oracle_information_kept_out_of_candidate']=C.sha(V.FREEZE)==receipt['source_manifest']
    dest=V.OUT/'oracle_v1';C.require(not dest.exists(),'oracle output exists');dest.mkdir()
    df.to_parquet(dest/'values.parquet',index=False)
    report=dict(asof=C.stamp(),n=len(targets),common_supported_actions=len(common),statuses=df.groupby(['case','evaluation_status']).size().to_dict(),
        available_metrics=metrics,common_support_metrics=common_metrics,controls=controls,
        limits=['Retrospective future-information diagnostic, not an achievable model or strategy','True coarse classes still omit within-class timing, size and price severity','Actual future plus client state still omits hidden current queue/remainder and unreceived fills','Consumed one-day action sample, no significance or profitability claim'])
    C.dump(dest/'report.json',report);C.dump(dest/'receipt.json',dict(asof=C.stamp(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip(),source_sha256=C.sha(Path(__file__)),
        candidate_freeze_sha256=C.sha(V.FREEZE),input_receipt_sha256=C.sha(source/'receipt.json'),outputs={p.name:C.sha(p) for p in dest.iterdir() if p.is_file()},seconds=time.time()-start))
    print(json.dumps(C.clean(dict(n=len(targets),statuses=report['statuses'],metrics=metrics))),flush=True)


if __name__=='__main__':run()
