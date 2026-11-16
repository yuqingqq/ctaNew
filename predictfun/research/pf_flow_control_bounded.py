"""Superseding action-value diagnostic on the controller's bounded neutral path.

Does not change or rerun any strategy case. Every label uses the canonical
engine, with the same cap in the reference and both counterfactual prefixes.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import subprocess
import time
import pf_flow_control_common as F
import pf_flow_control_paths as PATH
import pf_flow_control_policy as POLICY
import pf_flow_control_kernel as EXEC
import pf_flow_control_run as RUN
import pf_flow_control_checks as CHECK
import pf_flow_price_data as DATA
import pf_stop_queue_activation_lane as ORIGINAL
import pf_ev_lag_gap as CLOCK
C,np,pd,I,AC=F.C,F.np,F.pd,F.I,F.C.AC
FREEZE=F.REVIEW/'bounded_probe_v1_freeze.json'
OUT=F.OUT/'bounded_probe_v1'


def controls():
    def check_position(q):C.require(abs(q)<=F.CAP+1e-7,'neutral inventory outside controller support')
    check_position(0.);check_position(F.CAP)
    CHECK.refused(lambda:check_position(F.CAP+1.),'unbounded neutral inventory')
    return dict(bounded_position_positive=True,unbounded_position_refused=True)


def freeze():
    F.verify();C.require(not FREEZE.exists(),'bounded freeze exists')
    C.dump(FREEZE,dict(asof=C.stamp(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip(),
        source_sha256=C.sha(Path(__file__)),controller_freeze_sha256=C.sha(F.FREEZE),controls=controls(),
        supersedes=dict(artifact='pilot_v3/interpretation.json: neutral_advantage_prediction and original matched_null',
            reason='169/222 original diagnostic states exceed the 180-share controller net inventory limit. Keep original results but use a bounded neutral reference for action-value evaluation.'),
        population='Same 12 pilot markets, neutral no-learned-cancel reference with hard reservation cap180, size30, six rungs, original clocks. One hash-assigned age per generation; two eligible hash-selected probes per market/side/phase. No-fill actions retained; all statuses exported.',
        expiry='Require t+2s <= final shutdown at ws+298s; uniform potential mark and reference-prefix coverage checks.',
        target='Canonical keep/cancel branches at t+1s; same capped neutral prefix, then no placements. Wealth difference minus identical quadratic risk penalty difference. Common exposure before cancellation arrival must reconcile.',
        null=dict(draws=200,seed=20261116,groups=['market','side'],match='same count of predicted-positive cancellations; actual economic advantage'),
        model='Existing frozen JOINT and same empirical bank; no refit, threshold, policy, size or horizon change',
        multiplicity=dict(new_policies=0,new_models=0,superseding_diagnostics=1),
        limits='One consumed day, small neutral reference sample; no intervals/promotion. Bounded neutral state is closer support, not the learned controller own path. Queue/timing and path-transport assumptions remain.'))


def verify():
    F.verify();r=json.loads(FREEZE.read_text())
    C.require(r['source_sha256']==C.sha(Path(__file__)) and r['controller_freeze_sha256']==C.sha(F.FREEZE),'bounded diagnostic source changed')
    return r


def reference_and_candidates(W,data):
    with CLOCK.clocks(58.,37.):r=AC.replay_neutral(W,size=30.,shift_ms=50.,queue='M',nr=6,cap=F.CAP,mode='hard')
    info,_=DATA.provenance(data,W);tf=np.asarray(W[0]['tf']);ws=W[0]['ws'];m=int(W[0]['m'])
    gaps=np.flatnonzero(np.diff(tf)>500.)
    coverage=AC.Coverage(trade_start_ms=ws-2050.,trade_end_ms=ws+300950.,mark_end_ms=float(tf[-1]),
        expected_legs=len(info),book_start_ms=float(tf[0]),book_end_ms=float(tf[-1]),
        book_gaps=tuple(AC.MissingLeg(f'{m}_{j}',float(tf[j]),float(tf[j+1]),'book_gap') for j in gaps))
    ledger=AC.Ledger(W,r,submitted_quantity=30.,shift_ms=50.,leg_info=info,coverage=coverage)
    rows=[]
    for key,t in AC.generation_probes(ledger,day=F.TEST,ages_ms=C.AGES):
        side,rung,generation=key;order=ledger.orders[key]
        row=dict(m=m,side=side,rung=rung,generation=generation,decision_ms=t,x_age_ms=t-order['ta'],
                 phase=min(5,int((t-ws)//60000)+1),hash=PATH.keyhash(F.TEST,m,t)+f':{side}:{rung}:{generation}')
        if t+2000.>ws+298000.:row['status']='shutdown_horizon'
        else:
            z=AC.label_action(ledger,key,t,day=F.TEST,settings=C.SETTINGS)
            row['status']=z['status'];support=DATA.mark_support(W[side],t)
            if row['status'] in ('eligible','eligible_estimated_timing') and support!='ok':row['status']=support
        rows.append(row)
    return r,pd.DataFrame(rows)


def truth(W,row,reference,corrupt=False):
    t=float(row.decision_ms);endpoint=t+1000.;side=int(row.side);rung=int(row.rung);activation=t-row.x_age_ms
    summaries=[];prefixes=[]
    for cancel in (False,True):
        z=I.truncate(W,endpoint-W[0]['ws'])
        for s,w in z.items():
            mask=np.asarray(w['anc'])-50.<=endpoint
            for name in ('anc','ptl','shl'):w[name]=np.asarray(w[name])[mask]
        gate={(s,d):(lambda k:float(z[0]['tg'][k])>=t) for s in (0,1) for d in range(6)}
        dec={(side,rung):(lambda k,now,o:bool(cancel and now>=t and abs(o['ta']-activation)<.001))}
        with CLOCK.clocks(58.,37.):
            r=ORIGINAL.sim_joint_g(z,'M',50.,F.CAP,{s:[False]*z[s]['n'] for s in (0,1)},
                decide=dec,gate=gate,size=30.,nr=6,mode='hard',cooldown=0.)
        C.require(not any(r['stat']['viol'].values()),'bounded truth conservation')
        prefix=Counter(x for x in r['fl'] if x[2]<=t);expected=Counter(x for x in reference['fl'] if x[2]<=t)
        if corrupt:expected[(0,0,t-1.,50.,.01)]+=1
        C.require(prefix==expected,'bounded truth prefix differs')
        prefixes.append(Counter(x for x in r['fl'] if x[2]<t+37.))
        mt,mm=AC.valid_marks(W[0],0);j=np.searchsorted(mt,endpoint,side='right')-1
        C.require(j>=0 and endpoint-mt[j]<=1000.,'truth endpoint missing')
        pos=sum((1-2*s)*q for s,d,ft,p,q in r['fl']);cash=-sum((1-2*s)*p*q for s,d,ft,p,q in r['fl'])
        own=sum(q for idx,price,q,through,age,*_ in r['fills'][side,rung]
                if float(W[side]['anc'][idx])-50.>t and abs(float(W[side]['anc'][idx])-50.-age-activation)<.001)
        summaries.append(dict(wealth=cash+pos*mm[j],risk=-F.RISK*pos**2,quantity=own))
    C.require(prefixes[0]==prefixes[1],'unavoidable cancellation-prefix exposure changed')
    wealth=summaries[1]['wealth']-summaries[0]['wealth'];risk=summaries[1]['risk']-summaries[0]['risk']
    return dict(realized_advantage_cents=wealth+risk,realized_wealth_advantage_cents=wealth,
                realized_risk_advantage_cents=risk,actual_keep_quantity=summaries[0]['quantity'])


def run():
    declaration=verify();start=time.time();C.require(not OUT.exists(),'bounded output exists');OUT.mkdir(parents=True)
    pilot=json.loads((F.OUT/'pilot/receipt.json').read_text());raw=json.loads((C.OUT/F.TEST/'receipt.json').read_text())
    for p,h in raw['input_hashes'].items():C.require(C.sha(p)==h,'bounded source changed '+p)
    selected=pilot['selected_markets'];reg=pd.read_parquet(Path(I.K.IO)/'step1'/F.TEST/'registry.parquet')
    windows={int(x.mid):float(x.ws)*1000 for x in reg[reg.mid.isin(selected)].itertuples()}
    fair,_=I.fair_fix(F.TEST,windows);clock=json.loads((Path(I.K.IO)/'step1'/F.TEST/'clock_rl_d2.json').read_text())
    depth,_=I.X5.depth_arrays(I.X5.depth_hours(F.TEST),float(clock['bn_in'])+float(clock['d_dec']))
    data=I.load_day(F.TEST,set(selected),raw['imputation_offset_ms']);paths=PATH.load();allrows=[];results=[];flags=controls();stats=Counter()
    for W in I.pairs(data):
        reference,rows=reference_and_candidates(W,data)
        eligible=rows[rows.status.isin(['eligible','eligible_estimated_timing'])]
        probes=eligible.sort_values('hash').groupby(['side','phase'],sort=False).head(2)
        chosen=set(probes.index);rows['selected']=[i in chosen for i in rows.index];allrows.append(rows)
        X,_=I.market_inputs(W,fair[int(W[0]['m'])],depth);prob=POLICY.forecasts(W,X)
        ctx=POLICY.Controller(W,paths,prob,'JOINT');bytime={float(t):g for t,g in probes.groupby('decision_ms')};saved=[]
        def observe(k,t,ST,kpos):
            if t not in bytime:return
            C.require(abs(kpos)<=F.CAP+1e-7,'bounded neutral position outside support')
            context,status=ctx.context(k,t);stats[status]+=len(bytime[t])
            if context is None:return
            packed,weight,mid,diag=context;state=POLICY.snapshot(W,k,t,ST,kpos)
            j=int(W[0]['jk'][k]);bid=float(W[0]['P5'][j,0]);spread=float(W[1]['P5'][j,0]-bid)
            library=ctx.cache[(min(5,int((t-W[0]['ws'])//60000)+1),bid,spread,1000.)][0]
            empirical,_=PATH.weights(library)
            for row in bytime[t].itertuples():
                matches=[i for i,x in enumerate(state['info']) if x['kind']=='cancel' and x['side']==row.side and x['rung']==row.rung and abs(x['activation']-(t-row.x_age_ms))<.001]
                C.require(len(matches)==1,'bounded probe identity');i=matches[0];action=state['baseline'].copy();action[i]=2.
                out,qty=EXEC.evaluate(packed,state['orders'],np.array([state['baseline'],action]),kpos,mid,1000.)
                utility=POLICY.utility(out,kpos);v=weight@utility;e=empirical@utility
                wealth=weight@out[:,:,0]
                saved.append((row,dict(m=int(row.m),side=int(row.side),rung=int(row.rung),generation=int(row.generation),phase=int(row.phase),
                    decision_ms=t,known_inventory=kpos,predicted_advantage_cents=float(v[1]-v[0]),empirical_advantage_cents=float(e[1]-e[0]),
                    predicted_wealth_advantage_cents=float(wealth[1]-wealth[0]),predicted_risk_advantage_cents=float(v[1]-v[0]-wealth[1]+wealth[0]),
                    predicted_keep_quantity=float(weight@qty[:,0,i]),**diag)))
        replay=RUN.run_lane(W,cap=F.CAP,observer=observe)
        for field in ('fl','stat','orders','cancels'):np.testing.assert_equal(replay[field],reference[field])
        for row,record in saved:
            record.update(truth(W,row,reference));results.append(record)
        if saved and not flags.get('wrong_prefix_refused'):
            flags['wrong_prefix_refused']=CHECK.refused(lambda:truth(W,saved[0][0],reference,corrupt=True),'corrupt bounded prefix')
        C.guard('bounded probes')
        print(json.dumps(dict(m=int(W[0]['m']),selected=len(probes),evaluated=len(saved),seconds=round(time.time()-start))),flush=True)
    candidates=pd.concat(allrows,ignore_index=True);result=pd.DataFrame(results)
    C.require(not candidates.duplicated(['m','side','rung','generation']).any() and not result.duplicated(['m','side','rung','generation']).any(),'duplicated bounded action')
    C.require((result.known_inventory.abs()<=F.CAP+1e-7).all(),'bounded diagnostic support failure')
    candidates.to_parquet(OUT/'candidates.parquet',index=False);result.to_parquet(OUT/'probes.parquet',index=False)
    models={}
    for name,col in [('joint','predicted_advantage_cents'),('empirical','empirical_advantage_cents')]:
        pred=result[col];actual=result.realized_advantage_cents;positive=pred>0
        models[name]=dict(n=len(result),spearman=float(pred.corr(actual,method='spearman')),predicted_mean_cents=float(pred.mean()),
            actual_mean_cents=float(actual.mean()),positive_predictions=int(positive.sum()),
            positive_predicted_mean_cents=float(pred[positive].mean()),positive_realized_mean_cents=float(actual[positive].mean()))
    report=dict(asof=C.stamp(),markets=len(set(result.m)),generated_actions=len(candidates),candidate_statuses=candidates.status.value_counts().to_dict(),
        selected_actions=int(candidates.selected.sum()),evaluation_statuses=dict(stats),evaluated_actions=len(result),
        outside_controller_net_cap=int((result.known_inventory.abs()>F.CAP+1e-7).sum()),max_abs_known_inventory=float(result.known_inventory.abs().max()),
        models=models,quantity=dict(predicted_mean=float(result.predicted_keep_quantity.mean()),actual_mean=float(result.actual_keep_quantity.mean()),
            mae=float(np.abs(result.predicted_keep_quantity-result.actual_keep_quantity).mean())),
        components=dict(predicted_wealth_advantage_mean=float(result.predicted_wealth_advantage_cents.mean()),actual_wealth_advantage_mean=float(result.realized_wealth_advantage_cents.mean()),
            predicted_risk_advantage_mean=float(result.predicted_risk_advantage_cents.mean()),actual_risk_advantage_mean=float(result.realized_risk_advantage_cents.mean())),
        matched_null=RUN.matched_null(result),controls=flags,scope=declaration['limits'],supersedes=declaration['supersedes'])
    C.dump(OUT/'report.json',report)
    C.dump(OUT/'receipt.json',dict(asof=C.stamp(),source_manifest=C.sha(FREEZE),controller_freeze_sha256=C.sha(F.FREEZE),
        primary_replay_receipt_sha256=C.sha(F.OUT/'pilot/receipt.json'),outputs={p.name:C.sha(p) for p in OUT.iterdir() if p.is_file()},seconds=time.time()-start))
    print(json.dumps(C.clean(report)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','run']);args=parser.parse_args()
    freeze() if args.mode=='freeze' else run()
