"""Collect, decompose and audit fixed neutral actions; research diagnostics only."""
import argparse
import gc
import json
import time
import pf_value_decomposition as D
import pf_conditional_map_bank as BANK
import pf_value_bridge_run as MET
import pf_flow_control_run as RUN
F,C,np,pd,I,B,V,A=D.F,D.C,D.np,D.pd,D.I,D.B,D.V,D.A


def input_rows(day):
    src=B.OUT/('actions_'+day);r=json.loads((src/'receipt.json').read_text())
    C.require(r['source_manifest']==C.sha(B.FREEZE),'archived action contract')
    for p,h in r['outputs'].items():C.require(C.sha(src/p)==h,'archived action changed '+p)
    rows=pd.read_parquet(src/'values.parquet');oracle=pd.read_parquet(src/'oracle.parquet') if day==F.TEST else None
    return rows,oracle,r


def collect(day):
    D.verify();C.require(day in (F.CAL,F.TEST),'undeclared day');start=time.time()
    out=D.OUT/day;C.require(not out.exists(),'decomposition day exists');out.mkdir(parents=True)
    previous,previous_oracle,receipt=input_rows(day);targets=previous[previous.case=='SOFT_JOINT']
    archived={(r['uid'],r['case'],'ordinary'):r for r in previous.to_dict('records')}
    if previous_oracle is not None:archived.update({(r['uid'],r['case'],'true_pair'):r for r in previous_oracle.to_dict('records')})
    selected,data,fair,depth,raw=A.load(day);C.require(selected==receipt['selected_markets'],'market selection differs')
    paths=B.PATH.load();bank=BANK.load();rows=[];truth=[];controls={}
    for W in I.pairs(data):
        m=int(W[0]['m']);local=targets[targets.m==m];bytime={float(t):g for t,g in local.groupby('decision_ms')}
        X,_=I.market_inputs(W,fair[m],depth);prob=V.P.forecasts(W,X)
        pols={case:B.Controller(W,paths,bank,prob,X,case) for case in B.CASES}
        with A.CLOCK.clocks(58.,37.):reference=A.AC.replay_neutral(W,size=30.,shift_ms=50.,queue='M',nr=6,cap=F.CAP,mode='hard')
        def observe(k,t,ST,kpos):
            if t not in bytime:return
            state=V.P.snapshot(W,k,t,ST,kpos);j=int(W[0]['jk'][k]);bid=float(W[0]['P5'][j,0]);ask=float(W[1]['P5'][j,0]);mid=(bid+ask)/2
            path,path_status=B.PATH.extract(W,t,day);perfect=None
            if path is not None:
                C.require(path['bid']==bid and path['spread']==ask-bid,'actual future starting book')
                packed=B.K.pack([path],bid,1000.);B.K.current_book(packed,[W[s]['P5'][j] for s in (0,1)],[W[s]['S5'][j] for s in (0,1)])
                perfect=(packed,np.ones(1),mid,dict(unsupported_mass=0.,marginal_l1=0.,effective_paths=1.))
            true_pair=None
            if day==F.TEST:
                pair=[]
                for side,price in ((0,bid),(1,ask)):
                    z=A.DATA.public_target(W[side],t,price,mid);pair.append(int(C.classes(z['flow_quantity'],z['move_cents'])[2]))
                true_pair=15*pair[0]+pair[1]
            contexts={}
            for case,pol in pols.items():
                contexts[case,'ordinary']=pol.context(k,t)
                if true_pair is not None:contexts[case,'true_pair']=pol.context(k,t,known_pair=true_pair)
            contexts['ACTUAL_PATH_CLIENT','actual_path']=(perfect,path_status)
            for row in bytime[t].to_dict('records'):
                indices=[i for i,x in enumerate(state['info']) if x['kind']==row['kind'] and x['side']==row['side'] and x['rung']==row['rung'] and (row['kind']=='post' or abs(x['activation']-(t-row['x_age_ms']))<.001)]
                C.require(len(indices)==1 and np.isclose(kpos,row['known_inventory']),'causal action identity')
                index=indices[0];price=state['orders'][index,2]
                actual=D.truth(W,row,reference,mid,price)
                keys=('uid','day','m','side','rung','generation','phase','kind','decision_ms','x_age_ms','known_inventory','actual','actual_quantity','status')
                base={key:row[key] for key in keys};base.update(actual,price=price,mid=mid,
                    known_ahead=state['orders'][index,4],known_remaining=state['orders'][index,3],seconds_left=(W[0]['ws']+300000.-t)/1000.)
                truth.append(base)
                for (case,channel),(ctx,status) in contexts.items():
                    result=dict(base,case=case,channel=channel,evaluation_status=status)
                    if ctx is not None:result.update(D.estimate(state,index,row['kind'],ctx))
                    if channel!='actual_path':
                        old=archived[row['uid'],case,channel];C.require(status==old['evaluation_status'],'map support changed')
                        if ctx is not None:
                            for field in ('predicted','predicted_wealth','predicted_risk','predicted_quantity'):
                                C.require(np.isclose(result[field],old[field],rtol=1e-10,atol=1e-7),'archived prediction changed '+field)
                    rows.append(result)
        replay=RUN.run_lane(W,cap=F.CAP,observer=observe)
        for field in ('fl','stat','orders','cancels'):np.testing.assert_equal(replay[field],reference[field])
        controls[str(m)]=dict(canonical_prefixes_reconciled=True,archived_labels_and_predictions_reproduced=True)
        del pols,prob,reference,replay;gc.collect();C.guard('value decomposition')
        print(json.dumps(dict(stage='decompose',day=day,m=m,seconds=round(time.time()-start))),flush=True)
    g=pd.DataFrame(rows);a=pd.DataFrame(truth)
    C.require(not a.uid.duplicated().any() and set(a.uid)==set(targets.uid),'truth action identities')
    C.require(not g.duplicated(['uid','case','channel']).any(),'duplicated prediction identity')
    for _,z in g.groupby(['case','channel']):C.require(set(z.uid)==set(targets.uid),'missing action status')
    g.to_parquet(out/'values.parquet',index=False);a.to_parquet(out/'actual.parquet',index=False)
    C.dump(out/'receipt.json',dict(asof=C.stamp(),source_manifest=C.sha(D.FREEZE),day=day,selected_markets=selected,
        input_receipt_sha256=C.sha(B.OUT/('actions_'+day)/'receipt.json'),controls=controls,
        outputs={p.name:C.sha(p) for p in out.glob('*.parquet')},seconds=time.time()-start))
    report(day)


def rmse(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))


def metrics(g):
    e=g.predicted.to_numpy()-g.actual.to_numpy();den=float(e@e);parts={};signed=0.
    for name in D.PARTS:
        z=g['pred_'+name].to_numpy()-g['actual_'+name].to_numpy();v=float(z@e/den) if den>0 else None
        parts[name]=dict(rmse_cents=rmse(z),mean_error_cents=float(z.mean()),signed_mse_share=v)
        if v is not None:signed+=v
    C.require(den==0 or np.isclose(signed,1.,rtol=0,atol=1e-7),'component MSE attribution lost')
    orientation=np.where(g.kind=='post',1.,-1.);retained=orientation*g.actual.to_numpy();tail=retained < -30.
    return dict(n=len(g),rmse_cents=rmse(e),mae_cents=float(np.abs(e).mean()),spearman=float(g.predicted.corr(g.actual,method='spearman')),
        zero_prediction_rmse_cents=rmse(g.actual),components=parts,
        affected_quantity_mae_shares=float((g.pred_affected_quantity-g.actual_affected_quantity).abs().mean()),
        legacy_quantity_mae_shares=float((g.predicted_quantity-g.actual_quantity).abs().mean()),
        no_actual_affected_fill=int((g.actual_affected_quantity<=1e-9).sum()),
        focal_motion_rmse_cents=rmse(g.pred_focal_motion-g.actual_focal_motion),
        other_wealth_rmse_cents=rmse(g.pred_other_edge+g.pred_other_motion-g.actual_other_edge-g.actual_other_motion),
        actual_other_wealth_nonzero=int(((g.actual_other_edge+g.actual_other_motion).abs()>1e-9).sum()),
        mean_abs_focal_covariance=float(g.pred_focal_covariance.abs().mean()),
        severe_retained_loss_n=int(tail.sum()),severe_retained_loss_sse_share=float((e[tail]@e[tail])/den) if den>0 else None,
        matched_null=MET.matched_null(g))


def replacement(g,mode):
    C.require(mode in D.SUBSTITUTIONS,'unknown diagnostic substitution')
    p=g.predicted.to_numpy().copy();status=np.full(len(g),'eligible',object);orientation=np.where(g.kind=='post',1.,-1.)
    if mode.startswith('true_') and mode[5:] in D.PARTS:
        name=mode[5:];p+=(g['actual_'+name]-g['pred_'+name]).to_numpy()
    elif mode=='true_other_wealth':p+=(g.actual_other_edge+g.actual_other_motion-g.pred_other_edge-g.pred_other_motion).to_numpy()
    elif mode=='independent_focal_fill_move':p+=orientation*g.pred_affected_quantity.to_numpy()*g.pred_focal_move.to_numpy()-g.pred_focal_motion.to_numpy()
    elif mode=='true_endpoint_all_fills':p=(g.pred_edge+g.pred_delta_position*g.actual_endpoint_move+g.pred_risk).to_numpy()
    elif mode=='true_focal_quantity':
        ok=g.pred_affected_quantity.to_numpy()>1e-12;actual=g.actual_affected_quantity.to_numpy()
        status[~ok & (actual>1e-9)]='zero_predicted_affected_fill_support'
        move=np.where(ok,g.pred_focal_fill_move.to_numpy(),0.);edge=(1-2*g.side.to_numpy())*(g.mid-g.price).to_numpy()
        focal=orientation*actual*(edge+move)
        p+=focal-g.pred_focal_edge.to_numpy()-g.pred_focal_motion.to_numpy()
    return p,status


def diagnostic_selftest():
    g=pd.DataFrame([dict(kind='post',side=0,mid=51.,price=50.,predicted=-11.,
        pred_edge=10.,pred_motion=-20.,pred_risk=-1.,actual_edge=20.,actual_motion=-60.,actual_risk=-2.,
        pred_focal_edge=10.,pred_focal_motion=-20.,pred_other_edge=0.,pred_other_motion=0.,
        actual_other_edge=0.,actual_other_motion=0.,pred_affected_quantity=10.,actual_affected_quantity=20.,
        pred_focal_move=-1.,pred_focal_fill_move=-2.,pred_delta_position=10.,actual_endpoint_move=-3.)])
    expected=dict(original=-11.,true_edge=-1.,true_motion=-51.,true_risk=-12.,true_other_wealth=-11.,
        independent_focal_fill_move=-1.,true_endpoint_all_fills=-21.,true_focal_quantity=-21.)
    for mode,value in expected.items():
        p,status=replacement(g,mode);C.require(p[0]==value and status[0]=='eligible','substitution control '+mode)
    g['pred_affected_quantity']=0.;g['pred_focal_fill_move']=np.nan
    _,status=replacement(g,'true_focal_quantity');C.require(status[0]=='zero_predicted_affected_fill_support','missing fill support invented')
    return dict(component_substitutions_reconciled=True,missing_fill_support_reported=True,
        unknown_substitution_refused=D.CHECK.refused(lambda:replacement(g,'unknown'),'unknown mode'))


def summarize(g):
    good=g[g.evaluation_status=='eligible'].copy();out={};by_side={};sub={};tails={};worst=[]
    for (channel,case,kind),z in good.groupby(['channel','case','kind']):
        key=f'{channel}:{case}:{kind}';out[key]=metrics(z)
        retained=np.where(z.kind=='post',1.,-1.)*z.actual.to_numpy()
        labels=np.select([retained < -30.,retained < -1e-9,np.abs(retained)<=1e-9,retained<=30.],
            ['severe_loss','loss','zero','gain'],default='severe_gain')
        tails[key]={}
        for name in ('severe_loss','loss','zero','gain','severe_gain'):
            q=z[labels==name]
            tails[key][name]=dict(n=len(q),rmse_cents=rmse(q.predicted-q.actual) if len(q) else None,
                actual_value_sum_cents=float(q.actual.sum()),predicted_value_sum_cents=float(q.predicted.sum()))
        for mode in D.SUBSTITUTIONS:
            p,st=replacement(z,mode);ok=st=='eligible';q=z[ok].copy();q['predicted']=p[ok]
            sub[key+':'+mode]=dict(n=len(q),statuses=pd.Series(st).value_counts().to_dict(),
                rmse_cents=rmse(q.predicted-q.actual) if len(q) else None,
                original_rmse_same_support=rmse(z.loc[ok,'predicted']-z.loc[ok,'actual']) if len(q) else None,
                matched_null=MET.matched_null(q) if len(q) else None)
        w=z.assign(squared_error=(z.predicted-z.actual)**2).sort_values('squared_error',ascending=False).head(10)
        worst.extend(w.to_dict('records'))
    for (channel,case,kind,side),z in good.groupby(['channel','case','kind','side']):by_side[f'{channel}:{case}:{kind}:{side}']=metrics(z)
    return dict(metrics=out,by_side=by_side,substitutions=sub,tails=tails),pd.DataFrame(worst)


def report(day):
    src=D.OUT/day;r=json.loads((src/'receipt.json').read_text())
    for p,h in r['outputs'].items():C.require(C.sha(src/p)==h,'decomposition output changed')
    g=pd.read_parquet(src/'values.parquet');summary,worst=summarize(g)
    prior=json.loads((B.OUT/('actions_'+day)/'report.json').read_text())
    C.dump(src/'report.json',dict(asof=C.stamp(),day=day,selected_actions=g.uid.nunique(),markets=len(r['selected_markets']),
        statuses=g.groupby(['channel','case','evaluation_status']).size().to_dict(),
        source_candidate_statuses=prior['candidate_statuses'],**summary,limits=D.verify()['limits'],receipt_sha256=C.sha(src/'receipt.json')))
    worst.to_parquet(src/'largest_errors.parquet',index=False)
    C.dump(src/'report_receipt.json',dict(asof=C.stamp(),source_manifest=C.sha(D.FREEZE),
        outputs={p:C.sha(src/p) for p in ('report.json','largest_errors.parquet')}))
    print(json.dumps(dict(stage='report',day=day,rows=len(g))),flush=True)


def audit():
    D.verify();checks=dict(**D.selftest(),**diagnostic_selftest());files=[];counts={}
    for day in (F.CAL,F.TEST):
        src=D.OUT/day;r=json.loads((src/'receipt.json').read_text());rr=json.loads((src/'report_receipt.json').read_text())
        C.require(r['source_manifest']==rr['source_manifest']==C.sha(D.FREEZE),'diagnostic source identity')
        for p,h in {**r['outputs'],**rr['outputs']}.items():C.require(C.sha(src/p)==h,'diagnostic artifact changed')
        g=pd.read_parquet(src/'values.parquet');actual=pd.read_parquet(src/'actual.parquet').set_index('uid').sort_index()
        old,oracle,_=input_rows(day);old=old[old.case=='SOFT_JOINT'].set_index('uid').sort_index()
        C.require(actual.index.equals(old.index),'canonical action identity')
        for field in ('m','side','rung','generation','decision_ms','actual','actual_wealth','actual_risk','actual_quantity'):
            C.require(np.allclose(actual[field],old[field],rtol=0,atol=1e-7),'actual artifact changed '+field)
        good=g[g.evaluation_status=='eligible']
        for prefix in ('pred_','actual_'):
            D.reconcile({name:good[prefix+name].to_numpy() for name in ('value','wealth','risk','edge','motion','focal_edge','focal_motion','other_edge','other_motion','affected_quantity')})
        stored=json.loads((src/'report.json').read_text());recomputed,worst=summarize(g)
        for key,value in recomputed.items():C.require(C.clean(value)==stored[key],'summary differs '+key)
        pd.testing.assert_frame_equal(worst,pd.read_parquet(src/'largest_errors.parquet'))
        bad=dict(value=np.array([1.]),wealth=np.array([0.]),risk=np.array([0.]),edge=np.array([0.]),motion=np.array([0.]),
            focal_edge=np.array([0.]),other_edge=np.array([0.]),focal_motion=np.array([0.]),other_motion=np.array([0.]),affected_quantity=np.array([0.]))
        checks['corrupt_artifact_refused_'+day]=D.CHECK.refused(lambda:D.reconcile(bad),'wrong total')
        counts[day]=dict(actions=len(actual),prediction_rows=len(g),eligible_rows=len(good))
        files += [day+'/'+name for name in ('receipt.json','report.json','report_receipt.json')]
    C.dump(D.REVIEW/'audit_v1.json',dict(asof=C.stamp(),source_manifest=C.sha(D.FREEZE),controls=checks,counts=counts,
        canonical_replays_and_original_predictions_reproduced=True))
    for name in files:
        p=D.REVIEW/name;C.require(not p.exists(),'review output exists');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((D.OUT/name).read_bytes())
    C.dump(D.REVIEW/'index.json',dict(asof=C.stamp(),source_manifest=C.sha(D.FREEZE),
        outputs={p:C.sha(D.REVIEW/p) for p in files+['audit_v1.json']},limits=D.verify()['limits']))
    print(json.dumps(dict(audit='complete',counts=counts)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','selftest','collect','audit']);p.add_argument('day',nargs='?');a=p.parse_args()
    if a.mode=='freeze':D.freeze()
    elif a.mode=='selftest':
        import pf_conditional_map_checks as OLD
        print(json.dumps(dict(decomposition=D.selftest(),substitutions=diagnostic_selftest(),inherited=OLD.selftest())),flush=True)
    elif a.mode=='collect':collect(a.day)
    else:audit()
