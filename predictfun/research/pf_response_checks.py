"""Response representation controls and artifact-identity evaluation audit."""
import json
import pf_response_bridge as B
import pf_response_fit as MODEL
import pf_value_bridge_checks as OLD
import pf_flow_control_checks as CHECK
import pf_value_bridge_run as R
F,C,np,pd=B.F,B.C,B.np,B.pd


def selftest():
    result={}
    np.testing.assert_equal(B.category('q1000',[0,1,10,10.1,30,30.1,60,120,300,301]),[0,1,1,2,2,3,3,4,5,6])
    result['negative_volume_refused']=CHECK.refused(lambda:B.category('q1000',[-1]),'negative volume')
    result['nan_move_refused']=CHECK.refused(lambda:B.category('move',[np.nan]),'missing move')
    p=dict(bid=50.,spread=2.,marks=[51.,51.],trades=np.array([[80,0,50,5.],[200,0,49,40.],[800,1,55,400.]]))
    sig=B.signature(p,2.)
    C.require(sig[2]==1 and sig[3]==3 and sig[4]==3 and sig[5]==3 and sig[6]==0,'timing/depth positive control')
    q=dict(p,trades=p['trades'].copy());q['trades'][0,0]=300.
    changed=B.signature(q,2.);C.require(sig[2]!=changed[2] and sig[4]==changed[4],'early timing control')
    shifted=B.signature(p,4.);C.require(sig[1]!=shifted[1],'current spread relabel control')
    result['quantity_timing_depth_and_current_mid_controls']=True
    paths=[dict(base_weight=1.),dict(base_weight=1.)];labels=np.array([[0]*16,[1]*16])
    w,diag=B.project(paths,labels,[np.array([.9,.1])]*16)
    C.require(w[0]>.89 and np.isclose(w.sum(),1.) and (w>=0).all(),'projection positive control')
    result['bad_probability_refused']=CHECK.refused(lambda:B.project(paths,labels,[np.array([1.,1.])]*16),'invalid probability')
    result['unsupported_distribution_refused']=CHECK.refused(lambda:B.project(paths,labels,[np.array([0.,0.,1.])]*16),'unsupported mass')
    C.require(set(B.FEATURES)==set(C.MARKET+['response_depth','response_edge']),'future feature list')
    C.require(not set(B.FEATURES)&set([h[0] for h in B.HEADS.values()]+['own_quantity','value_cents','joint_class']),'future label input')
    result['future_labels_excluded_from_features']=True
    paths=B.PATH.load();reconciled=0
    for p in paths:
        sig=B.signature(p,p['spread'])
        for s in (0,1):
            j=s*len(B.specs());flow=sig[j+4];move=sig[j+1]
            coarseflow=0 if flow==0 else (1 if flow<=2 else 2)
            coarsemove=[0,0,0,1,2,3,4,4,4][move]
            C.require(5*coarseflow+coarsemove==p['classes'][s],'rich label does not reconcile to original')
        reconciled+=1
    result['native_donor_labels_reconciled']=reconciled
    result['inherited_execution_controls']=OLD.selftest()
    return result


def audit():
    B.verify();r,priors=MODEL.load();controls=selftest();inputs={};counts={}
    for day,h in r['data_receipts'].items():C.require(C.sha(C.OUT/day/'receipt.json')==h,'fit data identity changed')
    for day in (F.CAL,F.TEST):
        src=B.OUT/('actions_'+day);receipt=json.loads((src/'receipt.json').read_text());report=json.loads((src/'report.json').read_text())
        C.require(receipt['source_manifest']==C.sha(B.FREEZE) and receipt['fit_receipt_sha256']==C.sha(B.OUT/'fit/receipt.json'),'action source identity')
        C.require(C.sha(src/'values.parquet')==receipt['outputs']['values.parquet'],'action values changed')
        a=pd.read_parquet(src/'values.parquet');old=pd.read_parquet(B.V.OUT/('actions_'+day)/'values.parquet');old=old[old.case=='SOFT_JOINT'].set_index('uid').sort_index()
        for case,g in a.groupby('case'):
            g=g.set_index('uid').sort_index();C.require(g.index.equals(old.index),'action population changed')
            for col in ('decision_ms','generation','m','side','rung','actual','actual_wealth','actual_risk','actual_quantity'):
                C.require(np.allclose(g[col],old[col],equal_nan=True,rtol=0,atol=1e-10),'action label/identity changed '+col)
            ok=g[g.evaluation_status=='eligible']
            C.require(np.allclose(ok.predicted,ok.predicted_wealth+ok.predicted_risk,atol=1e-8),'predicted decomposition')
            C.require((ok.predicted_quantity>=-1e-7).all() and (ok.predicted_quantity<=30.+1e-7).all(),'own quantity impossible')
        good=a[a.evaluation_status=='eligible'];n=good.groupby('uid').case.nunique();common=good[good.uid.isin(n[n==len(B.CASES)].index)]
        for (case,kind),g in common.groupby(['case','kind']):
            computed=C.clean(R.action_metrics(g));C.require(computed==report['metrics'][case+':'+kind],'action report metrics differ')
        controls['corrupt_value_refused_'+day]=CHECK.refused(lambda:C.require(np.isclose(good.actual.iloc[0]+1.,good.actual_wealth.iloc[0]+good.actual_risk.iloc[0]),'corrupt label'),'corrupt value')
        counts[day]=dict(selected=a.uid.nunique(),common=common.uid.nunique());inputs[str(src/'receipt.json')]=C.sha(src/'receipt.json')
    src=B.OUT/'pilot';r=json.loads((src/'receipt.json').read_text());report=json.loads((src/'report.json').read_text())
    C.require(r['source_manifest']==C.sha(B.FREEZE),'pilot source identity')
    for n,h in r['outputs'].items():C.require(C.sha(src/n)==h,'pilot artifact changed')
    m=pd.read_parquet(src/'markets.parquet');fills=pd.read_parquet(src/'fills.parquet')
    for case,g in m.groupby('case'):
        C.require(len(g)==12 and (g.max_position<=F.CAP+1e-7).all(),'market population or cap')
        for _,row in g.iterrows():
            f=fills[(fills.case==case)&(fills.m==row.m)]
            summary=R.MET.summarize_fills(f)
            for k,v in summary.items():
                if isinstance(v,(int,float,np.number)):C.require(np.isclose(v,row[k],equal_nan=True,atol=1e-7),'fill accounting differs '+k)
        claimed=next(x for x in report['rows'] if x['case']==case)
        for key in ('terminal_pnl_usd','settlement_pnl_usd','shares','orders','pulls'):
            C.require(np.isclose(g[key].sum(),claimed[key]),'pilot report total differs '+key)
    old=pd.read_parquet(B.V.OUT/'pilot/markets.parquet');old=old[old.case=='SOFT_JOINT'].set_index('m').sort_index()
    new=m[m.case=='SOFT_JOINT'].set_index('m').sort_index();pd.testing.assert_frame_equal(old,new)
    controls['baseline_pilot_exact_identity']=True
    controls['corrupt_pilot_total_refused']=CHECK.refused(lambda:C.require(np.isclose(m.shares.sum()+1,m.shares.sum()),'wrong total'),'corrupt pilot')
    inputs[str(src/'receipt.json')]=C.sha(src/'receipt.json')
    C.dump(B.REVIEW/'audit_v1.json',dict(asof=C.stamp(),source_manifest=C.sha(B.FREEZE),controls=controls,actions=counts,pilot_markets=len(m),inputs=inputs))
    print(json.dumps(dict(audit='complete',actions=counts,pilot_markets=len(m))),flush=True)
