"""Positive controls, known-bad refusals, and source/label/metric reconciliation."""
import json
import pf_conditional_map as B
import pf_conditional_map_bank as BANK
import pf_value_bridge_checks as OLD
import pf_flow_control_checks as CHECK
import pf_value_bridge_run as R
F,C,np,pd=B.F,B.C,B.np,B.pd


def selftest():
    prices=[np.arange(50,45,-1),np.arange(52,57)];q=[np.ones(5)*30,np.ones(5)*60]
    x=B.geometry(prices,q,30.,[0.]*len(B.DYNAMICS));y=B.geometry(prices,[v*10 for v in q],300.,[0.]*len(B.DYNAMICS))
    C.require(not np.array_equal(x,y),'absolute liquidity states collapse')
    norm=dict(median=np.zeros(len(x)),iqr=np.ones(len(x)));z=B.normalize(np.stack([x,y]),norm)
    C.require(B.distance(z,z[0])[0]==0 and B.distance(z,z[0])[1]>0,'distance positive control')
    missing=x.copy();missing[-1]=np.nan
    C.require(B.distance(B.normalize(missing,norm),B.normalize(x,norm))>0,'missing input loses identity')
    mass=np.zeros(225);mass[:2]=[.3,.7];groups=np.array([0,0,1,1])
    w,error=B.anchored(np.array([0.,-100.,-100.,0.]),groups,mass)
    C.require(error<1e-12 and np.allclose(w[[0,3]],[.3,.7]),'conditional response cannot change while mass fixed')
    checks=dict(absolute_liquidity_separable=True,conditional_variation_with_exact_event_mass=True,
        unsupported_anchor_refused=CHECK.refused(lambda:B.anchored(np.zeros(2),np.zeros(2,int),mass),'lost event'),
        bad_probability_refused=CHECK.refused(lambda:B.anchored(np.zeros(4),groups,-mass),'negative mass'),
        invalid_size_refused=CHECK.refused(lambda:B.geometry(prices,[np.ones(5)*-1,q[1]],30.,[0.]*len(B.DYNAMICS)),'negative depth'))
    paths=[CHECK.toy([(100,0,50,10)],marks=(51.,51.)),CHECK.toy([(500,0,50,30)],marks=(51.,51.))]
    g=B.geometry([paths[0]['frames'][0,2:7],paths[0]['frames'][0,12:17]],[paths[0]['frames'][0,7:12],paths[0]['frames'][0,17:22]],30.,[0.]*len(B.DYNAMICS))
    paths[0]['trades']*=2;paths[0]['frames'][1:]*=2;paths[0]['marks']=[0.,100.]
    unchanged=B.geometry([paths[0]['frames'][0,2:7],paths[0]['frames'][0,12:17]],[paths[0]['frames'][0,7:12],paths[0]['frames'][0,17:22]],30.,[0.]*len(B.DYNAMICS))
    C.require(np.array_equal(g,unchanged),'future path changed state geometry');checks['future_path_not_a_conditioning_feature']=True
    zz=np.zeros((12,2*len(B.FEATURES)));zz[:,0]=np.arange(12)
    chosen,count=BANK.representatives(zz,[0,1,2],8)
    C.require(set([0,1,2])<=set(chosen) and len(chosen)==8 and count.sum()==12,'representative coverage/weights')
    checks['representative_weight_conservation']=True
    _,tied=BANK.representatives(np.zeros_like(zz),[0,1,2],8)
    C.require(np.allclose(tied,np.full(8,1.5)),'identical states collapse to one arbitrary future')
    checks['identical_states_split_mass_equally']=True
    checks['inherited_execution']=OLD.selftest()
    return checks


def audit():
    B.verify();checks=selftest();bank=BANK.load();br=json.loads((B.OUT/'bank/receipt.json').read_text());old={p['uid']:p for p in B.PATH.load()}
    current={p['uid']:p for p in bank['paths']};C.require(set(old)<=set(current),'legacy bank not retained')
    for uid,p in old.items():BANK.same_path(p,current[uid])
    C.require(np.isclose(sum(p['base_weight'] for p in current.values()),br['selected_times'],rtol=0,atol=1e-8),'training exposure mass lost')
    inputs={};counts={}
    for day in (F.CAL,F.TEST):
        src=B.OUT/('actions_'+day);receipt=json.loads((src/'receipt.json').read_text())
        C.require(receipt['source_manifest']==C.sha(B.FREEZE) and receipt['bank_receipt_sha256']==C.sha(B.OUT/'bank/receipt.json'),'action source identity')
        for name,h in receipt['outputs'].items():C.require(C.sha(src/name)==h,'action artifact changed')
        previous=pd.read_parquet(B.V.OUT/('actions_'+day)/'values.parquet');previous=previous[previous.case=='SOFT_JOINT'].set_index('uid').sort_index()
        for filename,reportname in [('values.parquet','report.json')]+([('oracle.parquet','oracle_report.json')] if day==F.TEST else []):
            a=pd.read_parquet(src/filename);report=json.loads((src/reportname).read_text())
            for case,g in a.groupby('case'):
                g=g.set_index('uid').sort_index();C.require(g.index.equals(previous.index),'action identity lost')
                for key in ('m','side','rung','generation','decision_ms','actual','actual_wealth','actual_risk','actual_quantity'):
                    C.require(np.allclose(g[key],previous[key],rtol=0,atol=1e-10),'label changed '+key)
                ok=g[g.evaluation_status=='eligible']
                C.require(np.allclose(ok.predicted,ok.predicted_wealth+ok.predicted_risk,atol=1e-8),'value decomposition')
                C.require((ok.predicted_quantity>=-1e-7).all() and (ok.predicted_quantity<=30.+1e-7).all(),'fill bound')
                if case!='SOFT_JOINT':C.require((ok.anchor_pair_l1<=1e-12).all(),'forecast event mass changed')
            good=a[a.evaluation_status=='eligible'];n=good.groupby('uid').case.nunique();common=good[good.uid.isin(n[n==len(B.CASES)].index)]
            for (case,kind),g in common.groupby(['case','kind']):C.require(C.clean(R.action_metrics(g))==report['metrics'][case+':'+kind],'metric reproduction')
            counts[day+':'+filename]=dict(selected=a.uid.nunique(),common=common.uid.nunique())
            checks['wrong_value_refused_'+day+filename]=CHECK.refused(lambda:C.require(np.isclose(good.actual.iloc[0]+1.,good.actual_wealth.iloc[0]+good.actual_risk.iloc[0]),'corrupt value'),'wrong value')
        inputs[str(src/'receipt.json')]=C.sha(src/'receipt.json')
    import pf_conditional_map_eval as E
    gate=json.loads((B.OUT/'gate.json').read_text());expected=E.gate()
    for key in ('run_pilot','comparisons','source_manifest','oracle_report_sha256'):C.require(gate[key]==expected[key],'gate changed')
    pilot_status='skipped_by_frozen_gate'
    if gate['run_pilot']:
        src=B.OUT/'pilot';r=json.loads((src/'receipt.json').read_text())
        for name,h in r['outputs'].items():C.require(C.sha(src/name)==h,'pilot artifact changed')
        m=pd.read_parquet(src/'markets.parquet');f=pd.read_parquet(src/'fills.parquet')
        for _,row in m.iterrows():
            summary=R.MET.summarize_fills(f[(f.case==row.case)&(f.m==row.m)])
            for key,v in summary.items():
                if isinstance(v,(int,float,np.number)):C.require(np.isclose(v,row[key],equal_nan=True,atol=1e-7),'pilot fill sum '+key)
            C.require(row.max_position<=F.CAP+1e-7,'pilot cap')
        C.require(len(m)==24 and m.groupby('case').m.nunique().eq(12).all(),'pilot population');pilot_status='complete_audited'
        baseline=pd.read_parquet(B.V.OUT/'pilot/markets.parquet');baseline=baseline[baseline.case=='SOFT_JOINT'].set_index('m').sort_index()
        pd.testing.assert_frame_equal(baseline,m[m.case=='SOFT_JOINT'].set_index('m').sort_index())
        inputs[str(src/'receipt.json')]=C.sha(src/'receipt.json')
    else:C.require(not (B.OUT/'pilot').exists(),'replay ran after failed gate')
    C.dump(B.REVIEW/'audit_v1.json',dict(asof=C.stamp(),source_manifest=C.sha(B.FREEZE),controls=checks,actions=counts,
        bank=dict(retained_legacy=len(old),representatives=len(current)),pilot_status=pilot_status,inputs=inputs))
    # Immutable small review copies; parquets and models remain ignored.
    files=['bank/receipt.json','gate.json']
    for day in (F.CAL,F.TEST):files += ['actions_'+day+'/'+n for n in ('report.json','receipt.json')]
    files+=['actions_'+F.TEST+'/oracle_report.json']
    if gate['run_pilot']:files+=['pilot/report.json','pilot/receipt.json']
    for name in files:
        dest=B.REVIEW/name;C.require(not dest.exists(),'review artifact exists');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((B.OUT/name).read_bytes())
    C.dump(B.REVIEW/'index.json',dict(asof=C.stamp(),source_manifest=C.sha(B.FREEZE),pilot_status=pilot_status,
        outputs={name:C.sha(B.REVIEW/name) for name in files+['audit_v1.json']},scope=B.verify()['limits']))
    print(json.dumps(dict(audit='complete',actions=counts,pilot=pilot_status)),flush=True)
