"""Positive and known-bad controls for the economic mapping and its evidence."""
import copy
import json
from pathlib import Path
import pf_value_bridge as V
import pf_flow_control_checks as OLD
F,C,np,pd,K=V.F,V.C,V.np,V.pd,V.K


def selftest():
    checks=OLD.selftest();paths=V.PATH.load()
    for p in paths:C.require(V.labels(p,p['spread'])==p['classes'],'own-spread relabel differs from raw target')
    checks['all_donor_labels_reconciled']=len(paths)
    p=OLD.toy([(50.,1.,54.,20.)],marks=(51.,51.),depth=0.);p['uid']='toy';p['classes']=V.labels(p,2.)
    old=p['classes'];new=V.labels(p,5.)
    C.require(old[1]//5==1 and new[1]//5==0,'ask reach did not change at current ask')
    checks['ask_reach_relabels_at_current_price']=True
    checks['stale_donor_label_refused']=OLD.refused(lambda:C.require(old==new,'stale donor labels'),'stale classes')
    a=copy.deepcopy(p);b=copy.deepcopy(p);b.update(uid='second',spread=3.)
    pa=V.pool([a,b],1,50.,2.);pb=V.pool([a,b],1,50.,3.)
    C.require({z['uid'] for z in pa}=={z['uid'] for z in pb}=={'toy','second'},'spread replaced support')
    C.require(pa[0]['base_weight']>pa[1]['base_weight'] and pb[0]['base_weight']<pb[1]['base_weight'],'spread prior is not conditional')
    near=V.pool([a,b],1,50.,2.+1e-6)
    C.require(max(abs(z['base_weight']-x['base_weight']) for z,x in zip(pa,near))<2e-6,'discontinuous spread prior')
    checks['shared_support_and_continuous_prior']=True
    checks['invalid_spread_refused']=OLD.refused(lambda:V.pool([a],1,99.,3.),'payoff query')
    # A simple post/wait kernel pair carries nonzero economic value; flipping
    # price direction must flip the sign. Quantity/cash accounting checked above.
    o=np.array([[0,0,50.,30.,0.,58.,1e30,0.,1.]])
    gain=OLD.toy([(80.,0.,50.,10.)],marks=(52.,52.),depth=0.)
    loss=OLD.toy([(80.,0.,50.,10.)],marks=(48.,48.),depth=0.)
    for p,sgn in [(gain,1),(loss,-1)]:
        z,q=K.evaluate(K.pack([p],50.,1000.),o,[[0.],[1.]],0.,51.,1000.)
        value=V.P.utility(z,0.)[0,1]-V.P.utility(z,0.)[0,0]
        C.require(value*sgn>0 and q[0,1,0]>0,'post value sign control')
    checks['positive_and_negative_post_value']=True
    return checks


def action_checks(df,candidates):
    keys=['day','m','kind','side','rung','generation']
    C.require(not candidates.duplicated(keys).any() and not df.duplicated(keys+['case']).any(),'duplicate action identity')
    selected=candidates[candidates.selected];identity=set(selected[keys].itertuples(index=False,name=None))
    for case in V.CASES:C.require(set(df[df.case==case][keys].itertuples(index=False,name=None))==identity,'selected action identity differs')
    C.require(selected.status.isin(['eligible','eligible_estimated_timing']).all(),'invalid selected action')
    C.require(np.isfinite(df.actual).all() and (df.known_inventory.abs()<=F.CAP+1e-7).all(),'invalid realized value or inventory')
    for prefix in ('actual','predicted'):
        g=df if prefix=='actual' else df[df.evaluation_status=='eligible']
        C.require(np.allclose(g[prefix],g[prefix+'_wealth']+g[prefix+'_risk'],atol=1e-7),'value decomposition '+prefix)
        C.require(g[prefix+'_quantity'].between(0.,30.+1e-7).all(),'quantity bound '+prefix)
    C.require((df.groupby('uid').actual.nunique()==1).all(),'different actual labels across models')


def audit():
    import pf_value_bridge_run as R
    V.verify();controls=selftest();summaries={}
    for day in ('20261026','20261027'):
        out=V.OUT/('actions_'+day);r=json.loads((out/'receipt.json').read_text())
        C.require(r['source_manifest']==C.sha(V.FREEZE),'wrong action freeze')
        for p,h in r['outputs'].items():C.require(C.sha(out/p)==h,'action artifact changed')
        df=pd.read_parquet(out/'values.parquet');c=pd.read_parquet(out/'candidates.parquet');action_checks(df,c)
        bad=df.copy();bad.loc[bad.index[0],'actual']+=1.
        controls[day+'_corrupt_value_refused']=OLD.refused(lambda:action_checks(bad,c),'corrupt economic label')
        report=json.loads((out/'report.json').read_text());eligible=df[df.evaluation_status=='eligible'];n=eligible.groupby('uid').case.nunique();matched=eligible[eligible.uid.isin(n[n==len(V.CASES)].index)]
        for (case,kind),g in matched.groupby(['case','kind']):C.require(C.clean(R.action_metrics(g))==report['metrics'][case+':'+kind],'action metrics/null mismatch')
        summaries[day]=dict(selected_actions=int(c.selected.sum()),generated_actions=len(c),report_sha256=C.sha(out/'report.json'),receipt_sha256=C.sha(out/'receipt.json'))
    out=V.OUT/'pilot';r=json.loads((out/'receipt.json').read_text())
    C.require(r['source_manifest']==C.sha(V.FREEZE),'wrong replay freeze')
    C.require(C.sha(r['compiled_kernel']['path'])==r['compiled_kernel']['sha256'],'changed compiled kernel')
    for name,h in r['outputs'].items():C.require(C.sha(out/name)==h,'replay artifact changed')
    f=pd.read_parquet(out/'fills.parquet');m=pd.read_parquet(out/'markets.parquet');original=pd.read_parquet(F.OUT/'pilot/markets.parquet')
    def reconcile(markets):
        for row in markets.itertuples():
            g=f[(f.case==row.case)&(f.m==row.m)]
            C.require(np.isclose(g.shares.sum(),row.shares) and np.isclose(g.terminal_pnl_usd.sum(),row.terminal_pnl_usd) and np.isclose(g.settlement_pnl_usd.sum(),row.settlement_pnl_usd),'fill P&L reconciliation')
            C.require(row.max_position<=F.CAP+1e-7,'inventory cap')
    reconcile(m);bad=m.copy();bad.loc[bad.index[0],'terminal_pnl_usd']+=1.
    controls['corrupt_portfolio_summary_refused']=OLD.refused(lambda:reconcile(bad),'corrupt P&L')
    base=m[m.case=='EXACT_JOINT'].copy();base['case']='JOINT';old=original[original.case=='JOINT']
    pd.testing.assert_frame_equal(base[old.columns].reset_index(drop=True),old.reset_index(drop=True),check_dtype=False)
    controls['exact_baseline_artifact_preserved']=True
    C.require(not (V.OUT/'audit_v1.json').exists(),'bridge audit exists')
    result=dict(asof=C.stamp(),source_manifest=C.sha(V.FREEZE),controls=controls,action_reports=summaries,
        pilot_markets=len(m),fills=len(f),pilot_report_sha256=C.sha(out/'report.json'),pilot_receipt_sha256=C.sha(out/'receipt.json'))
    C.dump(V.OUT/'audit_v1.json',result);print(json.dumps(C.clean(result)),flush=True)
