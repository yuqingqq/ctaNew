"""Matched neutral action calibration and continuous controller comparison."""
import argparse
from collections import Counter
import gc
import json
import time
import pf_conditional_map as B
import pf_conditional_map_bank as BANK
import pf_value_bridge_actions as A
import pf_value_bridge_run as R
import pf_flow_control_run as RUN
import pf_stop_ev_minutes_metrics as MET
import pf_stop_ev_minutes as AUDIT
F,C,np,pd,I,V=B.F,B.C,B.np,B.pd,B.I,B.V


def archived(subdir):
    src=V.OUT/subdir;r=json.loads((src/'receipt.json').read_text())
    for p,h in r['outputs'].items():C.require(C.sha(src/p)==h,'archived artifact changed '+p)
    return src,r


def actions(day):
    B.verify();C.require(day in (F.CAL,F.TEST),'undeclared action date')
    start=time.time();out=B.OUT/('actions_'+day);C.require(not out.exists(),'action output exists');out.mkdir(parents=True)
    src,receipt=archived('actions_'+day);old=pd.read_parquet(src/'values.parquet')
    targets=old[old.case=='SOFT_JOINT'].copy();selected,data,fair,depth,raw=A.load(day)
    C.require(selected==receipt['selected_markets'],'action markets changed');paths=B.PATH.load();bank=BANK.load();results=[];controls={};oracle=[];old_oracle={}
    if day==F.TEST:
        source=V.OUT/'oracle_v1';rr=json.loads((source/'receipt.json').read_text())
        C.require(C.sha(source/'values.parquet')==rr['outputs']['values.parquet'],'original oracle changed')
        oo=pd.read_parquet(source/'values.parquet');old_oracle={r['uid']:r for r in oo[oo.case=='SOFT_JOINT_TRUE_CLASS'].to_dict('records')}
    for W in I.pairs(data):
        m=int(W[0]['m']);local=targets[targets.m==m];bytime={float(t):g for t,g in local.groupby('decision_ms')}
        X,_=I.market_inputs(W,fair[m],depth);prob=V.P.forecasts(W,X)
        controllers={case:B.Controller(W,paths,bank,prob,X,case) for case in B.CASES}
        def observe(k,t,ST,kpos):
            if t not in bytime:return
            state=V.P.snapshot(W,k,t,ST,kpos)
            true_pair=None
            if day==F.TEST:
                j=int(W[0]['jk'][k]);bid=float(W[0]['P5'][j,0]);ask=float(W[1]['P5'][j,0]);classes=[]
                for side,price in ((0,bid),(1,ask)):
                    q=A.DATA.public_target(W[side],t,price,(bid+ask)/2.)
                    classes.append(int(C.classes(q['flow_quantity'],q['move_cents'])[2]))
                true_pair=15*classes[0]+classes[1]
            for row in bytime[t].to_dict('records'):
                indices=[i for i,x in enumerate(state['info']) if x['kind']==row['kind'] and x['side']==row['side'] and x['rung']==row['rung'] and (row['kind']=='post' or abs(x['activation']-(t-row['x_age_ms']))<.001)]
                C.require(len(indices)==1,'response action identity')
                C.require(np.isclose(kpos,row['known_inventory']),'response client position identity')
                for case,pol in controllers.items():
                    ctx,status=pol.context(k,t);est=dict(evaluation_status=status)
                    if ctx is not None:est.update(A.prediction(state,indices[0],row['kind'],ctx))
                    if case=='SOFT_JOINT':
                        C.require(status==row['evaluation_status'],'baseline status changed')
                        if ctx is not None:
                            for field in ('predicted','predicted_wealth','predicted_risk','predicted_quantity'):
                                C.require(np.isclose(est[field],row[field],rtol=1e-10,atol=1e-8),'baseline value changed '+field)
                    keep=('day','m','side','rung','generation','decision_ms','x_age_ms','kind','phase','hash','status','uid','known_inventory','actual','actual_wealth','actual_risk','actual_quantity','label_origin')
                    result={name:row[name] for name in keep if name in row};result.update(est,case=case);results.append(result)
                    if true_pair is not None:
                        cx,st=pol.context(k,t,known_pair=true_pair);e=dict(evaluation_status=st)
                        if cx is not None:e.update(A.prediction(state,indices[0],row['kind'],cx))
                        if case=='SOFT_JOINT':
                            previous=old_oracle[row['uid']]
                            C.require((cx is not None)==(previous['evaluation_status']=='eligible'),'oracle support changed')
                            if cx is not None:
                                for field in ('predicted','predicted_wealth','predicted_risk','predicted_quantity'):
                                    C.require(np.isclose(e[field],previous[field],rtol=1e-10,atol=1e-8),'oracle baseline differs '+field)
                        z={name:row[name] for name in keep if name in row};z.update(e,case=case,true_pair=true_pair);oracle.append(z)
        replay=RUN.run_lane(W,cap=F.CAP,observer=observe)
        with A.CLOCK.clocks(58.,37.):reference=A.AC.replay_neutral(W,size=30.,shift_ms=50.,queue='M',nr=6,cap=F.CAP,mode='hard')
        for field in ('fl','stat','orders','cancels'):np.testing.assert_equal(replay[field],reference[field])
        controls[str(m)]=dict(neutral_prefix_reconciled=True,baseline_predictions_reproduced=True)
        del controllers,prob;gc.collect();C.guard('response actions')
        print(json.dumps(dict(stage='actions',day=day,m=m,seconds=round(time.time()-start))),flush=True)
    df=pd.DataFrame(results);C.require(not df.duplicated(['uid','case']).any(),'duplicate response action')
    for case,g in df.groupby('case'):C.require(set(g.uid)==set(targets.uid),'missing response action')
    df.to_parquet(out/'values.parquet',index=False)
    if oracle:pd.DataFrame(oracle).to_parquet(out/'oracle.parquet',index=False)
    C.dump(out/'receipt.json',dict(asof=C.stamp(),source_manifest=C.sha(B.FREEZE),day=day,selected_markets=selected,
        bank_receipt_sha256=C.sha(B.OUT/'bank/receipt.json'),old_receipt_sha256=C.sha(src/'receipt.json'),controls=controls,
        outputs={p.name:C.sha(p) for p in out.glob('*.parquet')},seconds=time.time()-start))
    action_report(day)
    if oracle:oracle_report()


def action_report(day):
    out=B.OUT/('actions_'+day);r=json.loads((out/'receipt.json').read_text())
    C.require(C.sha(out/'values.parquet')==r['outputs']['values.parquet'],'action artifact changed')
    g=pd.read_parquet(out/'values.parquet');ok=g[g.evaluation_status=='eligible'];counts=ok.groupby('uid').case.nunique();ids=set(counts[counts==len(B.CASES)].index)
    common=ok[ok.uid.isin(ids)];metrics={};by_side={}
    for (case,kind),a in common.groupby(['case','kind']):metrics[case+':'+kind]=R.action_metrics(a)
    for (case,kind,side),a in common.groupby(['case','kind','side']):by_side[f'{case}:{kind}:{side}']=R.action_metrics(a)
    src,old=archived('actions_'+day);cand=pd.read_parquet(src/'candidates.parquet')
    result=dict(asof=C.stamp(),day=day,selected_actions=g.uid.nunique(),common_support_actions=len(ids),
        generated_actions=len(cand),candidate_statuses=cand.groupby(['kind','status']).size().to_dict(),
        evaluation_statuses=g.groupby(['case','evaluation_status']).size().to_dict(),metrics=metrics,by_side=by_side,
        receipt_sha256=C.sha(out/'receipt.json'),limits=B.verify()['limits'])
    C.dump(out/'report.json',result);print(json.dumps(C.clean(dict(stage='action_report',day=day,metrics=metrics))),flush=True)


def pilot():
    B.verify();C.require(gate()['run_pilot'],'replay gate failed');start=time.time();out=B.OUT/'pilot';C.require(not out.exists(),'pilot exists');out.mkdir(parents=True)
    selected,data,fair,depth,raw=A.load(F.TEST);old,oldreceipt=archived('pilot')
    C.require(selected==oldreceipt['selected_markets'],'pilot markets changed')
    paths=B.PATH.load();bank=BANK.load();markets=[];fills=[];inventory=[];traces=[]
    for W in I.pairs(data):
        m=int(W[0]['m']);X,_=I.market_inputs(W,fair[m],depth);prob=V.P.forecasts(W,X)
        pol=B.Controller(W,paths,bank,prob,X);r=RUN.run_lane(W,pol)
        f=MET.engine_fills(W,r,data,F.TEST);f['case']='STATE_CONDITIONAL';f=MET.value_fills(f,W);AUDIT.raw_audit(f,W);AUDIT.audit_controls(f,W)
        inv=MET.minute_inventory(W,r);summary=MET.summarize_fills(f)
        markets.append(dict(m=m,case='STATE_CONDITIONAL',**summary,orders=sum(len(z) for z in r['order_states'].values()),pulls=sum(r['stat']['pulls']),
            end_position=r['endpos'],max_position=r['stat']['maxpos'],both_quote_ms=sum(x['quote_both_ms'] for x in inv),any_quote_ms=sum(x['quote_any_ms'] for x in inv),
            seconds=300.,policy_stats=json.dumps(dict(pol.stats),sort_keys=True)))
        fills.append(f);inventory.extend(dict(m=m,case='STATE_CONDITIONAL',**x) for x in inv);traces.extend(pol.trace)
        print(json.dumps(dict(stage='pilot',m=m,seconds=round(time.time()-start),rss_mib=round(I.K.rss_mib()))),flush=True)
        del pol,r;gc.collect();C.guard('response pilot')
    C.require(set(x['m'] for x in markets)==set(selected),'missing pilot market')
    for name,frame in [('markets',pd.DataFrame(markets)),('fills',pd.concat(fills,ignore_index=True)),('inventory',pd.DataFrame(inventory)),('decisions',pd.DataFrame(traces))]:
        previous=pd.read_parquet(old/(name+'.parquet'));previous=previous[previous.case=='SOFT_JOINT'].copy()
        pd.concat([previous,frame],ignore_index=True).to_parquet(out/(name+'.parquet'),index=False)
    C.dump(out/'receipt.json',dict(asof=C.stamp(),source_manifest=C.sha(B.FREEZE),selected_markets=selected,
        old_receipt_sha256=C.sha(old/'receipt.json'),bank_receipt_sha256=C.sha(B.OUT/'bank/receipt.json'),
        outputs={p.name:C.sha(p) for p in out.glob('*.parquet')},seconds=time.time()-start))
    pilot_report()


def pilot_report():
    out=B.OUT/'pilot';r=json.loads((out/'receipt.json').read_text())
    for name,h in r['outputs'].items():C.require(C.sha(out/name)==h,'pilot artifact changed')
    m=pd.read_parquet(out/'markets.parquet');d=pd.read_parquet(out/'decisions.parquet');rows=[]
    for case,g in m.groupby('case'):
        total=g.select_dtypes('number').sum();q=float(total.shares);c=d[(d.case==case)&(d.kind=='cancel')];age=c.time-c.activation;stats=Counter()
        for s in g.policy_stats:stats.update(json.loads(s))
        rows.append(dict(case=case,markets=len(g),shares=q,orders=int(total.orders),pulls=int(total.pulls),
            terminal_pnl_usd=float(total.terminal_pnl_usd),settlement_pnl_usd=float(total.settlement_pnl_usd),
            spread_cents_per_share=float(total.spread_capture_usd*100/q),
            markout_1s_cents_per_share=float(total.markout_1000_usd*100/total.markout_1000_ok_shares),
            markout_5s_cents_per_share=float(total.markout_5000_usd*100/total.markout_5000_ok_shares),
            markout_1s_covered_shares=float(total.markout_1000_ok_shares),markout_5s_covered_shares=float(total.markout_5000_ok_shares),
            both_quote_percent=float(total.both_quote_ms/(total.seconds*1000)*100),any_quote_percent=float(total.any_quote_ms/(total.seconds*1000)*100),
            mean_abs_end_inventory=float(g.end_position.abs().mean()),max_inventory=float(g.max_position.max()),
            model_cancels=len(c),first_review_cancels=int((age<=50.+1e-7).sum()),median_cancel_age_ms=float(age.median()),
            unsupported_mass_mean=stats['unsupported_mass_sum']/stats['eligible'],marginal_l1_mean=stats['marginal_l1_sum']/stats['eligible'],
            effective_paths_mean=stats['effective_paths_sum']/stats['eligible'],policy_stats=dict(stats)))
    a=m[m.case=='STATE_CONDITIONAL'].set_index('m');b=m[m.case=='SOFT_JOINT'].set_index('m')
    paired={k:dict(delta=float((a[k]-b[k]).sum()),positive_markets=int((a[k]>b[k]).sum()),n=len(a)) for k in ('terminal_pnl_usd','settlement_pnl_usd','pulls','both_quote_ms','shares')}
    C.dump(out/'report.json',dict(asof=C.stamp(),day=F.TEST,rows=rows,paired=paired,limits=B.verify()['limits'],receipt_sha256=C.sha(out/'receipt.json')))


def oracle_report():
    src=B.OUT/('actions_'+F.TEST);r=json.loads((src/'receipt.json').read_text())
    C.require(C.sha(src/'oracle.parquet')==r['outputs']['oracle.parquet'],'oracle values changed')
    g=pd.read_parquet(src/'oracle.parquet');ok=g[g.evaluation_status=='eligible'];n=ok.groupby('uid').case.nunique();ids=set(n[n==len(B.CASES)].index)
    common=ok[ok.uid.isin(ids)];metrics={}
    for (case,kind),z in common.groupby(['case','kind']):metrics[case+':'+kind]=R.action_metrics(z)
    C.dump(src/'oracle_report.json',dict(asof=C.stamp(),n=g.uid.nunique(),common_support_actions=len(ids),
        statuses=g.groupby(['case','evaluation_status']).size().to_dict(),metrics=metrics,
        max_anchor_pair_l1=float(ok.anchor_pair_l1.max()),receipt_sha256=C.sha(src/'receipt.json'),
        role='Known future coarse pair diagnostic ONLY; current client states and raw conditional responses. Not achievable forecasting or validation.'))


def gate():
    B.verify();src=B.OUT/('actions_'+F.TEST);r=json.loads((src/'oracle_report.json').read_text());m=r['metrics'];comparisons={}
    for kind in ('cancel','post'):
        a=m['STATE_CONDITIONAL:'+kind];b=m['SOFT_JOINT:'+kind]
        C.require(a['n']==b['n'] and a['n']>0,'gate population')
        comparisons[kind]=dict(n=a['n'],baseline_rmse=b['rmse_cents'],candidate_rmse=a['rmse_cents'],
            no_worse=a['rmse_cents']<=b['rmse_cents'],strictly_better=a['rmse_cents']<b['rmse_cents'])
    run=all(v['no_worse'] for v in comparisons.values()) and any(v['strictly_better'] for v in comparisons.values())
    return dict(asof=C.stamp(),run_pilot=run,comparisons=comparisons,source_manifest=C.sha(B.FREEZE),
        oracle_report_sha256=C.sha(src/'oracle_report.json'),role='Frozen development compute screen, not validation or promotion')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','selftest','bank','actions','gated_pilot','audit']);p.add_argument('day',nargs='?');args=p.parse_args()
    if args.mode=='freeze':B.freeze()
    elif args.mode=='bank':BANK.build()
    elif args.mode=='actions':actions(args.day)
    elif args.mode=='gated_pilot':
        result=gate();C.require(not (B.OUT/'gate.json').exists(),'gate output exists');C.dump(B.OUT/'gate.json',result)
        if result['run_pilot']:pilot()
        else:print(json.dumps(dict(pilot='skipped_by_frozen_gate',**result)),flush=True)
    else:
        import pf_conditional_map_checks as CHECK
        if args.mode=='selftest':print(json.dumps(C.clean(CHECK.selftest())),flush=True)
        else:CHECK.audit()


