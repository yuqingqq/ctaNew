"""Matched neutral action calibration and continuous controller comparison."""
import argparse
from collections import Counter
import gc
import json
import time
import pf_response_bridge as B
import pf_response_fit as MODEL
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
    C.require(selected==receipt['selected_markets'],'action markets changed');paths=B.PATH.load();results=[];controls={}
    for W in I.pairs(data):
        m=int(W[0]['m']);local=targets[targets.m==m];bytime={float(t):g for t,g in local.groupby('decision_ms')}
        X,_=I.market_inputs(W,fair[m],depth);prob=V.P.forecasts(W,X);learned,emp=MODEL.forecasts(W,X)
        controllers={case:B.Controller(W,paths,prob,emp if case=='RICH_EMPIRICAL' else learned,case) for case in B.CASES}
        def observe(k,t,ST,kpos):
            if t not in bytime:return
            state=V.P.snapshot(W,k,t,ST,kpos)
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
        replay=RUN.run_lane(W,cap=F.CAP,observer=observe)
        with A.CLOCK.clocks(58.,37.):reference=A.AC.replay_neutral(W,size=30.,shift_ms=50.,queue='M',nr=6,cap=F.CAP,mode='hard')
        for field in ('fl','stat','orders','cancels'):np.testing.assert_equal(replay[field],reference[field])
        controls[str(m)]=dict(neutral_prefix_reconciled=True,baseline_predictions_reproduced=True)
        del controllers,learned,emp,prob;gc.collect();C.guard('response actions')
        print(json.dumps(dict(stage='actions',day=day,m=m,seconds=round(time.time()-start))),flush=True)
    df=pd.DataFrame(results);C.require(not df.duplicated(['uid','case']).any(),'duplicate response action')
    for case,g in df.groupby('case'):C.require(set(g.uid)==set(targets.uid),'missing response action')
    df.to_parquet(out/'values.parquet',index=False)
    C.dump(out/'receipt.json',dict(asof=C.stamp(),source_manifest=C.sha(B.FREEZE),day=day,selected_markets=selected,
        fit_receipt_sha256=C.sha(B.OUT/'fit/receipt.json'),old_receipt_sha256=C.sha(src/'receipt.json'),controls=controls,
        outputs={'values.parquet':C.sha(out/'values.parquet')},seconds=time.time()-start))
    action_report(day)


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
    B.verify();start=time.time();out=B.OUT/'pilot';C.require(not out.exists(),'pilot exists');out.mkdir(parents=True)
    selected,data,fair,depth,raw=A.load(F.TEST);old,oldreceipt=archived('pilot')
    C.require(selected==oldreceipt['selected_markets'],'pilot markets changed')
    paths=B.PATH.load();markets=[];fills=[];inventory=[];traces=[]
    for W in I.pairs(data):
        m=int(W[0]['m']);X,_=I.market_inputs(W,fair[m],depth);prob=V.P.forecasts(W,X);learned,_=MODEL.forecasts(W,X)
        pol=B.Controller(W,paths,prob,learned);r=RUN.run_lane(W,pol)
        f=MET.engine_fills(W,r,data,F.TEST);f['case']='RICH_JOINT';f=MET.value_fills(f,W);AUDIT.raw_audit(f,W);AUDIT.audit_controls(f,W)
        inv=MET.minute_inventory(W,r);summary=MET.summarize_fills(f)
        markets.append(dict(m=m,case='RICH_JOINT',**summary,orders=sum(len(z) for z in r['order_states'].values()),pulls=sum(r['stat']['pulls']),
            end_position=r['endpos'],max_position=r['stat']['maxpos'],both_quote_ms=sum(x['quote_both_ms'] for x in inv),any_quote_ms=sum(x['quote_any_ms'] for x in inv),
            seconds=300.,policy_stats=json.dumps(dict(pol.stats),sort_keys=True)))
        fills.append(f);inventory.extend(dict(m=m,case='RICH_JOINT',**x) for x in inv);traces.extend(pol.trace)
        print(json.dumps(dict(stage='pilot',m=m,seconds=round(time.time()-start),rss_mib=round(I.K.rss_mib()))),flush=True)
        del pol,r,learned;gc.collect();C.guard('response pilot')
    C.require(set(x['m'] for x in markets)==set(selected),'missing pilot market')
    for name,frame in [('markets',pd.DataFrame(markets)),('fills',pd.concat(fills,ignore_index=True)),('inventory',pd.DataFrame(inventory)),('decisions',pd.DataFrame(traces))]:
        previous=pd.read_parquet(old/(name+'.parquet'));previous=previous[previous.case=='SOFT_JOINT'].copy()
        pd.concat([previous,frame],ignore_index=True).to_parquet(out/(name+'.parquet'),index=False)
    C.dump(out/'receipt.json',dict(asof=C.stamp(),source_manifest=C.sha(B.FREEZE),selected_markets=selected,
        old_receipt_sha256=C.sha(old/'receipt.json'),fit_receipt_sha256=C.sha(B.OUT/'fit/receipt.json'),
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
    a=m[m.case=='RICH_JOINT'].set_index('m');b=m[m.case=='SOFT_JOINT'].set_index('m')
    paired={k:dict(delta=float((a[k]-b[k]).sum()),positive_markets=int((a[k]>b[k]).sum()),n=len(a)) for k in ('terminal_pnl_usd','settlement_pnl_usd','pulls','both_quote_ms','shares')}
    C.dump(out/'report.json',dict(asof=C.stamp(),day=F.TEST,rows=rows,paired=paired,limits=B.verify()['limits'],receipt_sha256=C.sha(out/'receipt.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','selftest','fit','predictions','actions','pilot','audit']);p.add_argument('day',nargs='?');args=p.parse_args()
    if args.mode=='freeze':B.freeze()
    elif args.mode=='fit':MODEL.fit()
    elif args.mode=='predictions':MODEL.report()
    elif args.mode=='actions':actions(args.day)
    elif args.mode=='pilot':pilot()
    else:
        import pf_response_checks as CHECK
        if args.mode=='selftest':print(json.dumps(C.clean(CHECK.selftest())),flush=True)
        else:CHECK.audit()
