"""Fixed mapping comparison: action calibration and continuous economic replay."""
import argparse
from collections import Counter
import gc
import json
import time
import pf_value_bridge as V
import pf_value_bridge_actions as A
import pf_flow_control_run as RUN
import pf_stop_ev_minutes_metrics as MET
import pf_stop_ev_minutes as AUDIT
F,C,np,pd,I=V.F,V.C,V.np,V.pd,V.I


def matched_null(g):
    rng=np.random.default_rng(20261116);selected=g.predicted.to_numpy()>0;actual=g.actual.to_numpy();groups=list(g.groupby(['m','side','kind'],sort=False).indices.values());null=[]
    for _ in range(200):
        a=selected.copy()
        for ix in groups:a[ix]=rng.permutation(selected[ix])
        C.require(all(a[ix].sum()==selected[ix].sum() for ix in groups),'null action counts differ');null.append(float(actual[a].sum()))
    return dict(n=len(g),selected=int(selected.sum()),draws=200,observed_cents=float(actual[selected].sum()),mean_cents=float(np.mean(null)),min_cents=float(min(null)),max_cents=float(max(null)))


def action_metrics(g):
    pred=g.predicted;actual=g.actual;positive=pred>0;error=pred-actual
    return dict(n=len(g),mean_predicted_cents=float(pred.mean()),mean_actual_cents=float(actual.mean()),
        rmse_cents=float(np.sqrt(np.mean(error**2))),zero_prediction_rmse_cents=float(np.sqrt(np.mean(actual**2))),
        mae_cents=float(error.abs().mean()),spearman=float(pred.corr(actual,method='spearman')),
        positive_predictions=int(positive.sum()),positive_predicted_mean_cents=float(pred[positive].mean()),positive_actual_mean_cents=float(actual[positive].mean()),
        selected_actual_sum_cents=float(actual[positive].sum()),selected_severe_negative_count=int((actual[positive]<-30.).sum()),
        predicted_quantity_mean=float(g.predicted_quantity.mean()),actual_quantity_mean=float(g.actual_quantity.mean()),quantity_mae=float((g.predicted_quantity-g.actual_quantity).abs().mean()),
        scenario_advantage_sd_mean=float(g.scenario_advantage_sd.mean()),effective_paths_mean=float(g.effective_paths.mean()),
        unsupported_mass_mean=float(g.unsupported_mass.mean()),marginal_l1_mean=float(g.marginal_l1.mean()),
        matched_null=matched_null(g))


def action_report(day):
    out=V.OUT/('actions_'+day);r=json.loads((out/'receipt.json').read_text())
    for name,h in r['outputs'].items():C.require(C.sha(out/name)==h,'action report input changed')
    df=pd.read_parquet(out/'values.parquet');c=pd.read_parquet(out/'candidates.parquet')
    ok=df[df.evaluation_status=='eligible'];n=ok.groupby('uid').case.nunique();common=set(n[n==len(V.CASES)].index);matched=ok[ok.uid.isin(common)]
    metrics={};by_side={};by_phase={}
    for (case,kind),g in matched.groupby(['case','kind']):metrics[case+':'+kind]=action_metrics(g)
    for (case,kind,side),g in matched.groupby(['case','kind','side']):by_side[f'{case}:{kind}:{side}']=action_metrics(g)
    for (case,kind,phase),g in matched.groupby(['case','kind','phase']):by_phase[f'{case}:{kind}:{phase}']=action_metrics(g)
    result=dict(asof=C.stamp(),day=day,markets=len(r['selected_markets']),generated_actions=len(c),candidate_statuses=c.groupby(['kind','status']).size().to_dict(),
        selected_actions=int(c.selected.sum()),evaluated_actions=df.uid.nunique(),evaluation_statuses=df.groupby(['case','evaluation_status']).size().to_dict(),
        common_supported_actions=len(common),metrics=metrics,by_side=by_side,by_phase=by_phase,
        label_origins=df.drop_duplicates('uid').label_origin.value_counts().to_dict(),
        limits=V.verify()['limits'],receipt_sha256=C.sha(out/'receipt.json'))
    C.dump(out/'report.json',result);print(json.dumps(C.clean(dict(day=day,n=len(common),metrics=metrics))),flush=True)
    return result


def pilot():
    V.verify();start=time.time();out=V.OUT/'pilot';C.require(not out.exists(),'pilot output exists');out.mkdir(parents=True)
    selected,data,fair,depth,raw=A.load(F.TEST);old=F.OUT/'pilot';oldreceipt=json.loads((old/'receipt.json').read_text())
    C.require(selected==oldreceipt['selected_markets'],'pilot population differs')
    for name,h in oldreceipt['outputs'].items():C.require(C.sha(old/name)==h,'old replay changed')
    paths=V.PATH.load();markets=[];fills=[];inventory=[];traces=[]
    for W in I.pairs(data):
        m=int(W[0]['m']);X,_=I.market_inputs(W,fair[m],depth);prob=V.P.forecasts(W,X)
        pol=V.Controller(W,paths,prob,'SOFT_JOINT');r=RUN.run_lane(W,pol)
        f=MET.engine_fills(W,r,data,F.TEST);f['case']='SOFT_JOINT';f=MET.value_fills(f,W);AUDIT.raw_audit(f,W);AUDIT.audit_controls(f,W)
        inv=MET.minute_inventory(W,r);summary=MET.summarize_fills(f)
        markets.append(dict(m=m,case='SOFT_JOINT',**summary,orders=sum(len(z) for z in r['order_states'].values()),pulls=sum(r['stat']['pulls']),
            end_position=r['endpos'],max_position=r['stat']['maxpos'],both_quote_ms=sum(x['quote_both_ms'] for x in inv),any_quote_ms=sum(x['quote_any_ms'] for x in inv),
            seconds=300.,policy_stats=json.dumps(dict(pol.stats),sort_keys=True)))
        fills.append(f);inventory.extend(dict(m=m,case='SOFT_JOINT',**x) for x in inv);traces.extend(pol.trace)
        print(json.dumps(dict(stage='pilot',m=m,seconds=round(time.time()-start),rss_mib=round(I.K.rss_mib()))),flush=True)
        del pol,r;gc.collect();C.guard('bridge pilot')
    C.require(set(x['m'] for x in markets)==set(selected),'missing pilot market')
    for name,frame in [('markets',pd.DataFrame(markets)),('fills',pd.concat(fills,ignore_index=True)),('inventory',pd.DataFrame(inventory)),('decisions',pd.DataFrame(traces))]:
        previous=pd.read_parquet(old/(name+'.parquet'));previous=previous[previous.case=='JOINT'].copy();previous['case']='EXACT_JOINT'
        pd.concat([previous,frame],ignore_index=True).to_parquet(out/(name+'.parquet'),index=False)
    binary=F.OUT/'build'/('kernel_'+C.sha(F.HERE/'pf_flow_control_kernel.cpp')[:16]+'.so')
    C.dump(out/'receipt.json',dict(asof=C.stamp(),source_manifest=C.sha(V.FREEZE),selected_markets=selected,old_receipt_sha256=C.sha(old/'receipt.json'),
        compiled_kernel=dict(path=str(binary),sha256=C.sha(binary)),outputs={p.name:C.sha(p) for p in out.glob('*.parquet')},seconds=time.time()-start))
    pilot_report()


def pilot_report():
    out=V.OUT/'pilot';r=json.loads((out/'receipt.json').read_text())
    for name,h in r['outputs'].items():C.require(C.sha(out/name)==h,'pilot report input changed')
    m=pd.read_parquet(out/'markets.parquet');d=pd.read_parquet(out/'decisions.parquet');rows=[]
    for case,g in m.groupby('case'):
        total=g.select_dtypes('number').sum();q=float(total.shares);c=d[(d.case==case)&(d.kind=='cancel')];age=c.time-c.activation
        stats=Counter()
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
    a=m[m.case=='SOFT_JOINT'].set_index('m');b=m[m.case=='EXACT_JOINT'].set_index('m')
    paired={k:dict(delta=float((a[k]-b[k]).sum()),positive_markets=int((a[k]>b[k]).sum()),n=len(a)) for k in ('terminal_pnl_usd','settlement_pnl_usd','pulls','both_quote_ms','shares')}
    result=dict(asof=C.stamp(),day=F.TEST,rows=rows,paired=paired,limits=V.verify()['limits'],receipt_sha256=C.sha(out/'receipt.json'))
    C.dump(out/'report.json',result);print(json.dumps(C.clean(result)),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','selftest','actions','pilot','report','audit']);p.add_argument('day',nargs='?');args=p.parse_args()
    if args.mode=='freeze':V.freeze()
    elif args.mode=='selftest':
        import pf_value_bridge_checks as CHECK
        print(json.dumps(CHECK.selftest()),flush=True)
    elif args.mode=='actions':A.run(args.day);action_report(args.day)
    elif args.mode=='pilot':pilot()
    elif args.mode=='report':
        for day in ('20261026','20261027'):action_report(day)
        pilot_report()
    else:
        import pf_value_bridge_checks as CHECK
        CHECK.audit()
