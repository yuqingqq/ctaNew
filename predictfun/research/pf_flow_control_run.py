"""Build, validate and replay the frozen joint-forecast research controller."""
import argparse
from collections import Counter
import copy
import gc
import json
from pathlib import Path
import time
import pf_flow_control_common as F
import pf_flow_control_paths as PATH
import pf_flow_control_policy as POLICY
import pf_flow_control_kernel as EXEC
import pf_flow_control_checks as CHECK
import pf_flow_control_lane as LANE
import pf_stop_queue_activation_lane as ORIGINAL
import pf_stop_ev_minutes_metrics as MET
import pf_stop_ev_minutes as AUDIT
import pf_flow_price_model as FM
import pf_ev_lag_gap as CLOCK
C,np,pd,I=F.C,F.np,F.pd,F.I
CASES=('KEEP_ALL_CAP','EMPIRICAL','JOINT','JOINT_H2')


def run_lane(W,pol=None,cap=F.CAP,observer=None):
    with CLOCK.clocks(58.,37.):
        n=W[0]['n'];aux={s:[False]*n for s in (0,1)}
        result=LANE.sim_joint_g(W,'M',50.,cap,aux,size=30.,nr=6,mode='hard' if cap is not None else 'none',cooldown=0.,
            decide={(s,d):pol.decide(s,d) for s in (0,1) for d in range(6)} if pol else None,
            gate={(s,d):pol.gate(s,d) for s in (0,1) for d in range(6)} if pol else None,
            on_step=pol.on_step if pol else observer)
    C.require(sum(result['stat']['viol'].values())==0,'continuous lane violations')
    return result


def branch_truth(W,row,reference):
    """Two canonical replays with the same neutral prefix and no later placement.

    This deliberately uses actual engine state only for the realized labels;
    the predictor receives the separate client-known snapshot.
    """
    t=float(row.decision_ms);endpoint=t+1000.;side=int(row.side);rung=int(row.rung)
    activation=t-float(row.x_age_ms);out=[];prefixes=[]
    for cancel in (False,True):
        z=I.truncate(W,endpoint-W[0]['ws'])
        for s,w in z.items():
            mask=np.asarray(w['anc'])-50.<=endpoint
            w['anc']=np.asarray(w['anc'])[mask];w['ptl']=np.asarray(w['ptl'])[mask];w['shl']=np.asarray(w['shl'])[mask]
        n=z[0]['n'];aux={s:[False]*n for s in (0,1)}
        gate={(s,d):(lambda k:float(z[0]['tg'][k])>=t) for s in (0,1) for d in range(6)}
        dec={(side,rung):(lambda k,now,o:bool(cancel and now>=t and abs(o['ta']-activation)<.001))}
        with CLOCK.clocks(58.,37.):
            result=ORIGINAL.sim_joint_g(z,'M',50.,None,aux,decide=dec,gate=gate,size=30.,nr=6,mode='none',cooldown=0.)
        C.require(sum(result['stat']['viol'].values())==0,'branch truth lane violations')
        prefix=Counter(x for x in result['fl'] if x[2]<=t)
        C.require(prefix==Counter(x for x in reference['fl'] if x[2]<=t),'counterfactual changed neutral prefix')
        prefixes.append(prefix)
        mt,mm=C.AC.valid_marks(W[0],0);j=np.searchsorted(mt,endpoint,side='right')-1
        C.require(j>=0 and endpoint-mt[j]<=1000.,'branch endpoint coverage')
        pos=sum((1-2*s)*q for s,d,ft,p,q in result['fl']);cash=-sum((1-2*s)*q*p for s,d,ft,p,q in result['fl'])
        out.append(cash+pos*mm[j]-F.RISK*pos*pos)
    C.require(prefixes[0]==prefixes[1],'common prefix differs')
    return float(out[1]-out[0])


def neutral_probes(W,paths,prob,actions):
    # Two generation actions per side/phase, selected before looking at values.
    a=actions[actions.m==W[0]['m']].copy()
    a['hash']=[PATH.keyhash(F.TEST,int(r.m),float(r.decision_ms))+f':{r.side}:{r.rung}' for r in a.itertuples()]
    a=a.sort_values('hash').groupby(['side','phase'],sort=False).head(2)
    bytime={float(t):g for t,g in a.groupby('decision_ms')};saved=[];statuses=Counter()
    ctx=POLICY.Controller(W,paths,prob,'JOINT')
    def observe(k,t,ST,kpos):
        if t not in bytime:return
        context,status=ctx.context(k,t);statuses[status]+=len(bytime[t])
        if context is None:return
        packed,weight,mid,diag=context;state=POLICY.snapshot(W,k,t,ST,kpos)
        for row in bytime[t].itertuples():
            matches=[i for i,x in enumerate(state['info']) if x['kind']=='cancel' and x['side']==row.side and x['rung']==row.rung and abs(x['activation']-(t-row.x_age_ms))<.001]
            C.require(len(matches)==1,'neutral probe generation mismatch')
            i=matches[0];action=state['baseline'].copy();action[i]=2.
            out,qty=EXEC.evaluate(packed,state['orders'],np.array([state['baseline'],action]),state['position'],mid,1000.)
            values=weight@POLICY.utility(out,state['position'])
            empirical,_=PATH.weights(ctx.cache[(min(5,int((t-W[0]['ws'])//60000)+1),float(W[0]['P5'][W[0]['jk'][k],0]),float(W[1]['P5'][W[0]['jk'][k],0]-W[0]['P5'][W[0]['jk'][k],0]),1000.)][0])
            evalue=empirical@POLICY.utility(out,state['position'])
            saved.append((row,dict(m=int(row.m),side=int(row.side),rung=int(row.rung),generation=int(row.generation),
                phase=int(row.phase),decision_ms=t,predicted_advantage_cents=float(values[1]-values[0]),
                empirical_advantage_cents=float(evalue[1]-evalue[0]),**diag)))
    reference=run_lane(W,cap=None,observer=observe)
    # Same complete neutral path must be unaffected by the observer's prediction work.
    with CLOCK.clocks(58.,37.):
        control=ORIGINAL.sim_joint_g(W,'M',50.,None,{s:[False]*W[s]['n'] for s in (0,1)},size=30.,nr=6,mode='none',cooldown=0.)
    for key in ('fl','stat','orders','cancels'):np.testing.assert_equal(reference[key],control[key])
    result=[]
    for row,record in saved:
        record['realized_advantage_cents']=branch_truth(W,row,reference);result.append(record)
    return result,dict(statuses),len(a)


def run(smoke=False):
    F.verify();start=time.time();checks=CHECK.selftest()
    out=F.OUT/('smoke' if smoke else 'pilot');C.require(not out.exists(),'run exists');out.mkdir(parents=True)
    paths=PATH.load();r=json.loads((C.OUT/F.TEST/'receipt.json').read_text());inputs={}
    for p,h in r['input_hashes'].items():C.require(C.sha(p)==h,'evaluation raw source changed '+p);inputs[p]=h
    reg=pd.read_parquet(Path(I.K.IO)/'step1'/F.TEST/'registry.parquet')
    reg=reg[reg.mid.isin(r['selected_ids'])].sort_values('ws').copy()
    day0=I._ms('2026-10-27');reg['block']=((reg.ws*1000-day0)//7200000).astype(int)
    selected=reg.groupby('block',sort=True).head(1).mid.astype(int).tolist()
    C.require(len(selected)==F.PILOT_MARKETS,'pilot selection count')
    if smoke:selected=selected[:1]
    windows={int(x.mid):float(x.ws)*1000 for x in reg[reg.mid.isin(selected)].itertuples()}
    fair,_=I.fair_fix(F.TEST,windows);hours=I.X5.depth_hours(F.TEST)
    clock=json.loads((Path(I.K.IO)/'step1'/F.TEST/'clock_rl_d2.json').read_text())
    depth,_=I.X5.depth_arrays(hours,float(clock['bn_in'])+float(clock['d_dec']))
    actions=FM.read(F.TEST);data=I.load_day(F.TEST,set(selected),r['imputation_offset_ms'])
    market_rows=[];fill_rows=[];inventory=[];trace=[];probes=[];probe_status=Counter();selected_probes=0
    for W in I.pairs(data):
        m=int(W[0]['m']);X,coverage=I.market_inputs(W,fair[m],depth);prob=POLICY.forecasts(W,X)
        for case in CASES:
            pol=POLICY.Controller(W,paths,prob,case);result=run_lane(W,pol)
            f=MET.engine_fills(W,result,data,F.TEST);f['case']=case;f=MET.value_fills(f,W)
            AUDIT.raw_audit(f,W);AUDIT.audit_controls(f,W)
            inv=MET.minute_inventory(W,result);summary=MET.summarize_fills(f)
            C.require(abs(result['endpos'])<=F.CAP+1e-7,'terminal inventory cap')
            market_rows.append(dict(m=m,case=case,**summary,orders=sum(len(z) for z in result['order_states'].values()),
                pulls=sum(result['stat']['pulls']),end_position=result['endpos'],max_position=result['stat']['maxpos'],
                both_quote_ms=sum(x['quote_both_ms'] for x in inv),any_quote_ms=sum(x['quote_any_ms'] for x in inv),
                seconds=300.,policy_stats=json.dumps(dict(pol.stats),sort_keys=True)))
            fill_rows.append(f);inventory.extend(dict(m=m,case=case,**x) for x in inv);trace.extend(pol.trace)
            print(json.dumps(dict(stage='replay',m=m,case=case,seconds=round(time.time()-start),rss_mib=round(I.K.rss_mib()))),flush=True)
            del pol,result;gc.collect();C.guard('controller case')
        p,st,n=neutral_probes(W,paths,prob,actions);probes.extend(p);probe_status.update(st);selected_probes+=n
        print(json.dumps(dict(stage='probes',m=m,n=len(p),seconds=round(time.time()-start))),flush=True)
    seen={x['m'] for x in market_rows};missing=sorted(set(selected)-seen)
    pd.DataFrame(market_rows).to_parquet(out/'markets.parquet',index=False)
    pd.concat(fill_rows,ignore_index=True).to_parquet(out/'fills.parquet',index=False)
    pd.DataFrame(inventory).to_parquet(out/'inventory.parquet',index=False)
    pd.DataFrame(trace).to_parquet(out/'decisions.parquet',index=False)
    pd.DataFrame(probes).to_parquet(out/'probes.parquet',index=False)
    outputs={p.name:C.sha(p) for p in out.glob('*.parquet')}
    C.dump(out/'receipt.json',dict(asof=C.stamp(),source_manifest=C.sha(F.FREEZE),
        test_day=F.TEST,smoke=smoke,selected_markets=selected,missing_markets=missing,completed_markets=len(seen),
        selected_neutral_probes=selected_probes,probe_statuses=dict(probe_status),eligible_neutral_probes=len(probes),
        inputs=inputs,bank_receipt_sha256=C.sha(F.OUT/'bank/receipt.json'),
        forecast_receipt_sha256=C.sha(C.OUT/('fit_'+F.TEST)/'receipt.json'),outputs=outputs,checks=checks,seconds=time.time()-start))
    C.dump(out/'audit.json',CHECK.audit(out));report(out)


def matched_null(p):
    if len(p)==0:return dict(n=0,draws=200)
    rng=np.random.default_rng(20261116);a=p.predicted_advantage_cents.to_numpy()>0;y=p.realized_advantage_cents.to_numpy()
    groups=list(p.groupby(['m','side'],sort=False).indices.values());null=[]
    for _ in range(200):
        shuffled=a.copy()
        for ix in groups:shuffled[ix]=rng.permutation(a[ix])
        C.require(all(shuffled[ix].sum()==a[ix].sum() for ix in groups),'unmatched null action counts')
        null.append(float(y[shuffled].sum()))
    return dict(n=len(p),selected=int(a.sum()),draws=200,observed_increment_cents=float(y[a].sum()),
        null_mean_cents=float(np.mean(null)),null_min_cents=float(min(null)),null_max_cents=float(max(null)),
        interpretation='Consumed neutral-generation action diagnostic, not portfolio P&L or untouched validation')


def report(out):
    out=Path(out);r=json.loads((out/'receipt.json').read_text());m=pd.read_parquet(out/'markets.parquet')
    p=pd.read_parquet(out/'probes.parquet');rows=[]
    for case,g in m.groupby('case',sort=False):
        total=g.select_dtypes('number').sum();shares=float(total.shares)
        row=dict(case=case,markets=len(g),shares=shares,orders=int(total.orders),pulls=int(total.pulls),
            terminal_pnl_usd=float(total.terminal_pnl_usd),settlement_pnl_usd=float(total.settlement_pnl_usd),
            spread_cents_per_share=float(total.spread_capture_usd*100/shares) if shares else None,
            markout_1s_cents_per_share=float(total.markout_1000_usd*100/total.markout_1000_ok_shares) if total.markout_1000_ok_shares else None,
            markout_5s_cents_per_share=float(total.markout_5000_usd*100/total.markout_5000_ok_shares) if total.markout_5000_ok_shares else None,
            markout_1s_covered_shares=float(total.markout_1000_ok_shares),markout_5s_covered_shares=float(total.markout_5000_ok_shares),
            under_100ms_shares=float(total.young_shares),cancel_pending_shares=float(total.pending_shares),
            both_quote_percent=float(total.both_quote_ms/(total.seconds*1000)*100),
            any_quote_percent=float(total.any_quote_ms/(total.seconds*1000)*100),
            mean_abs_ending_inventory=float(g.end_position.abs().mean()),max_abs_inventory=float(g.max_position.max()))
        stats=Counter()
        for text in g.policy_stats:stats.update(json.loads(text))
        row['policy_stats']=dict(stats)
        for key in ('unsupported_mass','max_unsupported_mass','marginal_l1','effective_paths'):
            row[key+'_mean']=stats[key+'_sum']/stats['eligible'] if stats['eligible'] else None
        rows.append(row)
    result=dict(asof=C.stamp(),test_day=F.TEST,smoke=r['smoke'],selected_markets=r['selected_markets'],
        missing_markets=r['missing_markets'],rows=rows,neutral_probe_statuses=r['probe_statuses'],matched_null=matched_null(p),
        limits=F.verify()['limits'],receipt_sha256=C.sha(out/'receipt.json'),audit_sha256=C.sha(out/'audit.json'))
    C.dump(out/'report.json',result)
    print(json.dumps(C.clean(result)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','selftest','bank','smoke','pilot','report','audit']);args=parser.parse_args()
    if args.command=='freeze':F.freeze()
    elif args.command=='selftest':print(json.dumps(CHECK.selftest()))
    elif args.command=='bank':PATH.build()
    elif args.command in ('smoke','pilot'):run(args.command=='smoke')
    elif args.command=='report':report(F.OUT/'pilot')
    elif args.command=='audit':print(json.dumps(CHECK.audit(F.OUT/'pilot')))
