"""Full-policy comparisons, entry-matched inventory paths and fill attribution."""
import datetime as dt
import json
import subprocess
import numpy as np
import pandas as pd
from .pf_passive_study import DAYS,ARMS,DATA,DEST,HERE,sha,write
from .pf_passive_run import folder,stem,name
from .pf_sizing_run import reconcile
from .pf_settlement_study import paired_inference


def trajectory(f):
    """Economic inventory area; classify every tranche without selecting windows."""
    t=f.t.to_numpy(float);q=f.q.to_numpy(float);sg=np.where(f.side.to_numpy()==0,1.,-1.)
    if not np.isfinite(t).all() or not np.isfinite(q).all() or (q<=0).any():raise ValueError('invalid fill')
    if len(t) and ((np.diff(t)<-1e-9).any() or t.min()<0 or t.max()>300):raise ValueError('invalid event chronology')
    net=float(np.sum((sg*q)[t<60]));known=float(np.sum((sg*q)[t<=59.9]))
    initial=net;area=0.;last=60.;reduce_q=0.;open_q=0.;flips=0;overshoot=0.;reduce_mo=0.;nreduce=0
    for i in np.flatnonzero(t>=60):
        area+=abs(net)*(t[i]-last);last=t[i]
        red=min(abs(net),q[i]) if net*sg[i]<-1e-8 else 0.
        if red>0:
            nreduce+=1
            if q[i]>abs(net)+1e-8:flips+=1;overshoot+=q[i]-abs(net)
        reduce_q+=red;open_q+=q[i]-red
        reduce_mo+=float(f.mo1.iloc[i])*red/q[i]
        net+=sg[i]*q[i]
    area+=abs(net)*(300-last)
    return dict(net60=initial,known_net60=known,net_end=net,area_share_seconds=area,
        reducing_shares=reduce_q,opening_shares_after60=open_q,zero_crossings_after60=flips,
        overshoot_shares=overshoot,reducing_mo1=reduce_mo,reducing_fill_tranches=nreduce)


def generation_counts(f,q,ws,counters):
    """Match actual fill activation identities to sent orders, including partial fills."""
    p=q[q.kind=='place'].copy()
    # Engine epoch-ms doubles have a 1/4096 ms quantum at the studied dates.
    p['activation_tick']=np.rint((p.t_ms+40)*4096).astype(np.int64)
    keys=['mid','lane','activation_tick']
    if p.duplicated(keys).any():raise ValueError('ambiguous create generation')
    z=f.copy();z['activation_tick']=np.rint((z.mid.map(ws)+z.t*1000-z.age_ms)*4096).astype(np.int64)
    gens=z[keys].drop_duplicates()
    if len(gens.merge(p[keys],on=keys,how='inner',validate='one_to_one'))!=len(gens):
        raise ValueError('fill generation does not match a sent order')
    accepted=len(p)-counters.get('postonly_refused',0)-counters.get('postonly_refused_own',0)
    if not 0<=len(gens)<=accepted:raise ValueError('acceptance/fill count mismatch')
    return dict(place_attempts=len(p),accepted_orders=accepted,filled_generations=len(gens),
                accepted_order_fill_fraction=len(gens)/accepted if accepted else None)


def summarize(day,tm,mode,label,spec):
    out=folder(day,tm,mode);checked=json.loads((out/'passive_checks.json').read_text())
    if checked['markets']!=288:raise ValueError('incomplete checker population')
    s=json.loads((out/'summary.json').read_text())
    if label=='baseline' and (out/'baseline_reference.json').exists():
        ref=json.loads((out/'baseline_reference.json').read_text())
        from pathlib import Path
        src=Path(ref['summary_path'])
        if sha(src)!=ref['summary_sha']:raise ValueError('baseline source changed')
        s=json.loads(src.read_text())
    a=s['arms'][name(spec)]
    w=pd.read_parquet(out/'win'/stem(spec));f=pd.read_parquet(out/'fills'/stem(spec));q=pd.read_parquet(out/'requests'/stem(spec))
    if len(w)!=288 or w.mid.duplicated().any():raise ValueError('market population')
    reconcile(w,f);ws=w.set_index('mid').ws;ctr=a['live']['counters'];groups={m:g for m,g in f.groupby('mid',sort=False)}
    tr=[]
    for row in w.itertuples():
        g=groups.get(row.mid,f.iloc[:0]);x=trajectory(g)
        if abs(x['net_end']-row.net_end)>1e-7:raise ValueError('inventory/fill reconciliation')
        tr.append(dict(day=day,mid=row.mid,**x))
    tr=pd.DataFrame(tr);inventory=tr[tr.known_net60.abs()>=1]
    missing=dict(mo1=int(f.mo1.isna().sum()),mark_stop=int(w.mtm_stop.isna().sum()))
    if any(missing.values()):raise ValueError('missing decision metric; report exclusions explicitly before use')
    row=dict(day=day,timing=tm,mode=mode,variant=label,markets=len(w),
        settle=float(w.pnl.sum()),plus1=float(f.mo1.sum()),mark_stop=float(w.mtm_stop.sum()),carry=float(w.carry_stop.sum()),
        shares=float(w.shares.sum()),fills=len(f),requests=q.kind.value_counts().to_dict(),refusals=int(w.refused.sum()),
        abs_net_end_mean=float(w.net_end.abs().mean()),abs_net_end_p95=float(w.net_end.abs().quantile(.95)),
        mean_time_weighted_abs_net_60_300=float(tr.area_share_seconds.mean()/240),
        inventory_population=dict(markets=len(inventory),known_entry_min_abs_shares=1,
            mean_abs_net_end=float(inventory.net_end.abs().mean()) if len(inventory) else None,
            flat_below_one_share_fraction=float((inventory.net_end.abs()<1).mean()) if len(inventory) else None),
        reducing_shares=float(tr.reducing_shares.sum()),opening_shares_after60=float(tr.opening_shares_after60.sum()),
        zero_crossings_after60=int(tr.zero_crossings_after60.sum()),overshoot_shares=float(tr.overshoot_shares.sum()),
        reducing_mo1=float(tr.reducing_mo1.sum()),
        fill_weighted_age_ms=float(np.average(f.age_ms,weights=f.q)) if len(f) else None,
        generations=generation_counts(f,q,ws,ctr),counters=ctr,withheld=a['withheld_decisions'],
        statuses=dict(eligible_markets=len(w),excluded_markets=0,missing_valuations=missing),
        artifact_sha={k:sha(out/k/stem(spec)) for k in ['win','fills','requests']})
    return row,w,tr


def main():
    results=[];daily=[];trace=[];inputs={};checks={}
    for mode in ['main','stress']:
        for tm in ['J','X']:
            for label,spec in ARMS.items():
                ws=[];ds=[];ts=[]
                for day in DAYS if mode=='main' else ['20261103']:
                    d,w,tr=summarize(day,tm,mode,label,spec);daily.append(d);ds.append(d);ws.append(w);ts.append(tr)
                    trace.extend(dict(r,variant=label,timing=tm,mode=mode) for r in tr.to_dict('records'))
                    path=folder(day,tm,mode)
                    checks[f'{mode}/{tm}/{day}']=json.loads((path/'passive_checks.json').read_text())
                    inputs[f'{mode}/{tm}/{day}']=dict(summary_sha=sha(path/'summary.json'),
                        execution=json.loads((path/'study_execution.json').read_text()),
                        baseline_reference=json.loads((path/'baseline_reference.json').read_text()) if (path/'baseline_reference.json').exists() else None,
                        passive_audit_sha=sha(path/'passive_audit.json'))
                w=pd.concat(ws);tr=pd.concat(ts);eq=np.r_[0.,w.pnl.cumsum().to_numpy()];inv=tr[tr.known_net60.abs()>=1]
                add=lambda k:sum(d[k] for d in ds)
                row=dict(mode=mode,timing=tm,variant=label,days=len(ds),markets=len(w),
                    **{k:add(k) for k in ['settle','plus1','mark_stop','carry','shares','fills','refusals',
                        'reducing_shares','opening_shares_after60','zero_crossings_after60','overshoot_shares','reducing_mo1']},
                    max_drawdown=float((eq-np.maximum.accumulate(eq)).min()),
                    mean_abs_net_end=float(tr.net_end.abs().mean()),p95_abs_net_end=float(tr.net_end.abs().quantile(.95)),
                    mean_time_weighted_abs_net_60_300=float(tr.area_share_seconds.mean()/240),
                    inventory_population=dict(markets=len(inv),mean_abs_net_end=float(inv.net_end.abs().mean()),
                        flat_below_one_share_fraction=float((inv.net_end.abs()<1).mean())),
                    requests={k:sum(d['requests'].get(k,0) for d in ds) for k in ['place','cancel']},
                    accepted_orders=sum(d['generations']['accepted_orders'] for d in ds),
                    filled_generations=sum(d['generations']['filled_generations'] for d in ds),
                    statuses=dict(eligible_markets=len(w),excluded_markets=0,missing_valuations=0),
                    per_day=ds)
                row['accepted_order_fill_fraction']=row['filled_generations']/row['accepted_orders'] if row['accepted_orders'] else None
                results.append(row)
    mainrows={(r['timing'],r['variant']):r for r in results if r['mode']=='main'}
    family=[(tm,v,m) for tm in ['J','X'] for v in ARMS if v!='baseline' for m in ['settle','plus1']]
    diffs=np.array([[mainrows[tm,v]['per_day'][i][m]-mainrows[tm,'baseline']['per_day'][i][m] for tm,v,m in family] for i in range(len(DAYS))])
    inference={f'{tm}/{v}/{m}':x for (tm,v,m),x in zip(family,paired_inference(diffs))}
    comparisons=[]
    for mode in ['main','stress']:
        idx={(r['timing'],r['variant']):r for r in results if r['mode']==mode}
        for tm in ['J','X']:
            b=idx[tm,'baseline']
            for v in ARMS:
                if v=='baseline':continue
                a=idx[tm,v];dd=np.array([x['settle']-y['settle'] for x,y in zip(a['per_day'],b['per_day'])])
                comparisons.append(dict(mode=mode,timing=tm,variant=v,
                    deltas={k:a[k]-b[k] for k in ['settle','plus1','mark_stop','carry','reducing_mo1','mean_abs_net_end',
                        'mean_time_weighted_abs_net_60_300','shares','refusals','accepted_order_fill_fraction']},
                    place_change_pct=100*(a['requests']['place']/b['requests']['place']-1),
                    days_settle_better=int((dd>1e-8).sum()),days_settle_worse=int((dd< -1e-8).sum()),
                    leave_one_out_settle_delta=[float(dd.sum()-dd.max()),float(dd.sum()-dd.min())] if mode=='main' else None))
    sources=list(HERE.glob('pf_passive*.py'))+[HERE/'run_passive_study.sh',HERE/'tests/test_passive_study.py',
        HERE/'pf_sizing_engine.py',HERE/'pf_step4_ladder_v3.py',HERE/'pf_settlement_study.py',HERE/'pf_clock_rl.py',HERE/'pf_io.py']
    datafiles=[DATA/'step4'/'minute_lines_t60k.json',DATA/'step4'/'inv_r60.json']
    for day in DAYS:
        datafiles += [DATA/'step5'/day/'preds_basis_study.parquet',DATA/'step1'/day/'registry.parquet',
                      DATA/'step1'/day/'clock_rl_d2.json',DATA/'step1'/day/'legs_timed2.parquet',DATA/'step1'/day/'legs_tk.parquet']
        datafiles += sorted((DATA/'step1'/day/'books').glob('*.parquet'))
    tests=(DEST/'logs/passive_final_controls.log').read_text()
    if '\nOK\n' not in tests or 'rc=0 ' not in tests:raise ValueError('source controls missing')
    declaration=DEST/('declaration_v2.json' if (DEST/'declaration_v2.json').exists() else 'declaration.json')
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),declaration=json.loads(declaration.read_text()),
        declaration_sha={p.name:sha(p) for p in sorted(DEST.glob('declaration*.json'))},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        source_sha={str(p.relative_to(HERE.parents[1])):sha(p) for p in sources},
        prepared_input_sha={str(p):sha(p) for p in datafiles},results=results,comparisons=comparisons,
        paired_inference=inference,checks=checks,execution_inputs=inputs,source_controls=tests,
        predicates={v:dict(main_settle_better_both=all(mainrows[t,v]['settle']>mainrows[t,'baseline']['settle'] for t in ['J','X']),
            main_plus1_better_both=all(mainrows[t,v]['plus1']>mainrows[t,'baseline']['plus1'] for t in ['J','X']),
            lower_inventory_both=all(mainrows[t,v]['mean_abs_net_end']<mainrows[t,'baseline']['mean_abs_net_end'] for t in ['J','X']))
            for v in ARMS if v!='baseline'},
        interpretation='consumed-day in-sample policy feasibility; no live-return, untouched-validation or full OMS parity claim',
        forecast_refit=False,forward_data_read=False,live_settings_changed=False,deployment=False,
        inventory_statistics='economic path from each fill; known net at 60s uses fixed 100ms fill knowledge delay; entry is shared before interventions',
        fill_statistics='accepted orders = place attempts minus post-only rejections; filled generations matched by market/lane/activation timestamp; conditional fill ages are descriptive',
        inference_limit='five complete UTC days; wide intervals and coarse sign-flip inference; no intervals for one-day stress')
    pd.DataFrame(trace).to_parquet(DEST/'inventory_paths.parquet',index=False)
    write(DEST/'report.json',out)
    write(HERE.parent/'reviews/passive_inventory_20261109/receipt.json',out)
    for r in results:
        print({k:r[k] for k in ['mode','timing','variant','settle','plus1','mark_stop','carry','mean_abs_net_end','requests','accepted_order_fill_fraction']})


if __name__=='__main__':main()
