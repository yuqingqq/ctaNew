"""All-market risk and FIFO trading attribution; no fill-selected fitting."""
import collections
import datetime as dt
import json
import math
import subprocess
import numpy as np
import pandas as pd
from .pf_entry_study import HERE,DAYS,ARMS,DATA,DEST,sha,write
from .pf_entry_run import folder,stem
from . import pf_passive_report as metrics
from .pf_settlement_study import paired_inference


def full_path(f):
    net=0.;last=0.;area=0.;energy=0.;over=0.;peak=0.;paired=0.;profit=0.;queue=collections.deque()
    if not f.side.isin([0,1]).all():raise ValueError('invalid fill side')
    for r in f.itertuples():
        if not all(math.isfinite(v) for v in [r.t,r.q,r.pt]) or not last<=r.t<=300 or r.q<=0 or not 0<r.pt<100:
            raise ValueError('invalid fill path')
        elapsed=r.t-last;area+=abs(net)*elapsed;energy+=net*net*elapsed;over+=(abs(net)>10+1e-7)*elapsed
        last=r.t;sg=1 if r.side==0 else -1;net+=sg*r.q;peak=max(peak,abs(net));left=float(r.q)
        while left>1e-9 and queue and queue[0][0]!=sg:
            os,oq,op=queue.popleft();q=min(left,oq);paired+=q;profit+=q*os*(r.pt-op)/100
            left-=q;oq-=q
            if oq>1e-9:queue.appendleft((os,oq,op))
        if left>1e-9:queue.append((sg,left,r.pt))
    elapsed=300-last;area+=abs(net)*elapsed;energy+=net*net*elapsed;over+=(abs(net)>10+1e-7)*elapsed
    if abs(float(f.q.sum())-2*paired-abs(net))>1e-6:raise ValueError('matched quantity reconciliation')
    return dict(net_end=net,mean_abs_inventory=area/300,inventory_square=energy/300,
                peak_abs_inventory=peak,seconds_above10=over,paired_shares=paired,paired_pnl=profit)


def summarize(day,tm,label,spec):
    metrics.folder=lambda d,t,mode:folder(d,t)
    d,w,tr=metrics.summarize(day,tm,'main',label,spec)
    # Entry policies change the population with inventory at 60s. Do not call that subgroup matched.
    d.pop('inventory_population');f=pd.read_parquet(folder(day,tm)/'fills'/stem(spec));groups={m:g for m,g in f.groupby('mid',sort=False)}
    paths=[]
    for row in w.itertuples():
        p=full_path(groups.get(row.mid,f.iloc[:0]))
        if abs(p['net_end']-row.net_end)>1e-7:raise ValueError('full-path net reconciliation')
        paths.append(dict(day=day,mid=row.mid,**p,residual_pnl=float(row.pnl-p['paired_pnl'])))
    paths=pd.DataFrame(paths)
    d.update(inventory_square=float(paths.inventory_square.mean()),mean_abs_inventory=float(paths.mean_abs_inventory.mean()),
        peak_abs_inventory=float(paths.peak_abs_inventory.max()),mean_seconds_above10=float(paths.seconds_above10.mean()),
        paired_shares=float(paths.paired_shares.sum()),paired_pnl=float(paths.paired_pnl.sum()),residual_pnl=float(paths.residual_pnl.sum()))
    if not math.isclose(d['settle'],d['paired_pnl']+d['residual_pnl'],abs_tol=1e-7):raise ValueError('PnL attribution')
    return d,w,paths


def aggregate(ds,ws,paths):
    w=pd.concat(ws).sort_values('ws');p=pd.concat(paths)
    if w.mid.duplicated().any() or len(w)!=2592 or len(ds)!=9:raise ValueError('aggregate population')
    pnl=np.array([d['settle'] for d in ds]);eq=np.r_[0.,w.pnl.cumsum().to_numpy()]
    sums=['settle','plus1','mark_stop','carry','shares','fills','refusals','paired_shares','paired_pnl','residual_pnl']
    result=dict(days=9,markets=len(w),**{k:sum(d[k] for d in ds) for k in sums},
        daily_pnl_sd=float(pnl.std(ddof=1)),worst_day=float(pnl.min()),sum_losing_days=float(pnl[pnl<0].sum()),
        max_drawdown=float((eq-np.maximum.accumulate(eq)).min()),
        worst5pct_window_mean=float(w.pnl.nsmallest(math.ceil(.05*len(w))).mean()),
        inventory_square=float(p.inventory_square.mean()),mean_abs_inventory=float(p.mean_abs_inventory.mean()),
        mean_abs_net_end=float(p.net_end.abs().mean()),p95_abs_net_end=float(p.net_end.abs().quantile(.95)),
        max_abs_inventory=float(p.peak_abs_inventory.max()),mean_seconds_above10=float(p.seconds_above10.mean()),
        active_markets=int((w.n_fills>0).sum()),requests={k:sum(d['requests'].get(k,0) for d in ds) for k in ['place','cancel']},
        accepted_orders=sum(d['generations']['accepted_orders'] for d in ds),
        filled_orders_lower=sum(d['generations']['filled_generations_lower'] for d in ds),
        filled_orders_upper=sum(d['generations']['filled_generations_upper'] for d in ds),
        statuses=dict(eligible_markets=len(w),excluded_markets=0,missing_valuations=0),per_day=ds)
    result['plus1_per100_filled_shares']=100*result['plus1']/result['shares'] if result['shares'] else None
    result['paired_pnl_per_pair']=result['paired_pnl']/result['paired_shares'] if result['paired_shares'] else None
    return result


def main():
    results=[];trace=[];checks={};execution={}
    for tm in ['J','X']:
        for label,spec in ARMS.items():
            ds=[];ws=[];ps=[]
            for day in DAYS:
                d,w,p=summarize(day,tm,label,spec);ds.append(d);ws.append(w);ps.append(p)
                trace.extend(dict(x,timing=tm,variant=label) for x in p.to_dict('records'))
                path=folder(day,tm)
                checks[f'{tm}/{day}']=json.loads((path/'passive_checks.json').read_text())
                execution[f'{tm}/{day}']=dict(summary_sha=sha(path/'summary.json'),
                    execution=json.loads((path/'study_execution.json').read_text()),entry_audit_sha=sha(path/'entry_audit.json'),
                    baseline_reference=json.loads((path/'baseline_reference.json').read_text()) if (path/'baseline_reference.json').exists() else None)
            results.append(dict(timing=tm,variant=label,**aggregate(ds,ws,ps)))
    idx={(r['timing'],r['variant']):r for r in results};pairs=[('reserved10','baseline'),('uniform10','baseline'),('reserved10','uniform10')]
    family=[(tm,a,b,m) for tm in ['J','X'] for a,b in pairs for m in ['settle','plus1','inventory_square']]
    delta=np.array([[idx[tm,a]['per_day'][i][m]-idx[tm,b]['per_day'][i][m] for tm,a,b,m in family] for i in range(len(DAYS))])
    inf={f'{tm}/{a}-{b}/{m}':x for (tm,a,b,m),x in zip(family,paired_inference(delta))}
    comparisons=[]
    for tm in ['J','X']:
        for av,bv in pairs:
            a,b=idx[tm,av],idx[tm,bv];dd=np.array([x['settle']-y['settle'] for x,y in zip(a['per_day'],b['per_day'])])
            comparisons.append(dict(timing=tm,candidate=av,baseline=bv,
                delta={k:a[k]-b[k] for k in ['settle','plus1','mark_stop','carry','shares','inventory_square','mean_abs_inventory',
                    'mean_abs_net_end','daily_pnl_sd','worst_day','sum_losing_days','paired_pnl','residual_pnl']},
                days_settle_better=int((dd>1e-8).sum()),days_settle_worse=int((dd< -1e-8).sum()),
                leave_one_out_settle_delta=[float(dd.sum()-dd.max()),float(dd.sum()-dd.min())],
                share_change_pct=100*(a['shares']/b['shares']-1),
                paired_volume_change_pct=100*(a['paired_shares']/b['paired_shares']-1)))
    declaration=json.loads((DEST/'declaration.json').read_text())
    for p,h in declaration['prepared_input_sha'].items():
        if sha(p)!=h:raise ValueError('input changed')
    controls=(DEST/'logs/entry_controls.log').read_text()
    if '\nOK\n' not in controls or 'rc=0 ' not in controls:raise ValueError('missing controls')
    from . import pf_io as io
    raw=sorted({io.BN_DIR+f'/{h}.csv.gz' for day in DAYS for h in io.hours_of(day,extra_before=1,extra_after=1)
                if h[:8] not in ['20261028','20261029','20261030'] and h[:8]<'20261104'})
    from pathlib import Path
    sources=list(HERE.glob('pf_entry*.py'))+[HERE/'run_entry_budget.sh',HERE/'tests/test_entry_budget.py',
        HERE/'pf_sizing_engine.py',HERE/'pf_step4_ladder_v3.py',HERE/'pf_passive_report.py',HERE/'pf_settlement_study.py',HERE/'pf_io.py',HERE/'pf_clock_rl.py']
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),declaration=declaration,declaration_sha=sha(DEST/'declaration.json'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        source_sha={str(p.relative_to(HERE.parents[1])):sha(p) for p in sources},
        binance_input_sha={p:sha(p) for p in raw if Path(p).exists()},missing_binance_hour_files=[p for p in raw if not Path(p).exists()],
        results=results,comparisons=comparisons,paired_inference=inf,checks=checks,execution_inputs=execution,source_controls=controls,
        predicates={tm:dict(reserved_pnl_better_than_uniform=idx[tm,'reserved10']['settle']>idx[tm,'uniform10']['settle'],
            reserved_risk_lower_than_uniform=idx[tm,'reserved10']['inventory_square']<idx[tm,'uniform10']['inventory_square'],
            reserved_pnl_better_than_baseline=idx[tm,'reserved10']['settle']>idx[tm,'baseline']['settle']) for tm in ['J','X']},
        interpretation='in-sample policy comparison on nine consumed days; no matched post-entry inventory subgroup or independent validation',
        forward_data_read=False,live_settings_changed=False,deployment=False)
    pd.DataFrame(trace).to_parquet(DEST/'inventory_paths.parquet',index=False)
    write(DEST/'report.json',out);write(HERE.parent/'reviews/entry_budget_20261109/receipt.json',out)
    for r in results:print({k:r[k] for k in ['timing','variant','settle','plus1','shares','inventory_square','mean_abs_net_end','daily_pnl_sd','worst_day','paired_pnl']})


if __name__=='__main__':
    import sys
    if len(sys.argv)==1:main()
    elif len(sys.argv)==4 and sys.argv[1]=='audit':
        for label,spec in ARMS.items():summarize(sys.argv[2],sys.argv[3],label,spec)
        print('AUDITED',sys.argv[2:])
    else:raise ValueError('unknown report mode')
