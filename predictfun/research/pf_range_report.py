"""All-market inventory, trading attribution and paired UTC-day comparisons."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from .pf_range_study import HERE, DAYS, ARMS, DEST, PRIOR, PAIRS, sha, write
from .pf_range_run import folder, location, stem
from . import pf_entry_report as old
from .pf_settlement_study import paired_inference


def summarize(day,tm,label,spec):
    old.folder=lambda d,t:location(label,d,t)
    d,w,p=old.summarize(day,tm,label,spec)
    f=pd.read_parquet(location(label,day,tm)/'fills'/stem(spec))
    flips=[]
    for mid,g in f.groupby('mid',sort=False):
        net=(g.q*np.where(g.side==0,1.,-1.)).cumsum().to_numpy();before=np.r_[0.,net[:-1]]
        flip=(before*net < -1e-8)
        flips.append(dict(mid=mid,zero_crossings=int(flip.sum()),zero_crossings_after60=int((flip&(g.t>=60)).sum())))
    z=pd.DataFrame(flips,columns=['mid','zero_crossings','zero_crossings_after60'])
    p=p.merge(z,on='mid',how='left',validate='one_to_one').fillna({'zero_crossings':0,'zero_crossings_after60':0})
    d['zero_crossings']=int(p.zero_crossings.sum());d['zero_crossings_after60']=int(p.zero_crossings_after60.sum())
    if label in ['range10','both_reserved'] and p.peak_abs_inventory.max()>10+1e-7:
        raise ValueError('independent inventory envelope reconciliation failed')
    return d,w,p


def main():
    results=[];paths=[];checks={};executions={};artifacts={}
    for tm in ['J','X']:
        for label,spec in ARMS.items():
            ds=[];ws=[];ps=[]
            for day in DAYS:
                d,w,p=summarize(day,tm,label,spec);ds.append(d);ws.append(w);ps.append(p)
                paths.extend(dict(x,timing=tm,variant=label) for x in p.to_dict('records'))
                out=folder(day,tm);loc=location(label,day,tm)
                checks[f'{tm}/{day}']=json.loads((out/'passive_checks.json').read_text())
                executions[f'{tm}/{day}']=dict(summary_sha=sha(out/'summary.json'),
                    execution=json.loads((out/'study_execution.json').read_text()),audit_sha=sha(out/'entry_audit.json'))
                artifacts[f'{tm}/{day}/{label}']={k:dict(path=str(loc/k/stem(spec)),sha=sha(loc/k/stem(spec))) for k in ['win','fills','requests']}
            r=old.aggregate(ds,ws,ps)
            r.update(zero_crossings=sum(d['zero_crossings'] for d in ds),zero_crossings_after60=sum(d['zero_crossings_after60'] for d in ds))
            results.append(dict(timing=tm,variant=label,**r))
    idx={(r['timing'],r['variant']):r for r in results}
    family=[(tm,a,b,m) for tm in ['J','X'] for a,b in PAIRS for m in ['settle','plus1','inventory_square']]
    delta=np.array([[idx[tm,a]['per_day'][i][m]-idx[tm,b]['per_day'][i][m] for tm,a,b,m in family] for i in range(len(DAYS))])
    inf={f'{tm}/{a}-{b}/{m}':x for (tm,a,b,m),x in zip(family,paired_inference(delta))}
    declaration=json.loads((DEST/'declaration.json').read_text())
    for p,h in declaration['prepared_input_sha'].items():
        if sha(p)!=h:raise ValueError('prepared input changed')
    if sha(PRIOR/'report.json')!=declaration['prior_receipt_sha']:raise ValueError('prior receipt changed')
    controls=(DEST/'logs/range_controls.log').read_text()
    if '\nOK\n' not in controls or 'rc=0 ' not in controls:raise ValueError('missing controls')
    prior=json.loads((PRIOR/'report.json').read_text())
    for p,h in prior['binance_input_sha'].items():
        if sha(p)!=h:raise ValueError('bounded raw input changed')
    sources=list(HERE.glob('pf_range*.py'))+[HERE/'run_inventory_range.sh',HERE/'tests/test_range_budget.py',
        HERE/'pf_inflight_engine.py',HERE/'pf_entry_engine.py',HERE/'pf_entry_report.py',HERE/'pf_sizing_engine.py',HERE/'pf_step4_ladder_v3.py',
        HERE/'pf_passive_report.py',HERE/'pf_settlement_study.py',HERE/'pf_io.py',HERE/'pf_clock_rl.py']
    for k,e in executions.items():
        if e['execution']['adapter_sha']!=sha(HERE/'pf_range_engine.py'):raise ValueError('per-run source mismatch')
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),declaration=declaration,declaration_sha=sha(DEST/'declaration.json'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        source_sha={str(p.relative_to(HERE.parents[1])):sha(p) for p in sources},results=results,paired_inference=inf,
        checks=checks,execution_inputs=executions,artifacts=artifacts,source_controls=controls,
        binance_input_sha=prior['binance_input_sha'],missing_binance_hour_files=prior['missing_binance_hour_files'],
        predicates={tm:dict(range_stayed_within10=idx[tm,'range10']['max_abs_inventory']<=10+1e-7,
            range_crossed_zero=idx[tm,'range10']['zero_crossings']>0,
            range_pnl_improves_strict=idx[tm,'range10']['settle']>idx[tm,'both_reserved']['settle'],
            range_pnl_improves_adding_only=idx[tm,'range10']['settle']>idx[tm,'reserved10']['settle']) for tm in ['J','X']},
        interpretation='exploratory after prior policy selection; all markets, no matched post-entry subgroup; no independent validation',
        forward_data_read=False,live_settings_changed=False,deployment=False)
    pd.DataFrame(paths).to_parquet(DEST/'inventory_paths.parquet',index=False)
    write(DEST/'report.json',out);write(HERE.parent/'reviews/inventory_range_20261109/receipt.json',out)
    for r in results:print({k:r[k] for k in ['timing','variant','settle','plus1','shares','inventory_square','mean_abs_net_end','daily_pnl_sd','worst_day','max_abs_inventory','zero_crossings','zero_crossings_after60','paired_pnl']})


if __name__=='__main__':
    import sys
    if len(sys.argv)==1:main()
    elif len(sys.argv)==4 and sys.argv[1]=='audit':
        for label,spec in ARMS.items():summarize(sys.argv[2],sys.argv[3],label,spec)
        print('AUDITED',sys.argv[2:])
    else:raise ValueError('unknown mode')
