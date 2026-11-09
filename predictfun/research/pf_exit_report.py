"""Closing-liquidity attribution, exact prefix checks, and day-level comparisons."""
import datetime as dt
import json
import subprocess
import numpy as np
import pandas as pd
from .pf_exit_study import HERE,DAYS,ARMS,DEST,PRIOR,PAIRS,sha,write
from .pf_exit_run import folder,location,stem,reconcile_metadata
from . import pf_entry_report as old
from .pf_settlement_study import paired_inference

SUMS=['reducing_shares','opening_shares_after60','zero_crossings_after60','overshoot_shares','reducing_mo1',
      'late_concession_dollars','late_concession_shares','late_concession_reducing_shares','post60_shares','post60_plus1','young_or_pending_shares','young_or_pending_plus1','older_resting_shares','older_resting_plus1']
MEANS=['mean_abs_net60','mean_abs_net120','mean_abs_net180','mean_abs_net240','mean_time_weighted_abs_net_60_300']



def fill_profile(f):
    if not np.isfinite(f.age_ms).all() or (f.age_ms<0).any() or not np.isfinite(f.mo1).all():
        raise ValueError('invalid age/markout attribution')
    mask=f.zombie|(f.age_ms<100)
    return dict(young_or_pending_shares=float(f.loc[mask,'q'].sum()),young_or_pending_plus1=float(f.loc[mask,'mo1'].sum()),
                older_resting_shares=float(f.loc[~mask,'q'].sum()),older_resting_plus1=float(f.loc[~mask,'mo1'].sum()))


def summarize(day,tm,label,spec):
    old.folder=lambda d,t:location(label,d,t)
    d,w,p=old.summarize(day,tm,label,spec)
    loc=location(label,day,tm)
    f=pd.read_parquet(loc/'fills'/stem(spec))
    if label!='baseline':
        meta=pd.read_parquet(loc/'late_fills.parquet')
        z=meta[meta.variant==label]
        reconcile_metadata(f,z)
        shifted=z[z.concession_ticks>0]
        d.update(late_concession_dollars=float(z.concession.sum()),late_concession_shares=float(shifted.q.sum()),
                 late_concession_reducing_shares=float(shifted.reducing_q.sum()))
    else:
        d.update(late_concession_dollars=0.,late_concession_shares=0.,late_concession_reducing_shares=0.)
    d.update(fill_profile(f))
    late=f[f.t>=60]
    d.update(post60_shares=float(late.q.sum()),post60_plus1=float(late.mo1.sum()))
    for at in [60,120,180,240]:
        fs=f[f.t<at]
        signed=fs.q*np.where(fs.side==0,1.,-1.)
        net=signed.groupby(fs.mid).sum().reindex(w.mid,fill_value=0.)
        d[f'mean_abs_net{at}']=float(net.abs().mean())
        p[f'net{at}']=p.mid.map(net)
    return d,w,p


def main():
    results=[];paths=[];checks={};executions={};artifacts={};mechanisms={}
    for tm in ['J','X']:
        for label,spec in ARMS.items():
            ds=[];ws=[];ps=[]
            for day in DAYS:
                d,w,p=summarize(day,tm,label,spec)
                ds.append(d);ws.append(w);ps.append(p)
                paths.extend(dict(x,timing=tm,variant=label) for x in p.to_dict('records'))
                out=folder(day,tm);loc=location(label,day,tm)
                checks[f'{tm}/{day}']=json.loads((out/'passive_checks.json').read_text())
                executions[f'{tm}/{day}']=dict(summary_sha=sha(out/'summary.json'),
                    execution=json.loads((out/'study_execution.json').read_text()),
                    audit_sha=sha(out/'entry_audit.json'),late_fills_sha=sha(out/'late_fills.parquet'),exit_counts_sha=sha(out/'exit_counts.json'))
                mechanisms[f'{tm}/{day}']=json.loads((out/'exit_counts.json').read_text())
                artifacts[f'{tm}/{day}/{label}']={k:dict(path=str(loc/k/stem(spec)),sha=sha(loc/k/stem(spec))) for k in ['win','fills','requests']}
            r=old.aggregate(ds,ws,ps)
            r.update({k:sum(d[k] for d in ds) for k in SUMS})
            r.update({k:sum(d[k] for d in ds)/len(ds) for k in MEANS})
            results.append(dict(timing=tm,variant=label,**r))
    idx={(r['timing'],r['variant']):r for r in results}
    family=[(tm,a,b,m) for tm in ['J','X'] for a,b in PAIRS for m in ['settle','plus1','inventory_square']]
    delta=np.array([[idx[tm,a]['per_day'][i][m]-idx[tm,b]['per_day'][i][m] for tm,a,b,m in family] for i in range(len(DAYS))])
    inf={f'{tm}/{a}-{b}/{m}':x for (tm,a,b,m),x in zip(family,paired_inference(delta))}
    comparisons=[]
    for tm in ['J','X']:
        for an,bn in PAIRS:
            a,b=idx[tm,an],idx[tm,bn]
            dd=np.array([x['settle']-y['settle'] for x,y in zip(a['per_day'],b['per_day'])])
            comparisons.append(dict(timing=tm,candidate=an,reference=bn,
                delta={k:a[k]-b[k] for k in ['settle','plus1','mark_stop','carry','paired_pnl','residual_pnl',
                    'mean_abs_net_end','inventory_square','daily_pnl_sd',*SUMS,*MEANS]},
                days_settle_better=int((dd>1e-8).sum()),days_settle_worse=int((dd< -1e-8).sum()),
                leave_one_day_out_delta=[float(dd.sum()-dd.max()),float(dd.sum()-dd.min())],
                inventory_change_pct=100*(a['mean_abs_net_end']/b['mean_abs_net_end']-1)))
    declaration=json.loads((DEST/'declaration.json').read_text())
    for p,h in declaration['prepared_input_sha'].items():
        if sha(p)!=h:raise ValueError('prepared input changed')
    for p,h in declaration['prepared_source_sha'].items():
        if sha(HERE.parents[1]/p)!=h:raise ValueError('declared source changed')
    for p,h in declaration['evidence_snapshot_sha'].items():
        if sha(p)!=h:raise ValueError('evidence snapshot changed')
    if sha(PRIOR/'report.json')!=declaration['prior_receipt_sha']:raise ValueError('reference receipt changed')
    prior=json.loads((PRIOR/'report.json').read_text())
    for p,h in prior['binance_input_sha'].items():
        if sha(p)!=h:raise ValueError('bounded raw input changed')
    controls=(DEST/'logs/exit_controls.log').read_text()
    if '\nOK\n' not in controls or 'rc=0 ' not in controls:raise ValueError('missing source controls')
    sources=list(HERE.glob('pf_exit*.py'))+list(HERE.glob('pf_late*.py'))+[HERE/'run_exit_liquidity.sh',HERE/'tests/test_exit_liquidity.py',
        HERE/'pf_inflight_engine.py',HERE/'pf_entry_engine.py',HERE/'pf_entry_report.py',HERE/'pf_sizing_engine.py',
        HERE/'pf_step4_ladder_v3.py',HERE/'pf_passive_report.py',HERE/'pf_settlement_study.py',HERE/'pf_io.py',HERE/'pf_clock_rl.py']
    for v in executions.values():
        if v['execution']['adapter_sha']!=sha(HERE/'pf_exit_engine.py'):raise ValueError('per-run source mismatch')
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),declaration=declaration,declaration_sha=sha(DEST/'declaration.json'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        source_sha={str(p.relative_to(HERE.parents[1])):sha(p) for p in sources},results=results,comparisons=comparisons,
        paired_inference=inf,checks=checks,execution_inputs=executions,artifacts=artifacts,source_controls=controls,mechanism_decision_counts=mechanisms,
        binance_input_sha=prior['binance_input_sha'],missing_binance_hour_files=prior['missing_binance_hour_files'],
        predicates={v:dict(lower_terminal_inventory_both=all(idx[t,v]['mean_abs_net_end']<idx[t,'baseline']['mean_abs_net_end'] for t in ['J','X']),
            higher_settlement_both=all(idx[t,v]['settle']>idx[t,'baseline']['settle'] for t in ['J','X']),
            same_mean_inventory_at60=all(idx[t,v]['mean_abs_net60']==idx[t,'baseline']['mean_abs_net60'] for t in ['J','X']),
            observed_value_preserved_and_inventory_lower_both=all(idx[t,v]['plus1']>=idx[t,'baseline']['plus1'] and idx[t,v]['mark_stop']>=idx[t,'baseline']['mark_stop'] and idx[t,v]['mean_abs_net_end']<idx[t,'baseline']['mean_abs_net_end'] for t in ['J','X'])) for v in ['exit_keep','exit_close25']},
        interpretation='conditional engineering comparison on consumed dates; fill model questioned by live evidence; no independently validated or live-achievable profit claim',
        attribution='no price concessions; FIFO includes drift; age/pending contribution is descriptive, not corrected-strategy PnL',
        forward_data_read=False,live_settings_changed=False,deployment=False)
    pd.DataFrame(paths).to_parquet(DEST/'inventory_paths.parquet',index=False)
    write(DEST/'report.json',out)
    write(HERE.parent/'reviews/exit_liquidity_20261109/receipt.json',out)
    for r in results:
        print({k:r[k] for k in ['timing','variant','settle','plus1','mark_stop','carry','paired_pnl','residual_pnl',
              'mean_abs_net_end','daily_pnl_sd','reducing_shares','young_or_pending_plus1']})


if __name__=='__main__':
    main()
