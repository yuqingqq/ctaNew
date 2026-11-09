"""Verify frozen closing-liquidity artifacts and audit originating-order exposure."""
import datetime as dt
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from .pf_exit_study import HERE,DEST,PRIOR,DAYS,ARMS,sha,write
from .pf_exit_run import folder


def opening_audit(z):
    if not len(z) or not np.isfinite(z[['q','reducing_q','send_rel','t']]).all().all():
        raise ValueError('missing/invalid fill provenance')
    if (z.q<=0).any() or (z.reducing_q<0).any() or (z.reducing_q>z.q+1e-7).any():
        raise ValueError('invalid reduction accounting')
    late=z[z.t>=60]
    opening=late.q-late.reducing_q
    inherited=float(opening[late.send_rel<60].sum())
    new=float(opening[late.send_rel>=60].sum())
    if new>1e-7:raise ValueError('new late order opens inventory')
    if (z.t>240.1).any():raise ValueError('fill after carry valuation')
    return dict(inherited_opening=inherited,new_opening=new,fills=len(z),max_fill_second=float(z.t.max()))


def controls():
    z=pd.DataFrame(dict(q=[10.,5.],reducing_q=[8.,5.],send_rel=[59.,75.],t=[61.,76.]))
    a=opening_audit(z)
    assert a['inherited_opening']==2 and a['new_opening']==0
    for col,value in [('send_rel',60.),('t',241.),('reducing_q',float('nan'))]:
        bad=z.copy();bad.loc[0,col]=value
        try:opening_audit(bad)
        except ValueError:pass
        else:raise AssertionError(f'failed refusal: {col}')
    return dict(inherited_opening_detected=True,new_opening_refused=True,late_valuation_refused=True,missing_provenance_refused=True)


def main():
    checked=controls()
    r=json.loads((DEST/'report.json').read_text())
    review=HERE.parent/'reviews/exit_liquidity_20261109'
    if sha(DEST/'report.json')!=sha(review/'receipt.json'):raise ValueError('receipt copies differ')
    if sha(DEST/'declaration.json')!=r['declaration_sha'] or sha(review/'declaration.json')!=r['declaration_sha']:
        raise ValueError('declaration copies differ')
    for p,h in r['source_sha'].items():
        if sha(HERE.parents[1]/p)!=h:raise ValueError('source changed')
    if sha(PRIOR/'report.json')!=r['declaration']['prior_receipt_sha']:
        raise ValueError('reference receipt changed')
    prior=json.loads((PRIOR/'report.json').read_text())
    for tm in ['J','X']:
        for day in DAYS:
            key=f'{tm}/{day}'
            loc=Path(prior['artifacts'][key+'/late_reserved']['fills']['path']).parent.parent
            if sha(loc/'late_fills.parquet')!=prior['execution_inputs'][key]['late_fills_sha']:
                raise ValueError('reference fill metadata changed')
    count=0
    for item in r['artifacts'].values():
        for v in item.values():
            if sha(v['path'])!=v['sha']:raise ValueError('result artifact changed')
            count+=1
    if len(r['results'])!=len(ARMS)*2 or len(r['checks'])!=len(DAYS)*2:
        raise ValueError('incomplete declared experiment')
    for result in r['results']:
        if result['markets']!=2592 or result['statuses']!={'eligible_markets':2592,'excluded_markets':0,'missing_valuations':0}:
            raise ValueError('unreconciled population')
    audit={};rows=0
    for tm in ['J','X']:
        by_variant={v:[] for v in ['exit_keep','exit_close25']}
        for day in DAYS:
            out=folder(day,tm);key=f'{tm}/{day}';e=r['execution_inputs'][key]
            for f,k in [('summary.json','summary_sha'),('entry_audit.json','audit_sha'),('late_fills.parquet','late_fills_sha'),('exit_counts.json','exit_counts_sha')]:
                if sha(out/f)!=e[k]:raise ValueError('run metadata changed')
            if json.loads((out/'study_execution.json').read_text())!=e['execution']:raise ValueError('execution identity')
            if json.loads((out/'passive_checks.json').read_text())!=r['checks'][key]:raise ValueError('reconciliation identity')
            z=pd.read_parquet(out/'late_fills.parquet')
            for v in by_variant:by_variant[v].append(z[z.variant==v])
        for v,parts in by_variant.items():
            a=opening_audit(pd.concat(parts));audit[f'{tm}/{v}']=a;rows+=a['fills']
    metrics=['settle','plus1','mark_stop','carry','paired_pnl','residual_pnl','shares','reducing_shares',
             'opening_shares_after60','inventory_square','mean_abs_inventory','mean_abs_net_end',
             'p95_abs_net_end','daily_pnl_sd','worst_day','max_drawdown','mean_abs_net60','mean_abs_net120',
             'mean_abs_net180','mean_abs_net240','young_or_pending_shares','young_or_pending_plus1',
             'older_resting_shares','older_resting_plus1']
    records=[{k:x[k] for k in ['timing','variant',*metrics]}|{'places':x['requests']['place'],'cancels':x['requests']['cancel']} for x in r['results']]
    table=pd.DataFrame(records)
    with (DEST/'comparison.csv').open('x') as f:table.to_csv(f,index=False)
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),report_as_of=r['as_of'],report_sha=sha(DEST/'report.json'),
        verifier_sha=sha(__file__),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        population=dict(days_per_timing=9,markets_per_arm_timing=2592,candidate_fill_rows=rows),
        verified_sources=len(r['source_sha']),verified_result_artifacts=count,verified_metadata_sets=len(r['checks']),
        verified_reference_metadata_sets=len(DAYS)*2,
        controls=checked,inherited_vs_new_late_opening=audit,
        predicates=dict(no_late_new_order_opens_reverse=all(a['new_opening']<=1e-7 for a in audit.values()),
                        all_fills_before_stop_mark=all(a['max_fill_second']<=240.1 for a in audit.values())),
        comparison_csv_sha=sha(DEST/'comparison.csv'),scope='provenance and accounting verification; no additional hypothesis selection')
    write(DEST/'verification.json',out);write(review/'verification.json',out)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
