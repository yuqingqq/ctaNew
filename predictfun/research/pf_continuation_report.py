"""Predeclared local target/opportunity diagnostics; no portfolio promotion."""
import datetime as dt
import json
import numpy as np
import pandas as pd
from . import pf_continuation_spec as S
from .pf_fill_value_report import interval,rank_match,policy_stats,controls
from .pf_order_policy_report import null

FAMILIES=('baseline','market_raw','market_cal','queue_raw','queue_cal')


def run():
    S.check();controls();fit=json.loads((S.OUT/'fit.json').read_text());p=S.OUT/'predictions.parquet'
    if S.sha(p)!=fit['predictions_sha']:raise ValueError('prediction identity')
    a=pd.read_parquet(p)
    if a.id.duplicated().any() or len(a)!=fit['n'] or set(a.day)!=set(S.TEST):raise ValueError('evaluation population')
    checked=0
    for fold in fit['folds']:
        for family,model in fold['models'].items():
            if S.sha(S.OUT/'models'/f"{fold['day']}_{family}.txt")!=model['model_sha'] or S.sha(S.OUT/'models'/f"{fold['day']}_{family}.npz")!=model['table_sha']:raise ValueError('saved model identity')
            if model['reload_max_abs_error']>1e-9:raise ValueError('saved model reconstruction')
            checked+=model['checked_forecasts']
    if checked!=4*len(a):raise ValueError('unchecked forecasts')
    a['label']=a.keep_10000_value-a.pull_10000_value
    a['hour']=(a.decision//3600000).astype(int);a['inv_sign']=np.sign(a.inventory)
    a['age_bin']=pd.cut(a.order_age_ms,[0,100,250,500,1000,2000,5000,np.inf],include_lowest=True).astype(str)
    keys=['day','side','hour','age_bin','inv_sign'];metrics=[];diffs=[];losses={}
    # The model's time forecast is a cumulative event curve, not a fabricated
    # precise venue fill timestamp. No-fill actions remain in every denominator.
    targets=[f'{b}_first_{h}' for b in ('keep','pull') for h in (250,1000,2000,10000,30000)]
    targets += [f'{b}_{h}_{k}' for b in ('keep','pull') for h in S.HORIZONS for k in ('q','same_q','value')]
    targets += ['flow_'+str(h) for h in S.HORIZONS]
    for target in targets:
        y=a[target].to_numpy()
        for family in FAMILIES:
            x=a[family+'_'+target].to_numpy();err=(x-y)**2;losses[family,target]=err
            metrics.append(dict(target=target,family=family,n=len(a),actual_mean=float(y.mean()),predicted_mean=float(x.mean()),mse=float(err.mean())))
        for model,base in [('queue_cal','queue_raw'),('queue_cal','market_cal'),('queue_cal','baseline')]:
            diffs.append(dict(target=target,model=model,baseline=base,**interval(a,losses[model,target]-losses[base,target])))
    horizons=[]
    for h in S.HORIZONS:
        y=a[f'keep_{h}_value']-a[f'pull_{h}_value']
        horizons.append(dict(horizon=h,n=len(a),keep_q=float(a[f'keep_{h}_q'].sum()),pull_q=float(a[f'pull_{h}_q'].sum()),
            label_sum_c=float(y.sum()),label_nonzero=int(y.abs().gt(1e-9).sum()),keep_first_probability=float(a[f'keep_first_{h}'].mean()),
            pull_first_probability=float(a[f'pull_first_{h}'].mean()),flow_sum=float(a[f'flow_{h}'].sum()),
            opposite_sign_vs_2s=int(((y*(a.keep_2000_value-a.pull_2000_value))<0).sum())))
    tail=dict(n=len(a),increment_2_to_10=interval(a,(a.keep_10000_value-a.pull_10000_value)-(a.keep_2000_value-a.pull_2000_value)),
              increment_10_to_30=interval(a,(a.keep_30000_value-a.pull_30000_value)-a.label),
              keep_no_same_side_fill_30s=int(a.keep_first_30000.eq(0).sum()),pull_no_same_side_fill_30s=int(a.pull_first_30000.eq(0).sum()))
    decisions={};values=[]
    for family in FAMILIES[1:]:
        score=a[family+'_score_10000'].to_numpy();pull=score<0;benefit=-a.label.to_numpy()
        z=null(a,pull,benefit,keys);comparisons={};matched={}
        for label,col in [('original_direct','original_direct_score'),('market_cal','market_cal_score_10000'),('guard_margin','guard_margin'),('same_family_raw',family.split('_')[0]+'_raw_score_10000')]:
            other=rank_match(a,pull,a[col].to_numpy())
            comparisons[label]=interval(a,(pull.astype(int)-other.astype(int))*benefit,mean=False)
            matched[label]=policy_stats(a,other)
        expected=np.zeros(len(a))
        for ix in a.groupby(keys,dropna=False,sort=True).indices.values():expected[ix]=pull[ix].mean()*benefit[ix]
        comparisons['random']=interval(a,pull*benefit-expected,mean=False)
        decisions[family]=dict(policy=policy_stats(a,pull),null=z,comparisons=comparisons,matched=matched)
        values.append(dict(family=family,n=len(a),rmse_c=float(np.sqrt(np.mean((score-a.label)**2))),pulls=int(pull.sum()),keeps=int((~pull).sum()),
                           predicted_kept_c=float(score[~pull].sum()),actual_kept_c=float(a.label.to_numpy()[~pull].sum())))
    def error(target,base):return next(v for v in diffs if v['model']=='queue_cal' and v['baseline']==base and v['target']==target)
    pred=dict(coherent_source_models_verified=checked==4*len(a),
        same_side_fill_10s_beats_baseline_both=all(error(f'{b}_first_10000','baseline')['delta']<0 for b in ('keep','pull')),
        incoming_flow_10s_beats_baseline=error('flow_10000','baseline')['delta']<0,
        queue_selection_beats_original=decisions['queue_cal']['comparisons']['original_direct']['delta']>0,
        queue_selection_interval_positive=decisions['queue_cal']['comparisons']['original_direct']['day_interval'][0]>0,
        live_calibration_complete=False,sequential_portfolio_tested=False,promoted=False)
    r=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),status='PASS',n=len(a),days=S.TEST,validation=False,
           declaration_sha=S.sha(S.OUT/'declaration.json'),fit_sha=S.sha(S.OUT/'fit.json'),predictions_sha=fit['predictions_sha'],
           populations=[json.loads((S.OUT/f'build_{d}.json').read_text()) for d in S.DAYS],paired_statuses=fit['statuses'],
           metrics=metrics,differences=diffs,horizons=horizons,tail=tail,valuations=values,decisions=decisions,predicates=pred,
           limits='Quarter-sampled neutral actions, consumed dates, inherited guard features. Full engine removes same-price reset approximation; observed execution timestamps/queue model remain uncalibrated. First-fill curves are horizon probabilities. No value beyond30s and no new sequential portfolio test. Old direct model uses a different short target and one additional prior training day.')
    S.write('result.json',r)
    lines=['# Full-engine continuation — numerical results',f"As of {r['as_of']}; n={len(a):,} sampled eligible actions on six consumed dates. See receipt populations for every sampling/exclusion status.",'',
        '| Horizon | KEEP shares | PULL shares | KEEP-minus-PULL local utility, cents | Nonzero actions |', '|---|---:|---:|---:|---:|']
    for x in horizons:lines.append(f"| {x['horizon']/1000:g}s | {x['keep_q']:.2f} | {x['pull_q']:.2f} | {x['label_sum_c']:.2f} | {x['label_nonzero']} |")
    lines+=['','| Model | Value RMSE, cents | Pulls | Keeps | Predicted / observed kept value, cents |','|---|---:|---:|---:|---:|']
    for x in values:lines.append(f"| {x['family']} | {x['rmse_c']:.4f} | {x['pulls']} | {x['keeps']} | {x['predicted_kept_c']:.2f} / {x['actual_kept_c']:.2f} |")
    lines+=['','Computed predicates: `'+json.dumps(pred,sort_keys=True)+'`','',r['limits'],
        '', 'These are overlapping local utility sums, not portfolio P&L. No strategy is promoted. Read the human review before interpreting these diagnostics.']
    for base in (S.WORK,S.ROOT):
        with (base/'predictfun/handoffs/20261112_continuation_results.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    S.progress('numerical_report','complete','Full-engine horizon, flow, fill-opportunity and local selection reports saved; human review pending.')

if __name__=='__main__':run()
