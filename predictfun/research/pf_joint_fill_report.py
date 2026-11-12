"""Paired support, calibration and decision-value assessment of the joint model."""
import datetime as dt
import json
import numpy as np
import pandas as pd
from . import pf_joint_fill_spec as S
from .pf_joint_fill import support,aggregate
from .pf_fill_value import targets,values,probability_loss,validate
from .pf_fill_value_report import auc,interval,rank_match,policy_stats,controls as report_controls
from .pf_order_policy_report import null

FAMILIES=['market_raw','market_cal','queue_raw','queue_cal']


def run():
    S.check();report_controls();fit=json.loads((S.OUT/'fit.json').read_text())
    if S.sha(S.OUT/'predictions.parquet')!=fit['predictions_sha']:raise ValueError('forecast hash')
    joint_n=0
    for fold in fit['folds']:
        day=fold['day'];p=S.OUT/'fold_predictions'/(day+'.parquet');q=p.with_name(day+'_joint.npz')
        if S.sha(p)!=fold['predictions_sha'] or S.sha(q)!=fold['joint_predictions_sha']:raise ValueError('fold identity')
        a=pd.read_parquet(p)
        with np.load(q) as arrays:
            for family in FAMILIES:
                j=arrays[family];support(j);joint_n+=len(j);z=aggregate(a,j)
                for branch,v in z.items():
                    for key,x in v.items():np.testing.assert_allclose(x,a[family+'_'+branch+'_'+key],atol=1e-10,rtol=0)
        for family,h in fold['models'].items():
            if S.sha(S.OUT/'models'/f'{day}_{family}.txt')!=h['model_sha'] or S.sha(S.OUT/'models'/f'{day}_{family}_joint.npz')!=h['joint_sha']:raise ValueError('saved model/distribution identity')
    a=pd.read_parquet(S.OUT/'predictions.parquet');validate(a)
    if len(a)!=fit['actions'] or joint_n!=len(a)*len(FAMILIES):raise ValueError('population')
    a['hour']=(a.decision//3600000).astype(int);a['inv_sign']=np.sign(a.inventory)
    a['age_bin']=pd.cut(a.order_age_ms,[0,100,250,500,1000,2000,5000,np.inf],include_lowest=True).astype(str)
    metrics=[];calibration=[];losses={};differences=[];support_rows=[]
    for branch in ('keep','reset'):
        truth=values(a,targets(a,branch));y=targets(a,branch)['fill'];qty=truth['quantity']
        for family in FAMILIES+['previous_queue','previous_baseline']:
            prefix=('queue' if family=='previous_queue' else 'baseline' if family=='previous_baseline' else family)+'_'+branch
            p=a[prefix+'_fill'].to_numpy();q=a[prefix+'_quantity'].to_numpy();q2=a[prefix+'_quantity2'].to_numpy()
            gain=a[prefix+'_gain'].to_numpy();loss=a[prefix+'_loss'].to_numpy();val=a[prefix+'_value'].to_numpy()
            brier=probability_loss(y,p);logloss=-(y*np.log(np.clip(p,1e-9,1-1e-9))+(1-y)*np.log(np.clip(1-p,1e-9,1-1e-9)))
            losses[branch,family]=dict(brier=brier,quantity_mse=(q-qty)**2,value_mse=(val-truth['value'])**2)
            metrics.append(dict(branch=branch,family=family,n=len(a),filled=int(y.sum()),fill_rate=float(y.mean()),mean_probability=float(p.mean()),brier=float(brier.mean()),
                log_loss=float(logloss.mean()),log_clip_rows=int(((p<1e-9)|(p>1-1e-9)).sum()),auc=auc(y,p),actual_quantity=float(qty.sum()),predicted_quantity=float(q.sum()),
                quantity_rmse=float(np.sqrt(np.mean((q-qty)**2))),value_rmse_c=float(np.sqrt(np.mean((val-truth['value'])**2))),
                gain_predicted_c=float(gain.sum()),loss_predicted_c=float(loss.sum())))
            if family in FAMILIES:
                rem=a.remaining.to_numpy();gcap=np.where(a.side.eq(0),100-a.order_price,a.order_price);lcap=100-gcap
                bad=(q< -1e-8)|(q>rem*p+1e-8)|(q2>rem*q+1e-8)|(q*q>p*q2+1e-8)|(gain< -1e-8)|(loss< -1e-8)|(gain/gcap+loss/lcap>q+1e-8)
                zero_bad=(q<=1e-12)&((gain+loss)>1e-8)
                support_rows.append(dict(family=family,branch=branch,n=len(a),violations=int(bad.sum()),zero_quantity_value_violations=int(zero_bad.sum())))
                if bad.any() or zero_bad.any():raise ValueError('joint forecast violated physical support')
                bins=np.minimum((p*10).astype(int),9)
                for b in range(10):
                    ix=bins==b
                    if ix.any():calibration.append(dict(kind='fill',family=family,branch=branch,bin=f'[{b/10:.1f},{(b+1)/10:.1f}]',n=int(ix.sum()),
                                                        predicted=float(p[ix].mean()),observed=float(y[ix].mean()),filled=int(y[ix].sum())))
        for family,base in [('market_cal','market_raw'),('queue_cal','queue_raw'),('queue_raw','market_raw'),('queue_cal','market_cal'),('queue_raw','previous_queue'),('queue_cal','previous_queue'),('queue_cal','previous_baseline')]:
            for key in ('brier','quantity_mse','value_mse'):
                differences.append(dict(branch=branch,model=family,baseline=base,metric=key,**interval(a,losses[branch,family][key]-losses[branch,base][key])))
    scorecols={family:family+'_score' for family in FAMILIES}
    scorecols.update(original_direct='queue_score',previous_components='queue_component_score')
    valuations=[];valerrors={};policies={}
    for family,col in scorecols.items():
        pred=a[col].to_numpy();y=a.label.to_numpy();err=pred-y;valerrors[family]=err**2;keep=pred>=0
        valuations.append(dict(model=family,n=len(a),rmse_c=float(np.sqrt(np.mean(err**2))),mae_c=float(np.abs(err).mean()),pulls=int((~keep).sum()),
                               keeps=int(keep.sum()),predicted_keep_c=float(pred[keep].sum()),actual_keep_c=float(y[keep].sum())))
        policies[family]=policy_stats(a,~keep)
    value_differences=[]
    for family,base in [('queue_cal','queue_raw'),('market_cal','market_raw'),('queue_cal','original_direct'),('queue_raw','original_direct'),('queue_cal','previous_components')]:
        value_differences.append(dict(model=family,baseline=base,**interval(a,valerrors[family]-valerrors[base])))
    decisions={}
    for family in FAMILIES:
        pull=a[scorecols[family]].lt(0).to_numpy();benefit=-a.label.to_numpy();n=null(a,pull,benefit,S.KEYS)
        expected=np.zeros(len(a))
        for ix in a.groupby(S.KEYS,dropna=False,sort=True).indices.values():expected[ix]=pull[ix].mean()*benefit[ix]
        comparisons={'random':interval(a,pull*benefit-expected,mean=False)};matched={}
        for base,col in [('original_direct','queue_score'),('previous_components','queue_component_score'),('guard_margin','guard_margin'),('same_family_raw',family.split('_')[0]+'_raw_score')]:
            mask=rank_match(a,pull,a[col].to_numpy());matched[base]=policy_stats(a,mask)
            comparisons[base]=interval(a,(pull.astype(int)-mask.astype(int))*benefit,mean=False)
        decisions[family]=dict(null=n,comparisons=comparisons,matched=matched)
        print('JOINT_POLICY_DONE',family,flush=True)
    def find(branch,model,base,metric):return next(x for x in differences if (x['branch'],x['model'],x['baseline'],x['metric'])==(branch,model,base,metric))
    selection=decisions['queue_cal']['comparisons']['original_direct']
    predicates=dict(all_forecasts_coherent=all(x['violations']==0 and x['zero_quantity_value_violations']==0 for x in support_rows),
                    all_calibrators_supported=all(v['status']=='PASS' for f in fit['folds'] for v in f['calibrators'].values()),
                    calibration_improves_queue_brier_both_branches=all(find(b,'queue_cal','queue_raw','brier')['delta']<0 for b in ('keep','reset')),
                    calibrated_queue_brier_beats_previous_both_branches=all(find(b,'queue_cal','previous_queue','brier')['delta']<0 for b in ('keep','reset')),
                    calibrated_queue_selection_beats_original=selection['delta']>0,
                    calibrated_queue_selection_day_interval_positive=selection['day_interval'][0]>0,
                    sequential_portfolio_tested=False,new_candidate_promoted=False)
    r=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),status='PASS',n=len(a),days=S.TEST,validation=False,
           declaration_sha=S.sha(S.OUT/'declaration.json'),fit_sha=S.sha(S.OUT/'fit.json'),predictions_sha=fit['predictions_sha'],
           joint_forecasts_verified=joint_n,support=support_rows,metrics=metrics,differences=differences,valuations=valuations,
           value_differences=value_differences,policies=policies,decisions=decisions,calibration=calibration,statuses=fit['statuses'],predicates=predicates,
           limits='Consumed simulator labels, short horizon and same-price reset retained. Core models exclude calibration day; prior controls used that extra day. Calibrated forecasts share joint-event weights, not separate output projections. No live calibration or portfolio profit claim.')
    S.write('result.json',r)
    lines=['# Coherent paired fill/value model — numerical results',f"As of {r['as_of']}; {len(a):,} actions, six consumed dates. Source/declaration, models, calibration and status records are in `reviews/joint_fill_20261112/`.", '',
           '| Branch | Model | Observed fill rate | Predicted fill rate | Brier | Quantity RMSE |', '|---|---|---:|---:|---:|---:|']
    for x in metrics:lines.append(f"| {x['branch']} | {x['family']} | {x['fill_rate']:.4f} | {x['mean_probability']:.4f} | {x['brier']:.6f} | {x['quantity_rmse']:.4f} |")
    lines+=['','| Model | Value RMSE, c | Pulls / keeps | Predicted / realized KEEP advantage on kept subset, c |','|---|---:|---:|---:|']
    for x in valuations:lines.append(f"| {x['model']} | {x['rmse_c']:.4f} | {x['pulls']:,} / {x['keeps']:,} | {x['predicted_keep_c']:.2f} / {x['actual_keep_c']:.2f} |")
    lines+=['','| Policy | Matched original-direct increment, c [day interval] | Matched guard-margin increment, c [day interval] |','|---|---:|---:|']
    for name,x in decisions.items():
        d=x['comparisons']['original_direct'];g=x['comparisons']['guard_margin']
        lines.append(f"| {name} | {d['delta']:+.2f} [{d['day_interval'][0]:+.2f}, {d['day_interval'][1]:+.2f}] | {g['delta']:+.2f} [{g['day_interval'][0]:+.2f}, {g['day_interval'][1]:+.2f}] |")
    lines+=['',f'Local values are overlapping short-horizon counterfactual utility, not portfolio P&L. All head values derive from a common mixture of paired valid outcomes. All {joint_n:,} joint forecasts and {sum(x["n"] for x in support_rows):,} branch forecasts are checked for physical support; detailed computed counts are in JSON. Probability log loss clips only for reporting at 1e-9, with clipped counts recorded; the forecasts themselves are unchanged.',
            '', 'Core training excludes the preceding calibration day, so comparing to older full-training models is not equal-data architectural attribution. Raw versus calibrated variants use exactly the same core model; their difference isolates the calibration step. Day intervals resample six UTC days 1,000 times; nulls use 500 matched-count shuffles. No untouched-validation claim.',
            '', 'Computed predicates: `'+json.dumps(S.native(predicates),sort_keys=True)+'`', '',
            'The short fill horizon, same-price reset semantics, exact fill timing and live queue/fill calibration remain outside this stage. Read the human review before interpreting candidate quality. No live changes or model promotion.']
    for base in (S.WORK,S.ROOT):
        with (base/'predictfun/handoffs/20261112_joint_fill_results.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    S.progress('numerical_report','complete','Joint support, chronological calibration and matched-value diagnostics complete; human review pending.')
    print('JOINT_FILL_REPORT_PASS',flush=True)


if __name__=='__main__':run()
