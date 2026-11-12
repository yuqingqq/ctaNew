"""Calibration, matched action value and day-unit uncertainty for fill forecasts."""
import datetime as dt
import json
import numpy as np
import pandas as pd
from . import pf_fill_value_spec as S
from .pf_fill_value import probability_loss, validate, targets, constrain, values
from .pf_order_policy_report import null


def auc(y,p):
    y=np.asarray(y,bool);p=np.asarray(p,float);n=int(y.sum());m=len(y)-n
    if not n or not m:return None
    ranks=pd.Series(p).rank(method='average').to_numpy()
    return float((ranks[y].sum()-n*(n+1)/2)/(n*m))


def interval(a,x,mean=True):
    if set(a.day)!=set(S.TEST) or not np.isfinite(x).all():raise ValueError('day interval population')
    x=np.asarray(x,float);s=np.array([x[a.day.eq(d).to_numpy()].sum() for d in S.TEST])
    n=np.array([int(a.day.eq(d).sum()) for d in S.TEST])
    ix=np.random.default_rng(S.SEED).integers(0,len(s),(1000,len(s)))
    draws=s[ix].sum(axis=1)
    if mean:draws=draws/n[ix].sum(axis=1)
    return dict(delta=float(x.mean() if mean else x.sum()),day_interval=np.quantile(draws,[.025,.975]).tolist(),
                daily=[dict(day=d,n=int(nn),sum=float(ss),mean=float(ss/nn)) for d,nn,ss in zip(S.TEST,n,s)])


def rank_match(a,pull,score):
    z=np.zeros(len(a),bool)
    ids=a.id.to_numpy();score=np.asarray(score,float)
    for ix in a.groupby(S.KEYS,dropna=False,sort=True).indices.values():
        n=int(pull[ix].sum());z[ix[np.lexsort((ids[ix],score[ix]))[:n]]]=True
        if int(z[ix].sum())!=n:raise ValueError('matched count')
    return z


def policy_stats(a,pull):
    y=-a.label.to_numpy();v=y[pull]
    good=v>S.EPS;bad=v<-S.EPS
    return dict(n=len(a),pulls=int(pull.sum()),benefit_c=float(v.sum()),
                beneficial=int(good.sum()),harmful=int(bad.sum()),indifferent=int((~good&~bad).sum()),
                good_benefit_c=float(v[good].sum()),bad_cost_c=float(-v[bad].sum()),
                kept_n=int((~pull).sum()),kept_keep_advantage_c=float(-y[~pull].sum()))


def controls():
    y=np.array([0,0,1,1]);p=np.array([.1,.2,.8,.9])
    if auc(y,p)!=1. or auc(y,1-p)!=0.:raise ValueError('AUC instrument')
    a=pd.DataFrame(dict(day=np.repeat(S.TEST,2)))
    z=interval(a,np.ones(len(a)))
    if z['delta']!=1 or z['day_interval']!=[1.,1.]:raise ValueError('day bootstrap instrument')
    try:interval(a.iloc[:-2],np.ones(len(a)-2))
    except ValueError:pass
    else:raise ValueError('missing day accepted')
    a=pd.DataFrame(dict(id=['a','b','c'],day=['x']*3,side=[0]*3,hour=[0]*3,age_bin=['a']*3,inv_sign=[0]*3))
    if not np.array_equal(rank_match(a,np.array([True,False,False]),[2.,-1.,0.]),[False,True,False]):raise ValueError('matched ranking instrument')


def run():
    S.check();controls()
    fit=json.loads((S.OUT/'fit.json').read_text());p=S.OUT/'predictions.parquet'
    if S.sha(p)!=fit['predictions_sha']:raise ValueError('prediction artifact changed')
    for f in fit['folds']:
        if S.sha(S.OUT/'fold_predictions'/f"{f['day']}.parquet")!=f['predictions_sha']:raise ValueError('fold predictions changed')
        for name,h in f['models_sha'].items():
            if S.sha(S.OUT/'models'/name)!=h:raise ValueError('model artifact changed')
    a=pd.read_parquet(p);validate(a)
    if len(a)!=fit['test_actions'] or set(a.day)!=set(S.TEST):raise ValueError('report cohort')
    a['hour']=(a.decision//3600000).astype(int)
    a['age_bin']=pd.cut(a.order_age_ms,[0,100,250,500,1000,2000,5000,np.inf],include_lowest=True).astype(str)
    a['inv_sign']=np.sign(a.inventory)
    metrics=[];calibration=[];losses={};branch_differences=[]
    for branch in ('keep','reset'):
        t=targets(a,branch);truth=values(a,t);y=t['fill'];q=truth['quantity']
        for family in ('baseline','market','queue'):
            prefix=family+'_'+branch
            raw={head:a[prefix+'_'+head+'_raw'].to_numpy() for head in S.HEADS}
            z=constrain(raw);v=values(a,z)
            for head in S.HEADS:np.testing.assert_allclose(z[head],a[prefix+'_'+head],atol=1e-12,rtol=0)
            for head in v:np.testing.assert_allclose(v[head],a[prefix+'_'+head],atol=1e-10,rtol=0)
            brier=probability_loss(y,z['fill']);logloss=-(y*np.log(z['fill'])+(1-y)*np.log1p(-z['fill']))
            qtyerr=v['quantity']-q;valueerr=v['value']-truth['value']
            losses[branch,family]=dict(brier=brier,quantity_mse=qtyerr**2,value_mse=valueerr**2)
            metrics.append(dict(branch=branch,family=family,n=len(a),filled=int(y.sum()),unfilled=int((1-y).sum()),
                fill_rate=float(y.mean()),mean_probability=float(z['fill'].mean()),brier=float(brier.mean()),log_loss=float(logloss.mean()),auc=auc(y,z['fill']),
                actual_quantity=float(q.sum()),predicted_quantity=float(v['quantity'].sum()),quantity_mae=float(np.abs(qtyerr).mean()),quantity_rmse=float(np.sqrt(np.mean(qtyerr**2))),
                gain_actual_c=float(t['gain'].sum()),gain_predicted_c=float(z['gain'].sum()),loss_actual_c=float(t['loss'].sum()),loss_predicted_c=float(z['loss'].sum()),
                value_actual_c=float(truth['value'].sum()),value_predicted_c=float(v['value'].sum()),value_mae_c=float(np.abs(valueerr).mean()),value_rmse_c=float(np.sqrt(np.mean(valueerr**2)))))
            bins=np.minimum((z['fill']*10).astype(int),9)
            for b in range(10):
                ix=bins==b
                if not ix.any():continue
                calibration.append(dict(kind='fill',family=family,branch=branch,bin=f'[{b/10:.1f},{(b+1)/10:.1f})',n=int(ix.sum()),
                                        predicted=float(z['fill'][ix].mean()),observed=float(y[ix].mean()),filled=int(y[ix].sum()),actual_qty=float(q[ix].sum()),predicted_qty=float(v['quantity'][ix].sum())))
        for family,base in [('market','baseline'),('queue','baseline'),('queue','market')]:
            for key in ('brier','quantity_mse','value_mse'):
                diff=losses[branch,family][key]-losses[branch,base][key]
                branch_differences.append(dict(branch=branch,family=family,baseline=base,metric=key,**interval(a,diff)))
    valuations=[];valuation_differences=[];policies={}
    scorecols={'baseline':'baseline_component_score','market_component':'market_component_score','queue_component':'queue_component_score',
               'market_direct':'market_score','queue_direct':'queue_score'}
    val_loss={}
    for name,col in scorecols.items():
        pred=a[col].to_numpy();y=a.label.to_numpy();err=pred-y;val_loss[name]=err**2
        valuations.append(dict(model=name,n=len(a),prediction_c=float(pred.sum()),actual_c=float(y.sum()),mae_c=float(np.abs(err).mean()),rmse_c=float(np.sqrt(np.mean(err**2))),
                               pull_n=int((pred<0).sum()),keep_n=int((pred>=0).sum()),predicted_keep_value_c=float(pred[pred>=0].sum()),actual_keep_value_c=float(y[pred>=0].sum())))
        bins=pd.cut(pred,[-np.inf,-5,-1,-.1,0,.1,1,5,np.inf],right=False)
        for b in bins.categories:
            ix=np.asarray(bins==b)
            if ix.any():calibration.append(dict(kind='value',model=name,bin=str(b),n=int(ix.sum()),predicted=float(pred[ix].mean()),observed=float(y[ix].mean())))
        policies[name]=policy_stats(a,pred<0)
    for family,base in [('market_component','market_direct'),('queue_component','queue_direct'),('queue_component','market_component')]:
        valuation_differences.append(dict(model=family,baseline=base,metric='value_mse',**interval(a,val_loss[family]-val_loss[base])))
    decisions={}
    for model in ('market_component','queue_component'):
        pull=a[scorecols[model]].lt(0).to_numpy();benefit=-a.label.to_numpy()
        diag=null(a,pull,benefit,S.KEYS)
        expected=np.zeros(len(a))
        for ix in a.groupby(S.KEYS,dropna=False,sort=True).indices.values():expected[ix]=pull[ix].mean()*benefit[ix]
        comparisons={'random':interval(a,pull*benefit-expected,mean=False)};matched={}
        for base,col in [('old_queue','queue_score'),('old_market','market_score'),('guard_margin','guard_margin')]:
            mask=rank_match(a,pull,a[col].to_numpy());matched[base]=policy_stats(a,mask)
            comparisons[base]=interval(a,(pull.astype(int)-mask.astype(int))*benefit,mean=False)
        decisions[model]=dict(null=diag,comparisons=comparisons,matched=matched)
    def bd(branch,family,base,metric):
        return next(x for x in branch_differences if (x['branch'],x['family'],x['baseline'],x['metric'])==(branch,family,base,metric))
    vc=next(x for x in valuation_differences if x['model']=='queue_component' and x['baseline']=='queue_direct')
    dc=decisions['queue_component']['comparisons']['old_queue']
    predicates=dict(queue_fill_brier_better_than_baseline_both_branches=all(bd(b,'queue','baseline','brier')['delta']<0 for b in ('keep','reset')),
                    queue_fill_brier_day_intervals_below_zero_both_branches=all(bd(b,'queue','baseline','brier')['day_interval'][1]<0 for b in ('keep','reset')),
                    queue_quantity_mse_better_than_baseline_both_branches=all(bd(b,'queue','baseline','quantity_mse')['delta']<0 for b in ('keep','reset')),
                    queue_component_value_mse_better_than_direct=vc['delta']<0,
                    queue_component_selection_better_than_matched_direct=dc['delta']>0,
                    queue_component_selection_day_interval_positive=dc['day_interval'][0]>0,
                    sequential_portfolio_tested=False,live_fill_calibrated=False,new_candidate_promoted=False)
    result=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),status='PASS',days=S.TEST,n=len(a),validation=False,
                declaration_sha=S.sha(S.OUT/'declaration.json'),fit_sha=S.sha(S.OUT/'fit.json'),predictions_sha=fit['predictions_sha'],
                metrics=metrics,branch_differences=branch_differences,valuations=valuations,valuation_differences=valuation_differences,
                policies=policies,decisions=decisions,calibration=calibration,statuses=fit['statuses'],predicates=predicates,
                controls='Original source/input hashes, synthetic positive/refusal controls, original fold/prediction parity, 120 saved-model reconstructions plus old direct models, constraint/accounting identities, AUC/day-bootstrap/matched-count controls and final artifact hashes pass.',
                limits='Existing short-horizon simulated labels only. Conditional LODO guard chronology; no incoming-volume/time-of-fill forecast or live calibration. Positive/negative targets split aggregate branch net markout, not per-fill gross outcomes. Utility sums overlap and are not portfolio P&L. No new entry gate or strategy promotion.')
    S.write('result.json',result)
    lines=['# Explicit fill opportunity and branch value — first-stage diagnostic',
           f"As of {result['as_of']}. Six consumed test dates, {len(a):,} neutral actions. No untouched validation. All heads include zero-fill actions.", '',
           'Two feature sets, KEEP/RESET branches, five heads per branch, six expanding folds: 120 models. Original parameters and 300 rounds retained; fill-event heads use binary objective. Training-only state-stratified means are the baseline. Exact definitions and source/input hashes are in `../reviews/fill_value_20261112/declaration.json`.', '',
           '| Branch | Model | Actions / filled | Observed fill rate | Mean probability | Brier | AUC | Actual / predicted shares | Quantity RMSE |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in metrics:lines.append(f"| {x['branch']} | {x['family']} | {x['n']:,} / {x['filled']:,} | {x['fill_rate']:.4f} | {x['mean_probability']:.4f} | {x['brier']:.6f} | {x['auc']:.4f} | {x['actual_quantity']:.2f} / {x['predicted_quantity']:.2f} | {x['quantity_rmse']:.4f} |")
    lines += ['', 'Brier and quantity error describe simulated fill prediction, not calibrated live execution. Baseline means use only earlier training days, matched on side/order age/remaining/displayed depth and shrunk with fixed weight 50. Higher displayed depth is not a measurement of shares ahead. Missing marks and all upstream statuses remain in the machine receipt.', '',
              '| Model | Actions | KEEP-minus-RESET RMSE, c | Pulls / keeps | Predicted / realized keep advantage on kept subset, c |', '|---|---:|---:|---:|---:|']
    for x in valuations:lines.append(f"| {x['model']} | {x['n']:,} | {x['rmse_c']:.4f} | {x['pull_n']:,} / {x['keep_n']:,} | {x['predicted_keep_value_c']:.2f} / {x['actual_keep_value_c']:.2f} |")
    lines += ['', '| Pull policy | Pull count | Summed local benefit, c | Increment vs matched old queue, c [day interval] | Increment vs matched guard margin, c [day interval] |', '|---|---:|---:|---:|---:|']
    for name,z in decisions.items():
        p=policies[name];d=z['comparisons'];q=d['old_queue'];g=d['guard_margin']
        lines.append(f"| {name} | {p['pulls']:,} | {p['benefit_c']:.2f} | {q['delta']:+.2f} [{q['day_interval'][0]:+.2f}, {q['day_interval'][1]:+.2f}] | {g['delta']:+.2f} [{g['day_interval'][0]:+.2f}, {g['day_interval'][1]:+.2f}] |")
    lines += ['', 'Same-count comparisons match day, side, UTC hour, order-age bucket and inventory sign. Guard margin is only the original value-line component, not its complete rules. Intervals resample six UTC days 1,000 times. Each component policy also has a predeclared 500-draw matched-count random null. All values here are overlapping local counterfactual utility, not portfolio P&L.', '',
              'Computed predicates: `'+json.dumps(S.native(predicates),sort_keys=True)+'`', '',
              'This first stage retains the existing roughly two-second fill horizon and same-price reset after one second. It does not yet value later unfilled queue progress, forecast fill timing or raw incoming volume, or fix label/re-entry mismatch. Probability forecasts are diagnostic and constrain feasible quantity moments; value heads estimate unconditional joint gains/losses directly. There is no independent edge-times-fill-probability assumption and no extra queue bonus. No model promotion or live changes.', '',
              'Full probability/value calibration bins, paired day differences, individual statuses, baselines, saved models, predictions and numerical checks are in the canonical receipt and data directory. Read the human review for interpretation.']
    rel='predictfun/handoffs/20261112_fill_value_results.md'
    for base in (S.WORK,S.ROOT):
        with (base/rel).open('x') as f:f.write('\n'.join(lines)+'\n')
    S.progress('numerical_report','complete','All models and diagnostics complete; calibrated-prediction and matched-value comparisons saved. Human review pending; no promotion.')
    print('FILL_VALUE_REPORT_PASS',S.OUT,flush=True)


if __name__=='__main__':run()
