"""Superseding pre-fit declaration: deduplicate identical actions across neutral paths."""
import json
import datetime as dt
import subprocess
import numpy as np
import pandas as pd
import lightgbm as lgb
from . import pf_order_policy_spec as S
from .pf_order_policy_inputs import MARKET,FEATURES
from .pf_model_spec import PARAMS,ROUNDS


def unique_actions(rows):
    keys=list(dict.fromkeys(['day','mid','t_send','t_act','decision','initial_size']+FEATURES))
    a=rows.sort_values('mode',ascending=False,kind='stable')  # canonical reference before calm_seed
    grouped=a.groupby(keys,dropna=False,sort=False).label.agg(['min','max'])
    if ((grouped['max']-grouped['min']).abs()>1e-7).any():raise ValueError('same action has conflicting labels')
    return a.drop_duplicates(keys,keep='first')


def declaration():
    S.check()
    if (S.OUT/'models').exists():raise ValueError('fit already started')
    source={str(p.relative_to(S.HERE.parents[1])):S.sha(p) for p in [S.HERE/'pf_own_order_fit_v2.py',S.HERE/'run_own_order_v2.sh',S.HERE/'tests/test_order_fit_v2.py']}
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),version=2,supersedes='training population of original declaration; no original model was fit',
        original_declaration_sha=S.sha(S.OUT/'declaration.json'),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S.HERE,text=True).strip(),source_sha=source,
        method='Train on unique action identity (day/window/send/activation/first-cancel/initial-size plus all causal observed features), retaining reference when paths agree. Duplicate keys with conflicting labels refuse the fit. Path-specific neutral evaluation remains separate. No threshold, feature, model parameter, policy, label, null or test date changes.',
        validation='Positive duplicate control and known-bad conflicting-label refusal. Full18 original controls retained. Declaration before any model fit or policy evaluation.')
    S.write(S.OUT/'training_v2.json',d);S.write(S.REVIEW/'training_v2.json',d)


def fit(rows,day,columns):
    if rows.empty or rows.day.max()>=day:raise ValueError('empty or future training split')
    if not np.isfinite(rows.label).all():raise ValueError('invalid training labels')
    return lgb.train(PARAMS,lgb.Dataset(rows[columns],label=rows.label,params=PARAMS),ROUNDS)


def run():
    S.check();parts=[];d=json.loads((S.OUT/'training_v2.json').read_text())
    for p,h in d['source_sha'].items():
        if S.sha(S.HERE.parents[1]/p)!=h:raise ValueError('superseding fit source changed')
    for day in S.DAYS:
        folder=S.OUT/'runs/build'/day;checks=json.loads((folder/'checks.json').read_text())
        if checks['status']!='PASS' or not checks['reference_exact']:raise ValueError('unverified neutral reference')
        for mode in ('reference','calm_seed'):
            p=folder/'generations'/f'{mode}.parquet'
            receipt=next(r for r in checks['variants'] if r['mode']==mode)
            if S.sha(p)!=receipt['generation_sha']:raise ValueError('generation file changed')
            a=pd.read_parquet(p);parts.append(a)
    a=pd.concat(parts,ignore_index=True);valid=a[(a.status=='eligible')&a.label.notna()]
    if a.id.duplicated().any():raise ValueError('duplicate generation')
    models=S.OUT/'models';models.mkdir(exist_ok=False);pred=[];folds=[]
    for day in S.TEST:
        pooled=valid[valid.day<day];train=unique_actions(pooled);test=valid[valid.day==day].copy()
        importance={}
        for model,cols in [('market',MARKET),('queue',FEATURES)]:
            m=fit(train,day,cols);m.save_model(str(models/f'{day}_{model}.txt'))
            test[model+'_score']=m.predict(test[cols],num_threads=1)
            test[model+'_keep']=test[model+'_score']>S.CONT_MARGIN_C*test.remaining
            importance[model]={k:float(v) for k,v in zip(cols,m.feature_importance('gain'))}
        pred.append(test);folds.append(dict(day=day,train_days=sorted(train.day.unique()),train_n=len(train),pooled_train_n=len(pooled),duplicate_actions_removed=len(pooled)-len(train),test_n=len(test),
            no_incremental_fill_train=int(train.label_q.eq(0).sum()),no_incremental_fill_test=int(test.label_q.eq(0).sum()),importance=importance))
        print('FOLD',day,'train',len(train),'test',len(test),flush=True)
    p=pd.concat(pred,ignore_index=True);p.to_parquet(S.OUT/'predictions.parquet',index=False)
    S.write(S.OUT/'fit.json',dict(training_version=2,superseding_declaration=d,superseding_declaration_sha=S.sha(S.OUT/'training_v2.json'),folds=folds,statuses=a.groupby(['day','mode']).status.value_counts().rename('n').reset_index().to_dict('records'),
                               predictions_sha=S.sha(S.OUT/'predictions.parquet'),generation_rows=len(a),eligible_rows=len(valid)))


if __name__=='__main__':
    import sys
    declaration() if len(sys.argv)>1 and sys.argv[1]=='declare' else run()
