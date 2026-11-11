"""Chronological continuation value models; no labels selected by eventual fill."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from . import pf_order_policy_spec as S
from .pf_order_policy_inputs import MARKET,FEATURES
from .pf_model_spec import PARAMS,ROUNDS


def fit(rows,day,columns):
    if rows.empty or rows.day.max()>=day:raise ValueError('empty or future training split')
    if not np.isfinite(rows.label).all():raise ValueError('invalid training labels')
    return lgb.train(PARAMS,lgb.Dataset(rows[columns],label=rows.label,params=PARAMS),ROUNDS)


def run():
    S.check();parts=[]
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
        train=valid[valid.day<day];test=valid[valid.day==day].copy()
        importance={}
        for model,cols in [('market',MARKET),('queue',FEATURES)]:
            m=fit(train,day,cols);m.save_model(str(models/f'{day}_{model}.txt'))
            test[model+'_score']=m.predict(test[cols],num_threads=1)
            test[model+'_keep']=test[model+'_score']>S.CONT_MARGIN_C*test.remaining
            importance[model]={k:float(v) for k,v in zip(cols,m.feature_importance('gain'))}
        pred.append(test);folds.append(dict(day=day,train_days=sorted(train.day.unique()),train_n=len(train),test_n=len(test),
            no_incremental_fill_train=int(train.label_q.eq(0).sum()),no_incremental_fill_test=int(test.label_q.eq(0).sum()),importance=importance))
        print('FOLD',day,'train',len(train),'test',len(test),flush=True)
    p=pd.concat(pred,ignore_index=True);p.to_parquet(S.OUT/'predictions.parquet',index=False)
    S.write(S.OUT/'fit.json',dict(folds=folds,statuses=a.groupby(['day','mode']).status.value_counts().rename('n').reset_index().to_dict('records'),
                               predictions_sha=S.sha(S.OUT/'predictions.parquet'),generation_rows=len(a),eligible_rows=len(valid)))


if __name__=='__main__':run()
