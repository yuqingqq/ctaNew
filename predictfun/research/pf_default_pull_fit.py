"""Expanding chronology, all neutral actions, pinned models and artifact replay."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from . import pf_default_pull_spec as S
from .pf_order_policy_inputs import FEATURES,MARKET
from .pf_own_order_fit_v2 import fit
from .pf_default_pull_verify import actions

def run():
    S.check();parts=[];statuses=[]
    for day in S.DAYS:
        out=S.folder(day,'build');c=json.loads((out/'checks.json').read_text())
        if c['status']!='PASS' or not c['reference_exact']:raise ValueError('unverified neutral path')
        v=next(x for x in c['variants'] if x['mode']=='default');p=out/'generations/default.parquet'
        if S.sha(p)!=v['generation_sha']:raise ValueError('neutral artifact changed')
        a=pd.read_parquet(p);actions(a);statuses.extend(dict(day=day,status=k,n=int(n)) for k,n in a.status.value_counts().items())
        parts.append(a[a.status.eq('eligible')&a.label.notna()])
    a=pd.concat(parts,ignore_index=True)
    if a.id.duplicated().any():raise ValueError('duplicate action population')
    models=S.OUT/'models';models.mkdir(exist_ok=False);pred=[];folds=[]
    for day in S.TEST:
        train=a[a.day<day];test=a[a.day==day].copy();importance={};hashes={}
        if len(train)<500 or test.empty:raise ValueError('insufficient action population')
        for which,cols in [('market',MARKET),('queue',FEATURES)]:
            m=fit(train,day,cols);p=models/f'{day}_{which}.txt';m.save_model(str(p));hashes[which]=S.sha(p)
            score=m.predict(test[cols],num_threads=1);loaded=lgb.Booster(model_file=str(p))
            actual=loaded.predict(test[cols],num_threads=1)
            np.testing.assert_allclose(actual,score,atol=1e-12,rtol=0)
            # Instrument positive/falsifier: a one-unit corrupted score must fail.
            corrupt=score.copy();corrupt[0]+=1
            if np.allclose(actual,corrupt,atol=1e-12,rtol=0):raise ValueError('score falsifier failed')
            test[which+'_score']=actual;importance[which]=dict(zip(cols,map(float,m.feature_importance('gain'))))
        pred.append(test);folds.append(dict(day=day,train_days=sorted(train.day.unique()),train_n=len(train),test_n=len(test),
            zero_keep_and_reset_train=int((train.label_q.eq(0)&train.reset_q.eq(0)).sum()),
            zero_keep_and_reset_test=int((test.label_q.eq(0)&test.reset_q.eq(0)).sum()),importance=importance,models_sha=hashes))
        print('FIT',day,len(train),len(test),flush=True)
    p=S.OUT/'predictions.parquet';pd.concat(pred,ignore_index=True).to_parquet(p,index=False)
    S.write(S.OUT/'fit.json',dict(folds=folds,statuses=statuses,eligible_actions=len(a),predictions_sha=S.sha(p)))

if __name__=='__main__':run()
