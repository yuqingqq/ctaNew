"""Descriptive order populations and saved-model identity, no new candidate or selection."""
import datetime as dt
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from . import pf_order_policy_spec as S
from .pf_order_policy_inputs import FEATURES,MARKET


def scores_equal(actual,expected):
    if not np.allclose(actual,expected,rtol=0,atol=1e-10,equal_nan=False):raise ValueError('saved-model prediction identity')


def run():
    S.check();report=json.loads((S.OUT/'comparison.json').read_text());fit=json.loads((S.OUT/'fit.json').read_text())
    p=S.OUT/'predictions.parquet'
    if S.sha(p)!=fit['predictions_sha']:raise ValueError('predictions artifact changed')
    pred=pd.read_parquet(p);models=[];counts=[]
    # Positive reconstruction and known-bad falsifier at the actual named model artifact.
    for day in S.TEST:
        rows=pred[pred.day==day]
        for model,cols in [('market',MARKET),('queue',FEATURES)]:
            path=S.OUT/'models'/f'{day}_{model}.txt';m=lgb.Booster(model_file=str(path));score=m.predict(rows[cols],num_threads=1)
            scores_equal(score,rows[model+'_score'])
            bad=score.copy();bad[0]+=1
            try:scores_equal(bad,rows[model+'_score'])
            except ValueError:pass
            else:raise ValueError('known-bad prediction accepted')
            models.append(dict(day=day,model=model,n=len(rows),sha=S.sha(path),score_identity=True,known_bad_refused=True,
                chosen=int(rows[model+'_keep'].sum()),label_nonzero=int(rows.label_q.gt(0).sum())))
    for receipt in report['receipts']:
        day,mode,scenario=(receipt[k] for k in ('day','mode','scenario'))
        group='stress' if scenario=='stress' else 'build' if mode in ('reference','calm_seed') else 'eval'
        base=S.OUT/'runs'/group/day;c=json.loads((base/'checks.json').read_text());check=next(x for x in c['variants'] if x['mode']==mode)
        file=base/'generations'/f'{mode}.parquet'
        if S.sha(file)!=check['generation_sha']:raise ValueError('generation artifact identity')
        g=pd.read_parquet(file)
        for population,mask in [('all',np.ones(len(g),bool)),('seed',g.seed.eq(1)),('kept',g.kept.eq(1)),('eligible',g.status.eq('eligible'))]:
            a=g[mask];valid=a.entry_value1.notna()&a.entry_value5.notna()
            counts.append(dict(day=day,mode=mode,scenario=scenario,population=population,n=len(a),filled=int(a.entry_q.gt(0).sum()),
                shares=float(a.entry_q.sum()),known_mark_n=int(valid.sum()),missing_mark_n=int((~valid).sum()),
                value1=float(a.loc[valid,'entry_value1'].sum()/100),value5=float(a.loc[valid,'entry_value5'].sum()/100),
                settlement=float(a.entry_settle.sum()/100)))
    result=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),source_sha=S.sha(__file__),base_comparison_sha=S.sha(S.OUT/'comparison.json'),
        note='Descriptive subsets of complete generation lifetimes; kept-order values include unavoidable pre-decision fills and must not be called incremental continuation gains. No new candidate or inferential test. Model predictions independently reconstructed; altered-score falsifiers refused.',models=models,populations=counts)
    S.write(S.OUT/'mechanism.json',result);S.write(S.REVIEW/'mechanism.json',result)
    print('MECHANISM_VERIFIED',len(models),'saved model artifacts',flush=True)


if __name__=='__main__':run()
