"""One explicitly exploratory extension after the logistic feasibility result.

Fixed pinned LightGBM parameters/rounds, binary objective for this new target;
learn a log-odds residual to the market instead of replacing its prior.
No tuning, new validation claim, or live strategy integration.
"""
import datetime as dt
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import logit, expit
from .pf_sizing_study import DAYS,DATA,DEST,HERE,sha,write
from .pf_settlement_study import causal_train,weights,score,paired_inference
from .pf_model_spec import PARAMS,ROUNDS,SETS

FEATURES=list(dict.fromkeys(SETS['ALL']+['p_bn']))

def fit_predict(tr,te):
    if tr.day.max()>=te.day.min():raise ValueError('time leakage')
    init=logit(np.clip(tr.pf_mid.to_numpy()/100,1e-5,1-1e-5))
    ds=lgb.Dataset(tr[FEATURES].to_numpy(dtype=np.float32),label=tr.y.to_numpy(),
                   weight=weights(tr),init_score=init,feature_name=FEATURES)
    params=dict(PARAMS,objective='binary',boost_from_average=False)
    m=lgb.train(params,ds,num_boost_round=ROUNDS)
    residual=m.predict(te[FEATURES].to_numpy(dtype=np.float32),raw_score=True,num_threads=1)
    p=expit(logit(np.clip(te.pf_mid.to_numpy()/100,1e-5,1-1e-5))+residual)
    return p,m

def controls():
    # With no splitting feature, a perfectly calibrated prior must be retained.
    x=np.zeros((1000,1));y=np.tile([1.,1.,1.,1.,0.],200)
    ds=lgb.Dataset(x,label=y,init_score=np.full(1000,logit(.8)))
    m=lgb.train(dict(PARAMS,objective='binary',boost_from_average=False),ds,num_boost_round=ROUNDS)
    raw=m.predict(x,raw_score=True,num_threads=1)
    if not np.allclose(expit(logit(.8)+raw),.8,atol=1e-6):raise AssertionError('lost market prior')
    if np.allclose(expit(raw),.8,atol=1e-6):raise AssertionError('offset-omission control failed')
    return dict(market_prior_retained=True,omitted_offset_detected=True)

def main():
    checked=controls();base=pd.read_parquet(DEST/'settlement_predictions.parquet')
    parts=[]
    for d in DAYS:
        r=pd.read_parquet(DATA/'step5'/d/'settlement_actions.parquet');r['y']=r.outcome.map({'Up':1.,'Down':0.})
        r=r[r.y.notna()&r.start_known&r.pf_mid.between(0,100)].copy();parts.append(r)
    allrows=pd.concat(parts,ignore_index=True);preds=[];modelhash={}
    for d in DAYS[3:]:
        tr=causal_train(allrows,d);te=allrows[allrows.day==d].copy();p,m=fit_predict(tr,te)
        te['market_residual_lgbm']=p;preds.append(te[['day','mid','tau','t_rel','market_residual_lgbm']])
        path=DEST/'probability_models'/f'{d}.txt';path.parent.mkdir(exist_ok=True)
        if path.exists():raise ValueError('refuse overwrite model')
        m.save_model(str(path));modelhash[d]=sha(path)
    pred=pd.concat(preds);out=base.merge(pred,on=['day','mid','tau','t_rel'],validate='one_to_one')
    if len(out)!=len(base):raise ValueError('forecast population changed')
    models=['market_mid','brownian_basis','market_calibrated_logistic','market_plus_physical_logistic','market_residual_lgbm']
    byday={d:{m:score(g.y,g[m],weights(g)) for m in models} for d,g in out.groupby('day')}
    family=[(m,b,metric) for m in models[2:] for b in ['market_mid','market_calibrated_logistic'] if m!=b for metric in ['brier','log_loss']]
    delta=np.array([[x[m][metric]-x[b][metric] for m,b,metric in family] for x in byday.values()])
    result=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),exploratory_extension_after_initial_result=True,
      n_days=6,n_markets=int(out.mid.nunique()),n_actions=len(out),models_in_family=5,
      overall={m:{k:float(np.mean([v[m][k] for v in byday.values()])) for k in ['brier','log_loss','accuracy']} for m in models},
      by_day=byday,paired_inference={f'{m} minus {b} {metric}':x for (m,b,metric),x in zip(family,paired_inference(delta))},
      controls=checked,model_sha=modelhash,code_sha=sha(__file__),parameters=dict(PARAMS,objective='binary',boost_from_average=False),
      rounds=ROUNDS,forward_data_read=False,interpretation='consumed-day feasibility only; fitted model includes market logit offset at inference')
    out.to_parquet(DEST/'settlement_predictions_v2.parquet',index=False)
    write(DEST/'settlement_nonlinear_report.json',result);print(result['overall'])

if __name__=='__main__':main()
