"""Chronological settlement-probability feasibility study on consumed days.

Forecast scores are descriptive, paired by UTC day. No claim of untouched
validation, independence of anchors, realized trading return, or deployment.
"""
import datetime as dt
import json
import numpy as np
import pandas as pd
from scipy.special import logit, ndtri
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from .pf_sizing_study import DAYS, DATA, DEST, OLD, HERE, sha, write

MODELS=['market_mid','brownian_basis','market_calibrated_logistic','market_plus_physical_logistic']

def validate_labels(r):
    if r.duplicated(['mid','t_rel']).any():raise ValueError('duplicate decision identity')
    if not r.y.isin([0.,1.]).all():raise ValueError('nonbinary label')
    wanted=r.outcome.map({'Up':1.,'Down':0.})
    if not np.array_equal(wanted,r.y):raise ValueError('label/outcome identity mismatch')

def causal_train(r,day):
    start=pd.Timestamp(day,tz='UTC').value
    a=r[(r.day<day)&(r.outcome_seen_ns<start)].copy()
    if len(a)==0 or a.day.max()>=day or not (a.outcome_seen_ns<start).all():
        raise ValueError('empty or future training labels')
    return a

def matrix(r,physical):
    p=np.clip(r.pf_mid.to_numpy()/100,1e-5,1-1e-5)
    l=logit(p);t=r.t_rel.to_numpy()/300
    parts=[l,t,l*t]
    if physical:
        pb=np.clip(r.p_bn.to_numpy()/100,1e-5,1-1e-5)
        z=ndtri(pb)
        parts += [z,z*t,r.bn_r1000.to_numpy(),r.bn_r3000.to_numpy(),
                  r.bn_imb.to_numpy(),r.pf_imb1.to_numpy(),r.pf_spread.to_numpy(),
                  np.log(np.maximum(r.sigma.to_numpy(),1e-9)),r.basis_bps.to_numpy()]
    return np.column_stack(parts)

def weights(r):
    return 1/r.groupby('mid').mid.transform('size').to_numpy()

def fit_predict(train,test,physical):
    if train.day.max()>=test.day.min():raise ValueError('train/test time leakage')
    model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),
                        LogisticRegression(C=1.,max_iter=2000,solver='lbfgs',random_state=20261108))
    model.fit(matrix(train,physical),train.y.to_numpy(),logisticregression__sample_weight=weights(train))
    return model.predict_proba(matrix(test,physical))[:,1]

def score(y,p,w):
    p=np.clip(np.asarray(p),1e-6,1-1e-6);y=np.asarray(y)
    return dict(brier=float(np.average((p-y)**2,weights=w)),
      log_loss=float(np.average(-(y*np.log(p)+(1-y)*np.log1p(-p)),weights=w)),
      accuracy=float(np.average((p>=.5)==y,weights=w)),
      mean_probability=float(np.average(p,weights=w)),realized_up=float(np.average(y,weights=w)))

def controls():
    y=np.tile([0.,1.],100);p=np.full(200,.5)
    oracle=np.where(y==1,.99,.01);bad=1-oracle
    assert score(y,oracle,np.ones(200))['brier']<score(y,p,np.ones(200))['brier']
    assert score(y,bad,np.ones(200))['brier']>score(y,p,np.ones(200))['brier']
    r=pd.DataFrame(dict(mid=[1,2],t_rel=[60,60],outcome=['Up','Down'],y=[0.,1.]))
    try:validate_labels(r)
    except ValueError:pass
    else:raise AssertionError('shuffled labels accepted')
    r=pd.DataFrame(dict(day=['20261103'],outcome_seen_ns=[0]))
    try:causal_train(r,'20261102')
    except ValueError:pass
    else:raise AssertionError('future training accepted')
    return dict(oracle_detected=True,anti_oracle_detected=True,shuffled_identity_refused=True,future_training_refused=True)

def paired_inference(deltas):
    a=np.asarray(deltas,float)
    if a.shape[0]<5:raise ValueError('insufficient complete day clusters')
    rng=np.random.default_rng(20261108)
    signs=rng.choice([-1,1],size=(2000,len(a)))
    null=(signs@a)/len(a);obs=a.mean(axis=0)
    sd=a.std(axis=0,ddof=1)/np.sqrt(len(a));sd=np.maximum(sd,1e-12)
    maxt=np.max(np.abs(null/sd),axis=1)
    boot=np.array([a[rng.integers(0,len(a),size=len(a))].mean(axis=0) for _ in range(2000)])
    return [dict(mean=float(v),interval_95=np.quantile(boot[:,i],[.025,.975]).tolist(),
      p_two_sided=float((1+np.sum(np.abs(null[:,i])>=abs(v)))/2001),
      p_max_t_family=float((1+np.sum(maxt>=abs(v/sd[i])))/2001)) for i,v in enumerate(obs)]

def main():
    checked=controls();parts=[];statuses={};inputs={}
    for day in DAYS:
        p=DATA/'step5'/day/'settlement_actions.parquet';r=pd.read_parquet(p);inputs[str(p)]=sha(p)
        r['y']=r.outcome.map({'Up':1.,'Down':0.})
        r['status']='eligible'
        r.loc[r.pf_mid.isna()|~r.pf_mid.between(0,100),'status']='missing_market_mid'
        r.loc[~r.start_known,'status']='start_price_not_known'
        r.loc[r.y.isna(),'status']='unresolved_or_nonbinary_outcome'
        statuses[day]=dict(rows=len(r),markets=int(r.mid.nunique()),by_status=r.status.value_counts().to_dict(),
          missing_physical=int(r.p_bn.isna().sum()),missing_label_known_time=int(r.outcome_seen_ns.isna().sum()),
          equal_endpoint_prices=int(r.loc[r.start_price==r.end_price,'mid'].nunique()))
        r=r[r.status=='eligible'].copy();validate_labels(r);parts.append(r)
    allrows=pd.concat(parts,ignore_index=True);preds=[];folds=[]
    for day in DAYS[3:]:
        tr=causal_train(allrows,day);te=allrows[allrows.day==day].copy()
        if tr.y.nunique()!=2:raise ValueError('training missing a label class')
        te['market_mid']=np.clip(te.pf_mid/100,1e-6,1-1e-6)
        te['brownian_basis']=np.clip((te.p_bn/100).fillna(te.market_mid),1e-6,1-1e-6)
        for physical,name in [(False,MODELS[2]),(True,MODELS[3])]:te[name]=fit_predict(tr,te,physical)
        folds.append(dict(day=day,train_days=sorted(tr.day.unique()),train_rows=len(tr),train_markets=int(tr.mid.nunique()),
          test_rows=len(te),test_markets=int(te.mid.nunique()),max_training_label_seen_ns=int(tr.outcome_seen_ns.max()),
          labels_unavailable_excluded=int(((allrows.day<day)&~(allrows.outcome_seen_ns<pd.Timestamp(day,tz='UTC').value)).sum())))
        preds.append(te[['day','mid','tau','t_rel','y','pf_mid','pf_spread','fee_bps']+MODELS])
    pr=pd.concat(preds,ignore_index=True);validate=pr.groupby(['day','mid','t_rel']).size().max()==1
    if not validate:raise ValueError('duplicate forecast action')
    byday={d:{m:score(g.y,g[m],weights(g)) for m in MODELS} for d,g in pr.groupby('day')}
    overall={m:{metric:float(np.mean([x[m][metric] for x in byday.values()])) for metric in ['brier','log_loss','accuracy','mean_probability','realized_up']} for m in MODELS}
    family=[(m,b,metric) for m in MODELS[2:] for b in ['market_mid','market_calibrated_logistic'] if m!=b for metric in ['brier','log_loss']]
    delta=np.array([[v[m][metric]-v[b][metric] for m,b,metric in family] for v in byday.values()])
    inf={f'{m} minus {b} {met}':x for (m,b,met),x in zip(family,paired_inference(delta))}
    calibration={}
    for m in MODELS:
        bins=np.minimum((pr[m]*10).astype(int),9);rows=[]
        for k in range(10):
            g=pr[bins==k]
            if len(g):rows.append(dict(bin=k,n_actions=len(g),n_markets=int(g.mid.nunique()),mean_p=float(g[m].mean()),up_rate=float(g.y.mean())))
        calibration[m]=rows
    out=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),controls=checked,statuses=statuses,
      folds=folds,n_test_days=len(byday),n_test_markets=int(pr.mid.nunique()),n_test_actions=len(pr),
      overall_day_balanced=overall,by_day=byday,paired_inference=inf,calibration=calibration,
      by_anchor={str(a):{m:score(g.y,g[m],weights(g)) for m in MODELS} for a,g in pr.groupby('t_rel')},
      source_sha=inputs,code_sha=sha(__file__),forward_data_read=False,
      interpretation='chronological feasibility on previously consumed days; scores are not trading PnL',
      conditional_payout='binary venue outcome label; equal endpoint prices do not override recorded venue outcome')
    pr.to_parquet(DEST/'settlement_predictions.parquet',index=False)
    write(DEST/'settlement_report.json',out)
    print(json.dumps({k:out[k] for k in ['n_test_days','n_test_markets','n_test_actions','overall_day_balanced','paired_inference']},indent=2))

if __name__=='__main__':main()
