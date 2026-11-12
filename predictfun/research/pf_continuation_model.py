"""Coherent shared outcome forecasts for full-policy continuations and incoming flow."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from . import pf_continuation_spec as S
from .pf_model_spec import PARAMS,ROUNDS
from .pf_order_policy_inputs import MARKET,FEATURES
from .pf_joint_fill import calibrate
from .pf_fill_value import safe_features,chronology

FIELDS=['mass']
for _b in ('keep','pull'):
    for _h in S.HORIZONS:
        FIELDS += [f'{_b}_{_h}_{k}' for k in ('q','q2','same_q','gain','loss','risk_pos','risk_neg','net2')]
    FIELDS += [f'{_b}_first_{h}' for h in (250,1000,2000,10000,30000)]
FIELDS += ['flow_'+str(h) for h in S.HORIZONS]
IX={s:i for i,s in enumerate(FIELDS)}


def tensor(a):
    if a.id.duplicated().any() or not a.status.eq('eligible').all():raise ValueError('invalid training population')
    v=np.ones((len(a),len(FIELDS)))
    for i,k in enumerate(FIELDS[1:],1):
        if k.endswith('risk_pos'):v[:,i]=np.maximum(a[k.rsplit('_',1)[0]],0)
        elif k.endswith('risk_neg'):v[:,i]=np.maximum(-a[k.rsplit('_',1)[0]],0)
        else:v[:,i]=a[k]
    event=a.keep_10000_q.gt(0).astype(int).to_numpy()+2*a.pull_10000_q.gt(0).astype(int).to_numpy()
    j=np.zeros((len(a),4,len(FIELDS)));j[np.arange(len(a)),event]=v
    support(j);return event,v,j.mean(axis=0)


def support(j):
    if j.ndim!=3 or j.shape[1:]!=(4,len(FIELDS)) or not np.isfinite(j).all() or (j< -1e-8).any():raise ValueError('invalid continuation outcomes')
    mass=j[:,:,0]
    if not np.allclose(mass.sum(axis=1),1,atol=1e-9,rtol=0):raise ValueError('mass')
    for b,bit in [('keep',1),('pull',2)]:
        last=np.zeros_like(mass)
        for h in S.HORIZONS:
            col=lambda k:j[:,:,IX[f'{b}_{h}_{k}']]
            q=col('q');q2=col('q2')
            if ((q<last-1e-8)|(q*q>mass*q2+1e-7)|(col('same_q')>q+1e-8)|(col('gain')+col('loss')>100*q+1e-7)).any():raise ValueError('incoherent quantity/value')
            if ((q<=1e-12)&((col('gain')+col('loss')+col('risk_pos')+col('risk_neg'))>1e-8)).any():raise ValueError('value without fills')
            if h==10000:
                inactive=[c for c in range(4) if not c&bit]
                if (q[:,inactive]>1e-8).any():raise ValueError('no-fill event with quantity')
            last=q
        last=np.zeros_like(mass)
        for h in (250,1000,2000,10000,30000):
            p=j[:,:,IX[f'{b}_first_{h}']]
            if ((p<last-1e-9)|(p>mass+1e-9)).any():raise ValueError('first-fill curve')
            last=p
    last=np.zeros_like(mass)
    for h in S.HORIZONS:
        v=j[:,:,IX[f'flow_{h}']]
        if (v<last-1e-8).any():raise ValueError('flow horizon')
        last=v


def tables(leaves,event,v,prior):
    n,t=leaves.shape;k=int(leaves.max())+1;out=np.zeros((t,k,4,len(FIELDS)))
    for i in range(t):
        ids=leaves[:,i];counts=np.bincount(ids,minlength=k);ix=ids*4+event
        for c in range(len(FIELDS)):out[i,:,:,c]=np.bincount(ix,weights=v[:,c],minlength=k*4).reshape(k,4)
        good=counts>0;out[i,good]/=counts[good,None,None];out[i,~good]=prior
    support(out.reshape(-1,4,len(FIELDS)));return out


def predict(model,features,a,table,prior):
    leaves=np.asarray(model.predict(a[features],pred_leaf=True,num_threads=1),int)
    if leaves.ndim==1:leaves=leaves[:,None]
    out=np.zeros((len(a),4,len(FIELDS)))
    if leaves.shape[1]!=len(table):raise ValueError('tree identity')
    for i in range(len(table)):out+=table[i,leaves[:,i]]
    out=.99*out/len(table)+.01*prior;support(out);return out


def reweight(j,weights):
    weights=np.asarray(weights)
    if weights.shape!=(4,) or (weights<=0).any() or not np.isfinite(weights).all():raise ValueError('invalid calibration weights')
    out=j*weights[None,:,None];out/=out[:,:,0].sum(axis=1)[:,None,None];support(out);return out


def fit_calibration(j,event,prior):
    # The inherited calibrator only needs event masses. Supply zero ancillary
    # moments to its support checker, then reweight our complete tensors.
    proxy=np.zeros((len(j),4,9));proxy[:,:,0]=j[:,:,0]
    p=np.zeros((4,9));p[:,0]=prior[:,0]
    return calibrate(proxy,event,p)


def flatten(j):
    support(j);v=j.sum(axis=1);out={k:v[:,i] for i,k in enumerate(FIELDS) if i}
    for b in ('keep','pull'):
        for h in S.HORIZONS:
            p=f'{b}_{h}_';out[p+'value']=out[p+'gain']-out[p+'loss']-out[p+'risk_pos']+out[p+'risk_neg']
    for h in S.HORIZONS:out['score_'+str(h)]=out[f'keep_{h}_value']-out[f'pull_{h}_value']
    return out


def run():
    S.check();parts=[];statuses=[]
    for d in S.DAYS:
        receipt=json.loads((S.OUT/f'build_{d}.json').read_text());p=S.OUT/'runs'/d/'paired.parquet'
        if receipt['status']!='PASS' or S.sha(p)!=receipt['hashes']['paired.parquet']:raise ValueError('unverified builder')
        a=pd.read_parquet(p);statuses += [dict(day=d,status=k,n=int(v)) for k,v in a.status.value_counts().items()]
        parts.append(a[a.status.eq('eligible')])
    a=pd.concat(parts,ignore_index=True);tensor(a);folds=[];predictions=[]
    models=S.OUT/'models';models.mkdir(exist_ok=False)
    for d in S.TEST:
        calday=S.DAYS[S.DAYS.index(d)-1];core=a[a.day<calday];cal=a[a.day.eq(calday)];test=a[a.day.eq(d)].copy()
        chronology(core,cal);chronology(cal,test)
        ev,v,prior=tensor(core);ce,_,_=tensor(cal)
        record=dict(day=d,core_days=sorted(core.day.unique()),calibration_day=calday,core_n=len(core),calibration_n=len(cal),test_n=len(test),models={})
        outputs={'baseline_'+k:x for k,x in flatten(np.repeat(prior[None],len(test),axis=0)).items()}
        for family,features in [('market',MARKET),('queue',FEATURES)]:
            safe_features(features)
            model=lgb.train(PARAMS,lgb.Dataset(core[features],label=core.keep_10000_value-core.pull_10000_value),ROUNDS)
            p=models/f'{d}_{family}.txt';model.save_model(str(p));model=lgb.Booster(model_file=str(p))
            leaves=np.asarray(model.predict(core[features],pred_leaf=True,num_threads=1),int)
            if leaves.ndim==1:leaves=leaves[:,None]
            table=tables(leaves,ev,v,prior);q=models/f'{d}_{family}.npz';np.savez_compressed(q,table=table,prior=prior)
            fit=fit_calibration(predict(model,features,cal,table,prior),ce,prior)
            raw=predict(model,features,test,table,prior);calibrated=reweight(raw,fit['weights'])
            with np.load(q) as saved:
                rebuilt=reweight(predict(lgb.Booster(model_file=str(p)),features,test,saved['table'],saved['prior']),json.loads(json.dumps(S.native(fit)))['weights'])
            np.testing.assert_allclose(rebuilt,calibrated,atol=1e-9,rtol=0)
            for variant,j in [('raw',raw),('cal',calibrated)]:
                for k,x in flatten(j).items():outputs[f'{family}_{variant}_'+k]=x
            record['models'][family]=dict(model_sha=S.sha(p),table_sha=S.sha(q),calibrator=fit,
                checked_forecasts=2*len(test),reload_max_abs_error=float(np.abs(rebuilt-calibrated).max()))
        test=pd.concat([test.reset_index(drop=True),pd.DataFrame(outputs)],axis=1)
        old=lgb.Booster(model_file=str(S.PRIOR/'models'/f'{d}_queue.txt'))
        test['original_direct_score']=old.predict(test[FEATURES],num_threads=1)
        predictions.append(test);folds.append(record)
        S.write('fit_'+d+'.json',record);S.progress('fit_'+d,'complete',f'Coherent 2/10/30s value, quantity, fill-time and flow forecasts saved; {len(test)} test actions.')
    p=S.OUT/'predictions.parquet';pd.concat(predictions,ignore_index=True).to_parquet(p,index=False)
    S.write('fit.json',dict(folds=folds,statuses=statuses,n=sum(len(x) for x in predictions),predictions_sha=S.sha(p),fields=FIELDS))
    from .pf_continuation_report import run as report
    report()

if __name__=='__main__':run()
