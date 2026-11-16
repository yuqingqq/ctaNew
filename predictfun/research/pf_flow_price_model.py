"""Fit a coarse joint forecast and integrate conditional responses, without future inputs."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf_flow_price_common as C
import lightgbm as lgb
np, pd = C.np, C.I.pd


def logloss(y,p):
    return float(-np.log(np.clip(p[np.arange(len(y)),np.asarray(y,int)],1e-12,1)).mean())


def temperature(p,t):
    z=np.log(np.clip(p,1e-12,1.))/t;z-=z.max(1,keepdims=True)
    z=np.exp(z);return z/z.sum(1,keepdims=True)


def choose_temperature(p,y):
    scores=[logloss(y,temperature(p,t)) for t in C.TEMPERATURES]
    return C.TEMPERATURES[int(np.argmin(scores))],scores


def fit(X,y,features,out,objective,nclass=None):
    C.require(X.shape[1]==len(features) and len(y)==len(X),'fit dimensions')
    params={**C.V.PARAMS,'objective':objective}
    if nclass is not None:params['num_class']=nclass
    model=lgb.train(params,lgb.Dataset(X,label=y,feature_name=features),num_boost_round=200)
    model.save_model(str(out));loaded=lgb.Booster(model_file=str(out))
    check=X[:min(257,len(X))]
    C.require(np.allclose(model.predict(check,num_threads=1),loaded.predict(check,num_threads=1),rtol=0,atol=1e-10),'model reload mismatch')
    return loaded


def read(day):
    p=C.OUT/day;receipt=json.loads((p/'receipt.json').read_text())
    C.require(receipt['output_sha256']==C.sha(p/'actions.parquet'),'changed dataset '+day)
    C.require(receipt['source_manifest']==C.sha(C.REVIEW/'freeze_v1.json'),'mixed data contract')
    df=pd.read_parquet(p/'actions.parquet');df=df[df.status.isin(['eligible','eligible_estimated_timing'])].copy()
    C.require(len(df)>0,'no eligible actions '+day)
    C.require(not df.duplicated(['day','m','side','rung','generation']).any(),'duplicate action')
    return df


def empirical_fit(df):
    global_counts=np.bincount(df.joint_class.astype(int),minlength=C.NCLASS)+1.
    prior=global_counts/global_counts.sum();groups={}
    for key,g in df.groupby(['side','rung','phase']):
        count=np.bincount(g.joint_class.astype(int),minlength=C.NCLASS)
        groups[tuple(map(int,key))]=(count+30.*prior)/(len(g)+30.)
    return prior,groups


def empirical_predict(df,obj):
    prior,groups=obj
    return np.array([groups.get(tuple(map(int,k)),prior) for k in df[['side','rung','phase']].itertuples(index=False,name=None)])


def integrate(prob, conditional):
    C.require(prob.shape==conditional.shape,'scenario integration shape')
    C.require(np.isfinite(prob).all() and (prob>=0).all() and np.allclose(prob.sum(1),1.,atol=1e-7),'invalid probability')
    return np.sum(prob*conditional,axis=1)


def fit_fold(day):
    C.verify();C.require(day in C.TESTS,'undeclared fold');start=time.time()
    out=C.OUT/('fit_'+day);C.require(not out.exists(),'fit exists');out.mkdir(parents=True)
    train_days,cal_day=C.FOLDS[day]
    C.require(max(train_days)<cal_day<day,'chronology')
    train=pd.concat([read(d) for d in train_days],ignore_index=True)
    cal=read(cal_day);test=read(day);C.guard('fit loaded')
    X=train[C.FEATURES].to_numpy(np.float32);Xc=cal[C.FEATURES].to_numpy(np.float32);Xt=test[C.FEATURES].to_numpy(np.float32)
    # No target or future class can enter the scenario-probability feature list.
    C.require(not (set(C.FEATURES)&set(C.TARGETS+C.COND+['joint_class','move_cents','flow_class','move_class'])),'target feature leak')
    configs=[('joint','joint_class',15,C.FEATURES),('mechanical','joint_class',15,C.MECHANICAL),
             ('flow','flow_class',3,C.FEATURES),('move','move_class',5,C.FEATURES)]
    predictions={};cal_predictions={};artifacts={};temps={};importance={}
    for name,label,nclass,features in configs:
        x=X if features==C.FEATURES else train[features].to_numpy(np.float32)
        xc=Xc if features==C.FEATURES else cal[features].to_numpy(np.float32)
        xt=Xt if features==C.FEATURES else test[features].to_numpy(np.float32)
        path=out/(name+'.txt');b=fit(x,train[label].to_numpy(int),features,path,'multiclass',nclass)
        raw=b.predict(xc,num_threads=1);t,loss=choose_temperature(raw,cal[label].to_numpy(int))
        cal_predictions[name]=temperature(raw,t);predictions[name]=temperature(b.predict(xt,num_threads=1),t)
        temps[name]=dict(value=t,grid_logloss=loss)
        importance[name]=dict(zip(features,map(float,b.feature_importance(importance_type='gain'))))
        artifacts[name]=dict(path=str(path),sha256=C.sha(path));del b;gc.collect();C.guard(name)
        print(json.dumps(dict(fold=day,head=name,seconds=round(time.time()-start))),flush=True)
    for P in (predictions,cal_predictions):
        P['independent']=(P['flow'][:,:,None]*P['move'][:,None,:]).reshape(-1,15)
    empirical=empirical_fit(train)
    predictions['empirical']=empirical_predict(test,empirical);cal_predictions['empirical']=empirical_predict(cal,empirical)
    C.dump(out/'empirical.json',dict(prior=empirical[0],groups={','.join(map(str,k)):v for k,v in empirical[1].items()}))
    # Conditional heads see the realized scenario only while fitting. Deployment
    # enumerates every possible scenario and integrates over forecast probabilities.
    XX=np.column_stack([X,train[['flow_class','move_class']].to_numpy(np.float32)])
    response={};cal_response={}
    names=C.FEATURES+C.COND
    for target in C.TARGETS:
        obj='binary' if target in ('severe','own_fill') else 'regression'
        path=out/('conditional_'+target+'.txt')
        b=fit(XX,train[target].to_numpy(float),names,path,obj)
        for frame,base,dest in ((test,Xt,response),(cal,Xc,cal_response)):
            values=np.empty((len(frame),15))
            for k in range(15):
                inp=np.column_stack([base,np.full(len(base),k//5),np.full(len(base),k%5)]).astype(np.float32)
                v=b.predict(inp,num_threads=1)
                if target=='flow_quantity':v=np.maximum(v,0.)
                if target=='own_quantity':v=np.clip(v,0.,frame.fp_remaining.to_numpy())
                # Structural no-flow scenario implies no own fills in the same interval.
                if k//5==0:v=np.zeros(len(v))
                values[:,k]=v
            dest[target]=values
        artifacts['conditional_'+target]=dict(path=str(path),sha256=C.sha(path))
        importance['conditional_'+target]=dict(zip(names,map(float,b.feature_importance(importance_type='gain'))))
        del b;gc.collect();C.guard(target)
        print(json.dumps(dict(fold=day,head=target,seconds=round(time.time()-start))),flush=True)
    path=out/'direct_value.txt';b=fit(X,train.value_cents.to_numpy(float),C.FEATURES,path,'regression')
    direct=b.predict(Xt,num_threads=1);direct_cal=b.predict(Xc,num_threads=1)
    artifacts['direct_value']=dict(path=str(path),sha256=C.sha(path));del b
    for frame,P,R,direct_pred,filename in ((test,predictions,response,direct,'test'),(cal,cal_predictions,cal_response,direct_cal,'calibration')):
        cols=['day','m','side','rung','generation','decision_ms','phase','status','fp_remaining','flow_class','move_class','joint_class','move_cents']+C.TARGETS
        result=frame[cols].copy();result['direct_value']=direct_pred
        for name in ['joint','mechanical','independent','empirical']:
            for k in range(15):result[f'{name}_p{k}']=P[name][:,k]
        for name in ['joint','independent','empirical']:
            for target,conditional in R.items():result[f'{name}_{target}']=integrate(P[name],conditional)
        result.to_parquet(out/(filename+'.parquet'),index=False)
    C.dump(out/'receipt.json',dict(asof=C.stamp(),test_day=day,training_days=train_days,calibration_day=cal_day,
        n_train=len(train),n_calibration=len(cal),n_test=len(test),temperature=temps,artifacts=artifacts,
        importance=importance,features=C.FEATURES,conditional_features=names,source_manifest=C.sha(C.REVIEW/'freeze_v1.json'),
        data_receipts={d:C.sha(C.OUT/d/'receipt.json') for d in (*train_days,cal_day,day)},
        outputs={n:C.sha(out/n) for n in ['test.parquet','calibration.parquet','empirical.json']},seconds=time.time()-start))
    print(json.dumps(dict(done='fit_'+day,seconds=round(time.time()-start))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('day');a=p.parse_args();fit_fold(a.day)
