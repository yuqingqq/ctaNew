"""Explicit fill and branch-value forecasts on all neutral order actions."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from . import pf_fill_value_spec as S
from .pf_model_spec import PARAMS, ROUNDS
from .pf_order_policy_inputs import MARKET, FEATURES
from .pf_default_pull_verify import actions

BASE_KEYS=['side','age_group','remaining_group','depth_group']


def safe_features(cols):
    if list(cols) not in (MARKET,FEATURES):raise ValueError('undeclared or future feature')


def chronology(train,test):
    if train.empty or test.empty or train.day.max()>=test.day.min():
        raise ValueError('empty or future training population')


def validate(a):
    if a.id.duplicated().any() or not a.status.eq('eligible').all():raise ValueError('action identity')
    cols=['remaining','inventory','side','label','label_q','reset_q','keep_value','reset_value',
          'label_value1','label_value5','reset_value1','reset_value5']
    if not np.isfinite(a[cols].to_numpy()).all() or not a.remaining.gt(0).all():raise ValueError('invalid target population')
    if not a.side.isin([0.,1.]).all():raise ValueError('invalid side')
    for branch,qty in [('keep','label_q'),('reset','reset_q')]:
        if ((a[qty]<-S.EPS)|(a[qty]>a.remaining+1e-7)).any():raise ValueError('quantity outside causal capacity')
        prefix='label' if branch=='keep' else 'reset'
        mark=.5*(a[prefix+'_value1']+a[prefix+'_value5'])
        dq=np.where(a.side.eq(0),1.,-1.)*a[qty]
        value=mark-.01*(2*a.inventory*dq+dq*dq)
        np.testing.assert_allclose(value,a[branch+'_value'],atol=1e-7,rtol=0)
        if (np.abs(mark[a[qty].eq(0)])>S.EPS).any():raise ValueError('unfilled markout')
    np.testing.assert_allclose(a.label,a.keep_value-a.reset_value,atol=1e-7,rtol=0)


def targets(a,branch):
    prefix='label' if branch=='keep' else 'reset'
    q=a[prefix+'_q'].to_numpy();fraction=np.clip(q/a.remaining.to_numpy(),0,1)
    mark=.5*(a[prefix+'_value1']+a[prefix+'_value5']).to_numpy()
    return dict(fill=(q>0).astype(float),fraction=fraction,fraction2=fraction*fraction,
                gain=np.maximum(mark,0),loss=np.maximum(-mark,0))


def constrain(raw):
    z={k:np.asarray(v,float) for k,v in raw.items()}
    if not all(np.isfinite(v).all() for v in z.values()):raise ValueError('nonfinite forecast')
    p=np.clip(z['fill'],1e-6,1-1e-6)
    f=np.clip(z['fraction'],0,p)
    f2=np.clip(z['fraction2'],f*f/p,f)
    return dict(fill=p,fraction=f,fraction2=f2,gain=np.maximum(z['gain'],0),loss=np.maximum(z['loss'],0))


def values(a,z):
    rem=a.remaining.to_numpy();inv=a.inventory.to_numpy();sg=np.where(a.side.eq(0),1.,-1.)
    q=rem*z['fraction'];q2=rem*rem*z['fraction2']
    risk=.01*(2*inv*sg*q+q2)
    return dict(quantity=q,quantity2=q2,mark=z['gain']-z['loss'],risk=risk,value=z['gain']-z['loss']-risk)


def groups(a):
    z=a[['side']].copy()
    z['age_group']=pd.cut(a.order_age_ms,[0,100,250,500,1000,2000,5000,np.inf],include_lowest=True,labels=False)
    z['remaining_group']=pd.cut(a.remaining,[0,1,3,5,np.inf],include_lowest=True,labels=False)
    z['depth_group']=pd.cut(a.queue_now,[-1,0,10,50,200,1000,np.inf],include_lowest=True,labels=False)
    if z.isna().any().any():raise ValueError('baseline state unavailable')
    return z


def baseline(train,test,y):
    a=groups(train);b=groups(test);out={};summary={}
    for name,target in y.items():
        a['target']=np.asarray(target,float)
        g=a.groupby(BASE_KEYS,dropna=False).target.agg(['sum','count']).reset_index()
        t=b.merge(g,on=BASE_KEYS,how='left',validate='many_to_one')
        mean=float(np.mean(target));out[name]=(t['sum'].fillna(0).to_numpy()+50*mean)/(t['count'].fillna(0).to_numpy()+50)
        summary[name]=dict(mean=mean,groups=g.to_dict('records'))
    return out,summary


def probability_loss(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not np.isin(y,[0,1]).all():
        raise ValueError('invalid probability')
    return (y-p)**2


def controls():
    y=np.array([0.,1.,0.,1.])
    if probability_loss(y,y).sum()!=0 or probability_loss(y,1-y).mean()!=1:raise ValueError('probability instrument')
    try:probability_loss(y,[0,2,0,1])
    except ValueError:pass
    else:raise ValueError('bad probability accepted')
    a=pd.DataFrame(dict(id=['a','b','c','d'],day=['20261023']*4,status=['eligible']*4,
        side=[0.,0.,1.,1.],inventory=[0.,1.,0.,-1.],remaining=[5.]*4,
        label_q=[0.,2.,5.,1.],reset_q=[1.,0.,2.,0.],
        label_value1=[0.,4.,-5.,2.],label_value5=[0.,4.,-5.,2.],
        reset_value1=[1.,0.,-2.,0.],reset_value5=[1.,0.,-2.,0.]))
    for branch in ('keep','reset'):
        z=targets(a,branch);v=values(a,z);a[branch+'_value']=v['value']
        if not np.allclose(v['quantity'],a['label_q' if branch=='keep' else 'reset_q']):raise ValueError('quantity instrument')
        if not np.allclose(v['mark'],a['label_value1' if branch=='keep' else 'reset_value1']):raise ValueError('mark instrument')
    a['label']=a.keep_value-a.reset_value;validate(a)
    for bad in (pd.concat([a,a.iloc[[0]]]),a.assign(label=999.),a.assign(label_q=6.),a.assign(remaining=np.nan)):
        try:validate(bad)
        except (ValueError,AssertionError):pass
        else:raise ValueError('known-bad action accepted')
    chronology(a,a.assign(day='20261024'));safe_features(MARKET);safe_features(FEATURES)
    for fn in (lambda:chronology(a,a),lambda:safe_features(FEATURES+['label_q'])):
        try:fn()
        except ValueError:pass
        else:raise ValueError('leakage refusal failed')
    bad=dict(fill=np.array([.2]),fraction=np.array([.8]),fraction2=np.array([-.1]),gain=np.array([-2.]),loss=np.array([3.]))
    c=constrain(bad)
    if c['fraction'][0]>.2 or not np.isclose(c['fraction2'][0],.2) or c['gain'][0]!=0:raise ValueError('moment support')
    b=pd.DataFrame(dict(side=[0.,0.],order_age_ms=[110.,110.],remaining=[5.,5.],queue_now=[10.,10.]))
    v,_=baseline(b,b,dict(fill=np.array([0.,1.])))
    np.testing.assert_allclose(v['fill'],[.5,.5],atol=1e-12)
    return dict(status='PASS',probability='perfect and reversed verified; invalid refused',
                actions='duplicate, corrupt value, overfill and missing capacity refused',
                features='future chronology and outcome feature refused',
                accounting='quantity, mark and inventory moments reconstructed; support and baseline controls passed')


def run():
    S.check();S.write('controls.json',controls())
    S.progress('controls','complete','Probability, quantity, value, chronology, feature and matched-baseline positive/refusal controls passed.')
    parts=[];statuses=[]
    for day in S.DAYS:
        a=pd.read_parquet(S.PRIOR/'runs/build'/day/'generations/default.parquet');actions(a)
        statuses += [dict(day=day,status=k,n=int(v)) for k,v in a.status.value_counts().items()]
        a=a[a.status.eq('eligible')&a.label.notna()].copy();validate(a);parts.append(a)
    a=pd.concat(parts,ignore_index=True);validate(a)
    old=pd.read_parquet(S.PRIOR/'predictions.parquet')
    oldfit=json.loads((S.PRIOR/'fit.json').read_text())
    if len(a)!=oldfit['eligible_actions']:raise ValueError('eligible cohort changed')
    modeldir=S.OUT/'models';modeldir.mkdir(exist_ok=False)
    preddir=S.OUT/'fold_predictions';preddir.mkdir(exist_ok=False)
    folds=[]
    for oldfold in oldfit['folds']:
        day=oldfold['day'];train=a[a.day<day];test=a[a.day.eq(day)].copy();chronology(train,test)
        if len(train)!=oldfold['train_n'] or len(test)!=oldfold['test_n']:raise ValueError('fold population changed')
        identity=test.id.tolist()
        test=test.merge(old[['id','market_score','queue_score']],on='id',validate='one_to_one',how='left')
        if test.id.tolist()!=identity or not np.isfinite(test[['market_score','queue_score']]).all().all():raise ValueError('old prediction join')
        fold=dict(day=day,train_n=len(train),test_n=len(test),train_days=sorted(train.day.unique()),
                  models_sha={},constraints={},baseline={},action_ids_sha=S.hashlib.sha256('\n'.join(identity).encode()).hexdigest())
        S.progress('fit_'+day,'running',f'Training 20 fixed-parameter heads on {len(train)} earlier-day actions; test actions {len(test)}.')
        for family,cols in [('market',MARKET),('queue',FEATURES)]:
            safe_features(cols)
            oldm=lgb.Booster(model_file=str(S.PRIOR/'models'/f'{day}_{family}.txt'))
            np.testing.assert_allclose(oldm.predict(test[cols],num_threads=1),test[family+'_score'],atol=1e-12,rtol=0)
            for branch in ('keep','reset'):
                y=targets(train,branch);raw={}
                if family=='market':
                    base,summary=baseline(train,test,y);fold['baseline'][branch]=summary
                    put(test,'baseline_'+branch,base,fold)
                for head in S.HEADS:
                    params=dict(PARAMS)
                    if head=='fill':params['objective']='binary'
                    m=lgb.train(params,lgb.Dataset(train[cols],label=y[head]),ROUNDS)
                    name=f'{day}_{family}_{branch}_{head}.txt';p=modeldir/name;m.save_model(str(p))
                    pred=m.predict(test[cols],num_threads=1)
                    loaded=lgb.Booster(model_file=str(p));rebuilt=loaded.predict(test[cols],num_threads=1)
                    np.testing.assert_allclose(pred,rebuilt,atol=1e-12,rtol=0)
                    corrupt=pred.copy();corrupt[0]+=1
                    if np.allclose(rebuilt,corrupt,atol=1e-12,rtol=0):raise ValueError('corrupted score accepted')
                    raw[head]=rebuilt;fold['models_sha'][name]=S.sha(p)
                put(test,family+'_'+branch,raw,fold)
                print('HEADS_DONE',day,family,branch,flush=True)
        for family in ('baseline','market','queue'):
            test[family+'_component_score']=test[family+'_keep_value']-test[family+'_reset_value']
        p=preddir/(day+'.parquet');test.to_parquet(p,index=False);fold['predictions_sha']=S.sha(p)
        S.write('folds/'+day+'.json',fold)
        folds.append({k:v for k,v in fold.items() if k!='baseline'})
        S.progress('fit_'+day,'complete','20 saved models reloaded and reconstructed; old direct scores and exact fold identities match.')
    pred=pd.concat([pd.read_parquet(preddir/(day+'.parquet')) for day in S.TEST],ignore_index=True)
    if set(pred.id)!=set(old.id) or len(pred)!=len(old) or pred.id.duplicated().any():raise ValueError('final prediction population')
    p=S.OUT/'predictions.parquet';pred.to_parquet(p,index=False)
    S.write('fit.json',dict(folds=folds,statuses=statuses,eligible_actions=len(a),test_actions=len(pred),
                           predictions_sha=S.sha(p),models=sum(len(f['models_sha']) for f in folds),
                           versions=dict(lightgbm=lgb.__version__,numpy=np.__version__,pandas=pd.__version__)))
    S.progress('fit','complete','All six expanding folds and 120 component models complete; prediction artifact frozen.')
    from .pf_fill_value_report import run as report
    report()


def put(test,prefix,raw,fold):
    z=constrain(raw)
    for head in S.HEADS:
        test[prefix+'_'+head+'_raw']=raw[head];test[prefix+'_'+head]=z[head]
        fold['constraints'][prefix+'_'+head]=int(np.sum(np.abs(raw[head]-z[head])>S.EPS))
    for key,value in values(test,z).items():test[prefix+'_'+key]=value


if __name__=='__main__':run()
