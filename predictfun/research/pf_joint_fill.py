"""A convex mixture of valid paired outcomes supplies every forecast component."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from . import pf_joint_fill_spec as S
from .pf_model_spec import PARAMS,ROUNDS
from .pf_order_policy_inputs import MARKET,FEATURES
from . import pf_fill_value as OLD
from .pf_default_pull_verify import actions


def atoms(a):
    OLD.validate(a)
    p=a.order_price.to_numpy();rem=a.remaining.to_numpy();buy=a.side.eq(0).to_numpy()
    if not np.isfinite(p).all() or ((p<=0)|(p>=100)).any():raise ValueError('price support')
    gcap=np.where(buy,100-p,p);lcap=100-gcap
    v=np.ones((len(a),9));event=np.zeros(len(a),int)
    for branch,offset,bit in [('keep',1,1),('reset',5,2)]:
        z=OLD.targets(a,branch);event+=bit*z['fill'].astype(int)
        v[:,offset]=z['fraction'];v[:,offset+1]=z['fraction2']
        v[:,offset+2]=z['gain']/(rem*gcap);v[:,offset+3]=z['loss']/(rem*lcap)
    j=np.zeros((len(a),4,9));j[np.arange(len(a)),event]=v
    support(j)
    return event,v,j.mean(axis=0)


def support(j):
    if j.ndim!=3 or j.shape[1:]!=(4,9) or not np.isfinite(j).all() or (j < -1e-10).any():raise ValueError('invalid joint distribution')
    mass=j[:,:,0]
    if not np.allclose(mass.sum(axis=1),1.,atol=1e-9,rtol=0):raise ValueError('probability mass')
    for off,bit in [(1,1),(5,2)]:
        f,f2,g,l=(j[:,:,off+k] for k in range(4))
        if ((f>mass+1e-9)|(f2>f+1e-9)|(f*f>mass*f2+1e-9)|(g+l>f+1e-9)).any():raise ValueError('incoherent moments')
        inactive=[k for k in range(4) if not k&bit]
        if (np.abs(j[:,inactive,off:off+4])>1e-9).any():raise ValueError('value or quantity without fill event')
        if (((f<=1e-12)&((g+l)>1e-9))).any():raise ValueError('zero quantity nonzero value')


def aggregate(a,j):
    support(j);out={};rem=a.remaining.to_numpy();p=a.order_price.to_numpy();sg=np.where(a.side.eq(0),1.,-1.)
    gcap=np.where(sg>0,100-p,p);lcap=100-gcap
    for branch,off,bit in [('keep',1,1),('reset',5,2)]:
        mass=j[:,[c for c in range(4) if c&bit],0].sum(axis=1)
        f,f2,g,l=(j[:,:,off+k].sum(axis=1) for k in range(4))
        q=rem*f;q2=rem*rem*f2;gain=rem*gcap*g;loss=rem*lcap*l
        risk=.01*(2*a.inventory.to_numpy()*sg*q+q2)
        out[branch]=dict(fill=mass,quantity=q,quantity2=q2,gain=gain,loss=loss,mark=gain-loss,risk=risk,value=gain-loss-risk)
        if ((gain>gcap*q+1e-8)|(loss>lcap*q+1e-8)|((gain/gcap+loss/lcap)>q+1e-8)).any():raise ValueError('price/value support')
    return out


def leaf_tables(leaves,event,v,prior):
    n,t=leaves.shape;k=int(leaves.max())+1
    table=np.zeros((t,k,4,9));counts=np.zeros((t,k),int)
    for j in range(t):
        ids=leaves[:,j];counts[j]=np.bincount(ids,minlength=k);ix=ids*4+event
        for h in range(9):table[j,:,:,h]=np.bincount(ix,weights=v[:,h],minlength=k*4).reshape(k,4)
        valid=counts[j]>0;table[j,valid]/=counts[j,valid,None,None];table[j,~valid]=prior
    support(table.reshape(-1,4,9))
    return table,counts


def forecast(model,cols,a,table,prior):
    leaves=np.asarray(model.predict(a[cols],pred_leaf=True,num_threads=1),int)
    if leaves.ndim==1:leaves=leaves[:,None]
    if leaves.shape[1]!=len(table) or leaves.max()>=table.shape[1]:raise ValueError('leaf identity')
    out=np.zeros((len(a),4,9))
    for t in range(len(table)):out+=table[t,leaves[:,t]]
    out=.99*out/len(table)+.01*prior
    support(out);return out


def apply_calibration(j,weights):
    w=np.asarray(weights,float)
    if w.shape!=(4,) or not np.isfinite(w).all() or (w<=0).any():raise ValueError('invalid joint calibration weights')
    out=j*w[None,:,None];den=out[:,:,0].sum(axis=1)
    out/=den[:,None,None];support(out);return out


def calibrate(j,event,prior):
    support(j);p=j[:,:,0];core=prior[:,0];active=core>0;counts=np.bincount(event,minlength=4)
    if counts[~active].sum():
        return dict(status='REFUSED_UNSEEN_EVENT',weights=[1.]*4,n=len(event),unseen_actions=int(counts[~active].sum()),counts=counts.tolist())
    target=(counts+50*core)/(len(event)+50);q=p[:,active];tar=target[active];b=np.zeros(q.shape[1]);logp=np.log(q)
    def objective(b):
        z=logp+b;mx=z.max(axis=1);return float((mx+np.log(np.exp(z-mx[:,None]).sum(axis=1))).mean()-tar@b)
    for iteration in range(100):
        z=logp+b;z-=z.max(axis=1)[:,None];w=np.exp(z);w/=w.sum(axis=1)[:,None]
        grad=w.mean(axis=0)-tar
        if np.max(np.abs(grad))<1e-9:break
        h=np.diag(w.mean(axis=0))-w.T@w/len(w)
        step=np.linalg.solve(h[:-1,:-1]+1e-12*np.eye(len(b)-1),grad[:-1])
        delta=np.r_[step,0.];scale=1.;old=objective(b)
        for _ in range(40):
            trial=b-scale*delta
            if objective(trial)<=old+1e-13:break
            scale*=.5
        else:raise ValueError('calibration line search')
        b=trial
    else:raise ValueError('calibration convergence')
    weights=np.ones(4);weights[active]=np.exp(b-b.mean())
    calibrated=apply_calibration(j,weights);mean=calibrated[:,:,0].mean(axis=0)
    if np.max(np.abs(mean-target))>1e-7:raise ValueError('calibration target mismatch')
    return dict(status='PASS',weights=weights.tolist(),n=len(event),counts=counts.tolist(),iterations=iteration,
                target=target.tolist(),mean_before=p.mean(axis=0).tolist(),mean_after=mean.tolist())


def controls():
    OLD.controls()
    a=pd.DataFrame(dict(id=['a','b','c','d'],day=['20261023']*4,status=['eligible']*4,side=[0.,0.,1.,1.],
        order_price=[30.,40.,60.,70.],inventory=[0.,1.,0.,-1.],remaining=[5.]*4,label_q=[0.,2.,5.,1.],reset_q=[0.,0.,2.,0.],
        label_value1=[0.,4.,-5.,2.],label_value5=[0.,4.,-5.,2.],reset_value1=[0.,0.,-2.,0.],reset_value5=[0.,0.,-2.,0.]))
    for branch in ('keep','reset'):a[branch+'_value']=OLD.values(a,OLD.targets(a,branch))['value']
    a['label']=a.keep_value-a.reset_value;event,v,prior=atoms(a)
    j=np.zeros((len(a),4,9));j[np.arange(len(a)),event]=v
    z=aggregate(a,j)
    for branch in ('keep','reset'):np.testing.assert_allclose(z[branch]['value'],a[branch+'_value'],atol=1e-10,rtol=0)
    mixture=np.repeat(prior[None],len(a),axis=0);support(mixture)
    calibrated=calibrate(mixture,event,prior);support(apply_calibration(mixture,calibrated['weights']))
    changed=calibrate(np.repeat(prior[None],100,axis=0),np.array([0]*70+[1]*20+[3]*10),prior)
    if np.allclose(changed['weights'],1.):raise ValueError('calibration positive control did not move')
    unseen=calibrate(mixture,np.array([0,1,2,3]),prior)
    if unseen['status']!='REFUSED_UNSEEN_EVENT':raise ValueError('unseen event refusal')
    bad=j.copy();bad[0,0,3]=.1
    bad2=mixture.copy();bad2[:,:,2]=100
    for b in (bad,bad2):
        try:support(b)
        except ValueError:pass
        else:raise ValueError('impossible outcome accepted')
    try:apply_calibration(j,[1.,-1.,1.,1.])
    except ValueError:pass
    else:raise ValueError('negative mixture weight accepted')
    leaves=np.array([[0,0],[0,1],[1,0],[1,1]])
    table,c=leaf_tables(leaves,event,v,prior)
    np.testing.assert_allclose(table[0,0],j[:2].mean(axis=0),atol=1e-12)
    return dict(status='PASS',scope='Valid paired atoms reconstruct; convex mixtures and shared calibration satisfy support; impossible quantity/value and negative-weight refusals; known calibration target moves and unseen event refuses; chronology/feature controls inherited.')


def run():
    S.check();S.write('controls.json',controls());S.progress('controls','complete','Joint support, calibration, accounting and leakage positive/refusal controls passed.')
    parts=[];statuses=[]
    for day in S.DAYS:
        a=pd.read_parquet(S.PRIOR/'runs/build'/day/'generations/default.parquet');actions(a)
        statuses += [dict(day=day,status=k,n=int(n)) for k,n in a.status.value_counts().items()]
        a=a[a.status.eq('eligible')&a.label.notna()].copy();atoms(a);parts.append(a)
    a=pd.concat(parts,ignore_index=True);OLD.validate(a)
    previous=pd.read_parquet(S.PREVIOUS/'predictions.parquet');oldfit=json.loads((S.PRIOR/'fit.json').read_text())
    modeldir=S.OUT/'models';modeldir.mkdir(exist_ok=False);preds=S.OUT/'fold_predictions';preds.mkdir(exist_ok=False)
    folds=[]
    for oldfold in oldfit['folds']:
        day=oldfold['day'];calday=oldfold['train_days'][-1];core=a[a.day<calday];cal=a[a.day.eq(calday)];test=a[a.day.eq(day)].copy()
        OLD.chronology(core,cal);OLD.chronology(cal,test)
        if len(core)+len(cal)!=oldfold['train_n'] or len(test)!=oldfold['test_n']:raise ValueError('fold population')
        cols=['id','market_score','queue_score','market_component_score','queue_component_score']
        cols += [c for c in previous if c.startswith(('baseline_keep_','baseline_reset_','queue_keep_','queue_reset_','market_keep_','market_reset_'))]
        test=test.merge(previous[cols],on='id',how='left',validate='one_to_one')
        if test.queue_score.isna().any():raise ValueError('prior action identity')
        event,v,prior=atoms(core);calevent,_,_=atoms(cal)
        fold=dict(day=day,core_days=sorted(core.day.unique()),calibration_day=calday,core_n=len(core),calibration_n=len(cal),test_n=len(test),models={},calibrators={})
        tensors={}
        S.progress('fit_'+day,'running',f'Core {len(core)} actions before calibration day {calday}; test {len(test)}. Shared outcomes for both actions.')
        for family,features in [('market',MARKET),('queue',FEATURES)]:
            OLD.safe_features(features)
            model=lgb.train(PARAMS,lgb.Dataset(core[features],label=core.label),ROUNDS)
            path=modeldir/f'{day}_{family}.txt';model.save_model(str(path));model=lgb.Booster(model_file=str(path))
            leaves=np.asarray(model.predict(core[features],pred_leaf=True,num_threads=1),int)
            if leaves.ndim==1:leaves=leaves[:,None]
            table,counts=leaf_tables(leaves,event,v,prior)
            p=modeldir/f'{day}_{family}_joint.npz';np.savez_compressed(p,table=table,counts=counts,prior=prior)
            with np.load(p) as saved:
                np.testing.assert_array_equal(table,saved['table']);np.testing.assert_array_equal(prior,saved['prior'])
                raw=forecast(model,features,test,saved['table'],saved['prior'])
                calraw=forecast(model,features,cal,saved['table'],saved['prior'])
            fit=calibrate(calraw,calevent,prior);fold['calibrators'][family]=fit
            calibrated=apply_calibration(raw,fit['weights']);tensors[family+'_raw']=raw;tensors[family+'_cal']=calibrated
            for variant,j in [('raw',raw),('cal',calibrated)]:
                prefix=family+'_'+variant;out=aggregate(test,j)
                for branch,z in out.items():
                    for key,x in z.items():test[prefix+'_'+branch+'_'+key]=x
                test[prefix+'_score']=out['keep']['value']-out['reset']['value']
            # Rebuild the complete forecast from the stored model, table and JSON-safe calibrator.
            savedfit=json.loads(json.dumps(S.native(fit)))
            rebuilt=apply_calibration(forecast(lgb.Booster(model_file=str(path)),features,test,table,prior),savedfit['weights'])
            np.testing.assert_allclose(rebuilt,calibrated,atol=1e-12,rtol=0)
            corrupt=calibrated.copy();corrupt[0,0,0]+=1
            try:support(corrupt)
            except ValueError:pass
            else:raise ValueError('corrupt forecast accepted')
            fold['models'][family]=dict(model_sha=S.sha(path),joint_sha=S.sha(p),trees=len(table))
            print('JOINT_MODEL_DONE',day,family,fit['status'],flush=True)
        p=preds/(day+'.parquet');test.to_parquet(p,index=False);q=preds/(day+'_joint.npz');np.savez_compressed(q,**tensors)
        fold.update(predictions_sha=S.sha(p),joint_predictions_sha=S.sha(q));S.write('folds/'+day+'.json',fold);folds.append(fold)
        S.progress('fit_'+day,'complete','Both paired-outcome models and calibrators saved/reconstructed; all predicted joint distributions satisfy support.')
    pred=pd.concat([pd.read_parquet(preds/(d+'.parquet')) for d in S.TEST],ignore_index=True)
    if len(pred)!=len(previous) or set(pred.id)!=set(previous.id) or pred.id.duplicated().any():raise ValueError('final action population')
    p=S.OUT/'predictions.parquet';pred.to_parquet(p,index=False)
    S.write('fit.json',dict(folds=folds,statuses=statuses,actions=len(pred),models=12,predictions_sha=S.sha(p),versions=dict(lightgbm=lgb.__version__,numpy=np.__version__)))
    from .pf_joint_fill_report import run as report
    report()


if __name__=='__main__':run()
