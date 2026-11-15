"""Controlled, consumed-data check of STOP EV estimator and target representation.

Frozen comparison: original v2 hurdle (saved control), direct L2 LightGBM,
at-price + through-price L2 LightGBM, and direct Extra Trees. Same certified
actions, 83 inputs, chronological folds and economic target. LGB parameters
unchanged apart from required objective. Extra Trees: 64 trees, depth 10,
minimum leaf 200, features .8, seed 1, one thread; native missing inputs.
No tuning, winsorization, tail deletion, raw reads or horizon changes.

Every candidate is reported raw and with ONE predeclared mean-residual offset
estimated on the intermediate reserved day. Four setups x two calibration
forms = eight descriptive variants (three new learned setups); B2/B1 saved
controls. This is not a forward race or deployment selection. No significance
test: matched expectations calculated exactly, no sampled null or p-values.
Same-count comparisons select within day/hour/side/rung/exact known remainder;
20% highest values and 10% greatest risks, ceil counts, descriptive only.
Tail cutoff = training all-action 1st percentile. All five days shown.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import roc_auc_score

from pf_stop_ev_generalization_review import validate, selftest as parity_controls

ROOT=Path('/home/yuqing/ctaNew/data/predict_fun/research/stop_ev_model_20261115')
TESTS=('20261027','20261031','20261101','20261102','20261103')
KEY=['day','m','side','rung','generation','decision_ms']
GROUP=['day','hour','side','rung','x_known_remaining']
ET=dict(n_estimators=64,max_depth=10,min_samples_leaf=200,max_features=.8,
        criterion='squared_error',random_state=1,n_jobs=1)


def require(ok,msg):
    if not bool(ok):raise ValueError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def guard():require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2048*1024,'RSS cap')


def corr(a,b):
    if len(a)<3 or np.std(a)==0 or np.std(b)==0:return None
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def auc(y,p):return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None


def matched(d,p,fraction):
    p=np.asarray(p,float);require(np.isfinite(p).all(),'nonfinite score')
    sel=np.zeros(len(d),bool);expect=0.
    for ix in d.reset_index(drop=True).groupby(GROUP,dropna=False).indices.values():
        ix=np.asarray(ix);n=int(np.ceil(fraction*len(ix)))
        chosen=ix[np.argsort(-p[ix],kind='stable')[:n]];sel[chosen]=True
        expect+=n*float(d.value_cents.to_numpy()[ix].mean())
    return sel,float(expect)


def metrics(d,p,cut):
    p=np.asarray(p,float);y=d.value_cents.to_numpy(float);f=d.filled_quantity.to_numpy()>0
    require(len(p)==len(d) and np.isfinite(p).all(),'bad prediction vector')
    keep=p>0;dec=pd.qcut(pd.Series(p).rank(method='first'),10,labels=False).to_numpy()
    top=dec==9;sel,expect=matched(d,p,.2);risk,_=matched(d,-p,.1);tail=y<=cut
    return dict(n=len(y),filled=int(f.sum()),value_mean=float(y.mean()),pred_mean=float(p.mean()),
        optimism=float((p-y).mean()),ic=corr(p,y),mse=float(np.mean((p-y)**2)),
        top_n=int(top.sum()),top_pred=float(p[top].mean()),top_value=float(y[top].mean()),
        kept_n=int(keep.sum()),kept_value_cents=float(y[keep].sum()),
        same_count_n=int(sel.sum()),same_count_value_cents=float(y[sel].sum()),
        same_count_matched_expected_cents=expect,same_count_increment_cents=float(y[sel].sum()-expect),
        tail_n=int(tail.sum()),tail_auc=auc(tail,-p),risk_count=int(risk.sum()),
        risk_tail_n=int((risk&tail).sum()),risk_tail_capture=float((risk&tail).sum()/tail.sum()) if tail.any() else None,
        gross_loss_cents=float(np.maximum(-y,0).sum()))


def selftest():
    ctl=parity_controls()
    toy=pd.DataFrame({c:[0]*40 for c in GROUP});toy['value_cents']=[-5.]*20+[3.]*20
    sel,ex=matched(toy,toy.value_cents,.2)
    require(sel.sum()==8 and toy.value_cents[sel].sum()==24 and ex==-8,'matched positive control')
    q=np.zeros(40);q[0]=np.nan
    try:matched(toy,q,.2)
    except ValueError:ctl['bad_score_refused']=True
    else:raise ValueError('nonfinite score accepted')
    x=np.zeros((400,2),np.float32);x[:30,0]=np.nan;y=np.r_[np.ones(390),np.full(10,-100.)]
    b=lgb.train(dict(objective='regression',num_threads=1,verbose=-1,min_data_in_leaf=200),lgb.Dataset(x,label=y),2)
    require(np.allclose(b.predict(x),y.mean(),atol=1e-8),'mean target tail control')
    e=ExtraTreesRegressor(**{**ET,'n_estimators':2}).fit(x,y)
    require(np.allclose(e.predict(x),y.mean(),atol=1e-8),'Extra Trees mean/missing control')
    ctl.update(matched_positive=True,tail_mean_preserved=True,extra_trees_native_missing=True)
    return ctl


def run(out):
    require(not out.exists(),'output exists');out.mkdir(parents=True);checks=selftest();hashes={};folds={};pooled=[];all_offsets={}
    def read(p):
        h=sha(p);require(str(p) not in hashes or hashes[str(p)]==h,'input changed');hashes[str(p)]=h
        return pd.read_parquet(p) if p.suffix=='.parquet' else json.loads(p.read_text())
    for day in TESTS:
        started=time.time();fit=read(ROOT/f'fit2_{day}.json');tr=fit['train_days'];cal=fit['cal'];F=fit['features']
        require(max(tr)<cal<day,'chronology');require(len(F)==83,'unexpected feature contract')
        def panel(d):
            p=ROOT/f'ev_x2_{d}.parquet';z=read(p)
            require(hashes[str(p)]==fit['inputs_sha256'][d],'source hash')
            require(not z.duplicated(KEY).any(),'duplicate action')
            require(np.allclose(z.value_cents,z.at_price_value_cents+z.through_price_value_cents,atol=1e-8,rtol=0),'component identity')
            return z
        T=pd.concat([panel(d) for d in tr],ignore_index=True);T=T[T.cert_est].reset_index(drop=True)
        X=T[F].to_numpy(np.float32);y=T.value_cents.to_numpy(float);filled=T.filled_quantity.to_numpy()>0
        require(hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()==fit['train_matrix_sha256'],'training matrix identity')
        C=panel(cal);A=panel(day);savedC=read(ROOT/f'pred2_{day}_{cal}.parquet');savedA=read(ROOT/f'pred2_{day}_{day}.parquet')
        validate(C,savedC);validate(A,savedA)
        csel=C.cert_est.to_numpy();asel=A.cert_est.to_numpy()
        C=C[csel].reset_index(drop=True);A=A[asel].reset_index(drop=True)
        savedC=savedC[csel].reset_index(drop=True);savedA=savedA[asel].reset_index(drop=True)
        XC=C[F].to_numpy(np.float32);XA=A[F].to_numpy(np.float32);cut=float(np.quantile(y,.01))
        heads={};arts={};fd=out/day;fd.mkdir()
        for h in ['p','v']:
            meta=fit['artifacts']['full_'+h];p=Path(meta['path']);hashes[str(p)]=sha(p)
            require(hashes[str(p)]==meta['sha256'],'control head hash');heads[h]=lgb.Booster(model_file=str(p))
            for xx,ss in [(XC,savedC),(XA,savedA)]:
                require(np.array_equal(heads[h].predict(xx),ss['p_fill' if h=='p' else 'v_fill']),'control head parity')
        predtr={'hurdle':heads['p'].predict(X)*heads['v'].predict(X)}
        predc={'hurdle':savedC.pred.to_numpy()};preda={'hurdle':savedA.pred.to_numpy()}
        params={**fit['params'],'objective':'regression'}
        ds=lgb.Dataset(X,label=y,feature_name=F,free_raw_data=False)
        for name,target in [('direct',y),('at_price',T.at_price_value_cents.to_numpy()),('through_price',T.through_price_value_cents.to_numpy())]:
            ds.set_label(target);b=lgb.train(params,ds,fit['rounds']);path=fd/(name+'.txt');b.save_model(str(path))
            reload=lgb.Booster(model_file=str(path));pr=b.predict(XA)
            require(np.array_equal(pr,reload.predict(XA)),'new head reload')
            arts[name]=dict(path=str(path),sha256=sha(path),reload_exact=True)
            predtr[name]=b.predict(X);predc[name]=b.predict(XC);preda[name]=pr
            del b,reload;guard()
        for ps in [predtr,predc,preda]:ps['components']=ps['at_price']+ps['through_price']
        e=ExtraTreesRegressor(**ET).fit(X,y);path=fd/'extra_trees.joblib';joblib.dump(e,path)
        reloaded=joblib.load(path);pr=e.predict(XA);require(np.array_equal(pr,reloaded.predict(XA)),'ET reload')
        arts['extra_trees']=dict(path=str(path),sha256=sha(path),reload_exact=True)
        predtr['extra_trees']=e.predict(X);predc['extra_trees']=e.predict(XC);preda['extra_trees']=pr
        del e,reloaded,ds;guard()
        variants={};offsets={};P=A[KEY+['hour','x_known_remaining','value_cents','filled_quantity','cert_ws']].copy()
        for name in ['hurdle','direct','components','extra_trees']:
            p=preda[name];pc=predc[name];offset=float(np.mean(C.value_cents.to_numpy()-pc));offsets[name]=offset
            variants[name]=dict(train=metrics(T,predtr[name],cut),cal=metrics(C,pc,cut),
                test=metrics(A,p,cut),calibrated_test=metrics(A,p+offset,cut),
                strict_test=metrics(A[A.cert_ws].reset_index(drop=True),p[A.cert_ws],cut),
                side={str(s):metrics(A[A.side==s].reset_index(drop=True),p[A.side==s],cut) for s in [0,1]})
            P[name]=p;P[name+'_calibrated']=p+offset
        P['b2']=savedA.b2;P['tail_cut']=cut;P.to_parquet(fd/'predictions.parquet',index=False)
        q=savedA.filled_quantity.to_numpy()>0;v=savedA.v_fill.to_numpy();fy=A.value_cents.to_numpy()
        diag=dict(fill_auc=auc(q,savedA.p_fill),fill_brier=float(np.mean((savedA.p_fill-q)**2)),
            filled_n=int(q.sum()),filled_value_ic=corr(v[q],fy[q]),
            filled_value_r2=float(1-np.sum((v[q]-fy[q])**2)/np.sum((fy[q]-fy[q].mean())**2)),
            filled_mean=float(fy[q].mean()),filled_head_mean=float(v[q].mean()),
            components={k:dict(pred_mean=float(preda[k].mean()),true_mean=float(A[k+'_value_cents'].mean()),
                mse=float(np.mean((preda[k]-A[k+'_value_cents'].to_numpy())**2))) for k in ['at_price','through_price']})
        rec=dict(day=day,asof=datetime.now(timezone.utc).isoformat(),train_days=tr,cal_day=cal,
            train_n=len(T),cal_n=len(C),test_n=len(A),training_tail_cut_cents=cut,
            excluded_test=len(asel)-int(asel.sum()),prep_status=read(ROOT/f'ev_x2_{day}.json'),
            params=params,extra_trees_params=ET,rounds=fit['rounds'],features=F,
            variants=variants,calibration_offsets=offsets,diagnostics=diag,
            b2=metrics(A,savedA.b2,cut),b1=metrics(A,savedA.b1,cut),artifacts=arts,
            seconds=time.time()-started)
        (fd/'receipt.json').write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
        folds[day]=rec;pooled.append(P);all_offsets[day]=offsets
        print(json.dumps(dict(day=day,seconds=rec['seconds'],test={k:z['test'] for k,z in variants.items()})),flush=True)
        del T,C,A,X,XC,XA,savedA,savedC,predtr,predc,preda,heads;gc.collect();guard()
    P=pd.concat(pooled,ignore_index=True);pools={}
    # Pooled metrics allow the fold-specific training tail cutoff per row.
    for col in ['hurdle','direct','components','extra_trees','b2','hurdle_calibrated','direct_calibrated','components_calibrated','extra_trees_calibrated']:
        pools[col]=metrics(P,P[col],P.tail_cut.to_numpy())
    require(len({pools[k]['same_count_n'] for k in pools})==1,'matched count differs')
    for p,h in hashes.items():require(sha(p)==h,'input changed during run')
    result=dict(asof=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),
        dependencies={'generalization_review':sha(Path(__file__).with_name('pf_stop_ev_generalization_review.py'))},
        versions=dict(lightgbm=lgb.__version__,sklearn=sklearn.__version__),controls=checks,input_sha256=hashes,
        n_unique_test_actions=len(P),folds=folds,pooled=pools,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
        interpretation='Consumed development days, local one-second value, no portfolio claim; eight variants, no tuning or candidate promoted; no inferential null or intervals.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(asof=result['asof'],pooled=pools,peak_rss_mib=result['peak_rss_mib'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);run(ap.parse_args().out)
