"""Post hoc cause analysis on consumed STOP-EV actions; no policy promotion.

Declared before execution: reproduce all five existing folds exactly, then
change only conditional-value objective Huber -> squared error. Same binary
fill head, training rows, features, seeds, rounds and parameters. Two variants
(one original control, one loss ablation), no tuning or candidate selection.
Report every fold, original positive-prediction and original top-decile sets,
calibration, filled-value tails, probability/value residual decomposition,
certificate strata and train/test outcome shifts. Certificate strata are
descriptive selected populations, never evidence of causal timing attribution.
No new null, interval, significance or out-of-sample promotion claim. Existing
chronological test days are consumed. No protected raw data is opened.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import resource
import subprocess

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT=Path('/home/yuqing/ctaNew');DATA=ROOT/'data/predict_fun/research'
SRC=DATA/'stop_ev_model_20261115';CERT=DATA/'stop_ev_timing_certified_20261115_v1'
DAYS=('20261027','20261031','20261101','20261102','20261103')
ALLOWED=('20261023','20261024','20261025','20261026',*DAYS)
KEY=['day','m','side','rung','generation','decision_ms']


def now():return datetime.now(timezone.utc).isoformat()


def require(ok,msg):
    if not bool(ok):raise ValueError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def guard(*_):
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2048*1024,'2 GiB RSS limit')


def dump(p,obj):
    with p.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')


def corr(a,b):
    x=float(pd.Series(a).rank().corr(pd.Series(b).rank()))
    return x if np.isfinite(x) else None


def stat(p,pc='pred',vc='v_fill'):
    if not len(p):return dict(n=0)
    y=p.value_cents.to_numpy(float);pred=p[pc].to_numpy(float);v=p[vc].to_numpy(float)
    prob=p.p_fill.to_numpy(float);fill=p.filled_quantity.to_numpy(float)>0;kept=pred>0
    pt=(prob-fill)*v;vt=np.where(fill,v-y,0.)
    require(np.allclose(pred-y,pt+vt,rtol=0,atol=1e-8),'decomposition identity')
    gross_loss=-y[y<0].sum();n_tail=max(1,int(np.ceil(len(y)*.01)))
    return dict(n=len(p),filled=int(fill.sum()),realized_mean=float(y.mean()),predicted_mean=float(pred.mean()),
        optimism=float((pred-y).mean()),fill_rate=float(fill.mean()),predicted_fill_rate=float(prob.mean()),
        fill_brier=float(((prob-fill)**2).mean()),ic=corr(pred,y),mse=float(((pred-y)**2).mean()),
        probability_residual_term=float(pt.mean()),value_residual_term=float(vt.mean()),
        filled_actual_mean=float(y[fill].mean()) if fill.any() else None,
        filled_predicted_value_mean=float(v[fill].mean()) if fill.any() else None,
        kept=int(kept.sum()),kept_value=float(y[kept].sum()),kept_expected=float(pred[kept].sum()),
        zero_fraction=float((y==0).mean()),gross_loss=float(gross_loss),gross_gain=float(y[y>0].sum()),
        worst_one_percent_loss_share=float(-np.minimum(np.sort(y)[:n_tail],0.).sum()/gross_loss) if gross_loss else None,
        at_price=float(p.at_price_value_cents.sum()),through_price=float(p.through_price_value_cents.sum()))


def outcome(d):
    y=d.value_cents.to_numpy(float);fill=d.filled_quantity.to_numpy(float)>0
    return dict(n=len(d),filled=int(fill.sum()),fill_rate=float(fill.mean()),value_mean=float(y.mean()),
        filled_value_mean=float(y[fill].mean()),filled_median=float(np.median(y[fill])),
        filled_q01=float(np.quantile(y[fill],.01)),filled_q05=float(np.quantile(y[fill],.05)),
        at_price_mean=float(d.at_price_value_cents.mean()),through_price_mean=float(d.through_price_value_cents.mean()))


def check(p):
    require(not p.duplicated(KEY).any(),'duplicate action')
    require(np.isfinite(p[['pred','p_fill','v_fill','value_cents','filled_quantity']]).all().all(),'nonfinite input')
    require(p.p_fill.between(0,1).all(),'bad probability')
    require(np.allclose(p.pred,p.p_fill*p.v_fill,rtol=0,atol=1e-12),'bad hurdle product')
    require((p.loc[p.filled_quantity==0,'value_cents']==0).all(),'nonfill value nonzero')
    require(np.allclose(p.value_cents,p.at_price_value_cents+p.through_price_value_cents,rtol=0,atol=1e-8),'bad components')


def controls(p):
    check(p);out={}
    for name,col,val in [('product','pred',float(p.iloc[0].pred)+1.),('probability','p_fill',2.),('component','at_price_value_cents',float(p.iloc[0].at_price_value_cents)+1.)]:
        b=p.copy();b.loc[b.index[0],col]=val
        try:check(b)
        except ValueError:out[name]=True
        else:out[name]=False
    b=pd.concat([p,p.iloc[:1]],ignore_index=True)
    try:check(b)
    except ValueError:out['duplicate']=True
    else:out['duplicate']=False
    require(all(out.values()),'corrupt input accepted')
    return out


def run(out):
    out.mkdir(parents=True,exist_ok=False);hashes={};source_hash=sha(__file__);records={};allp=[];ctl=None
    def read(path,cols=None):
        require(path.is_file(),'missing artifact '+str(path));hashes[str(path)]=sha(path)
        return pd.read_parquet(path,columns=cols) if path.suffix=='.parquet' else json.loads(path.read_text())
    for day in DAYS:
        print('starting fold',day,flush=True)
        fit=read(SRC/f'fit_{day}.json');trdays=fit['train_days']
        require(all(d in ALLOWED for d in trdays) and max(trdays)<fit['cal']<day,'chronology/day boundary')
        F=fit['features'];params=fit['params'];rounds=fit['rounds']
        tr=pd.concat([read(SRC/f'ev_x_{d}.parquet') for d in trdays],ignore_index=True)
        te=read(SRC/f'ev_x_{day}.parquet');p=read(SRC/f'pred_{day}_{day}.parquet')
        require(p[KEY].equals(te[KEY]),'saved prediction row order differs')
        require(np.array_equal(p.value_cents,te.value_cents),'saved labels differ')
        for c in ('at_price_value_cents','through_price_value_cents'):p[c]=te[c].values
        check(p)
        if ctl is None:ctl=controls(p)
        guard();X=tr[F].to_numpy(np.float32);Xt=te[F].to_numpy(np.float32)
        y=tr.value_cents.to_numpy(float);filled=tr.filled_quantity.to_numpy(float)>0
        delta=float(np.median(np.abs(y[filled])));delta=delta if delta>0 else 1.
        require(delta==fit['delta'] and len(tr)==fit['n_train'],'training inputs differ')
        b=lgb.train(params,lgb.Dataset(X,filled.astype(float),feature_name=F),rounds,callbacks=[guard])
        h=lgb.train({**params,'objective':'huber','alpha':delta},lgb.Dataset(X[filled],y[filled],feature_name=F),rounds,callbacks=[guard])
        pp=b.predict(Xt);vh=h.predict(Xt)
        parity=dict(p_fill=float(np.max(np.abs(pp-p.p_fill))),v_fill=float(np.max(np.abs(vh-p.v_fill))),
                    product=float(np.max(np.abs(pp*vh-p.pred))))
        require(max(parity.values())<=1e-10,'control does not reproduce saved predictions')
        m=lgb.train({**params,'objective':'regression_l2'},lgb.Dataset(X[filled],y[filled],feature_name=F),rounds,callbacks=[guard])
        vm=m.predict(Xt);p['l2_v']=vm;p['l2_pred']=pp*vm
        for mode in ('certified_ws','certified_estimated'):
            cr=read(CERT/day/'receipt.json');path=CERT/day/(mode+'.parquet')
            c=read(path,KEY+['status','value_cents']);require(hashes[str(path)]==cr['artifacts'][path.name],'certificate changed')
            c.day=c.day.astype('int64');j=p[KEY+['value_cents']].merge(c,on=KEY,how='left',validate='one_to_one',indicator=True,suffixes=('','_c'))
            require((j._merge=='both').all(),'certificate action missing');ok=j.status.str.startswith('eligible')
            require(np.array_equal(j.loc[ok,'value_cents'],j.loc[ok,'value_cents_c']),'certificate value mismatch')
            p[mode]=ok.values
        rec=dict(asof=now(),train=outcome(tr),test=outcome(te),delta=delta,control_parity=parity,
            huber=stat(p),squared_error=stat(p,'l2_pred','l2_v'),
            original_kept=dict(huber=stat(p[p.pred>0]),squared_error=stat(p[p.pred>0],'l2_pred','l2_v')))
        fd=out/day;fd.mkdir()
        for name,model in [('fill',b),('huber_value',h),('l2_value',m)]:
            path=fd/(name+'.txt');model.save_model(str(path))
            pred0=model.predict(Xt);pred1=lgb.Booster(model_file=str(path)).predict(Xt)
            require(np.array_equal(pred0,pred1),'model reload differs')
        p.to_parquet(fd/'predictions.parquet',index=False)
        rec['artifacts']={f.name:sha(f) for f in fd.iterdir()}
        dump(fd/'receipt.json',rec);records[day]=rec;allp.append(p)
        print(json.dumps(dict(day=day,parity=parity,huber_bias=rec['huber']['optimism'],l2_bias=rec['squared_error']['optimism'])),flush=True)
        del tr,te,X,Xt,b,h,m,p;gc.collect();guard()
    p=pd.concat(allp,ignore_index=True);dec=pd.qcut(p.pred.rank(method='first'),10,labels=False)
    group=lambda q:dict(huber=stat(q),squared_error=stat(q,'l2_pred','l2_v'))
    summary=dict(all=group(p),original_kept=group(p[p.pred>0]),original_top_decile=group(p[dec==9]),
        by_day={d:dict(huber=r['huber'],squared_error=r['squared_error'],original_kept=r['original_kept']) for d,r in records.items()},
        by_side={str(s):group(q) for s,q in p.groupby('side')},
        timing=dict(strict=group(p[p.certified_ws]),estimated_only=group(p[p.certified_estimated & ~p.certified_ws]),
                    uncertified=group(p[~p.certified_estimated])),
        deciles={str(i):group(p[dec==i]) for i in range(10)})
    # Formula-based tail audit only: not a candidate or deployable policy.
    tail=[]
    for day,q in p.groupby('day'):
        delta=records[str(day)]['delta'];f=q.filled_quantity>0;y=q.value_cents.to_numpy(float)
        tail.append(dict(day=int(day),n=len(q),filled=int(f.sum()),delta=delta,
            below_minus_delta=int((y < -delta).sum()),above_delta=int((y>delta).sum()),
            negative_tail_sum=float(y[y < -delta].sum()),positive_tail_sum=float(y[y>delta].sum()),
            clipped_value_sum=float(np.clip(y,-delta,delta).sum()),actual_value_sum=float(y.sum())))
    require(sha(__file__)==source_hash,'source changed')
    for path,h in hashes.items():require(sha(path)==h,'input changed')
    result=dict(asof=now(),source_model_asof='2026-11-15T14:00:57Z',source_sha256=source_hash,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),
        input_sha256=hashes,versions=dict(lightgbm=lgb.__version__,numpy=np.__version__,pandas=pd.__version__),
        controls=ctl,control_parity={d:r['control_parity'] for d,r in records.items()},summary=summary,tail=tail,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
        interpretation='Controlled objective ablation on already-consumed data, conditional on original timing labels. No tuning, significance, untouched validation, live fills or portfolio claim. Residual decomposition is an accounting identity, not causal attribution.')
    dump(out/'report.json',result)
    print(json.dumps(dict(asof=result['asof'],all=summary['all'],original_top_decile=summary['original_top_decile'],original_kept=summary['original_kept'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
