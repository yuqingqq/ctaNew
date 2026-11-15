"""Read-only, post hoc review of the completed STOP-EV model artifacts.

Recompute published metrics, verify prediction/feature-label identity, and join
the independently audited timing certificates by exact neutral action/time.
No model fitting, candidate selection, new significance test or raw tape reads.
Only the five already-consumed test days. Do not treat certificate failures as
observed label errors or compare predictive skill on selected new subsets.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource

import numpy as np
import pandas as pd

ROOT = Path('/home/yuqing/ctaNew')
DATA = ROOT/'data/predict_fun/research'
DAYS = ('20261027','20261031','20261101','20261102','20261103')
KEY = ['day','m','side','rung','generation','decision_ms']


def require(ok, message):
    if not bool(ok): raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_predictions(p, x):
    require(not p.duplicated(KEY).any(), 'duplicate prediction action')
    require(not x.duplicated(KEY).any(), 'duplicate source action')
    require(np.isfinite(p[['pred','p_fill','v_fill','b1','b2','value_cents']]).all().all(), 'nonfinite prediction')
    require(p.p_fill.between(0,1).all(), 'invalid fill probability')
    require(np.allclose(p.pred,p.p_fill*p.v_fill,rtol=0,atol=1e-12), 'hurdle product differs')
    cols=['value_cents','filled_quantity','x_known_remaining','status']
    j=p[KEY+cols].merge(x[KEY+cols],on=KEY,how='outer',validate='one_to_one',indicator=True,suffixes=('_p','_x'))
    require((j._merge=='both').all(), 'prediction/source population differs')
    for c in cols: require((j[c+'_p']==j[c+'_x']).all(), 'prediction/source '+c+' differs')


def huber_location(values, delta):
    lo, hi = float(min(values)), float(max(values))
    for _ in range(100):
        mid=(lo+hi)/2
        if np.clip(np.asarray(values)-mid,-delta,delta).mean()>0:lo=mid
        else:hi=mid
    return (lo+hi)/2


def main(out):
    src=DATA/'stop_ev_model_20261115';cert=DATA/'stop_ev_timing_certified_20261115_v1'
    hashes={};panels=[];counts={};controls={}
    def record(p): hashes[str(p)]=sha(p)
    report_path=src/'stop_ev_model.json';record(report_path)
    report=json.loads(report_path.read_text());folds={}
    for day in DAYS:
        pp=src/f'pred_{day}_{day}.parquet';xp=src/f'ev_x_{day}.parquet';fp=src/f'fit_{day}.json'
        for p in (pp,xp,fp):record(p)
        pred=pd.read_parquet(pp);x=pd.read_parquet(xp,columns=KEY+['value_cents','filled_quantity','x_known_remaining','status'])
        check_predictions(pred,x)
        if not controls:
            cases={'duplicate':pd.concat([pred,pred.iloc[:1]],ignore_index=True)}
            for name,col,value in [('probability','p_fill',2.),('label','value_cents',float(pred.iloc[0].value_cents)+1.)]:
                b=pred.copy();b.loc[b.index[0],col]=value;cases[name]=b
            for name,b in cases.items():
                try:check_predictions(b,x)
                except ValueError:controls[name]=True
                else:controls[name]=False
            require(all(controls.values()),'bad artifact accepted')
        fit=json.loads(fp.read_text());ic=np.asarray(fit['T5_ic'])
        folds[day]=dict(draws=len(ic),original_gate=bool(np.abs(ic).mean()<=.01 and np.abs(ic).max()<=.03),
                       revised_gate=bool(ic.mean()<=.01 and ic.max()<=.03),stored_gate=fit['T5_pass'])
        crp=cert/day/'receipt.json';record(crp);cr=json.loads(crp.read_text());counts[day]={}
        for mode in ('certified_ws','certified_estimated'):
            path=cert/day/(mode+'.parquet');record(path)
            require(hashes[str(path)]==cr['artifacts'][path.name], 'certificate artifact changed')
            c=pd.read_parquet(path,columns=KEY+['status','value_cents'])
            c['day']=c['day'].astype('int64')
            j=pred.merge(c,on=KEY,how='left',validate='one_to_one',indicator=True,suffixes=('','_cert'))
            require((j._merge=='both').all(),'certificate join missing action/time')
            ok=j.status_cert.str.startswith('eligible')
            require((j.loc[ok,'value_cents']==j.loc[ok,'value_cents_cert']).all(),'certified overlap label differs')
            counts[day][mode]=dict(predictions=len(j),certified=int(ok.sum()),without_certificate=int((~ok).sum()),
                original_ws_claim=int((j.status=='eligible').sum()),
                original_ws_claim_uncertified=int(((j.status=='eligible') & ~ok).sum()),
                statuses={k:int(v) for k,v in j.status_cert.value_counts().items()})
        panels.append(pred)
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2048*1024,'2 GiB RSS guard')
    p=pd.concat(panels,ignore_index=True);keep=p.pred>0
    ic=float(p.pred.rank().corr(p.value_cents.rank()));icb=float(p.b2.rank().corr(p.value_cents.rank()))
    kept=float(p.loc[keep,'value_cents'].sum());gain=float(-p.loc[~keep,'value_cents'].sum())
    b2gain=float(-p.loc[p.b2<=0,'value_cents'].sum())
    for actual,expected in [(ic,report['pooled']['ic']),(icb,report['pooled']['ic_b2']),
                            (gain,report['pooled']['gain']),(kept,report['metrics']['local_kept_value_cents']),
                            (b2gain,report['pooled']['gain_b2'])]:
        require(abs(actual-expected)<1e-7,'published metric differs')
    dec=pd.qcut(p.pred.rank(method='first'),10,labels=False)
    calibration=[dict(n=int((dec==i).sum()),prediction=float(p.loc[dec==i,'pred'].mean()),
                      realized=float(p.loc[dec==i,'value_cents'].mean())) for i in range(10)]
    model_source=ROOT/'predictfun/research/pf_stop_ev_model.py';record(model_source)
    tree=ast.parse(model_source.read_text());t5=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='t5_pass')
    scope=dict(np=np,T5_MEAN=.01,T5_MAX=.03);exec(compile(ast.Module(body=[t5],type_ignores=[]),str(model_source),'exec'),scope)
    negative_gate=bool(scope['t5_pass']([-1.]*10));positive_gate=bool(scope['t5_pass']([1.]*10))
    require(negative_gate and not positive_gate,'gate counterexample no longer reproduces')
    values=np.array([1.]*900+[-100.]*100);h=huber_location(values,1.)
    require(abs(h-8/9)<1e-9 and values.mean()==-9.1,'Huber sign counterexample failed')
    require(abs(huber_location(values,1000.)-values.mean())<1e-9,'quadratic-limit positive control failed')
    for name,digest in hashes.items():require(sha(name)==digest,'artifact changed during review')
    result=dict(asof=datetime.now(timezone.utc).isoformat(),source_asof=report['metrics']['asof'],
        checker_sha256=sha(__file__),input_sha256=hashes,artifact_identity_pass=True,controls=controls,
        n=len(p),days=list(DAYS),metrics=dict(ic=ic,ic_b2=icb,kept=int(keep.sum()),pull_fraction=float((~keep).mean()),
        kept_value_cents=kept,local_gain_cents=gain,b2_gain_cents=b2gain,increment_above_b2_cents=gain-b2gain),
        calibration=calibration,timing_membership=counts,t5=folds,
        counterexamples=dict(t5_accepts_perfect_negative_skill=negative_gate,t5_rejects_perfect_positive_skill=not positive_gate,
            huber=dict(n=1000,delta=1.,mean=float(values.mean()),minimizer=h,quadratic_control_pass=True)),
        limitations='Post hoc saved-artifact review on consumed days. No refit, new null, performance comparison on certificate subsets, raw-tape read, or physical timing validation.')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['asof','source_asof','n','metrics','controls','t5','counterexamples']}))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);main(a.parse_args().out)
