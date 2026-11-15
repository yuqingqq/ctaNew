"""Read-only audit of the completed v2 fits, on already-consumed days.

Recompute training IC from saved heads and exact declared training matrices;
compare with the reserved intermediate day and later test day. Reproduce all
saved full-head predictions, certificates, keys and values before reporting.
No fits, selection, significance tests or new data. Training metrics are
resubstitution, not validation; calendar outcomes and fitting optimism cannot
be causally separated by this descriptive comparison.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import subprocess

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT=Path('/home/yuqing/ctaNew/data/predict_fun/research/stop_ev_model_20261115')
TESTS=('20261027','20261031','20261101','20261102','20261103')
KEY=['day','m','side','rung','generation','decision_ms']


def require(ok,msg):
    if not bool(ok):raise ValueError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def ic(a,b):return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def validate(d,p):
    require(not d.duplicated(KEY).any() and not p.duplicated(KEY).any(),'duplicate action')
    require(d[KEY].equals(p[KEY]),'action identity')
    for c in ['status','value_cents','filled_quantity','cert_est','cert_ws']:
        require(np.array_equal(d[c],p[c]),'source field '+c)
    require(np.isfinite(p[['pred','p_fill','v_fill']].to_numpy()).all(),'nonfinite prediction')
    require(np.array_equal(p.pred,p.p_fill*p.v_fill),'product identity')


def selftest():
    d=pd.DataFrame({c:[1,2] for c in KEY})
    for c in ['status','value_cents','filled_quantity','cert_est','cert_ws']:d[c]=[1,2]
    p=d.copy();p['p_fill']=[.1,.2];p['v_fill']=[2.,3.];p['pred']=p.p_fill*p.v_fill
    validate(d,p);n=0
    for c in ['decision_ms','value_cents','pred']:
        q=p.copy();q.loc[0,c]+=1
        try:validate(d,q)
        except ValueError:n+=1
    require(n==3,'corruptions accepted')
    return dict(valid_identity_pass=True,corruptions_refused=n)


def metric(y,p,f):
    y=np.asarray(y,float);p=np.asarray(p,float);f=np.asarray(f,bool);keep=p>0
    dec=pd.qcut(pd.Series(p).rank(method='first'),10,labels=False).to_numpy();top=dec==9
    return dict(n=len(y),filled=int(f.sum()),ic=ic(p,y),pred_mean=float(p.mean()),value_mean=float(y.mean()),
        optimism=float((p-y).mean()),mse=float(np.mean((p-y)**2)),
        top_n=int(top.sum()),top_pred=float(p[top].mean()),top_value=float(y[top].mean()),
        kept_n=int(keep.sum()),kept_value_cents=float(y[keep].sum()))


def run(out):
    require(not out.exists(),'output already exists');checks=selftest();inputs={};folds={}
    def read(p):
        h=sha(p);require(str(p) not in inputs or inputs[str(p)]==h,'changed input');inputs[str(p)]=h
        return pd.read_parquet(p) if p.suffix=='.parquet' else json.loads(p.read_text())
    for t in TESTS:
        fit=read(ROOT/f'fit2_{t}.json');F=fit['features'];tr=fit['train_days'];cal=fit['cal']
        require(max(tr)<cal<t,'chronology')
        heads={}
        for h in ['p','v']:
            a=fit['artifacts']['full_'+h];p=Path(a['path']);inputs[str(p)]=sha(p)
            require(inputs[str(p)]==a['sha256'],'model hash');heads[h]=lgb.Booster(model_file=str(p))
        def panel(day):
            p=ROOT/f'ev_x2_{day}.parquet';d=read(p)
            require(inputs[str(p)]==fit['inputs_sha256'][day],'training input hash');return d
        d=pd.concat([panel(day) for day in tr],ignore_index=True);d=d[d.cert_est].reset_index(drop=True)
        x=d[F].to_numpy(np.float32);y=d.value_cents.to_numpy(float)
        require(hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()==fit['train_matrix_sha256'],'training matrix')
        train=metric(y,heads['p'].predict(x)*heads['v'].predict(x),d.filled_quantity>0)
        require(abs(train['ic']-fit['T6_ic_train'])<1e-12,'training IC')
        require(len(d)==fit['n_train'] and train['filled']==fit['filled_train'],'training counts')
        del x,d,y
        rec=dict(train_days=tr,cal_day=cal,train=train,forward={})
        for day in [cal,t]:
            d=panel(day);p=read(ROOT/f'pred2_{t}_{day}.parquet');validate(d,p);x=d[F].to_numpy(np.float32)
            for h,c in [('p','p_fill'),('v','v_fill')]:
                require(np.array_equal(heads[h].predict(x),p[c]),'saved head prediction')
            q=p[p.cert_est];rec['forward'][day]=metric(q.value_cents,q.pred,q.filled_quantity>0)
            if day==t:
                q=p[p.cert_ws];rec['strict_test']=metric(q.value_cents,q.pred,q.filled_quantity>0)
            del d,p,x,q
        folds[t]=rec
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2048*1024,'RSS cap')
    for p,h in inputs.items():require(sha(p)==h,'input changed during audit')
    result=dict(asof=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),
        controls=checks,input_sha256=inputs,folds=folds,all_saved_head_predictions_exact=True,
        interpretation='Consumed-data description only. In-sample training IC is not validation. No fitting, selected model, threshold or inferential claim.')
    out.parent.mkdir(exist_ok=True,parents=True)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ['asof','controls','folds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
