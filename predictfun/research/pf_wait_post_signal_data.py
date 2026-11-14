"""Consumed-date distribution audit and three frozen local signal probes.

No outcome rows are removed or targets clipped. All preprocessing learns from
earlier training dates only. RAW_DIRECT isolates fitting POST-minus-WAIT;
COMPACT_DIRECT uses economic scales and fewer redundant time windows;
COMPACT_RIDGE uses the same processed inputs and fixed regularization.
The saved v2 models are the control, not refitted. No portfolio/forward claim.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pf_wait_post_diagnose as D
M=D.M; BASE=D.BASE
ARMS=('FROZEN_V2','RAW_DIRECT','COMPACT_DIRECT','COMPACT_RIDGE')
STATE=['rung','price','depth','spread','known_position','own_exposure','other_exposure',
       'time_remaining','frame_age','idle_age_ms','side_price_change_while_empty','shown_known','size_better']
RIDGE_ALPHA=100.


def chronology(train,test_day):
    if not len(train) or not all(str(d)<str(test_day) for d in train.day.unique()):raise ValueError('nonchronological fit')


def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')


def compact(f):
    """Only current row's already-causal inputs. No smoothing across rows/days."""
    g={};sg=f.side_sign.to_numpy();spread=np.maximum(f.spread.to_numpy(),1.)
    sigma=np.where(f.sigma.to_numpy()>1e-8,f.sigma.to_numpy(),np.nan)
    for h in (100,250,1000,3000):
        g['bn_z'+str(h)]=sg*f['bn_r'+str(h)].to_numpy()/(1e4*sigma*np.sqrt(h/1000.))
    g['bn_imb']=sg*f.bn_imb.to_numpy()
    for h in (250,1000,5000):
        v=sg*f['bn_flow'+str(h)].to_numpy();g['flow_log'+str(h)]=np.sign(v)*np.log1p(abs(v))
    for h in (250,1000,5000):g['pf_move_spread'+str(h)]=sg*f['pf_dm'+str(h)].to_numpy()/spread
    for name in ('pf_imb1','pf_imb3'):g[name]=sg*f[name].to_numpy()
    g['micro_spread']=sg*f.pf_micro_dev.to_numpy()/spread
    g['gap_spread']=sg*f.gap.to_numpy()/spread
    g['own_book_log']=np.where(sg>0,f.pf_lbs,f.pf_las)
    g['other_book_log']=np.where(sg>0,f.pf_las,f.pf_lbs)
    for name in ('shown_known','size_better'):
        v=f[name].to_numpy();g[name+'_lots_log']=np.where(v>=0,np.log1p(np.maximum(v,0)/M.SIZE),np.nan)
    for name in ('bn_ntr1000','pf_nfr1000','bn_age','pf_age','frame_age','idle_age_ms'):
        v=f[name].to_numpy();g[name+'_log']=np.where(v>=0,np.log1p(np.maximum(v,0)),np.nan)
    g['price_probability']=f.price.to_numpy()/100.
    g['spread']=spread;g['rung']=f.rung.to_numpy()
    g['position_lots']=sg*f.known_position.to_numpy()/M.SIZE
    for name in ('own_exposure','other_exposure','capacity_left'):g[name+'_lots']=f[name].to_numpy()/M.SIZE
    g['remaining_fraction']=f.time_remaining.to_numpy()/60000.
    g['waiting_price_change_spread']=f.side_price_change_while_empty.to_numpy()/spread
    g['sigma_log']=np.log(sigma);g['basis_bps']=f.basis_bps.to_numpy()
    return pd.DataFrame(g,index=f.index)


class Processor:
    """Training-only bounds/median/IQR; extremes retained via indicator features."""
    def fit(self,frame):
        a=frame.to_numpy(float)
        if np.isinf(a).any():raise ValueError('infinite input')
        self.columns=list(frame.columns);self.stats={};self.output=[];self.kept=[];self.removed={};seen=set()
        for j,n in enumerate(self.columns):
            v=a[:,j];good=v[np.isfinite(v)]
            if not len(good):self.removed[n]='all_missing_in_training';continue
            lo,q1,med,q3,hi=np.quantile(good,[.005,.25,.5,.75,.995])
            scale=max(q3-q1,float(np.std(good))*.1,1e-8)
            self.stats[n]=dict(low=float(lo),median=float(med),high=float(hi),scale=float(scale))
            x=self.one(v,self.stats[n])
            for k,suffix in enumerate(('', '_missing','_low','_high')):
                z=x[:,k];key=z.tobytes();name=n+suffix
                if np.ptp(z)==0:self.removed[name]='constant_in_training';continue
                if key in seen:self.removed[name]='duplicate_in_training';continue
                seen.add(key);self.kept.append((n,k));self.output.append(name)
        if not self.output:raise ValueError('no varying training input')
        return self
    @staticmethod
    def one(v,s):
        missing=~np.isfinite(v);z=np.where(missing,s['median'],v)
        return np.column_stack(((np.clip(z,s['low'],s['high'])-s['median'])/s['scale'],missing,z<s['low'],z>s['high']))
    def transform(self,frame):
        if list(frame.columns)!=self.columns:raise ValueError('input schema changed')
        if np.isinf(frame.to_numpy(float)).any():raise ValueError('infinite input')
        parts={n:self.one(frame[n].to_numpy(float),s) for n,s in self.stats.items()}
        return np.column_stack([parts[n][:,k] for n,k in self.kept]).astype(np.float32)
    def metadata(self):return dict(columns=self.columns,output=self.output,stats=self.stats,removed=self.removed)


def feature_shift(train,test):
    result=[]
    for name in M.FEATURES:
        a=train[name].to_numpy(float);b=test[name].to_numpy(float);aa=a[np.isfinite(a)];bb=b[np.isfinite(b)]
        row=dict(feature=name,train_n=len(a),test_n=len(b),train_missing=int((~np.isfinite(a)).sum()),test_missing=int((~np.isfinite(b)).sum()))
        if len(aa) and len(bb):
            q=np.quantile(aa,[0,.005,.25,.5,.75,.995,1]);scale=max(q[4]-q[2],np.std(aa)*.1,1e-8)
            row.update(train_quantiles=q.tolist(),test_quantiles=np.quantile(bb,[0,.005,.25,.5,.75,.995,1]).tolist(),
                       outside_train_range=int(((bb<q[0])|(bb>q[-1])).sum()),outside_train_central99=int(((bb<q[1])|(bb>q[-2])).sum()),
                       median_shift_iqr=float((np.median(bb)-q[3])/scale),ks_distance=float(ks_2samp(aa,bb,method='asymp').statistic))
        result.append(row)
    return result


def distances(train,test,columns):
    """Reference/calibration split by whole training window, never by action row."""
    keys=train.day.astype(str)+':'+train.market.astype(str)
    buckets=keys.map(lambda z:int.from_bytes(hashlib.sha256(z.encode()).digest()[:4],'big')%2).to_numpy()
    ref=train.loc[buckets==0,columns];cal=train.loc[buckets==1,columns]
    if len(ref)<100 or len(cal)<100:raise ValueError('insufficient whole-window reference/calibration support')
    # Train-only deterministic caps; never sample on any target or prediction.
    ref=ref.sample(min(len(ref),512),random_state=M.SEED);cal=cal.sample(min(len(cal),1024),random_state=M.SEED)
    processor=Processor().fit(ref);xr=processor.transform(ref);xc=processor.transform(cal);xt=processor.transform(test[columns])
    def nearest(x):return cdist(x,xr).min(axis=1)/math.sqrt(xr.shape[1])
    dc=nearest(xc);dt=nearest(xt);threshold=float(np.quantile(dc,.95))
    return dict(reference_n=len(ref),calibration_n=len(cal),test_n=len(test),features=len(processor.output),
                training_calibration_q95=threshold,test_distance_quantiles=np.quantile(dt,[.5,.9,.95,.99]).tolist(),
                outside=int((dt>threshold).sum())),dt>threshold


def metrics(frame,post,delta,mean_post,mean_delta):
    mask=(post>0)&(delta>=0)
    return dict(n=len(frame),post=D.metric(frame.qpost,post,mean_post),delta=D.metric(frame.delta,delta,mean_delta),
                selection=dict(n=int(mask.sum()),predicted_post=float(np.mean(post[mask])) if mask.any() else None,
                               actual_post=float(frame.qpost.to_numpy()[mask].mean()) if mask.any() else None,
                               actual_delta=float(frame.delta.to_numpy()[mask].mean()) if mask.any() else None))


def ridge(x,y,z):
    xm=x.mean(axis=0,dtype=np.float64);ym=float(y.mean());xx=x.astype(float)-xm
    coef=np.linalg.solve(xx.T@xx+RIDGE_ALPHA*np.eye(x.shape[1]),xx.T@(y-ym))
    return (z-xm)@coef+ym,dict(coef=coef.tolist(),x_mean=xm.tolist(),y_mean=ym,alpha=RIDGE_ALPHA)


def audit(out):
    p=M.stage(out,'audit');results=[];runtime=[];receipts={}
    diag=Path('/home/yuqing/ctaNew/data/predict_fun/research/wait_post_diagnosis_20261114_v3')
    for day in M.TESTS:
        train=D.training(day);_,test,receipt=M.read_eval(day,BASE);receipts[day]=receipt['source_status']
        chronology(train,day)
        for side in (0,1):
            tr=train[train.side==side];te=test[test.side==side];support=D.Support(tr)
            counts=np.array([support.check(x)[0] for x in te[M.FEATURES].to_numpy(np.float32)])
            market,market_ood=distances(tr,te,list(M.K.G32));state,state_ood=distances(tr,te,STATE)
            strata=[]
            masks={'state_seen5':counts>=5,'state_sparse':(counts>0)&(counts<5),'state_unseen':counts==0,
                   'market_in_support':~market_ood,'market_outside':market_ood,'state_distance_in_support':~state_ood,'state_distance_outside':state_ood}
            for label,mask in masks.items():
                if mask.any():strata.append(dict(stratum=label,**metrics(te[mask],te.post_hat.to_numpy()[mask],(te.post_hat-te.wait_hat).to_numpy()[mask],tr.qpost.mean(),tr.delta.mean())))
            results.append(dict(day=day,side=side,n_train=len(tr),n_test=len(te),train_dates=list(M.FOLDS[day]),
                                train_initial_generation_fraction=float((tr.generation==0).mean()),test_initial_generation_fraction=float((te.generation==0).mean()),
                                feature_shift=feature_shift(tr,te),market_distance=market,state_distance=state,strata=strata))
        q=diag/('replay_'+day);receipt=json.loads((q/'receipt.json').read_text())
        if M.digest(q/'entries.parquet')!=receipt['files']['entries.parquet']:raise ValueError('runtime population changed')
        e=pd.read_parquet(q/'entries.parquet',filters=[('arm','==','VALUE')])
        for side,g in e.groupby('side'):
            posted=g[g.posted];runtime.append(dict(day=day,side=int(side),generations=len(g),posts=len(posted),
                initial_generation_fraction=float((g.generation==0).mean()),post_unseen=int(posted.post_unseen.sum()),
                post_sparse=int(posted.post_sparse.sum()),post_outside_marginal=int(posted.post_outside.sum()),
                first_unseen=int(g.first_unseen.sum()),first_outside_marginal=int(g.first_outside.sum())))
    write(p/'result.json',dict(asof=M.now(),cells=results,runtime=runtime,source_exclusions=receipts,
          forward_validation=False,limitations=['Nearest-distance cut is training-calibration95%, descriptive only; not a policy gate.','Runtime full population contains order-state support flags, not full market-feature vectors. Biased saved runtime inference samples are not substituted.','Sparse joint bins do not imply every feature is outside training range.']))
    print('Distribution audit complete',M.now(),flush=True)


def probes(out):
    import lightgbm as lgb
    p=M.stage(out,'probes');all_predictions=[];cells=[];metadata=[]
    for day in M.TESTS:
        train=D.training(day);_,test,receipt=M.read_eval(day,BASE)
        for side in (0,1):
            tr=train[train.side==side];te=test[test.side==side];y=tr.qpost.to_numpy();d=tr.delta.to_numpy()
            chronology(tr,day)
            params=dict(M.E.PD.LGB_PARAMS,objective='regression')
            preds={'FROZEN_V2':(te.post_hat.to_numpy(),(te.post_hat-te.wait_hat).to_numpy())}
            model=lgb.train(params,lgb.Dataset(tr[M.FEATURES].to_numpy(np.float32),d,feature_name=M.FEATURES),M.E.PD.N_ROUNDS)
            model.save_model(str(p/f'{day}_{side}_raw_delta.txt'))
            preds['RAW_DIRECT']=(te.post_hat.to_numpy(),model.predict(te[M.FEATURES].to_numpy(np.float32)))
            processor=Processor().fit(compact(tr));x=processor.transform(compact(tr));z=processor.transform(compact(te))
            fitted=[];linear=[];linear_meta={}
            for target,values in (('post',y),('delta',d)):
                model=lgb.train(params,lgb.Dataset(x,values,feature_name=processor.output),M.E.PD.N_ROUNDS)
                model.save_model(str(p/f'{day}_{side}_compact_{target}.txt'));fitted.append(model.predict(z))
                prediction,info=ridge(x,values,z);linear.append(prediction);linear_meta[target]=info
            preds['COMPACT_DIRECT']=tuple(fitted);preds['COMPACT_RIDGE']=tuple(linear)
            metadata.append(dict(day=day,side=side,train_dates=list(M.FOLDS[day]),train_n=len(tr),processor=processor.metadata(),ridge=linear_meta))
            for arm,(post,delta) in preds.items():
                cells.append(dict(day=day,side=side,arm=arm,**metrics(te,post,delta,y.mean(),d.mean()),
                                  price_rung_post=D.metric(te.qpost,D.baseline(tr,te,'qpost'),y.mean()),
                                  price_rung_delta=D.metric(te.delta,D.baseline(tr,te,'delta'),d.mean())))
                frame=te[['day','market','side','rung','generation','probe_t','qpost','qwait','delta']].copy()
                frame['arm']=arm;frame['post_hat']=post;frame['delta_hat']=delta;frame['post_action']=(post>0)&(delta>=0);all_predictions.append(frame)
        print('Local signal probes',day,'complete',M.now(),flush=True)
    pred=pd.concat(all_predictions,ignore_index=True);pred.to_parquet(p/'predictions.parquet',index=False)
    nulls=[]
    for (arm,side),g in pred.groupby(['arm','side']):nulls.append(dict(arm=arm,side=int(side),**M.timing_null(g,500)))
    write(p/'result.json',dict(asof=M.now(),cells=cells,matched_tests=nulls,params=params,rounds=M.E.PD.N_ROUNDS,
                             candidate_count=3,reference='FROZEN_V2',rows_sha256=M.digest(p/'predictions.parquet'),
                             preprocessing=metadata,forward_validation=False,new_portfolio_candidate_accepted=False,
                             limitations=['Same consumed dates and frozen neutral-KEEP labels, so this cannot validate actual combined keeper/entry performance.','No inverted-policy arm, target clipping, fill conditioning or learned test-set filter.','All three candidates reported, no post hoc threshold tuning or promotion.']))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['selftest','audit','probes']);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.command=='selftest':
        import unittest,test_wait_post_signal_data
        r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_wait_post_signal_data))
        if not r.wasSuccessful():raise SystemExit(1)
    elif a.command=='audit':audit(a.out)
    else:probes(a.out)


if __name__=='__main__':main()
