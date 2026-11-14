"""Correct known hash-sampling probabilities; retain every eligible action.

First generations were selected with probability1/2, later ones1/32. This
follow-up weights both fitting and evaluation by their inverse probabilities.
It corrects that sampling design, not policy-state coverage or censoring.
All earlier probes are reevaluated on the same weighted population. Two new
fixed candidates fit raw POST and POST-minus-WAIT; no threshold tuning.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import pf_wait_post_signal_data as S
M=S.M;D=S.D;BASE=S.BASE
PHASE1=Path('/home/yuqing/ctaNew/data/predict_fun/research/wait_post_signal_data_20261114_v1')
ARMS=('IPW_RAW_DIRECT','IPW_COMPACT_DIRECT')


def weights(frame):
    g=frame.generation.to_numpy(float)
    if not np.isfinite(g).all() or (g<0).any() or (g!=np.floor(g)).any():raise ValueError('invalid generation')
    return np.where(g==0,2.,32.)


def effective(w):return float(w.sum()**2/np.sum(w*w))


def metric(y,p,w,baseline):
    y=np.asarray(y,float);p=np.asarray(p,float);w=np.asarray(w,float)
    if not len(y) or y.shape!=p.shape or y.shape!=w.shape or not np.isfinite([y,p,w]).all() or (w<=0).any():raise ValueError('invalid weighted population')
    sse=float(np.sum(w*(p-y)**2));den=float(np.sum(w*(y-baseline)**2));mass=float(w.sum())
    return dict(n=len(y),weight_mass=mass,effective_n=effective(w),actual_mean=float(np.average(y,weights=w)),predicted_mean=float(np.average(p,weights=w)),
                rmse=float(np.sqrt(sse/mass)),sse=sse,baseline_sse=den,r2_vs_prior_mean=1-sse/den if den else None)


def baseline(train,test,target):
    a=train[['rung','price',target]].copy();a['weight']=weights(train);a['value']=a['weight']*a[target];a['pricebin']=(a.price//10).astype(int)
    rung=a.groupby('rung')[['weight','value']].sum();cells=a.groupby(['rung','pricebin']).agg(weight=('weight','sum'),value=('value','sum'),n=('weight','size'))
    whole=a.value.sum()/a.weight.sum();out=[]
    for r,p in zip(test.rung,(test.price//10).astype(int)):
        if (r,p) in cells.index and cells.loc[(r,p),'n']>=100:v=cells.loc[(r,p),'value']/cells.loc[(r,p),'weight']
        elif r in rung.index:v=rung.loc[r,'value']/rung.loc[r,'weight']
        else:v=whole
        out.append(float(v))
    return np.asarray(out)


def matched(frame,draws=500):
    if draws<200:raise ValueError('too few matched draws')
    f=frame.copy();w=weights(f);selected=f.post_action.to_numpy(bool)
    if not np.array_equal(w,f.sampling_weight.to_numpy()):raise ValueError('wrong sampling weight')
    count=int(selected.sum());mass=float(w[selected].sum());eff=effective(w[selected]) if count else 0.
    base=dict(n=len(f),posts=count,selected_weight_mass=mass,selected_effective_n=eff,draws=draws)
    if min(count,len(f)-count)<30 or eff<30:return dict(**base,passes=False,status='insufficient_action_variation')
    obs=float(np.average(f.delta.to_numpy()[selected],weights=w[selected]));absolute=float(np.average(f.qpost.to_numpy()[selected],weights=w[selected]))
    f['hour']=(f.probe_t//3600000).astype('int64');f['initial']=f.generation==0
    rng=np.random.default_rng(M.SEED);null=np.zeros(draws);matched_mass=0.
    for _,g in f.groupby(['day','side','hour','rung','initial']):
        k=int(g.post_action.sum());wg=g.sampling_weight.to_numpy()
        if np.ptp(wg)!=0:raise ValueError('null stratum mixes sampling probabilities')
        if not k:continue
        # Uniform k-subsets; vectorized across draws, same count/weight in every stratum.
        choices=np.argsort(rng.random((draws,len(g))),axis=1)[:,:k]
        null+=g.delta.to_numpy()[choices].sum(axis=1)*wg[0];matched_mass+=k*wg[0]
    if matched_mass!=mass:raise ValueError('weighted selection mass differs')
    null/=mass;p95=float(np.quantile(null,.95))
    return dict(**base,status='matched_weighted_action_value',observed_delta=obs,observed_absolute_post=absolute,null_p95=p95,
                passes=bool(obs>p95+1e-12 and obs>0 and absolute>0))


def probes(out):
    import lightgbm as lgb
    p=M.stage(out,'probes');prior=json.loads((PHASE1/'probes/result.json').read_text())
    if M.digest(PHASE1/'probes/predictions.parquet')!=prior['rows_sha256']:raise ValueError('phase1 predictions changed')
    controls=pd.read_parquet(PHASE1/'probes/predictions.parquet');parts=[];cells=[];support=[];transforms=[]
    params=dict(M.E.PD.LGB_PARAMS,objective='regression')
    for day in M.TESTS:
        train=D.training(day);_,test,receipt=M.read_eval(day,BASE)
        for side in (0,1):
            tr=train[train.side==side];te=test[test.side==side];S.chronology(tr,day)
            w=weights(tr);wt=weights(te);fit_weight=w/w.mean()
            means={h:float(np.average(tr[t],weights=w)) for h,t in (('post','qpost'),('delta','delta'))}
            processor=S.Processor().fit(S.compact(tr));x=processor.transform(S.compact(tr));z=processor.transform(S.compact(te))
            transforms.append(dict(day=day,side=side,processor=processor.metadata(),train_dates=list(M.FOLDS[day])))
            for split,f,ww in (('train',tr,w),('test',te,wt)):
                support.append(dict(day=day,side=side,split=split,n=len(f),effective_n=effective(ww),initial_fraction=float((f.generation==0).mean()),weighted_initial_fraction=float(ww[f.generation.to_numpy()==0].sum()/ww.sum())))
            forecasts={}
            for arm in ARMS:
                xx,zz,names=(tr[M.FEATURES].to_numpy(np.float32),te[M.FEATURES].to_numpy(np.float32),M.FEATURES) if arm=='IPW_RAW_DIRECT' else (x,z,processor.output)
                pred=[]
                for head,target in (('post','qpost'),('delta','delta')):
                    model=lgb.train(params,lgb.Dataset(xx,tr[target].to_numpy(),weight=fit_weight,feature_name=names),M.E.PD.N_ROUNDS)
                    model.save_model(str(p/f'{day}_{side}_{arm}_{head}.txt'));pred.append(model.predict(zz))
                forecasts[arm]=tuple(pred)
            for arm,g in controls[(controls.day==day)&(controls.side==side)].groupby('arm'):
                keys=['day','market','side','rung','generation']
                if not np.array_equal(g[keys].to_numpy(),te[keys].to_numpy()) or not np.allclose(g.qpost,te.qpost,atol=0,rtol=0):raise ValueError('control action identity differs')
                forecasts[arm]=(g.post_hat.to_numpy(),g.delta_hat.to_numpy())
            for arm,(post,delta) in forecasts.items():
                selected=(post>0)&(delta>=0);selw=wt[selected]
                cells.append(dict(day=day,side=side,arm=arm,n=len(te),
                    post=metric(te.qpost,post,wt,means['post']),delta=metric(te.delta,delta,wt,means['delta']),
                    baseline_post=metric(te.qpost,baseline(tr,te,'qpost'),wt,means['post']),
                    baseline_delta=metric(te.delta,baseline(tr,te,'delta'),wt,means['delta']),
                    selection=dict(n=int(selected.sum()),weight_mass=float(selw.sum()),
                        actual_post=float(np.average(te.qpost.to_numpy()[selected],weights=selw)) if len(selw) else None,
                        actual_delta=float(np.average(te.delta.to_numpy()[selected],weights=selw)) if len(selw) else None)))
                f=te[['day','market','side','rung','generation','probe_t','qpost','qwait','delta']].copy()
                f['arm']=arm;f['post_hat']=post;f['delta_hat']=delta;f['post_action']=selected;f['sampling_weight']=wt;parts.append(f)
        print('Sampling correction',day,'complete',M.now(),flush=True)
    predictions=pd.concat(parts,ignore_index=True);predictions.to_parquet(p/'predictions.parquet',index=False)
    tests=[dict(arm=a,side=int(s),**matched(g)) for (a,s),g in predictions.groupby(['arm','side'])]
    S.write(p/'result.json',dict(asof=M.now(),cells=cells,matched_tests=tests,support=support,preprocessing=transforms,
        rows_sha256=M.digest(p/'predictions.parquet'),phase1_result_sha256=M.digest(PHASE1/'probes/result.json'),params=params,rounds=M.E.PD.N_ROUNDS,
        new_candidates=2,cumulative_candidates=5,forward_validation=False,portfolio_evaluated=False,
        limitations=['Inverse inclusion weighting corrects only the1/2 versus1/32 neutral-generation sampling, not missing joint-policy states or unreached probes.',
                     'Weighted mass estimates representation; it is not an observed action count. Report actual n and effective n.',
                     'Training-fitted preprocessing is unchanged to isolate weighting; bounds are not refitted on test or runtime data.',
                     'All six arms reevaluated with identical weights and generation-matched null; weighted results cannot be directly compared with phase1 unweighted results.',
                     'Targets remain frozen neutral-KEEP continuation values. All days consumed; no portfolio promotion or forward claim.']))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['selftest','probes']);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.command=='selftest':
        import unittest,test_wait_post_sampling
        r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_wait_post_sampling))
        if not r.wasSuccessful():raise SystemExit(1)
    else:probes(a.out)


if __name__=='__main__':main()
