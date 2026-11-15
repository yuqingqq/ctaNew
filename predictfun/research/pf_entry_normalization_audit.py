"""Audit saved entry inputs/preprocessors; no historical model fitting or selection.

Units, side signs, denominator validity, causal liquidity support, distribution
scaling, sampling-weight alignment and discarded preprocessing information.
The component study is on hold. All outcomes and frozen models are unchanged.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pf_entry_information as I
import pf_wait_post_state_clean as C

N=I.N
BASE=Path(N.M.K.IO)/'entry_information_20261115_v3'
CLEAN=Path(N.M.K.IO)/'wait_post_state_clean_20261115_v1/probes'
OUT=Path(N.M.K.IO)/'entry_normalization_20261115_v1'


def wquantile(x,w,q):
    x=np.asarray(x,float);w=np.asarray(w,float);q=np.asarray(q,float)
    if x.shape!=w.shape or np.any(w<=0) or not np.isfinite(w).all() or np.any((q<0)|(q>1)):
        raise ValueError('invalid quantile population')
    good=np.isfinite(x);x=x[good];w=w[good]
    if not len(x):return np.full(len(q),np.nan)
    order=np.argsort(x,kind='stable');x=x[order];w=w[order]
    c=(np.cumsum(w)-.5*w)/w.sum()
    return np.interp(q,c,x)


def rate(mask,w):
    mask=np.asarray(mask,bool);w=np.asarray(w,float)
    return dict(n=len(mask),affected_n=int(mask.sum()),fraction=float(mask.mean()),weighted_fraction=float(np.average(mask,weights=w)))


def inspect_processor(p,tr,te,w,wt):
    if list(tr.columns)!=p.columns or list(te.columns)!=p.columns:
        raise ValueError('processor schema mismatch')
    trainparts={n:p.one(tr[n].to_numpy(float),s) for n,s in p.stats.items()}
    testparts={n:p.one(te[n].to_numpy(float),s) for n,s in p.stats.items()}
    kept={trainparts[n][:,j].tobytes():(name,n,j) for (n,j),name in zip(p.kept,p.output)}
    missing=np.zeros(len(te),bool);extreme=np.zeros(len(te),bool);duplicate=np.zeros(len(te),bool)
    losses=[];scales=[]
    for n in p.columns:
        a=tr[n].to_numpy(float);b=te[n].to_numpy(float)
        if n not in p.stats:
            bad=np.isfinite(b)
            if bad.any():losses.append(dict(feature=n,kind='training_all_missing_now_present',**rate(bad,wt)))
            continue
        s=p.stats[n];bounds=(b<s['low'])|(b>s['high'])
        wq=wquantile(a,w,[.005,.25,.5,.75,.995])
        scales.append(dict(feature=n,train_n=len(tr),test_n=len(te),train_missing=int((~np.isfinite(a)).sum()),test_missing=int((~np.isfinite(b)).sum()),
            unweighted_median=s['median'],weighted_median=float(wq[2]),median_shift_in_current_scale=float((wq[2]-s['median'])/s['scale']),
            unweighted_low=s['low'],weighted_low=float(wq[0]),unweighted_high=s['high'],weighted_high=float(wq[4]),
            test_clipping=rate(bounds,wt),train_clipping=rate((a<s['low'])|(a>s['high']),w)))
        for j,suffix in enumerate(('', '_missing','_low','_high')):
            name=n+suffix
            if name in p.output:continue
            reason=p.removed.get(name)
            if reason=='constant_in_training':
                bad=testparts[n][:,j]!=trainparts[n][0,j];alias=None
            elif reason=='duplicate_in_training':
                alias,nn,jj=kept[trainparts[n][:,j].tobytes()]
                bad=~np.isclose(testparts[n][:,j],testparts[nn][:,jj],atol=1e-10,rtol=1e-10)
            else:continue
            if bad.any():
                losses.append(dict(feature=name,kind=reason,kept_alias=alias,component=j,**rate(bad,wt)))
                if j==1:missing|=bad
                if j in (2,3):extreme|=bad
                if reason=='duplicate_in_training':duplicate|=bad
    return dict(lost_missing_indicator=rate(missing,wt),lost_extreme_indicator=rate(extreme,wt),
                diverged_training_duplicates=rate(duplicate,wt),losses=losses,scales=scales)


def unit_checks(f):
    checks={}
    def record(name,ok):
        a=np.asarray(ok,bool);checks[name]=int((~a).sum())
    side=f.side.to_numpy();sg=np.where(side==0,1.,-1.)
    record('side_sign',f.side_sign.to_numpy()==sg)
    for n in N.M.V1.DIRECTIONAL:
        record('signed_'+n,np.isclose(f['side_'+n],sg*f[n],equal_nan=True,atol=1e-7,rtol=1e-6))
    scale=np.expm1(f.own_book_liquidity_log.to_numpy(float))
    record('positive_liquidity_scale',np.isfinite(scale)&(scale>0))
    record('same_joint_scale_both_sides',np.isclose(f.own_book_liquidity_log,f.opp_book_liquidity_log,atol=1e-7,rtol=1e-6))
    record('positive_sigma',np.isfinite(f.sigma)&(f.sigma>0))
    record('valid_spread',np.isfinite(f.spread)&(f.spread>=1))
    record('price_cents',np.isfinite(f.price)&(f.price>=1)&(f.price<=99))
    record('known_reference_mask',(f.start_known!=0)|f.gap.isna())
    record('contract_time_units',np.isclose(f.contract_remaining_s,300.-f.t_rel,atol=3e-5,rtol=0))
    record('cutoff_time_units',np.isclose(f.time_remaining,1000.*(60.-f.t_rel),atol=.01,rtol=0))
    for n in ('post_fill_fraction','wait_fill_fraction'):
        record(n+'_bounded',np.isfinite(f[n])&(f[n]>=-1e-8)&(f[n]<=1+1e-8))
    compact=C.compact(f)
    record('position_lots',np.isclose(compact.position_lots,sg*f.known_position/N.M.SIZE,atol=1e-9,rtol=1e-6))
    for h in (100,250,1000,3000):
        expected=sg*f['bn_r'+str(h)].to_numpy()/(1e4*f.sigma.to_numpy()*np.sqrt(h/1000.))
        record('return_bps_sigma_'+str(h),np.isclose(compact['bn_z'+str(h)],expected,equal_nan=True,atol=1e-9,rtol=1e-6))
    record('infinite_compact',~np.isinf(compact.to_numpy(float)).any(axis=1))
    N.check_targets(f)
    return checks


def scaled_views(f):
    scale=np.expm1(f.own_book_liquidity_log.to_numpy(float))
    values={}
    public=['shown_known','size_better']+I.GEOMETRY[8:]+[n for n in I.DYNAMICS if n.startswith('candidate_')]
    for n in public:
        v=f[n].to_numpy(float)/(N.M.SIZE if n in ('shown_known','size_better') else 1.)
        values[n]=(v,v/scale,'known_liquidity_ratio')
    for h in (100,250,1000,3000):
        n='bn_r'+str(h);v=f[n].to_numpy(float)
        values[n]=(v,v/(1e4*f.sigma.to_numpy()*np.sqrt(h/1000.)),'known_volatility_ratio')
    return values


def shift_pair(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);aa=a[np.isfinite(a)];bb=b[np.isfinite(b)]
    if not len(aa) or not len(bb):return dict(train_valid=len(aa),test_valid=len(bb),ks=None)
    return dict(train_valid=len(aa),test_valid=len(bb),ks=float(ks_2samp(aa,bb,method='asymp').statistic),
                train_median=float(np.median(aa)),test_median=float(np.median(bb)))


def audit(out):
    out.mkdir();saved=N.E.read_json(CLEAN/'result.json')
    meta={(r['day'],r['side']):r for r in saved['preprocessing']}
    allrows={d:I.load(BASE,d) for d in N.DAYS}
    units=[];population=[];days=[];processors=[];scales=[];losses=[];shifts=[];actionflags=[]
    for day,f in allrows.items():
        checks=unit_checks(f);units.append(dict(day=day,n=len(f),violations=checks))
        for side,g in f.groupby('side'):
            w=N.S.weights(g);s=np.expm1(g.own_book_liquidity_log.to_numpy(float))*N.M.SIZE
            population.append(dict(day=day,side=int(side),n=len(g),windows=g.market.nunique(),
                scale_shares_quantiles=dict(zip(('p05','p50','p95'),map(float,np.quantile(s,[.05,.5,.95])))),
                sigma_quantiles=dict(zip(('p05','p50','p95'),map(float,np.quantile(g.sigma,[.05,.5,.95])))),
                before25s=rate(g.t_rel.to_numpy()<25.,w)))
    for day in N.M.TESTS:
        train=pd.concat([allrows[d] for d in N.M.FOLDS[day]],ignore_index=True);test=allrows[day]
        for side in (0,1):
            tr=train[train.side==side];te=test[test.side==side];w=N.S.weights(tr);wt=N.S.weights(te)
            a=C.compact(tr);b=C.compact(te);p=N.S.S.Processor().fit(a);old=meta[day,side]
            if json.dumps(p.metadata(),sort_keys=True)!=json.dumps(old['processor'],sort_keys=True) or [list(x) for x in p.kept]!=old['kept']:
                raise ValueError('saved preprocessing does not reproduce')
            before=copy.deepcopy(p.metadata());p.transform(b)
            if p.metadata()!=before:raise ValueError('test transformation mutated fitted state')
            z=inspect_processor(p,a,b,w,wt)
            processors.append(dict(day=day,side=side,train_n=len(tr),test_n=len(te),train_dates=list(N.M.FOLDS[day]),saved_exact=True,
                initial_row_fraction=float((tr.generation==0).mean()),initial_weight_fraction=float(np.average(tr.generation==0,weights=w)),
                **{k:v for k,v in z.items() if k not in ('scales','losses')}))
            scales.extend(dict(day=day,side=side,**v) for v in z['scales'])
            losses.extend(dict(day=day,side=side,**v) for v in z['losses'])
            at=scaled_views(tr);bt=scaled_views(te)
            for n,(raw,norm,kind) in at.items():
                original=shift_pair(raw,bt[n][0]);normalized=shift_pair(norm,bt[n][1])
                shifts.append(dict(day=day,side=side,feature=n,kind=kind,raw=original,normalized=normalized,
                    ks_reduced=bool(normalized['ks']<original['ks']) if original['ks'] is not None and normalized['ks'] is not None else None))
            print(day,side,'saved preprocessing reproduced',len(z['losses']),'discarded representations vary later',flush=True)
            if N.E.rss_mib()>2048:raise RuntimeError('RSS self-guard exceeded')
    test_n=sum(len(allrows[d]) for d in N.M.TESTS)
    result=dict(asof=N.M.now(),status='complete_normalization_audit',new_candidates=0,model_refits=0,forward_validation=False,
       source_rows={d:N.E.read_json(BASE/('rows_'+d)/'receipt.json') for d in N.DAYS},n_test_actions=test_n,
       label_probes=sum(len(f) for f in allrows.values()),test_dates=list(N.M.TESTS),unit_checks=units,populations=population,
       processors=processors,preprocessing_losses=losses,scaling_stats=scales,distribution_shifts=shifts,
       saved_cleanup_result_sha256=N.M.digest(CLEAN/'result.json'),
       limits=['Read-only development audit. No new historical model fit, economic selection or significance test.',
          'Lost flags/duplicates concern CLEAN_TREE/CLEAN_RIDGE preprocessing, not the raw 325-feature information study.',
          'A flag constant in training cannot teach a fitted response; preserving it supports later missingness/shift monitoring.',
          'Unweighted scaling is train-only, but differs from the inverse-inclusion weighted modeling population.',
          'Distribution-shift reduction is not a prediction or profitability improvement.',
          'The 30-second liquidity median is frame-weighted and uses only supplied frames; before 25 seconds the loader offers under 30 seconds of history.',
          'Fixed share/lot and bps/fraction conversions differ from state-dependent liquidity/volatility normalization.',
          'Raw economic labels retained; no label normalization, clipping, changed target or component study.'])
    N.write(out/'result.json',result)
    pd.DataFrame(losses).to_csv(out/'discarded_information.csv',index=False)
    pd.DataFrame([{k:v for k,v in x.items() if k not in ('test_clipping','train_clipping')}|{'test_clipped_n':x['test_clipping']['affected_n'],'test_clipped_weight_fraction':x['test_clipping']['weighted_fraction']} for x in scales]).to_csv(out/'scaling_stats.csv',index=False)
    pd.DataFrame([dict(day=x['day'],side=x['side'],feature=x['feature'],kind=x['kind'],raw_ks=x['raw']['ks'],normalized_ks=x['normalized']['ks'],ks_reduced=x['ks_reduced']) for x in shifts]).to_csv(out/'distribution_shift.csv',index=False)


class Checks(unittest.TestCase):
    def test_missing_and_extreme_flags(self):
        tr=pd.DataFrame(dict(x=np.arange(100.),discrete=np.tile([0.,1.],50)))
        te=tr.copy();te.loc[0,'x']=np.nan;te.loc[1,'discrete']=100.
        p=N.S.S.Processor().fit(tr);r=inspect_processor(p,tr,te,np.ones(100),np.ones(100))
        self.assertEqual(r['lost_missing_indicator']['affected_n'],1)
        self.assertEqual(r['lost_extreme_indicator']['affected_n'],1)
        clean=inspect_processor(p,tr,tr,np.ones(100),np.ones(100))
        self.assertEqual(clean['lost_missing_indicator']['affected_n'],0)
        with self.assertRaises(ValueError):inspect_processor(p,tr,te[['x']],np.ones(100),np.ones(100))

    def test_accidental_duplicate(self):
        tr=pd.DataFrame(dict(x=np.arange(100.),y=np.arange(100.)));te=tr.copy();te.y=99-te.y
        p=N.S.S.Processor().fit(tr);r=inspect_processor(p,tr,te,np.ones(100),np.ones(100))
        self.assertGreater(r['diverged_training_duplicates']['affected_n'],90)
        self.assertEqual(inspect_processor(p,tr,tr,np.ones(100),np.ones(100))['diverged_training_duplicates']['affected_n'],0)

    def test_weighted_quantiles(self):
        a=np.r_[np.zeros(90),np.ones(10)*100];w=np.r_[np.ones(90)*2,np.ones(10)*32]
        self.assertEqual(float(np.median(a)),0.)
        self.assertGreater(wquantile(a,w,[.5])[0],90.)
        with self.assertRaises(ValueError):wquantile(a,-w,[.5])

    def test_train_only(self):
        a=pd.DataFrame(dict(x=np.arange(100.)))
        p=N.S.S.Processor().fit(a);meta=copy.deepcopy(p.metadata());p.transform(a*1e9)
        self.assertEqual(meta,p.metadata());self.assertNotEqual(meta,N.S.S.Processor().fit(a*1e9).metadata())

    def test_positive_unit_scale_vs_economic_scale(self):
        a=np.arange(1.,101.)
        self.assertAlmostEqual(shift_pair(a,a*3)['ks'],shift_pair(a/30,a*3/30)['ks'])
        self.assertGreater(shift_pair(a,a*3)['ks'],.5)
        self.assertEqual(shift_pair(a/10,a*3/30)['ks'],0.)
        shares=90.;size=30.;self.assertEqual(shares/size,3.)
        bps=1e4*.001;sig=.001;self.assertEqual(bps/(1e4*sig),1.)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['selftest','audit']);p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args()
    if a.command=='selftest':
        r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
        if not r.wasSuccessful():raise SystemExit(1)
    else:audit(a.out)
