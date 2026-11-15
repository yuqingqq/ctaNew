"""Condition-preserving entry representations, with frozen action targets.

RAW reuses the exact saved ALL model. RELATIVE changes public size coordinates
but retains the causal liquidity scale and all own risk quantities. CONDITION
adds known history support, volatility units and fair-price response context.
No feature clipping, empirical duplicate removal, target edits or row filtering.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_entry_information as I
import pf_entry_normalization_audit as A
N = I.N
BASE = Path(N.M.K.IO) / 'entry_information_20261115_v3'
ARMS = ('RAW', 'RELATIVE', 'CONDITION')
# Divisors convert each original quantity to lots before liquidity scaling.
PUBLIC = {n: N.M.SIZE for n in ('shown_known', 'size_better', 'ahead_known', 'initial_ahead_known',
                               'pf_pend_bid1000', 'pf_pend_ask1000', 'pf_pend_tot5000')}
PUBLIC.update({n: 1. for n in I.GEOMETRY[8:] + [n for n in I.DYNAMICS if n.startswith('candidate_')]})
PUBLIC.update({f'{s}_{d}_{n}': 1. for s in ('own', 'opp') for d in range(N.M.NR) for n in ('shown_lots', 'queue_lots')})
SUPPORT = ['liquidity_history_span_ms', 'liquidity_history_available_ms', 'liquidity_history_frames',
           'liquidity_history_max_gap_ms', 'liquidity_history_full', 'liquidity_history_age_ms']
HORIZONS = (25, 50, 100, 250, 500, 1000, 3000)
CONTEXT = SUPPORT + [f'entry_bn_z{h}' for h in HORIZONS]
CONTEXT += ['entry_fair_edge_ticks', 'entry_fair_edge_spreads', 'entry_candidate_size_zero']
CONTEXT += [f'entry_{name}{h}' for h in (100, 250, 1000) for name in ('fair_move_ticks', 'edge_plus_move_ticks')]
CONTEXT += [f'entry_{name}{h}' for h in (250, 1000) for name in ('candidate_history_known', 'shrink_fraction')]
FEATURES = {'RAW': I.FEATURES,
            'RELATIVE': [n + '_liquidity_ratio' if n in PUBLIC else n for n in I.FEATURES]}
FEATURES['CONDITION'] = FEATURES['RELATIVE'] + CONTEXT


def history(W, row):
    """Actual support of the existing frame-weighted scale; no future frames."""
    t = float(row['probe_t']); side = int(row['side']); w = W[side]
    known = np.asarray(w['kf'], float)
    if not np.array_equal(known, W[1-side]['kf']) or np.any(np.diff(known) < 0):
        raise ValueError('unaligned/unsorted received book clocks')
    j = int(np.searchsorted(known, t, side='right')) - 1
    if j < 0: raise ValueError('no known frame')
    start = int(np.searchsorted(known, known[j] - 30000., side='left'))
    seg = known[start:j+1]
    values = [known[j]-seg[0], min(30000., known[j]-known[0]), len(seg),
              float(np.diff(seg).max()) if len(seg)>1 else 0., float(known[0]<=known[j]-30000.), t-known[j]]
    both = np.r_[W[0]['S5'][start:j+1, 0], W[1]['S5'][start:j+1, 0]]
    both = both[np.isfinite(both) & (both > 0)]
    if not len(both): raise ValueError('no positive known liquidity')
    scale = np.log1p(np.median(both) / N.M.SIZE)
    if 'own_book_liquidity_log' in row and not np.isclose(scale, row['own_book_liquidity_log'], atol=1e-7, rtol=1e-6):
        raise ValueError('support/scale does not match frozen row')
    return np.asarray(values, np.float32)


def future_check(W, row, expected):
    changed = copy.deepcopy(W); t = row['probe_t']
    for s in (0, 1):
        k = changed[s]['kf'] > t
        changed[s]['kf'][k] += 987654.
        changed[s]['S5'][k] = 999999.
    np.testing.assert_equal(history(changed, row), expected)
    bad = dict(row, own_book_liquidity_log=-1.)
    try: history(W, bad)
    except ValueError: pass
    else: raise ValueError('planted scale mismatch escaped')


def represent(f, arm):
    if arm not in ARMS or not set(I.FEATURES).issubset(f.columns): raise ValueError('bad representation schema')
    g = f[I.FEATURES].copy()
    scale = np.expm1(f.own_book_liquidity_log.to_numpy(float))
    if np.any(~np.isfinite(scale) | (scale <= 0)): raise ValueError('invalid liquidity denominator')
    if arm != 'RAW':
        for n, divisor in PUBLIC.items(): g[n] = f[n].to_numpy(float) / divisor / scale
        g = g.rename(columns={n:n+'_liquidity_ratio' for n in PUBLIC})
    if arm == 'CONDITION':
        if not set(SUPPORT).issubset(f.columns): raise ValueError('missing history support')
        sig = f.sigma.to_numpy(float); sign = f.side_sign.to_numpy(float)
        if np.any(~np.isfinite(sig) | (sig<=0)) or not np.isin(sign, [-1.,1.]).all(): raise ValueError('invalid volatility/side')
        x = {n:f[n].to_numpy(float) for n in SUPPORT}
        for h in HORIZONS: x[f'entry_bn_z{h}'] = sign*f['bn_r'+str(h)].to_numpy(float)/(1e4*sig*np.sqrt(h/1000.))
        # Fair is an assumption-based, known-time probability; unknown reference stays missing.
        edge = sign*(f.pf_mid.to_numpy()+f.gap.to_numpy()-f.price.to_numpy())
        edge = np.where(f.start_known.to_numpy()!=0, edge, np.nan)
        x['entry_fair_edge_ticks'] = edge
        if np.any(f.spread.to_numpy()<=0): raise ValueError('invalid spread')
        x['entry_fair_edge_spreads'] = edge/f.spread.to_numpy()
        x['entry_candidate_size_zero'] = (f.shown_known.to_numpy()==0).astype(float)
        for h in (100,250,1000):
            move = sign*f.fair_sensitivity.to_numpy()*f['bn_r'+str(h)].to_numpy()
            move = np.where(f.start_known.to_numpy()!=0, move, np.nan)
            x[f'entry_fair_move_ticks{h}'] = move
            x[f'entry_edge_plus_move_ticks{h}'] = edge+move
        for h in (250,1000):
            x[f'entry_candidate_history_known{h}'] = np.isfinite(f[f'candidate_change{h}_lots']).astype(float)
            shown = f.shown_known.to_numpy(float)/N.M.SIZE
            x[f'entry_shrink_fraction{h}'] = np.divide(f[f'candidate_shrink{h}_lots'].to_numpy(),shown,
                out=np.full(len(f),np.nan),where=np.isfinite(shown)&(shown>0))
        g = pd.concat([g, pd.DataFrame(x,index=f.index)[CONTEXT]],axis=1)
    if list(g.columns)!=FEATURES[arm] or np.isinf(g.to_numpy()).any(): raise ValueError('invalid transformed feature')
    return g


def build(day, out):
    p=N.M.stage(out,'rows_'+day); rows=I.load(BASE,day).copy(); rows[SUPPORT]=np.nan
    receipt=N.E.read_json(BASE/('rows_'+day)/'receipt.json'); allowed=set(N.E.read_json(I.BASE/('rows_'+day)/'receipt.json')['windows'])
    data=N.M.K.load_day(day); data['W0']={m:t for m,t in data['W0'].items() if m in allowed}
    groups=rows.groupby('market',sort=False).indices; seen=set(); visited=[]; checks=0
    for W in N.E.PAIRS.window_pairs(data):
        m=W[0]['m']; visited.append(m)
        for number,idx in enumerate(groups.get(m,[])):
            row=rows.iloc[idx]; v=history(W,row); rows.loc[rows.index[idx],SUPPORT]=v
            if number==0: future_check(W,row,v); checks+=1
            seen.add(idx)
        if len(visited)%64==0: print(day,len(visited),'windows',len(seen),'actions',flush=True)
        I.check_memory()
    if seen!=set(range(len(rows))) or len(set(visited))!=len(visited) or set(visited)!=allowed: raise ValueError('incomplete action/window join')
    original=I.load(BASE,day); pd.testing.assert_frame_equal(rows[original.columns],original)
    for arm in ARMS: represent(rows,arm)
    rows.to_parquet(p/'rows.parquet',index=False)
    N.write(p/'receipt.json',dict(asof=N.M.now(),day=day,n=len(rows),windows=len(visited),future_checks=checks,
        source_receipt_sha256=N.M.digest(BASE/('rows_'+day)/'receipt.json'),source_status=receipt['source_status'],
        statuses=receipt['statuses'],sealed_lookback_windows=receipt['sealed_lookback_windows'],rows_sha256=N.M.digest(p/'rows.parquet')))


def load(day,out):
    p=out/('rows_'+day); r=N.E.read_json(p/'receipt.json')
    if r['day']!=day or N.M.digest(p/'rows.parquet')!=r['rows_sha256'] or N.M.digest(BASE/('rows_'+day)/'receipt.json')!=r['source_receipt_sha256']:
        raise ValueError('changed row lineage')
    f=pd.read_parquet(p/'rows.parquet'); old=I.load(BASE,day); pd.testing.assert_frame_equal(f[old.columns],old)
    if len(f)!=r['n'] or f[SUPPORT].isna().any().any(): raise ValueError('missing support rows')
    return f


def strata(tr,te):
    w=N.S.weights(tr)
    cuts={n:float(A.wquantile(tr[n],w,[.5])[0]) for n in ('own_book_liquidity_log','sigma','fair_distance50')}
    result={n:np.where(te[n].isna(),'unknown',np.where(te[n]<=c,'low','high')) for n,c in cuts.items()}
    result.update(initial=np.where(te.generation==0,'initial','later'),
        history=np.where(te.liquidity_history_full>0,'full','partial'),
        reference=np.where(te.start_known>0,'known','unknown'),
        candidate_history=np.where(te.candidate_change250_lots.notna(),'known','unknown'))
    return cuts,result


def matched(f,draws=500):
    r=I.matched(f,draws)
    if r['status']=='matched':
        r['bonferroni_p']=min(1.,r['p_value']*len(ARMS)*2)
        r['passes']=bool(r['bonferroni_p']<.05 and r['observed_delta']>0 and r['observed_absolute_post']>0)
    return r


def measure(day,out):
    import lightgbm as lgb
    if day not in N.M.TESTS or not all(d<day for d in N.M.FOLDS[day]): raise ValueError('invalid chronology')
    p=N.M.stage(out,'fit_'+day); train=pd.concat([load(d,out) for d in N.M.FOLDS[day]],ignore_index=True); test=load(day,out)
    priorpath=BASE/('fit_'+day); prior=N.E.read_json(priorpath/'receipt.json')
    if N.M.digest(priorpath/'predictions.parquet')!=prior['predictions_sha256']: raise ValueError('changed RAW predictions')
    oldpred=pd.read_parquet(priorpath/'predictions.parquet'); oldpred=oldpred[oldpred.arm=='ALL']
    params=dict(N.E.PD.LGB_PARAMS,objective='regression')
    if params!=prior['params'] or N.E.PD.N_ROUNDS!=prior['rounds']: raise ValueError('pinned learner changed')
    cells=[]; groups=[]; parts=[]; hashes={}; controls={}; cutrecords=[]
    for side in (0,1):
        tr=train[train.side==side]; te=test[test.side==side]; w=N.S.weights(tr); wt=N.S.weights(te)
        if len(tr)<1000 or len(te)<200 or N.S.effective(w)<200: raise ValueError('insufficient support')
        cuts,ss=strata(tr,te);cutrecords.append(dict(side=side,weighted_training_medians=cuts))
        for arm in ARMS:
            xx=represent(tr,arm);zz=represent(te,arm); pred={}
            for target in I.TARGETS:
                if arm=='RAW':
                    path=priorpath/f'ALL_{side}_{target}.txt'; h=N.M.digest(path)
                    if h!=prior['models'][path.name]: raise ValueError('changed RAW model')
                    model=lgb.Booster(model_file=str(path)); controls[str(path)]=h
                else:
                    model=lgb.train(params,lgb.Dataset(xx.to_numpy(np.float32),tr[target].to_numpy(),weight=w/w.mean(),feature_name=FEATURES[arm]),N.E.PD.N_ROUNDS)
                    path=p/f'{arm}_{side}_{target}.txt'; model.save_model(str(path)); hashes[path.name]=N.M.digest(path)
                if model.feature_name()!=FEATURES[arm]: raise ValueError('saved model schema mismatch')
                pp=model.predict(zz.to_numpy(np.float32)); pred[target]=pp; mean=float(np.average(tr[target],weights=w))
                if arm=='RAW':
                    expected=oldpred[oldpred.side==side].set_index(N.KEYS).loc[te.set_index(N.KEYS).index,'delta_hat' if target=='delta' else 'post_hat'].to_numpy()
                    np.testing.assert_allclose(pp,expected,rtol=0,atol=1e-12)
                base=N.S.baseline(tr,te,target)
                cells.append(dict(day=day,side=side,arm=arm,target=target,features=len(FEATURES[arm]),
                    price_rung_sse=N.S.metric(te[target],base,wt,mean)['sse'],**N.S.metric(te[target],pp,wt,mean)))
                for condition,values in ss.items():
                    for value in sorted(set(values)):
                        ix=values==value
                        groups.append(dict(day=day,side=side,arm=arm,target=target,condition=condition,value=value,
                            **N.S.metric(te[target].to_numpy()[ix],pp[ix],wt[ix],mean)))
            f=te[N.KEYS+['probe_t','qpost','qwait','delta']].copy();f['arm']=arm;f['sampling_weight']=wt
            f['post_hat']=pred['qpost'];f['delta_hat']=pred['delta'];f['post_action']=(pred['qpost']>0)&(pred['delta']>=0);parts.append(f)
            print(day,side,arm,'complete',flush=True);I.check_memory()
    pd.concat(parts,ignore_index=True).to_parquet(p/'predictions.parquet',index=False)
    N.write(p/'receipt.json',dict(asof=N.M.now(),day=day,train_dates=list(N.M.FOLDS[day]),cells=cells,conditions=groups,
        condition_cuts=cutrecords,models=hashes,raw_controls=controls,params=params,rounds=N.E.PD.N_ROUNDS,
        predictions_sha256=N.M.digest(p/'predictions.parquet'),row_receipts={d:N.M.digest(out/('rows_'+d)/'receipt.json') for d in (*N.M.FOLDS[day],day)}))


def report(out):
    import lightgbm as lgb
    p=N.M.stage(out,'report'); cells=[]; groups=[]; parts=[];fits={};verified=0
    for day in N.M.TESTS:
        fp=out/('fit_'+day); r=N.E.read_json(fp/'receipt.json');fits[day]=N.M.digest(fp/'receipt.json')
        if N.M.digest(fp/'predictions.parquet')!=r['predictions_sha256']: raise ValueError('changed predictions')
        for n,h in r['models'].items():
            if N.M.digest(fp/n)!=h: raise ValueError('changed model')
            arm=n.split('_')[0]
            if lgb.Booster(model_file=str(fp/n)).feature_name()!=FEATURES[arm]: raise ValueError('artifact schema mismatch')
            verified+=1
        for n,h in r['raw_controls'].items():
            if N.M.digest(n)!=h: raise ValueError('changed raw control')
            verified+=1
        for d,h in r['row_receipts'].items():
            if N.M.digest(out/('rows_'+d)/'receipt.json')!=h: raise ValueError('changed rows')
        cells.extend(r['cells']);groups.extend(r['conditions']);parts.append(pd.read_parquet(fp/'predictions.parquet'))
    cf=pd.DataFrame(cells); pooled=[]
    for (side,arm,target),g in cf.groupby(['side','arm','target']):
        b=cf[(cf.side==side)&(cf.arm=='RAW')&(cf.target==target)].set_index('day')
        rel=cf[(cf.side==side)&(cf.arm=='RELATIVE')&(cf.target==target)]
        pooled.append(dict(side=int(side),arm=arm,target=target,n=int(g.n.sum()),
            r2_vs_prior_mean=float(1-g.sse.sum()/g.baseline_sse.sum()),error_reduction_vs_raw=float(1-g.sse.sum()/b.sse.sum()),
            error_reduction_vs_relative=float(1-g.sse.sum()/rel.sse.sum()),error_reduction_vs_price_rung=float(1-g.sse.sum()/g.price_rung_sse.sum()),
            days_better_than_raw=int(sum(row.sse<b.loc[row.day,'sse']-1e-9 for row in g.itertuples()))))
    pred=pd.concat(parts,ignore_index=True);tests=[dict(arm=a,side=int(s),**matched(g)) for (a,s),g in pred.groupby(['arm','side'])]
    counts=pred[pred.arm=='RAW'].groupby(['day','side']).size()
    rows={d:N.E.read_json(out/('rows_'+d)/'receipt.json') for d in N.DAYS}
    N.write(p/'result.json',dict(asof=N.M.now(),status='complete_development_representation_study',n_test_actions=int(counts.sum()),
        test_populations=[dict(day=d,side=int(s),n=int(n)) for (d,s),n in counts.items()],new_candidates=2,control_candidates=1,
        new_historical_model_fits=40,verified_model_artifacts=verified,primary_target='delta',secondary_target='qpost',
        pooled=pooled,cells=cells,matched_tests=tests,economic_passes=sum(x['passes'] for x in tests),
        fit_receipts=fits,rows=rows,features=FEATURES,public_scaling_divisors=PUBLIC,
        null_design='500 draws match day, side, hour, rung, initial/later generation, action count and sampling mass; Bonferroni across 6 arm/side checks.',
        forward_validation=False,portfolio_evaluated=False,promoted=False,
        limits=['All dates consumed development; no confidence intervals or causal claims.',
            'Normalization preserves absolute causal liquidity and all own risk quantities; no empirical column deletion or clipping.',
            'CONDITION bundles support, volatility and fair-response representations; no individual attribution within that bundle.',
            'History span and availability expose truncated input; the causal frame-weighted liquidity estimate is unchanged.',
            'Raw neutral-KEEP terminal labels unchanged; stop-model benefits remain outside the target.',
            'Coverage statuses do not create examples of rare later-generation or jointly waiting states.']))
    pd.DataFrame(pooled).to_csv(p/'scores.csv',index=False);cf.to_csv(p/'fold_scores.csv',index=False)
    pd.DataFrame(groups).to_csv(p/'conditions.csv',index=False)
    pred.to_parquet(p/'predictions.parquet',index=False)


def fixture():
    f=pd.DataFrame(np.ones((4,len(I.FEATURES))),columns=I.FEATURES)
    f['side_sign']=[1.,-1.,1.,-1.];f['side']=[0,1,0,1];f['sigma']=.001
    f['own_book_liquidity_log']=np.log1p(10.);f['opp_book_liquidity_log']=np.log1p(10.)
    f['pf_mid']=50.;f['price']=49.;f['gap']=2.;f['fair_sensitivity']=.5
    for n in SUPPORT:f[n]=1.
    return f


class Checks(unittest.TestCase):
    def test_scale_invariance_keeps_regime_and_own_risk(self):
        a=fixture();b=a.copy()
        for n in PUBLIC:b[n]*=2
        b.own_book_liquidity_log=np.log1p(20.);b.opp_book_liquidity_log=np.log1p(20.)
        x=represent(a,'RELATIVE');y=represent(b,'RELATIVE')
        names=[n+'_liquidity_ratio' for n in PUBLIC];np.testing.assert_allclose(x[names],y[names])
        self.assertFalse(np.array_equal(x,y,equal_nan=True))
        for n in ('known_position','own_exposure','other_exposure','own_0_remaining_lots'):np.testing.assert_equal(x[n],y[n])
        for n,d in PUBLIC.items():np.testing.assert_allclose(x[n+'_liquidity_ratio']*np.expm1(a.own_book_liquidity_log)*d,a[n])
        b.own_book_liquidity_log=0.
        with self.assertRaises(ValueError):represent(b,'RELATIVE')

    def test_condition_sensitivity_and_opposite_sides(self):
        f=fixture();x=represent(f,'CONDITION')
        self.assertEqual(x.entry_fair_edge_ticks.iloc[0],3.);self.assertEqual(x.entry_fair_edge_ticks.iloc[1],-3.)
        z=f.copy();z.fair_sensitivity*=2;y=represent(z,'CONDITION')
        np.testing.assert_allclose(y.entry_fair_move_ticks100,2*x.entry_fair_move_ticks100)
        z=f.copy();z.sigma*=2;y=represent(z,'CONDITION');np.testing.assert_allclose(y.entry_bn_z100,x.entry_bn_z100/2)
        with self.assertRaises(ValueError):represent(f.drop(columns='price'),'CONDITION')

    def test_missing_is_not_zero(self):
        f=fixture();f.loc[0,'start_known']=0.;f.loc[0,'candidate_change250_lots']=np.nan
        f.loc[1,'candidate_change250_lots']=0.;f.loc[2,'shown_known']=0.
        x=represent(f,'CONDITION');self.assertTrue(np.isnan(x.entry_fair_edge_ticks.iloc[0]))
        self.assertTrue(np.isnan(x.entry_fair_move_ticks100.iloc[0]));self.assertEqual(x.entry_candidate_history_known250.iloc[0],0.)
        self.assertEqual(x.entry_candidate_history_known250.iloc[1],1.);self.assertTrue(np.isnan(x.entry_shrink_fraction250.iloc[2]))
        self.assertEqual(x.entry_candidate_size_zero.iloc[2],1.)

    def test_known_support_future_and_bad_clock(self):
        w={'kf':np.array([0.,100.,200.,400.]),'S5':np.ones((4,5))*30.};W={0:copy.deepcopy(w),1:copy.deepcopy(w)}
        row=dict(probe_t=250.,side=0);x=history(W,row)
        np.testing.assert_equal(x,[200.,200.,3.,100.,0.,50.]);future_check(W,row,x)
        Z=copy.deepcopy(W)
        for s in (0,1):Z[s]['kf'][0]=-30100.
        self.assertEqual(history(Z,row)[4],1.)
        Z[1]['kf'][0]=0.
        with self.assertRaises(ValueError):history(Z,row)

    def test_weighted_strata_and_no_future_fit(self):
        tr=fixture();tr['generation']=[0,0,0,1];tr['fair_distance50']=[1.,1.,1.,10.]
        te=tr.copy();a,_=strata(tr,te);b,_=strata(tr,te.assign(fair_distance50=1e9))
        self.assertEqual(a,b);self.assertGreater(a['fair_distance50'],1.)
        with self.assertRaises(ValueError):A.wquantile([1.,2.],[-1.,1.],[.5])

    def test_selection_instrument(self):
        n=3000;y=np.tile([-1.,1.],n//2)
        f=pd.DataFrame(dict(day='synthetic',side=0,probe_t=0.,rung=0,generation=0,delta=y,qpost=2*y,post_action=y>0,sampling_weight=2.))
        self.assertTrue(matched(f)['passes']);self.assertFalse(matched(f.assign(post_action=y<0))['passes'])
        with self.assertRaises(ValueError):matched(f,199)

    def test_estimator_distinguishes_hidden_regime(self):
        import lightgbm as lgb
        a=np.tile([-1.,1.],2000);params=dict(N.E.PD.LGB_PARAMS,objective='regression')
        null=lgb.train(params,lgb.Dataset(np.zeros((len(a),1)),a),N.E.PD.N_ROUNDS)
        model=lgb.train(params,lgb.Dataset(a[:,None],a),N.E.PD.N_ROUNDS)
        p=model.predict(a[:,None]);b=null.predict(np.zeros((len(a),1)))
        self.assertLess(np.mean((a-p)**2),np.mean((a-b)**2)*.01)
        self.assertGreater(np.mean((-a-p)**2),np.mean((-a-b)**2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['selftest','build','measure','report']);p.add_argument('day',nargs='?');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.command=='selftest':
        r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
        if not r.wasSuccessful():raise SystemExit(1)
    elif a.command=='build':build(a.day,a.out)
    elif a.command=='measure':measure(a.day,a.out)
    else:report(a.out)
