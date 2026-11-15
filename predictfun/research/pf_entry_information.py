"""Entry-information inventory and predictive-contribution experiment.

Frozen action labels are unchanged. Primary target is POST-minus-WAIT; absolute
POST is secondary. All associations are descriptive, not causal effects. Nine
fixed feature-family probes, separate sides, five consumed chronological test
days. No model promotion, keeper training, outcome filtering or new thresholds.
"""
import argparse
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_wait_post_state as N

BASE = N.DEFAULT_OUT / 'full'
OUT = Path(N.M.K.IO) / 'entry_information_20261115_v1'
SEED = 202611151
TARGETS = ('delta', 'qpost')
DRAWS = 500
RECOVERED = ['own_d250', 'opp_d250', 'act_rel', 'd_bnimb250']
GEOMETRY = [f'{s}_level_gap{i}' for s in ('own', 'opp') for i in range(2, 6)]
GEOMETRY += [f'{s}_depth_ticks{i}_lots' for s in ('own', 'opp') for i in range(1, 6)]
DYNAMICS = [f'{s}_best_move{h}' for s in ('own', 'opp') for h in (250, 1000)]
DYNAMICS += [f'candidate_{a}{h}_lots' for h in (250, 1000) for a in ('change', 'shrink', 'grow')]
DYNAMICS += [f'own_imb_change{h}' for h in (250, 1000)]
FAIR = ['side_fair_change250', 'side_fair_change1000', 'side_gap_change250',
        'side_gap_change1000', 'fair_sensitivity', 'fair_distance50', 'contract_remaining_s']
NEW = RECOVERED + GEOMETRY + DYNAMICS + FAIR
FEATURES = N.FEATURES + NEW
BN_RETURN = [n for n in N.FEATURES if n.startswith('bn_r') or n.startswith('side_bn_r')]
BN_PRESSURE = [n for n in N.FEATURES if n in ('bn_imb', 'bn_ntr1000', 'side_bn_imb') or 'bn_flow' in n]
PF_HISTORY = [n for n in N.FEATURES if 'pf_dm' in n or n.startswith('pf_pend')]
FAMILIES = dict(BN_RETURN=BN_RETURN, BN_PRESSURE=BN_PRESSURE, PF_HISTORY=PF_HISTORY,
                RECOVERED=RECOVERED, GEOMETRY=GEOMETRY, LEVEL_DYNAMICS=DYNAMICS, FAIR_DYNAMICS=FAIR)
HISTORY = sum(FAMILIES.values(), [])
STATE = [n for n in N.FEATURES if n not in HISTORY]
ARMS = {'STATE': STATE, **{n: STATE + fs for n, fs in FAMILIES.items()}, 'ALL': FEATURES}


def write(path, obj):
    N.write(path, obj)


def check_memory():
    if N.E.rss_mib() > 2048:
        raise RuntimeError('RSS self-guard exceeded')


def validate_frame(f):
    if f.empty or f.duplicated(N.KEYS).any() or not (f.status == 'eligible').all():
        raise ValueError('invalid action population')
    N.check_targets(f)
    if np.isinf(f[FEATURES].to_numpy(float)).any():
        raise ValueError('infinite feature')
    N.S.weights(f)


def known_index(w, t):
    return int(np.searchsorted(w['kf'], t, side='right')) - 1


def old_market(w, k):
    x = np.asarray(w['X'][k, :len(N.M.K.G32)], float).copy()
    if w['vkl'][k] == -1:
        x[N.M.K.G32.index('gap')] = np.nan
    return x


def history_market(w, t, name):
    k = int(np.searchsorted(w['tg'], t, side='right')) - 1
    if k < 0 or t - w['tg'][k] > 60.:
        return np.nan
    return old_market(w, k)[N.M.K.G32.index(name)]


def extra_features(W, row, cache=None):
    """Only received frames/grid rows at or before this exact action time."""
    side = int(row['side']); t = float(row['probe_t']); k = int(row['probe_k'])
    w = W[side]; sign = 1. if side == 0 else -1.
    if k < 0 or k >= len(w['tg']) or w['tg'][k] != t:
        raise ValueError('action/grid timestamp mismatch')
    if len(w['X'][k]) != len(N.M.K.G32) + 5:
        raise ValueError('upstream schema changed')
    f = cache or N.Features(W)
    result = dict(zip(RECOVERED, map(float, w['X'][k, 33:37])))
    js = {s: known_index(W[s], t) for s in (0, 1)}
    for label, s in (('own', side), ('opp', 1-side)):
        j = js[s]; ww = W[s]; sg = 1 if s == 0 else -1
        if j < 0 or ww['P5'][j, 0] < 0:
            raise ValueError('missing known book')
        p = ww['P5'][j]; z = ww['S5'][j]; gaps = sg * (p[0] - p)
        for i in range(2, 6):
            result[f'{label}_level_gap{i}'] = float(gaps[i-1]) if p[i-1] >= 0 else np.nan
        for i in range(1, 6):
            result[f'{label}_depth_ticks{i}_lots'] = float(z[(p >= 0) & (gaps <= i-1)].sum() / N.M.SIZE)
        for h in (250, 1000):
            jp = known_index(ww, t-h)
            result[f'{label}_best_move{h}'] = float(sign*(p[0]-ww['P5'][jp, 0])) if jp >= 0 and ww['P5'][jp, 0] >= 0 else np.nan
    j = js[side]; lv = f.level(side, float(row['price']))
    for h in (250, 1000):
        jp = known_index(w, t-h)
        seg = lv[jp:j+1] if jp >= 0 else np.array([np.nan])
        valid = np.isfinite(seg).all()
        diff = np.diff(seg)
        result[f'candidate_change{h}_lots'] = float((seg[-1]-seg[0])/N.M.SIZE) if valid else np.nan
        result[f'candidate_shrink{h}_lots'] = float(np.maximum(-diff, 0).sum()/N.M.SIZE) if valid else np.nan
        result[f'candidate_grow{h}_lots'] = float(np.maximum(diff, 0).sum()/N.M.SIZE) if valid else np.nan
        result[f'own_imb_change{h}'] = sign*(history_market(w, t, 'pf_imb1')-history_market(w, t-h, 'pf_imb1'))
    gap = history_market(w, t, 'gap'); fair = gap + history_market(w, t, 'pf_mid')
    for h in (250, 1000):
        gap0 = history_market(w, t-h, 'gap'); fair0 = gap0 + history_market(w, t-h, 'pf_mid')
        result[f'side_fair_change{h}'] = sign*(fair-fair0)
        result[f'side_gap_change{h}'] = sign*(gap-gap0)
    rem = (w['ws']+300000.-t)/1000.; sig = history_market(w, t, 'sigma')
    result['contract_remaining_s'] = rem
    result['fair_distance50'] = abs(fair-50.)
    good = np.isfinite([fair, sig, rem]).all() and sig > 0 and rem > 0
    z = ndtri(np.clip(fair/100., 1e-6, 1.-1e-6)) if good else np.nan
    result['fair_sensitivity'] = float(100.*np.exp(-.5*z*z)/np.sqrt(2*np.pi)*1e-4/(sig*np.sqrt(rem))) if good else np.nan
    if set(result) != set(NEW) or np.isinf(list(result.values())).any():
        raise ValueError('bad extra feature schema/value')
    return np.asarray([result[n] for n in NEW], np.float32)


def future_check(W, row, expected):
    changed = copy.deepcopy(W); t = row['probe_t']
    for s in (0, 1):
        w = changed[s]; future = np.asarray(w['kf']) > t
        w['S5'][future] = 987654.
        w['P5'][future] = np.where(w['P5'][future] >= 0, 9, -1)
        w['X'][np.asarray(w['tg']) > t] = -123456.
    np.testing.assert_equal(extra_features(changed, row), expected)
    changed = copy.deepcopy(W)
    x = changed[int(row['side'])]['X']
    k = int(row['probe_k'])
    x[k, 33] = float(x[k, 33])+10. if np.isfinite(x[k, 33]) else 10.
    if np.array_equal(extra_features(changed, row), expected, equal_nan=True):
        raise ValueError('known-input positive control failed')


def build(day, out):
    if day not in N.DAYS:
        raise ValueError('undeclared day')
    p = N.M.stage(out, 'rows_'+day)
    rows = N.read_rows(BASE, day).copy()
    rows[NEW] = np.nan
    data = N.M.K.load_day(day)
    receipts = N.E.read_json(BASE/('rows_'+day)/'receipt.json')
    allowed = set(receipts['windows'])
    data['W0'] = {m:t for m,t in data['W0'].items() if m in allowed}
    seen = set(); checks = 0; visited = []
    groups = rows.groupby('market', sort=False).indices
    for W in N.E.PAIRS.window_pairs(data):
        m = W[0]['m']; visited.append(m); cache = N.Features(W)
        for number, idx in enumerate(groups.get(m, [])):
            row = rows.iloc[idx]; side = int(row.side); k = int(row.probe_k)
            raw = old_market(W[side], k).astype(np.float32)
            np.testing.assert_allclose(raw, row[N.M.K.G32].to_numpy(np.float32), atol=1e-7, rtol=1e-6, equal_nan=True)
            values = extra_features(W, row, cache)
            rows.loc[rows.index[idx], NEW] = values
            if number == 0:
                future_check(W, row, values); checks += 1
            seen.add(idx)
        if len(visited) % 32 == 0:
            print(day, len(visited), 'windows', len(seen), 'actions', round(N.E.rss_mib()), 'MiB', flush=True)
        check_memory()
    if len(seen) != len(rows) or len(set(visited)) != len(visited) or set(visited) != allowed:
        raise ValueError('feature join omitted or duplicated actions/windows')
    validate_frame(rows)
    rows.to_parquet(p/'rows.parquet', index=False)
    write(p/'receipt.json', dict(asof=N.M.now(), day=day, n=len(rows), windows=len(visited), features=FEATURES,
          source_receipt_sha256=N.M.digest(BASE/('rows_'+day)/'receipt.json'), source_status=receipts['source_status'],
          statuses=receipts['statuses'], sealed_lookback_windows=receipts['sealed_lookback_windows'], future_checks=checks,
          rows_sha256=N.M.digest(p/'rows.parquet')))


def load(out, day):
    p = out/('rows_'+day); meta = N.E.read_json(p/'receipt.json')
    if meta['day'] != day or meta['features'] != FEATURES or N.M.digest(p/'rows.parquet') != meta['rows_sha256']:
        raise ValueError('row receipt differs')
    if N.M.digest(BASE/('rows_'+day)/'receipt.json') != meta['source_receipt_sha256']:
        raise ValueError('source row receipt differs')
    f = pd.read_parquet(p/'rows.parquet'); validate_frame(f)
    if len(f) != meta['n']:
        raise ValueError('row count differs')
    return f


def wcorr(x, y, w, ranks=False):
    x = np.asarray(x, float); y = np.asarray(y, float); w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x,y,w = x[ok],y[ok],w[ok]
    if len(x) < 30 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    if ranks:
        x,y = rankdata(x),rankdata(y)
    x=x-np.average(x,weights=w); y=y-np.average(y,weights=w)
    return float(np.sum(w*x*y)/np.sqrt(np.sum(w*x*x)*np.sum(w*y*y)))


def bins(train, test):
    a = np.asarray(train,float); b=np.asarray(test,float); good=np.isfinite(a)
    cuts = np.unique(np.quantile(a[good], [.2,.4,.6,.8])) if good.any() else np.array([])
    return np.where(good,np.searchsorted(cuts,a),-1), np.where(np.isfinite(b),np.searchsorted(cuts,b),-1), cuts


def marginal(train, test, name, target):
    w=N.S.weights(train); wt=N.S.weights(test); y=train[target].to_numpy(); z=test[target].to_numpy()
    a=train[name].to_numpy(); b=test[name].to_numpy(); prior=float(np.average(y,weights=w))
    ib,jb,cuts=bins(a,b); pred=np.full(len(test),prior); profiles=[]
    for j in sorted(set(ib)|set(jb)):
        tr=ib==j; te=jb==j
        enough=tr.sum()>=30 and N.S.effective(w[tr])>=20
        mean=float(np.average(y[tr],weights=w[tr])) if tr.any() else None
        if enough: pred[te]=mean
        profiles.append(dict(bin=int(j), train_n=int(tr.sum()), test_n=int(te.sum()), train_mean=mean,
             test_mean=float(np.average(z[te],weights=wt[te])) if te.any() else None, used=bool(enough)))
    stats=N.S.metric(z,pred,wt,prior)
    return dict(feature=name,target=target,train_n=len(train),test_n=len(test),train_missing=int((~np.isfinite(a)).sum()),
       test_missing=int((~np.isfinite(b)).sum()),train_ic=wcorr(a,y,w,True),test_ic=wcorr(b,z,wt,True),
       train_pearson=wcorr(a,y,w),test_pearson=wcorr(b,z,wt),cuts=cuts.tolist(),profile=profiles,**stats)


def measure(day, out):
    import lightgbm as lgb
    if day not in N.M.TESTS or not all(d < day for d in N.M.FOLDS[day]):
        raise ValueError('invalid chronology')
    p=N.M.stage(out,'fit_'+day)
    train=pd.concat([load(out,d) for d in N.M.FOLDS[day]],ignore_index=True); test=load(out,day)
    cells=[]; associations=[]; predictions=[]; hashes={}; importance=[]
    params=dict(N.E.PD.LGB_PARAMS,objective='regression')
    for side in (0,1):
        tr=train[train.side==side]; te=test[test.side==side]; w=N.S.weights(tr); wt=N.S.weights(te)
        if len(tr)<1000 or len(te)<200 or N.S.effective(w)<200:
            raise ValueError('insufficient support')
        for name in FEATURES:
            for target in TARGETS:
                associations.append(dict(day=day,side=side,**marginal(tr,te,name,target)))
        for arm,names in ARMS.items():
            x=tr[names].to_numpy(np.float32); z=te[names].to_numpy(np.float32); pred={}
            for target in TARGETS:
                model=lgb.train(params,lgb.Dataset(x,tr[target].to_numpy(),weight=w/w.mean(),feature_name=names),N.E.PD.N_ROUNDS)
                path=p/f'{arm}_{side}_{target}.txt'; model.save_model(str(path)); hashes[path.name]=N.M.digest(path)
                pred[target]=model.predict(z)
                cells.append(dict(day=day,side=side,arm=arm,target=target,features=len(names),
                     **N.S.metric(te[target],pred[target],wt,float(np.average(tr[target],weights=w)))))
                importance.extend(dict(day=day,side=side,arm=arm,target=target,feature=n,gain=float(g))
                                  for n,g in zip(names,model.feature_importance('gain')))
            f=te[N.KEYS+['probe_t','qpost','qwait','delta']].copy()
            f['post_hat']=pred['qpost']; f['delta_hat']=pred['delta']; f['post_action']=(pred['qpost']>0)&(pred['delta']>=0)
            f['sampling_weight']=wt; f['arm']=arm; predictions.append(f)
            print(day,'bid' if side==0 else 'ask',arm,'complete',N.M.now(),flush=True)
            check_memory()
    pd.concat(predictions,ignore_index=True).to_parquet(p/'predictions.parquet',index=False)
    pd.DataFrame(importance).to_parquet(p/'importance.parquet',index=False)
    write(p/'associations.json',associations)
    write(p/'receipt.json',dict(asof=N.M.now(),day=day,train_dates=list(N.M.FOLDS[day]),cells=cells,
         models=hashes,params=params,rounds=N.E.PD.N_ROUNDS,association_sha256=N.M.digest(p/'associations.json'),
         predictions_sha256=N.M.digest(p/'predictions.parquet'),importance_sha256=N.M.digest(p/'importance.parquet'),
         row_receipts={d:N.M.digest(out/('rows_'+d)/'receipt.json') for d in (*N.M.FOLDS[day],day)}))


def matched(f, draws=DRAWS):
    if draws<200:
        raise ValueError('too few matched draws')
    f=f.copy(); w=N.S.weights(f); selected=f.post_action.to_numpy(bool)
    if not np.array_equal(w,f.sampling_weight.to_numpy()):
        raise ValueError('wrong weights')
    count=int(selected.sum()); eff=N.S.effective(w[selected]) if count else 0.
    result=dict(n=len(f),posts=count,effective_selected=eff,draws=draws)
    if min(count,len(f)-count)<30 or eff<30:
        return dict(**result,status='insufficient_action_variation',passes=False)
    obs=float(np.average(f.delta.to_numpy()[selected],weights=w[selected]))
    absolute=float(np.average(f.qpost.to_numpy()[selected],weights=w[selected]))
    f['hour']=(f.probe_t//3600000).astype('int64'); f['initial']=f.generation==0
    null=np.zeros(draws); mass=0.; rng=np.random.default_rng(SEED)
    for _,g in f.groupby(['day','side','hour','rung','initial']):
        k=int(g.post_action.sum()); ww=g.sampling_weight.to_numpy()
        if np.ptp(ww)!=0:
            raise ValueError('mixed weight stratum')
        if k:
            choices=np.argsort(rng.random((draws,len(g))),axis=1)[:,:k]
            null+=g.delta.to_numpy()[choices].sum(axis=1)*ww[0];mass+=k*ww[0]
    if mass!=float(w[selected].sum()):
        raise ValueError('null selection mass mismatch')
    null/=mass; pv=float((1+np.sum(null>=obs-1e-12))/(draws+1)); adjusted=min(1.,pv*len(ARMS)*2)
    return dict(**result,status='matched',observed_delta=obs,observed_absolute_post=absolute,
         null_mean=float(null.mean()),null_p95=float(np.quantile(null,.95)),p_value=pv,bonferroni_p=adjusted,
         passes=bool(adjusted<.05 and obs>0 and absolute>0))


def catalog():
    records=[]
    for name in FEATURES:
        group=next((g for g,cols in FAMILIES.items() if name in cols),'STATE')
        source='frozen action-row input' if name in N.FEATURES else 'known book history' if name in GEOMETRY+DYNAMICS else 'known grid history'
        detail='Original model input; directional aliases contain no new raw information.'
        if name in RECOVERED: detail='Already computed upstream; omitted by the original 32-market-column slice.'
        if name in GEOMETRY: detail='Price spacing or cumulative displayed size within a fixed tick distance; both sides.'
        if name in DYNAMICS: detail='Received-frame best moves, same-price size changes/churn, or imbalance changes over trailing 250/1000 ms.'
        if name in FAIR: detail='Model-implied fair-value transform, not observed truth; start-unknown values stay missing; contract expiry differs from research cutoff.'
        records.append(dict(feature=name,group=group,source=source,original_input=name in N.FEATURES,description=detail))
    return dict(asof=N.M.now(),feature_count=len(FEATURES),original_count=len(N.FEATURES),additional_count=len(NEW),features=records,
       arm_features=ARMS,available_sources=[
         dict(source='Binance bookTicker and aggregate trades',representation='returns, imbalance, signed flow, activity, known-time age',limit='Existing feed clock uses assumed event-to-known latency; no deeper Binance L2 in this study.'),
         dict(source='Predict received top-five book frames and pending-size increments',representation='current book, price/size history, pending flow, candidate-level history',limit='Displayed size changes do not identify individual cancellations versus trades or exact queue position.'),
         dict(source='Contract metadata and known reference-price estimate',representation='expiry, volatility, basis, masked implied fair value',limit='Reference must be known; implied fair value is a model assumption.'),
         dict(source='Client-known order/fill/cancel state',representation='all ladder slots, received fills, pending operations, exposure, ages',limit='No actual unreceived fill, queue or rejection state.'),
         dict(source='Venue trade legs and inferred execution anchors',representation='label/replay only',limit='Not treated as a causal decision-time trade tape without receipt-clock reconstruction.'),
         dict(source='Legacy guard score and other fitted models',representation='excluded from new inputs',limit='Training-fold lineage not verified for this entry study; avoid indirect held-out leakage.'),
         dict(source='Future book, settlement outcome, actual queue and future fills',representation='outcomes only',limit='Never features.')])


def report(out):
    p=N.M.stage(out,'report'); cells=[]; associations=[]; parts=[]; fits={}
    for day in N.M.TESTS:
        path=out/('fit_'+day); r=N.E.read_json(path/'receipt.json'); fits[day]=N.M.digest(path/'receipt.json')
        for filename,key in (('predictions.parquet','predictions_sha256'),('associations.json','association_sha256'),('importance.parquet','importance_sha256')):
            if N.M.digest(path/filename)!=r[key]: raise ValueError('changed fit artifact')
        for filename,h in r['models'].items():
            if N.M.digest(path/filename)!=h: raise ValueError('changed fitted model')
        for d,h in r['row_receipts'].items():
            if N.M.digest(out/('rows_'+d)/'receipt.json')!=h: raise ValueError('changed row receipt')
        cells.extend(r['cells']); associations.extend(N.E.read_json(path/'associations.json'))
        parts.append(pd.read_parquet(path/'predictions.parquet'))
    cf=pd.DataFrame(cells); pooled=[]
    for (side,arm,target),g in cf.groupby(['side','arm','target']):
        b=cf[(cf.side==side)&(cf.arm=='STATE')&(cf.target==target)].set_index('day')
        improvements=[float(b.loc[r.day,'sse']-r.sse) for r in g.itertuples()]
        pooled.append(dict(side=int(side),arm=arm,target=target,n=int(g.n.sum()),r2_vs_prior_mean=float(1-g.sse.sum()/g.baseline_sse.sum()),
             error_reduction_vs_state=float(1-g.sse.sum()/b.sse.sum()),days_better_than_state=int(sum(x>1e-9 for x in improvements)),
             positive_all_days=bool(all(x>1e-9 for x in improvements))))
    af=pd.DataFrame(associations); feature_summary=[]
    for (side,name,target),g in af.groupby(['side','feature','target']):
        good=g.test_ic.notna(); pairs=g.train_ic.notna()&good
        signs=(g.loc[pairs,'train_ic']*g.loc[pairs,'test_ic'])>0
        feature_summary.append(dict(side=int(side),feature=name,target=target,n=int(g.n.sum()),
             test_missing=int(g.test_missing.sum()),mean_day_rank_ic=float(g.loc[good,'test_ic'].mean()) if good.any() else None,
             min_day_rank_ic=float(g.loc[good,'test_ic'].min()) if good.any() else None,max_day_rank_ic=float(g.loc[good,'test_ic'].max()) if good.any() else None,
             sign_agreement_days=int(signs.sum()),correlation_days=int(pairs.sum()),
             univariate_r2=float(1-g.sse.sum()/g.baseline_sse.sum())))
    pred=pd.concat(parts,ignore_index=True)
    tests=[dict(arm=a,side=int(s),**matched(g)) for (a,s),g in pred.groupby(['arm','side'])]
    counts=pred[pred.arm=='STATE'].groupby(['day','side']).size().to_dict()
    rows_meta={d:N.E.read_json(out/('rows_'+d)/'receipt.json') for d in N.DAYS}
    result=dict(asof=N.M.now(),status='complete_consumed_date_information_study',primary_target='delta',secondary_target='qpost',
       new_diagnostic_candidates=len(ARMS),new_production_candidates=0,forward_validation=False,causal_effects=False,portfolio_evaluated=False,
       n_test_actions=int(sum(counts.values())),test_populations=[dict(day=d,side=int(s),n=int(n)) for (d,s),n in counts.items()],
       pooled=pooled,cells=cells,matched_tests=tests,fit_receipts=fits,rows=rows_meta,feature_count=len(FEATURES),
       marginal_tests=len(feature_summary),marginal_inference='Descriptive screen; no p-values, no selection or confirmation claim.',
       null_design='500 draws match day/side/hour/rung/initial-later generation, count and inverse-sampling mass; Bonferroni over all 18 arm-side tests.',
       limits=['All dates consumed. Groups declared before these results; they are research probes, not validated policies.',
         'Association is not causal influence; correlated and derived features share information.',
         'STATE already includes current book, fair-value gap, freshness, best age and contract context; additions measure incremental history/geometry value.',
         'Labels are unchanged neutral-KEEP continuation values; cancellation benefits are not credited here.',
         'Top-five public books and assumed feed/fill clocks limit signal and queue precision.',
         'First 60 seconds of each contract only; sparse later-generation/joint-waiting support remains.',
         'Raw targets and all eligible actions retained; features missing outside visible depth remain missing. No intervals or model promotion.'])
    write(p/'result.json',result); write(p/'inventory.json',catalog())
    pd.DataFrame(catalog()['features']).to_csv(p/'inventory.csv',index=False)
    pd.DataFrame(feature_summary).to_csv(p/'feature_summary.csv',index=False)
    pd.DataFrame(pooled).to_csv(p/'group_summary.csv',index=False)
    pred.to_parquet(p/'predictions.parquet',index=False)


class Checks(unittest.TestCase):
    def test_inventory_complete_disjoint(self):
        self.assertEqual(len(FEATURES),325)
        self.assertEqual(len(FEATURES),len(set(FEATURES)))
        self.assertEqual(set(STATE+HISTORY),set(FEATURES))
        self.assertEqual(len(STATE+HISTORY),len(FEATURES))
        self.assertNotIn('v',FEATURES)

    def test_bins_fit_training_only(self):
        a=np.arange(100.); b=np.array([-1000.,50.,1000.,np.nan])
        i,j,c=bins(a,b); _,_,c2=bins(a,b*100)
        np.testing.assert_equal(c,c2); self.assertEqual(j[-1],-1)
        self.assertAlmostEqual(wcorr(a,a,np.ones(100),True),1.)
        self.assertAlmostEqual(wcorr(a,-a,np.ones(100),True),-1.)

    def test_marginal_signal_and_missing_fallback(self):
        a=np.tile(np.arange(10.),100)
        tr=pd.DataFrame(dict(generation=0,signal=a,delta=a))
        te=pd.DataFrame(dict(generation=0,signal=a,delta=a))
        r=marginal(tr,te,'signal','delta')
        self.assertGreater(r['r2_vs_prior_mean'],.9)
        bad=marginal(tr,te.assign(delta=9.-a),'signal','delta')
        self.assertLess(bad['r2_vs_prior_mean'],0.)
        missing=marginal(tr,te.assign(signal=np.nan),'signal','delta')
        self.assertAlmostEqual(missing['r2_vs_prior_mean'],0.)

    def test_group_estimator_signal_and_reversal(self):
        import lightgbm as lgb
        a=np.tile([-1.,1.],2000); z=np.tile([-1.,1.],1000)
        params=dict(N.E.PD.LGB_PARAMS,objective='regression')
        base=lgb.train(params,lgb.Dataset(np.zeros((len(a),1)),3*a),N.E.PD.N_ROUNDS)
        model=lgb.train(params,lgb.Dataset(a[:,None],3*a),N.E.PD.N_ROUNDS)
        bp=base.predict(np.zeros((len(z),1)));p=model.predict(z[:,None])
        self.assertLess(np.sum((p-3*z)**2),.1*np.sum((bp-3*z)**2))
        self.assertGreater(np.sum((p+3*z)**2),np.sum((bp+3*z)**2))

    def test_feature_clocks_and_controls(self):
        tg=np.arange(40)*50.; W={}
        for s in (0,1):
            P=np.tile(np.arange(50,45,-1) if s==0 else np.arange(52,57),(40,1))
            X=np.zeros((40,37));X[:,N.M.K.G32.index('pf_mid')]=51.;X[:,N.M.K.G32.index('sigma')]=1e-4
            X[:,N.M.K.G32.index('gap')]=np.arange(40)*.01;X[:,33]=np.arange(40)
            W[s]=dict(kf=tg.copy(),tg=tg.copy(),X=X,vkl=[0.]*40,P5=P,S5=np.ones((40,5))*30.,ws=0.)
        row=dict(side=0,probe_t=1000.,probe_k=20,price=50.)
        x=extra_features(W,row);future_check(W,row,x)
        self.assertAlmostEqual(x[NEW.index('side_fair_change250')],.05,places=5)
        self.assertAlmostEqual(x[NEW.index('own_depth_ticks5_lots')],5.)
        W[0]['vkl'][20]=-1.
        self.assertTrue(np.isnan(extra_features(W,row)[NEW.index('fair_sensitivity')]))
        W[0]['X'][20,33]=np.nan
        future_check(W,row,extra_features(W,row))
        with self.assertRaises(ValueError): extra_features(W,dict(row,probe_t=999.))

    def test_matched_value_positive_and_bad_controls(self):
        n=400; y=np.tile([-1.,1.],n//2)
        f=pd.DataFrame(dict(day='x',side=0,rung=0,generation=0,probe_t=np.arange(n),delta=y,qpost=y,
                            post_action=y>0,sampling_weight=2.))
        good=matched(f);self.assertTrue(good['passes'])
        bad=matched(f.assign(post_action=y<0));self.assertFalse(bad['passes'])
        with self.assertRaises(ValueError): matched(f,199)
        with self.assertRaises(ValueError): matched(f.assign(sampling_weight=1.))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['selftest','build','measure','report'])
    p.add_argument('day',nargs='?');p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    if a.command=='selftest':
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
        if not result.wasSuccessful():raise SystemExit(1)
    elif a.command=='build':build(a.day,a.out)
    elif a.command=='measure':measure(a.day,a.out)
    else:report(a.out)


if __name__=='__main__':main()
