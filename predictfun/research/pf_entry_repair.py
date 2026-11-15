"""Frozen development study: loader/labels, coverage, then training history.

30-share timing labels retain neutral KEEP and the frozen entry continuation.
50/100-share arms measure queue retention on their OWN neutral paths. They do
not reuse 30-share action identities or multiply its P&L. All sizes keep the
same 1,500-share hard cap. No model promotion or forward-validation claim.
"""
import argparse
import copy
from pathlib import Path
import re
import sys
import time
import unittest
from unittest import mock

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_entry_representation as R
import pf_entry_repair_lane as L
import pf_clock_rl as CK

I = R.I
N = I.N
M = N.M
K = M.K
DAYS = tuple(N.DAYS)
SIZES = (30., 50., 100.)
REFERENCE = Path(K.IO) / 'entry_representation_20261115_v2'
VERSION = 'entry-density-4-preserved-age-v1'
ARMS = ('SPARSE_ALL', 'DENSE_ALL', 'DENSE_RECENT2')
OLD_LOADER_SHA = '4f173446bf07d7141a96bddf02470e38e66aa64b2b28a380eeb7dec9343530cc'
NEW_LOADER_SHA = '56377b13c71362281036a8711e94a4b9979d163f02c22f6bb821e799fcb2f8e3'
ORIGINAL_REQUIRE = M.require
LIMITS = [
    'Consumed development days only. No forward validation or model promotion.',
    'The inherited per-day clock is a retrospective nuisance calibration; boundary-filtered clock differences are reported, not silently substituted into saved features.',
    'Raw-to-step5 causality and clock reconstruction remain an upstream qualification; downstream future checks cannot close them.',
    'Queue-zero and queue-block are simulator sensitivities, not observed queues or P&L bounds. Next-frame activation correction and E2 fill clocks remain assumptions.',
    'Entry labels use neutral KEEP and a frozen bootstrap entry continuation, not the separate selective cancellation model.',
    'Density repairs later-generation sampling only, not joint-policy state coverage, missing history, or new calendar regimes.',
    'Size diagnostics use same windows, separate neutral paths, constant cap and no own market impact. They are not fitted entry-model results.',
]


def require_day(day):
    if day not in DAYS:
        raise ValueError('day outside declared development partition')


def require_migrated(meta):
    """Only the reviewed 6340dd32 opt-in loader change may cross old receipts.

    Model bytes, schema and every other source must still match. Old receipts
    are never rewritten. Rebuilt rows must additionally reproduce the old ones.
    """
    if M.digest(HERE/'pf_kp_s4.py') != NEW_LOADER_SHA:
        raise ValueError('unreviewed loader implementation')
    migrated = copy.deepcopy(meta)
    if migrated['sources']['pf_kp_s4.py'] != OLD_LOADER_SHA:
        raise ValueError('unexpected legacy loader identity')
    migrated['sources']['pf_kp_s4.py'] = NEW_LOADER_SHA
    ORIGINAL_REQUIRE(migrated)


def predictor():
    with mock.patch.object(M, 'require', require_migrated):
        return M.load_models(N.BASE/'bootstrap')


def original_rows(day):
    with mock.patch.object(M, 'require', require_migrated):
        return M.load_rows(N.BASE,'learned',day)


def allowed_part(path, day):
    require_day(day)
    if re.fullmatch(r'part_\d{8}_\d{2}\.parquet', Path(path).name) is None or int(Path(path).name[14:16]) > 23:
        raise ValueError('malformed book part')
    return K.part_allowed(str(path), day)


def load_day(day):
    """Opt in before reading; a second read tripwire makes regressions fail closed."""
    require_day(day)
    read = pd.read_parquet
    def guarded(path, *args, **kwargs):
        p = Path(path)
        if p.parent.name == 'books' and not allowed_part(p, day):
            raise ValueError('refused non-admissible book part before read')
        return read(path, *args, **kwargs)
    with mock.patch.object(K, 'SKIP_FOREIGN_PARTS', True), mock.patch.object(pd, 'read_parquet', guarded):
        data = K.load_day(day)
    sealed = []
    if day == '20261031':
        bound = K._ms('2026-10-31') + 1800000.
        sealed = [m for m, t in data['W0'].items() if t < bound]
        data['W0'] = {m:t for m,t in data['W0'].items() if m not in sealed}
    return data, sealed


def sampling(day, market, key):
    h = M.hash_key(day, market, key)
    old_mod = 2 if key[2] == 0 else 32
    mod = 2 if key[2] == 0 else 4
    return dict(sampling_version=VERSION, inclusion_probability=1./mod,
                sampling_weight=float(mod), old_selected=h % old_mod == 0,
                selected=h % mod == 0, requested_delay=M.AGES[(h // old_mod) % len(M.AGES)])


def weights(f):
    if not (f.sampling_version == VERSION).all():
        raise ValueError('undeclared sampling version')
    p = f.inclusion_probability.to_numpy(float)
    g = f.generation.to_numpy(float)
    if not np.isfinite(g).all() or (g < 0).any() or (g != np.floor(g)).any():
        raise ValueError('invalid generation')
    expected = np.where(g == 0, .5, .25)
    if not np.isfinite(p).all() or (p <= 0).any() or (p > 1).any() or not np.array_equal(p, expected):
        raise ValueError('invalid inclusion probability')
    w = 1./p
    if not np.array_equal(w, f.sampling_weight.to_numpy(float)):
        raise ValueError('wrong inclusion weight')
    return w


def run(W, size=30., queue='zero', choose=None, observe=None, snapshot_filter=None):
    aux = {s:[False]*len(W[s]['tg']) for s in (0,1)}
    result = L.sim_joint(W, 'M', 50., M.CAP, aux, size=size, nr=M.NR, mode='hard',
        policy=choose or (lambda v: True), observe=observe, snapshot_filter=snapshot_filter,
        neutral_keeper=True, unknown_queue=queue)
    if any(result['stat']['viol'].values()):
        raise ValueError('engine accounting violation')
    N.E.terminal(W, result)
    return result


def order_summary(W, result, size):
    rows = []
    for (side,rung), orders in result['order_states'].items():
        for generation,o in enumerate(orders):
            fills = o['fl']; shares = float(sum(q for _,q in fills))
            finish = min(float(o['tr']), float(result['terminal_time']), float(o['tfill']))
            rows.append(dict(side=side,rung=rung,generation=generation,refused=bool(o['ref']),
                initial_queue_unknown=bool(o.get('initial_queue_unknown',False)),
                shares=shares, fill_events=len(fills), multiple_fills=len(fills)>=2,
                exhausted=shares>=size-1e-8, resting_ms=max(0.,finish-o['ta']) if not o['ref'] else 0.))
    return rows


def clock_audit(day):
    """Read only admissible parts; do not reconstruct the forbidden old input set."""
    values = []; parts = 0; skipped = 0
    for path in sorted((Path(K.IO)/'step1'/day/'books').glob('part_*.parquet')):
        if not allowed_part(path, day):
            skipped += 1
            continue
        f = pd.read_parquet(path, columns=['t_ms','recv_ns'])
        values.append(f.recv_ns.to_numpy(np.float64)/1e6-f.t_ms.to_numpy(np.float64)); parts += 1
        I.check_memory()
    lag = np.concatenate(values)
    shift = float(np.median(lag)-CK.PF_P50)
    saved = N.E.read_json(Path(K.IO)/'step1'/day/'clock_rl_d2.json')
    return dict(n_frames=len(lag),parts=parts,parts_skipped=skipped,
        allowed_parts_shift_ms=shift,inherited_shift_ms=saved['shift_ms'],
        difference_ms=shift-saved['shift_ms'],exactly_equal=shift==saved['shift_ms'],
        interpretation='diagnostic only; stored features and labels retain their common inherited clock')


def validity(day, out, max_windows=None):
    require_day(day); p=M.stage(out,'validity_'+day); start=time.monotonic()
    clock=clock_audit(day); data,sealed=load_day(day)
    old=R.load(day,REFERENCE); oldgroups=old.groupby('market',sort=False).indices
    model=predictor(); records=[]; checked=0; visited=[]
    for W in N.E.PAIRS.window_pairs(data):
        market=W[0]['m']; visited.append(market)
        for size in SIZES:
            for queue in ('zero','block'):
                result=run(copy.deepcopy(W),size,queue)
                stats=order_summary(W,result,size)
                term=N.E.terminal(W,result)
                for side in (0,1):
                    rows=[x for x in stats if x['side']==side]
                    records.append(dict(day=day,market=market,side=side,size=size,queue=queue,
                        orders=len(rows),accepted=sum(not x['refused'] for x in rows),
                        reposts=sum(x['generation']>0 for x in rows),
                        unknown_initial=sum(x['initial_queue_unknown'] for x in rows),
                        filled_orders=sum(x['shares']>0 for x in rows),multi_fill_orders=sum(x['multiple_fills'] for x in rows),
                        exhausted=sum(x['exhausted'] for x in rows),shares=sum(x['shares'] for x in rows),
                        resting_order_ms=sum(x['resting_ms'] for x in rows),
                        max_abs_inventory=result['stat']['maxpos'],
                        hard_blocks=result['stat']['hard_block'][side],
                        joint_terminal_pnl=term['pnl']))
        # Exact target/feature check on first old eligible row per side in every window.
        births=[]; N.run(W,lambda v:True,observe=lambda v,a: births.append(v) if v['status']=='eligible' else None)
        bykey={M.key(b):b for b in births}; model.clear(); feat=N.Features(W)
        windowold=old.iloc[oldgroups.get(market,[])]
        for _,g in windowold.groupby('side',sort=True):
            row=g.iloc[0]; b=bykey[(int(row.side),int(row.rung),int(row.generation))]
            value,x=N.paired(W,b,row.requested_delay,model,feat)
            for key in ('qpost','qwait','delta','probe_t','probe_k','local_post','local_delta','external_post','external_delta'):
                np.testing.assert_allclose(value[key],row[key],atol=1e-8,rtol=0)
            np.testing.assert_allclose(x,row[N.FEATURES].to_numpy(float),atol=0,rtol=0,equal_nan=True)
            extra=I.extra_features(W,row); np.testing.assert_allclose(extra,row[I.NEW].to_numpy(float),atol=0,rtol=0,equal_nan=True)
            np.testing.assert_allclose(R.history(W,row),row[R.SUPPORT].to_numpy(float),atol=0,rtol=0,equal_nan=True)
            # Each queue scenario rebuilds its OWN neutral birth; do not pair generation ids across paths.
            checked+=1
        if len(visited)%32==0: print('validity',day,len(visited),'windows',flush=True)
        I.check_memory()
        if max_windows and len(visited)>=max_windows: break
    if max_windows is None and set(visited) != set(data['W0']):
        raise ValueError('validity window population incomplete')
    f=pd.DataFrame(records);f.to_parquet(p/'size_windows.parquet',index=False)
    N.write(p/'receipt.json',dict(asof=M.now(),day=day,partial=max_windows is not None,
        windows=visited,sealed_lookback_windows=sealed,source_status=data['stat'],clock=clock,
        target_feature_rows_checked=checked,rows_sha256=M.digest(p/'size_windows.parquet'),
        source_receipt_sha256=M.digest(REFERENCE/('rows_'+day)/'receipt.json'),
        sizes=list(SIZES),fixed_cap_shares=M.CAP,seconds=time.monotonic()-start,limits=LIMITS))


def build(day,out,max_windows=None):
    """Denser training rows; old requested ages and existing labels must match."""
    require_day(day);p=M.stage(out,'rows_'+day);start=time.monotonic();data,sealed=load_day(day)
    prior=original_rows(day); old=R.load(day,REFERENCE).set_index(N.KEYS)
    orig={m:g for m,g in prior.groupby('market',sort=False)}
    allrows=[];visited=[];checks=0; model=predictor()
    for row in prior[prior.market.isin(sealed)].to_dict('records'):
        s=sampling(day,row['market'],(row['side'],row['rung'],row['generation']))
        allrows.append({**{k:row[k] for k in N.KEYS+['birth_t']},**s,'status':'sealed_lookback_window'})
    for W in N.E.PAIRS.window_pairs(data):
        market=W[0]['m']; births=[]
        N.run(W,lambda v:True,observe=lambda v,a:births.append(v) if v['status']=='eligible' else None)
        source={tuple(r[k] for k in ('side','rung','generation')):r for r in orig[market].to_dict('records')}
        if {M.key(b) for b in births}!=set(source) or len(births)!=len(source):raise ValueError('neutral action population changed')
        features=N.Features(W);model.clear()
        for b in births:
            oldrow=source[M.key(b)];s=sampling(day,market,M.key(b))
            if s['requested_delay']!=oldrow['requested_delay'] or s['old_selected']!=(oldrow['status']!='not_sampled'):
                raise ValueError('old age or inclusion identity changed')
            row={k:oldrow[k] for k in N.KEYS+['birth_t']};row.update(s);row['status']='not_sampled'
            row['reference']='neutral_POST_KEEP_size30_cap1500';row['continuation']='frozen_bootstrap_start_gap_mask'
            if s['selected']:
                value,x=N.paired(W,b,s['requested_delay'],model,features);row.update(value)
                if x is not None:
                    row.update(zip(N.FEATURES,map(float,x)))
                    row.update(zip(I.NEW,map(float,I.extra_features(W,row,features))))
                    row.update(zip(R.SUPPORT,map(float,R.history(W,row))))
                    if s['old_selected']:
                        saved=old.loc[tuple(row[k] for k in N.KEYS)]
                        cols=I.FEATURES+R.SUPPORT+['qpost','qwait','delta','probe_t','local_post','local_delta','external_post','external_delta']
                        np.testing.assert_allclose([row[k] for k in cols],saved[cols].to_numpy(float),rtol=0,atol=1e-8,equal_nan=True)
                        checks+=1
            allrows.append(row)
        visited.append(market)
        if len(visited)%8==0 or max_windows: print('density',day,len(visited),'windows',round(time.monotonic()-start,1),'s',flush=True)
        I.check_memory()
        if max_windows and len(visited)>=max_windows:break
    if max_windows is None and set(visited) != set(data['W0']):
        raise ValueError('training window population incomplete')
    f=pd.DataFrame(allrows)
    if f.duplicated(N.KEYS).any():raise ValueError('duplicate action')
    eligible=f[f.status=='eligible'];N.check_targets(eligible);weights(eligible);R.represent(eligible,'CONDITION')
    expected=old.reset_index();expected=expected[expected.market.isin(visited)]
    if checks!=len(expected):raise ValueError('old eligible population lost')
    f.to_parquet(p/'rows.parquet',index=False)
    N.write(p/'receipt.json',dict(asof=M.now(),day=day,partial=max_windows is not None,windows=visited,
        statuses=f.status.value_counts().to_dict(),source_status=data['stat'],sealed_lookback_windows=sealed,
        original_actions_checked=checks,rows_sha256=M.digest(p/'rows.parquet'),sampling_version=VERSION,
        size=30,cap=M.CAP,continuation_sha256=M.digest(N.BASE/'bootstrap/receipt.json'),
        source_receipt_sha256=M.digest(REFERENCE/('rows_'+day)/'receipt.json'),seconds=time.monotonic()-start,limits=LIMITS))


def load(day,out):
    p=out/('rows_'+day);r=N.E.read_json(p/'receipt.json')
    if r['partial'] or r['day']!=day or r['sampling_version']!=VERSION or M.digest(p/'rows.parquet')!=r['rows_sha256']:
        raise ValueError('partial or changed rows')
    f=pd.read_parquet(p/'rows.parquet')
    if f.duplicated(N.KEYS).any():raise ValueError('duplicate actions')
    f=f[f.status=='eligible'].copy();weights(f);N.check_targets(f)
    return f


def baseline(tr,te,w,target):
    a=tr[['rung','price',target]].copy();a['w']=w;a['y']=a[target]*w;a['pricebin']=(a.price//10).astype(int)
    rung=a.groupby('rung')[['w','y']].sum();cells=a.groupby(['rung','pricebin']).agg(w=('w','sum'),y=('y','sum'),n=('w','size'))
    mean=a.y.sum()/a.w.sum();pred=[]
    for r,p in zip(te.rung,(te.price//10).astype(int)):
        v=cells.loc[(r,p)] if (r,p) in cells.index and cells.loc[(r,p),'n']>=100 else rung.loc[r] if r in rung.index else None
        pred.append(float(v.y/v.w) if v is not None else float(mean))
    return np.array(pred)


def measure(day,out):
    import lightgbm as lgb
    if day not in M.TESTS:raise ValueError('undeclared evaluation day')
    dates=list(M.FOLDS[day]);p=M.stage(out,'fit_'+day)
    trall=pd.concat([load(d,out) for d in dates],ignore_index=True);test=R.load(day,REFERENCE)
    cells=[];parts=[];models={};support=[]
    params=dict(N.E.PD.LGB_PARAMS,objective='regression')
    for arm in ARMS:
        selected_dates=dates[-2:] if arm=='DENSE_RECENT2' else dates
        train=trall[trall.day.isin(selected_dates)].copy()
        if arm=='SPARSE_ALL':train=train[train.old_selected].copy()
        for side in (0,1):
            tr=train[train.side==side];te=test[test.side==side]
            w=N.S.weights(tr) if arm=='SPARSE_ALL' else weights(tr);wt=N.S.weights(te)
            xx=R.represent(tr,'CONDITION').to_numpy(np.float32);zz=R.represent(te,'CONDITION').to_numpy(np.float32)
            support.append(dict(arm=arm,side=side,train_dates=selected_dates,n=len(tr),later=int((tr.generation>0).sum()),effective_n=N.S.effective(w)))
            pred={}
            for target in I.TARGETS:
                model=lgb.train(params,lgb.Dataset(xx,tr[target].to_numpy(),weight=w/w.mean(),feature_name=R.FEATURES['CONDITION']),N.E.PD.N_ROUNDS)
                fn=p/f'{arm}_{side}_{target}.txt';model.save_model(str(fn));models[fn.name]=M.digest(fn)
                pred[target]=model.predict(zz)
                np.testing.assert_allclose(lgb.Booster(model_file=str(fn)).predict(zz),pred[target],rtol=0,atol=0)
                if arm=='SPARSE_ALL':
                    oldmodel=REFERENCE/('fit_'+day)/f'CONDITION_{side}_{target}.txt'
                    np.testing.assert_allclose(lgb.Booster(model_file=str(oldmodel)).predict(zz),pred[target],rtol=0,atol=1e-10)
                mean=float(np.average(tr[target],weights=w));base=baseline(tr,te,w,target)
                cells.append(dict(day=day,arm=arm,side=side,target=target,
                    price_rung_sse=N.S.metric(te[target],base,wt,mean)['sse'],**N.S.metric(te[target],pred[target],wt,mean)))
            f=te[N.KEYS+['probe_t','qpost','qwait','delta']].copy();f['arm']=arm;f['sampling_weight']=wt
            f['post_hat']=pred['qpost'];f['delta_hat']=pred['delta'];f['post_action']=(pred['qpost']>0)&(pred['delta']>=0);parts.append(f)
            print('fit',day,arm,side,len(tr),flush=True);I.check_memory()
    pd.concat(parts,ignore_index=True).to_parquet(p/'predictions.parquet',index=False)
    N.write(p/'receipt.json',dict(asof=M.now(),day=day,cells=cells,support=support,models=models,params=params,rounds=N.E.PD.N_ROUNDS,
        predictions_sha256=M.digest(p/'predictions.parquet'),row_receipts={d:M.digest(out/('rows_'+d)/'receipt.json') for d in dates},
        evaluation_receipt_sha256=M.digest(REFERENCE/('rows_'+day)/'receipt.json')))


def report(out):
    p=M.stage(out,'report');cells=[];parts=[];sizes=[];validity_receipts={};fits={}
    for d in DAYS:
        v=out/('validity_'+d);r=N.E.read_json(v/'receipt.json')
        if r['partial'] or M.digest(v/'size_windows.parquet')!=r['rows_sha256']:raise ValueError('incomplete validity')
        validity_receipts[d]=r;sizes.append(pd.read_parquet(v/'size_windows.parquet'))
    for d in M.TESTS:
        fp=out/('fit_'+d);r=N.E.read_json(fp/'receipt.json')
        if M.digest(fp/'predictions.parquet')!=r['predictions_sha256']:raise ValueError('changed predictions')
        for n,h in r['models'].items():
            if M.digest(fp/n)!=h:raise ValueError('changed saved model')
        for day,h in r['row_receipts'].items():
            if M.digest(out/('rows_'+day)/'receipt.json')!=h:raise ValueError('changed training receipt')
        fits[d]=M.digest(fp/'receipt.json');cells.extend(r['cells']);parts.append(pd.read_parquet(fp/'predictions.parquet'))
    cf=pd.DataFrame(cells);pooled=[]
    for (arm,side,target),g in cf.groupby(['arm','side','target']):
        control=cf[(cf.arm=='SPARSE_ALL')&(cf.side==side)&(cf.target==target)].set_index('day')
        dense=cf[(cf.arm=='DENSE_ALL')&(cf.side==side)&(cf.target==target)].set_index('day')
        pooled.append(dict(arm=arm,side=int(side),target=target,n=int(g.n.sum()),
            error_reduction_vs_sparse=float(1-g.sse.sum()/control.sse.sum()),
            error_reduction_vs_dense_all=float(1-g.sse.sum()/dense.sse.sum()),
            r2_vs_prior_mean=float(1-g.sse.sum()/g.baseline_sse.sum()),
            error_reduction_vs_price_rung=float(1-g.sse.sum()/g.price_rung_sse.sum()),
            days_better_than_sparse=int(sum(x.sse<control.loc[x.day,'sse']-1e-9 for x in g.itertuples()))))
    pred=pd.concat(parts,ignore_index=True);tests=[]
    for (arm,side),g in pred.groupby(['arm','side']):
        v=I.matched(g,500)
        if v['status']=='matched':
            v['bonferroni_p']=min(1.,v['p_value']*6)
            v['passes']=bool(v['bonferroni_p']<.05 and v['observed_delta']>0 and v['observed_absolute_post']>0)
        tests.append(dict(arm=arm,side=int(side),**v))
    sf=pd.concat(sizes,ignore_index=True);sf.to_parquet(p/'size_windows.parquet',index=False)
    size_totals=[]
    for (day,size,queue),g in sf.groupby(['day','size','queue']):
        n=int(g.accepted.sum());filled=int(g.filled_orders.sum());win=g.drop_duplicates('market')
        size_totals.append(dict(day=day,size=float(size),queue=queue,windows=len(win),orders=int(g.orders.sum()),accepted=n,
            shares=float(g.shares.sum()),reposts=int(g.reposts.sum()),multi_fill_orders=int(g.multi_fill_orders.sum()),
            multi_fill_fraction_all_orders=float(g.multi_fill_orders.sum()/n) if n else None,
            multi_fill_fraction_filled=float(g.multi_fill_orders.sum()/filled) if filled else None,
            mean_resting_ms=float(g.resting_order_ms.sum()/n) if n else None,
            unknown_initial=int(g.unknown_initial.sum()),joint_terminal_pnl=float(win.joint_terminal_pnl.sum()),
            max_abs_inventory=float(win.max_abs_inventory.max()),hard_blocks=int(g.hard_blocks.sum())))
    N.write(p/'result.json',dict(asof=M.now(),n_test_actions=int(len(pred)//len(ARMS)),pooled=pooled,cells=cells,
        matched_tests=tests,economic_passes=sum(bool(t.get('passes',False)) for t in tests),size_totals=size_totals,
        validity=validity_receipts,fit_receipts=fits,new_timing_candidates=2,control_candidates=1,
        size_queue_scenarios=6,model_changes_tested=0,forward_validation=False,promoted=False,limits=LIMITS,
        null='500 draws match day/side/hour/rung/initial-later and exact original evaluation weight mass; Bonferroni 6 arm-side tests',
        next_stage='Review coverage and fixed-evaluation history results before separately freezing model changes or size-specific entry labels.'))
    pd.DataFrame(pooled).to_csv(p/'scores.csv',index=False);pd.DataFrame(size_totals).to_csv(p/'sizes.csv',index=False)


def selftest():
    import test_entry_repair
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_entry_repair))
    if not r.wasSuccessful():raise SystemExit(1)


def main():
    a=argparse.ArgumentParser();a.add_argument('command',choices=['selftest','validity','build','measure','report']);a.add_argument('day',nargs='?')
    a.add_argument('--out',type=Path);a.add_argument('--max-windows',type=int);args=a.parse_args()
    if args.command=='selftest':selftest()
    elif args.command in ('validity','build'):globals()[args.command](args.day,args.out,args.max_windows)
    elif args.command=='measure':measure(args.day,args.out)
    else:report(args.out)


if __name__=='__main__':main()
