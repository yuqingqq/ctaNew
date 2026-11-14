"""Predeclared review controls and bounded chronological diagnostic of frozen v1.

This does not repair/change v1's target, features, model parameters or policy.
It trains on the first 24 eligible windows of each of 10-23,24,25, skips 10-26
calibration (threshold fixed at zero), and replays the first 12 windows of
10-27. Prefixes are defined before reading their results. They are NOT complete
days, cannot establish development acceptance, and receive no intervals.
All original entry comparators, clocks and neutral keeper are retained.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_wait_post as W
import pf_entry_value as E

TRAIN = ('20261023', '20261024', '20261025')
TEST = '20261027'
TRAIN_WINDOWS = 24
TEST_WINDOWS = 12


def review_controls():
    from test_entry_value import context, Score
    cases = []
    # Both branches fill after both order activations, so the one-step target
    # is zero for BOTH positive and negative terminal entry value.
    for adverse in (False, True):
        if adverse:
            import pf_joint_ladder as JL
            bid=[(0., [(50,100),(49,100),(48,100),(47,100),(46,100)],52), (500.,[(40,100)],42)]
            ask=[(0., [(52,100),(53,100),(54,100),(55,100),(56,100)],50), (500.,[(42,100)],40)]
            tape={0:JL._ctx(0,bid,[(400.,49,30.)],t_end=1000.),1:JL._ctx(1,ask,[],t_end=1000.)}
            for w in tape.values():
                w['vkl']=[-1.]*len(w['tg']); w['m']=1
                w['tf_mff']=(w['tf'],np.array([51.,41.]))
        else:
            tape = context([(400., 49, 30.)])
        seen = []; base = E.run(tape, lambda v: True, observe=lambda v, a: seen.append(v))
        entry = next(v for v in seen if v['kind']=='entry' and v['side']==0 and v['rung']==0)
        pair = E.paired_probe(tape, entry, base)
        learned = E.run(tape, W.policy(tape, {0: Score(0.), 1: Score(0.)}))
        cases.append(dict(adverse=adverse, **pair, zero_score_pnl=E.terminal(tape, learned)['pnl'],
                          zero_score_orders=sum(map(len, learned['order_states'].values()))))
    assert cases[0]['value_yes'] > 0 and cases[1]['value_yes'] < 0
    assert all(abs(r['target_c']) < 1e-10 and r['zero_score_orders']==0 for r in cases)
    # Positive contrast: a trade between activation times must NOT return zero.
    tape = context([(95., 49, 30.)]); seen = []
    base = E.run(tape, lambda v: True, observe=lambda v, a: seen.append(v))
    entry = next(v for v in seen if v['kind']=='entry' and v['side']==0 and v['rung']==0)
    control = E.paired_probe(tape, entry, base)
    assert control['target_c'] > 0
    # Wrong/shared side routing is detected with opposite fixed forecasts.
    tape = context(); out = E.run(tape, W.policy(tape, {0: Score(1.), 1: Score(-1.)}))
    assert out['order_states'][(0,0)] and not out['order_states'][(1,0)]
    return dict(zero_contrast_cases=cases, between_activations_positive_control=control,
                target_does_not_identify_absolute_entry_value=True, independent_side_routing=True)


def metadata():
    return dict(candidate=W.stamp(), review_source_sha256=E.digest(Path(__file__)),
                train=list(TRAIN), test=TEST, train_prefix_windows=TRAIN_WINDOWS,
                test_prefix_windows=TEST_WINDOWS, partial_population=True, forward_validation=False)


def check_review(meta):
    expected=json.loads(json.dumps(metadata()))
    if meta != expected: raise ValueError('pilot source/design changed')


def run(out):
    import lightgbm as lgb
    out=Path(out); out.mkdir(); initial=metadata(); E.write_json(out/'design.json', initial)
    controls=review_controls(); E.write_json(out/'controls.json', controls)
    progress=lambda stage: E.write_json(out/'progress.json',dict(stage=stage,asof=W.now()))
    frame=[]; support={}
    for day in (*TRAIN, TEST):
        progress('rows_'+day)
        W.rows_day(day,out,max_windows=TRAIN_WINDOWS if day in TRAIN else TEST_WINDOWS)
        p=out/('rows_'+day); meta=E.read_json(p/'receipt.json'); W.require(meta)
        if meta['day'] != day or not meta['partial'] or E.digest(p/'rows.parquet') != meta['rows_sha256']:
            raise ValueError('pilot row identity mismatch')
        rows=pd.read_parquet(p/'rows.parquet')
        if not (rows.day == day).all() or rows.duplicated(['day','market','side','rung','generation']).any():
            raise ValueError('pilot action population mismatch')
        support[day]={k:meta[k] for k in ('asof','windows','markets','source_status','action_status','support')}
        frame.append(rows)
    data=pd.concat(frame,ignore_index=True); eligible=data[data.status=='eligible']
    train=eligible[eligible.day.isin(TRAIN)]; test=eligible[eligible.day==TEST].copy()
    models={}; counts={}; params=dict(E.PD.LGB_PARAMS,objective='regression')
    progress('fit'); started=time.monotonic()
    for side,name in W.SIDE_NAMES.items():
        r=train[train.side==side]
        if len(r)<200 or r.day.nunique()<2 or (r.target_c.abs()>1e-10).sum()<20:
            E.write_json(out/'insufficient_support.json',dict(side=name,n=len(r),nonzero=int((r.target_c.abs()>1e-10).sum()),asof=W.now()))
            raise ValueError('predeclared training support gate failed: '+name)
        if len(r)>E.PD.SUB_N: r=r.sample(E.PD.SUB_N,random_state=E.SEED)
        y=r.target_c.to_numpy(float)
        if not np.isfinite(y).all(): raise ValueError('invalid target')
        models[side]=lgb.train(params,lgb.Dataset(r[W.FEATURES].to_numpy(np.float32),np.clip(y,-10.,10.),feature_name=W.FEATURES),E.PD.N_ROUNDS)
        models[side].save_model(str(out/(name+'.txt')))
        counts[name]=dict(n=len(r),nonzero=int((abs(y)>1e-10).sum()),positive=int((y>1e-10).sum()),negative=int((y< -1e-10).sum()))
        mask=test.side==side; test.loc[mask,'prediction']=models[side].predict(test.loc[mask,W.FEATURES].to_numpy(np.float32))
    test.to_parquet(out/'test_predictions.parquet',index=False)
    E.write_json(out/'fit.json',dict(asof=W.now(),train=list(TRAIN),test=TEST,calibration_day='20261026',
        calibration_used=False,threshold=0,training_support=counts,params=params,rounds=E.PD.N_ROUNDS,
        models={name:E.digest(out/(name+'.txt')) for name in W.SIDE_NAMES.values()},wall_seconds=time.monotonic()-started))
    progress('replay'); tape=E.K.load_day(TEST); records=[]; inference=[]; diagnostics=collections.Counter(); wanted=support[TEST]['markets']; seen=[]
    for tape_window in E.PAIRS.window_pairs(tape):
        market=tape_window[0]['m']
        if len(seen)>=TEST_WINDOWS: break
        if market!=wanted[len(seen)]: raise ValueError('pilot replay/label window identities differ')
        seen.append(market)
        for shift in W.CFGS:
            for arm in W.ARMS:
                audit=collections.Counter(); call_count=[0]; timing=[]
                def capture(v,x,score,action):
                    call_count[0]+=1
                    if call_count[0]%500==1:
                        inference.append(dict(market=market,side=v['side'],shift=int(shift),k=v['k'],prediction=score,
                                              **dict(zip(W.FEATURES,map(float,x)))))
                choose=W.policy(tape_window,models,arm,audit,capture)
                def timed(v):
                    if v['kind']!='entry' or arm!='VALUE': return choose(v)
                    start=time.perf_counter_ns(); result=choose(v); timing.append((time.perf_counter_ns()-start)/1e6); return result
                result=E.run(tape_window,timed,shift=shift,entry_mode='legacy' if arm=='TIMER' else 'value')
                diagnostics.update({f'{arm}:{shift}:{k}':v for k,v in audit.items()})
                records.append(dict(day=TEST,market=market,arm=arm,shift=int(shift),**E.metrics(tape_window,result),
                    entry_calls=len(timing),entry_call_ms_sum=float(sum(timing)),entry_call_ms_max=float(max(timing)) if timing else 0.))
        print('pilot replay',len(seen),'/',TEST_WINDOWS,'RSS',round(E.rss_mib()),flush=True)
        pd.DataFrame(records).to_parquet(out/'windows.partial.parquet',index=False)
        if E.rss_mib()>2048: raise RuntimeError('RSS cap exceeded')
    if seen!=wanted: raise ValueError('pilot window population changed')
    full=pd.DataFrame(records)
    if len(full)!=TEST_WINDOWS*len(W.ARMS)*len(W.CFGS) or full.duplicated(['market','arm','shift']).any():
        raise ValueError('pilot replay incomplete or duplicated')
    full.to_parquet(out/'windows.parquet',index=False); saved=pd.DataFrame(inference)
    saved.to_parquet(out/'inference.parquet',index=False)
    for side,g in saved.groupby('side'):
        if not np.allclose(models[side].predict(g[W.FEATURES].to_numpy(np.float32)),g.prediction,atol=1e-9,rtol=0):
            raise ValueError('inference reconstruction failed')
    cols=['pnl','pnl_without_young','shares','places','pulls','inventory_square_seconds','any_quote_steps','both_quote_steps','grid_steps','entry_calls','entry_call_ms_sum']
    summary=full.groupby(['arm','shift'])[cols].sum().reset_index()
    summary['any_quote_fraction']=summary.any_quote_steps/summary.grid_steps
    summary['both_quote_fraction']=summary.both_quote_steps/summary.grid_steps
    summary['c_per_share']=np.where(summary.shares>0,100*summary.pnl/summary.shares,np.nan)
    summary.to_parquet(out/'summary.parquet',index=False)
    local=[]
    for side,g in test.groupby('side'):
        local.append(dict(side=int(side),n=len(g),nonzero=int((g.target_c.abs()>1e-10).sum()),post_share=float((g.prediction>0).mean()),
            mean_prediction=float(g.prediction.mean()),mean_target=float(g.target_c.mean()),local_null=E.local_null(g)))
    check_review(json.loads(json.dumps(initial)))
    result=dict(asof=W.now(),status='diagnostic_complete_review_not_passed',n_test_days=1,complete_test_days=0,
        n_test_windows=len(wanted),trained_models=True,development_acceptance=False,forward_validation=False,
        training_support=counts,source_support=support,local_selection=local,decision_counts=dict(diagnostics),
        summary=json.loads(summary.to_json(orient='records')),controls=controls,
        result_sha256={n:E.digest(out/n) for n in ('windows.parquet','inference.parquet','test_predictions.parquet','summary.parquet')},
        limits=['first 12 test windows only, first 24 windows on each training day; no intervals or generalization claim',
                'one-decision target and repeated-wait population mismatch remain; not a repaired model',
                'neutral keeper, gross simulated cash+inventory; not combined strategy or live economics'])
    E.write_json(out/'result.json',result); progress('complete')
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['controls','pilot']);p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.command=='controls': print(json.dumps(review_controls(),indent=2))
    else: run(args.out)
