"""Walk-forward guard predictions (review check 3, declared in STATUS.yml 2026-11-04 13:41Z). Model and thresholds exactly as
pf_step5_guard_preds (M1: LightGBM, pinned PARAMS / ROUNDS, v per side on pf_model_spec ALL, grid step4/badfill_info/grid.parquet;
c(k) = k-quantile of the model's in-sample predictions), but each day d is predicted by a fit on the read days BEFORE d only
(10-27: 10-23..26; 10-31: + 10-27; 11-01; 11-02). Output step5/<day>/preds_w<k>.parquet, same columns as preds_g<k>.
Usage: PF_GUARD_KS=60 python3 -m predictfun.research.pf_step5_guard_preds_wf
"""
import os
import numpy as np, pandas as pd, lightgbm as lgb
from . import pf_io as io
from .pf_model_spec import SETS, PARAMS, ROUNDS

READ = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
TARGET = ['20261027', '20261031', '20261101', '20261102']
FE = SETS['ALL']; KS = tuple(int(x) for x in os.environ.get('PF_GUARD_KS', '60').split(','))
G = pd.read_parquet(f'{io.OUT}/step4/badfill_info/grid.parquet', columns=['day', 'v_bid', 'v_ask'] + FE)
rep = {}
for d in TARGET:
    train = G.day.isin([x for x in READ if x < d]).values; ms, cs = {}, {}
    for side in ('bid', 'ask'):
        X = G.loc[train, FE].values.astype(np.float32)
        ds = lgb.Dataset(X, G.loc[train, f'v_{side}'].clip(-10, 10).values, params=PARAMS, free_raw_data=True).construct()
        m = lgb.train(PARAMS, ds, ROUNDS); del ds
        p = m.predict(X, num_threads=1); ms[side] = m; cs[side] = {k: float(np.quantile(p, k / 100.0)) for k in KS}; del X, p
    r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows_rl_d2.parquet', columns=list(dict.fromkeys(['mid', 'tau', 't_rel', 'pf_mid'] + FE)))
    src = 'preds_c3' if d >= '20261101' else 'preds_o3'
    pr = pd.read_parquet(f'{io.OUT}/step5/{d}/{src}.parquet', columns=['mid', 'tau', 'pred_1000', 'pred_250'])
    assert len(pr) == len(r) and (pr.mid.values == r.mid.values).all() and np.allclose(pr.tau.values, r.tau.values), f'{d}: row alignment'
    live = ((r.t_rel < 240) & r.pf_mid.notna()).values; X = r[FE].values.astype(np.float32)
    vh = {s: np.where(live, ms[s].predict(X, num_threads=1), np.nan) for s in ('bid', 'ask')}
    for k in KS:
        o = pr.copy(); o['v_bid'] = (vh['bid'] - cs['bid'][k]).astype(np.float32); o['v_ask'] = (vh['ask'] - cs['ask'][k]).astype(np.float32)
        o.to_parquet(f'{io.OUT}/step5/{d}/preds_w{k}.parquet', index=False)
    rep[d] = dict(train_days=[x for x in READ if x < d], thresholds=cs, pulled_share={k: {s: float(np.mean(vh[s][live] <= cs[s][k])) for s in vh} for k in KS})
    print(d, rep[d], flush=True); del r, pr, X, ms
    io.mem_check(f'wf {d}')
io.write_json(f'{io.OUT}/step5/guard_preds_wf_report.json', rep)
