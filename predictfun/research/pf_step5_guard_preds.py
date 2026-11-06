"""Stage 3 guard predictions (declared in STATUS.yml 2026-11-04 04:28Z). Model M1 of pf_badfill_info: LightGBM (pinned PARAMS /
ROUNDS) for v per side (the value of keeping an order at the known best through the next exposure, clipped +-10) on the current
features (pf_model_spec ALL), trained on the neutral grid (step4/badfill_info/grid.parquet).
  read days: leave one day out over the 8 read days; the held-out day's every 50 ms row is predicted
  11-03 (never seen by any model here): one fit on all 8 read days
Pull thresholds c(k), k = 10 / 20 / 30% (PF_GUARD_KS, e.g. 40,50): the k-quantile of the model's in-sample predictions on its training rows, per side. Output
per k: step5/<day>/preds_g<k>.parquet = mid, tau, pred_1000 / pred_250 copied from preds_o3 (read days) or preds_c3 (fresh days),
v_bid / v_ask = prediction - c(k) (the engine's V0 token pulls a side when its value <= 0); NaN outside live rows (t_rel >= 240 or
no known mid). Usage: python3 -m predictfun.research.pf_step5_guard_preds [fresh_day ...]
"""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from . import pf_io as io
from .pf_model_spec import SETS, PARAMS, ROUNDS

READ = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
FRESH = [d for d in sys.argv[1:]]
import os
FE = SETS['ALL']; KS = tuple(int(x) for x in os.environ.get('PF_GUARD_KS', '10,20,30').split(','))
GRID = os.environ.get('PF_GRID', f'{io.OUT}/step4/badfill_info/grid.parquet'); ROWS = os.environ.get('PF_ROWS', 'rows_rl_d2'); PSTEM = os.environ.get('PF_PSTEM', 'g')
G = pd.read_parquet(GRID, columns=['day', 'v_bid', 'v_ask'] + FE)   # 2026-11-04 review: PF_GRID / PF_ROWS / PF_PSTEM (defaults = the original run)
rep = {}
NOSRC = os.environ.get('PF_NOSRC', '') == '1'


def fit(train):
    ms, cs = {}, {}
    for side in ('bid', 'ask'):
        X = G.loc[train, FE].values.astype(np.float32)
        ds = lgb.Dataset(X, G.loc[train, f'v_{side}'].clip(-10, 10).values, params=PARAMS, free_raw_data=True).construct()
        m = lgb.train(PARAMS, ds, ROUNDS); del ds
        p = m.predict(X, num_threads=1); ms[side] = m; cs[side] = {k: float(np.quantile(p, k / 100.0)) for k in KS}; del X, p
    return ms, cs


def write(d, ms, cs, src):
    r = pd.read_parquet(f'{io.OUT}/step5/{d}/{ROWS}.parquet', columns=list(dict.fromkeys(['mid', 'tau', 't_rel', 'pf_mid'] + FE)))
    if NOSRC:                       # PF_NOSRC=1 (2026-11-05): rows without a 1 s / 250 ms prediction file (every-frame rows); merged downstream
        pr = r[['mid', 'tau']].copy(); pr['pred_1000'] = np.nan; pr['pred_250'] = np.nan
    else:
        pr = pd.read_parquet(f'{io.OUT}/step5/{d}/{src}.parquet', columns=['mid', 'tau', 'pred_1000', 'pred_250'])
    assert len(pr) == len(r) and (pr.mid.values == r.mid.values).all() and np.allclose(pr.tau.values, r.tau.values), f'{d}: row alignment'
    live = ((r.t_rel < 240) & r.pf_mid.notna()).values; X = r[FE].values.astype(np.float32)
    vh = {s: np.where(live, ms[s].predict(X, num_threads=1), np.nan) for s in ('bid', 'ask')}
    for k in KS:
        o = pr.copy(); o['v_bid'] = (vh['bid'] - cs['bid'][k]).astype(np.float32); o['v_ask'] = (vh['ask'] - cs['ask'][k]).astype(np.float32)
        o.to_parquet(f'{io.OUT}/step5/{d}/preds_{PSTEM}{k}.parquet', index=False)
    rep[d] = dict(thresholds={s: cs[s] for s in cs}, pulled_share={k: {s: float(np.mean(vh[s][live] <= cs[s][k])) for s in ('bid', 'ask')} for k in KS})
    print(d, src, rep[d], flush=True); del r, pr, X


for d in READ:
    ms, cs = fit((G.day != d).values); write(d, ms, cs, 'preds_c3' if d >= '20261101' else 'preds_o3'); del ms
    io.mem_check(f'fold {d}')
if FRESH:
    ms, cs = fit(np.ones(len(G), bool))
    for d in FRESH:
        if d in READ:
            sys.exit(f'REFUSED: {d} is a read day')
        write(d, ms, cs, 'preds_c3')
io.write_json(f"{io.OUT}/step5/guard_preds_report{'' if PSTEM == 'g' else '_' + PSTEM}{'' if KS == (10, 20, 30) else '_' + '_'.join(map(str, KS))}.json", rep)
