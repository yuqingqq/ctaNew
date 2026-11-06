"""Round O3 predictions (2026-11-03): pred_1000 = model A at the Tokyo clock with 2 ms decision (preds_rl_a_d2, unchanged)
plus pred_250 = a y250 model refit on Tokyo-clock rows (rows_rl_d2), leave-one-day-out over the six read days, same spec
otherwise (ALL features, t < 240 s with valid known mid, y250 and y1000, every 2nd row, target clipped to +-5 c, pinned
PARAMS / ROUNDS). y250 = the venue mid 250 ms ahead minus the known mid, about the time our cancel needs to land.
Writes step5/<day>/preds_o3.parquet and step5/o3_preds_report.json. Usage: python3 -m predictfun.research.pf_step5_o3_preds
"""
import numpy as np, pandas as pd, lightgbm as lgb
from . import pf_io as io
from .pf_model_spec import SETS, PARAMS, ROUNDS

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031']
FE = SETS['ALL']; TC = ['t_rel', 'pf_mid', 'y250', 'y1000']; STEM = 'rows_rl_d2'


def mask(r):
    return ((r.t_rel < 240) & r.pf_mid.notna() & r.y250.notna() & r.y1000.notna()).values


def fit(train):
    n = {}
    for d in train:
        r = pd.read_parquet(f'{io.OUT}/step5/{d}/{STEM}.parquet', columns=TC); n[d] = len(np.flatnonzero(mask(r))[::2]); del r
    X = np.empty((sum(n.values()), len(FE)), np.float32); y = np.empty(len(X), np.float32); i = 0
    for d in train:
        r = pd.read_parquet(f'{io.OUT}/step5/{d}/{STEM}.parquet', columns=list(dict.fromkeys(FE + TC)))
        k = np.flatnonzero(mask(r))[::2]
        X[i:i + len(k)] = r[FE].values[k].astype(np.float32); y[i:i + len(k)] = r.y250.values[k].clip(-5, 5); i += len(k); del r, k
    ds = lgb.Dataset(X, y, params=PARAMS, free_raw_data=True).construct(); del X, y
    m = lgb.train(PARAMS, ds, ROUNDS); del ds
    io.mem_check('fit y250')
    return m


rep = {}
for d in DAYS:
    r = pd.read_parquet(f'{io.OUT}/step5/{d}/{STEM}.parquet', columns=['mid', 'tau'] + list(dict.fromkeys(FE + TC)))
    ok = ((r.t_rel < 240) & r.pf_mid.notna()).values; ev = ok & r.y250.notna().values
    m = fit([e for e in DAYS if e != d]); p = m.predict(r[FE].values.astype(np.float32), num_threads=1).astype(np.float32); del m
    a = pd.read_parquet(f'{io.OUT}/step5/{d}/preds_rl_a_d2.parquet')
    assert (a.mid.values == r.mid.values).all() and np.allclose(a.tau.values, r.tau.values), 'row alignment'
    a['pred_250'] = np.where(ok, p, np.nan).astype(np.float32)
    a.to_parquet(f'{io.OUT}/step5/{d}/preds_o3.parquet', index=False)
    yt = r.y250.values[ev]; pp = p[ev]
    rep[d] = dict(n=int(ev.sum()), r2_250=round(float(1 - np.sum((yt - pp) ** 2) / np.sum(yt ** 2)), 4), corr_250=round(float(np.corrcoef(pp, yt)[0, 1]), 4),
                  flagged={th: round(float((np.abs(p[ok]) >= th).mean()), 4) for th in (0.25, 0.5, 1.0)})
    print(d, rep[d], '[mem]', io.rss_mib(), flush=True)
    del r, a, p
io.write_json(f'{io.OUT}/step5/o3_preds_report.json', rep)
