"""Leave-one-day-out models for the measured Tokyo clock (2026-11-02; user: tests at real latency values).

For each of the six read days d, two y1000 models fit on the other five days (spec = preds_lodo6 / C2: ALL features,
train rows t < 240 s with valid known mid, y250 and y1000, every 2nd row, target clipped to +-5 c, pinned PARAMS/ROUNDS):
  A  trained on rows.parquet (tokyo20 clock) = the frozen live model's spec  -> step5/<d>/preds_rl_a.parquet
  B  trained on rows_rl.parquet (Tokyo clock) = a model refit for real latency -> step5/<d>/preds_rl_b.parquet
Both predict rows_rl.parquet of d (features as the live system would see them). A's fits equal preds_lodo6's y1000 fits
(same rows, params, seed), so A isolates the clock. Training matrix preallocated per fold; one fold at a time.
Usage: python3 -m predictfun.research.pf_step5_lodo_rl
Report: step5/lodo_rl_report.json (out-of-day R2 / corr of A and B on the Tokyo-clock target, and of preds_lodo6 on tokyo20)
"""
import numpy as np, pandas as pd, lightgbm as lgb
from . import pf_io as io
from .pf_model_spec import SETS, PARAMS, ROUNDS
from . import pf_clock_rl as CK
import os
ONLY_A = os.environ.get('PF_RL_ONLY_A') == '1'      # decision-time variants: only the frozen-spec model A

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031']
FE = SETS['ALL']; TC = ['t_rel', 'pf_mid', 'y250', 'y1000']


def mask(r):
    return ((r.t_rel < 240) & r.pf_mid.notna() & r.y250.notna() & r.y1000.notna()).values


def fit(train, stem):
    n = {}
    for d in train:
        r = pd.read_parquet(f'{io.OUT}/step5/{d}/{stem}.parquet', columns=TC); n[d] = len(np.flatnonzero(mask(r))[::2]); del r
    X = np.empty((sum(n.values()), len(FE)), np.float32); y = np.empty(len(X), np.float32); i = 0
    for d in train:
        r = pd.read_parquet(f'{io.OUT}/step5/{d}/{stem}.parquet', columns=list(dict.fromkeys(FE + TC)))
        k = np.flatnonzero(mask(r))[::2]
        X[i:i + len(k)] = r[FE].values[k].astype(np.float32); y[i:i + len(k)] = r.y1000.values[k].clip(-5, 5); i += len(k); del r, k
    ds = lgb.Dataset(X, y, params=PARAMS, free_raw_data=True).construct(); del X, y
    m = lgb.train(PARAMS, ds, ROUNDS); del ds
    io.mem_check(f'fit {stem}')
    return m


def score(p, y):
    return dict(n=int(len(y)), r2=round(float(1 - np.sum((y - p) ** 2) / np.sum(y ** 2)), 4), corr=round(float(np.corrcoef(p, y)[0, 1]), 4))


rep = {}
for d in DAYS:
    others = [e for e in DAYS if e != d]
    r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows_rl{CK.SFX}.parquet', columns=['mid', 'tau'] + list(dict.fromkeys(FE + TC)))
    ok = ((r.t_rel < 240) & r.pf_mid.notna()).values; ev = ok & r.y1000.notna().values; Xd = r[FE].values.astype(np.float32)
    yv = r.y1000.values[ev]; base = r[['mid', 'tau']].copy(); del r
    rep[d] = {}
    for tag, stem in ((('a', 'rows'),) if ONLY_A else (('a', 'rows'), ('b', 'rows_rl' + CK.SFX))):
        m = fit(others, stem); p = m.predict(Xd, num_threads=1).astype(np.float32); del m
        out = base.copy(); out['pred_250'] = np.float32(np.nan); out['pred_1000'] = np.where(ok, p, np.nan).astype(np.float32)
        out.to_parquet(f'{io.OUT}/step5/{d}/preds_rl_{tag}{CK.SFX}.parquet', index=False)
        rep[d][tag] = dict(score(p[ev], yv), flagged_025=round(float((np.abs(p[ok]) >= 0.25).mean()), 4))
        print(d, tag, rep[d][tag], '[mem]', io.rss_mib(), flush=True)
        del out, p
    t20 = pd.read_parquet(f'{io.OUT}/step5/{d}/rows.parquet', columns=['t_rel', 'pf_mid', 'y1000'])
    p6 = pd.read_parquet(f'{io.OUT}/step5/{d}/preds_lodo6.parquet', columns=['pred_1000']).pred_1000.values
    e6 = ((t20.t_rel < 240) & t20.pf_mid.notna() & t20.y1000.notna()).values & np.isfinite(p6)
    rep[d]['lodo6_on_tokyo20'] = score(p6[e6], t20.y1000.values[e6]); del t20, p6, Xd
    io.mem_check(f'day {d}')
io.write_json(f'{io.OUT}/step5/lodo_rl_report{CK.SFX}.json', rep)
