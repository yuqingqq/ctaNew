"""Step 5d (exploratory, 2026-11-01): leave-one-day-out over every read day, both targets, ALL features, pinned model.

Each day is predicted by a model fit on the other days (5 of 6 here vs 2 in step 5b's LODO and 3 in C1's apply).
Rows: t < 240 s with a valid known mid and target; train on every 2nd row; targets clipped to +-5 c; PARAMS / ROUNDS from
pf_model_spec (unchanged). Writes step5/<day>/preds_lodo6.parquet (mid, tau, pred_250, pred_1000) and
step5/lodo6_report.json (out-of-day R2 vs zero and corr per day and target).
Usage: python3 -m predictfun.research.pf_step5_lodo6 20261023 20261024 20261025 20261026 20261027 20261031
"""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
from . import pf_io as io
from .pf_model_spec import SETS, PARAMS, ROUNDS

days = sys.argv[1:]
FE = SETS['ALL']; TG = ('y250', 'y1000')
COLS = list(dict.fromkeys(FE + list(TG) + ['t_rel', 'pf_mid']))
Xs, Ys = {}, {}
for d in days:                                     # training subsample (every 2nd valid row), float32
    r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows.parquet', columns=COLS)
    ok = ((r.t_rel < 240) & r.pf_mid.notna() & r.y250.notna() & r.y1000.notna()).values
    Xs[d] = r.loc[ok, FE].values.astype(np.float32)[::2]
    Ys[d] = {t: r.loc[ok, t].values[::2].clip(-5, 5).astype(np.float32) for t in TG}
    del r
    io.mem_check(f'load {d}')
rep = dict(days=days, params='pf_model_spec PARAMS/ROUNDS (pinned)', folds={})
for d in days:
    others = [e for e in days if e != d]
    r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows.parquet', columns=['mid', 'tau'] + COLS)
    ok = ((r.t_rel < 240) & r.pf_mid.notna()).values
    Xd = r[FE].values.astype(np.float32)
    ev = {t: ok & r[t].notna().values for t in TG}; yv = {t: r[t].values[ev[t]] for t in TG}
    out = r[['mid', 'tau']].copy(); rep['folds'][d] = {}
    del r
    for tgt in TG:
        Xtr = np.concatenate([Xs[e] for e in others])
        ds = lgb.Dataset(Xtr, np.concatenate([Ys[e][tgt] for e in others]), params=PARAMS, free_raw_data=True).construct()
        del Xtr
        m = lgb.train(PARAMS, ds, ROUNDS); del ds
        p = m.predict(Xd, num_threads=1).astype(np.float32); del m
        out[f'pred_{tgt[1:]}'] = np.where(ok, p, np.nan).astype(np.float32)
        yt = yv[tgt]; pp = p[ev[tgt]]
        rep['folds'][d][tgt] = dict(n=int(ev[tgt].sum()), r2=round(float(1 - np.sum((yt - pp) ** 2) / np.sum(yt ** 2)), 4),
                                    corr=round(float(np.corrcoef(pp, yt)[0, 1]), 4))
        print(d, tgt, rep['folds'][d][tgt], '[mem]', io.rss_mib(), flush=True)
        io.mem_check(f'{d} {tgt}')
    out.to_parquet(f'{io.OUT}/step5/{d}/preds_lodo6.parquet', index=False)
    del Xd, out
io.write_json(f'{io.OUT}/step5/lodo6_report.json', rep)
