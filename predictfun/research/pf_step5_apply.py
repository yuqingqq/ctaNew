"""Step 5c: train ONE model on the development days and apply it unchanged to later days.
Frozen 2026-11-01 before any later day was read: feature set ALL, target y1000, PARAMS/ROUNDS from
pf_step5_model.py, every 2nd row of t < 240 s with valid known mid and target, targets clipped to +-5 c.
Usage: python3 -m predictfun.research.pf_step5_apply --train 20261023,20261024,20261025 --apply 20261026,20261027,20261031
Writes step5/<day>/preds.parquet (pred_1000; pred_250 = NaN) for every --apply day, plus step5/apply_report.json
(out-of-sample R2/corr and flag diagnostics on the applied days).
"""
import sys, json, argparse
import numpy as np, pandas as pd, lightgbm as lgb
from . import pf_io as io
from .pf_model_spec import SETS, PARAMS, ROUNDS

ap = argparse.ArgumentParser(); ap.add_argument('--train'); ap.add_argument('--apply')
a = ap.parse_args()
train = a.train.split(','); apply_ = a.apply.split(',')
FE = SETS['ALL']; COLS = list(dict.fromkeys(FE + ['y1000', 't_rel', 'pf_mid']))
Xs, ys = [], []
for d in train:
    r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows.parquet', columns=COLS)
    ok = ((r.t_rel < 240) & r.pf_mid.notna() & r.y1000.notna()).values
    Xs.append(r.loc[ok, FE].values.astype(np.float32)[::2]); ys.append(r.loc[ok, 'y1000'].values[::2])
    del r
X = np.concatenate(Xs); y = np.concatenate(ys).clip(-5, 5); del Xs, ys
m = lgb.train(PARAMS, lgb.Dataset(X, y, free_raw_data=True), ROUNDS)
del X, y
m.save_model(f'{io.OUT}/step5/model_y1000_{"_".join(train)}.txt')
rep = dict(train=train, apply=apply_, frozen='ALL features, y1000, pinned PARAMS, ROUNDS=%d' % ROUNDS, days={})
for d in apply_:
    r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows.parquet', columns=['mid', 'tau'] + COLS)
    p = m.predict(r[FE].values.astype(np.float32), num_threads=1).astype(np.float32)
    ok = ((r.t_rel < 240) & r.pf_mid.notna() & r.y1000.notna()).values
    pr = r[['mid', 'tau']].copy(); pr['pred_250'] = np.float32(np.nan)
    pr['pred_1000'] = np.where(ok, p, np.nan).astype(np.float32)
    pr.to_parquet(f'{io.OUT}/step5/{d}/preds.parquet', index=False)
    yt = r.y1000.values[ok]; pp = p[ok]
    fl = {}
    for th in (0.5, 1.0):
        dn = pp <= -th; up = pp >= th
        fl[f'th{th}'] = dict(share=round(float((dn | up).mean()), 4), y_when_down=round(float(yt[dn].mean()), 3), y_when_up=round(float(yt[up].mean()), 3))
    rep['days'][d] = dict(n=int(ok.sum()), r2=round(float(1 - np.sum((yt - pp) ** 2) / np.sum(yt ** 2)), 4),
                          corr=round(float(np.corrcoef(pp, yt)[0, 1]), 4), flags=fl)
    print(d, rep['days'][d], flush=True)
    del r, pr
    io.mem_check(f'apply {d}')
io.write_json(f'{io.OUT}/step5/apply_report_{"_".join(apply_)}.json', rep)
