"""Step 5b (exploratory): leave-one-day-out LightGBM for the Predict mid move 250 ms / 1 s ahead.

Rows: step5/<day>/rows.parquet, live t < 240 s, valid known mid and target. Train on every 2nd row (100 ms).
Feature sets: BN (Binance only), PF (Predict book/flow + contract), ALL. Params pinned (no tuning).
Scores on the held-out day: R2 vs zero, Pearson corr, and the realised move when the prediction (or the
G1/G2 rule) flags a side. Out-of-day predictions (ALL) are written for the policy replay.
Usage: python3 -m predictfun.research.pf_step5_model 20261023 20261024 20261025
"""
import sys, os, json
import numpy as np, pandas as pd, lightgbm as lgb
from . import pf_io as io

days = sys.argv[1:]
from .pf_model_spec import BN, PF, SETS, PARAMS, ROUNDS
COLS = list(dict.fromkeys(BN + PF + ['y250', 'y1000', 't_rel', 'pf_mid']))
X, Y, IDX, NROW = {}, {}, {}, {}
for d in days:
    r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows.parquet', columns=COLS)
    NROW[d] = len(r)
    ok = ((r.t_rel < 240) & r.pf_mid.notna() & r.y250.notna() & r.y1000.notna()).values
    IDX[d] = np.where(ok)[0]
    X[d] = r.loc[ok, SETS['ALL']].values.astype(np.float32)
    Y[d] = {t: r.loc[ok, t].values.astype(np.float64) for t in ('y250', 'y1000')}
    Y[d]['r1000'] = r.loc[ok, 'bn_r1000'].values; Y[d]['r250'] = r.loc[ok, 'bn_r250'].values
    del r
    print(d, 'rows', NROW[d], 'ok', len(IDX[d]), '[mem]', io.rss_mib(), flush=True)
out = {}; preds = {d: {} for d in days}
for tgt in ('y250', 'y1000'):
    for name, feats in SETS.items():
        res = {}
        for d in days:
            others = [e for e in days if e != d]
            ci = [SETS['ALL'].index(f) for f in feats]
            Xtr = np.concatenate([X[e][::2][:, ci] for e in others]); y = np.concatenate([Y[e][tgt][::2] for e in others]).clip(-5, 5)
            ds = lgb.Dataset(Xtr, y, free_raw_data=True)
            m = lgb.train(PARAMS, ds, ROUNDS)
            del Xtr, ds
            p = m.predict(X[d][:, ci], num_threads=1)
            yt = Y[d][tgt]
            r2 = 1 - np.sum((yt - p) ** 2) / np.sum(yt ** 2)
            flags = {}
            for th in (0.25, 0.5, 1.0):
                dn = p <= -th; up = p >= th
                flags[f'th{th}'] = dict(share=round(float((dn | up).mean()), 4),
                                        y_when_down=round(float(yt[dn].mean()), 3) if dn.any() else None,
                                        y_when_up=round(float(yt[up].mean()), 3) if up.any() else None)
            res[d] = dict(n_test=int(len(yt)), r2=round(float(r2), 4), corr=round(float(np.corrcoef(p, yt)[0, 1]), 4), flags=flags)
            if name == 'ALL':
                full = np.full(NROW[d], np.nan, np.float32); full[IDX[d]] = p; preds[d][tgt] = full
                if tgt == 'y1000':
                    imp = pd.Series(m.feature_importance('gain'), index=feats).sort_values(ascending=False)
                    res[d]['top_gain'] = (imp / imp.sum()).round(3).head(10).to_dict()
            del m
            io.mem_check(f'{tgt} {name} {d}')
        out[f'{tgt}/{name}'] = res
        print(tgt, name, {d: (v['r2'], v['corr']) for d, v in res.items()}, flush=True)
# rule benchmark: D1's G1 / G2 flags on the same rows
bench = {}
for d in days:
    r1, r2_, y1, y2 = Y[d]['r1000'], Y[d]['r250'], Y[d]['y1000'], Y[d]['y250']
    dn = (r1 <= -0.5) | (r2_ <= -0.5); up = (r1 >= 0.5) | (r2_ >= 0.5)
    bench[d] = dict(share=round(float((dn | up).mean()), 4), y1000_when_down=round(float(y1[dn].mean()), 3),
                    y1000_when_up=round(float(y1[up].mean()), 3), y250_when_down=round(float(y2[dn].mean()), 3))
out['rule_G1_G2_flags'] = bench
for d in days:
    pr = pd.read_parquet(f'{io.OUT}/step5/{d}/rows.parquet', columns=['mid', 'tau'])
    for tgt in ('y250', 'y1000'):
        pr[f'pred_{tgt[1:]}'] = preds[d].get(tgt, np.full(len(pr), np.nan, np.float32))
    pr.to_parquet(f'{io.OUT}/step5/{d}/preds.parquet', index=False)
    del pr
io.write_json(f'{io.OUT}/step5/model_report_{"_".join(days)}.json', out)
print(json.dumps(out, indent=1, default=str)[:6000])
