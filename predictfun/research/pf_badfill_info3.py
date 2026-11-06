"""Latency-aware guard screen (declared in STATUS.yml 2026-11-04 05:32Z; user: "take latency into account and use the above new guard
model"). Same neutral grid, labels, folds and metrics as pf_badfill_info (step4/badfill_info/grid.parquet). Descriptive, 8 read days.
  M1  LightGBM (pinned PARAMS / ROUNDS) for v per side on the current features (pf_model_spec ALL), refit with the same seed
  M5  M1 + latency-aware inputs from the lead-anchor rows (step4/anchor_eval/rows_lead.parquet, joined on day / mid / tau):
      age_f = ms since the known Predict frame was stamped, dbn_bp = Binance move since that frame, dp = its implied change in
      Predict's price (p_bn at tau minus p_bn at the frame's stamp)
Scores pull the lowest 5-30% of exposures per side per held-out day; bad value avoided / good value lost / net, AUC for bad,
day-clustered 95% intervals (G = 8). Rows the join misses are a status.
Usage: python3 -m predictfun.research.pf_badfill_info3
"""
import os
import numpy as np, pandas as pd, lightgbm as lgb
from . import pf_io as io
from .pf_model_spec import SETS, PARAMS, ROUNDS

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
FE = SETS['ALL']; LAT = ['age_f', 'dbn_bp', 'dp']; PULLS = (0.05, 0.10, 0.20, 0.30)
OUTD = f'{io.OUT}/step4/badfill_info3'; os.makedirs(OUTD, exist_ok=True)
R = pd.read_parquet(f'{io.OUT}/step4/badfill_info/grid.parquet', columns=list(dict.fromkeys(['day', 'mid', 'tau', 'v_bid', 'v_ask', 'bad_bid', 'bad_ask', 'pred_1000', 'y1000'] + FE)))
L = pd.read_parquet(f'{io.OUT}/step4/anchor_eval/rows_lead.parquet', columns=['day', 'mid', 'tau'] + LAT)
n0 = len(R); R = R.merge(L, on=['day', 'mid', 'tau'], how='left', validate='one_to_one'); del L
miss = int(R.dp.isna().sum()); print(f'grid rows {n0}; join misses (status) {miss}; |dp| > 0.5 c in {np.mean(np.abs(R.dp) > 0.5):.1%}', flush=True)
RNG = np.random.default_rng(7)
for side, sg in (('bid', 1.0), ('ask', -1.0)):
    R[f'base_{side}'] = sg * R.pred_1000.values; R[f'ideal_{side}'] = sg * R.y1000.values; R[f'rand_{side}'] = RNG.random(len(R))
    for name, cols in (('M1', FE), ('M5', FE + LAT)):
        pred = np.full(len(R), np.nan)
        for d in DAYS:
            tr = (R.day != d).values; te = ~tr
            ds = lgb.Dataset(R.loc[tr, cols].values.astype(np.float32), R.loc[tr, f'v_{side}'].clip(-10, 10).values, params=PARAMS, free_raw_data=True).construct()
            m = lgb.train(PARAMS, ds, ROUNDS); del ds
            pred[te] = m.predict(R.loc[te, cols].values.astype(np.float32), num_threads=1)
            if name == 'M5' and d == DAYS[0]:
                imp = pd.Series(m.feature_importance('gain'), index=cols).sort_values(ascending=False)
                print(f'M5 {side} gain share of the latency inputs (fold 1): ' + ', '.join(f'{k} {imp[k] / imp.sum():.1%} (rank {list(imp.index).index(k) + 1})' for k in LAT), flush=True)
            del m
        R[f'{name}_{side}'] = pred; io.mem_check(f'{name} {side}'); print(f'fitted {name} {side}', flush=True)


def auc(score, y):
    o = np.argsort(score, kind='stable'); rk = np.empty(len(score)); rk[o] = np.arange(1, len(score) + 1)
    n1 = y.sum(); n0_ = len(y) - n1
    return float((n1 * (len(y) + 1) - rk[y].sum()) / (n1 * n0_)) if n1 and n0_ else np.nan


SC = ['base', 'M1', 'M5', 'ideal', 'rand']; rows = []
for d in DAYS:
    x = R[R.day == d]
    vb = np.r_[x.v_bid.values, x.v_ask.values]; isbad = np.r_[x.bad_bid.values, x.bad_ask.values]; badv = vb[vb < 0].sum(); goodv = vb[vb > 0].sum()
    for sc in SC:
        sco = np.r_[x[f'{sc}_bid'].values, x[f'{sc}_ask'].values]; rec = dict(day=d, score=sc, auc_bad=auc(sco, isbad))
        for k in PULLS:
            pulled = np.zeros(len(vb), bool)
            for half in (slice(0, len(x)), slice(len(x), 2 * len(x))):
                s_ = sco[half]; n = int(round(k * len(s_))); pulled[np.arange(len(vb))[half][np.argsort(s_, kind='stable')[:n]]] = True
            rec[f'bad_avoided_{k}'] = float(vb[pulled & (vb < 0)].sum() / badv); rec[f'good_lost_{k}'] = float(vb[pulled & (vb > 0)].sum() / goodv)
            rec[f'net_{k}'] = float(-vb[pulled].sum() / -badv)
        rows.append(rec)
D = pd.DataFrame(rows); D.to_parquet(f'{OUTD}/by_day.parquet', index=False)
print('\nmean over 8 days: bad value avoided / good value lost / net (share of bad value); M5 vs M1 and vs base, day-clustered 95% CI')
out = {}
for k in PULLS:
    print(f'  pull {k:.0%} of exposures per side')
    for sc in SC:
        x = D[D.score == sc]; line = f"    {sc:5s} {x[f'bad_avoided_{k}'].mean():6.1%} / {x[f'good_lost_{k}'].mean():6.1%} / {x[f'net_{k}'].mean():+6.1%}"
        for ref in (('M1', 'base') if sc == 'M5' else (('base',) if sc == 'M1' else ())):
            dd = x[f'net_{k}'].values - D[D.score == ref][f'net_{k}'].values; se = dd.std(ddof=1) / np.sqrt(len(dd))
            out[f'{sc}_vs_{ref}_{k}'] = dict(mean=float(dd.mean()), lo=float(dd.mean() - 2.365 * se), hi=float(dd.mean() + 2.365 * se), days_better=int((dd > 0).sum()))
            line += f"  | vs {ref} {dd.mean():+5.1%} ({dd.mean() - 2.365 * se:+.1%}..{dd.mean() + 2.365 * se:+.1%}, {int((dd > 0).sum())}/8)"
        print(line)
print('AUC for bad: ' + ', '.join(f"{sc} {D[D.score == sc].auc_bad.mean():.3f}" for sc in SC))
better = sum(out[f'M5_vs_M1_{k}']['lo'] > 0 for k in PULLS)
print(f'M5 beats M1 with the CI excluding 0 at {better} of {len(PULLS)} pull rates -> step 2 (replay) warranted: {better >= 3}')
io.write_json(f'{OUTD}/report.json', dict(join_misses=miss, comparisons=out, m5_beats_m1_pulls=better, replay_warranted=bool(better >= 3),
              means={sc: D[D.score == sc].drop(columns=['day', 'score']).mean().round(4).to_dict() for sc in SC}))
