"""Fitted anchor (declared in STATUS.yml 2026-11-04 04:30Z; user: "use Binance market to fit the [Predict.fun] price ... to get a
better anchor"). Descriptive, 8 read days, nothing selected.

A5 = LightGBM (pinned PARAMS / ROUNDS), leave one day out, predicting Predict's CURRENT venue mid at tau (the mid already set at the
venue, which reaches us ~50 ms later) from fresh Binance-side inputs only (no Predict input): z = ln(S (1 + basis) / K) /
(sigma sqrt(tau_rem)), ln(S (1 + basis) / K) in bp, sigma sqrt(tau_rem), tau_rem, basis_bps, bn_r25..bn_r3000, bn_imb, bn_flow250 /
1000 / 5000, bn_ntr1000, bn_age (S = Binance mid known at tau, K = the window's start price). Rows: those of pf_anchor_eval
(step4/anchor_eval/rows.parquet: every 500 ms, live 0-240 s, with the outcome and Predict's mid changes now / +1 / +10 / +60 s).
Measures and gate as the redeclared stage 1: settlement Brier vs A0 (Predict mid as known, ~50 ms late), convergence slopes of
A5 - A0, gate = Brier within +0.005 of A0 on >= 6/8 days AND +10 s slope >= 0.3 with CI excluding 0. Also the fit's error to the
current venue mid, against A0's (the late mid's) error to it.
Usage: python3 -m predictfun.research.pf_anchor_fit
"""
import numpy as np, pandas as pd, lightgbm as lgb
from . import pf_io as io
from .pf_model_spec import PARAMS, ROUNDS

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
H = (0.0, 1000.0, 10000.0, 60000.0)
BNX = ['bn_r25', 'bn_r50', 'bn_r100', 'bn_r250', 'bn_r500', 'bn_r1000', 'bn_r3000', 'bn_imb', 'bn_flow250', 'bn_flow1000', 'bn_flow5000', 'bn_ntr1000', 'bn_age']
R = pd.read_parquet(f'{io.OUT}/step4/anchor_eval/rows.parquet')
parts = []
for d in DAYS:
    x = pd.read_parquet(f'{io.OUT}/step5/{d}/rows_rl_d2.parquet', columns=['mid', 'tau', 'bn_mid'] + BNX)
    reg = pd.read_parquet(f'{io.OUT}/step1/{d}/registry.parquet', columns=['mid', 'start_price'])
    r = R[R.day == d].merge(x, on=['mid', 'tau'], how='left', validate='one_to_one').merge(reg, on='mid', how='left')
    parts.append(r); del x
R = pd.concat(parts, ignore_index=True); del parts
lsk = 1e4 * np.log(R.bn_mid * (1 + R.basis_bps.fillna(0) / 1e4) / R.start_price); sst = R.sigma * np.sqrt(R.tau_rem)
R['lnSK_bp'] = lsk; R['sst'] = sst; R['z'] = (lsk / 1e4) / sst
FX = ['z', 'lnSK_bp', 'sst', 'tau_rem', 'basis_bps'] + BNX
R['cur'] = R.A0 + R.y0                                                    # Predict's current venue mid (not yet received)
ok = np.isfinite(R.cur.values) & np.isfinite(R.z.values)
print(f'rows {len(R)}, usable {int(ok.sum())} (no current mid or no z: {int((~ok).sum())})', flush=True)
R['A5'] = np.nan
for d in DAYS:
    tr = ok & (R.day != d).values; te = (R.day == d).values
    ds = lgb.Dataset(R.loc[tr, FX].values.astype(np.float32), R.loc[tr, 'cur'].values, params=PARAMS, free_raw_data=True).construct()
    m = lgb.train(PARAMS, ds, ROUNDS); del ds
    R.loc[te, 'A5'] = np.clip(m.predict(R.loc[te, FX].values.astype(np.float32), num_threads=1), 0.5, 99.5)
    if d == DAYS[0]:
        imp = pd.Series(m.feature_importance('gain'), index=FX).sort_values(ascending=False)
        print('top inputs (fold 1):', ', '.join(f'{k} {v / imp.sum():.0%}' for k, v in imp.head(8).items()), flush=True)
    del m
R = R[ok].reset_index(drop=True)
print('\nfit to the current venue mid (c, mean abs error):', f"A5 {np.mean(np.abs(R.A5 - R.cur)):.3f}", f"A2 p_bn {np.mean(np.abs(R.A2 - R.cur)):.3f}",
      f"A0 late Predict mid {np.mean(np.abs(R.A0 - R.cur)):.3f}")
br = {c: np.array([np.mean((R[R.day == d][c] / 100 - R[R.day == d].y) ** 2) for d in DAYS]) for c in ('A0', 'A2', 'A5')}
gate = {}
print('\n(a) settlement Brier, mean over days')
for c in ('A0', 'A2', 'A5'):
    line = f'  {c} {br[c].mean():.4f}'
    if c != 'A0':
        dd = br[c] - br['A0']; se = dd.std(ddof=1) / np.sqrt(len(dd))
        line += f"  vs A0 {dd.mean():+.4f} ({dd.mean() - 2.365 * se:+.4f}..{dd.mean() + 2.365 * se:+.4f}), better {int((dd < 0).sum())}/8, within +0.005 {int((dd <= 0.005).sum())}/8"
        gate[f'{c}_brier_ok'] = bool((dd <= 0.005).sum() >= 6)
    print(line)
print('\n(b) convergence: Predict mid change regressed on A - A0 (slope = share of the gap that closes), mean over days, 95% CI')
conv = {}
for c in ('A2', 'A5'):
    g = (R[c] - R.A0).values; cells = []
    for h in H:
        col = 'y1000' if h == 1000.0 else f'y{int(h)}'; sl, r2 = [], []
        for d in DAYS:
            m = (R.day == d).values & np.isfinite(R[col].values) & np.isfinite(g)
            x, y = g[m], R[col].values[m]; b = np.sum(x * y) / np.sum(x * x); sl.append(b); r2.append(1 - np.sum((y - b * x) ** 2) / np.sum((y - y.mean()) ** 2))
        sl = np.array(sl); se = sl.std(ddof=1) / np.sqrt(len(sl))
        conv[f'{c}|{int(h)}'] = dict(slope=float(sl.mean()), lo=float(sl.mean() - 2.365 * se), hi=float(sl.mean() + 2.365 * se), r2=float(np.mean(r2)))
        cells.append(f"{'now (unseen)' if h == 0 else '+' + str(int(h / 1000)) + ' s'} {sl.mean():+.3f} ({sl.mean() - 2.365 * se:+.3f}..{sl.mean() + 2.365 * se:+.3f}) R2 {np.mean(r2):.3f}")
    print(f'  {c} ' + '  '.join(cells))
    gate[f'{c}_leads'] = bool(conv[f'{c}|10000']['slope'] >= 0.3 and conv[f'{c}|10000']['lo'] > 0)
good = {c: bool(gate.get(f'{c}_leads') and gate.get(f'{c}_brier_ok')) for c in ('A2', 'A5')}
print('\ngate (computed):', gate, '-> good anchor:', good)
R[['day', 'mid', 'tau', 'A0', 'A2', 'A5', 'cur', 'y', 'y0', 'y1000', 'y10000', 'y60000']].to_parquet(f'{io.OUT}/step4/anchor_eval/rows_fit.parquet', index=False)
io.write_json(f'{io.OUT}/step4/anchor_eval/report_fit.json', dict(brier={c: br[c].tolist() for c in br}, convergence=conv, gate=gate, good_anchor=good))
