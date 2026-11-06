"""Lead-adjusted anchor (declared in STATUS.yml 2026-11-04 05:16Z; user: "predict price + binance move makes sense, it should be
more fresh without that 50ms latency"). Descriptive, 8 read days, nothing selected.

Rows of the stage-1 test (step4/anchor_eval/rows.parquet: every 500 ms, live 0-240 s). For each row at tau: t_f = the venue time
(updateTimestampMs) of the Predict frame known at tau (Tokyo clock, pf_clock_rl with decision 2 ms); dp = p_bn(tau) - p_bn(t_f), with
S = the Binance USD-M mid (E clock) at tau - 3.5 ms (known) and at t_f, the row's basis / start price K / sigma, tau_rem at each time.
  A6 = A0 + dp                     full pass-through of the Binance-implied change since Predict's frame
  A7 = A0 + beta dp                beta fitted leave one day out on Predict's +1 s change (least squares through 0)
  A8 = A0 + f(x)                   LightGBM (pinned PARAMS / ROUNDS), LODO, Predict's +1 s change on x = [dp, Binance move since t_f
                                   (bp), bn_r25..bn_r1000, tau - t_f, tau_rem, sigma sqrt(tau_rem)]
Measures and gate as the redeclared stage 1: settlement Brier vs A0; convergence slopes of A - A0 to Predict's mid change now (the
move made but not yet received), +1 s, +10 s, +60 s; gate = Brier within +0.005 of A0 on >= 6/8 days AND +10 s slope >= 0.3 with
CI excluding 0. Hours inside 10-28..30 and after 11-02 are not loaded (rows whose times fall there are a status).
Usage: PF_RL_DDEC=2 python3 -m predictfun.research.pf_anchor_lead
"""
import os
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.special import ndtr
from . import pf_io as io
from . import pf_clock_rl as CK
from . import pf_venue_feeds as VF
from .pf_model_spec import PARAMS, ROUNDS

assert CK.D_DEC == 2, 'run with PF_RL_DDEC=2'
DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
H = (0.0, 1000.0, 10000.0, 60000.0); BNR = ['bn_r25', 'bn_r50', 'bn_r100', 'bn_r250', 'bn_r500', 'bn_r1000']
st = dict(no_frame=0, no_binance=0)


def bn_series(day):
    k, m = [], []
    for h in VF.hours(day):
        p = f'{io.BN_DIR}/{h}.csv.gz'
        if os.path.exists(p):
            x = pd.read_csv(p, header=None, usecols=[1, 4, 6]).apply(pd.to_numeric, errors='coerce').dropna(); x = x[(x[4] > 0) & (x[6] > x[4])]
            kk, mm = VF.series(x[1].values, np.log((x[4].values + x[6].values) / 2.0)); del x; k.append(kk); m.append(mm)
    return VF.series(np.concatenate(k), np.concatenate(m))


R = pd.read_parquet(f'{io.OUT}/step4/anchor_eval/rows.parquet'); parts = []
for d in DAYS:
    S1 = f'{io.OUT}/step1/{d}'; shift = CK.load_shift(S1)
    r = R[R.day == d].reset_index(drop=True)
    x = pd.read_parquet(f'{io.OUT}/step5/{d}/rows_rl_d2.parquet', columns=['mid', 'tau'] + BNR)
    reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'ws', 'start_price'])
    r = r.merge(x, on=['mid', 'tau'], how='left', validate='one_to_one').merge(reg, on='mid', how='left'); del x
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=['mid', 't_ms', 'recv_ns']) for f in sorted(os.listdir(f'{S1}/books'))])
    bk.sort_values(['mid', 't_ms', 'recv_ns'], inplace=True, kind='stable')
    tf = np.full(len(r), np.nan)
    for mid, g in bk.groupby('mid', sort=False):
        idx = np.flatnonzero(r.mid.values == mid)
        if not len(idx):
            continue
        kf = np.maximum.accumulate(CK.pf_known(g.t_ms.values, g.recv_ns.values, shift)); j = np.searchsorted(kf, r.tau.values[idx], side='right') - 1
        tf[idx] = np.where(j >= 0, g.t_ms.values.astype(float)[np.clip(j, 0, None)], np.nan)
    del bk
    K, LM = bn_series(d)
    def lnS(t):
        i = np.searchsorted(K, t, side='right') - 1
        return np.where(i >= 0, LM[np.clip(i, 0, None)], np.nan)
    ln_now = lnS(r.tau.values - 3.5); ln_f = lnS(tf)
    bas = 1 + r.basis_bps.fillna(0).values / 1e4; Kp = r.start_price.values; sg = r.sigma.values
    rem_now = r.tau_rem.values; rem_f = rem_now + (r.tau.values - tf) / 1000.0
    p_now = 100 * ndtr((ln_now + np.log(bas / Kp)) / (sg * np.sqrt(np.maximum(rem_now, 1e-3))))
    p_f = 100 * ndtr((ln_f + np.log(bas / Kp)) / (sg * np.sqrt(np.maximum(rem_f, 1e-3))))
    r['dp'] = p_now - p_f; r['dbn_bp'] = 1e4 * (ln_now - ln_f); r['age_f'] = r.tau.values - tf; r['sst'] = sg * np.sqrt(rem_now)
    st['no_frame'] += int(np.isnan(tf).sum()); st['no_binance'] += int((np.isnan(ln_now) | np.isnan(ln_f)).sum())
    parts.append(r); io.mem_check(f'day {d}')
R = pd.concat(parts, ignore_index=True); del parts
R = R[np.isfinite(R.dp.values)].reset_index(drop=True)
print(f'statuses {st}; rows used {len(R)}; frame age at tau: median {R.age_f.median():.0f} ms, p90 {R.age_f.quantile(0.9):.0f} ms; |dp| > 0.5 c in {np.mean(np.abs(R.dp) > 0.5):.1%} of rows', flush=True)
R['A6'] = R.A0 + R.dp
R['A7'] = np.nan; R['A8'] = np.nan; FX = ['dp', 'dbn_bp'] + BNR + ['age_f', 'tau_rem', 'sst']; betas = {}
for d in DAYS:
    tr = (R.day != d).values & np.isfinite(R.y1000.values); te = (R.day == d).values
    b = float(np.sum(R.dp.values[tr] * R.y1000.values[tr]) / np.sum(R.dp.values[tr] ** 2)); betas[d] = b
    R.loc[te, 'A7'] = R.A0[te] + b * R.dp[te]
    ds = lgb.Dataset(R.loc[tr, FX].values.astype(np.float32), R.loc[tr, 'y1000'].clip(-10, 10).values, params=PARAMS, free_raw_data=True).construct()
    m = lgb.train(PARAMS, ds, ROUNDS); del ds
    R.loc[te, 'A8'] = R.A0[te] + m.predict(R.loc[te, FX].values.astype(np.float32), num_threads=1); del m
print('fitted pass-through beta by held-out day:', {k[4:]: round(v, 3) for k, v in betas.items()})
CAND = ['A0', 'A6', 'A7', 'A8']; LAB = dict(A0='Predict mid (late)', A6='+ dp (full)', A7='+ beta dp', A8='+ learned')
br = {c: np.array([np.mean((np.clip(R[R.day == d][c], 0, 100) / 100 - R[R.day == d].y) ** 2) for d in DAYS]) for c in CAND}
gate = {}
print('\n(a) settlement Brier, mean over days')
for c in CAND:
    line = f'  {c} {LAB[c]:18s} {br[c].mean():.4f}'
    if c != 'A0':
        dd = br[c] - br['A0']; se = dd.std(ddof=1) / np.sqrt(len(dd))
        line += f"  vs A0 {dd.mean():+.5f} ({dd.mean() - 2.365 * se:+.5f}..{dd.mean() + 2.365 * se:+.5f}), better {int((dd < 0).sum())}/8, within +0.005 {int((dd <= 0.005).sum())}/8"
        gate[f'{c}_brier_ok'] = bool((dd <= 0.005).sum() >= 6)
    print(line)
print('\n(b) convergence: Predict mid change regressed on A - A0 (share of the adjustment that Predict then makes), mean over days, 95% CI')
conv = {}
for c in ('A6', 'A7', 'A8'):
    g = (R[c] - R.A0).values; cells = []
    for h in H:
        col = 'y1000' if h == 1000.0 else f'y{int(h)}'; sl, r2 = [], []
        for d in DAYS:
            mm = (R.day == d).values & np.isfinite(R[col].values) & np.isfinite(g)
            x, y = g[mm], R[col].values[mm]; b = np.sum(x * y) / np.sum(x * x); sl.append(b); r2.append(1 - np.sum((y - b * x) ** 2) / np.sum((y - y.mean()) ** 2))
        sl = np.array(sl); se = sl.std(ddof=1) / np.sqrt(len(sl))
        conv[f'{c}|{int(h)}'] = dict(slope=float(sl.mean()), lo=float(sl.mean() - 2.365 * se), hi=float(sl.mean() + 2.365 * se), r2=float(np.mean(r2)))
        cells.append(f"{'now (unseen)' if h == 0 else '+' + str(int(h / 1000)) + ' s'} {sl.mean():+.3f} ({sl.mean() - 2.365 * se:+.3f}..{sl.mean() + 2.365 * se:+.3f}) R2 {np.mean(r2):.3f}")
    print(f'  {c} {LAB[c]:12s} ' + '  '.join(cells))
    gate[f'{c}_leads'] = bool(conv[f'{c}|10000']['slope'] >= 0.3 and conv[f'{c}|10000']['lo'] > 0)
good = {c: bool(gate.get(f'{c}_leads') and gate.get(f'{c}_brier_ok')) for c in ('A6', 'A7', 'A8')}
print('\ngate (computed):', gate, '-> good anchor:', good)
R[['day', 'mid', 'tau', 'A0', 'A6', 'A7', 'A8', 'dp', 'dbn_bp', 'age_f', 'y', 'y0', 'y1000', 'y10000', 'y60000']].to_parquet(f'{io.OUT}/step4/anchor_eval/rows_lead.parquet', index=False)
io.write_json(f'{io.OUT}/step4/anchor_eval/report_lead.json', dict(statuses=st, betas=betas, brier={c: br[c].tolist() for c in br}, convergence=conv, gate=gate, good_anchor=good))
