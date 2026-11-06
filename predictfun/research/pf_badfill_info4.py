"""Longer look-back guard screen (declared in STATUS.yml 2026-11-04 06:42Z; user: "test a longer look back window"). Same neutral
grid, labels, folds, metrics and rule as pf_badfill_info3 (step4/badfill_info/grid.parquet). Descriptive, 8 read days.
  M1  LightGBM (pinned PARAMS / ROUNDS) for v per side on the current features (pf_model_spec ALL; longest windows 3-5 s), refit
  M6  M1 + longer look-back inputs known at tau (Binance at tau - 3.5 ms, E / T clocks; Predict frames known per pf_clock_rl):
      Binance: log-mid return over 5 / 10 / 30 / 60 s (bp); realised volatility of 1 s returns over 10 / 60 s; largest |1 s
      return| in the last 10 s; aggressor flow (BTC) over 10 / 30 s
      Predict: known mid change over 10 / 30 / 60 s; best-price changes among known frames in the last 10 / 60 s; sweeps (best
      2+ ticks against the side between consecutive known frames) in the last 30 / 60 s, per side
Rule (coded): replay only if M6 beats M1 with the CI excluding 0 at >= 3 of the 4 pull rates. Data inside 10-28..30 and after
11-02 is never loaded (lookbacks that reach it are NaN).
Usage: PF_RL_DDEC=2 python3 -m predictfun.research.pf_badfill_info4
"""
import os
import numpy as np, pandas as pd, lightgbm as lgb
from . import pf_io as io
from . import pf_clock_rl as CK
from . import pf_venue_feeds as VF
from .pf_model_spec import SETS, PARAMS, ROUNDS

assert CK.D_DEC == 2, 'run with PF_RL_DDEC=2'
DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
FE = SETS['ALL']; PULLS = (0.05, 0.10, 0.20, 0.30); OUTD = f'{io.OUT}/step4/badfill_info4'; os.makedirs(OUTD, exist_ok=True)
BT = io.BN_DIR.replace('bookTicker', 'trade')
LONG = ['bl_r5', 'bl_r10', 'bl_r30', 'bl_r60', 'bl_rv10', 'bl_rv60', 'bl_max10', 'bl_fl10', 'bl_fl30', 'pl_dm10', 'pl_dm30', 'pl_dm60', 'pl_nch10', 'pl_nch60',
        'pl_swb30', 'pl_swb60', 'pl_swa30', 'pl_swa60']


def bn(day):
    k, m = [], []
    for h in VF.hours(day):
        p = f'{io.BN_DIR}/{h}.csv.gz'
        if os.path.exists(p):
            x = pd.read_csv(p, header=None, usecols=[1, 4, 6]).apply(pd.to_numeric, errors='coerce').dropna(); x = x[(x[4] > 0) & (x[6] > x[4])]
            kk, mm = VF.series(x[1].values, np.log((x[4].values + x[6].values) / 2.0)); del x; k.append(kk); m.append(mm)
    S = VF.series(np.concatenate(k), np.concatenate(m)); k, q = [], []
    for h in VF.hours(day):
        p = f'{BT}/{h}.csv.gz'
        if os.path.exists(p):
            x = pd.read_csv(p, header=None, usecols=[2, 5, 6]).apply(pd.to_numeric, errors='coerce').dropna()
            k.append(x[2].values); q.append(np.where(x[6].values == 1, -1.0, 1.0) * x[5].values)
    return S, VF.flow(np.concatenate(k), np.concatenate(q))


def lm_at(S, t):
    K, V = S; i = np.searchsorted(K, t, side='right') - 1
    return np.where(i >= 0, V[np.clip(i, 0, None)], np.nan)


def long_features(d, G):
    S1 = f'{io.OUT}/step1/{d}'; shift = CK.load_shift(S1)
    S, FL = bn(d); t = G.tau.values.astype(float) - 3.5; out = {}
    now = lm_at(S, t)
    for h in (5, 10, 30, 60):
        out[f'bl_r{h}'] = 1e4 * (now - lm_at(S, t - h * 1000.0))
    samp = np.column_stack([lm_at(S, t - i * 1000.0) for i in range(61)])          # 1 s samples, newest first
    r1 = 1e4 * (samp[:, :-1] - samp[:, 1:])
    out['bl_rv10'] = np.sqrt(np.nansum(r1[:, :10] ** 2, 1)); out['bl_rv60'] = np.sqrt(np.nansum(r1[:, :60] ** 2, 1)); out['bl_max10'] = np.nanmax(np.abs(r1[:, :10]), 1)
    bad = np.isnan(samp).any(1); out['bl_rv60'][bad] = np.nan
    for h in (10, 30):
        out[f'bl_fl{h}'] = VF.vol(FL, t, h * 1000.0)
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=['mid', 't_ms', 'recv_ns', 'b1_p', 'a1_p']) for f in sorted(os.listdir(f'{S1}/books'))])
    bk.sort_values(['mid', 't_ms', 'recv_ns'], inplace=True, kind='stable')
    for c in LONG[9:]:
        out[c] = np.full(len(G), np.nan)
    for mid, g in bk.groupby('mid', sort=False):
        idx = np.flatnonzero(G.mid.values == mid)
        if not len(idx):
            continue
        kf = np.maximum.accumulate(CK.pf_known(g.t_ms.values, g.recv_ns.values, shift)); tau = G.tau.values[idx].astype(float)
        b1 = g.b1_p.values.astype(float); a1 = g.a1_p.values.astype(float); valid = (b1 >= 0) & (a1 > b1)
        mf = pd.Series(np.where(valid, (b1 + a1) / 2.0, np.nan)).ffill().values
        ch = np.r_[False, (mf[1:] != mf[:-1]) & np.isfinite(mf[1:]) & np.isfinite(mf[:-1])]; cch = np.cumsum(ch)
        b1v = np.where(b1 >= 0, b1, np.nan); a1v = np.where(a1 >= 0, a1, np.nan)
        swb = np.cumsum(np.r_[False, (b1v[:-1] - b1v[1:]) >= 2]); swa = np.cumsum(np.r_[False, (a1v[1:] - a1v[:-1]) >= 2])
        j = np.searchsorted(kf, tau, side='right') - 1; jc = np.clip(j, 0, None); okj = j >= 0
        for h in (10, 30, 60):
            jh = np.searchsorted(kf, tau - h * 1000.0, side='right') - 1; jhc = np.clip(jh, 0, None); okh = okj & (jh >= 0)
            out[f'pl_dm{h}'][idx] = np.where(okh, mf[jc] - mf[jhc], np.nan)
            if h in (10, 60):
                out[f'pl_nch{h}'][idx] = np.where(okh, cch[jc] - cch[jhc], np.nan)
            if h in (30, 60):
                out[f'pl_swb{h}'][idx] = np.where(okh, swb[jc] - swb[jhc], np.nan); out[f'pl_swa{h}'][idx] = np.where(okh, swa[jc] - swa[jhc], np.nan)
    return pd.DataFrame({c: out[c].astype(np.float32) for c in LONG})


R = pd.read_parquet(f'{io.OUT}/step4/badfill_info/grid.parquet', columns=list(dict.fromkeys(['day', 'mid', 'tau', 'v_bid', 'v_ask', 'bad_bid', 'bad_ask', 'pred_1000', 'y1000'] + FE)))
parts = []
for d in DAYS:
    G = R[R.day == d].reset_index(drop=True); X = long_features(d, G)
    parts.append(pd.concat([G, X], axis=1)); io.mem_check(f'features {d}')
    print(f'features {d}: rows {len(G)}, NaN share {float(np.isnan(X.values).mean()):.3f}', flush=True)
R = pd.concat(parts, ignore_index=True); del parts
R[['day', 'mid', 'tau'] + LONG].to_parquet(f'{OUTD}/long_features.parquet', index=False)
RNG = np.random.default_rng(7)
for side, sg in (('bid', 1.0), ('ask', -1.0)):
    R[f'base_{side}'] = sg * R.pred_1000.values; R[f'ideal_{side}'] = sg * R.y1000.values; R[f'rand_{side}'] = RNG.random(len(R))
    for name, cols in (('M1', FE), ('M6', FE + LONG)):
        pred = np.full(len(R), np.nan)
        for d in DAYS:
            tr = (R.day != d).values; te = ~tr
            ds = lgb.Dataset(R.loc[tr, cols].values.astype(np.float32), R.loc[tr, f'v_{side}'].clip(-10, 10).values, params=PARAMS, free_raw_data=True).construct()
            m = lgb.train(PARAMS, ds, ROUNDS); del ds
            pred[te] = m.predict(R.loc[te, cols].values.astype(np.float32), num_threads=1)
            if name == 'M6' and d == DAYS[0]:
                imp = pd.Series(m.feature_importance('gain'), index=cols).sort_values(ascending=False)
                print(f"M6 {side}: gain share of the longer look-back inputs {imp[LONG].sum() / imp.sum():.1%}; top: " + ', '.join(f'{k} {imp[k] / imp.sum():.1%}' for k in imp[LONG].sort_values(ascending=False).head(5).index), flush=True)
            del m
        R[f'{name}_{side}'] = pred; io.mem_check(f'{name} {side}'); print(f'fitted {name} {side}', flush=True)


def auc(score, y):
    o = np.argsort(score, kind='stable'); rk = np.empty(len(score)); rk[o] = np.arange(1, len(score) + 1)
    n1 = y.sum(); n0_ = len(y) - n1
    return float((n1 * (len(y) + 1) - rk[y].sum()) / (n1 * n0_)) if n1 and n0_ else np.nan


SC = ['base', 'M1', 'M6', 'ideal', 'rand']; rows = []
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
print('\nmean over 8 days: bad value avoided / good value lost / net (share of bad value); M6 vs M1 and vs base, day-clustered 95% CI')
out = {}
for k in PULLS:
    print(f'  pull {k:.0%} of exposures per side')
    for sc in SC:
        x = D[D.score == sc]; line = f"    {sc:5s} {x[f'bad_avoided_{k}'].mean():6.1%} / {x[f'good_lost_{k}'].mean():6.1%} / {x[f'net_{k}'].mean():+6.1%}"
        for ref in (('M1', 'base') if sc == 'M6' else (('base',) if sc == 'M1' else ())):
            dd = x[f'net_{k}'].values - D[D.score == ref][f'net_{k}'].values; se = dd.std(ddof=1) / np.sqrt(len(dd))
            out[f'{sc}_vs_{ref}_{k}'] = dict(mean=float(dd.mean()), lo=float(dd.mean() - 2.365 * se), hi=float(dd.mean() + 2.365 * se), days_better=int((dd > 0).sum()))
            line += f"  | vs {ref} {dd.mean():+5.1%} ({dd.mean() - 2.365 * se:+.1%}..{dd.mean() + 2.365 * se:+.1%}, {int((dd > 0).sum())}/8)"
        print(line)
print('AUC for bad: ' + ', '.join(f"{sc} {D[D.score == sc].auc_bad.mean():.3f}" for sc in SC))
better = sum(out[f'M6_vs_M1_{k}']['lo'] > 0 for k in PULLS)
print(f'M6 beats M1 with the CI excluding 0 at {better} of {len(PULLS)} pull rates -> replay warranted: {better >= 3}')
io.write_json(f'{OUTD}/report.json', dict(comparisons=out, m6_beats_m1_pulls=better, replay_warranted=bool(better >= 3),
              means={sc: D[D.score == sc].drop(columns=['day', 'score']).mean().round(4).to_dict() for sc in SC}))
