"""Do Predict-native book signals help the guard reach the top makers' bad-fill level? (declared in STATUS.yml 2026-11-04 03:53Z,
step 1 of the guard-target plan). Descriptive, 8 read days, nothing selected.

Same neutral exposure grid, labels, scores and metrics as pf_badfill_info (step4/badfill_info/grid.parquet: an order resting at
the known best, every 500 ms, live 0-240 s; v = s (m(t_sw + 1 s) - p) if swept through within (tau + 22, tau + 522] ms, else 0;
bad = v <= -2). New Predict-native features, from the venue frames known at tau (Tokyo clock, pf_clock_rl with decision 2 ms):
  per side (bid b / ask a): best size now; log best size ratio vs 250 / 1000 ms ago (same price, else NaN; +1 share floor); best
  price change in ticks over 250 / 1000 ms; depth of levels 1-3 and its log ratio vs 250 / 1000 ms ago; price gaps level 1-2 and 2-3; sweeps inferred from
  the book (best moving 2+ ticks against that side between consecutive known frames) in the last 1 s / 5 s and ms since the last;
  market: resting order count change over 250 / 1000 ms; rows: tau_rem, p_bn, pf_onesided.
Scores: base (s * pred_1000), M1 (current features; refit, same seed as before), M3 = M1 + the new features, M4 = M3 + the outside
venue features of M2, ideal, random. LODO by day, per-day metrics, day-clustered 95% intervals (G = 8).
Usage: PF_RL_DDEC=2 python3 -m predictfun.research.pf_badfill_info2
"""
import os
import numpy as np, pandas as pd, lightgbm as lgb
from . import pf_io as io
from . import pf_clock_rl as CK
from .pf_model_spec import SETS, PARAMS, ROUNDS

assert CK.D_DEC == 2, 'run with PF_RL_DDEC=2'
DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
FE = SETS['ALL']; PULLS = (0.05, 0.10, 0.20, 0.30); OUTD = f'{io.OUT}/step4/badfill_info2'; os.makedirs(OUTD, exist_ok=True)
NEWF = [f'{f}_{k}{w}' for f in ('spot', 'hl') for k in ('mv', 'v') for w in (100, 250, 1000)] + ['cl_dev', 'cl_age', 'cl_mv2000']
PFN = [f'{x}_{s}' for s in ('b', 'a') for x in ('sz1', 'szr250', 'szr1000', 'dp250', 'dp1000', 'dep3', 'depr250', 'depr1000', 'gap12', 'gap23', 'sw1000', 'sw5000', 'swage')] + \
      ['doc250', 'doc1000', 'tau_rem', 'p_bn', 'pf_onesided']
COLS = ['mid', 't_ms', 'recv_ns', 'order_count'] + [f'{s}{i}_{k}' for s in 'ba' for i in range(1, 6) for k in 'ps']


def pf_features(d, G):
    S1 = f'{io.OUT}/step1/{d}'; shift = CK.load_shift(S1)
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=COLS) for f in sorted(os.listdir(f'{S1}/books'))])
    bk.sort_values(['mid', 't_ms', 'recv_ns'], inplace=True, kind='stable'); BK = {m: g for m, g in bk.groupby('mid', sort=False)}; del bk
    out = np.full((len(G), len(PFN) - 3), np.nan, np.float32)
    for mid, idx in G.groupby('mid', sort=False).indices.items():
        g = BK.get(mid)
        if g is None:
            continue
        kf = np.maximum.accumulate(CK.pf_known(g.t_ms.values, g.recv_ns.values, shift)); tau = G.tau.values[idx].astype(float)
        j = np.searchsorted(kf, tau, side='right') - 1; j250 = np.searchsorted(kf, tau - 250.0, side='right') - 1; j1k = np.searchsorted(kf, tau - 1000.0, side='right') - 1
        ok = (j >= 0); jc, j2c, j1c = np.clip(j, 0, None), np.clip(j250, 0, None), np.clip(j1k, 0, None)
        oc = g.order_count.values.astype(float); cols = []
        for side, pre, sgn in ((0, 'b', 1.0), (1, 'a', -1.0)):
            P = g[[f'{pre}{i}_p' for i in range(1, 6)]].values.astype(float); S = g[[f'{pre}{i}_s' for i in range(1, 6)]].values.astype(float)
            P = np.where(P < 0, np.nan, P); S = np.where(np.isnan(P), 0.0, S)
            b1 = P[:, 0]; s1 = S[:, 0]; dep = S[:, :3].sum(1)
            jump = np.r_[False, sgn * (b1[:-1] - b1[1:]) >= 2]                 # best moved 2+ ticks against this side
            cs = np.cumsum(jump); last = np.maximum.accumulate(np.where(jump, np.arange(len(jump)), -1))
            j5k = np.clip(np.searchsorted(kf, tau - 5000.0, side='right') - 1, 0, None)
            same250 = b1[jc] == b1[j2c]; same1k = b1[jc] == b1[j1c]
            cols += [s1[jc],
                     np.where(same250 & (j250 >= 0), np.log((s1[jc] + 1) / (s1[j2c] + 1)), np.nan), np.where(same1k & (j1k >= 0), np.log((s1[jc] + 1) / (s1[j1c] + 1)), np.nan),
                     np.where(j250 >= 0, b1[jc] - b1[j2c], np.nan), np.where(j1k >= 0, b1[jc] - b1[j1c], np.nan),
                     dep[jc], np.where(j250 >= 0, np.log((dep[jc] + 1) / (dep[j2c] + 1)), np.nan), np.where(j1k >= 0, np.log((dep[jc] + 1) / (dep[j1c] + 1)), np.nan),
                     sgn * (P[jc, 0] - P[jc, 1]), sgn * (P[jc, 1] - P[jc, 2]),
                     (cs[jc] - cs[j1c]).astype(float), (cs[jc] - cs[j5k]).astype(float),
                     np.where(last[jc] >= 0, np.minimum(tau - kf[np.clip(last[jc], 0, None)], 10000.0), 10000.0)]
        cols += [np.where(j250 >= 0, oc[jc] - oc[j2c], np.nan), np.where(j1k >= 0, oc[jc] - oc[j1c], np.nan)]
        X = np.column_stack(cols).astype(np.float32); X[~ok] = np.nan; out[idx] = X
    del BK
    return out


R = pd.read_parquet(f'{io.OUT}/step4/badfill_info/grid.parquet')
parts = []
SMOKE = os.environ.get('PF_BI2_SMOKE')
for d in ([SMOKE] if SMOKE else DAYS):
    G = R[R.day == d].reset_index(drop=True)
    X = pf_features(d, G)
    for i, c in enumerate(PFN[:-3]):
        G[c] = X[:, i]
    r = pd.read_parquet(f'{io.OUT}/step5/{d}/rows_rl_d2.parquet', columns=['mid', 'tau', 'tau_rem', 'p_bn', 'pf_onesided'])
    G = G.merge(r, on=['mid', 'tau'], how='left', validate='one_to_one'); parts.append(G); del r
    io.mem_check(f'features {d}'); print(f'features {d}: rows {len(G)}, NaN share of new features {float(np.isnan(G[PFN].values.astype(float)).mean()):.3f}', flush=True)
R = pd.concat(parts, ignore_index=True); del parts
if SMOKE:
    print(R[PFN].describe().T[['count', 'mean', 'std', 'min', 'max']].round(3).to_string()); raise SystemExit(0)
R[['day', 'mid', 'tau'] + PFN].to_parquet(f'{OUTD}/pf_features.parquet', index=False)
RNG = np.random.default_rng(7)
for side, sg in (('bid', 1.0), ('ask', -1.0)):
    R[f'base_{side}'] = sg * R.pred_1000.values; R[f'ideal_{side}'] = sg * R.y1000.values; R[f'rand_{side}'] = RNG.random(len(R))
    for name, cols in (('M1', FE), ('M3', FE + PFN), ('M4', FE + PFN + NEWF)):
        pred = np.full(len(R), np.nan)
        for d in DAYS:
            tr = (R.day != d).values; te = ~tr
            ds = lgb.Dataset(R.loc[tr, cols].values.astype(np.float32), R.loc[tr, f'v_{side}'].clip(-10, 10).values, params=PARAMS, free_raw_data=True).construct()
            m = lgb.train(PARAMS, ds, ROUNDS); del ds
            pred[te] = m.predict(R.loc[te, cols].values.astype(np.float32), num_threads=1)
            if name == 'M3' and d == DAYS[0]:
                imp = pd.Series(m.feature_importance('gain'), index=cols).sort_values(ascending=False)
                print(f'M3 {side} top features by gain (fold {d}):', ', '.join(f'{k} {v / imp.sum():.1%}' for k, v in imp.head(12).items()), flush=True)
            del m
        R[f'{name}_{side}'] = pred
        io.mem_check(f'{name} {side}'); print(f'fitted {name} {side}', flush=True)


def auc(score, y):
    o = np.argsort(score, kind='stable'); rk = np.empty(len(score)); rk[o] = np.arange(1, len(score) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((n1 * (len(y) + 1) - rk[y].sum()) / (n1 * n0)) if n1 and n0 else np.nan


SC = ['base', 'M1', 'M3', 'M4', 'ideal', 'rand']; rows = []
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
print('\nmean over 8 days: bad value avoided / good value lost / net (share of bad value); vs base and vs M1 with day-clustered 95% CI')
for k in PULLS:
    print(f'  pull {k:.0%} of exposures per side')
    for sc in SC:
        x = D[D.score == sc]; line = f"    {sc:5s} {x[f'bad_avoided_{k}'].mean():6.1%} / {x[f'good_lost_{k}'].mean():6.1%} / {x[f'net_{k}'].mean():+6.1%}"
        for ref in (('base', 'M1') if sc in ('M3', 'M4') else (('base',) if sc == 'M1' else ())):
            dd = x[f'net_{k}'].values - D[D.score == ref][f'net_{k}'].values; se = dd.std(ddof=1) / np.sqrt(len(dd))
            line += f"  | vs {ref} {dd.mean():+5.1%} ({dd.mean() - 2.365 * se:+.1%}..{dd.mean() + 2.365 * se:+.1%}, {int((dd > 0).sum())}/8)"
        print(line)
print('AUC for bad: ' + ', '.join(f"{sc} {D[D.score == sc].auc_bad.mean():.3f}" for sc in SC))
io.write_json(f'{OUTD}/report.json', dict(means={sc: D[D.score == sc].drop(columns=['day', 'score']).mean().round(4).to_dict() for sc in SC}))
