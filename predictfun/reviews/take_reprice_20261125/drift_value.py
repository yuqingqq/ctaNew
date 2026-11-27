"""BTC's drift and the time left as a fair value for Predict's price (USER 2026-11-27: "How do we integrate price drift and remaining time").
The window settles Up if Binance spot (Chainlink's input) ends above the start price, so with drift x = ln(spot / start) and volatility
s per sqrt(s), the Up price BTC implies is FV = 100 x Phi(x / (s sqrt(time left))), and Predict should move delta = 100 phi(z) / (s sqrt(
time left)) cents per bp of BTC. REPRICE fired on FV's 250 ms change (futures-based); the take model never had FV. Development evidence
on the four consumed days, every market (REGIME_TAKES' population: 201,847 moments); declared before the run.
Inputs (known at the decision, its way): spot's last trade 100 ms before the moment (and 250 ms / 1 s / 5 s earlier) against the registry
start (= spot at the start, CHEAP_SIDE_ALL); volatility = REGIME_TAKES' 15 min 30 s-return volatility; time left from the registry; drift,
z, FV, the gap FV - Predict's price of the side bought, delta, FV's change over 250 ms / 1 s / 5 s, and Predict's lag = FV's 1 s change -
Predict's mid 1 s change.
Readouts (stage read):
1 calibration (Up space, one row per moment): FV vs Predict's mid, by bucket, against how often Up won; Brier scores of both (and 50).
2 gap quintiles (both directions): the 1 s move, the take at the known best valued 1 s after the match and held to settlement.
3 rules: the top 1 / 5 % by gap: the same values at three limits.
4 models (pinned trees, leave-one-day-out): the 1 s markout move (TAKE_MARKOUT's target) with TAKE_MODEL's 79 inputs (must reproduce
  REGIME_TAKES) and + the 9 drift inputs; the settlement move (100 x payout - mid, its way) with 79, 79 + 9, the 9 alone, and Predict's
  price + time left + the 9; held-out rank correlation; top 1 % by predicted edge at the known best: value at three limits (1 s) or held.
Paying (computed): > 0 pooled AND >= 3 of 4 days AND >= 150 orders. Settlement values are binary per market: noisy.
Outputs (write-once): OUT/drift_value/<day>.parquet (points), drift_value.json (read).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/drift_value.py \
       selftest | points <day> [<dir>] | read [<rehearsal dir> <market fraction>]
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import cross_venue as CV  # noqa: E402
import regime_takes as RT  # noqa: E402

TK, TM, MM, TR, C, F, WT = RT.TK, RT.TM, RT.MM, RT.TR, RT.C, RT.F, RT.WT
OUT = WT.OUT / 'drift_value'
DAYS, Q, fee = RT.DAYS, RT.Q, RT.fee
LAGS = dict(now=100., b250=350., b1s=1100., b5s=5100.)                      # spot taken this many ms before the moment
DRIFT = ['dv_drift_its', 'dv_z_its', 'dv_fv_its', 'dv_gap_its', 'dv_delta', 'dv_dfv250_its', 'dv_dfv1s_its', 'dv_dfv5s_its', 'dv_lag1s_its']
BINS = ((0, 10), (10, 20), (20, 35), (35, 65), (65, 80), (80, 90), (90, 100.01))
ORDERS = ('s0', 's3', 'any')


def payout(r):
    if float(r['start_price']) == float(r['end_price']):
        return .5
    return 1. if r['outcome'] == 'Up' else 0.


def fair(drift_bps, vol_bps, tau_s):
    """-> (z, FV in c, delta in c per bp)."""
    s = vol_bps * np.sqrt(np.maximum(tau_s, .5))
    z = drift_bps / s
    return z, 100. * norm.cdf(z), 100. * norm.pdf(z) / s


def points(day, od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1201))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    L = pd.read_parquet(RT.OUT / f'{day}.parquet', columns=['row', 'day', 'mid', 't', 'd'])
    rg = pd.read_parquet(MM.S1 / day / 'registry.parquet').set_index('mid')
    C.require(set(L.mid) <= set(rg.index), 'a market missing from the registry')
    tv, pv, _, _, cov, _ = CV.load('binance_spot', day)
    t = L.t.to_numpy(float)
    st = rg.loc[L.mid, 'start_price'].to_numpy(float)
    o = dict(row=L.row.to_numpy(), tau=(rg.loc[L.mid, 'we'].to_numpy(float) * 1000. - t) / 1000.,
             payout=np.array([payout(rg.loc[m]) for m in L.mid]))
    for k, lag in LAGS.items():
        j = np.searchsorted(tv, t - lag, 'right') - 1
        s = np.where(j >= 0, pv[np.maximum(j, 0)], np.nan)
        o[f'drift_{k}'] = 1e4 * np.log(s / st)
    P = pd.DataFrame(o)
    od.mkdir(parents=True, exist_ok=True)
    P.to_parquet(fp, index=False)
    print(day, len(P), {c: int(P[c].isna().sum()) for c in P.columns if c.startswith('drift_')}, flush=True)


def gap_quintile_stats(g, V, HV, FL, mo, ok, day, mid):
    q, edges = RT.quintiles(g)
    out = dict(edges=[float(x) for x in edges] if edges is not None else None, quintiles={})
    for b in range(5):
        m = ok & (q == b)
        out['quintiles'][str(b)] = dict(mean_gap_c=float(g[m].mean()), take_1s=RT.cell_stats(V['s0'], FL['s0'], mo, m, day, mid),
                                        held=RT.cell_stats(HV['s0'], FL['s0'], mo, m, day, mid))
    return out


def read(od=None, frac=None):
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3601))
    reh = od is not None
    od = Path(od) if od else None
    fn = (od or H) / 'drift_value.json'
    C.require(not fn.exists(), 'drift_value.json exists (write-once)')
    X, feats = RT.frame(RT.OUT)
    Pt = pd.concat([pd.read_parquet((od or OUT) / f'{dh}.parquet') for dh in DAYS if ((od or OUT) / f'{dh}.parquet').exists()], ignore_index=True)
    C.require(len(Pt) == len(X) and set(Pt.row) == set(X.row), 'the drift points do not cover REGIME_TAKES')
    X = X.merge(Pt, on='row', how='left', validate='1:1')
    if reh and frac:
        keep = set(np.random.default_rng(1).choice(np.unique(X.mid), max(8, int(frac * X.mid.nunique())), replace=False).tolist())
        X = X[X.mid.isin(keep)].reset_index(drop=True)
    C.require(np.allclose(X.tau.to_numpy(float), X.p_tau.to_numpy(float), atol=1e-6), 'time left differs from public_score')
    d, lim, md, day, mid = X.d.to_numpy(float), X.lim.to_numpy(float), X.md.to_numpy(float), X.day.to_numpy(), X.mid.to_numpy()
    pay, ip = X.payout.to_numpy(float), X.pf_its_price.to_numpy(float)
    vol, tau = X.btc_vol_15m.to_numpy(float), X.tau.to_numpy(float)
    z, fv, delta = fair(X.drift_now.to_numpy(float), vol, tau)
    fv_its = np.where(d > 0, fv, 100. - fv)
    X['dv_drift_its'] = d * X.drift_now; X['dv_z_its'] = d * z; X['dv_fv_its'] = fv_its; X['dv_gap_its'] = fv_its - ip; X['dv_delta'] = delta
    for k, c_ in (('b250', 'dv_dfv250_its'), ('b1s', 'dv_dfv1s_its'), ('b5s', 'dv_dfv5s_its')):
        X[c_] = d * (fv - fair(X[f'drift_{k}'].to_numpy(float), vol, tau)[1])
    X['dv_lag1s_its'] = X.dv_dfv1s_its - X.pf_mid_ret_1000
    mk = X.m1s.to_numpy(float); mo = d * (mk - md)
    cost = d * (lim - md) + fee(lim)
    PA = X[[f'pA_{k}' for k in range(5)]].to_numpy(float); SA = X[[f'sA_{k}' for k in range(5)]].to_numpy(float)
    V, HV, FL = {}, {}, {}
    for o, L_ in (('s0', lim), ('s3', lim + 3. * d), ('any', np.where(d > 0, 99., 1.))):
        fl, co, fe = TK.vwalk(PA, SA, d, L_)
        V[o] = d * (mk * fl - co) - fe; HV[o] = d * (fl * 100. * pay - co) - fe; FL[o] = fl
    ok = np.isfinite(mo)
    okf = ok & np.isfinite(fv)
    out = dict(asof='2026-11-27', rows=int(len(X)), rows_with_fv=int(okf.sum()), calibration={}, gap={}, rules={}, models={})
    # 1 calibration, Up space
    up = okf & (d > 0)
    for nm, pr in (('fv', fv), ('predict_mid', X.pf_mid.to_numpy(float))):
        out['calibration'][nm] = {}
        for lo, hi in BINS:
            m = up & (pr >= lo) & (pr < hi)
            out['calibration'][nm][f'{lo}-{min(hi, 100):g}'] = dict(n=int(m.sum()), mean_c=float(pr[m].mean()) if m.any() else None, up_won_pct=float(100. * pay[m].mean()) if m.any() else None,
                                                                 markets=int(np.unique(mid[m]).size))
        out['calibration'][f'brier_{nm}'] = float(np.mean((pr[up] / 100. - pay[up]) ** 2))
        out['calibration'][f'brier_{nm}_by_day'] = {dh: float(np.mean((pr[up & (day == dh)] / 100. - pay[up & (day == dh)]) ** 2)) for dh in DAYS}
    out['calibration']['brier_half'] = float(np.mean((.5 - pay[up]) ** 2))
    out['calibration']['spearman_fv_predict_mid'] = float(spearmanr(fv[up], X.pf_mid.to_numpy(float)[up]).correlation)
    # 2 gap quintiles
    g = X.dv_gap_its.to_numpy(float)
    out['gap'] = gap_quintile_stats(g, V, HV, FL, mo, okf, day, mid)
    out['gap']['spearman_gap_move_1s'] = float(spearmanr(g[okf], mo[okf]).correlation)
    out['gap']['spearman_gap_settlement'] = float(spearmanr(g[okf], (d * (100. * pay - md))[okf]).correlation)
    out['gap']['spearman_lag1s_move_1s'] = float(spearmanr(X.dv_lag1s_its.to_numpy(float)[okf], mo[okf]).correlation)
    # 3 rules
    for q in (.01, .05):
        idx = TK.top(np.where(okf, g, np.nan), okf, q)
        m = np.zeros(len(X), bool); m[idx] = True
        out['rules'][f'gap_top{q * 100:g}'] = dict(mean_gap_c=float(g[m].mean()), its_price_median_c=float(np.median(ip[m])), time_left_median_s=float(np.median(tau[m])),
                                                   take_1s={o: RT.cell_stats(V[o], FL[o], mo, m, day, mid) for o in ORDERS},
                                                   held={o: RT.cell_stats(HV[o], FL[o], mo, m, day, mid) for o in ORDERS})
    # 4 models
    PRED = ['pf_its_price', 'p_tau']
    ys = d * (100. * pay - md)
    specs = [('move_1s', 'take_model_inputs', feats, mo, V), ('move_1s', 'plus_drift', feats + DRIFT, mo, V),
             ('settlement', 'take_model_inputs', feats, ys, HV), ('settlement', 'plus_drift', feats + DRIFT, ys, HV),
             ('settlement', 'drift_only', DRIFT, ys, HV), ('settlement', 'price_time_left_drift', PRED + DRIFT, ys, HV)]
    for tg, sn, cols, y, VV in specs:
        pred = np.full(len(X), np.nan)
        for dh in DAYS:
            tr, te = ok & (day != dh), day == dh
            pred[te] = HistGradientBoostingRegressor(random_state=0, **TM.HGB).fit(X.loc[tr, cols], y[tr]).predict(X.loc[te, cols])
        idx = TK.top(pred - cost, ok, .01)
        m = np.zeros(len(X), bool); m[idx] = True
        r = dict(spearman=float(spearmanr(pred[ok], y[ok]).correlation), top1={o: RT.cell_stats(VV[o], FL[o], mo, m, day, mid) for o in ORDERS},
                 top1_its_price_median_c=float(np.median(ip[m])), top1_time_left_median_s=float(np.median(tau[m])))
        out['models'].setdefault(tg, {})[sn] = r
        print(tg, sn, round(r['spearman'], 3), {o: (round(c['value_c'], 2), c['days_positive']) for o, c in r['top1'].items()}, flush=True)
    if not reh:
        ref = json.loads((H / 'regime_takes.json').read_text())['models']['take_model_inputs']
        mine = out['models']['move_1s']['take_model_inputs']
        C.require(abs(mine['spearman'] - ref['spearman']) < 1e-9 and abs(mine['top1']['s0']['value_c'] - ref['top1']['s0']['value_c']) < 1e-9, 'REGIME_TAKES not reproduced')
        out['reproduced_regime_takes'] = True
    cells = [(f'{tg}/{sn}', o, c) for tg, ms in out['models'].items() for sn, r in ms.items() for o, c in r['top1'].items()] + \
            [(f'rule/{k}/{kind}', o, c) for k, r in out['rules'].items() for kind in ('take_1s', 'held') for o, c in r[kind].items()]
    out['verdict'] = dict(cells=len(cells), paying=[dict(cell=a, order=o, value_c=c['value_c'], orders=c['orders'], days_positive=c['days_positive']) for a, o, c in cells if c.get('paying')],
                          fv_brier_better_than_predict=bool(out['calibration']['brier_fv'] < out['calibration']['brier_predict_mid']))
    TR.dump(fn, out)
    print(json.dumps(dict(calibration={k: v for k, v in out['calibration'].items() if k.startswith('brier') or k.startswith('spearman')}, verdict=out['verdict']), indent=1))


def selftest():
    z, fv, dl = fair(np.array([0., 5., -5.]), np.array([.5, .5, .5]), np.array([100., 100., 100.]))
    assert abs(fv[0] - 50.) < 1e-12 and abs(fv[1] - (100. - fv[2])) < 1e-9 and fv[1] > 50.
    assert abs(z[1] - 1.) < 1e-12                                             # 5 bps against 0.5 x sqrt(100) = 5 bps
    assert abs(dl[0] - 100. * norm.pdf(0.) / 5.) < 1e-12 and dl[0] > dl[1]   # most sensitive at the money
    _, fv_end, dl_end = fair(np.array([1.]), np.array([.5]), np.array([1.]))
    _, fv_open, dl_open = fair(np.array([1.]), np.array([.5]), np.array([299.]))
    assert fv_end[0] > fv_open[0] and dl_end[0] > dl_open[0]                 # the same drift matters more near the end
    assert payout(dict(start_price=1., end_price=1., outcome='Up')) == .5 and payout(dict(start_price=1., end_price=2., outcome='Up')) == 1.
    print('drift_value selftest PASS')


if __name__ == '__main__':
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == 'points':
        points(a[0], a[1] if len(a) > 1 else None)
    elif cmd == 'read':
        read(a[0] if a else None, float(a[1]) if len(a) > 1 else None)
    else:
        dict(selftest=selftest)[cmd]()
