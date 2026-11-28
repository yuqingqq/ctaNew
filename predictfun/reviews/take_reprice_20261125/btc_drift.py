"""BTC's drift from the window's start price against Predict's price (USER 2026-11-27: "We check the btc price drift,"; follow-up to
PREDICT_INPUTS: nothing in the models compares Predict's price with where BTC stands against the start). Development evidence on the four
consumed days; declared before the run.
BTC = Binance spot's last trade (aggTrades, Binance Vision; Chainlink's settlement input is Binance spot's mid), taken 100 ms before the
moment (we sit ~72 ms from Binance); start = the registry's start price (equal to spot at the window start: checked, readout 1). Live,
the official start reaches our recorder a median 19 s after the window opens; spot at the start is known within ~0.1 s.
Per moment of TAKE_MODEL's population (stage points): drift = ln(spot / start); BTC volatility per second = trailing overlapping 30 s returns of spot (last trade
on a 1 s grid) / 30 over 15 min of the same UTC day (primary; 60 min, and 1 s returns over 15 min, as sensitivities: 1 s returns are
positively autocorrelated on these days and understate the 5 min variance); >= 300 returns, else a status; z = drift / (vol x sqrt(time
left)); the price BTC implies for Up = 100 x Phi(z) (no drift term); the drift 30 s before; the drift's high and low since the window start; for readouts
only, never inputs: the drift 30 s after the match and at the end (registry end price).
Readouts (stage read):
1 alignment (positive control, stops the read if it fails): spot at the window start vs the registry start (|median| < 0.5 bps); the
  sign of spot's end drift vs the outcome >= 95% of markets, every disagreement at |drift| < 1 bp.
2 Predict vs BTC: rank correlation of Predict's mid with the implied price (one row per moment); Predict's price minus the implied price
  for the side bought, by Predict-price bucket.
3 calibration on the same moments: by the implied price's bucket (each volatility) and by Predict's price bucket,
  how often the side won; sides Predict prices under 10c split by the implied price (under / over 10c).
4 does BTC revert: the drift's change over the next 30 s (inside the window) and to the end, by z (Up space, one row per moment);
  the slope of the change on the drift, pooled and by day; the variance ratio: each window's whole move against 300 s of each
  volatility at its first moment (1 = a calibrated volatility under a random walk), by day.
5 decisions: TAKE_MARKOUT's targets (1 s / 30 s / settlement) with drift inputs added (drift, z, implied price, Predict minus implied,
  the last 30 s drift change, high and low since the start, all its way; volatility): sets all, all + drift, Predict + drift, price
  level + time left + drift, drift only; pinned trees, leave-one-day-out: held-out rank correlation, top 1 % by predicted edge at the
  known best (TAKE_MARKOUT's rule), its value sold back after h (settlement: held), days > 0.
Outputs (write-once): OUT/btc_drift/<day>.parquet + .json (points), btc_drift.json (read).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/btc_drift.py \
       selftest | points <day> [<dir> <n rows>] | read [<dir>]
"""
import collections
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
import take_markout as TK  # noqa: E402
import take_markout_exit as TE  # noqa: E402
import take_model_insample as TI  # noqa: E402

TM, C, TR, F, Q, DAYS, fee = TK.TM, TK.C, TK.TR, TK.F, TK.Q, TK.DAYS, TK.fee
MM, WT = TK.MM, TK.WT
OUT = WT.OUT / 'btc_drift'
KNOWN_MS, MIN_S = 100., 300
VOLS = (('r30_15', 30, 15), ('r30_60', 30, 60), ('r1_15', 1, 15))       # name, return step (s), window (min); the first is primary
PV = VOLS[0][0]
ACT_MS = TM.FEED + TM.DECIDE + 22. + TM.HOLD
BINS = ((0, 10), (10, 20), (20, 35), (35, 65), (65, 80), (80, 90), (90, 100.01))
ZB = ((-np.inf, -2.), (-2., -1.), (-1., -.25), (-.25, .25), (.25, 1.), (1., 2.), (2., np.inf))
DRIFT = ['bd_drift_its', 'bd_z_its', 'bd_implied_its', 'bd_gap_its', 'bd_chg30_its', 'bd_high_its', 'bd_low_its', 'bd_vol']


def lab(lo, hi, cap=100):
    return f'{lo}-{min(hi, cap):g}'


def payout(r):
    if float(r['start_price']) == float(r['end_price']):
        return .5
    return 1. if r['outcome'] == 'Up' else 0.


class Spot:
    """Binance spot last trade and its trailing 1 s volatility over one UTC day."""

    def __init__(self, t, p, lo, hi):
        self.t, self.p, self.lo = t, p, lo
        n = int((hi - lo) // 1000.)
        g = lo + 1000. * np.arange(n + 1)
        j = np.searchsorted(t, g, 'right') - 1
        self.g = np.where(j >= 0, p[np.maximum(j, 0)], np.nan)
        lg = np.log(self.g)
        self.cr2, self.cn = {}, {}
        for step in sorted({v[1] for v in VOLS}):
            r = np.full(len(lg), np.nan); r[step:] = lg[step:] - lg[:-step]          # r[j] = the step-second return ending at grid j
            ok = np.isfinite(r)
            self.cr2[step] = np.concatenate([[0.], np.cumsum(np.where(ok, r * r, 0.))])
            self.cn[step] = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])

    def last(self, x):
        j = np.searchsorted(self.t, x, 'right') - 1
        return np.where(j >= 0, self.p[np.maximum(j, 0)], np.nan)

    def k(self, x):
        return np.clip(np.floor((np.asarray(x, float) - self.lo) / 1000.).astype(np.int64), 0, len(self.g) - 1)

    def vol(self, x, minutes, step=1):
        """Per-second volatility (log units) from the overlapping step-second returns ending at grid seconds (k - 60 x minutes, k], k the
        grid second at or before x."""
        k = self.k(x); a = np.maximum(k - 60 * minutes, 0)
        n = self.cn[step][k + 1] - self.cn[step][a + 1]; s = self.cr2[step][k + 1] - self.cr2[step][a + 1]
        return np.where(n >= MIN_S, np.sqrt(s / np.maximum(n, 1) / step), np.nan)


def its(d, up_value):
    return np.where(d > 0, up_value, 100. - up_value)


# ---------------------------------------------------------------------------------------------------------------- points
def points(day, od=None, limit=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1500, 1501))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    L0 = pd.read_parquet(TM.OUT / f'{day}_labels.parquet', columns=['row', 'day', 'mid', 't', 'd'])
    if limit:
        L0 = L0.iloc[:limit].reset_index(drop=True)
    rg = pd.read_parquet(MM.S1 / day / 'registry.parquet').set_index('mid')
    C.require(set(L0.mid) <= set(rg.index), 'a market missing from the registry')
    tv, pv, _, _, cov, _ = CV.load('binance_spot', day)
    S = Spot(tv, pv, cov[0], cov[1])
    status = collections.Counter()
    # alignment, every market of the population
    al = []
    for m in sorted(set(L0.mid)):
        r = rg.loc[m]
        ws, we = float(r.ws) * 1000., float(r.we) * 1000.
        s0, s1 = float(S.last(np.array([ws]))[0]), float(S.last(np.array([we]))[0])
        st, en, y = float(r.start_price), float(r.end_price), payout(r)
        al.append(dict(mid=int(m), start_gap_bps=1e4 * np.log(s0 / st) if np.isfinite(s0) else np.nan,
                       end_drift_bps=1e4 * np.log(s1 / st) if np.isfinite(s1) else np.nan, payout=y))
    out = []
    for m, g in L0.groupby('mid'):
        r = rg.loc[m]
        ws, we, st = float(r.ws) * 1000., float(r.we) * 1000., float(r.start_price)
        t, d = g.t.to_numpy(float), g.d.to_numpy(float)
        tk = t - KNOWN_MS
        s = S.last(tk)
        drift = np.log(s / st)
        tau = (we - t) / 1000.
        o = dict(row=g.row.to_numpy(), day=day, mid=int(m), d=d, tau=tau, drift=drift, payout=payout(r),
                 drift_end=np.log(float(r.end_price) / st))
        for nm, step, mn in VOLS:
            v = S.vol(tk, mn, step)
            o[f'vol_{nm}'] = v
            z = drift / (v * np.sqrt(np.maximum(tau, 1e-3)))
            o[f'z_{nm}'] = z
            o[f'implied_{nm}'] = 100. * norm.cdf(z)
        b30 = tk - 30000.
        o['drift_m30'] = np.where(b30 >= ws, np.log(S.last(b30) / st), np.nan)
        kw, kk = S.k(ws), S.k(tk)
        seg = np.log(S.g[kw:int(kk.max()) + 1] / st)
        hi_, lo_ = np.fmax.accumulate(seg), np.fmin.accumulate(seg)
        ix = kk - kw
        o['high'] = np.fmax(np.where(ix >= 0, hi_[np.maximum(ix, 0)], np.nan), drift)
        o['low'] = np.fmin(np.where(ix >= 0, lo_[np.maximum(ix, 0)], np.nan), drift)
        o['drift_p30'] = np.log(S.last(t + ACT_MS + 30000.) / st)
        out.append(pd.DataFrame(o))
    P = pd.concat(out, ignore_index=True)
    for c in ('drift', f'vol_{PV}', 'drift_m30', 'drift_p30'):
        status[f'nan_{c}'] = int(P[c].isna().sum())
    status['rows'] = len(P)
    A = pd.DataFrame(al)
    od.mkdir(parents=True, exist_ok=True)
    P.to_parquet(fp, index=False)
    TR.dump(od / f'{day}.json', dict(asof='2026-11-27', day=day, status=dict(status), spot_trades=int(len(tv)),
                                     alignment=dict(markets=int(len(A)), start_gap_bps_median=float(A.start_gap_bps.median()),
                                                    start_gap_bps_abs_max=float(A.start_gap_bps.abs().max()), rows=A.to_dict('records'))))
    C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    print(day, dict(status), 'start gap median bps', round(float(A.start_gap_bps.median()), 4), flush=True)


# ---------------------------------------------------------------------------------------------------------------- read
def alignment(al):
    A = pd.DataFrame(al)
    dec = A[A.payout != .5]
    agree = np.sign(dec.end_drift_bps) == np.where(dec.payout == 1., 1., -1.)
    dis = dec[~agree]
    ok = (abs(A.start_gap_bps.median()) < .5) and agree.mean() >= .95 and (dis.end_drift_bps.abs() < 1.).all()
    return dict(markets=int(len(A)), ties=int((A.payout == .5).sum()), start_gap_bps_median=float(A.start_gap_bps.median()),
                end_sign_agreement=float(agree.mean()), disagreements=int((~agree).sum()), disagreement_abs_drift_bps_max=float(dis.end_drift_bps.abs().max()) if len(dis) else 0.,
                passed=bool(ok))


def lodo(X, cols, y, ok, day):
    pred = np.full(len(X), np.nan)
    for dh in DAYS:
        tr, te = ok & (day != dh), day == dh
        if te.sum() == 0 or tr.sum() == 0:
            continue
        pred[te] = HistGradientBoostingRegressor(random_state=0, **TM.HGB).fit(X.loc[tr, cols], y[tr]).predict(X.loc[te, cols])
    return pred


def read(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    reh = od is not None
    od = Path(od) if od else None
    fn = (od or H) / 'btc_drift.json'
    C.require(not fn.exists(), 'btc_drift.json exists (write-once)')
    src = od or OUT
    al = [r for dh in DAYS if (src / f'{dh}.json').exists() for r in json.loads((src / f'{dh}.json').read_text())['alignment']['rows']]
    A = alignment(al)
    C.require(A['passed'], f'alignment failed: {A}')
    Pt = pd.concat([pd.read_parquet(src / f'{dh}.parquet') for dh in DAYS if (src / f'{dh}.parquet').exists()], ignore_index=True)
    X, feats = TI.frame()
    K = pd.concat([pd.read_parquet(TK.OUT / f'{d}_labels.parquet') for d in DAYS], ignore_index=True)
    E = pd.concat([pd.read_parquet(TE.OUT / f'{d}_labels.parquet') for d in DAYS], ignore_index=True)
    if not reh:
        C.require(len(Pt) == len(X) and set(Pt.row) == set(X.row), 'the drift points do not cover the population')
    X = X[X.row.isin(set(Pt.row))].reset_index(drop=True)
    X = X.merge(K.drop(columns=['day', 'mid', 't', 'd']), on='row', how='left', validate='1:1').merge(E.drop(columns=['day', 'mid', 'd']), on='row', how='left', validate='1:1')
    X = X.merge(Pt.drop(columns=['day', 'mid', 'd']), on='row', how='left', validate='1:1', suffixes=('', '_bd'))
    C.require(np.array_equal(X.payout.to_numpy(float), X.payout_bd.to_numpy(float)), 'payouts differ between TAKE_MARKOUT_EXIT and the registry')
    C.require(np.allclose(X.tau.to_numpy(float), X.p_tau.to_numpy(float), atol=1e-6), 'time left differs from public_score')
    d, day, mid = X.d.to_numpy(float), X.day.to_numpy(), X.mid.to_numpy()
    lim, md, pay = X.lim.to_numpy(float), X.md.to_numpy(float), X.payout.to_numpy(float)
    win = np.where(d > 0, pay, 1. - pay)
    ip = X.pf_its_price.to_numpy(float)
    up = d > 0
    # drift inputs, its way
    X['bd_drift_its'] = 1e4 * d * X.drift
    X['bd_z_its'] = d * X[f'z_{PV}']
    X['bd_implied_its'] = its(d, X[f'implied_{PV}'].to_numpy(float))
    X['bd_gap_its'] = ip - X.bd_implied_its
    X['bd_chg30_its'] = 1e4 * d * (X.drift - X.drift_m30)
    X['bd_high_its'] = 1e4 * np.where(up, X.high, -X.low)
    X['bd_low_its'] = 1e4 * np.where(up, X.low, -X.high)
    X['bd_vol'] = 1e4 * X[f'vol_{PV}']
    out = dict(asof='2026-11-27', rows=int(len(X)), rehearsal=reh, alignment=A, status={dh: json.loads((src / f'{dh}.json').read_text())['status'] for dh in DAYS if (src / f'{dh}.json').exists()})
    # 2 Predict vs BTC (Up space, one row per moment)
    okz = up & np.isfinite(X[f'implied_{PV}'].to_numpy(float))
    out['predict_vs_btc'] = dict(n=int(okz.sum()), spearman_mid_implied=float(spearmanr(X.pf_mid[okz], X[f'implied_{PV}'][okz]).correlation), gap_by_predict_price={})
    for lo, hi in BINS:
        m = np.isfinite(X.bd_implied_its.to_numpy()) & (ip >= lo) & (ip < hi)
        out['predict_vs_btc']['gap_by_predict_price'][lab(lo, hi)] = dict(n=int(m.sum()), predict_c=float(ip[m].mean()), implied_c=float(X.bd_implied_its[m].mean()),
                                                                        gap_c=float(X.bd_gap_its[m].mean()), gap_median_c=float(np.median(X.bd_gap_its[m])))
    # 3 calibration on the same moments
    cal = dict(by_implied={}, by_predict={}, cheap_split={})
    for nm, _, _ in VOLS:
        iv = its(d, X[f'implied_{nm}'].to_numpy(float))
        okv = np.isfinite(iv)
        cal['by_implied'][nm] = {}
        for lo, hi in BINS:
            m = okv & (iv >= lo) & (iv < hi)
            mk = pd.DataFrame(dict(mid=mid[m], d=d[m], w=win[m])).drop_duplicates(['mid', 'd'])
            cal['by_implied'][nm][lab(lo, hi)] = dict(n=int(m.sum()), implied_c=float(iv[m].mean()) if m.any() else None, won_pct=float(100. * win[m].mean()) if m.any() else None,
                                                            market_sides=int(len(mk)))
    ok15 = np.isfinite(X.bd_implied_its.to_numpy())
    for lo, hi in BINS:
        m = ok15 & (ip >= lo) & (ip < hi)
        cal['by_predict'][lab(lo, hi)] = dict(n=int(m.sum()), predict_c=float(ip[m].mean()), implied_c=float(X.bd_implied_its[m].mean()), won_pct=float(100. * win[m].mean()))
    cheap = ok15 & (ip < 10)
    for nm, m in (('implied_under_10c', cheap & (X.bd_implied_its.to_numpy() < 10)), ('implied_10c_or_more', cheap & (X.bd_implied_its.to_numpy() >= 10))):
        cal['cheap_split'][nm] = dict(n=int(m.sum()), markets=int(np.unique(mid[m]).size), predict_c=float(ip[m].mean()) if m.any() else None,
                                     implied_c=float(X.bd_implied_its[m].mean()) if m.any() else None, won_pct=float(100. * win[m].mean()) if m.any() else None)
    out['calibration'] = cal
    # 4 does BTC revert (Up space)
    zi = X[f'z_{PV}'].to_numpy(float)
    ch30 = np.where(X.tau.to_numpy(float) >= 31., 1e4 * (X.drift_p30 - X.drift).to_numpy(float), np.nan)     # the next 30 s inside the window
    che = 1e4 * (X.drift_end - X.drift).to_numpy(float)
    dr = 1e4 * X.drift.to_numpy(float)
    rv = dict(by_z={}, slope={})
    for a, b in ZB:
        m = up & np.isfinite(zi) & (zi >= a) & (zi < b)
        rv['by_z'][f'{a:g}..{b:g}'] = dict(n=int(m.sum()), markets=int(np.unique(mid[m]).size), drift_bps=float(dr[m].mean()) if m.any() else None,
                                          change_30s_bps=float(np.nanmean(ch30[m])) if m.any() else None, change_to_end_bps=float(np.nanmean(che[m])) if m.any() else None,
                                          up_won_pct=float(100. * pay[m].mean()) if m.any() else None, implied_up_c=float(np.nanmean(X[f'implied_{PV}'].to_numpy()[m])) if m.any() else None)
    for nm, ch in (('next_30s', ch30), ('to_end', che)):
        m = up & np.isfinite(ch) & np.isfinite(dr)
        rv['slope'][nm] = dict(n=int(m.sum()), slope=float(np.polyfit(dr[m], ch[m], 1)[0]),
                               by_day={dh: float(np.polyfit(dr[m & (day == dh)], ch[m & (day == dh)], 1)[0]) for dh in DAYS if (m & (day == dh)).sum() > 50})
    # variance ratio: the window's whole move against 300 s of the 1 s volatility at its first moment (1 = random walk)
    rv['variance_ratio'] = {}
    for nm, _, _ in VOLS:
        W = pd.DataFrame(dict(mid=mid, day=day, tau=X.tau.to_numpy(float), v=X[f'vol_{nm}'].to_numpy(float), e=X.drift_end.to_numpy(float)))[up]
        W = W[np.isfinite(W.v)].sort_values('tau', ascending=False, kind='stable').drop_duplicates('mid')
        vr = {dh: float((W.e[W.day == dh] ** 2).sum() / (300. * (W.v[W.day == dh] ** 2)).sum()) for dh in DAYS if (W.day == dh).any()}
        vr.update(pooled=float((W.e ** 2).sum() / (300. * (W.v ** 2)).sum()), markets=int(len(W)))
        rv['variance_ratio'][nm] = vr
    out['reversion'] = rv
    # 5 decisions
    cost = d * (lim - md) + fee(lim)
    PA = X[[f'pA172_{k}' for k in range(5)]].to_numpy(float); SA = X[[f'sA172_{k}' for k in range(5)]].to_numpy(float)
    fl, co, fe = TK.vwalk(PA, SA, d, lim)
    targets = {}
    for hs in ('1s', '30s'):
        mk = X[f'm{hs}_172'].to_numpy(float)
        eP = X[[f'eP{hs}_{k}' for k in range(5)]].to_numpy(float); eS = X[[f'eS{hs}_{k}' for k in range(5)]].to_numpy(float)
        back, hold, _ = TE.exit_values(fl, co, fe, d, eP, eS, pay)
        targets[hs] = (np.where(np.isfinite(mk), d * (mk - md), np.nan), np.isfinite(mk), back)
    targets['settlement'] = (d * (100. * pay - md), np.ones(len(X), bool), hold)
    PRED = ['pf_mid', 'pf_extreme', 'pf_its_price', 'p_tau', 'pf_spread', 'pf_far1', 'pf_near1', 'pf_far5', 'pf_near5', 'pf_mid_ret_1000', 'p_pf5', 'q_net_share5', 'q_fill_r1s']
    C.require(all(c in feats for c in PRED), 'a Predict input is missing')
    sets = {'all': feats, 'all_plus_drift': feats + DRIFT, 'predict_plus_drift': PRED + DRIFT, 'price_level_time_left_plus_drift': ['pf_its_price', 'p_tau'] + DRIFT,
            'drift_only': DRIFT}
    out['decisions'] = {}
    for tg, (y, ok, val) in targets.items():
        o = {}
        for sn, cols in sets.items():
            pred = lodo(X, cols, y, ok, day)
            idx = TK.top(pred - cost, ok, .01)
            v = val[idx] / Q
            by = {dh: (float(v[day[idx] == dh].mean()) if (day[idx] == dh).any() else None) for dh in DAYS}
            o[sn] = dict(spearman=float(spearmanr(pred[ok], y[ok]).correlation), top1_value_c=float(v.mean()), top1_move_c=float(y[idx].mean()), by_day=by,
                         days_positive=int(sum(1 for x in by.values() if x is not None and x > 0)), top1_its_price_median_c=float(np.median(ip[idx])))
            print(tg, sn, round(o[sn]['spearman'], 3), round(o[sn]['top1_value_c'], 2), o[sn]['days_positive'], flush=True)
        out['decisions'][tg] = o
        C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    TR.dump(fn, out)
    print(json.dumps(dict(alignment=A, predict_vs_btc=out['predict_vs_btc'], cheap_split=cal['cheap_split'], slope=rv['slope']), indent=1, default=str))


def selftest():
    # spot: a flat start then a step; the vol of a known +-1 bp alternation
    t = np.arange(0., 4000. * 1000., 1000.) + 500.
    p = 100. * np.exp(1e-4 * (np.arange(len(t)) % 2))
    S = Spot(t, p, 0., 4000. * 1000.)
    v = S.vol(np.array([3999. * 1000.]), 15, 1)[0]
    assert abs(v - 1e-4) < 1e-9, v
    assert np.isnan(S.vol(np.array([200. * 1000.]), 15, 1)[0])           # under 5 min of data
    v30 = S.vol(np.array([3999. * 1000.]), 15, 30)[0]
    assert v30 < 1e-6, v30                                                # an even-step alternation has no 30 s variance
    t2 = np.arange(0., 4000. * 1000., 1000.) + 500.
    p2 = 100. * np.exp(1e-4 * np.arange(len(t2)))                         # a steady trend: 30 s returns = 30 x the 1 s return
    S2 = Spot(t2, p2, 0., 4000. * 1000.)
    v1, v30 = S2.vol(np.array([3999. * 1000.]), 15, 1)[0], S2.vol(np.array([3999. * 1000.]), 15, 30)[0]
    assert abs(v1 - 1e-4) < 1e-9 and abs(v30 - 1e-4 * np.sqrt(30.)) < 1e-8, (v1, v30)
    assert S.last(np.array([400.]))[0] != S.last(np.array([400.]))[0]     # nothing before the first trade
    assert abs(S.last(np.array([1600.]))[0] - p[1]) < 1e-12
    # its way: a buy of Down sees 100 - the Up price
    assert its(np.array([1., -1.]), np.array([30., 30.])).tolist() == [30., 70.]
    # alignment passes on agreeing markets and refuses a sign error and a start offset
    good = [dict(start_gap_bps=0., end_drift_bps=3., payout=1.), dict(start_gap_bps=0., end_drift_bps=-2., payout=0.)] * 20 + [dict(start_gap_bps=0., end_drift_bps=0., payout=.5)]
    assert alignment(good)['passed']
    bad = good + [dict(start_gap_bps=0., end_drift_bps=5., payout=0.)]
    assert not alignment(bad)['passed']
    off = [dict(r, start_gap_bps=2.) for r in good]
    assert not alignment(off)['passed']
    assert payout(dict(start_price=1., end_price=1., outcome='Up')) == .5 and payout(dict(start_price=1., end_price=2., outcome='Down')) == 0.
    print('btc_drift selftest PASS')


if __name__ == '__main__':
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == 'points':
        points(a[0], a[1] if len(a) > 1 else None, int(a[2]) if len(a) > 2 else None)
    elif cmd == 'read':
        read(a[0] if a else None)
    else:
        dict(selftest=selftest)[cmd]()
