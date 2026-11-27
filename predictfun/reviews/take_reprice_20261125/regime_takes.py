"""Are some market regimes better for taking? (USER 2026-11-27: "If the model can't generalize, maybe it is the market regime issue, some
market regime is better for taking, we can also check the taking rate, etc to consider market regime, what do you think"; "Ok go").
Development evidence on the four consumed days, on EVERY market (CHEAP_SIDE_ALL: 'complete' markets are outcome-selected). Declared
before the run.
Population: public_score3's random moments (both directions) in every market whose recorded path holds the decision (>= 1 s in), the
match and 1 s after it (TAKE_MODEL's rule, any path status), with a valid known book and a valid mid 1 s after the match; statuses
counted. TAKE_MODEL's rows are among them: their labels are recomputed and must match TAKE_MODEL's (parity, 0 differences).
Take (TAKE_MODEL's): a 50-share IOC decided at the moment, matched after 22 ms + 150 ms against the venue book (snapshot at or before),
limit = the known best + 0 / 3c / every shown level, valued at the mid 1 s after the match, fee per level; per intended share, misses 0.
Regimes (known at the decision, minutes-scale; quintiles over all labelled moments, NaN its own bucket):
  take_rate_60s / take_rate_300s: taker orders per minute on Predict's BTC 5 m markets over the last 60 / 300 s (every taker; WS-timed
    fills only, known 200 ms after their WS time; legs grouped by taker order; the same UTC day only);
  top10_rate_300s: the same for the ten most profitable takers of these days (identities chosen on these days: in-sample);
  btc_vol_15m: Binance spot volatility from overlapping 30 s returns over the last 15 min (spot known 100 ms late);
  spread_60s: Predict's mean spread over the last 60 s of known snapshots; book_updates_60s: known snapshots per second over it.
Readouts (stage read), per regime variable and quintile: random takes (every moment in it) and the model's top 1 % within it (by
predicted edge at the known best: TAKE_MARKOUT's target, the 1 s markout move, pinned trees, leave-one-day-out; once on TAKE_MODEL's 79
inputs, once with the six regime inputs added): value per intended share at each limit, fill, move, days > 0, markets. Overall: both
models' held-out rank correlation and top 1 %. Paying (computed): > 0 pooled AND >= 3 of 4 days AND >= 150 orders. Null (declared,
200 draws): each day's regime values rolled in time by a random 1/8-7/8 of its moments (keeps their persistence, breaks their timing),
recounting the paying cells; the observed count against the null's 95th percentile.
Outputs (write-once): OUT/regime_takes/<day>.parquet + .json (labels), regime_takes.json (read).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/regime_takes.py \
       selftest | labels <day> [<dir> <n markets>] | read [<dir>]
"""
import collections
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import cross_venue as CV  # noqa: E402
import fire_times as FT  # noqa: E402
import take_markout as TK  # noqa: E402

TM, VR, MM, TR, C, F, WT = TK.TM, TK.VR, TK.MM, TK.TR, TK.C, TK.F, TK.WT
OUT = WT.OUT / 'regime_takes'
DAYS, Q, fee = TK.DAYS, TK.Q, TK.fee
NET = 22.
REGIMES = ('take_rate_60s', 'take_rate_300s', 'top10_rate_300s', 'btc_vol_15m', 'spread_60s', 'book_updates_60s')
ORDERS = ('s0', 's3', 'any')
N_NULL, MIN_ORDERS, SEED = 200, 150, 20261127
LEG_KNOWN_MS, SPOT_KNOWN_MS = 200., 100.


def walk_parity(r0, got):
    return [k for k, v in got.items() if not abs(float(v) - float(r0[k])) <= (1e-12 if k.startswith('fill_') else 1e-9)]


def order_times(day, takers=None):
    """WS-timed taker orders of the day (first leg's WS time per taker order), optionally for a set of takers -> sorted ms."""
    L = pd.read_parquet(WT.OUT / f'winners_legs_{day}.parquet', columns=['t_ms2', 'how2', 'taker', 'taker_hash'])
    L = L[(L.how2 == 1) & L.t_ms2.notna()]
    if takers is not None:
        L = L[L.taker.isin(takers)]
    return np.sort(L.groupby('taker_hash').t_ms2.min().to_numpy(float))


def rate(times, t, window_ms, day_lo):
    """Orders per minute over (t - known - window, t - known], the window cut at the day start (>= 60 s needed, else NaN)."""
    hi = t - LEG_KNOWN_MS
    lo = np.maximum(hi - window_ms, day_lo)
    n = np.searchsorted(times, hi, 'right') - np.searchsorted(times, lo, 'right')
    span = hi - lo
    return np.where(span >= 60000., n / (span / 60000.), np.nan)


class SpotVol:
    """Binance spot last trade on a 1 s grid; trailing volatility per second from overlapping 30 s returns."""

    def __init__(self, t, p, lo, hi, step=30):
        n = int((hi - lo) // 1000.)
        g = lo + 1000. * np.arange(n + 1)
        j = np.searchsorted(t, g, 'right') - 1
        lg = np.log(np.where(j >= 0, p[np.maximum(j, 0)], np.nan))
        r = np.full(len(lg), np.nan); r[step:] = lg[step:] - lg[:-step]
        ok = np.isfinite(r)
        self.lo, self.step, self.n = lo, step, len(lg)
        self.cr2 = np.concatenate([[0.], np.cumsum(np.where(ok, r * r, 0.))])
        self.cn = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])

    def vol(self, x, minutes=15, min_n=300):
        k = np.clip(np.floor((np.asarray(x, float) - SPOT_KNOWN_MS - self.lo) / 1000.).astype(np.int64), 0, self.n - 1)
        a = np.maximum(k - 60 * minutes, 0)
        m = self.cn[k + 1] - self.cn[a + 1]; s = self.cr2[k + 1] - self.cr2[a + 1]
        return np.where(m >= min_n, 1e4 * np.sqrt(s / np.maximum(m, 1) / self.step), np.nan)


def labels(day, od=None, limit=None):
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    Z = pd.read_parquet(WT.OUT / 'public_score3_points.parquet', columns=['day', 't', 'd', 'kind'])
    Z['row'] = np.arange(len(Z))
    Z = Z[(Z.day == day) & (Z.kind == 'random')].reset_index(drop=True)
    reg = MM.registry(day).set_index('mid')
    ws_to_mid = {int(float(reg.loc[m, 'ws'])): int(m) for m in reg.index}
    Z['mid'] = (np.floor(Z.t / 300000.) * 300).astype(int).map(ws_to_mid)
    status = collections.Counter()
    status['no_market'] = int(Z.mid.isna().sum())
    Z = Z[Z.mid.notna()].reset_index(drop=True)
    Z['mid'] = Z.mid.astype(int)
    L0 = pd.read_parquet(TM.OUT / f'{day}_labels.parquet').set_index('row')
    day_lo, day_hi = (float(x) for x in CV.WT.LV.day_ms(day))
    T_all = order_times(day)
    T_top = order_times(day, set(FT.top10()))
    tv, pv, _, _, cov, _ = CV.load('binance_spot', day)
    SV = SpotVol(tv, pv, cov[0], cov[1])
    by_mid = Z.groupby('mid').indices
    rows, parity, path_st = [], collections.Counter(), {}
    src = MM.KP.Tape(); off = src.offset(day); n_m = 0
    for h, g in reg.groupby('hour'):
        mids = [int(m) for m in g.index if int(m) in by_mid]
        if limit is not None:
            mids = mids[:max(0, limit - n_m)]
        if not mids:
            continue
        for m, W, lst in VR.worlds_all(day, mids, src, off, reg):
            n_m += 1
            if lst != 'loaded':
                status[f'loader_{lst}'] += len(by_mid[m]); path_st[m] = f'loader_{lst}'
                continue
            ws = float(reg.loc[m, 'ws']) * 1000.
            p, st, info = VR.vm_path(W, ws, 50.)
            path_st[m] = st
            if p is None:
                status[f'path_{st}'] += len(by_mid[m])
                continue
            t0, end = float(info['t_start']), float(info['end_rel'])
            f = np.asarray(p['frames'], float)
            fk = f[:, 1]
            spr = np.where((f[:, 2] >= 0) & (f[:, 12] <= 100) & (f[:, 2] < f[:, 12]), f[:, 12] - f[:, 2], np.nan)
            cs = np.concatenate([[0.], np.cumsum(np.nan_to_num(spr))]); cv = np.concatenate([[0], np.cumsum(np.isfinite(spr))])
            for i in by_mid[m]:
                t, d, rw = float(Z.t.iat[i]), float(Z.d.iat[i]), int(Z.row.iat[i])
                g_ = t + TM.FEED + TM.DECIDE - t0
                act = g_ + NET + TM.HOLD
                if g_ < 1000. or act + 1000. > end:
                    status['outside_path'] += 1
                    continue
                jk = int(np.searchsorted(fk, g_, 'right') - 1)
                j1 = int(np.searchsorted(fk, g_ - 1000., 'right') - 1)
                if jk < 0 or j1 < 0:
                    status['no_known_frame'] += 1
                    continue
                rk = f[jk]
                bid, ask = rk[2], rk[12]
                if not (0 <= bid < ask <= 100):
                    status['invalid_known_book'] += 1
                    continue
                jv = int(np.searchsorted(f[:, 0], act, 'right') - 1)
                ja = int(np.searchsorted(f[:, 0], act + 1000., 'right') - 1)
                ma = TM.mid_of(f[ja]) if (act + 1000.) - f[ja, 0] <= 1000. else np.nan
                if not np.isfinite(ma):
                    status['no_mid_after'] += 1
                    continue
                side = 1 if d > 0 else 0
                lim = ask if d > 0 else bid
                md = (bid + ask) / 2.
                m1 = TM.mid_of(f[j1])
                bs = np.where(rk[2:7] >= 0, rk[7:12], 0.); as_ = np.where(rk[12:17] >= 0, rk[17:22], 0.)
                r = dict(row=rw, day=day, mid=m, t=t, d=d, lim=lim, md=md, m1s=ma, path_status=st, in_take_model=rw in L0.index,
                         pf_mid=md, pf_spread=ask - bid, pf_far1=as_[0] if d > 0 else bs[0], pf_near1=bs[0] if d > 0 else as_[0],
                         pf_far5=as_.sum() if d > 0 else bs.sum(), pf_near5=bs.sum() if d > 0 else as_.sum(),
                         pf_mid_ret_1000=d * (md - m1) if np.isfinite(m1) else np.nan, pf_extreme=abs(md - 50.), pf_its_price=md if d > 0 else 100. - md)
                pa, sa = TK.side_levels(f[jv], side)
                for k in range(5):
                    r[f'pA_{k}'], r[f'sA_{k}'] = pa[k], sa[k]
                j60 = int(np.searchsorted(fk, g_ - 60000., 'right'))
                span = g_ - max(g_ - 60000., float(fk[0]))
                nv = cv[jk + 1] - cv[j60]
                r['spread_60s'] = (cs[jk + 1] - cs[j60]) / nv if (nv > 0 and span >= 10000.) else np.nan
                r['book_updates_60s'] = (jk + 1 - j60) / (span / 1000.) if span >= 10000. else np.nan
                if r['in_take_model']:
                    r0 = L0.loc[rw]
                    q, vw = TM.walk(f[jv], side, d, lim, Q)
                    got = dict(pf_mid=md, v_172_0=q * (d * (ma - vw) - fee(vw)) / Q if q > 0 else 0., fill_172_0=q / Q, mv_172=d * (ma - md))
                    parity['checked'] += 1
                    for k in walk_parity(r0, got):
                        parity[f'{k}_differs'] += 1
                status['ok'] += 1
                rows.append(r)
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    L = pd.DataFrame(rows)
    tt = L.t.to_numpy(float)
    L['take_rate_60s'] = rate(T_all, tt, 60000., day_lo)
    L['take_rate_300s'] = rate(T_all, tt, 300000., day_lo)
    L['top10_rate_300s'] = rate(T_top, tt, 300000., day_lo)
    L['btc_vol_15m'] = SV.vol(tt)
    bad = {k: v for k, v in parity.items() if k.endswith('_differs')}
    C.require(not bad, f'parity with TAKE_MODEL failed: {bad}')
    if limit is None:
        C.require(int(L.in_take_model.sum()) == len(L0), f'TAKE_MODEL rows found {int(L.in_take_model.sum())} of {len(L0)}')
    od.mkdir(parents=True, exist_ok=True)
    L.to_parquet(fp, index=False)
    TR.dump(od / f'{day}.json', dict(asof='2026-11-27', day=day, markets=n_m, status=dict(status), parity=dict(parity), path_status=dict(collections.Counter(path_st.values())),
                                     ws_taker_orders=int(len(T_all)), top10_orders=int(len(T_top)), regime_nan={k: int(L[k].isna().sum()) for k in REGIMES}))
    print(day, n_m, dict(status), dict(parity), {k: int(L[k].isna().sum()) for k in REGIMES}, flush=True)


# ---------------------------------------------------------------------------------------------------------------- read
def quintiles(x):
    ok = np.isfinite(x)
    q = np.full(len(x), -1, np.int64)
    edges = None
    if ok.sum() >= 5:
        edges = np.quantile(x[ok], [.2, .4, .6, .8])
        q[ok] = np.searchsorted(edges, x[ok], 'right')
    return q, edges


def frame(src):
    L = pd.concat([pd.read_parquet(src / f'{dh}.parquet') for dh in DAYS if (src / f'{dh}.parquet').exists()], ignore_index=True)
    C.require(L.row.is_unique, 'a moment twice')
    Pz = pd.read_parquet(WT.OUT / 'public_score3_points.parquet')
    Pz['row'] = np.arange(len(Pz))
    Pz = Pz[Pz.row.isin(set(L.row))]
    NWp = pd.concat([pd.read_parquet(WT.OUT / 'norm_windows' / f'{d}.parquet', columns=['day', 't', 'd', 'kind'] + TM.NW_COLS) for d in DAYS], ignore_index=True)
    NWp = NWp[NWp.kind == 'random'].drop_duplicates(['day', 't', 'd'])
    X = L.merge(Pz, on='row', how='inner', suffixes=('', '_p'), validate='1:1')
    C.require(len(X) == len(L) and np.allclose(X.t.to_numpy(), X.t_p.to_numpy()), 'labels and features do not align')
    X = X.merge(NWp.drop(columns=['kind']), on=['day', 't', 'd'], how='left', validate='m:1')
    feats = [c for c in Pz.columns if c not in TM.EXCL and c != 'row'] + [c for c in L.columns if c.startswith('pf_')] + TM.NW_COLS
    C.require(len(feats) == 79 and len(set(feats)) == 79, f'{len(feats)} inputs, TAKE_MODEL has 79')
    return X, feats


def cell_stats(v, fl, mv, m, day, mid):
    n = int(m.sum())
    if n == 0:
        return dict(orders=0)
    by = {dh: (float(v[m & (day == dh)].mean() / Q) if (m & (day == dh)).any() else None) for dh in DAYS}
    dp = int(sum(1 for x in by.values() if x is not None and x > 0))
    val = float(v[m].mean() / Q)
    return dict(orders=n, markets=int(np.unique(mid[m]).size), value_c=val, fill=float(fl[m].sum() / (Q * n)), move_c=float(mv[m].mean()), by_day=by, days_positive=dp,
                paying=bool(val > 0 and dp >= 3 and n >= MIN_ORDERS))


def top_within(score, groups, ok, q=.01):
    """Per group the top q (>= 1) by score among ok rows -> a boolean mask."""
    sel = np.zeros(len(score), bool)
    for gv in np.unique(groups[ok]):
        i = np.flatnonzero(ok & (groups == gv))
        k = max(1, int(round(q * len(i))))
        sel[i[np.argsort(-score[i], kind='stable')[:k]]] = True
    return sel


def paying_count(Qs, V, score_e, ok, day, mid, fl, mv):
    """Paying cells over regimes x quintiles x {random, model top 1 %} x orders for given quintile labels."""
    n = 0
    for rg_, q in Qs.items():
        tops = top_within(score_e, q, ok & (q >= 0))
        for b in range(5):
            mb = ok & (q == b)
            for sel in (mb, mb & tops):
                for o in ORDERS:
                    v = V[o]
                    m = sel
                    if not m.any() or m.sum() < MIN_ORDERS:
                        continue
                    val = v[m].mean()
                    if val <= 0:
                        continue
                    dp = sum(1 for dh in DAYS if (m & (day == dh)).any() and v[m & (day == dh)].mean() > 0)
                    n += int(dp >= 3)
    return n


def read(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3601))
    reh = od is not None
    od = Path(od) if od else None
    fn = (od or H) / 'regime_takes.json'
    C.require(not fn.exists(), 'regime_takes.json exists (write-once)')
    src = od or OUT
    X, feats = frame(src)
    d, lim, md, day, mid = X.d.to_numpy(float), X.lim.to_numpy(float), X.md.to_numpy(float), X.day.to_numpy(), X.mid.to_numpy()
    mk = X.m1s.to_numpy(float)
    mo = d * (mk - md)
    cost = d * (lim - md) + fee(lim)
    PA = X[[f'pA_{k}' for k in range(5)]].to_numpy(float); SA = X[[f'sA_{k}' for k in range(5)]].to_numpy(float)
    V, FL = {}, {}
    for o, L_ in (('s0', lim), ('s3', lim + 3. * d), ('any', np.where(d > 0, 99., 1.))):
        fl, co, fe = TK.vwalk(PA, SA, d, L_)
        V[o] = d * (mk * fl - co) - fe; FL[o] = fl
    ok = np.isfinite(mo)
    out = dict(asof='2026-11-27', rows=int(len(X)), rehearsal=reh, path_status_moments=X.path_status.value_counts().to_dict(),
               label_status={dh: json.loads((src / f'{dh}.json').read_text()) for dh in DAYS if (src / f'{dh}.json').exists()}, models={}, regimes={})
    # models
    preds = {}
    for nm, cols in (('take_model_inputs', feats), ('with_regime_inputs', feats + list(REGIMES))):
        pred = np.full(len(X), np.nan)
        for dh in DAYS:
            tr, te = ok & (day != dh), day == dh
            if te.sum() == 0 or tr.sum() == 0:
                continue
            pred[te] = HistGradientBoostingRegressor(random_state=0, **TM.HGB).fit(X.loc[tr, cols], mo[tr]).predict(X.loc[te, cols])
        preds[nm] = pred
        e = pred - cost
        top = TK.top(e, ok, .01)
        tm = np.zeros(len(X), bool); tm[top] = True
        out['models'][nm] = dict(spearman=float(spearmanr(pred[ok], mo[ok]).correlation), top1={o: cell_stats(V[o], FL[o], mo, tm, day, mid) for o in ORDERS})
        print(nm, round(out['models'][nm]['spearman'], 3), {o: round(c['value_c'], 2) for o, c in out['models'][nm]['top1'].items()}, flush=True)
    out['all_moments'] = {o: cell_stats(V[o], FL[o], mo, ok, day, mid) for o in ORDERS}
    # regimes
    e = preds['with_regime_inputs'] - cost
    Qs = {}
    for rg_ in REGIMES:
        q, edges = quintiles(X[rg_].to_numpy(float))
        Qs[rg_] = q
        tops = top_within(e, q, ok & (q >= 0))
        o_ = dict(edges=[float(x) for x in edges] if edges is not None else None, nan=int((q < 0).sum()), quintiles={})
        for b in list(range(5)) + [-1]:
            mb = ok & (q == b)
            if not mb.any():
                continue
            o_['quintiles'][str(b) if b >= 0 else 'nan'] = dict(mean_value=float(np.nanmean(X[rg_].to_numpy(float)[mb])) if b >= 0 else None,
                                                             random={o: cell_stats(V[o], FL[o], mo, mb, day, mid) for o in ORDERS},
                                                             model_top1={o: cell_stats(V[o], FL[o], mo, mb & tops, day, mid) for o in ORDERS})
        out['regimes'][rg_] = o_
        print(rg_, {b: tuple(round(c[s]['s0' if k < 2 else 'any'].get('value_c', np.nan), 2) for k, s in enumerate(('random', 'model_top1', 'model_top1')))
                    for b, c in o_['quintiles'].items()}, flush=True)
    cells = [(rg_, b, s, o, c) for rg_, r in out['regimes'].items() for b, qd in r['quintiles'].items() if b != 'nan'
             for s in ('random', 'model_top1') for o, c in qd[s].items()]
    paying = [dict(regime=rg_, quintile=b, selection=s, order=o, value_c=c['value_c'], orders=c['orders'], days_positive=c['days_positive'])
              for rg_, b, s, o, c in cells if c.get('paying')]
    obs = paying_count(Qs, V, e, ok, day, mid, FL, mo)
    C.require(obs == len(paying), f'the paying count differs: {obs} vs {len(paying)}')
    # null: each day's regime values rolled in time
    rng = np.random.default_rng(SEED)
    order_t = {dh: np.flatnonzero(day == dh)[np.argsort(X.t.to_numpy(float)[day == dh], kind='stable')] for dh in DAYS if (day == dh).any()}
    null = []
    for _ in range(N_NULL if not reh else 20):
        Qn = {}
        for rg_, q in Qs.items():
            qn = q.copy()
            for dh, ix in order_t.items():
                k = int(rng.integers(len(ix) // 8, 7 * len(ix) // 8 + 1))
                qn[ix] = np.roll(q[ix], k)
            Qn[rg_] = qn
        null.append(paying_count(Qn, V, e, ok, day, mid, FL, mo))
    null = np.asarray(null)
    out['verdict'] = dict(cells=len(cells), paying=paying, paying_count=len(paying), null_draws=int(len(null)), null_mean=float(null.mean()), null_p95=float(np.quantile(null, .95)),
                          beats_null=bool(len(paying) > np.quantile(null, .95)),
                          best_cell=max(({k: v for k, v in dict(regime=rg_, quintile=b, selection=s, order=o, value_c=c['value_c'], orders=c['orders'], days_positive=c['days_positive']).items()}
                                         for rg_, b, s, o, c in cells if c.get('orders', 0) >= MIN_ORDERS), key=lambda x: x['value_c']))
    TR.dump(fn, out)
    print(json.dumps(out['verdict'], indent=1, default=str))


def selftest():
    times = np.array([1000., 61000., 62000., 119000.])
    r = rate(times, np.array([120200.]), 60000., 0.)                     # (60 s, 120 s]: 61, 62, 119 -> 3 per minute
    assert abs(r[0] - 3.) < 1e-12, r
    assert np.isnan(rate(times, np.array([30200.]), 60000., 0.)[0])        # under 60 s of the day
    r5 = rate(times, np.array([120200.]), 300000., 0.)                     # cut at the day start: (0, 120 s] -> 4 in 2 min
    assert abs(r5[0] - 2.) < 1e-12, r5
    t = np.arange(0., 4000. * 1000., 1000.) + 500.
    p = 100. * np.exp(1e-4 * np.arange(len(t)))
    sv = SpotVol(t, p, 0., 4000. * 1000.)
    v = sv.vol(np.array([3999. * 1000.]))[0]
    assert abs(v - 1e4 * 1e-4 * np.sqrt(30.)) < 1e-6, v
    q, edges = quintiles(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., np.nan]))
    assert q[-1] == -1 and sorted(set(q[:-1].tolist())) == [0, 1, 2, 3, 4]
    g = np.array([0, 0, 0, 1, 1, 1]); sc = np.array([1., 3., 2., 5., 4., 6.])
    assert top_within(sc, g, np.ones(6, bool), q=.34).tolist() == [False, True, False, False, False, True]
    m = np.ones(4, bool)
    c = cell_stats(np.array([50., -10., 20., 40.]), np.array([50., 50., 50., 50.]), np.zeros(4), m, np.array(DAYS), np.array([1, 2, 3, 4]))
    assert abs(c['value_c'] - 100. / 4 / Q) < 1e-12 and c['days_positive'] == 3 and c['paying'] is False     # 4 orders < 150
    assert walk_parity(dict(v=1., fill_x=1.), dict(v=1., fill_x=1.)) == [] and walk_parity(dict(v=1.), dict(v=1.1)) == ['v']
    print('regime_takes selftest PASS')


if __name__ == '__main__':
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == 'labels':
        labels(a[0], a[1] if len(a) > 1 else None, int(a[2]) if len(a) > 2 else None)
    elif cmd == 'read':
        read(a[0] if a else None)
    else:
        dict(selftest=selftest)[cmd]()
