"""Outside-venue feeds for research features (2026-11-04; loaders as in pf_cross_venue). Hours inside 10-28..30 and after 11-02
are never loaded; only the hour before the day is added (lookback), never the hour after.
Feeds: Binance USD-M bookTicker mid (E), Binance spot aggTrades (transact time; last price and signed aggressor qty),
Hyperliquid BTC bbo mid and trades (t_ms; 'B' = buyer aggressor, checked 99% in pf_cross_venue), Predict's Chainlink relay
(receipt time). Series are (times ms, log values) with repeats dropped; flows are (unique times, cumulative signed qty).
"""
import os, json, zipfile
import numpy as np, pandas as pd
from . import pf_io as io

SKIP = ('20261028', '20261029', '20261030'); LAST = '20261102'
HL = f'{io.ROOT}/data/mm_hf/hl_raw'; SPOT = f'{io.ROOT}/data/binance_spot_vision/aggTrades/BTCUSDT'; CL = f'{io.REF_RUN}/reference_prices'


def hours(day):
    return [h for h in io.hours_of(day, extra_before=1, extra_after=0) if h[:8] not in SKIP and h[:8] <= LAST]


def series(t, v):
    o = np.argsort(t, kind='stable'); t, v = np.asarray(t, np.float64)[o], np.asarray(v, np.float64)[o]
    keep = np.r_[True, v[1:] != v[:-1]] if len(v) else np.zeros(0, bool)
    return t[keep], v[keep]


def flow(t, q):
    o = np.argsort(t, kind='stable'); t, q = np.asarray(t, np.float64)[o], np.asarray(q, np.float64)[o]
    if not len(t):
        return t, np.zeros(1)
    u, idx = np.unique(t, return_index=True)
    return u, np.r_[0.0, np.cumsum(np.add.reduceat(q, idx))]


def load(day, days_available):
    F = {}; hs = hours(day)
    k, m = [], []
    for h in hs:
        p = f'{io.BN_DIR}/{h}.csv.gz'
        if os.path.exists(p):
            x = pd.read_csv(p, header=None, usecols=[1, 4, 6]).apply(pd.to_numeric, errors='coerce').dropna(); x = x[(x[4] > 0) & (x[6] > x[4])]
            kk, mm = series(x[1].values, np.log((x[4].values + x[6].values) / 2.0)); del x; k.append(kk); m.append(mm)
    F['bnf'] = series(np.concatenate(k), np.concatenate(m))
    k, m = [], []
    for h in hs:
        p = f'{HL}/bbo/BTC/{h}.csv.gz'
        if os.path.exists(p):
            x = pd.read_csv(p, header=None, usecols=[1, 2, 4]).apply(pd.to_numeric, errors='coerce').dropna(); x = x[(x[2] > 0) & (x[4] > x[2])]
            k.append(x[1].values); m.append(np.log((x[2].values + x[4].values) / 2.0))
    F['hl'] = series(np.concatenate(k), np.concatenate(m))
    k, q = [], []
    for h in hs:
        p = f'{HL}/trades/BTC/{h}.csv.gz'
        if os.path.exists(p):
            x = pd.read_csv(p, header=None, usecols=[1, 4, 5]).dropna()
            k.append(pd.to_numeric(x[1]).values.astype(float)); q.append(np.where(x[5].astype(str).values == 'B', 1.0, -1.0) * pd.to_numeric(x[4]).values)
    F['hl_tr'] = flow(np.concatenate(k), np.concatenate(q))
    k, pr, q = [], [], []
    prev = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime('%Y%m%d')
    for dd in ([prev] if prev in days_available else []) + [day]:
        if dd in SKIP or dd > LAST:
            continue
        p = f"{SPOT}/BTCUSDT-aggTrades-{dd[:4]}-{dd[4:6]}-{dd[6:]}.zip"
        if not os.path.exists(p):
            continue
        with zipfile.ZipFile(p) as z:
            with z.open(z.namelist()[0]) as fh:
                x = pd.read_csv(fh, header=None, usecols=[1, 2, 5, 6])
        t = x[5].values.astype(np.float64); t = t / 1000.0 if t.max() > 1e14 else t
        if dd != day:
            keep = t >= pd.Timestamp(day, tz='UTC').value / 1e6 - 3600e3; x = x[keep]; t = t[keep]
        k.append(t); pr.append(np.log(x[1].values.astype(float))); q.append(np.where(x[6].astype(str).str.lower().values == 'true', -1.0, 1.0) * x[2].values.astype(float))
    k = np.concatenate(k); F['spot'] = series(k, np.concatenate(pr)); F['spot_tr'] = flow(k, np.concatenate(q))
    k, m = [], []
    for h in hs:
        p = f'{CL}/{h}.jsonl.gz'
        if not os.path.exists(p):
            continue
        for line in io._lines(p):
            r = json.loads(line)
            if r.get('kind') == 'PRICE' and r.get('price') and not r.get('duplicate_previous_payload'):
                k.append(r['recv_ns'] / 1e6); m.append(np.log(float(r['price'])))
    F['cl'] = series(np.array(k), np.array(m))
    return F


def last(series_, t):
    """Value of the series in force at each t (NaN before the first) and its age (ms)."""
    K, V = series_; i = np.searchsorted(K, t, side='right') - 1; ok = i >= 0
    return np.where(ok, V[np.clip(i, 0, None)], np.nan), np.where(ok, t - K[np.clip(i, 0, None)], np.nan)


def move(series_, t, w):
    a, _ = last(series_, t - w); b, _ = last(series_, t)
    return (b - a) * 1e4


def vol(fl, t, w):
    U, C = fl
    if not len(U):
        return np.full(len(t), np.nan)
    a = np.searchsorted(U, t - w, side='right'); b = np.searchsorted(U, t, side='right')
    return np.where(U[0] <= t - w, C[b] - C[a], np.nan)
