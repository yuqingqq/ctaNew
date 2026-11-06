"""Do the sweepers (and the makers who dodge them) see flow on other venues first? (declared in STATUS.yml 2026-11-03 23:49Z;
user: "check if others see the flow in other exchanges or perps/spot flows"). Descriptive, 8 read days, nothing selected.

Events: market sweeps over 2+ prices and our replay swept fills (step4/sweep_precursor/{sweeps,ours}.parquet; bad = informed
<= -2 c/sh for sweeps, net <= -2 for ours), plus control times (10 per live window, uniform in 0-240 s, random direction).
Each event has an adverse direction for the swept makers (dd -1: BTC falling hurts bid makers). Signals before the event at t,
signed so that positive = adverse:
  mid feeds    Binance USD-M bookTicker mid (E), Hyperliquid BTC perp bbo mid (t_ms), Binance spot last trade price
               (transact time), Predict's Chainlink relay price (receipt time; publish times are whole seconds)
               move over [t - w, t] in bp; lag since a 0.5 bp adverse crossing within 1 s
  flow feeds   aggressor volume (BTC, buys - sells, signed adverse) over [t - w, t]: Binance USD-M trades (T), Hyperliquid
               trades (side verified against the prevailing bbo below), Binance spot aggTrades
A signal is 'present' when it exceeds the 90th percentile of the same signal at control times (so 10% at controls).
'In time for us' = the same window ending at t - 25.5 ms (our cancel lands 25.5 ms after we know a Binance-speed signal).
Hours inside 10-28..30 (sealed) and after 11-02 are never loaded; events whose lookback reaches them, or that have no feed
data in the lookback, are statuses (NaN signal).
Usage: python3 -m predictfun.research.pf_cross_venue
"""
import os, glob, json, zipfile
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
RUN = os.environ.get('PF_CV_DAYS', ','.join(DAYS)).split(',')
SKIP = ('20261028', '20261029', '20261030'); LAST = '20261102'
S4 = f'{io.OUT}/step4'; OUTD = f"{S4}/cross_venue{os.environ.get('PF_CV_SFX', '')}"; os.makedirs(OUTD, exist_ok=True)
BT = io.BN_DIR.replace('bookTicker', 'trade'); HL = f'{io.ROOT}/data/mm_hf/hl_raw'; SPOT = f'{io.ROOT}/data/binance_spot_vision/aggTrades/BTCUSDT'
CL = f'{io.REF_RUN}/reference_prices'
WS_ = (100.0, 250.0, 1000.0); LATE = 25.5; RNG = np.random.default_rng(20261103)


def hours(day):
    return [h for h in io.hours_of(day, extra_before=1, extra_after=0) if h[:8] not in SKIP and h[:8] <= LAST]


def series(t, v):
    o = np.argsort(t, kind='stable'); t, v = t[o].astype(np.float64), v[o].astype(np.float64)
    keep = np.r_[True, v[1:] != v[:-1]]
    return t[keep], v[keep]


def load_day(day):
    F = {}
    hs = hours(day)
    # Binance USD-M bookTicker mid
    k, m = [], []
    for h in hs:
        p = f'{io.BN_DIR}/{h}.csv.gz'
        if os.path.exists(p):
            x = pd.read_csv(p, header=None, usecols=[1, 4, 6]).apply(pd.to_numeric, errors='coerce').dropna(); x = x[(x[4] > 0) & (x[6] > x[4])]
            kk, mm = series(x[1].values, np.log((x[4].values + x[6].values) / 2.0)); del x       # keep mid changes per hour (memory)
            k.append(kk); m.append(mm)
    F['bnf'] = series(np.concatenate(k), np.concatenate(m))
    # Binance USD-M trades: recv_ns,E,T,trade_id,price,qty,is_buyer_maker -> signed aggressor qty per ms
    k, q = [], []
    for h in hs:
        p = f'{BT}/{h}.csv.gz'
        if os.path.exists(p):
            x = pd.read_csv(p, header=None, usecols=[2, 5, 6]).apply(pd.to_numeric, errors='coerce').dropna()
            k.append(x[2].values); q.append(np.where(x[6].values == 1, -1.0, 1.0) * x[5].values)
    F['bnf_tr'] = flow(np.concatenate(k), np.concatenate(q))
    # Hyperliquid bbo mid and trades
    k, m = [], []
    for h in hs:
        p = f'{HL}/bbo/BTC/{h}.csv.gz'
        if os.path.exists(p):
            x = pd.read_csv(p, header=None, usecols=[1, 2, 4]).apply(pd.to_numeric, errors='coerce').dropna(); x = x[(x[2] > 0) & (x[4] > x[2])]
            k.append(x[1].values); m.append(np.log((x[2].values + x[4].values) / 2.0))
    F['hl'] = series(np.concatenate(k), np.concatenate(m))
    k, px, sz, sd = [], [], [], []
    for h in hs:
        p = f'{HL}/trades/BTC/{h}.csv.gz'
        if os.path.exists(p):
            x = pd.read_csv(p, header=None, usecols=[1, 3, 4, 5]).dropna()
            k.append(pd.to_numeric(x[1]).values); px.append(pd.to_numeric(x[3]).values); sz.append(pd.to_numeric(x[4]).values); sd.append(x[5].astype(str).values)
    k = np.concatenate(k).astype(float); px = np.concatenate(px); sz = np.concatenate(sz); sd = np.concatenate(sd)
    F['hl_side_check'] = hl_side_check(F['hl'], k, px, sd)
    F['hl_tr'] = flow(k, np.where(sd == 'B', 1.0, -1.0) * sz)
    # Binance spot aggTrades: agg_id,price,qty,first,last,transact_time(us),is_buyer_maker,best_match; this day and the previous (lookback)
    k, pr, q = [], [], []
    for dd in (prev_day(day), day):
        if dd is None or dd in SKIP or dd > LAST:
            continue
        p = f"{SPOT}/BTCUSDT-aggTrades-{dd[:4]}-{dd[4:6]}-{dd[6:]}.zip"
        if not os.path.exists(p):
            continue
        with zipfile.ZipFile(p) as z:
            with z.open(z.namelist()[0]) as fh:
                x = pd.read_csv(fh, header=None, usecols=[1, 2, 5, 6])
        t = x[5].values.astype(np.float64); t = t / 1000.0 if t.max() > 1e14 else t      # microseconds since 2025 -> ms
        if dd != day:
            keep = t >= day_ms(day) - 3600e3; x = x[keep]; t = t[keep]
        k.append(t); pr.append(np.log(x[1].values.astype(float))); q.append(np.where(x[6].astype(str).str.lower().values == 'true', -1.0, 1.0) * x[2].values.astype(float))
    k = np.concatenate(k); F['spot'] = series(k, np.concatenate(pr)); F['spot_tr'] = flow(k, np.concatenate(q))
    # Predict Chainlink relay (receipt time)
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


def prev_day(day):
    i = DAYS.index(day)
    d0 = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime('%Y%m%d')
    return d0 if (i > 0 and DAYS[i - 1] == d0) else None


def day_ms(day):
    return pd.Timestamp(day, tz='UTC').value / 1e6


def flow(t, q):
    o = np.argsort(t, kind='stable'); t, q = t[o].astype(np.float64), q[o]
    u, idx = np.unique(t, return_index=True); s = np.add.reduceat(q, idx)
    return u, np.r_[0.0, np.cumsum(s)]


def hl_side_check(hlmid, t, px, sd):
    """Share of 'B' prints at or above the prevailing HL mid (buyer aggressor) and of 'A' prints at or below it."""
    K, LM = hlmid; i = np.searchsorted(K, t, side='left') - 1; ok = i >= 0; mid = np.exp(LM[np.clip(i, 0, None)])
    b = ok & (sd == 'B'); a = ok & (sd == 'A')
    return dict(B_at_or_above_mid=float(np.mean(px[b] >= mid[b])), A_at_or_below_mid=float(np.mean(px[a] <= mid[a])), n=int(ok.sum()))


def move(series_, t, w, dd):
    K, LM = series_; i1 = np.searchsorted(K, t, side='right') - 1; i0 = np.searchsorted(K, t - w, side='right') - 1
    ok = (i0 >= 0) & (i1 >= 0)
    return np.where(ok, dd * (LM[np.clip(i1, 0, None)] - LM[np.clip(i0, 0, None)]) * 1e4, np.nan)


def vol(fl, t, w, dd):
    U, C = fl; a = np.searchsorted(U, t - w, side='right'); b = np.searchsorted(U, t, side='right')
    ok = (len(U) > 0) & (U[0] <= t - w) if len(U) else np.zeros(len(t), bool)
    return np.where(ok, dd * (C[b] - C[a]), np.nan)


def since_cross(series_, t, dd, X=0.5):
    K, LM = series_; out = np.full(len(t), np.nan)
    i1 = np.searchsorted(K, t, side='right') - 1; i0 = np.maximum(np.searchsorted(K, t - 1000.0, side='right') - 1, 0)
    for n in range(len(t)):
        b = i1[n]
        if b < 0:
            continue
        seg = LM[i0[n]:b + 1]; idx = np.flatnonzero(dd[n] * (seg[-1] - seg) >= X * 1e-4 - 1e-12)
        if len(idx):
            out[n] = t[n] - K[i0[n] + idx[-1] + 1]
    return out


SW = pd.read_parquet(f'{S4}/sweep_precursor/sweeps.parquet'); OU = pd.read_parquet(f'{S4}/sweep_precursor/ours.parquet')
MIDF = ('bnf', 'hl', 'spot', 'cl'); FLF = ('bnf_tr', 'hl_tr', 'spot_tr')
rows, side_checks = [], {}
for d in RUN:
    F = load_day(d); side_checks[d] = F['hl_side_check']
    reg = pd.read_parquet(f'{io.OUT}/step1/{d}/registry.parquet', columns=['ws'])
    wsx = reg.ws.values.astype(float) * 1000.0
    ct = (wsx[:, None] + RNG.uniform(0, 240000.0, (len(wsx), 10))).ravel(); cdd = RNG.choice([-1, 1], len(ct))
    s = SW[SW.day == d]; o = OU[(OU.day == d) & np.isfinite(OU.net)]
    E = pd.concat([pd.DataFrame(dict(kind='sweep', cls=np.where(s.informed <= -2, 'bad', np.where(s.informed >= 0, 'good', 'mid')), t=s.ta.values, dd=s.dd.values,
                                     q=s.q.values, Q500=s.Q500.values, Q75=s.Q75.values, V=s.V.values)),
                   pd.DataFrame(dict(kind=o.grp.values, cls=np.where(o.net <= -2, 'bad', 'rest'), t=o.ta.values, dd=o.dd.values, q=o.q.values,
                                     Q500=o.Q500.values, Q75=o.Q75.values, V=o.V.values)),
                   pd.DataFrame(dict(kind='control', cls='control', t=ct, dd=cdd, q=1.0, Q500=np.nan, Q75=np.nan, V=np.nan))], ignore_index=True)
    t = E.t.values.astype(float); dd = E.dd.values.astype(float)
    for f in MIDF:
        for w in WS_:
            E[f'{f}_mv{w:g}'] = move(F[f], t, w, dd)
        E[f'{f}_mv250_late'] = move(F[f], t - LATE, 250.0, dd)
        E[f'{f}_lag'] = since_cross(F[f], t, dd.astype(int))
    for f in FLF:
        for w in WS_:
            E[f'{f}_v{w:g}'] = vol(F[f], t, w, dd)
        E[f'{f}_v250_late'] = vol(F[f], t - LATE, 250.0, dd)
    E['day'] = d; rows.append(E); del F
    io.mem_check(f'day {d}')
A = pd.concat(rows, ignore_index=True); A.to_parquet(f'{OUTD}/events.parquet', index=False)
SIG = [f'{f}_mv250' for f in MIDF] + [f'{f}_v250' for f in FLF]
C = A[A.kind == 'control']; thr = {c: float(np.nanpercentile(C[c], 90)) for c in SIG}
thr.update({c + '_late': thr[c] for c in SIG})
print('Hyperliquid side check (share of prints on the expected side of the prevailing mid):', {d[4:]: (round(v['B_at_or_above_mid'], 3), round(v['A_at_or_below_mid'], 3)) for d, v in side_checks.items()})
print('NaN statuses (no data in the lookback) per signal:', {c: int(A[c].isna().sum()) for c in SIG})
bn_none = A.bnf_lag.isna()
groups = [('controls', A.kind == 'control'), ('market sweeps: bad', (A.kind == 'sweep') & (A.cls == 'bad')), ('market sweeps: good', (A.kind == 'sweep') & (A.cls == 'good')),
          ('bad, no Binance-futures 0.5 bp move', (A.kind == 'sweep') & (A.cls == 'bad') & bn_none),
          ('ours ladder: bad', (A.kind == 'ours: ladder') & (A.cls == 'bad')), ('ours ladder: rest', (A.kind == 'ours: ladder') & (A.cls == 'rest')),
          ('ours single: bad', (A.kind == 'ours: single quote') & (A.cls == 'bad')), ('ours single: rest', (A.kind == 'ours: single quote') & (A.cls == 'rest'))]
lab = {'bnf_mv250': 'BN perp mid', 'hl_mv250': 'HL perp mid', 'spot_mv250': 'BN spot px', 'cl_mv250': 'Chainlink', 'bnf_tr_v250': 'BN perp flow', 'hl_tr_v250': 'HL flow', 'spot_tr_v250': 'BN spot flow'}
out = {}
for late in (False, True):
    print(f"\nshare with an adverse signal above the controls' 90th percentile, 250 ms window ending at {'t - 25.5 ms (in time for our cancel)' if late else 't'}")
    print(f"{'group':38s} {'n':>7s} " + ' '.join(f'{lab[c]:>12s}' for c in SIG) + f" {'any':>6s}")
    for g, m in groups:
        x = A[m]; cols = [c + ('_late' if late else '') for c in SIG]
        pres = np.column_stack([x[c].values > thr[c] for c in cols]); anyp = pres.any(1)
        r = {lab[c]: float(np.average(pres[:, i], weights=x.q)) for i, c in enumerate(SIG)}; r['any'] = float(np.average(anyp, weights=x.q))
        out[f"{g}|{'late' if late else 'at t'}"] = r
        print(f"{g:38s} {len(x):7d} " + ' '.join(f'{r[lab[c]]:12.1%}' for c in SIG) + f" {r['any']:6.1%}")
bad = A[(A.kind == 'sweep') & (A.cls == 'bad')]
print('\nbad market sweeps: lag since a 0.5 bp adverse crossing in each mid feed (share none / within 25.5 ms / earlier; median lag of those that crossed)')
for f, nm in (('bnf', 'BN perp'), ('hl', 'HL perp'), ('spot', 'BN spot'), ('cl', 'Chainlink')):
    l_ = bad[f'{f}_lag']; w = bad.q
    print(f"  {nm:10s} none {np.average(l_.isna(), weights=w):5.1%}  <= 25.5 ms {np.average(l_.fillna(1e9) <= LATE, weights=w):5.1%}  earlier {np.average(l_.fillna(-1) > LATE, weights=w):5.1%}  median {l_.median():6.0f} ms")
both = bad[bad.bnf_lag.notna() & bad.hl_lag.notna()]; both2 = bad[bad.bnf_lag.notna() & bad.spot_lag.notna()]
print(f"  where both crossed: HL before BN perp in {np.mean(both.hl_lag > both.bnf_lag):.0%} (n {len(both)}, median lead {np.median(both.hl_lag - both.bnf_lag):+.0f} ms); "
      f"spot before BN perp in {np.mean(both2.spot_lag > both2.bnf_lag):.0%} (n {len(both2)}, median lead {np.median(both2.spot_lag - both2.bnf_lag):+.0f} ms)")
sw = A[A.kind == 'sweep'].copy(); pulled = np.maximum(sw.Q500 - sw.Q75 - sw.V, 0) / sw.Q500.where(sw.Q500 > 0); sw['p30'] = pulled >= 0.3
cols = [c + '_late' for c in SIG]; sw['anyx'] = np.column_stack([sw[c].values > thr[c] for c in cols if not c.startswith('bnf')]).any(1)
sw['bnx'] = np.column_stack([sw[c].values > thr[c] for c in ('bnf_mv250_late', 'bnf_tr_v250_late')]).any(1)
print('\nthe market\'s makers: share of sweeps where the at-risk size was pulled >= 30% by t - 75 ms, by signal (in time for us)')
for nm, m in (('no signal in any feed', ~sw.anyx & ~sw.bnx), ('Binance perp signal only', sw.bnx & ~sw.anyx), ('HL / spot / Chainlink signal, no Binance perp', sw.anyx & ~sw.bnx), ('both', sw.anyx & sw.bnx)):
    x = sw[m & pulled.notna()]
    print(f"  {nm:46s} n {len(x):6d}  pulled >= 30%: {np.average(x.p30, weights=x.q):5.1%}  bad share {np.average(x.cls == 'bad', weights=x.q):5.1%}")
io.write_json(f'{OUTD}/report.json', dict(thresholds=thr, hl_side_check=side_checks, presence=out))
