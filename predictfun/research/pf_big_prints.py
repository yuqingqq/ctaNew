"""Do big trades on other exchanges come with our bad fills? (2026-11-04; user: "check if there are any big fill happens from
other exchanges", then "we pull data from more exchanges"; part C of the check declared 2026-11-04 01:57Z, added 02:06Z, venues
extended 02:19Z - see STATUS). Descriptive, 8 read days.

Venues (BTC trades, taker side; sizes in BTC): Binance USD-M perp (our trade stream, T), Binance spot (Vision aggTrades),
Hyperliquid perp (our collector), Bybit perp and spot (public.bybit.com, UTC days), OKX perp and spot (static.okx.com; each file
runs 16:00-16:00 UTC; perp size x 0.01 BTC a contract), Coinbase spot, Kraken spot, Deribit perp (public APIs via
data_collectors/fetch_public_trades.py). Side convention checked per venue: share of taker buys on an uptick and of taker sells
on a downtick (trades whose price differs from the previous one); both well above 50% if the side is read right.
Big order: all trades of one venue with the same millisecond and taker side summed; big = at or above the day's 99th
percentile of those sizes on that venue (99.9th in the JSON). Adverse = taker side that hurts the swept makers (bid makers:
sellers); the same check in the favourable direction is reported beside it.
Events (Predict venue time t): our replay's fills of P60 with the real model (anchor test runs, tag anc; bad <= -2 c/sh at 1 s,
good >= 0), market sweeps over 2+ prices (step4/sweep_precursor, bad / good by informed at 1 s) and control times (10 per live
window, random direction). Windows: well before [t-1000, t-250), just before [t-250, t-25.5) (in time for our cancel), same
moment [t-25.5, t+25], just after (t+25, t+250]. Exchange clocks vs Predict venue time; under 10 ms not resolved.
Coverage: an event counts for a venue only if [t-1000, t+250] lies inside that venue's loaded data (else NaN, a status).
Data inside 10-28..30 and after 11-02 is never loaded.
Usage: python3 -m predictfun.research.pf_big_prints
"""
import os, zipfile
import numpy as np, pandas as pd
from . import pf_io as io
from . import pf_venue_feeds as VF

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; FRESH = ('20261101', '20261102')
S4 = f'{io.OUT}/step4'; OUTD = f'{S4}/big_prints'; os.makedirs(OUTD, exist_ok=True)
BT = io.BN_DIR.replace('bookTicker', 'trade'); EX = f'{io.ROOT}/data/exchange_public'; RNG = np.random.default_rng(20261104)
WIN = [('well before', -1000.0, -250.0), ('just before', -250.0, -25.5), ('same moment', -25.5, 25.0), ('just after', 25.0, 250.0)]
P60 = 'MICRO+G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1+S60/mid@A1:5:25:0'
VENUES = ['Binance perp', 'Binance spot', 'Hyperliquid', 'Bybit perp', 'Bybit spot', 'OKX perp', 'OKX spot', 'Coinbase', 'Kraken', 'Deribit']
stem = lambda n: n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')
H = 3600e3; D_ = 86400e3


def dms(d):
    return pd.Timestamp(d, tz='UTC').value / 1e6


def iso(d):
    return f'{d[:4]}-{d[4:6]}-{d[6:]}'


def prev_in(d):
    p = (pd.Timestamp(d) - pd.Timedelta(days=1)).strftime('%Y%m%d')
    return p if p in DAYS else None


def side_check(t, p, s):
    o = np.lexsort((np.arange(len(t)), t)); p = p[o]; s = s[o]; dp = np.r_[0.0, np.diff(p)]; ch = dp != 0
    b = ch & (s > 0); a = ch & (s < 0)
    return dict(buy_on_uptick=float(np.mean(dp[b] > 0)) if b.any() else np.nan, sell_on_downtick=float(np.mean(dp[a] < 0)) if a.any() else np.nan)


def pack(t, p, q, s, cover):
    """Trades -> (order times, signed sizes, coverage intervals, side check)."""
    t = np.asarray(t, float); p = np.asarray(p, float); q = np.abs(np.asarray(q, float)); s = np.asarray(s, float)
    sc = side_check(t, p, s)
    df = pd.DataFrame(dict(t=np.floor(t), s=s, q=q)).groupby(['t', 's'], sort=True).q.sum().reset_index()
    return df.t.values.astype(float), (df.s * df.q).values, cover, sc


def daily_files(d, path_of, read):
    """UTC-day files: this day and the last hour of the previous (if it is a read day). Returns trades and coverage."""
    parts, cov = [], []
    for dd, lo in ((prev_in(d), dms(d) - H), (d, dms(d))):
        if dd is None or not os.path.exists(path_of(dd)):
            continue
        x = read(path_of(dd)); x = x[(x[0] >= lo) & (x[0] < dms(dd) + D_)]; parts.append(x); cov.append((lo, dms(dd) + D_))
    x = pd.concat(parts) if parts else pd.DataFrame({0: [], 1: [], 2: [], 3: []})
    return x, cov


def merge(cov):
    cov = sorted(cov); out = []
    for a, b in cov:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def rd_binance_spot(p):
    with zipfile.ZipFile(p) as z:
        with z.open(z.namelist()[0]) as fh:
            x = pd.read_csv(fh, header=None, usecols=[1, 2, 5, 6])
    t = x[5].values.astype(float); t = t / 1000.0 if t.max() > 1e14 else t
    return pd.DataFrame({0: t, 1: x[1].values.astype(float), 2: x[2].values.astype(float), 3: np.where(x[6].astype(str).str.lower().values == 'true', -1.0, 1.0)})


def rd_bybit_perp(p):
    x = pd.read_csv(p, usecols=['timestamp', 'side', 'size', 'price'])
    return pd.DataFrame({0: x.timestamp.values * 1000.0, 1: x.price.values, 2: x['size'].values, 3: np.where(x.side.values == 'Buy', 1.0, -1.0)})


def rd_bybit_spot(p):
    x = pd.read_csv(p, usecols=['timestamp', 'price', 'volume', 'side'])
    return pd.DataFrame({0: x.timestamp.values.astype(float), 1: x.price.values, 2: x.volume.values, 3: np.where(x.side.str.lower().values == 'buy', 1.0, -1.0)})


def rd_rest(p):
    x = pd.read_csv(p).sort_values(['t_ms', 'id'], kind='stable')      # the fetcher wrote rows sorted by (time, price): restore sequence order
    return pd.DataFrame({0: x.t_ms.values, 1: x.price.values.astype(float), 2: x.size_btc.values, 3: x.taker.values.astype(float)})


def rd_okx(p, mult):
    with zipfile.ZipFile(p) as z:
        with z.open(z.namelist()[0]) as fh:
            x = pd.read_csv(fh, usecols=['side', 'price', 'size', 'created_time'], encoding_errors='replace')
    return pd.DataFrame({0: x.created_time.values.astype(float), 1: x.price.values, 2: x['size'].values * mult, 3: np.where(x.side.values == 'buy', 1.0, -1.0)})


def okx(d, sub, name, mult):
    """OKX files run [X-1 16:00, X 16:00) UTC; UTC day d needs files d and d+1. Each file is cut at once to [lo, hi): the day,
    plus the hour before only when the previous day is a read day, so no row of 10-28..30 or 11-03 is kept."""
    lo = dms(d) - H if prev_in(d) else dms(d); hi = dms(d) + D_
    parts, cov = [], []
    for f in (d, (pd.Timestamp(d) + pd.Timedelta(days=1)).strftime('%Y%m%d')):
        p = f'{EX}/{sub}/{name}-{iso(f)}.zip'
        if os.path.exists(p):
            x = rd_okx(p, mult); x = x[(x[0] >= lo) & (x[0] < hi)]; parts.append(x)
            a, b = max(dms(f) - 8 * H, lo), min(dms(f) + 16 * H, hi)
            if b > a:
                cov.append((a, b))
    x = pd.concat(parts) if parts else pd.DataFrame({0: [], 1: [], 2: [], 3: []})
    return x, cov


def load_orders(d):
    O = {}
    k, p, q, s, cov = [], [], [], [], []
    for h in VF.hours(d):
        f = f'{BT}/{h}.csv.gz'
        if os.path.exists(f):
            x = pd.read_csv(f, header=None, usecols=[2, 4, 5, 6]).apply(pd.to_numeric, errors='coerce').dropna()
            k.append(x[2].values); p.append(x[4].values); q.append(x[5].values); s.append(np.where(x[6].values == 1, -1.0, 1.0))
            h0 = pd.to_datetime(h, format='%Y%m%d_%H', utc=True).value / 1e6; cov.append((h0, h0 + H))
    O['Binance perp'] = pack(np.concatenate(k), np.concatenate(p), np.concatenate(q), np.concatenate(s), merge(cov))
    x, cov = daily_files(d, lambda dd: f'{VF.SPOT}/BTCUSDT-aggTrades-{iso(dd)}.zip', rd_binance_spot); O['Binance spot'] = pack(x[0], x[1], x[2], x[3], merge(cov))
    k, p, q, s, cov = [], [], [], [], []
    for h in VF.hours(d):
        f = f'{VF.HL}/trades/BTC/{h}.csv.gz'
        if os.path.exists(f):
            x = pd.read_csv(f, header=None, usecols=[1, 3, 4, 5]).dropna()
            k.append(pd.to_numeric(x[1]).values.astype(float)); p.append(pd.to_numeric(x[3]).values); q.append(pd.to_numeric(x[4]).values)
            s.append(np.where(x[5].astype(str).values == 'B', 1.0, -1.0))
            h0 = pd.to_datetime(h, format='%Y%m%d_%H', utc=True).value / 1e6; cov.append((h0, h0 + H))
    O['Hyperliquid'] = pack(np.concatenate(k), np.concatenate(p), np.concatenate(q), np.concatenate(s), merge(cov))
    for v, path_of, rd in (('Bybit perp', lambda dd: f'{EX}/bybit_perp/BTCUSDT{iso(dd)}.csv.gz', rd_bybit_perp),
                           ('Bybit spot', lambda dd: f'{EX}/bybit_spot/BTCUSDT_{iso(dd)}.csv.gz', rd_bybit_spot),
                           ('Coinbase', lambda dd: f'{EX}/coinbase/{dd}.csv.gz', rd_rest), ('Kraken', lambda dd: f'{EX}/kraken/{dd}.csv.gz', rd_rest),
                           ('Deribit', lambda dd: f'{EX}/deribit/{dd}.csv.gz', rd_rest)):
        x, cov = daily_files(d, path_of, rd); O[v] = pack(x[0], x[1], x[2], x[3], merge(cov))
    for v, sub, name, mult in (('OKX perp', 'okx_swap', 'BTC-USDT-SWAP-trades', 0.01), ('OKX spot', 'okx_spot', 'BTC-USDT-trades', 1.0)):
        x, cov = okx(d, sub, name, mult); O[v] = pack(x[0], x[1], x[2], x[3], merge(cov))
    return O


def covered(cov, t):
    ok = np.zeros(len(t), bool)
    for a, b in cov:
        ok |= (t - 1000.0 >= a) & (t + 250.0 < b)
    return ok


def big_in(T, Q, thr, t, dd, a, b):
    """Per event: an order of size >= thr with taker sign == dd within [t + a, t + b)?"""
    big = np.abs(Q) >= thr; Tb = T[big]; Sb = np.sign(Q[big]); out = np.zeros(len(t), bool)
    for sgn in (-1.0, 1.0):
        Ts = Tb[Sb == sgn]; m = dd == sgn
        out[m] = np.searchsorted(Ts, t[m] + b, side='left') > np.searchsorted(Ts, t[m] + a, side='left')
    return out


SW = pd.read_parquet(f'{S4}/sweep_precursor/sweeps.parquet'); rows, thr_rep, checks = [], {}, {}
RUN = os.environ.get('PF_BP_DAYS', ','.join(DAYS)).split(',')
if 'PF_BP_DAYS' in os.environ:
    OUTD = OUTD + '_smoke'; os.makedirs(OUTD, exist_ok=True)
for d in RUN:
    O = load_orders(d)
    reg = pd.read_parquet(f'{io.OUT}/step1/{d}/registry.parquet', columns=['mid', 'ws']).set_index('mid')
    f = pd.read_parquet(f"{S4}/live_tokyoreal_custom_e2_{'preds_c3' if d in FRESH else 'preds_o3'}_anc/{d}/fills/{stem(P60)}.parquet", columns=['mid', 't', 'side', 'q', 'mo1'])
    c = 100 * f.mo1 / f.q
    ours = pd.DataFrame(dict(kind='ours', cls=np.where(c <= -2, 'bad', np.where(c >= 0, 'good', 'small loss')), t=reg.loc[f.mid.values].ws.values * 1000.0 + f.t.values * 1000.0,
                             dd=np.where(f.side.values == 0, -1.0, 1.0), q=f.q.values))
    s = SW[SW.day == d]
    mk = pd.DataFrame(dict(kind='market sweep', cls=np.where(s.informed <= -2, 'bad', np.where(s.informed >= 0, 'good', 'mid')), t=s.ta.values, dd=s.dd.values.astype(float), q=s.q.values))
    ws = reg.ws.values.astype(float) * 1000.0; ct = (ws[:, None] + RNG.uniform(0, 240000.0, (len(ws), 10))).ravel()
    ctl = pd.DataFrame(dict(kind='control', cls='control', t=ct, dd=RNG.choice([-1.0, 1.0], len(ct)), q=1.0))
    E = pd.concat([ours, mk, ctl], ignore_index=True); E['day'] = d
    t = E.t.values.astype(float); dd = E.dd.values; cols = {}
    anyb = {wl: np.zeros(len(E), bool) for wl, _, _ in WIN}
    for v in VENUES:
        T, Q, cov, sc = O[v]; checks[f'{d}|{v}'] = dict(sc, orders=int(len(Q)), covered_hours=round(sum(b - a for a, b in cov) / H, 2))
        if not len(Q):
            for wl, _, _ in WIN:
                for nm in ('p99', 'p999'):
                    cols[f'{v}|{wl}|{nm}'] = np.full(len(E), np.nan); cols[f'{v}|{wl}|{nm}|fav'] = np.full(len(E), np.nan)
            continue
        p99, p999 = np.percentile(np.abs(Q), [99, 99.9]); thr_rep[f'{d}|{v}'] = dict(p99_btc=float(p99), p999_btc=float(p999))
        cv = covered(cov, t)
        for wl, a, b in WIN:
            for nm, th in (('p99', p99), ('p999', p999)):
                x = big_in(T, Q, th, t, dd, a, b); xf = big_in(T, Q, th, t, -dd, a, b)
                if nm == 'p99':
                    anyb[wl] |= x & cv
                x = x.astype(float); x[~cv] = np.nan; xf = xf.astype(float); xf[~cv] = np.nan
                cols[f'{v}|{wl}|{nm}'] = x; cols[f'{v}|{wl}|{nm}|fav'] = xf
    for wl, _, _ in WIN:
        cols[f'any venue|{wl}|p99'] = anyb[wl].astype(float)
    E = pd.concat([E, pd.DataFrame(cols, index=E.index)], axis=1)
    rows.append(E); del O
    io.mem_check(f'day {d}')
A = pd.concat(rows, ignore_index=True); A.to_parquet(f'{OUTD}/events.parquet', index=False)
C = pd.DataFrame(checks).T
print('side check (median over days: taker buys on an uptick / taker sells on a downtick) and big-order threshold (p99, BTC):')
for v in VENUES:
    cc = C.loc[[k for k in C.index if k.endswith('|' + v)]]; th = [x['p99_btc'] for k, x in thr_rep.items() if k.endswith('|' + v)]
    print(f"  {v:13s} {cc.buy_on_uptick.median():.1%} / {cc.sell_on_downtick.median():.1%}   p99 {np.median(th) if th else np.nan:7.3f}   orders/day {cc.orders.median():9.0f}   covered h/day {cc.covered_hours.median():5.1f}")
G = [('controls', (A.kind == 'control')), ('ours: good fills', (A.kind == 'ours') & (A.cls == 'good')), ('ours: bad fills', (A.kind == 'ours') & (A.cls == 'bad')),
     ('market sweeps: good', (A.kind == 'market sweep') & (A.cls == 'good')), ('market sweeps: bad', (A.kind == 'market sweep') & (A.cls == 'bad'))]
out = {}
for wl, _, _ in WIN:
    print(f"\nshare of events with an adverse big order (p99) {wl} ({'in time for our cancel' if wl == 'just before' else ''}); favourable rate in brackets")
    print(f"  {'group':22s} {'n':>6s} " + ' '.join(f'{v[:13]:>15s}' for v in VENUES) + f" {'any venue':>10s}")
    for g, m in G:
        x = A[m]; cells = []
        for v in VENUES:
            a_ = np.nanmean(x[f'{v}|{wl}|p99']) if x[f'{v}|{wl}|p99'].notna().any() else np.nan; f_ = np.nanmean(x[f'{v}|{wl}|p99|fav']) if x[f'{v}|{wl}|p99|fav'].notna().any() else np.nan
            out[f'{v}|{g}|{wl}'] = dict(adverse=float(a_), favourable=float(f_)); cells.append(f'{a_:6.1%} ({f_:5.1%})')
        an = float(x[f'any venue|{wl}|p99'].mean()); out[f'any venue|{g}|{wl}'] = an
        print(f"  {g:22s} {len(x):6d} " + ' '.join(f'{c:>15s}' for c in cells) + f" {an:10.1%}")
stat = {v: int(A[f'{v}|just before|p99'].isna().sum()) for v in VENUES}
print('\nstatuses: events outside a venue\'s loaded data (NaN):', stat)
io.write_json(f'{OUTD}/report.json', dict(thresholds=thr_rep, checks=checks, shares=out, statuses=stat))
