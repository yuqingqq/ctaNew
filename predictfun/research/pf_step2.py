"""Step 2 (exploratory): do D1's Binance-driven signals carry over to Predict?

All series on one receipt clock (this host): Predict book frames (recv_ns) and the
Binance USD-M futures bookTicker (recv_ns), plus the Predict reference relay.
Grid: 100 ms receipt-time points over each window's live span [ws, ws+300).

A. BTC guard premise: Binance 1 s (and 0.25 s) return buckets -> Predict YES mid change
   over +0.25/1/5/30 s (cents), window-clustered.
B. Pass-through: slope of the mid change on the theoretical change delta * r1.
C. Impulse response after a fresh >= 1 bps / 1 s Binance move: cumulative mid change / theory.
D. Tick leg re-derived for Predict: p_bn = Phi(ln(S*basis/K) / (sigma*sqrt(300 - t))), K = start
   price, basis = trailing-60 s median(relay / futures mid). Calibration vs outcome and
   whether (p_bn - mid) predicts the next mid change.

Usage: python3 -m predictfun.research.pf_step2 20261023
Needs step1/<day>/ (books, registry). Output: step2/<day>/summary.json (+ grid.parquet).
"""
import sys, os, math, collections, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np, pandas as pd
from scipy.special import ndtr
from . import pf_io as io

day = sys.argv[1]
CLOCK = sys.argv[2] if len(sys.argv) > 2 else 'recv'     # recv: this host's receipt; tokyo20: Predict t_ms+20, Binance E+5
S1 = f'{io.OUT}/step1/{day}'
OUT = f'{io.OUT}/step2/{day}' + ('' if CLOCK == 'recv' else f'_{CLOCK}')
os.makedirs(OUT, exist_ok=True)
GRID_NS = 100_000_000
H = [0.25, 1.0, 5.0, 30.0]

reg = pd.read_parquet(f'{S1}/registry.parquet').set_index('mid')

# ---- Binance futures mid on a 100 ms receipt grid (last row per bucket), day +-1 h ----
parts = []
for h in io.hours_of(day, extra_after=1, extra_before=1):
    p = f'{io.BN_DIR}/{h}.csv.gz'
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p, header=None, names=['recv', 'E', 'T', 'u', 'bid', 'bq', 'ask', 'aq'], usecols=['recv', 'E', 'bid', 'ask'],
                     engine='c', on_bad_lines='skip')
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df['recv'] = df['recv'].astype('int64') if CLOCK == 'recv' else ((df['E'] + 5) * 1_000_000).astype('int64')
    df = df[(df.bid > 0) & (df.ask > df.bid)]
    df['b'] = df.recv // GRID_NS
    g = df.groupby('b', sort=True).agg(mid=('bid', 'last'), ask=('ask', 'last'))
    g['mid'] = (g['mid'] + g['ask']) / 2
    parts.append(g['mid'])
    io.mem_check(f'bn {h}')
bn = pd.concat(parts).sort_index()
bn = bn[~bn.index.duplicated(keep='last')]
bn_b = bn.index.values.astype(np.int64); bn_m = bn.values
print(f'[bn] buckets {len(bn_b)} [mem] {io.rss_mib()} MiB', flush=True)

def bn_at(bucket, max_age=10):
    """Last Binance mid at or before each grid bucket; NaN if older than max_age buckets."""
    i = np.searchsorted(bn_b, bucket, side='right') - 1
    ok = i >= 0
    out = np.full(len(bucket), np.nan)
    out[ok] = bn_m[i[ok]]
    age = np.where(ok, bucket - bn_b[np.clip(i, 0, None)], 10 ** 9)
    out[age > max_age] = np.nan
    return out

# ---- relay prints (receipt time) ----
rr = []
for h in io.hours_of(day, extra_after=1, extra_before=1):
    for recv, pub, price, dup in io.ref_rows(h):
        rr.append((recv, pub, float(price)))
rel = pd.DataFrame(rr, columns=['recv', 'pub', 'price']).sort_values('recv')
rel_b = (rel.recv.values // GRID_NS).astype(np.int64)
rel_ratio = rel.price.values / bn_at(rel_b, max_age=10)

# ---- Predict books ----
bf = sorted(os.listdir(f'{S1}/books'))
books = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=['mid', 'recv_ns', 't_ms', 'b1_p', 'b1_s', 'a1_p', 'a1_s']) for f in bf], ignore_index=True)
if CLOCK == 'tokyo20':
    books['recv_ns'] = ((books.t_ms + 20) * 1_000_000).astype('int64')
books.sort_values(['mid', 'recv_ns'], inplace=True, kind='stable')
print(f'[books] {len(books)} [mem] {io.rss_mib()} MiB', flush=True)

rows = []
stat = collections.Counter()
for mid, g in books.groupby('mid', sort=False):
    if mid not in reg.index:
        continue
    R = reg.loc[mid]; ws = int(R.ws)
    if R.start_price is None or (isinstance(R.start_price, float) and math.isnan(R.start_price)):
        stat['no_start_price'] += 1
        continue
    K = float(R.start_price)
    y = 1.0 if R.outcome == 'Up' else (0.0 if R.outcome == 'Down' else np.nan)
    grid = (ws * 1_000_000_000 + np.arange(0, 3000 + 300) * GRID_NS)   # live + 30 s for targets
    gb = grid // GRID_NS
    rv = g.recv_ns.values
    i = np.searchsorted(rv, grid, side='right') - 1
    okf = i >= 0
    ii = np.clip(i, 0, None)
    b1 = g.b1_p.values[ii]; a1 = g.a1_p.values[ii]; bs = g.b1_s.values[ii]; as_ = g.a1_s.values[ii]
    age_s = np.where(okf, (grid - rv[ii]) / 1e9, np.inf)
    valid = okf & (b1 >= 0) & (a1 > b1) & (age_s <= 5.0)
    midc = np.where(valid, (a1 + b1) / 2.0, np.nan)                     # cents
    micro = np.where(valid, (b1 * as_ + a1 * bs) / np.maximum(bs + as_, 1e-12), np.nan)
    spread = np.where(valid, a1 - b1, np.nan)
    S = bn_at(gb)
    S_1 = bn_at(gb - 10); S_q = bn_at(gb - 3)                           # 1 s / 0.3 s back (0.25 s ~ 3 buckets)
    r1 = 1e4 * np.log(S / S_1); rq = 1e4 * np.log(S / S_q)
    # sigma per sqrt(second) from 1 s returns over [ws-1800, ws)
    sb = (ws - 1800) * 10 + np.arange(0, 1800) * 10
    sp = bn_at(sb.astype(np.int64))
    lr = np.diff(np.log(sp)); lr = lr[np.isfinite(lr)]
    sigma = float(np.sqrt(np.mean(lr ** 2))) if len(lr) >= 1000 else np.nan
    # basis: trailing 60 s median of relay/futures ratio, known by the grid point
    j1 = np.searchsorted(rel_b, gb, side='right')
    j0 = np.searchsorted(rel_b, gb - 600, side='left')
    basis = np.array([np.nanmedian(rel_ratio[a:b]) if b - a >= 5 else np.nan for a, b in zip(j0[::10], j1[::10])])
    basis = np.repeat(basis, 10)[:len(grid)]
    t = (grid / 1e9) - ws
    tau = np.maximum(300.0 - t, 1e-6)
    z = np.log(S * basis / K) / (sigma * np.sqrt(tau))
    p_bn = ndtr(z) * 100.0                                               # cents
    delta_per_bps = (np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)) / (sigma * np.sqrt(tau)) * 1e-4 * 100.0   # cents per 1 bps
    n = 3000
    rec = dict(mid=np.full(n, mid), ws=np.full(n, ws), t=t[:n], midc=midc[:n], micro=micro[:n], spread=spread[:n],
               r1=r1[:n], rq=rq[:n], p_bn=p_bn[:n], delta=delta_per_bps[:n], sigma=np.full(n, sigma), basis_bps=1e4 * (basis[:n] - 1),
               y=np.full(n, y))
    for h in H:
        k = int(round(h * 10))
        rec[f'd{h:g}'] = midc[k:k + n] - midc[:n]
    rows.append(pd.DataFrame(rec))
    io.mem_check(f'grid {mid}')
G = pd.concat(rows, ignore_index=True)
del rows, books
G.to_parquet(f'{OUT}/grid.parquet', index=False)
print(f'[grid] rows {len(G)} windows {G.mid.nunique()} [mem] {io.rss_mib()} MiB', flush=True)

def clustered_mean(df, col):
    """Mean of col with a window-clustered SE (sum by window)."""
    d = df[['mid', col]].dropna()
    if len(d) == 0:
        return dict(n=0)
    s = d.groupby('mid')[col].agg(['sum', 'count'])
    m = s['sum'].sum() / s['count'].sum()
    resid = s['sum'] - m * s['count']
    se = math.sqrt((resid ** 2).sum()) / s['count'].sum() if len(s) > 1 else float('nan')
    return dict(n=int(s['count'].sum()), windows=int(len(s)), mean=round(float(m), 4), se=round(float(se), 4))

def slope(df, x, ycol):
    d = df[['mid', x, ycol]].dropna()
    if len(d) < 100:
        return dict(n=int(len(d)))
    X = d[x].values; Y = d[ycol].values
    b = (X * Y).sum() / (X * X).sum()
    e = Y - b * X
    gs = pd.DataFrame({'mid': d.mid.values, 'xe': X * e}).groupby('mid').xe.sum()
    se = math.sqrt((gs ** 2).sum()) / (X * X).sum()
    return dict(n=int(len(d)), windows=int(d.mid.nunique()), slope=round(float(b), 4), se=round(float(se), 4))

live = G[(G.t >= 0) & (G.t < 240)]
A = {}
for lab, col in (('r1', 'r1'), ('r025', 'rq')):
    for name, m in (('down_le_-0.5', live[col] <= -0.5), ('flat', live[col].abs() < 0.5), ('up_ge_0.5', live[col] >= 0.5)):
        sub = live[m]
        A[f'{lab}:{name}'] = dict(share_of_time=round(float(m.mean()), 4), **{f'd{h:g}': clustered_mean(sub, f'd{h:g}') for h in H})
B = {}
live = live.assign(theory=live.delta * live.r1)
for h in H:
    B[f'd{h:g}_on_delta_x_r1'] = slope(live, 'theory', f'd{h:g}')
# C: impulse response to fresh 1 s moves >= 1 bps (first grid point of an episode)
ev = []
for mid, g in G.groupby('mid', sort=False):
    r = g.r1.values; tt = g.t.values; mc = g.midc.values; de = g.delta.values
    big = np.abs(r) >= 1.0
    start = np.where(big & ~np.r_[False, big[:-1]] & (tt >= 1) & (tt < 230))[0]
    for s in start:
        if s + 100 >= len(mc) or not np.isfinite(mc[s]):
            continue
        th = de[s] * r[s]
        if not np.isfinite(th) or abs(th) < 0.25:
            continue
        ev.append([np.sign(r[s])] + [(mc[s + k] - mc[s]) / th if np.isfinite(mc[s + k]) else np.nan for k in (1, 3, 5, 10, 20, 50, 100)])
EV = pd.DataFrame(ev, columns=['sign', 'l0.1', 'l0.3', 'l0.5', 'l1', 'l2', 'l5', 'l10'])
C = dict(n_events=int(len(EV)), median_ratio=EV.drop(columns='sign').median().round(3).to_dict(), mean_ratio=EV.drop(columns='sign').mean().round(3).to_dict())
# D: calibration + fair-value gap
D = {}
for T in (60, 120, 180, 240, 290):
    s = G[(G.t >= T - 0.05) & (G.t < T + 0.05) & G.y.notna()]
    s = s.dropna(subset=['midc', 'p_bn'])
    D[f'brier_t{T}'] = dict(n=int(len(s)), mid=round(float(((s.midc / 100 - s.y) ** 2).mean()), 4) if len(s) else None,
                            p_bn=round(float(((s.p_bn / 100 - s.y) ** 2).mean()), 4) if len(s) else None,
                            avg=round(float((((s.midc + s.p_bn) / 200 - s.y) ** 2).mean()), 4) if len(s) else None)
live = live.assign(gap=live.p_bn - live.midc)
for h in H:
    D[f'd{h:g}_on_gap'] = slope(live, 'gap', f'd{h:g}')
D['gap_abs_median_c'] = round(float(live.gap.abs().median()), 3)
D['basis_bps'] = dict(zip(['p05', 'p50', 'p95'], np.nanpercentile(G.basis_bps, [5, 50, 95]).round(2).tolist()))
D['sigma_per_sqrt_s_bps'] = dict(zip(['p05', 'p50', 'p95'], (1e4 * np.nanpercentile(G.groupby('mid').sigma.first(), [5, 50, 95])).round(3).tolist()))
summary = dict(day=day, clock=CLOCK, mode='EXPLORATORY', n_windows=int(G.mid.nunique()), statuses=dict(stat),
               grid_valid_share_live=round(float(G[(G.t >= 0) & (G.t < 300)].midc.notna().mean()), 4),
               A_guard_buckets=A, B_passthrough=B, C_impulse=C, D_tick_leg=D, mem_mib=io.rss_mib())
io.write_json(f'{OUT}/summary.json', summary)
print('[done]', day, 'windows', summary['n_windows'])
