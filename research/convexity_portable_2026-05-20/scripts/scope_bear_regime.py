"""
scope_bear_regime.py — EDA (NOT a production backtest) for BEAR-regime action.

Question: in BEAR regime (BTC trailing-30d return < -0.10, PIT), what is the best
K=3 long/short action: (a) mean-rev (V0 pred), (b) trend-follow (30d momentum),
(c) FLAT. Honest read on whether bear is harvestable given only ~1-2 episodes.

Gross PnL only (costs would reduce further). Does NOT modify any existing file.
"""
import numpy as np
import pandas as pd
import glob, os

K = 3
ANN = np.sqrt(6 * 365)  # 6 cycles/day * 365
PANEL = 'outputs/vBTC_features/panel_3yr_v0.parquet'
PREDS = 'research/convexity_portable_2026-05-20/results/_cache/x70_v0_3yr_preds.parquet'
KLINES = 'data/ml/test/parquet/klines/BTCUSDT/5m/*.parquet'


def sharpe(x):
    x = np.asarray(x, float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / x.std(ddof=1) * ANN


# ---- 1. BTC 4h closes -> PIT 30d return regime ----
btc = pd.concat([pd.read_parquet(f, columns=['open_time', 'close']) for f in sorted(glob.glob(KLINES))])
btc['open_time'] = pd.to_datetime(btc['open_time'], utc=True)
btc = btc.sort_values('open_time').drop_duplicates('open_time')
# 4h cycles
btc4 = btc[(btc['open_time'].dt.hour % 4 == 0) & (btc['open_time'].dt.minute == 0)].copy()
btc4 = btc4.set_index('open_time')
btc4['ret_30d'] = btc4['close'] / btc4['close'].shift(180) - 1.0  # 180 4h-bars = 30d
btc4['bear'] = btc4['ret_30d'] < -0.10
bear_times = set(btc4.index[btc4['bear'].fillna(False)])
print(f"BTC 4h cycles: {len(btc4)}, bear cycles: {len(bear_times)}")

# ---- 2. Load preds (V0 mean-rev), align to 4h cycles ----
preds = pd.read_parquet(PREDS, columns=['symbol', 'open_time', 'pred', 'alpha_A', 'return_pct'])
preds = preds[(preds['open_time'].dt.hour % 4 == 0) & (preds['open_time'].dt.minute == 0)].copy()

# ---- 3. mom_30d from panel (PIT trailing 30d sym return) ----
# Use ret_3d? No — need 30d. Build from panel return_pct cumulative is messy; instead use
# panel close-equivalent: panel has return_1d. Build mom_30d from raw 4h fwd return is wrong.
# Cleanest PIT 30d momentum: use the panel's own price proxy. Panel lacks close, so
# reconstruct trailing 30d return per symbol from 4h-aligned return_pct (4h-fwd) shifted.
panel = pd.read_parquet(PANEL, columns=['symbol', 'open_time', 'return_pct'])
panel = panel[(panel['open_time'].dt.hour % 4 == 0) & (panel['open_time'].dt.minute == 0)].copy()
panel = panel.sort_values(['symbol', 'open_time'])
# return_pct is 4h-FWD return for the bar. Trailing 30d return at time t =
# product over the prior 180 bars of (1+fwd_ret) of bars whose fwd window ended <= t.
# Simpler & PIT-safe: cumulative log of return_pct then diff over 180 bars, shifted by 1
# so we only use returns realized strictly before t.
panel['lr'] = np.log1p(panel['return_pct'].clip(-0.5, 2))
g = panel.groupby('symbol', sort=False)
panel['clr'] = g['lr'].cumsum()
# trailing 30d return ending at the PREVIOUS bar (PIT): sum of last 180 fwd-returns that
# completed before t. fwd ret at bar i covers [i, i+1cycle]; completes at i+1. So returns
# realized before t are those with index < t. Use shift(1) on cumulative then 180-lag.
panel['clr_prev'] = g['clr'].shift(1)
panel['mom_30d'] = panel['clr_prev'] - g['clr_prev'].shift(180)
mom = panel[['symbol', 'open_time', 'mom_30d']]

df = preds.merge(mom, on=['symbol', 'open_time'], how='left')
df['bear'] = df['open_time'].isin(bear_times)
bear = df[df['bear']].dropna(subset=['return_pct']).copy()
print(f"Bear rows (sym-cycles): {len(bear)}, unique bear cycles in preds: {bear['open_time'].nunique()}")


def cycle_pnl(grp, sel_col, ascending_short=True, value='return_pct', need=K):
    """long top-K by sel_col, short bottom-K. Returns mean(long)-mean(short)."""
    g = grp.dropna(subset=[sel_col, value])
    if len(g) < 2 * need:
        return np.nan
    s = g.sort_values(sel_col)
    short = s.head(need)[value].mean()
    long = s.tail(need)[value].mean()
    return long - short


def run(sub, label):
    cyc = sub.groupby('open_time')
    mr_ret = cyc.apply(lambda x: cycle_pnl(x, 'pred', value='return_pct'), include_groups=False)
    mr_alp = cyc.apply(lambda x: cycle_pnl(x, 'pred', value='alpha_A'), include_groups=False)
    tf_ret = cyc.apply(lambda x: cycle_pnl(x, 'mom_30d', value='return_pct'), include_groups=False)
    tf_alp = cyc.apply(lambda x: cycle_pnl(x, 'mom_30d', value='alpha_A'), include_groups=False)
    out = pd.DataFrame({'mr_ret': mr_ret, 'mr_alp': mr_alp, 'tf_ret': tf_ret, 'tf_alp': tf_alp}).dropna()
    print(f"\n=== {label}  (n cycles={len(out)}) ===")
    for c in out.columns:
        x = out[c]
        print(f"  {c:8s}  Sharpe {sharpe(x):+6.2f}   mean {x.mean()*1e4:+7.2f} bps   sum {x.sum()*1e4:+9.0f} bps")
    # FLAT benchmark = 0
    print(f"  {'FLAT':8s}  Sharpe   0.00   mean   +0.00 bps   sum        +0 bps")
    return out


# ---- 4. windows ----
full = run(bear, 'BEAR 3yr')
recent = run(bear[bear['open_time'] >= '2025-05-01'], 'BEAR recent-12mo (>=2025-05-01)')

# ---- 5. episode breakdown ----
# contiguous bear episodes: gaps > 1 cycle (4h) split episodes
bt = sorted(bear_times)
episodes = []
cur = [bt[0]]
for t in bt[1:]:
    if (t - cur[-1]) > pd.Timedelta('4h'):
        episodes.append((cur[0], cur[-1], len(cur)))
        cur = [t]
    else:
        cur.append(t)
episodes.append((cur[0], cur[-1], len(cur)))
# merge tiny gaps (< 5 days) into same logical episode for reporting clarity
print(f"\n=== RAW contiguous bear episodes: {len(episodes)} ===")
for s, e, n in episodes:
    if n >= 6:  # >=1 day
        print(f"  {s.date()} -> {e.date()}  ({n} cycles, {n/6:.0f}d)")

# group into logical episodes: merge if gap < 7d
logical = []
cs, ce, cn = episodes[0]
for s, e, n in episodes[1:]:
    if (s - ce) < pd.Timedelta('7d'):
        ce = e; cn += n
    else:
        logical.append((cs, ce, cn)); cs, ce, cn = s, e, n
logical.append((cs, ce, cn))
logical = [(s, e, n) for s, e, n in logical if n >= 6]
print(f"\n=== LOGICAL bear episodes (merged gaps <7d, >=1d): {len(logical)} ===")

cyc = bear.groupby('open_time')
allpnl = pd.DataFrame({
    'mr_ret': cyc.apply(lambda x: cycle_pnl(x, 'pred', value='return_pct'), include_groups=False),
    'tf_ret': cyc.apply(lambda x: cycle_pnl(x, 'mom_30d', value='return_pct'), include_groups=False),
    'mr_alp': cyc.apply(lambda x: cycle_pnl(x, 'pred', value='alpha_A'), include_groups=False),
}).dropna()

print(f"\n{'episode':28s} {'cyc':>5s} {'MR_ret Sh':>10s} {'MR bps':>9s} {'TF_ret Sh':>10s} {'TF bps':>9s} {'MR_alp Sh':>10s}")
for s, e, n in logical:
    seg = allpnl[(allpnl.index >= s) & (allpnl.index <= e)]
    if len(seg) < 5:
        continue
    print(f"{str(s.date())+'..'+str(e.date()):28s} {len(seg):5d} "
          f"{sharpe(seg['mr_ret']):+10.2f} {seg['mr_ret'].mean()*1e4:+9.1f} "
          f"{sharpe(seg['tf_ret']):+10.2f} {seg['tf_ret'].mean()*1e4:+9.1f} "
          f"{sharpe(seg['mr_alp']):+10.2f}")
