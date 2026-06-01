"""
Scope: PnL potential of TREND-FOLLOWING selection in BULL regimes.

Cross-sectional mean-reversion fails in bull markets (prior: ~-2.59 ann Sharpe
at 4h-cohort level). Hypothesis: trend-following (long winners / short losers)
captures positive PnL in bull regimes instead.

EDA only - gross PnL, no costs. PIT (no look-ahead).
"""
import glob
import numpy as np
import pandas as pd

PANEL = "/home/yuqing/ctaNew/outputs/vBTC_features/panel_3yr_v0.parquet"
BTC_GLOB = "/home/yuqing/ctaNew/data/ml/test/parquet/klines/BTCUSDT/5m/*.parquet"
K = 3
ANN = np.sqrt(6 * 365)  # 4h non-overlapping cycles ~ independent

# ---------------------------------------------------------------- load panel
cols = ["symbol", "open_time", "return_pct", "alpha_vs_btc_realized", "ret_3d"]
df = pd.read_parquet(PANEL, columns=cols)
# 4h-aligned cycles only
df = df[(df.open_time.dt.hour % 4 == 0) & (df.open_time.dt.minute == 0)].copy()
df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)

# ---------------------------------------------- build PIT trailing momentum
# return_pct is the 4h-FORWARD return realized over [open_time, open_time+4h].
# So the realized return of the *previous* 4h window is return_pct.shift(1).
# Trailing N-day momentum (PIT, known at open_time) = compounded realized
# returns over the past N days = product over the last N*6 windows ending at
# the window prior to open_time.
g = df.groupby("symbol", group_keys=False)
log1p = np.log1p(df["return_pct"])  # log of each 4h-forward window return

def trailing(days):
    n = days * 6  # 4h bars per day = 6
    # sum of past `n` realized log-returns, shifted by 1 so it's strictly past
    s = g.apply(lambda x: np.log1p(x["return_pct"]).shift(1)
                .rolling(n, min_periods=n).sum())
    s.index = df.index
    return np.expm1(s)

df["mom_7d"] = trailing(7)
df["mom_30d"] = trailing(30)
# ret_3d already in panel (trailing 3-day return feature)

# ---------------------------------------------------------- BTC 30d regime
btc = pd.concat([pd.read_parquet(f, columns=["open_time", "close"])
                 for f in sorted(glob.glob(BTC_GLOB))], ignore_index=True)
btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
btc = btc.sort_values("open_time").drop_duplicates("open_time")
btc4 = btc[(btc.open_time.dt.hour % 4 == 0) & (btc.open_time.dt.minute == 0)].copy()
btc4 = btc4.set_index("open_time")
btc4["btc_ret_30d"] = btc4["close"] / btc4["close"].shift(180) - 1  # 180 4h-bars
regime = btc4["btc_ret_30d"]

df["btc_ret_30d"] = df["open_time"].map(regime)
df["bull"] = df["btc_ret_30d"] > 0.10

# -------------------------------------------------------- selection / PnL
def cycle_pnl(sub, signal, ascending=False):
    """Rank by signal; long top-K, short bottom-K. Return (raw, alpha) PnL."""
    sub = sub.dropna(subset=[signal, "return_pct", "alpha_vs_btc_realized"])
    if len(sub) < 2 * K:
        return None
    ranked = sub.sort_values(signal, ascending=ascending)
    longs = ranked.head(K)   # high signal
    shorts = ranked.tail(K)  # low signal
    raw = longs["return_pct"].mean() - shorts["return_pct"].mean()
    alpha = (longs["alpha_vs_btc_realized"].mean()
             - shorts["alpha_vs_btc_realized"].mean())
    return raw, alpha

def run(panel, signal, label):
    """Trend-follow (long winners). Returns per-cycle PnL series."""
    out = []
    for t, sub in panel.groupby("open_time"):
        r = cycle_pnl(sub, signal, ascending=False)
        if r is not None:
            out.append((t, r[0], r[1]))
    res = pd.DataFrame(out, columns=["open_time", "raw", "alpha"]).set_index("open_time")
    return res

def stats(s):
    if len(s) == 0:
        return dict(n=0, mean_bps=np.nan, sharpe=np.nan)
    return dict(n=len(s), mean_bps=s.mean() * 1e4,
                sharpe=(s.mean() / s.std() * ANN) if s.std() > 0 else np.nan)

# -------------------------------------------------------------- evaluate
def evaluate(panel, tag):
    bull = panel[panel["bull"]].copy()
    print(f"\n{'='*70}\n{tag}: bull cycles = {bull.open_time.nunique()} "
          f"/ {panel.open_time.nunique()} total\n{'='*70}")
    for sig in ["ret_3d", "mom_7d", "mom_30d"]:
        res = run(bull, sig, sig)
        raw, alp = stats(res["raw"]), stats(res["alpha"])
        # mean-reversion = exact negative of trend-follow (long losers/short winners)
        mr_raw, mr_alp = -res["raw"], -res["alpha"]
        print(f"\n  signal={sig}  (n_cycles={raw['n']})")
        print(f"    TREND-FOLLOW  raw   Sharpe {raw['sharpe']:+.2f}  "
              f"mean {raw['mean_bps']:+.2f} bps")
        print(f"    TREND-FOLLOW  alpha Sharpe {alp['sharpe']:+.2f}  "
              f"mean {alp['mean_bps']:+.2f} bps")
        print(f"    MEAN-REVERT   raw   Sharpe {stats(mr_raw)['sharpe']:+.2f}  "
              f"mean {stats(mr_raw)['mean_bps']:+.2f} bps")
        print(f"    MEAN-REVERT   alpha Sharpe {stats(mr_alp)['sharpe']:+.2f}  "
              f"mean {stats(mr_alp)['mean_bps']:+.2f} bps")
    return bull

# 12-month window
p12 = df[df["open_time"] >= pd.Timestamp("2025-05-01", tz="UTC")]
evaluate(p12, "12-MONTH (>= 2025-05-01)")

# full 3-year
evaluate(df, "FULL 3-YEAR")

# ---------------------------------------------- per bull-period breakdown (3yr)
print(f"\n{'='*70}\nPER BULL-PERIOD BREAKDOWN (3yr, signal=mom_7d, trend-follow raw)\n{'='*70}")
bull = df[df["bull"]].copy()
res = run(bull, "mom_7d", "mom_7d")
# group contiguous bull cycles into episodes
times = pd.Series(sorted(bull.open_time.unique()))
gap = times.diff() > pd.Timedelta(hours=8)
episode = gap.cumsum()
ep_map = dict(zip(times, episode))
res = res.reset_index()
res["ep"] = res["open_time"].map(ep_map)
for ep, sub in res.groupby("ep"):
    s = sub["raw"]
    sh = (s.mean() / s.std() * ANN) if s.std() > 0 else np.nan
    print(f"  ep{int(ep):>2}  {sub.open_time.min().date()}..{sub.open_time.max().date()}  "
          f"n={len(sub):>4}  mean {s.mean()*1e4:+7.2f} bps  Sharpe {sh:+.2f}")
