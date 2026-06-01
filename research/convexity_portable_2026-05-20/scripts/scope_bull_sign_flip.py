"""
Scope PnL potential of a SIGN-FLIP strategy in BULL regimes.

Standard strategy: long top-K=3 by pred, short bottom-K=3 (cross-sectional MR).
Hypothesis: cross-sectional alpha is negative in bull -> sign-flip captures +PnL.
Critical: check reliability per-fold AND replication 12mo vs 3yr (overfit guard).

EDA only. Does NOT modify any existing files.
"""
import glob
import numpy as np
import pandas as pd

K = 3
ANN = np.sqrt(6 * 365)  # 6 cycles/day, 365 days
PREDS = "/home/yuqing/ctaNew/research/convexity_portable_2026-05-20/results/_cache/x70_v0_3yr_preds.parquet"
BTC_GLOB = "/home/yuqing/ctaNew/data/ml/test/parquet/klines/BTCUSDT/5m/*.parquet"

# ----------------------------------------------------------------------------
# 1. Load preds, subset to 4h-aligned cycles
# ----------------------------------------------------------------------------
df = pd.read_parquet(PREDS, columns=["symbol", "open_time", "alpha_A", "return_pct", "pred", "fold"])
df = df[(df.open_time.dt.hour % 4 == 0) & (df.open_time.dt.minute == 0)].copy()
print(f"4h-aligned pred rows: {len(df):,}  cycles: {df.open_time.nunique():,}")

# ----------------------------------------------------------------------------
# 2. PIT bull regime from BTC 4h closes: ret_30d = close/close.shift(180)-1 > 0.10
# ----------------------------------------------------------------------------
btc = pd.concat([pd.read_parquet(f, columns=["open_time", "close"]) for f in sorted(glob.glob(BTC_GLOB))])
btc = btc.drop_duplicates("open_time").sort_values("open_time")
btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
# 4h closes: take rows at hour%4==0 minute==0
btc4 = btc[(btc.open_time.dt.hour % 4 == 0) & (btc.open_time.dt.minute == 0)].copy()
btc4["btc_ret_30d"] = btc4["close"] / btc4["close"].shift(180) - 1.0   # 180 * 4h = 30d
btc4["bull"] = btc4["btc_ret_30d"] > 0.10
regime = btc4[["open_time", "btc_ret_30d", "bull"]]

df = df.merge(regime, on="open_time", how="inner")
df = df.dropna(subset=["btc_ret_30d"])
print(f"After regime merge: {len(df):,} rows  bull-cycle frac: {df.groupby('open_time').bull.first().mean():.3f}")

# ----------------------------------------------------------------------------
# 3. Per-cycle standard PnL: mean(long-topK alpha) - mean(short-botK alpha)
#    Sign-flip = -standard.
# ----------------------------------------------------------------------------
def cycle_pnl(g, col):
    if len(g) < 2 * K:
        return np.nan
    s = g.sort_values("pred")
    short = s.iloc[:K]   # bottom-K by pred
    long = s.iloc[-K:]   # top-K by pred
    return long[col].mean() - short[col].mean()

rows = []
for (t, fold), g in df.groupby(["open_time", "fold"]):
    bull = bool(g["bull"].iloc[0])
    rows.append((t, fold, bull, cycle_pnl(g, "alpha_A"), cycle_pnl(g, "return_pct")))
cyc = pd.DataFrame(rows, columns=["open_time", "fold", "bull", "std_alpha", "std_ret"]).dropna()
# sign-flip = negative of standard
cyc["flip_alpha"] = -cyc["std_alpha"]
cyc["flip_ret"] = -cyc["std_ret"]
print(f"\nTotal cycles: {len(cyc):,}  bull cycles: {cyc.bull.sum():,}")

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def stats(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return len(x), np.nan, np.nan
    mean_bps = x.mean() * 1e4
    sharpe = x.mean() / x.std(ddof=1) * ANN
    return len(x), mean_bps, sharpe

def report_block(label, sub):
    print(f"\n=== {label}  (bull cycles n={sub.bull.sum()}) ===")
    b = sub[sub.bull]
    for col, nm in [("flip_alpha", "SIGN-FLIP alpha_A"), ("flip_ret", "SIGN-FLIP return_pct"),
                    ("std_alpha", "STANDARD alpha_A"), ("std_ret", "STANDARD return_pct")]:
        n, mb, sh = stats(b[col])
        print(f"  {nm:24s} n={n:5d}  mean={mb:+8.2f} bps  Sharpe={sh:+6.2f}")

# ----------------------------------------------------------------------------
# 4. 12-month window first
# ----------------------------------------------------------------------------
recent = cyc[cyc.open_time >= pd.Timestamp("2025-05-01", tz="UTC")]
report_block("RECENT 12 MONTHS (>= 2025-05-01)", recent)

# ----------------------------------------------------------------------------
# 5. Full 3-year window
# ----------------------------------------------------------------------------
report_block("FULL 3 YEARS", cyc)

# ----------------------------------------------------------------------------
# 6. Per-bull-fold breakdown of sign-flip PnL (consistency / overfit check)
# ----------------------------------------------------------------------------
print("\n=== PER-FOLD SIGN-FLIP PnL IN BULL CYCLES ===")
print(f"{'fold':>4} {'bull_n':>7} {'date_start':>12} {'date_end':>12} "
      f"{'flip_alpha_bps':>15} {'flip_alpha_Sh':>14} {'flip_ret_bps':>13} {'flip_ret_Sh':>12}")
pos_alpha = 0
fold_list = []
for fold, g in cyc.groupby("fold"):
    b = g[g.bull]
    if len(b) == 0:
        continue
    na, mba, sha = stats(b["flip_alpha"])
    nr, mbr, shr = stats(b["flip_ret"])
    d0 = b.open_time.min().date()
    d1 = b.open_time.max().date()
    print(f"{fold:>4} {len(b):>7} {str(d0):>12} {str(d1):>12} "
          f"{mba:>+15.2f} {sha:>+14.2f} {mbr:>+13.2f} {shr:>+12.2f}")
    fold_list.append(fold)
    if mba > 0:
        pos_alpha += 1
print(f"\nBull folds with POSITIVE sign-flip alpha PnL: {pos_alpha}/{len(fold_list)}")

# Per-contiguous-bull-period (gap > 1 day breaks a period)
print("\n=== CONTIGUOUS BULL PERIODS (sign-flip alpha) ===")
b = cyc[cyc.bull].sort_values("open_time").copy()
gap = b.open_time.diff() > pd.Timedelta("1D")
b["period"] = gap.cumsum()
print(f"{'period':>6} {'n':>5} {'start':>12} {'end':>12} {'flip_alpha_bps':>15} {'flip_alpha_Sh':>14}")
pos_per = 0
nper = 0
for p, g in b.groupby("period"):
    if len(g) < 10:
        continue
    n, mb, sh = stats(g["flip_alpha"])
    print(f"{p:>6} {len(g):>5} {str(g.open_time.min().date()):>12} "
          f"{str(g.open_time.max().date()):>12} {mb:>+15.2f} {sh:>+14.2f}")
    nper += 1
    if mb > 0:
        pos_per += 1
print(f"\nContiguous bull periods (n>=10) with POSITIVE sign-flip alpha: {pos_per}/{nper}")
