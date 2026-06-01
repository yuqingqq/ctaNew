"""
Scope: does a BULL-SPECIFIC TRAINED model beat the simple mom_30d RULE in bull?

Prior (scope_bull_trend_follow.py): rule-based trend-following (long top-3 by
momentum / short bottom-3) has genuine positive gross PnL in bull regimes.
Question here: a Ridge model TRAINED only on bull-regime cycles (with momentum
+ panel features, target=target_z) — does it learn a BETTER bull signal, or
just overfit the ~6 bull folds?

Method:
  - 4h-aligned cycles. PIT bull regime: BTC trailing-30d return > +0.10.
  - PIT trailing momentum (mom_7d/14d/30d) + rel_strength vs BTC, built from
    realized 4h-fwd returns shifted strictly into the past; plus panel feats.
  - POOLED Ridge with per-symbol sym-dummies (justify below). Features
    cross-sectionally z-scored within each cycle so the linear model learns a
    cross-sectional ranking (which is what K=3 long/short consumes). Pooled
    chosen over per-symbol because each symbol sees too few bull rows to fit a
    stable per-symbol coefficient vector — pooling shares the bull signal.
  - Expanding walk-forward folds. Train on BULL rows BEFORE test window,
    predict BULL rows IN test window.
  - Selection: long top-3 / short bottom-3 by pred. Cost 4.5 bps/leg.
  - Compare vs SIMPLE rule (long top-3 mom_30d / short bottom-3) on SAME cycles.
  - Test 12mo (>=2025-05-01) AND full 3yr separately; per-bull-fold breakdown.

EDA / exploratory with cost. PIT (no look-ahead).
"""
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

PANEL = "/home/yuqing/ctaNew/outputs/vBTC_features/panel_3yr_v0.parquet"
BTC_GLOB = "/home/yuqing/ctaNew/data/ml/test/parquet/klines/BTCUSDT/5m/*.parquet"
K = 3
COST_BPS = 4.5            # per leg
ANN = np.sqrt(6 * 365)
RIDGE_ALPHA = 10.0

# Cost convention (worst-case): every cycle fully rotates both the long and the
# short basket. PnL = mean(long ret) - mean(short ret) measured over one 4h hold.
# Each of the 2 sides is entered (1 leg) and exited (1 leg) = 2 legs/side * 2
# sides = 4 legs charged against the per-name-notional spread => 4 * 4.5 = 18 bps.
# This is the harsh "rotate every cycle" assumption; a held/overlapped book pays
# less. We report GROSS too so this is comparable to the prior gross-only scope.
COST_FRAC = 4 * COST_BPS * 1e-4

# ---------------------------------------------------------------- load panel
# NOTE: funding_rate / funding_rate_z_7d / funding_rate_1d_change only have data
# from 2025-01 in this 3yr panel. Including them in the dropna would discard ALL
# 2023-2024 bull cycles (2/3 of the bull history) and collapse the 3yr test into
# 2025+ only. We EXCLUDE funding from the model feature set so the bull-trained
# model can be tested over the full 3yr bull universe. All retained features have
# full 2023-2026 coverage.
PANEL_FEATS = ["return_1d", "atr_pct", "vwap_slope_96", "bars_since_high",
               "autocorr_pctile_7d", "obv_z_1d", "corr_to_btc_1d",
               "beta_to_btc_change_5d", "idio_vol_to_btc_1h", "idio_vol_to_btc_1d",
               "rvol_7d", "ret_3d", "btc_rvol_7d"]
cols = (["symbol", "open_time", "return_pct", "alpha_vs_btc_realized", "target_z"]
        + PANEL_FEATS)
df = pd.read_parquet(PANEL, columns=cols)
df = df[(df.open_time.dt.hour % 4 == 0) & (df.open_time.dt.minute == 0)].copy()
df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)

# ------------------------------------------- PIT trailing momentum features
g = df.groupby("symbol", group_keys=False)


def trailing(days):
    n = days * 6
    s = g.apply(lambda x: np.log1p(x["return_pct"]).shift(1)
                .rolling(n, min_periods=n).sum())
    s.index = df.index
    return np.expm1(s)


df["mom_7d"] = trailing(7)
df["mom_14d"] = trailing(14)
df["mom_30d"] = trailing(30)

# ---------------------------------------------------------- BTC 30d regime
btc = pd.concat([pd.read_parquet(f, columns=["open_time", "close"])
                 for f in sorted(glob.glob(BTC_GLOB))], ignore_index=True)
btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
btc = btc.sort_values("open_time").drop_duplicates("open_time")
btc4 = btc[(btc.open_time.dt.hour % 4 == 0) & (btc.open_time.dt.minute == 0)].copy()
btc4 = btc4.set_index("open_time")
btc4["btc_ret_30d"] = btc4["close"] / btc4["close"].shift(180) - 1
btc4["btc_ret_7d"] = btc4["close"] / btc4["close"].shift(42) - 1
df["btc_ret_30d"] = df["open_time"].map(btc4["btc_ret_30d"])
df["btc_ret_7d"] = df["open_time"].map(btc4["btc_ret_7d"])
df["bull"] = df["btc_ret_30d"] > 0.10

# relative strength vs BTC (PIT)
df["rel_strength_7d"] = df["mom_7d"] - df["btc_ret_7d"]
df["rel_strength_30d"] = df["mom_30d"] - df["btc_ret_30d"]

MOM_FEATS = ["mom_7d", "mom_14d", "mom_30d", "rel_strength_7d", "rel_strength_30d"]
FEATS = MOM_FEATS + PANEL_FEATS

# ----------------------------------------- cross-sectional z within cycle
# z-score each feature within each cycle so the model learns cross-sectional
# rank structure (what the K=3 long/short consumes), regime-scale invariant.
df_model = df.dropna(subset=FEATS + ["target_z", "return_pct",
                                     "alpha_vs_btc_realized"]).copy()


def xs_z(frame, feats):
    out = frame.copy()
    gb = out.groupby("open_time")
    for f in feats:
        mu = gb[f].transform("mean")
        sd = gb[f].transform("std")
        out[f + "_z"] = ((out[f] - mu) / (sd + 1e-9)).fillna(0.0)
    return out


df_model = xs_z(df_model, FEATS)
ZFEATS = [f + "_z" for f in FEATS]

# ---------------------------------------------- expanding walk-forward folds
# Bull cycles cluster in episodes (2023-Q1/Q2/Q4, 2024-Q1, 2024-Q4, 2025-Q2...).
# A calendar-uniform grid leaves early folds with no bull test rows. Instead cut
# the test-window edges at quantiles of the BULL-cycle timestamps so each of the
# 8 folds holds a comparable count of bull cycles, with a burn-in: the first fold
# is reserved entirely for training history (never tested). Within each test fold
# we train on BULL rows strictly before the window, predict BULL rows in window.
bull_times = np.sort(df_model.loc[df_model["bull"], "open_time"].unique())
N_FOLDS = 9                       # fold 0 = burn-in (train-only), folds 1..8 tested
q = np.linspace(0, 1, N_FOLDS + 1)
edges_raw = pd.to_datetime(pd.Series(bull_times).quantile(q).values, utc=True)
edges = pd.to_datetime(np.unique(edges_raw.values))
edges = pd.DatetimeIndex(edges).tz_localize("UTC") if edges.tz is None else pd.DatetimeIndex(edges)
edges = edges.insert(len(edges), df_model["open_time"].max() + pd.Timedelta(hours=4))
# test windows: skip the first edge (burn-in)
folds = [(edges[i], edges[i + 1]) for i in range(1, len(edges) - 1)]


def fit_predict():
    """Per-fold: Ridge on bull rows before test window -> predict bull in window."""
    preds = []
    syms = sorted(df_model["symbol"].unique())
    sym_idx = {s: i for i, s in enumerate(syms)}
    for fi, (lo, hi) in enumerate(folds):
        tr = df_model[(df_model["open_time"] < lo) & (df_model["bull"])]
        te = df_model[(df_model["open_time"] >= lo) & (df_model["open_time"] < hi)
                      & (df_model["bull"])]
        if len(tr) < 100 or len(te) == 0:
            continue
        # pooled with sym dummies
        Xtr = tr[ZFEATS].values.astype(np.float64)
        Xte = te[ZFEATS].values.astype(np.float64)
        dtr = np.zeros((len(tr), len(syms)), np.float64)
        dte = np.zeros((len(te), len(syms)), np.float64)
        for i, s in enumerate(tr["symbol"].values):
            dtr[i, sym_idx[s]] = 1.0
        for i, s in enumerate(te["symbol"].values):
            dte[i, sym_idx[s]] = 1.0
        Xtr = np.hstack([Xtr, dtr])
        Xte = np.hstack([Xte, dte])
        ytr = tr["target_z"].values.astype(np.float64)
        m = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        sub = te[["symbol", "open_time", "return_pct", "alpha_vs_btc_realized",
                  "mom_30d", "bull"]].copy()
        sub["pred"] = p
        sub["fold"] = fi
        preds.append(sub)
    return pd.concat(preds, ignore_index=True)


pred_df = fit_predict()
print(f"Bull-trained predictions: {len(pred_df):,} rows over "
      f"{pred_df['fold'].nunique()} folds, "
      f"{pred_df['open_time'].nunique()} bull cycles")

# -------------------------------------------------------- selection / PnL
def cycle_pnl(sub, signal, ascending=False):
    sub = sub.dropna(subset=[signal, "return_pct", "alpha_vs_btc_realized"])
    if len(sub) < 2 * K:
        return None
    ranked = sub.sort_values(signal, ascending=ascending)
    longs = ranked.head(K)
    shorts = ranked.tail(K)
    raw = longs["return_pct"].mean() - shorts["return_pct"].mean()
    alpha = (longs["alpha_vs_btc_realized"].mean()
             - shorts["alpha_vs_btc_realized"].mean())
    return raw, alpha


def run_signal(panel, signal):
    out = []
    for t, sub in panel.groupby("open_time"):
        r = cycle_pnl(sub, signal, ascending=False)  # long high signal
        if r is not None:
            out.append((t, r[0], r[1]))
    return pd.DataFrame(out, columns=["open_time", "raw", "alpha"]).set_index("open_time")


COST_HELD = 2 * COST_BPS * 1e-4   # held/overlapped book: ~2 legs (9 bps)


def stats(s, cost=0.0):
    if len(s) == 0:
        return dict(n=0, mean_bps=np.nan, sharpe=np.nan)
    s2 = s - cost
    return dict(n=len(s), mean_bps=s2.mean() * 1e4,
                sharpe=(s2.mean() / s2.std() * ANN) if s2.std() > 0 else np.nan)


def report_block(tag, pred_sub):
    print(f"\n{'='*72}\n{tag}: {pred_sub['open_time'].nunique()} bull cycles\n{'='*72}")
    # bull-trained model
    res_m = run_signal(pred_sub, "pred")
    # simple mom_30d rule on the SAME cycles (same rows / universe)
    res_r = run_signal(pred_sub, "mom_30d")
    for name, res in [("BULL-TRAINED model", res_m), ("SIMPLE mom_30d rule", res_r)]:
        for kind in ["raw", "alpha"]:
            g_ = stats(res[kind], cost=0.0)
            h_ = stats(res[kind], cost=COST_HELD)
            n_ = stats(res[kind], cost=COST_FRAC)
            print(f"  {name:22s} {kind:5s}  gross Sh {g_['sharpe']:+.2f} "
                  f"({g_['mean_bps']:+6.1f} bps) | held(9bp) Sh {h_['sharpe']:+.2f} "
                  f"| rotate(18bp) Sh {n_['sharpe']:+.2f}")
    return res_m, res_r


# 12-month
p12 = pred_df[pred_df["open_time"] >= pd.Timestamp("2025-05-01", tz="UTC")]
report_block("12-MONTH (>= 2025-05-01)", p12)
# full 3yr
m_full, r_full = report_block("FULL 3-YEAR", pred_df)

# ----------------------------------------------- per-bull-fold breakdown
print(f"\n{'='*72}\nPER-FOLD CONSISTENCY (full 3yr, GROSS raw Sharpe; +bps gross mean)\n{'='*72}")
print(f"  {'fold':>4} {'window':<25} {'n_cyc':>5}  {'TRAINED gSh':>11} {'(bps)':>8}  "
      f"{'RULE gSh':>9} {'(bps)':>8}  {'T>R?':>5}")
wins = 0
nf = 0
for fi in sorted(pred_df["fold"].unique()):
    ps = pred_df[pred_df["fold"] == fi]
    rm = run_signal(ps, "pred")["raw"]
    rr = run_signal(ps, "mom_30d")["raw"]
    if len(rm) < 5:        # skip degenerate single-cycle tail fold
        continue
    nf += 1
    sm = stats(rm)
    sr = stats(rr)
    better = sm["mean_bps"] > sr["mean_bps"]
    wins += int(better)
    win = f"{ps['open_time'].min().date()}..{ps['open_time'].max().date()}"
    print(f"  {fi:>4} {win:<25} {len(rm):>5}  {sm['sharpe']:>+11.2f} {sm['mean_bps']:>+8.1f}  "
          f"{sr['sharpe']:>+9.2f} {sr['mean_bps']:>+8.1f}  {'YES' if better else 'no':>5}")

# ----------------------------------------------- head-to-head verdict
print(f"\n{'='*72}\nHEAD-TO-HEAD (full 3yr): does TRAINED beat RULE?\n{'='*72}")
mr = run_signal(pred_df, "pred")["raw"]
rr = run_signal(pred_df, "mom_30d")["raw"]
for cost, lbl in [(0.0, "GROSS"), (COST_HELD, "NET held 9bp"), (COST_FRAC, "NET rotate 18bp")]:
    sm, sr = stats(mr, cost=cost), stats(rr, cost=cost)
    print(f"  {lbl:16s}  TRAINED Sh {sm['sharpe']:+.2f}  RULE Sh {sr['sharpe']:+.2f}  "
          f"Δ {sm['sharpe'] - sr['sharpe']:+.2f}")
print(f"  TRAINED beats RULE (gross mean PnL) in {wins}/{nf} bull folds")
