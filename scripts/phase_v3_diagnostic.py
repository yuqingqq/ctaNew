"""V3.1 per-cycle PnL × market data diagnostic.

Loads V3.1 equal_6 per-cycle simulation, joins BTC market regime + cross-sectional
dispersion at entry, and bucket-analyzes conditional Sharpe / mean PnL to find
data-driven insights for the next architectural step.

Sections:
  1. Per-fold + skip-rate × outcome anatomy
  2. BTC regime: vol, trend, dollar volume bucket conditional PnL
  3. Cross-sectional dispersion (pred_disp at entry) conditional PnL
  4. Time-of-day effect (4h bucket within UTC day)
  5. Day-of-week effect
  6. Cost vs gross relationship (when is cost dominant)
  7. Loss clustering / serial-correlation
  8. Concentration (what fraction of net comes from top-N cycles)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

OUT = REPO / "outputs/vBTC_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)
CYCLES_PER_YEAR = (288 * 365) / 48


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def load_btc_4h():
    """Load BTC 5m klines, resample to 4h bars at the same grid as cycle times."""
    bdir = REPO / "data/ml/test/parquet/klines/BTCUSDT/5m"
    files = sorted(bdir.glob("*.parquet"))
    dfs = [pd.read_parquet(f, columns=["open_time", "close", "volume", "high", "low"])
           for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.dropna().drop_duplicates("open_time").set_index("open_time").sort_index()
    # log returns at 5m
    df["ret_5m"] = np.log(df["close"]).diff()
    # 4h-aligned features (use trailing windows ending at t)
    out = pd.DataFrame(index=df.index)
    out["btc_close"] = df["close"]
    out["btc_ret_4h"] = df["close"].pct_change(48)
    out["btc_ret_24h"] = df["close"].pct_change(288)
    out["btc_ret_3d"] = df["close"].pct_change(288 * 3)
    out["btc_rvol_24h"] = df["ret_5m"].rolling(288).std() * np.sqrt(288)  # daily realized vol
    out["btc_rvol_3d"] = df["ret_5m"].rolling(288 * 3).std() * np.sqrt(288)
    out["btc_dvol_24h"] = (df["close"] * df["volume"]).rolling(288).sum()  # 24h dollar volume
    out["btc_range_4h"] = (df["high"].rolling(48).max() - df["low"].rolling(48).min()) / df["close"]
    return out


def attach_btc_at(df, btc_features):
    """For each cycle time t, attach BTC features at t (nearest <= t)."""
    df = df.sort_values("time").copy()
    df["time"] = pd.to_datetime(df["time"], utc=True).astype("datetime64[ns, UTC]")
    btc = btc_features.copy()
    btc.index = pd.to_datetime(btc.index, utc=True).astype("datetime64[ns, UTC]")
    btc.index.name = "time"
    df = pd.merge_asof(df, btc.reset_index(), on="time", direction="backward",
                         tolerance=pd.Timedelta("5min"))
    return df


def load_predictions():
    """Load all_predictions parquet for cross-sectional dispersion at cycle times."""
    df = pd.read_parquet(REPO / "outputs/vBTC_audit_panel/all_predictions.parquet")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df


def cs_dispersion_at(pred_df, ts):
    """Cross-sectional pred std and IQR at time ts."""
    g = pred_df[pred_df["open_time"] == ts]
    if len(g) == 0: return np.nan, np.nan, np.nan, 0
    p = g["pred"].dropna().to_numpy()
    if len(p) < 5: return np.nan, np.nan, np.nan, len(p)
    return float(p.std()), float(np.percentile(p, 75) - np.percentile(p, 25)), \
           float(p.max() - p.min()), len(p)


def bucket_analysis(df, col, q=5, label=None):
    """Quintile bucket analysis on `col`."""
    sub = df[df[col].notna()].copy()
    if len(sub) < q * 5:
        print(f"  too few cycles for {col} ({len(sub)})")
        return
    sub["bucket"] = pd.qcut(sub[col], q, labels=False, duplicates="drop")
    rows = []
    for b in sorted(sub["bucket"].dropna().unique()):
        g = sub[sub["bucket"] == b]
        net = g["net_pnl_bps"].to_numpy()
        gross = g["gross_pnl_bps"].to_numpy()
        rows.append({
            "bucket": int(b),
            f"{col}_min": float(g[col].min()),
            f"{col}_max": float(g[col].max()),
            "n_cycles": len(g),
            "mean_net": float(net.mean()),
            "mean_gross": float(gross.mean()),
            "mean_cost": float(g["cost_bps"].mean()),
            "sharpe_net": _sharpe(net),
            "pct_positive": float((net > 0).mean() * 100),
            "mean_turnover": float(g["turnover"].mean()),
            "mean_gross_exp": float(g["gross_exposure"].mean()),
        })
    res = pd.DataFrame(rows)
    label = label or col
    print(f"\n  --- {label} (n_total={len(sub)}) ---")
    print(res.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    return res


def main():
    print("=== V3.1 per-cycle × market data diagnostic ===\n", flush=True)

    # ---------- 0. Load data ----------
    df = pd.read_csv(REPO / "outputs/vBTC_sleeve_horizon/per_cycle_robust_equal_6.csv")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    print(f"V3.1 cycles: {len(df)}  (time {df.time.min()} → {df.time.max()})", flush=True)

    print("  loading BTC 5m klines...", flush=True)
    t0 = time.time()
    btc = load_btc_4h()
    df = attach_btc_at(df, btc)
    print(f"  attached BTC features ({time.time()-t0:.0f}s)", flush=True)

    print("  loading prediction panel for CS dispersion...", flush=True)
    t0 = time.time()
    pred = load_predictions()
    # Compute pred dispersion at each unique cycle time
    disp_rows = []
    for ts in df["time"].unique():
        s, iqr, rng, n = cs_dispersion_at(pred, ts)
        disp_rows.append({"time": ts, "cs_pred_std": s, "cs_pred_iqr": iqr,
                          "cs_pred_range": rng, "cs_n_pred": n})
    disp_df = pd.DataFrame(disp_rows)
    disp_df["time"] = pd.to_datetime(disp_df["time"], utc=True)
    df = df.merge(disp_df, on="time", how="left")
    print(f"  CS dispersion attached ({time.time()-t0:.0f}s)", flush=True)

    # Time features
    df["hour_utc"] = df["time"].dt.hour
    df["dow"] = df["time"].dt.dayofweek
    df["traded"] = df["gross_exposure"] > 0.01

    df.to_csv(OUT / "v3_1_cycles_enriched.csv", index=False)
    print(f"\n  saved enriched cycles to {OUT/'v3_1_cycles_enriched.csv'}", flush=True)
    print(f"  cols: {list(df.columns)}\n", flush=True)

    # ---------- 1. Per-fold + skip anatomy ----------
    print("=== 1. Per-fold + skip anatomy ===", flush=True)
    rows = []
    for f, g in df.groupby("fold"):
        skip_rate = (~g["traded"]).mean()
        traded_g = g[g["traded"]]
        nontr = g[~g["traded"]]
        rows.append({
            "fold": int(f),
            "cycles": len(g),
            "skip_rate": float(skip_rate * 100),
            "sharpe_all": _sharpe(g["net_pnl_bps"]),
            "sharpe_traded": _sharpe(traded_g["net_pnl_bps"]) if len(traded_g) else 0,
            "mean_net_all": float(g["net_pnl_bps"].mean()),
            "mean_net_traded": float(traded_g["net_pnl_bps"].mean()) if len(traded_g) else 0,
            "pct_pos_traded": float((traded_g["net_pnl_bps"] > 0).mean() * 100) if len(traded_g) else 0,
            "sum_net": float(g["net_pnl_bps"].sum()),
            "cost_share": float(g["cost_bps"].sum() / max(g["gross_pnl_bps"].sum(), 1e-9) * 100)
                          if g["gross_pnl_bps"].sum() > 0 else np.nan,
        })
    fold_df = pd.DataFrame(rows)
    print(fold_df.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    # ---------- 2. BTC regime ----------
    print("\n=== 2. BTC regime conditional PnL (traded cycles only) ===", flush=True)
    tr = df[df["traded"]].copy()
    bucket_analysis(tr, "btc_rvol_24h", q=5, label="BTC 24h realized vol")
    bucket_analysis(tr, "btc_rvol_3d", q=5, label="BTC 3d realized vol")
    bucket_analysis(tr, "btc_ret_24h", q=5, label="BTC 24h return (entry-time)")
    bucket_analysis(tr, "btc_ret_3d", q=5, label="BTC 3d return")
    bucket_analysis(tr, "btc_range_4h", q=5, label="BTC 4h high-low range %")
    bucket_analysis(tr, "btc_dvol_24h", q=5, label="BTC 24h dollar volume")

    # ---------- 3. CS dispersion ----------
    print("\n=== 3. Cross-sectional pred dispersion conditional PnL ===", flush=True)
    bucket_analysis(tr, "cs_pred_std", q=5, label="CS pred std (entry)")
    bucket_analysis(tr, "cs_pred_iqr", q=5, label="CS pred IQR (entry)")
    bucket_analysis(tr, "cs_pred_range", q=5, label="CS pred range (entry)")

    # ---------- 4. Time-of-day ----------
    print("\n=== 4. Time-of-day (UTC hour) effect ===", flush=True)
    tod = tr.groupby("hour_utc")["net_pnl_bps"].agg(["mean", "count", _sharpe]).reset_index()
    tod.columns = ["hour_utc", "mean_net", "n", "sharpe"]
    tod = tod.sort_values("mean_net")
    print(tod.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    # ---------- 5. Day-of-week ----------
    print("\n=== 5. Day-of-week effect ===", flush=True)
    dow = tr.groupby("dow")["net_pnl_bps"].agg(["mean", "count", _sharpe]).reset_index()
    dow.columns = ["dow", "mean_net", "n", "sharpe"]
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow["name"] = dow["dow"].map(lambda i: dow_names[int(i)])
    print(dow.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    # ---------- 6. Cost dominance ----------
    print("\n=== 6. Cost vs gross dominance ===", flush=True)
    tr["cost_share"] = tr["cost_bps"] / (tr["gross_pnl_bps"].abs() + 0.01)
    print(f"  median cost/|gross|: {tr['cost_share'].median():.2f}")
    print(f"  cycles where cost > |gross|: {(tr['cost_bps'] > tr['gross_pnl_bps'].abs()).mean() * 100:.0f}%")
    # When gross is small (low-edge regime), is cost dominant?
    tr["abs_gross"] = tr["gross_pnl_bps"].abs()
    bucket_analysis(tr, "abs_gross", q=5, label="|gross PnL| (proxy for signal strength)")

    # ---------- 7. Loss clustering ----------
    print("\n=== 7. Loss clustering (serial autocorrelation) ===", flush=True)
    for lag in [1, 2, 3, 6, 12, 24]:
        r = tr["net_pnl_bps"].autocorr(lag=lag)
        print(f"  autocorr lag={lag:>3} cycles ({lag*4}h): {r:+.3f}")

    # ---------- 8. Concentration ----------
    print("\n=== 8. PnL concentration ===", flush=True)
    sorted_net = df["net_pnl_bps"].sort_values(ascending=False).to_numpy()
    cum = np.cumsum(sorted_net)
    total = cum[-1]
    print(f"  total net: {total:+.0f} bps")
    for pct in [1, 5, 10, 20, 50]:
        k = int(len(sorted_net) * pct / 100)
        share = cum[k - 1] / total * 100 if total != 0 else np.nan
        print(f"  top {pct:>3}% cycles ({k:>4} cycles) contribute {share:+.0f}% of total")
    # Also: what fraction of LOSS comes from worst N%
    sorted_loss = df["net_pnl_bps"].sort_values().to_numpy()
    cum_loss = np.cumsum(sorted_loss)
    losses_only = sorted_loss[sorted_loss < 0]
    if len(losses_only) > 0:
        loss_sum = losses_only.sum()
        for pct in [1, 5, 10, 20]:
            k = int(len(sorted_loss) * pct / 100)
            share = cum_loss[k - 1] / loss_sum * 100 if loss_sum != 0 else np.nan
            print(f"  worst {pct:>3}% cycles ({k:>4} cycles) contribute {share:+.0f}% of total losses")

    # ---------- 9. Distribution shape ----------
    print("\n=== 9. PnL distribution shape ===", flush=True)
    net_arr = df["net_pnl_bps"].to_numpy()
    pcts = [1, 5, 25, 50, 75, 95, 99]
    print(f"  percentiles ({', '.join(f'p{p}' for p in pcts)}):")
    print(f"  {', '.join(f'{np.percentile(net_arr, p):+.1f}' for p in pcts)}")
    print(f"  skew {pd.Series(net_arr).skew():+.2f}  kurtosis {pd.Series(net_arr).kurtosis():+.2f}")

    print(f"\n=== Enriched data + summaries saved to {OUT} ===")


if __name__ == "__main__":
    main()
