"""Build BTC-only feature panel for the universe-portable α-residual strategy.

Engineers 25 features that are all universe-invariant: no basket references, no
cross-sectional ranks, no sym_id. Derived from per-symbol klines + funding +
BTC reference series only.

Output: outputs/vBTC_features_btc_only/panel_btc_only.parquet

Feature groups:
  (1) BTC residual momentum (3 horizons)
  (2) BTC residual price level (3)
  (3) BTC β/corr state (4)
  (4) BTC residual risk (3)
  (5) BTC market regime — same for all syms at t (3)
  (6) Single-name flow/funding (6)
  (7) Stable per-symbol context — replaces sym_id (3)

All windows are trailing/PIT, shifted by 1 bar where needed.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT_DIR = REPO / "outputs/vBTC_features_btc_only"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Bar/horizon convention: 5-min bars per the existing panel
BAR_5MIN = 5
BARS_PER_HOUR = 12
BARS_PER_DAY = 288
BETA_WINDOW = 90 * BARS_PER_DAY  # PIT 90d rolling β


def load_klines(sym):
    """Load all 5min klines for a symbol; returns (open_time, close, quote_volume)."""
    sd = KLINES_DIR / sym / "5m"
    if not sd.exists(): return None
    files = sorted(sd.glob("*.parquet"))
    if not files: return None
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["open_time", "close", "quote_volume"])
            dfs.append(df)
        except Exception:
            try:
                df = pd.read_parquet(f, columns=["open_time", "close"])
                df["quote_volume"] = np.nan
                dfs.append(df)
            except Exception: pass
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["open_time"]).drop_duplicates("open_time").sort_values("open_time")
    return df


def compute_pit_beta(sym_ret, btc_ret, window_bars):
    """PIT rolling β shifted by 1 bar."""
    cov = sym_ret.rolling(window_bars, min_periods=1000).cov(btc_ret)
    var = btc_ret.rolling(window_bars, min_periods=1000).var()
    beta = (cov / var.replace(0, np.nan)).shift(1)
    return beta


def main():
    print("=== Build BTC-only feature panel ===\n", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel_syms = sorted(panel["symbol"].unique())
    print(f"Existing panel: {len(panel):,} rows × {len(panel_syms)} symbols", flush=True)

    # ============================================================
    # Step 1: Load BTC's full 5min series (closes + returns + volumes)
    # ============================================================
    print("\n--- Step 1: load BTC 5min klines ---", flush=True)
    btc_df = load_klines("BTCUSDT")
    btc_df = btc_df.set_index("open_time")
    btc_df["btc_ret_5m"] = btc_df["close"].pct_change()
    print(f"  BTC: {len(btc_df):,} 5m bars from {btc_df.index.min()} to {btc_df.index.max()}",
          flush=True)

    # BTC market regime features (same for all symbols at time t)
    btc_df["btc_ret_48b"] = btc_df["close"].pct_change(48)  # 4h backward return
    btc_df["btc_realized_vol_1d"] = btc_df["btc_ret_5m"].rolling(BARS_PER_DAY).std()
    btc_df["btc_realized_vol_30d"] = btc_df["btc_ret_5m"].rolling(30 * BARS_PER_DAY).std()

    # ============================================================
    # Step 2: For each symbol, load klines and build per-symbol features
    # ============================================================
    print("\n--- Step 2: per-symbol feature engineering ---", flush=True)
    t_start = time.time()
    new_features_per_sym = []

    for i, sym in enumerate(panel_syms):
        sym_df = load_klines(sym)
        if sym_df is None or len(sym_df) < 1000:
            print(f"  skip {sym}: insufficient klines", flush=True)
            continue
        sym_df = sym_df.set_index("open_time")
        sym_df["sym_ret_5m"] = sym_df["close"].pct_change()
        # Align with BTC
        df = sym_df.join(btc_df[["close", "btc_ret_5m", "btc_ret_48b",
                                  "btc_realized_vol_1d", "btc_realized_vol_30d"]],
                          how="inner", rsuffix="_btc")

        # --- Group 1: BTC residual momentum (3 horizons) ---
        # Need PIT β first
        beta_pit = compute_pit_beta(df["sym_ret_5m"], df["btc_ret_5m"], BETA_WINDOW)
        df["beta_btc_pit"] = beta_pit

        for h_bars, name in [(12, "12b"), (48, "48b"), (288, "288b")]:
            sym_h_ret = df["close"].pct_change(h_bars)
            btc_h_ret = df["close_btc"].pct_change(h_bars)
            df[f"idio_ret_to_btc_{name}"] = sym_h_ret - df["beta_btc_pit"] * btc_h_ret

        # --- Group 2: BTC residual price level ---
        df["log_price_ratio"] = np.log(df["close"]) - np.log(df["close_btc"])
        df["dom_btc_change_48b"] = df["log_price_ratio"].diff(48)
        df["dom_btc_change_288b"] = df["log_price_ratio"].diff(288)
        ma_1d = df["log_price_ratio"].rolling(BARS_PER_DAY).mean()
        std_1d = df["log_price_ratio"].rolling(BARS_PER_DAY).std()
        df["dom_btc_z_1d"] = ((df["log_price_ratio"] - ma_1d) / std_1d.replace(0, np.nan))

        # --- Group 3: BTC β/corr state ---
        df["beta_to_btc_change_5d"] = df["beta_btc_pit"].diff(5 * BARS_PER_DAY)
        # corr_to_btc_1d: rank-correlation of returns over 1d trailing window
        df["corr_to_btc_1d"] = df["sym_ret_5m"].rolling(BARS_PER_DAY).corr(df["btc_ret_5m"])
        df["corr_to_btc_change_3d"] = df["corr_to_btc_1d"].diff(3 * BARS_PER_DAY)

        # --- Group 4: BTC residual risk ---
        # Build a per-bar idio return series (5min) using PIT β
        idio_ret_5m = df["sym_ret_5m"] - df["beta_btc_pit"] * df["btc_ret_5m"]
        df["idio_vol_to_btc_1h"] = idio_ret_5m.rolling(BARS_PER_HOUR).std()
        df["idio_vol_to_btc_1d"] = idio_ret_5m.rolling(BARS_PER_DAY).std()
        btc_vol_1d = df["btc_ret_5m"].rolling(BARS_PER_DAY).std()
        df["idio_vol_ratio_to_btc"] = df["idio_vol_to_btc_1d"] / btc_vol_1d.replace(0, np.nan)

        # --- Group 5: BTC market regime (already on df via join) ---
        # btc_ret_48b, btc_realized_vol_1d, btc_realized_vol_30d → already there

        # --- Group 6: Single-name flow features ---
        # atr_pct, obv_z_1d, vwap_slope_96, funding_* — already in existing panel; we'll merge from there
        # We'll only set placeholders here; the merge below will fill them

        # --- Group 7: Stable per-symbol context ---
        # listing_age_days: how many days since this symbol's first kline
        first_kline = sym_df.index.min()
        df["listing_age_days"] = ((df.index - first_kline).total_seconds() / 86400).astype(float)
        # log_quote_volume_90d: log of trailing 90d median quote volume
        if "quote_volume" in df.columns:
            qv_median_90d = df["quote_volume"].rolling(90 * BARS_PER_DAY).median()
            df["log_quote_volume_90d"] = np.log(qv_median_90d.replace(0, np.nan))
        else:
            df["log_quote_volume_90d"] = np.nan
        # residual_vol_90d_own_pctile: where current idio_vol_1d sits in own trailing 90d distribution
        # Use rolling rank
        roll_90d = df["idio_vol_to_btc_1d"].rolling(90 * BARS_PER_DAY, min_periods=1000)
        df["residual_vol_90d_own_pctile"] = roll_90d.apply(
            lambda x: (x.iloc[-1] > x).mean() if len(x) > 100 else np.nan, raw=False)

        # Save only the new feature columns + symbol/open_time
        new_cols = [
            "beta_btc_pit",
            "idio_ret_to_btc_12b", "idio_ret_to_btc_48b", "idio_ret_to_btc_288b",
            "dom_btc_change_48b", "dom_btc_change_288b", "dom_btc_z_1d",
            "beta_to_btc_change_5d",  # overrides existing column with PIT-clean version
            "corr_to_btc_1d", "corr_to_btc_change_3d",
            "idio_vol_to_btc_1h", "idio_vol_to_btc_1d", "idio_vol_ratio_to_btc",
            "btc_ret_48b", "btc_realized_vol_1d", "btc_realized_vol_30d",
            "listing_age_days", "log_quote_volume_90d", "residual_vol_90d_own_pctile",
        ]
        df_out = df[new_cols].copy()
        df_out["symbol"] = sym
        df_out = df_out.reset_index().rename(columns={"open_time": "open_time"})
        new_features_per_sym.append(df_out)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(panel_syms)} symbols done ({time.time()-t_start:.0f}s)",
                  flush=True)

    print(f"  total feature engineering: {time.time()-t_start:.0f}s", flush=True)
    all_features = pd.concat(new_features_per_sym, ignore_index=True)
    print(f"  new feature rows: {len(all_features):,}", flush=True)

    # ============================================================
    # Step 3: Merge new features into existing panel (keep existing alpha_A, target_A,
    # return_pct, funding_rate*, atr_pct, obv_z_1d, vwap_slope_96, etc.)
    # ============================================================
    print("\n--- Step 3: merge with existing panel ---", flush=True)
    # Drop columns we're recomputing to avoid conflicts
    cols_to_drop_from_existing = [
        "beta_to_btc_change_5d", "corr_to_btc_1d", "idio_vol_to_btc_1h",
    ]
    panel_kept = panel.drop(columns=[c for c in cols_to_drop_from_existing if c in panel.columns])
    panel_aug = panel_kept.merge(all_features, on=["symbol", "open_time"], how="inner")
    print(f"  merged panel: {len(panel_aug):,} rows × {len(panel_aug.columns)} columns", flush=True)

    # ============================================================
    # Step 4: Build β-residual target with PIT β + locked σ_idio
    # ============================================================
    print("\n--- Step 4: build β-residual target (z-scored, PIT) ---", flush=True)
    # alpha_beta = return_pct (forward 4h) - β_PIT × BTC_return (forward 4h)
    # need BTC's forward 4h return aligned by open_time
    btc_ret_fwd = panel_aug[panel_aug.symbol == "BTCUSDT"][["open_time", "return_pct"]].rename(
        columns={"return_pct": "btc_ret_fwd"}).drop_duplicates("open_time")
    panel_aug = panel_aug.merge(btc_ret_fwd, on="open_time", how="left")
    panel_aug["alpha_beta"] = panel_aug["return_pct"] - panel_aug["beta_btc_pit"] * panel_aug["btc_ret_fwd"]

    # σ_idio from fold-0 training residuals (locked PIT)
    from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice
    folds_all = _multi_oos_splits(panel_aug)
    train0, _, _ = _slice(panel_aug, folds_all[0])
    sigma_idio = train0.groupby("symbol")["alpha_beta"].std().to_dict()
    fallback = panel_aug["alpha_beta"].std()
    panel_aug["sigma_idio_btc"] = panel_aug["symbol"].map(sigma_idio).fillna(fallback).clip(lower=1e-6)
    panel_aug["target_beta_btc"] = panel_aug["alpha_beta"] / panel_aug["sigma_idio_btc"]
    print(f"  target stats: p1={panel_aug['target_beta_btc'].quantile(0.01):.2f}, "
          f"p99={panel_aug['target_beta_btc'].quantile(0.99):.2f}, "
          f"|x|>5: {(panel_aug['target_beta_btc'].abs()>5).sum():,}/{len(panel_aug):,}",
          flush=True)

    # ============================================================
    # Step 5: Save
    # ============================================================
    out_path = OUT_DIR / "panel_btc_only.parquet"
    panel_aug.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(panel_aug):,} rows × {len(panel_aug.columns)} cols)",
          flush=True)

    # ============================================================
    # Step 6: Quick feature sanity check — cross-sectional IC of new features
    # ============================================================
    print("\n--- Step 6: per-cycle IC sanity check on new features ---", flush=True)
    new_feature_list = [
        "idio_ret_to_btc_12b", "idio_ret_to_btc_48b", "idio_ret_to_btc_288b",
        "dom_btc_z_1d", "dom_btc_change_48b", "dom_btc_change_288b",
        "beta_btc_pit", "beta_to_btc_change_5d", "corr_to_btc_1d", "corr_to_btc_change_3d",
        "idio_vol_to_btc_1h", "idio_vol_to_btc_1d", "idio_vol_ratio_to_btc",
        "btc_ret_48b", "btc_realized_vol_1d", "btc_realized_vol_30d",
        "atr_pct", "obv_z_1d", "vwap_slope_96",
        "funding_rate", "funding_rate_z_7d", "funding_rate_1d_change",
        "listing_age_days", "log_quote_volume_90d", "residual_vol_90d_own_pctile",
    ]
    new_feature_list = [f for f in new_feature_list if f in panel_aug.columns]
    print(f"  {len(new_feature_list)} features in panel", flush=True)
    # Sample at entry cadence
    times = sorted(panel_aug["open_time"].unique())
    keep_t = set(times[::48])
    samp = panel_aug[panel_aug["open_time"].isin(keep_t)].dropna(subset=["alpha_beta"])
    rows = []
    for feat in new_feature_list:
        ics = []
        for t, g in samp.dropna(subset=[feat]).groupby("open_time"):
            if len(g) < 10: continue
            ic = g[feat].rank().corr(g["alpha_beta"].rank())
            if not pd.isna(ic): ics.append(ic)
        if ics:
            ics = np.array(ics)
            rows.append({"feature": feat,
                          "mean_ic": float(ics.mean()),
                          "median_ic": float(np.median(ics)),
                          "n_cycles": len(ics)})
    df_ic = pd.DataFrame(rows).sort_values("mean_ic", key=abs, ascending=False)
    pd.set_option("display.width", 200)
    print(df_ic.to_string(index=False, float_format=lambda x: f"{x:+.4f}"), flush=True)
    df_ic.to_csv(OUT_DIR / "new_features_ic_check.csv", index=False)


if __name__ == "__main__":
    main()
