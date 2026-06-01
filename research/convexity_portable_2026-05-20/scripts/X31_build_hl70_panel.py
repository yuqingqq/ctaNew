"""X31 — Build HL-70 panel = existing HL-50 (panel_v2) + 20 missing HL syms.

For the 20 missing syms, computes:
  - BASE features (return_1d, atr_pct, obv_z_1d, vwap_slope_96, bars_since_high,
    autocorr_pctile_7d, BTC-cross features)
  - Funding features (from funding_<sym>.parquet cache)
  - Cohort features (rebuilt fresh via build_cohort_fixed)
  - Target alpha_vs_btc_realized = my_fwd_return - β × BTC_fwd_return

Leverages cached xs_feats_<sym>.parquet for most BASE features.

Output: outputs/vBTC_features/panel_hl70.parquet
"""
from __future__ import annotations
import sys, time, warnings, gc, resource
from pathlib import Path
import pandas as pd, numpy as np

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

KLINES_DIR = REPO / "data/ml/test/parquet/klines"
HORIZON = 48  # 4h ahead (48 × 5min)


def log_mem(label=""):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  [MEM {label}] peak_rss={rss_mb:.0f}MB", flush=True)


def load_btc_closes():
    """Load BTC 5m closes from kline cache."""
    sd = KLINES_DIR / "BTCUSDT" / "5m"
    dfs = []
    for f in sorted(sd.glob("*.parquet")):
        try: dfs.append(pd.read_parquet(f, columns=["open_time", "close"]))
        except Exception: pass
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.set_index("open_time").sort_index()
    return df["close"].astype(np.float32)


def compute_btc_cross_features(my_close, btc_close):
    """Compute corr_to_btc_1d, beta_to_btc_change_5d, idio_vol_to_btc_1h/1d
    for a single symbol vs BTC."""
    # Normalize index precision (both to ns UTC)
    my_close = my_close.copy()
    btc_close = btc_close.copy()
    my_close.index = pd.DatetimeIndex(my_close.index).tz_convert("UTC").astype("datetime64[ns, UTC]")
    btc_close.index = pd.DatetimeIndex(btc_close.index).tz_convert("UTC").astype("datetime64[ns, UTC]")
    my_ret = np.log(my_close / my_close.shift(1))
    btc_ret = np.log(btc_close / btc_close.shift(1))
    common_idx = my_ret.index.intersection(btc_ret.index)
    my_ret = my_ret.reindex(common_idx)
    btc_ret = btc_ret.reindex(common_idx)

    # corr_to_btc_1d: rolling 1-day Pearson correlation (288 bars)
    corr_1d = my_ret.rolling(288, min_periods=72).corr(btc_ret).shift(1)

    # beta_to_btc: rolling 1d cov/var
    cov_1d = my_ret.rolling(288, min_periods=72).cov(btc_ret)
    var_1d = btc_ret.rolling(288, min_periods=72).var()
    beta_1d = (cov_1d / var_1d.replace(0, np.nan)).shift(1)
    beta_change_5d = beta_1d.diff(288 * 5)

    # idio vol to btc: residual return vol
    idio_ret = my_ret - beta_1d * btc_ret
    idio_vol_1h = idio_ret.rolling(12, min_periods=6).std().shift(1)
    idio_vol_1d = idio_ret.rolling(288, min_periods=72).std().shift(1)

    return pd.DataFrame({
        "corr_to_btc_1d": corr_1d.astype(np.float32),
        "beta_to_btc_change_5d": beta_change_5d.astype(np.float32),
        "idio_vol_to_btc_1h": idio_vol_1h.astype(np.float32),
        "idio_vol_to_btc_1d": idio_vol_1d.astype(np.float32),
    })


def compute_target_alpha_vs_btc(my_close, btc_close):
    """Compute alpha_vs_btc_realized = my_fwd_return - β × BTC_fwd_return.
    Same definition as in 51-panel."""
    my_close = my_close.copy()
    btc_close = btc_close.copy()
    my_close.index = pd.DatetimeIndex(my_close.index).tz_convert("UTC").astype("datetime64[ns, UTC]")
    btc_close.index = pd.DatetimeIndex(btc_close.index).tz_convert("UTC").astype("datetime64[ns, UTC]")
    my_ret = np.log(my_close / my_close.shift(1))
    btc_ret = np.log(btc_close / btc_close.shift(1))
    common_idx = my_ret.index.intersection(btc_ret.index)
    my_ret = my_ret.reindex(common_idx)
    btc_ret = btc_ret.reindex(common_idx)

    # Forward returns over 48 5m bars
    my_close_aligned = my_close.reindex(common_idx)
    btc_close_aligned = btc_close.reindex(common_idx)
    my_fwd = (my_close_aligned.shift(-HORIZON) / my_close_aligned - 1).astype(np.float32)
    btc_fwd = (btc_close_aligned.shift(-HORIZON) / btc_close_aligned - 1).astype(np.float32)

    # Beta (rolling 1d, shifted 1 for PIT)
    cov_1d = my_ret.rolling(288, min_periods=72).cov(btc_ret)
    var_1d = btc_ret.rolling(288, min_periods=72).var()
    beta_1d = (cov_1d / var_1d.replace(0, np.nan)).shift(1)

    # alpha = my_fwd - β × btc_fwd
    alpha = my_fwd - beta_1d * btc_fwd
    return alpha.astype(np.float32), my_fwd, btc_fwd


def build_sym_minirow(sym, btc_close):
    """Build minimum panel row set for one new sym."""
    # Load xs_feats (BASE features that don't depend on BTC)
    xs_path = REPO / f"data/ml/cache/xs_feats_{sym}.parquet"
    if not xs_path.exists():
        print(f"  [{sym}] xs_feats missing, skipping")
        return None
    xs = pd.read_parquet(xs_path)
    # Normalize index precision to ns UTC
    xs.index = pd.DatetimeIndex(xs.index).tz_convert("UTC").astype("datetime64[ns, UTC]")

    # Load this sym's 5m closes
    sd = KLINES_DIR / sym / "5m"
    if not sd.exists():
        print(f"  [{sym}] klines missing, skipping")
        return None
    dfs = []
    for f in sorted(sd.glob("*.parquet")):
        try: dfs.append(pd.read_parquet(f, columns=["open_time", "close"]))
        except Exception: pass
    if not dfs:
        print(f"  [{sym}] no klines parquets")
        return None
    klines = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    klines["open_time"] = pd.to_datetime(klines["open_time"], utc=True)
    klines = klines.set_index("open_time").sort_index()
    my_close = klines["close"].astype(np.float32)

    # BTC-cross features
    btc_cross = compute_btc_cross_features(my_close, btc_close)

    # Target alpha_vs_btc_realized + return_pct
    alpha_vs_btc, my_fwd, _ = compute_target_alpha_vs_btc(my_close, btc_close)

    # obv_z_1d from obv_signal (z over 288 bars)
    obv = xs["obv_signal"] if "obv_signal" in xs.columns else None
    if obv is not None:
        obv_z = ((obv - obv.rolling(288, min_periods=72).mean()) /
                  obv.rolling(288, min_periods=72).std().replace(0, np.nan)).shift(1).astype(np.float32)
    else:
        obv_z = pd.Series(np.nan, index=xs.index, dtype=np.float32)

    # Load funding (calc_time column has the timestamps)
    fund_path = REPO / f"data/ml/cache/funding_{sym}.parquet"
    fr = None
    if fund_path.exists():
        fund = pd.read_parquet(fund_path)
        ts_col = "calc_time" if "calc_time" in fund.columns else "open_time"
        if ts_col in fund.columns:
            fund[ts_col] = pd.to_datetime(fund[ts_col], utc=True).astype("datetime64[ns, UTC]")
            fund = fund.set_index(ts_col).sort_index()
            # Drop duplicates by index
            fund = fund[~fund.index.duplicated(keep="last")]
        if "funding_rate" in fund.columns:
            fr = fund["funding_rate"].reindex(xs.index, method="ffill")
    if fr is not None:
        fr_z_7d = ((fr - fr.rolling(288 * 7, min_periods=288).mean()) /
                    fr.rolling(288 * 7, min_periods=288).std().replace(0, np.nan)).shift(1).astype(np.float32)
        fr_1d_chg = fr.diff(288).shift(1).astype(np.float32)
        fr = fr.shift(1).astype(np.float32)
    else:
        fr = pd.Series(np.nan, index=xs.index, dtype=np.float32)
        fr_z_7d = pd.Series(np.nan, index=xs.index, dtype=np.float32)
        fr_1d_chg = pd.Series(np.nan, index=xs.index, dtype=np.float32)

    # Build panel-row-like dataframe
    out = pd.DataFrame({
        "symbol": sym,
        "open_time": xs.index,
        "return_pct": my_fwd.reindex(xs.index),  # forward return
        "exit_time": xs.index + pd.Timedelta(minutes=5 * HORIZON),
        "alpha_vs_btc_realized": alpha_vs_btc.reindex(xs.index),
        # BASE features from xs_feats
        "return_1d": xs.get("return_1d", pd.Series(np.nan, index=xs.index)).astype(np.float32),
        "atr_pct": xs.get("atr_pct", pd.Series(np.nan, index=xs.index)).astype(np.float32),
        "vwap_slope_96": xs.get("vwap_slope_96", pd.Series(np.nan, index=xs.index)).astype(np.float32),
        "bars_since_high": xs.get("bars_since_high", pd.Series(np.nan, index=xs.index)).astype(np.float32),
        "autocorr_pctile_7d": xs.get("autocorr_pctile_7d", pd.Series(np.nan, index=xs.index)).astype(np.float32),
        "obv_z_1d": obv_z,
        # BTC-cross
        "corr_to_btc_1d": btc_cross["corr_to_btc_1d"].reindex(xs.index),
        "beta_to_btc_change_5d": btc_cross["beta_to_btc_change_5d"].reindex(xs.index),
        "idio_vol_to_btc_1h": btc_cross["idio_vol_to_btc_1h"].reindex(xs.index),
        "idio_vol_to_btc_1d": btc_cross["idio_vol_to_btc_1d"].reindex(xs.index),
        # Funding
        "funding_rate": fr,
        "funding_rate_z_7d": fr_z_7d,
        "funding_rate_1d_change": fr_1d_chg,
    })
    out = out.reset_index(drop=True)
    return out


def main():
    t0 = time.time()
    print("=== X31 build HL-70 panel (existing 50 + 20 new) ===\n", flush=True)
    log_mem("start")

    # Identify missing HL syms
    hl_map = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
    hl70 = set(hl_map[hl_map.on_hl]["symbol"].tolist())
    existing_panel_syms = set(pd.read_parquet(
        REPO / "outputs/vBTC_features/panel_variants_with_funding_v2.parquet",
        columns=["symbol"])["symbol"].unique())
    missing = sorted(hl70 - existing_panel_syms)
    print(f"  HL-70 total: {len(hl70)}, missing from panel_v2: {len(missing)}")

    # Load BTC closes
    print(f"\n--- Load BTC closes ---")
    btc_close = load_btc_closes()
    print(f"  BTC closes: {len(btc_close):,} bars")
    log_mem("after_btc")

    # Build mini-panel for each new sym
    print(f"\n--- Build features for {len(missing)} new syms ---")
    sym_dfs = []
    for i, sym in enumerate(missing, 1):
        tf = time.time()
        sym_df = build_sym_minirow(sym, btc_close)
        if sym_df is None: continue
        print(f"  [{i}/{len(missing)}] {sym}: {len(sym_df):,} rows [{time.time()-tf:.0f}s]", flush=True)
        sym_dfs.append(sym_df)
        if i % 5 == 0: log_mem(f"after {i}")
    new_panel = pd.concat(sym_dfs, ignore_index=True)
    del sym_dfs; gc.collect()
    print(f"\n  New panel: {len(new_panel):,} rows × {new_panel.shape[1]} cols")
    log_mem("after_new_panel")

    # Load existing panel_v2 and select compatible cols
    print(f"\n--- Concat with existing panel_v2 ---")
    needed_cols = ["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"]
    # Get x6.BASE features
    base_cols = ["return_1d", "atr_pct", "obv_z_1d", "vwap_slope_96",
                  "bars_since_high", "autocorr_pctile_7d",
                  "corr_to_btc_1d", "beta_to_btc_change_5d",
                  "idio_vol_to_btc_1h", "idio_vol_to_btc_1d",
                  "funding_rate", "funding_rate_z_7d", "funding_rate_1d_change"]
    existing_panel = pd.read_parquet(
        REPO / "outputs/vBTC_features/panel_variants_with_funding_v2.parquet",
        columns=needed_cols + base_cols)
    existing_panel["open_time"] = pd.to_datetime(existing_panel["open_time"], utc=True)
    existing_panel["exit_time"] = pd.to_datetime(existing_panel["exit_time"], utc=True)
    print(f"  existing panel_v2: {len(existing_panel):,} rows × {existing_panel.shape[1]} cols")

    # Align new_panel cols to existing_panel cols
    for c in existing_panel.columns:
        if c not in new_panel.columns:
            new_panel[c] = np.nan
    new_panel = new_panel[existing_panel.columns]

    combined = pd.concat([existing_panel, new_panel], ignore_index=True)
    combined = combined.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    print(f"\n  Combined HL-70 panel: {len(combined):,} rows × {combined.shape[1]} cols")
    print(f"  syms: {combined['symbol'].nunique()}")

    out_path = REPO / "outputs/vBTC_features/panel_hl70.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path} ({out_path.stat().st_size/1e6:.0f}MB) [{time.time()-t0:.0f}s]")
    log_mem("end")


if __name__ == "__main__":
    main()
