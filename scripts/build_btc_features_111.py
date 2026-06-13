"""Build BTC-frame + V2 features for 111-panel.

Adapts build_btc_only_features.py to the expanded 111-symbol panel.
Output includes:
  - BTC-frame features (corr_to_btc_1d, idio_vol_to_btc_*, dom_btc_*, etc.)
  - V2 extras (return_8h, vol_zscore_4h_over_7d)
  - obv_z_1d (1d OBV z-score)
  - bars_since_high_xs_rank (cross-sectional rank in 111-universe)
  - alpha_beta target (PIT β × forward 4h BTC return)

Saves: outputs/vBTC_features_btc_only_111_pit/panel_btc_only_111.parquet
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

PANEL_111 = REPO / "outputs/vBTC_features_expanded/panel_variants_with_funding.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT_DIR = REPO / "outputs/vBTC_features_btc_only_111_pit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BARS_PER_HOUR = 12
BARS_PER_DAY = 288
BETA_WINDOW = 90 * BARS_PER_DAY


def load_klines(sym):
    sd = KLINES_DIR / sym / "5m"
    if not sd.exists(): return None
    files = sorted(sd.glob("*.parquet"))
    if not files: return None
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["open_time","close","quote_volume"])
            dfs.append(df)
        except Exception:
            try:
                df = pd.read_parquet(f, columns=["open_time","close"])
                df["quote_volume"] = np.nan
                dfs.append(df)
            except Exception: pass
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["open_time"]).drop_duplicates("open_time").sort_values("open_time")
    return df


def compute_pit_beta(sym_ret, btc_ret, window_bars):
    cov = sym_ret.rolling(window_bars, min_periods=1000).cov(btc_ret)
    var = btc_ret.rolling(window_bars, min_periods=1000).var()
    beta = (cov / var.replace(0, np.nan)).shift(1)
    return beta


def main():
    print("=== Build BTC-frame + V2 features for 111-panel ===\n", flush=True)
    t0 = time.time()

    panel = pd.read_parquet(PANEL_111)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel_syms = sorted(panel["symbol"].unique())
    print(f"111-panel: {len(panel):,} rows × {len(panel_syms)} symbols", flush=True)

    # Step 1: BTC kline series
    print("\n--- Step 1: load BTC 5min klines ---", flush=True)
    btc_df = load_klines("BTCUSDT")
    btc_df = btc_df.set_index("open_time")
    btc_df["btc_ret_5m"] = btc_df["close"].pct_change()
    btc_df["btc_ret_48b"] = btc_df["close"].pct_change(48)
    btc_df["btc_realized_vol_1d"] = btc_df["btc_ret_5m"].rolling(BARS_PER_DAY).std()
    print(f"  BTC: {len(btc_df):,} bars from {btc_df.index.min()} to {btc_df.index.max()}",
          flush=True)

    # Step 2: per-symbol feature engineering
    print("\n--- Step 2: per-symbol feature engineering ---", flush=True)
    t_start = time.time()
    new_features_per_sym = []
    skipped = []

    for i, sym in enumerate(panel_syms):
        sym_df = load_klines(sym)
        if sym_df is None or len(sym_df) < 1000:
            skipped.append(sym)
            continue
        sym_df = sym_df.set_index("open_time")
        sym_df["sym_ret_5m"] = sym_df["close"].pct_change()
        df = sym_df.join(btc_df[["close","btc_ret_5m","btc_ret_48b",
                                  "btc_realized_vol_1d"]],
                          how="inner", rsuffix="_btc")

        # PIT β
        beta_pit = compute_pit_beta(df["sym_ret_5m"], df["btc_ret_5m"], BETA_WINDOW)
        df["beta_btc_pit"] = beta_pit

        # BTC residual price level
        df["log_price_ratio"] = np.log(df["close"]) - np.log(df["close_btc"])
        df["dom_btc_change_288b"] = df["log_price_ratio"].diff(288)
        ma_1d = df["log_price_ratio"].rolling(BARS_PER_DAY).mean()
        std_1d = df["log_price_ratio"].rolling(BARS_PER_DAY).std()
        df["dom_btc_z_1d"] = ((df["log_price_ratio"] - ma_1d) / std_1d.replace(0, np.nan))

        # β/corr state
        df["beta_to_btc_change_5d"] = df["beta_btc_pit"].diff(5 * BARS_PER_DAY)
        df["corr_to_btc_1d"] = df["sym_ret_5m"].rolling(BARS_PER_DAY).corr(df["btc_ret_5m"])
        df["corr_to_btc_change_3d"] = df["corr_to_btc_1d"].diff(3 * BARS_PER_DAY)

        # BTC residual risk
        idio_ret_5m = df["sym_ret_5m"] - df["beta_btc_pit"] * df["btc_ret_5m"]
        df["idio_vol_to_btc_1h"] = idio_ret_5m.rolling(BARS_PER_HOUR).std()
        df["idio_vol_to_btc_1d"] = idio_ret_5m.rolling(BARS_PER_DAY).std()

        # OBV z-score 1d (uses quote_volume from klines)
        if "quote_volume" in sym_df.columns:
            # OBV = cumulative signed volume; signed by return direction
            signed_vol = np.sign(df["sym_ret_5m"].fillna(0)) * df.get("quote_volume",
                                                                      pd.Series(np.nan, index=df.index))
            obv = signed_vol.cumsum()
            obv_ma_1d = obv.rolling(BARS_PER_DAY).mean()
            obv_std_1d = obv.rolling(BARS_PER_DAY).std()
            df["obv_z_1d"] = ((obv - obv_ma_1d) / obv_std_1d.replace(0, np.nan))
        else:
            df["obv_z_1d"] = np.nan

        # return_8h = trailing 8h return (96 5m bars) — PIT shift(1) per Step 29
        df["return_8h"] = df["close"].pct_change(96).shift(1)

        # vol_zscore_4h_over_7d — Step 29 convention: shift(BAR_4H + 1) = shift(49)
        if "quote_volume" in sym_df.columns:
            vol_4h = df.get("quote_volume", pd.Series(np.nan, index=df.index)).rolling(48).mean()
            vol_7d_ma = vol_4h.rolling(7 * BARS_PER_DAY).mean()
            vol_7d_std = vol_4h.rolling(7 * BARS_PER_DAY).std()
            df["vol_zscore_4h_over_7d"] = ((vol_4h - vol_7d_ma) / vol_7d_std.replace(0, np.nan)).shift(49)
        else:
            df["vol_zscore_4h_over_7d"] = np.nan

        new_cols = [
            "beta_btc_pit",
            "dom_btc_change_288b", "dom_btc_z_1d",
            "beta_to_btc_change_5d",
            "corr_to_btc_1d", "corr_to_btc_change_3d",
            "idio_vol_to_btc_1h", "idio_vol_to_btc_1d",
            "obv_z_1d", "return_8h", "vol_zscore_4h_over_7d",
        ]
        df_out = df[new_cols].copy()
        df_out["symbol"] = sym
        df_out = df_out.reset_index()
        new_features_per_sym.append(df_out)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(panel_syms)} symbols done ({time.time()-t_start:.0f}s)",
                  flush=True)

    print(f"  total: {time.time()-t_start:.0f}s, skipped: {len(skipped)} ({skipped[:5]})",
          flush=True)

    all_features = pd.concat(new_features_per_sym, ignore_index=True)
    print(f"  new feature rows: {len(all_features):,}", flush=True)

    # Step 3: Merge with existing 111-panel — keep only V2-essential columns to reduce memory
    print("\n--- Step 3: merge with 111-panel (V2-essential cols only) ---", flush=True)
    KEEP_FROM_111 = ["symbol","open_time","return_pct","exit_time","autocorr_pctile_7d",
                     "return_1d","atr_pct","vwap_slope_96",
                     "funding_rate","funding_rate_z_7d","funding_rate_1d_change",
                     "bars_since_high"]
    panel_trimmed = panel[[c for c in KEEP_FROM_111 if c in panel.columns]].copy()
    panel_trimmed["symbol"] = panel_trimmed["symbol"].astype("category")
    all_features["symbol"] = all_features["symbol"].astype("category")
    panel_aug = panel_trimmed.merge(all_features, on=["symbol","open_time"], how="inner")
    del panel_trimmed, all_features
    import gc; gc.collect()
    print(f"  merged: {len(panel_aug):,} rows × {len(panel_aug.columns)} cols", flush=True)

    # Step 4: bars_since_high_xs_rank — cross-sectional within 111-universe
    print("\n--- Step 4: bars_since_high_xs_rank (cross-sectional) ---", flush=True)
    if "bars_since_high" not in panel_aug.columns:
        print("  bars_since_high missing — compute from close prices", flush=True)
        # Per symbol: bars since trailing N-bar high
        def compute_bsh(g):
            close = g["close"] if "close" in g.columns else None
            if close is None: return pd.Series(np.nan, index=g.index)
            # Use rolling argmax over BARS_PER_DAY
            roll_max = close.rolling(BARS_PER_DAY).max()
            is_high = (close == roll_max)
            # Bars since last high
            n = len(g)
            bsh = np.full(n, np.nan)
            last_high_idx = -1
            for k in range(n):
                if is_high.iloc[k]:
                    last_high_idx = k
                if last_high_idx >= 0:
                    bsh[k] = k - last_high_idx
            return pd.Series(bsh, index=g.index)
        # Skip if can't compute — fall back to existing values
        if "close" not in panel_aug.columns:
            print("  no close column — skipping bsh computation, may have NaN", flush=True)
    # Cross-sectional rank within 111-universe at each timestamp
    if "bars_since_high" in panel_aug.columns:
        panel_aug["bars_since_high_xs_rank"] = (
            panel_aug.groupby("open_time")["bars_since_high"].rank(pct=True))
    print(f"  bsh_xs_rank coverage: {panel_aug['bars_since_high_xs_rank'].notna().sum():,}",
          flush=True)

    # Step 5: alpha_beta target — memory-efficient via map (avoid full merge)
    print("\n--- Step 5: build alpha_beta target ---", flush=True)
    btc_ret_fwd_map = (panel_aug[panel_aug.symbol == "BTCUSDT"]
                        .drop_duplicates("open_time")
                        .set_index("open_time")["return_pct"].to_dict())
    print(f"  BTC return_fwd map size: {len(btc_ret_fwd_map):,}", flush=True)
    panel_aug["btc_ret_fwd"] = panel_aug["open_time"].map(btc_ret_fwd_map).astype("float32")
    panel_aug["alpha_beta"] = (panel_aug["return_pct"]
                                - panel_aug["beta_btc_pit"] * panel_aug["btc_ret_fwd"]).astype("float32")
    import gc; gc.collect()

    # σ_idio from fold-0 train (per symbol, frozen)
    from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice
    folds_all = _multi_oos_splits(panel_aug)
    train0, _, _ = _slice(panel_aug, folds_all[0])
    sigma_idio = train0.groupby("symbol")["alpha_beta"].std().to_dict()
    fold0_known = pd.Series(sigma_idio).dropna()
    fallback = float(fold0_known.median()) if len(fold0_known) > 0 else 0.005
    panel_aug["sigma_idio"] = panel_aug["symbol"].map(sigma_idio).fillna(fallback).clip(lower=1e-6)

    # Exclude BTCUSDT from final panel (reviewer finding — BTC has alpha_beta ≈ 0
    # and sigma_idio clipped to 1e-6; should not be in training/eval for BTC-hedged
    # alt strategy)
    n_before = len(panel_aug)
    panel_aug = panel_aug[panel_aug["symbol"] != "BTCUSDT"].copy()
    print(f"\nExcluded BTCUSDT: {n_before - len(panel_aug):,} rows removed", flush=True)

    # Save
    out_path = OUT_DIR / "panel_btc_only_111.parquet"
    panel_aug.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(panel_aug):,} rows × {len(panel_aug.columns)} cols)",
          flush=True)
    print(f"Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
