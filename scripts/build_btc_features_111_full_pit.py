"""Build FULL-PIT BTC-frame + V2 features for 111-input panel.

Adds to strict-PIT: also shifts base OHLCV-derived features copied from the
111-panel by 1 bar per symbol. Without this, return_1d, atr_pct, vwap_slope_96,
and bars_since_high used current-bar close/high/low/volume — leaking bar t's
data into a position opened at open_time t.

Base features SHIFTED per symbol (Step 3 of build):
  - return_1d        = close.pct_change(288)
  - atr_pct          (uses HLC of current bar)
  - vwap_slope_96    (uses current OHLCV)
  - bars_since_high  (uses current close to detect rolling high)
  - bars_since_low   (uses current close to detect rolling low)
  - bars_since_high_xs_rank / bars_since_low_xs_rank computed AFTER shift,
    then cross-sectional ranked

Base features NOT shifted (knew-before-bar-open):
  - funding_rate, funding_rate_z_7d, funding_rate_1d_change
    (Binance funding rate at t is announced before window starts)
  - return_pct, exit_time (target side; needed downstream as-is for sleeve MTM)
  - autocorr_pctile_7d (auxiliary control flag, used as filter not feature)

Other differences (inherited from strict-PIT version):
  1. beta_btc_pit: .shift(49) matching 01_build_target.py
  2. All rolling-ending-at-current BTC-frame features: .shift(1)
     (corr_to_btc_1d, idio_vol_to_btc_{1h,1d}, dom_btc_z_1d, dom_btc_change_288b,
      obv_z_1d, btc_realized_vol_1d, btc_ret_48b)
  3. corr_to_btc_change_3d / beta_to_btc_change_5d inherit shifts from their bases
  4. return_8h .shift(1), vol_zscore_4h_over_7d .shift(49) per Step 29
  5. BTCUSDT excluded from output

Saves: outputs/vBTC_features_btc_only_111_full_pit/panel_btc_only_111.parquet
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
OUT_DIR = REPO / "outputs/vBTC_features_btc_only_111_full_pit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BARS_PER_HOUR = 12
BARS_PER_DAY = 288
BETA_WINDOW = 90 * BARS_PER_DAY
HORIZON_BARS = 48
BETA_SHIFT = HORIZON_BARS + 1  # 49 — strict PIT matching 01_build_target.py


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


def bars_since_extreme(close: pd.Series, *, low: bool = False) -> pd.Series:
    """Bars since the trailing 1d high/low, shifted so row t uses bars <= t-1."""
    picker = np.argmin if low else np.argmax
    return close.rolling(BARS_PER_DAY).apply(
        lambda w: len(w) - 1 - int(picker(w)), raw=True).shift(1)


def compute_pit_beta_strict(sym_ret, btc_ret, window_bars):
    """Strict PIT rolling β — shift by HORIZON+1 = 49 bars to push window fully
    behind the forward 48-bar target window. Matches 01_build_target.py."""
    cov = sym_ret.rolling(window_bars, min_periods=1000).cov(btc_ret)
    var = btc_ret.rolling(window_bars, min_periods=1000).var()
    beta = (cov / var.replace(0, np.nan)).shift(BETA_SHIFT)
    return beta


def main():
    print("=== Build FULL-PIT BTC-frame + V2 features for 111-input panel ===\n",
          flush=True)
    print(f"  beta shift: {BETA_SHIFT} (was 1 in non-strict version)", flush=True)
    print(f"  rolling features shifted by 1 (window ends at bar t-1, not t)\n", flush=True)
    t0 = time.time()

    panel = pd.read_parquet(PANEL_111)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel_syms = sorted(panel["symbol"].unique())
    print(f"111-input panel: {len(panel):,} rows × {len(panel_syms)} symbols",
          flush=True)

    # Step 1: BTC kline series
    print("\n--- Step 1: load BTC 5min klines ---", flush=True)
    btc_df = load_klines("BTCUSDT")
    btc_df = btc_df.set_index("open_time")
    btc_df["btc_ret_5m"] = btc_df["close"].pct_change()
    btc_df["btc_ret_48b"] = btc_df["close"].pct_change(48).shift(1)  # SHIFTED
    btc_df["btc_realized_vol_1d"] = btc_df["btc_ret_5m"].rolling(BARS_PER_DAY).std().shift(1)  # SHIFTED
    print(f"  BTC: {len(btc_df):,} bars from {btc_df.index.min()} to {btc_df.index.max()}",
          flush=True)

    # Step 2: per-symbol feature engineering
    print("\n--- Step 2: per-symbol feature engineering (FULL PIT) ---", flush=True)
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

        # FULL PIT β: shift by 49 bars
        beta_pit = compute_pit_beta_strict(df["sym_ret_5m"], df["btc_ret_5m"],
                                            BETA_WINDOW)
        df["beta_btc_pit"] = beta_pit

        # BTC residual price level — shift by 1 so endpoint is bar t-1
        df["log_price_ratio"] = np.log(df["close"]) - np.log(df["close_btc"])
        df["dom_btc_change_288b"] = df["log_price_ratio"].diff(288).shift(1)  # SHIFTED
        ma_1d = df["log_price_ratio"].rolling(BARS_PER_DAY).mean()
        std_1d = df["log_price_ratio"].rolling(BARS_PER_DAY).std()
        df["dom_btc_z_1d"] = (((df["log_price_ratio"] - ma_1d)
                                / std_1d.replace(0, np.nan))).shift(1)  # SHIFTED

        # β/corr state — corr/diff need shifting
        df["beta_to_btc_change_5d"] = df["beta_btc_pit"].diff(5 * BARS_PER_DAY)
        # beta_btc_pit already shifted by 49, so beta_to_btc_change_5d is OK
        df["corr_to_btc_1d"] = (df["sym_ret_5m"].rolling(BARS_PER_DAY)
                                   .corr(df["btc_ret_5m"])).shift(1)  # SHIFTED
        df["corr_to_btc_change_3d"] = df["corr_to_btc_1d"].diff(3 * BARS_PER_DAY)
        # corr_to_btc_1d already shifted, so the diff is OK

        # BTC residual risk — rolling.std() with shift(1)
        # Note: idio_ret_5m uses beta_btc_pit (shifted 49) and sym_ret_5m/btc_ret_5m
        # at row t. To get strict PIT vol-at-open_time-t, shift the result by 1.
        idio_ret_5m = df["sym_ret_5m"] - df["beta_btc_pit"] * df["btc_ret_5m"]
        df["idio_vol_to_btc_1h"] = (idio_ret_5m.rolling(BARS_PER_HOUR)
                                       .std()).shift(1)  # SHIFTED
        df["idio_vol_to_btc_1d"] = (idio_ret_5m.rolling(BARS_PER_DAY)
                                       .std()).shift(1)  # SHIFTED

        # OBV z-score 1d — rolling z, shift by 1
        if "quote_volume" in sym_df.columns:
            signed_vol = (np.sign(df["sym_ret_5m"].fillna(0))
                            * df.get("quote_volume", pd.Series(np.nan, index=df.index)))
            obv = signed_vol.cumsum()
            obv_ma_1d = obv.rolling(BARS_PER_DAY).mean()
            obv_std_1d = obv.rolling(BARS_PER_DAY).std()
            df["obv_z_1d"] = (((obv - obv_ma_1d)
                                / obv_std_1d.replace(0, np.nan))).shift(1)  # SHIFTED
        else:
            df["obv_z_1d"] = np.nan

        # return_8h: trailing 8h return (96 5m bars) — Step 29 convention shift(1)
        df["return_8h"] = df["close"].pct_change(96).shift(1)

        # Symmetric distance-from-extreme feature. `bars_since_high` is carried
        # from the source panel and shifted in Step 3; this low-side variant is
        # computed directly from the same 5m close series with the shift applied.
        df["bars_since_low"] = bars_since_extreme(df["close"], low=True)

        # vol_zscore_4h_over_7d — Step 29 convention shift(49)
        if "quote_volume" in sym_df.columns:
            vol_4h = df.get("quote_volume", pd.Series(np.nan, index=df.index)).rolling(48).mean()
            vol_7d_ma = vol_4h.rolling(7 * BARS_PER_DAY).mean()
            vol_7d_std = vol_4h.rolling(7 * BARS_PER_DAY).std()
            df["vol_zscore_4h_over_7d"] = (((vol_4h - vol_7d_ma)
                                              / vol_7d_std.replace(0, np.nan))).shift(49)
        else:
            df["vol_zscore_4h_over_7d"] = np.nan

        new_cols = [
            "beta_btc_pit",
            "dom_btc_change_288b", "dom_btc_z_1d",
            "beta_to_btc_change_5d",
            "corr_to_btc_1d", "corr_to_btc_change_3d",
            "idio_vol_to_btc_1h", "idio_vol_to_btc_1d",
            "obv_z_1d", "return_8h", "vol_zscore_4h_over_7d",
            "bars_since_low",
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

    # Step 3: Merge with 111-panel (V2-essential cols only), then shift OHLCV-derived
    # base features by 1 bar per symbol (reviewer-identified leak: stored values use
    # current-bar close/high/low/volume; under open_time t execution we should only
    # see bar t-1's close).
    print("\n--- Step 3: merge with 111-panel + shift OHLCV-derived base features ---",
          flush=True)
    KEEP_FROM_111 = ["symbol","open_time","return_pct","exit_time","autocorr_pctile_7d",
                      "return_1d","atr_pct","vwap_slope_96",
                      "funding_rate","funding_rate_z_7d","funding_rate_1d_change",
                      "bars_since_high"]
    panel_trimmed = panel[[c for c in KEEP_FROM_111 if c in panel.columns]].copy()

    # Shift OHLCV-derived base features by 1 row per symbol (preserve return_pct,
    # exit_time, autocorr_pctile_7d, funding_*; those are not OHLCV-leaky)
    SHIFT_BASE_FEATURES = ["return_1d", "atr_pct", "vwap_slope_96", "bars_since_high"]
    panel_trimmed = panel_trimmed.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    for f in SHIFT_BASE_FEATURES:
        if f in panel_trimmed.columns:
            panel_trimmed[f] = panel_trimmed.groupby("symbol")[f].shift(1)
    print(f"  shifted base features by 1 row per symbol: {SHIFT_BASE_FEATURES}",
          flush=True)

    panel_trimmed["symbol"] = panel_trimmed["symbol"].astype("category")
    all_features["symbol"] = all_features["symbol"].astype("category")
    panel_aug = panel_trimmed.merge(all_features, on=["symbol","open_time"], how="inner")
    del panel_trimmed, all_features
    import gc; gc.collect()
    print(f"  merged: {len(panel_aug):,} rows × {len(panel_aug.columns)} cols",
          flush=True)

    # Step 4: bars_since_*_xs_rank — cross-sectional ranks using PIT-shifted
    # distance-from-extreme features.
    print("\n--- Step 4: bars_since_*_xs_rank (cross-sectional, post-shift) ---",
          flush=True)
    if "bars_since_high" in panel_aug.columns:
        panel_aug["bars_since_high_xs_rank"] = (
            panel_aug.groupby("open_time")["bars_since_high"].rank(pct=True))
    if "bars_since_low" in panel_aug.columns:
        panel_aug["bars_since_low_xs_rank"] = (
            panel_aug.groupby("open_time")["bars_since_low"].rank(pct=True))
    print(f"  bsh_xs_rank coverage: {panel_aug['bars_since_high_xs_rank'].notna().sum():,}",
          flush=True)
    print(f"  bsl_xs_rank coverage: {panel_aug['bars_since_low_xs_rank'].notna().sum():,}",
          flush=True)

    # Step 5: alpha_beta target — strict PIT beta × forward return
    print("\n--- Step 5: build alpha_beta target (FULL PIT) ---", flush=True)
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

    # Exclude BTCUSDT
    n_before = len(panel_aug)
    panel_aug = panel_aug[panel_aug["symbol"] != "BTCUSDT"].copy()
    print(f"\nExcluded BTCUSDT: {n_before - len(panel_aug):,} rows removed", flush=True)

    out_path = OUT_DIR / "panel_btc_only_111.parquet"
    panel_aug.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(panel_aug):,} rows × {len(panel_aug.columns)} cols)",
          flush=True)
    print(f"Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
