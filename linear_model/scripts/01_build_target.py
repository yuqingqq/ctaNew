"""Step 1: Build z-scored target with σ_idio frozen for inference recovery.

Pipeline:
  α_β = return_pct - β_pit × btc_ret_t
  σ_idio per symbol = std(α_β) over fold-0 training rows (FROZEN)
  target_z = α_β / σ_idio          ← LGBM/Ridge trains on this (mean~0, std~1)

At inference (Step 4 backtest):
  pred_bps = pred_z × σ_idio[symbol] × 1e4

Gate (Step 4): trade only if |pred_bps| > threshold (sweep {0, 4.5, 9, 15, 25}).

Winsorize target_z at fold-0 ±5σ to limit extreme-label leverage on Ridge MSE
(while keeping ~99.99% of rows unclipped).

Outputs:
  data/targets.parquet      symbol, open_time, alpha_beta, target_z,
                            target_bps_raw, sigma_idio_ref, beta_pit, exit_time
  data/sigma_idio.csv       per-symbol σ_idio (51 rows)
  data/beta_pit.parquet     PIT β series
  data/target_meta.csv      run metadata
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice

PANEL_BASE = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
OUT_DIR    = REPO / "linear_model/data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BAR_PER_DAY    = 288
BETA_WIN_DAYS  = 90
BETA_WIN_BARS  = BETA_WIN_DAYS * BAR_PER_DAY
HORIZON_BARS   = 48                    # target is 4h-forward return (= 48 5m bars)
BETA_SHIFT     = HORIZON_BARS + 1      # = 49 — pushes rolling window fully into the past
MIN_WARMUP_BARS = 14 * BAR_PER_DAY     # 14 days warmup before β is trusted
WINSORIZE_Z    = 5.0                   # clip target_z at ±5σ

# NOTE: production pipeline (diag_winner17_beta_residual_51_vs_111.py) uses .shift(1),
# which embeds 47 bars of forward-window prices in β. Empirically verified return_pct
# correlates +1.0 with forward 4h return. We use .shift(49) here for strict PIT.
# This means linear-model β differs slightly from production β; baseline comparison
# in Step 5 should either accept the small inconsistency or re-run LGBM with shift(49).


def compute_pit_beta(panel):
    btc_ret = panel[panel.symbol == "BTCUSDT"][["open_time", "return_pct"]].rename(
        columns={"return_pct": "btc_ret"}).drop_duplicates("open_time")
    out = []
    for sym, g in panel.groupby("symbol"):
        gg = g[["open_time", "return_pct"]].merge(btc_ret, on="open_time", how="left")
        gg = gg.sort_values("open_time").reset_index(drop=True)
        if sym == "BTCUSDT":
            gg["beta_pit"] = 1.0
        else:
            y = gg["return_pct"]; x = gg["btc_ret"]
            cov = y.rolling(BETA_WIN_BARS, min_periods=MIN_WARMUP_BARS).cov(x)
            var = x.rolling(BETA_WIN_BARS, min_periods=MIN_WARMUP_BARS).var()
            # shift(49) = HORIZON+1: push window fully into the past so β at time t
            # uses ONLY close prices through close[t-2] (no overlap with target window)
            gg["beta_pit"] = (cov / var.replace(0, np.nan)).shift(BETA_SHIFT)
        gg["symbol"] = sym
        out.append(gg)
    return pd.concat(out, ignore_index=True)[["symbol", "open_time", "beta_pit"]]


def main():
    print("=== Step 1: Build z-scored target + frozen σ_idio ===\n", flush=True)
    t0 = time.time()
    panel = pd.read_parquet(PANEL_BASE, columns=["symbol", "open_time",
                                                  "return_pct", "exit_time",
                                                  "autocorr_pctile_7d"])
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    print(f"  Loaded base panel: {len(panel):,} rows × {panel['symbol'].nunique()} symbols",
          flush=True)

    # 1a: rolling 90d β
    print(f"\n  Computing PIT β ({BETA_WIN_DAYS}d × {BAR_PER_DAY} bar window)...",
          flush=True)
    pit_beta = compute_pit_beta(panel)
    print(f"    {pit_beta['beta_pit'].notna().sum():,} non-NaN β values "
          f"({time.time()-t0:.0f}s)", flush=True)
    pit_beta.to_parquet(OUT_DIR / "beta_pit.parquet", index=False)

    # 1b: compute α_β
    print("\n  Computing α_β = return_pct - β_pit × btc_ret_t ...", flush=True)
    panel = panel.merge(pit_beta, on=["symbol", "open_time"], how="left")
    btc_ret_t = panel[panel.symbol == "BTCUSDT"][["open_time", "return_pct"]].rename(
        columns={"return_pct": "btc_ret_t"}).drop_duplicates("open_time")
    panel = panel.merge(btc_ret_t, on="open_time", how="left")
    panel["alpha_beta"] = panel["return_pct"] - panel["beta_pit"] * panel["btc_ret_t"]
    panel["target_bps_raw"] = panel["alpha_beta"] * 1e4

    # 1c: fold-0 training σ_idio per symbol (FROZEN — used for both train target and inference)
    folds_all = _multi_oos_splits(panel)
    train0, _, _ = _slice(panel, folds_all[0])
    print(f"\n  fold-0 train: {len(train0):,} rows "
          f"({train0.open_time.min()} → {train0.open_time.max()})", flush=True)

    sigma_idio = train0.groupby("symbol")["alpha_beta"].std()
    # PIT-clean fallback: cross-symbol MEDIAN of fold-0 σ_idio values.
    # Previously used panel["alpha_beta"].std() which is full-panel including OOS — LEAK.
    # For symbols missing from fold-0 train (HYPE listed 2025-06-03 after fold-0 end,
    # ASTER 2025-09-23), use the cross-symbol median computed on FOLD-0 ONLY.
    fold0_known = sigma_idio.dropna()
    fallback = float(fold0_known.median())
    print(f"\n  PIT-clean fallback (cross-symbol median of fold-0 σ): "
          f"{fallback:.5f} ({fallback*1e4:.0f} bps)", flush=True)
    n_missing = sigma_idio.isna().sum() + len(set(panel.symbol.unique()) - set(sigma_idio.index))
    missing_syms = [s for s in panel.symbol.unique() if s not in fold0_known.index]
    print(f"  Symbols using fallback: {missing_syms}", flush=True)
    all_syms = sorted(panel.symbol.unique())
    sigma_idio = sigma_idio.reindex(all_syms).fillna(fallback).clip(lower=1e-6)
    print(f"\n  σ_idio per symbol (51 values, frozen from fold-0):", flush=True)
    print(f"    median: {sigma_idio.median():.5f} ({sigma_idio.median()*1e4:.0f} bps)",
          flush=True)
    print(f"    min:    {sigma_idio.min():.5f} ({sigma_idio.min()*1e4:.0f} bps) "
          f"({sigma_idio.idxmin()})", flush=True)
    print(f"    max:    {sigma_idio.max():.5f} ({sigma_idio.max()*1e4:.0f} bps) "
          f"({sigma_idio.idxmax()})", flush=True)

    sigma_idio.rename("sigma_idio").to_csv(OUT_DIR / "sigma_idio.csv")

    panel["sigma_idio_ref"] = panel["symbol"].map(sigma_idio.to_dict())
    panel["target_z"] = panel["alpha_beta"] / panel["sigma_idio_ref"]

    # 1d: distribution check
    train_z = panel.loc[panel.index.isin(train0.index), "target_z"].dropna()
    print(f"\n  target_z on fold-0 train (should be ~N(0,1) per symbol):", flush=True)
    print(f"    mean      = {train_z.mean():+.4f}", flush=True)
    print(f"    std       = {train_z.std():.4f}", flush=True)
    print(f"    p1/p99    = [{train_z.quantile(0.01):+.2f}, "
          f"{train_z.quantile(0.99):+.2f}]", flush=True)
    print(f"    p0.1/p99.9 = [{train_z.quantile(0.001):+.2f}, "
          f"{train_z.quantile(0.999):+.2f}]", flush=True)
    print(f"    min/max   = [{train_z.min():+.2f}, {train_z.max():+.2f}]", flush=True)
    print(f"    |z|>5    : {(train_z.abs() > 5).sum():,} rows "
          f"({(train_z.abs() > 5).mean()*100:.3f}%)", flush=True)

    # 1e: winsorize target_z at ±5σ for Ridge MSE stability
    panel["target_z"] = panel["target_z"].clip(lower=-WINSORIZE_Z, upper=WINSORIZE_Z)
    n_clip = (panel["alpha_beta"] / panel["sigma_idio_ref"]).abs().gt(WINSORIZE_Z).sum()
    print(f"\n  Winsorize target_z at ±{WINSORIZE_Z}σ", flush=True)
    print(f"    rows clipped: {n_clip:,} ({n_clip/len(panel)*100:.3f}%)", flush=True)

    # 1f: drop BTC rows (not a trade target — α_β ≡ 0 by construction would bias training)
    n_before = len(panel)
    panel_out = panel[panel["symbol"] != "BTCUSDT"].copy()
    n_dropped_btc = n_before - len(panel_out)
    print(f"\n  Dropped BTCUSDT rows (not a trade target): {n_dropped_btc:,}", flush=True)

    # 1g: save
    out_cols = ["symbol", "open_time", "exit_time",
                "return_pct", "alpha_beta", "target_bps_raw", "target_z",
                "sigma_idio_ref", "beta_pit", "autocorr_pctile_7d"]
    panel_out[out_cols].to_parquet(OUT_DIR / "targets.parquet", index=False)

    pd.DataFrame([{
        "fold0_train_start": train0.open_time.min(),
        "fold0_train_end":   train0.open_time.max(),
        "fold0_train_rows":  len(train0),
        "n_symbols":         panel.symbol.nunique(),
        "n_rows_total":      len(panel),
        "n_alpha_beta_valid": int(panel["alpha_beta"].notna().sum()),
        "n_target_clipped":  int(n_clip),
        "winsorize_z":       WINSORIZE_Z,
        "beta_win_days":     BETA_WIN_DAYS,
        "sigma_idio_median": float(sigma_idio.median()),
        "sigma_idio_min":    float(sigma_idio.min()),
        "sigma_idio_max":    float(sigma_idio.max()),
    }]).to_csv(OUT_DIR / "target_meta.csv", index=False)

    print(f"\n  Saved:", flush=True)
    print(f"    {OUT_DIR / 'targets.parquet'}", flush=True)
    print(f"    {OUT_DIR / 'beta_pit.parquet'}", flush=True)
    print(f"    {OUT_DIR / 'sigma_idio.csv'}", flush=True)
    print(f"    {OUT_DIR / 'target_meta.csv'}", flush=True)
    print(f"\n  Total time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
