"""Per-symbol model diagnostic: train AVAXUSDT-only model, compare to universal predictions.

Tests whether per-symbol architecture solves the universal-model contamination problem
identified on the 111-panel.

Setup:
  - Symbol: AVAXUSDT (mature alt with positive IC under 51-model)
  - Features: WINNER_17 (same as universal model)
  - Target: target_beta (z-scored β-residual)
  - Folds: same _multi_oos_splits as universal, but training data restricted to AVAX rows only
  - Seeds: same SEEDS

Compare per-cycle IC for AVAXUSDT across:
  1. Universal 51-model (WINNER_17 trained on 51-panel)
  2. Universal 111-model (WINNER_17 trained on 111-panel)
  3. Per-symbol AVAX-on-51-panel model (this test)
  4. Per-symbol AVAX-on-111-panel-data model (same AVAX rows from 111-panel, isolated)

If per-symbol IC matches or beats universal on 51-panel AND is stable when moving to
111-panel data (same AVAX rows, no cross-symbol contamination), per-symbol architecture
is validated.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train

PANEL_51 = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
PANEL_111 = REPO / "outputs/vBTC_features_expanded/panel_variants_with_funding.parquet"
APD_UNI_51 = REPO / "outputs/vBTC_winner17_b_residual/51-panel_predictions.parquet"
APD_UNI_111 = REPO / "outputs/vBTC_winner17_b_residual/111-panel_predictions.parquet"
OUT_DIR = REPO / "outputs/vBTC_per_symbol_avax"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "AVAXUSDT"
SEEDS = (42, 1337, 7, 19, 2718)
ALL_FOLDS = list(range(10))
OOS_FOLDS = list(range(1, 10))
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
BETA_WIN_PIT_DAYS = 90

V6_CLEAN_28 = list(XS_FEATURE_COLS_V6_CLEAN)
ALL_DROPS = [
    "return_1d_xs_rank", "bk_ret_48b", "volume_ma_50",
    "ema_slope_20_1h", "ema_slope_20_1h_xs_rank",
    "vwap_zscore_xs_rank", "vwap_zscore",
    "atr_pct_xs_rank", "dom_z_7d_vs_bk", "obv_z_1d_xs_rank",
    "obv_signal", "price_volume_corr_10",
    "hour_cos", "hour_sin",
]
FUNDING_LEAN = ["funding_rate", "funding_rate_z_7d"]
ADD_CROSS_BTC = ["corr_to_btc_1d", "idio_vol_to_btc_1h", "beta_to_btc_change_5d"]
ADD_MORE_FUNDING = ["funding_rate_1d_change", "funding_streak_pos"]
WINNER_21 = ([f for f in V6_CLEAN_28 if f not in ALL_DROPS]
             + FUNDING_LEAN + ADD_CROSS_BTC + ADD_MORE_FUNDING)
DEAD_WEIGHT = {"mfi", "price_volume_corr_20", "idio_ret_48b_vs_bk", "funding_streak_pos"}
WINNER_17 = [f for f in WINNER_21 if f not in DEAD_WEIGHT]


def compute_pit_beta(panel, beta_win_days):
    btc_ret = panel[panel.symbol == "BTCUSDT"][["open_time", "return_pct"]].rename(
        columns={"return_pct": "btc_ret"}).drop_duplicates("open_time")
    bar_window = beta_win_days * 288
    out = []
    for sym, g in panel.groupby("symbol"):
        gg = g[["open_time", "return_pct"]].merge(btc_ret, on="open_time", how="left")
        gg = gg.sort_values("open_time").reset_index(drop=True)
        if sym == "BTCUSDT":
            gg["beta_pit"] = 1.0
        else:
            y = gg["return_pct"]; x = gg["btc_ret"]
            cov_xy = y.rolling(bar_window, min_periods=1000).cov(x)
            var_x = x.rolling(bar_window, min_periods=1000).var()
            beta = (cov_xy / var_x.replace(0, np.nan)).shift(1)
            gg["beta_pit"] = beta
        gg["symbol"] = sym
        out.append(gg)
    pit = pd.concat(out, ignore_index=True)[["symbol", "open_time", "beta_pit"]]
    return pit


def prepare_panel(panel_path, label):
    print(f"\n--- Preparing {label} ---", flush=True)
    panel = pd.read_parquet(panel_path)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    folds_all = _multi_oos_splits(panel)
    pit_beta = compute_pit_beta(panel, BETA_WIN_PIT_DAYS)
    panel = panel.merge(pit_beta, on=["symbol", "open_time"], how="left")
    btc_ret_map = panel[panel["symbol"] == "BTCUSDT"][["open_time", "return_pct"]].rename(
        columns={"return_pct": "btc_ret_t"}).drop_duplicates("open_time")
    panel = panel.merge(btc_ret_map, on="open_time", how="left")
    panel["alpha_beta"] = panel["return_pct"] - panel["beta_pit"] * panel["btc_ret_t"]
    # Per-symbol σ_idio from fold-0 training (locked)
    train0, _, _ = _slice(panel, folds_all[0])
    sigma_idio = train0.groupby("symbol")["alpha_beta"].std().to_dict()
    fallback = panel["alpha_beta"].std()
    panel["sigma_idio_ref"] = panel["symbol"].map(sigma_idio).fillna(fallback).clip(lower=1e-6)
    panel["target_beta"] = panel["alpha_beta"] / panel["sigma_idio_ref"]
    return panel, folds_all


def train_per_symbol_avax(panel, folds_all, feat_set, label):
    """Train AVAX-only LGBM with the same fold structure, 5 seeds."""
    print(f"\n--- Training AVAX-only model ({label}) ---", flush=True)
    print(f"  features: {len(feat_set)}", flush=True)
    avax = panel[panel["symbol"] == SYMBOL].copy()
    print(f"  AVAX rows: {len(avax):,}", flush=True)

    all_preds = []
    t_start = time.time()
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        train, cal, test = _slice(panel, folds_all[fid])  # slice from full panel for time boundaries
        # Restrict to AVAX rows in each slice
        tr = train[(train["symbol"] == SYMBOL) & (train["autocorr_pctile_7d"] >= THRESHOLD)]
        ca = cal[(cal["symbol"] == SYMBOL) & (cal["autocorr_pctile_7d"] >= THRESHOLD)]
        test_r = test[test["symbol"] == SYMBOL].copy()
        if len(tr) < 500 or len(ca) < 100 or len(test_r) < 50:
            print(f"  fold {fid}: insufficient AVAX data (train={len(tr)}, cal={len(ca)}, test={len(test_r)})",
                  flush=True)
            continue
        Xt = tr[feat_set].to_numpy(np.float32)
        Xc = ca[feat_set].to_numpy(np.float32)
        Xtest = test_r[feat_set].to_numpy(np.float32)
        yt = tr["target_beta"].to_numpy(np.float32)
        yc = ca["target_beta"].to_numpy(np.float32)
        mt = ~np.isnan(yt); mc = ~np.isnan(yc)
        if mt.sum() < 500 or mc.sum() < 100: continue
        preds = []
        for s in SEEDS:
            m = _train(Xt[mt], yt[mt], Xc[mc], yc[mc], seed=s)
            preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
        avg_pred = np.mean(preds, axis=0)
        test_r["pred_persym"] = avg_pred
        test_r["fold"] = fid
        all_preds.append(test_r[["symbol", "open_time", "alpha_beta", "return_pct", "pred_persym", "fold"]])
        print(f"  fold {fid}: train n={len(tr):,}, test n={len(test_r):,} ({time.time()-t0:.0f}s)",
              flush=True)
    print(f"  total: {time.time()-t_start:.0f}s", flush=True)
    return pd.concat(all_preds, ignore_index=True)


def per_cycle_ic_for_symbol(apd, target_col="alpha_A", pred_col="pred"):
    """Apd is universal-model predictions; compute AVAX's per-symbol over-time IC."""
    df = apd[(apd["symbol"] == SYMBOL) & (apd["fold"].isin(OOS_FOLDS))].dropna(
        subset=[target_col, pred_col])
    if len(df) < 100: return None
    # For per-symbol time series IC, we use Spearman of (pred, alpha) over time for this symbol
    return float(df[pred_col].rank().corr(df[target_col].rank()))


def main():
    print("=== Per-symbol model diagnostic on AVAXUSDT ===\n", flush=True)

    # === Per-symbol AVAX on 51-panel ===
    panel51, folds51 = prepare_panel(PANEL_51, "51-panel")
    feat_set = [f for f in WINNER_17 if f in panel51.columns]
    avax_pred_51 = train_per_symbol_avax(panel51, folds51, feat_set, "51-panel data")
    avax_pred_51.to_csv(OUT_DIR / "avax_persym_51panel.csv", index=False)

    # === Per-symbol AVAX on 111-panel data (only AVAX rows) ===
    panel111, folds111 = prepare_panel(PANEL_111, "111-panel")
    feat_set_111 = [f for f in WINNER_17 if f in panel111.columns]
    avax_pred_111 = train_per_symbol_avax(panel111, folds111, feat_set_111, "111-panel data (AVAX only)")
    avax_pred_111.to_csv(OUT_DIR / "avax_persym_111panel.csv", index=False)

    # === Universal model predictions for AVAX ===
    apd_uni_51 = pd.read_parquet(APD_UNI_51)
    apd_uni_51["open_time"] = pd.to_datetime(apd_uni_51["open_time"], utc=True)
    apd_uni_111 = pd.read_parquet(APD_UNI_111)
    apd_uni_111["open_time"] = pd.to_datetime(apd_uni_111["open_time"], utc=True)

    # Compute per-symbol time-series IC across all OOS rows for AVAX
    print("\n" + "="*80)
    print("  AVAXUSDT per-cycle (per-symbol time-series) IC comparison")
    print("="*80)
    ic_uni_51 = per_cycle_ic_for_symbol(apd_uni_51, "alpha_A", "pred")
    ic_uni_111 = per_cycle_ic_for_symbol(apd_uni_111, "alpha_A", "pred")

    # For per-symbol model output, compute IC
    avax_pred_51["fold"] = avax_pred_51["fold"]
    avax_pred_51_oos = avax_pred_51[avax_pred_51["fold"].isin(OOS_FOLDS)].dropna(
        subset=["alpha_beta", "pred_persym"])
    ic_ps_51 = float(avax_pred_51_oos["pred_persym"].rank().corr(
        avax_pred_51_oos["alpha_beta"].rank())) if len(avax_pred_51_oos) > 100 else None

    avax_pred_111_oos = avax_pred_111[avax_pred_111["fold"].isin(OOS_FOLDS)].dropna(
        subset=["alpha_beta", "pred_persym"])
    ic_ps_111 = float(avax_pred_111_oos["pred_persym"].rank().corr(
        avax_pred_111_oos["alpha_beta"].rank())) if len(avax_pred_111_oos) > 100 else None

    print(f"\n  1. Universal 51-panel model (WINNER_17): AVAX IC = {ic_uni_51:+.4f}", flush=True)
    print(f"  2. Universal 111-panel model (WINNER_17): AVAX IC = {ic_uni_111:+.4f}", flush=True)
    print(f"  3. Per-symbol AVAX on 51-panel data: AVAX IC = {ic_ps_51:+.4f}", flush=True)
    print(f"  4. Per-symbol AVAX on 111-panel data: AVAX IC = {ic_ps_111:+.4f}", flush=True)

    print(f"\n  Δ (per-symbol vs universal):", flush=True)
    print(f"    51-panel: {ic_ps_51 - ic_uni_51:+.4f}", flush=True)
    print(f"    111-panel: {ic_ps_111 - ic_uni_111:+.4f}", flush=True)

    print(f"\n  Δ (51 → 111 panel) STABILITY:", flush=True)
    print(f"    Universal model: {ic_uni_111 - ic_uni_51:+.4f}", flush=True)
    print(f"    Per-symbol model: {ic_ps_111 - ic_ps_51:+.4f}", flush=True)
    print(f"\n  If per-symbol is genuinely isolated, (4) should equal (3) ± noise.", flush=True)

    # Decision criteria
    print(f"\n  Decision criteria:")
    print(f"    A. per-symbol IC ≥ universal IC on 51 (per-symbol matches/beats baseline)", flush=True)
    print(f"       → {ic_ps_51:+.4f} vs {ic_uni_51:+.4f}: "
          f"{'PASS' if ic_ps_51 >= ic_uni_51 else 'FAIL'}", flush=True)
    print(f"    B. per-symbol IC on 111-data ≈ per-symbol IC on 51-data (isolation)", flush=True)
    print(f"       → {abs(ic_ps_111 - ic_ps_51):.4f} < 0.005: "
          f"{'PASS' if abs(ic_ps_111 - ic_ps_51) < 0.005 else 'NEEDS REVIEW'}", flush=True)
    print(f"    C. per-symbol IC on 111-data ≥ universal IC on 111-data (per-symbol fixes contamination)",
          flush=True)
    print(f"       → {ic_ps_111:+.4f} vs {ic_uni_111:+.4f}: "
          f"{'PASS' if ic_ps_111 > ic_uni_111 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
