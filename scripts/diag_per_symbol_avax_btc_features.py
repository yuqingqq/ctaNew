"""AVAX per-symbol model with universe-invariant WINNER_BTC features.

Cleaner test than the WINNER_17 version because every feature is universe-invariant
(no _vs_bk, no xs_rank). So AVAX's feature values at time t are identical regardless
of what other symbols are in the panel.

Compares:
  1. Universal BTC-only 51-panel model — AVAX time-series IC
     (from outputs/vBTC_audit_panel_btc_only/all_predictions.parquet)
  2. Per-symbol AVAX with WINNER_BTC features (this test)

Pass criteria:
  A. Per-symbol IC > Universal IC by ≥ +0.005
  B. Per-symbol IC > 0 (model is at least slightly predictive on AVAX)
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train

PANEL_BTC = REPO / "outputs/vBTC_features_btc_only/panel_btc_only_clean.parquet"
APD_UNI = REPO / "outputs/vBTC_audit_panel_btc_only/all_predictions.parquet"
OUT_DIR = REPO / "outputs/vBTC_per_symbol_avax_btc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "AVAXUSDT"
SEEDS = (42, 1337, 7, 19, 2718)
ALL_FOLDS = list(range(10))
OOS_FOLDS = list(range(1, 10))
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC

WINNER_BTC = [
    "idio_ret_to_btc_12b", "idio_ret_to_btc_48b", "idio_ret_to_btc_288b",
    "dom_btc_z_1d", "dom_btc_change_48b", "dom_btc_change_288b",
    "beta_to_btc", "beta_to_btc_change_5d", "corr_to_btc_1d", "corr_to_btc_change_3d",
    "idio_vol_to_btc_1h", "idio_vol_to_btc_1d", "idio_vol_ratio_to_btc",
    "btc_ret_48b", "btc_realized_vol_1d", "btc_realized_vol_30d",
    "atr_pct", "obv_z_1d", "vwap_slope_96",
    "funding_rate", "funding_rate_z_7d", "funding_rate_1d_change",
    "listing_age_days", "log_quote_volume_90d", "residual_vol_90d_own_pctile",
]


def main():
    print("=== AVAX per-symbol model with WINNER_BTC features (universe-invariant) ===\n",
          flush=True)
    panel = pd.read_parquet(PANEL_BTC)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    print(f"Panel: {len(panel):,} rows × {panel['symbol'].nunique()} symbols", flush=True)

    feat_set = [f for f in WINNER_BTC if f in panel.columns]
    if len(feat_set) != len(WINNER_BTC):
        missing = set(WINNER_BTC) - set(feat_set)
        print(f"  WARNING: missing features {missing}", flush=True)
    print(f"WINNER_BTC features available: {len(feat_set)}/{len(WINNER_BTC)}", flush=True)

    folds_all = _multi_oos_splits(panel)
    avax = panel[panel["symbol"] == SYMBOL]
    print(f"AVAX rows: {len(avax):,}", flush=True)

    # === Train per-symbol AVAX model ===
    print(f"\n--- Training AVAX-only model with WINNER_BTC ({len(feat_set)} features) ---",
          flush=True)
    all_preds = []
    t_start = time.time()
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        train, cal, test = _slice(panel, folds_all[fid])
        tr = train[(train["symbol"] == SYMBOL) & (train["autocorr_pctile_7d"] >= THRESHOLD)]
        ca = cal[(cal["symbol"] == SYMBOL) & (cal["autocorr_pctile_7d"] >= THRESHOLD)]
        test_r = test[test["symbol"] == SYMBOL].copy()
        if len(tr) < 500 or len(ca) < 100 or len(test_r) < 50:
            print(f"  fold {fid}: insufficient (train={len(tr)}, cal={len(ca)}, test={len(test_r)})",
                  flush=True)
            continue
        Xt = tr[feat_set].to_numpy(np.float32)
        Xc = ca[feat_set].to_numpy(np.float32)
        Xtest = test_r[feat_set].to_numpy(np.float32)
        yt = tr["target_beta_btc"].to_numpy(np.float32)
        yc = ca["target_beta_btc"].to_numpy(np.float32)
        mt = ~np.isnan(yt); mc = ~np.isnan(yc)
        if mt.sum() < 500 or mc.sum() < 100: continue
        preds = []
        for s in SEEDS:
            m = _train(Xt[mt], yt[mt], Xc[mc], yc[mc], seed=s)
            preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
        avg_pred = np.mean(preds, axis=0)
        test_r["pred_persym"] = avg_pred
        test_r["fold"] = fid
        all_preds.append(test_r[["symbol", "open_time", "alpha_beta", "return_pct",
                                  "pred_persym", "fold"]])
        print(f"  fold {fid}: train n={len(tr):,}, test n={len(test_r):,} ({time.time()-t0:.0f}s)",
              flush=True)
    print(f"  total: {time.time()-t_start:.0f}s", flush=True)
    persym = pd.concat(all_preds, ignore_index=True)
    persym.to_csv(OUT_DIR / "avax_persym_btc_features.csv", index=False)

    # === Universal BTC-only model AVAX predictions ===
    apd_uni = pd.read_parquet(APD_UNI)
    apd_uni["open_time"] = pd.to_datetime(apd_uni["open_time"], utc=True)
    avax_uni = apd_uni[(apd_uni["symbol"] == SYMBOL) & (apd_uni["fold"].isin(OOS_FOLDS))]
    avax_uni = avax_uni.dropna(subset=["alpha_A", "pred"])
    if len(avax_uni) < 100:
        print(f"  ERROR: too few AVAX rows from universal model: {len(avax_uni)}", flush=True)
        sys.exit(1)
    ic_uni = float(avax_uni["pred"].rank().corr(avax_uni["alpha_A"].rank()))

    # === Per-symbol AVAX IC ===
    persym_oos = persym[persym["fold"].isin(OOS_FOLDS)].dropna(subset=["alpha_beta", "pred_persym"])
    ic_persym = float(persym_oos["pred_persym"].rank().corr(persym_oos["alpha_beta"].rank()))

    print("\n" + "="*80)
    print("  AVAXUSDT time-series IC comparison (WINNER_BTC features, universe-invariant)")
    print("="*80)
    print(f"\n  Universal BTC-only model (WINNER_BTC, 51 syms pooled): "
          f"AVAX IC = {ic_uni:+.4f}  (n={len(avax_uni):,})", flush=True)
    print(f"  Per-symbol AVAX model    (WINNER_BTC, AVAX only):     "
          f"AVAX IC = {ic_persym:+.4f}  (n={len(persym_oos):,})", flush=True)
    print(f"\n  Δ (per-symbol − universal) = {ic_persym - ic_uni:+.4f}", flush=True)

    # Reference: WINNER_17 per-symbol AVAX on 51-panel (from prior test)
    print(f"\n  Reference (prior test with WINNER_17 features):", flush=True)
    print(f"    Universal WINNER_17 51-model: AVAX IC = -0.0152", flush=True)
    print(f"    Per-symbol WINNER_17 AVAX:    AVAX IC = +0.0099", flush=True)

    print(f"\n  Decision criteria:", flush=True)
    print(f"    A. per-symbol IC > universal IC by ≥ +0.005", flush=True)
    print(f"       Δ = {ic_persym - ic_uni:+.4f}: "
          f"{'PASS' if (ic_persym - ic_uni) >= 0.005 else 'FAIL'}", flush=True)
    print(f"    B. per-symbol IC > 0 (model is at least slightly predictive)", flush=True)
    print(f"       IC = {ic_persym:+.4f}: "
          f"{'PASS' if ic_persym > 0 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
