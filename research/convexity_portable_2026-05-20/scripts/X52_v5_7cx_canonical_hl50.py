"""X52 — V5 with 7 crossX on CANONICAL HL-50 (= panel_v2 syms, no FARTCOIN).

X29 V5 (5 crossX) on canonical HL-50 = +1.66.
X51 V5 (7 crossX) on top-50-by-vol HL-50 (includes FARTCOIN) = -1.89.

This is the apples-to-apples test on canonical 50 syms.
"""
from __future__ import annotations
import csv, sys, time, warnings, importlib.util
from pathlib import Path
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE_DIR = OUT / "_cache"

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)


def main():
    # Load the panel built by X51 (HL-70 with 7 crossX + full aggT + v3)
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_hl70_v5_full.parquet")
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    print(f"Panel: {len(panel):,} rows × {panel['symbol'].nunique()} syms × {panel.shape[1]} cols")

    # Get canonical HL-50 = panel_v2 syms minus BTC
    canonical_syms = sorted(set(pd.read_parquet(
        REPO / "outputs/vBTC_features/panel_variants_with_funding_v2.parquet",
        columns=["symbol"])["symbol"].unique()) - {"BTCUSDT"})
    print(f"Canonical HL-50: {len(canonical_syms)} syms")

    sub = panel[panel["symbol"].isin(canonical_syms)].copy()
    sub_folds = x6.get_folds(sub)
    print(f"Sub-panel: {len(sub):,} rows × {sub['symbol'].nunique()} syms, {len(sub_folds)} folds")

    # V5 features = BASE + cohort + aggT + 7 crossX + v3
    cx_cols = [c for c in panel.columns if c.endswith("_basis_z") or
               (c.startswith("bn_spot") and c.endswith("_z")) or
               (c.startswith("bn_perp") and c.endswith("_z")) or
               (c.startswith("okx_") and c.endswith("_z"))]
    cx_cols = [c for c in panel.columns if c.endswith("_z") and ("basis" in c or "_spot" in c or "_perp" in c)]
    # Restrict to actual crossX z features
    cx_cols = [c for c in panel.columns if c in {
        "bn_perp_okx_perp_z", "bn_perp_okx_spot_z", "okx_perp_spot_z",
        "bn_perp_cb_spot_z", "okx_cb_spot_z",
        "bn_spot_okx_spot_z", "bn_spot_cb_spot_z",  # NEW
    }]
    print(f"crossX features ({len(cx_cols)}): {cx_cols}")

    aggT = ["signed_volume_4h", "tfi_4h", "aggr_ratio_4h", "buy_count_4h", "avg_trade_size_4h"]
    feats = list(dict.fromkeys(x6.BASE + x6.COHORT_EXTRAS + aggT + cx_cols + x6.V3_EXTRAS))
    feats = [f for f in feats if f in sub.columns]
    print(f"V5 features ({len(feats)}): {feats[:5]}...{feats[-3:]}")

    # X21 fix
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")

    tf = time.time()
    apd = x6.train_per_sym_ridge(sub, sub_folds, feats, label="x52_canonical_hl50")
    pred_path = CACHE_DIR / "x52_v5_7cx_canonical_hl50_preds.parquet"
    apd.to_parquet(pred_path, index=False)
    ic = float(apd["pred"].corr(apd["alpha_A"]))
    print(f"\nTrained: {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]")

    m = x6.run_sleeve_on_preds(pred_path, "x52_v5_7cx_canonical_hl50")
    print(f"Sleeve: Sharpe={m.get('sharpe', '?'):+.2f} folds={m.get('folds_pos','?')} "
          f"conc={m.get('concentration','?')} PnL={m.get('totPnL','?')}")
    print(f"\nReference:")
    print(f"  V5 (5 crossX) on canonical HL-50 (X29): +1.66")
    print(f"  V5 (7 crossX) on HL-70 (X51): +1.19")
    print(f"  V0 on canonical HL-50: +2.01")


if __name__ == "__main__":
    main()
