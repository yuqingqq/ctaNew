"""X67 — V5_minus_v3_7cx on largest fully-data-covered subset (50 syms).

50 syms have ALL: BN-perp + BN-spot + OKX-swap + OKX-spot + CB-spot + flow features.
Different composition from canonical HL-50:
  ADDED: BERA, KAITO, LAYER, ME, MOVE, S, TRUMP, ASTER (from HL-70 extras)
  REMOVED: HYPE, GMX, JUP, ORDI, RUNE, TAO, VVV (missing some data)

Note: VVV (load-bearing for V0 +2.01) is removed. Will test impact.
"""
from __future__ import annotations
import csv, sys, importlib.util
from pathlib import Path
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)

FULL_COV_50 = [
    "AAVEUSDT","ADAUSDT","APTUSDT","ARBUSDT","ASTERUSDT","ATOMUSDT","AVAXUSDT","AXSUSDT",
    "BCHUSDT","BERAUSDT","BIOUSDT","BNBUSDT","DOGEUSDT","DOTUSDT","ENAUSDT","ETCUSDT",
    "ETHUSDT","FILUSDT","HBARUSDT","ICPUSDT","INJUSDT","JTOUSDT","KAITOUSDT","LAYERUSDT",
    "LDOUSDT","LINKUSDT","LTCUSDT","MEUSDT","MOVEUSDT","NEARUSDT","ONDOUSDT","OPUSDT",
    "PENDLEUSDT","PENGUUSDT","PUMPUSDT","SEIUSDT","SOLUSDT","STRKUSDT","SUIUSDT","SUSDT",
    "TIAUSDT","TONUSDT","TRBUSDT","TRUMPUSDT","UNIUSDT","VIRTUALUSDT","WIFUSDT","WLDUSDT",
    "XRPUSDT","ZECUSDT",
]


def main():
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_hl70_v5_full.parquet")
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"].isin(FULL_COV_50)]
    panel = x6.build_target_z(panel)
    if "bars_since_high_xs_rank" not in panel.columns:
        panel["bars_since_high_xs_rank"] = (panel.groupby("open_time")["bars_since_high"]
                                            .rank(pct=True).astype("float32"))
    folds = x6.get_folds(panel)
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")

    cx_7 = ["bn_perp_okx_perp_z","bn_perp_okx_spot_z","okx_perp_spot_z",
            "bn_perp_cb_spot_z","okx_cb_spot_z","bn_spot_okx_spot_z","bn_spot_cb_spot_z"]
    aggT = ["signed_volume_4h","tfi_4h","aggr_ratio_4h","buy_count_4h","avg_trade_size_4h"]
    feats = list(dict.fromkeys(x6.BASE + x6.COHORT_EXTRAS + aggT + cx_7))
    feats = [f for f in feats if f in panel.columns]

    print(f"=== X67 V5_minus_v3_7cx on fully-covered 50 ===")
    print(f"Panel: {len(panel):,} rows × {panel['symbol'].nunique()} syms × {len(folds)} folds")
    print(f"Features ({len(feats)}): {feats[:5]}...")

    # Check crossX coverage now (should be ~100% at 4h-aligned bars)
    panel_4h = panel[(panel["open_time"].dt.hour % 4 == 0) & (panel["open_time"].dt.minute == 0)]
    print(f"\n4h-aligned bars: {len(panel_4h):,}")
    for c in cx_7:
        cov = panel_4h[c].notna().mean() * 100
        syms_50 = panel_4h.groupby("symbol")[c].apply(lambda x: x.notna().mean() > 0.5).sum()
        print(f"  {c}: {cov:.1f}% 4h-bar coverage, {syms_50}/50 syms >50%")

    apd = x6.train_per_sym_ridge(panel, folds, feats, label="x67_fc50")
    pred_path = CACHE / "x67_V5_minus_v3_7cx_fullcov50_preds.parquet"
    apd.to_parquet(pred_path, index=False)
    ic = float(apd["pred"].corr(apd["alpha_A"]))
    print(f"\nTrained: {len(apd):,} rows, IC={ic:+.4f}")

    m = x6.run_sleeve_on_preds(pred_path, "x67_fc50")
    print(f"Sleeve: Sharpe={m.get('sharpe', '?'):+.2f} folds={m.get('folds_pos','?')} "
          f"conc={m.get('concentration','?')} PnL={m.get('totPnL','?')}")
    print(f"\nReference: canonical HL-50 +1.74, HL-70 +1.67")


if __name__ == "__main__":
    main()
