"""X53 — Audit panel_hl70_v5_full data + aggT ablation on HL-70.

PART A: Data quality audit
  - Per-sym, per-feature coverage in panel_hl70_v5_full
  - Identify any unexpected NaN/0 patterns
  - Per-sym listing dates (history sufficient for each fold?)
  - aggT coverage rate (now have flow for 70/70, but coverage?)

PART B: aggT ablation on HL-70
  - V5_minus_aggT: BASE + cohort + 7 crossX + v3 (28 feats, NO aggT)
  - V5_full: BASE + cohort + aggT + 7 crossX + v3 (33 feats, baseline +1.19)
  - V5_aggT_only: BASE + cohort + aggT (22 feats)
  - Compare to V0 (-0.11) and V5_full (+1.19)
"""
from __future__ import annotations
import csv, sys, time, warnings, importlib.util
from pathlib import Path
import pandas as pd, numpy as np

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)


def main():
    print("=" * 70)
    print("X53 PART A — Data quality audit of panel_hl70_v5_full")
    print("=" * 70)

    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_hl70_v5_full.parquet")
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"] != "BTCUSDT"]
    print(f"\nPanel: {len(panel):,} rows × {panel['symbol'].nunique()} syms × {panel.shape[1]} cols")
    print(f"Date range: {panel['open_time'].min()} → {panel['open_time'].max()}")

    # Per-sym start dates
    print("\n--- Per-sym listing dates (newest 10) ---")
    first_seen = panel.groupby("symbol")["open_time"].min().sort_values()
    for sym, t in first_seen.tail(10).items():
        n_rows = (panel["symbol"] == sym).sum()
        print(f"  {sym}: first={t}, n_rows={n_rows:,}")

    # Coverage by feature group (only counting rows where sym had data)
    feat_groups = {
        "BASE": x6.BASE,
        "cohort": x6.COHORT_EXTRAS,
        "aggT": ["aggr_ratio_4h", "tfi_4h", "signed_volume_4h", "buy_count_4h", "avg_trade_size_4h"],
        "v3_idio": x6.V3_EXTRAS,
        "funding": ["funding_rate", "funding_rate_z_7d", "funding_rate_1d_change"],
        "crossX_7": [c for c in panel.columns if c in {
            "bn_perp_okx_perp_z", "bn_perp_okx_spot_z", "okx_perp_spot_z",
            "bn_perp_cb_spot_z", "okx_cb_spot_z",
            "bn_spot_okx_spot_z", "bn_spot_cb_spot_z"}],
    }

    print("\n--- Feature group coverage ---")
    for grp, feats in feat_groups.items():
        print(f"\n{grp}:")
        for f in feats:
            if f not in panel.columns:
                print(f"  ⚠️ {f}: NOT IN PANEL")
                continue
            nn = panel[f].notna().mean() * 100
            syms_with_any = panel.groupby("symbol")[f].apply(lambda x: x.notna().any()).sum()
            syms_with_50 = panel.groupby("symbol")[f].apply(lambda x: x.notna().mean() > 0.5).sum()
            print(f"  {f}: {nn:.1f}% overall, {syms_with_any}/{panel['symbol'].nunique()} syms_any, "
                  f"{syms_with_50}/{panel['symbol'].nunique()} syms>50%")

    # Identify syms with low feature coverage (potentially problematic)
    print("\n--- Per-sym total feature coverage (sample of low-coverage) ---")
    aggT_cov = panel.groupby("symbol")["aggr_ratio_4h"].apply(lambda x: x.notna().mean())
    cx_cov = panel.groupby("symbol")["bn_spot_okx_spot_z"].apply(lambda x: x.notna().mean())
    print(f"\nLowest-coverage syms:")
    syms_summary = pd.DataFrame({"aggT_cov": aggT_cov, "bn_spot_okx_cov": cx_cov}).sort_values("aggT_cov")
    print(syms_summary.head(15).to_string())

    # =====================================================================
    print("\n\n" + "=" * 70)
    print("X53 PART B — aggT ablation on HL-70")
    print("=" * 70)

    panel = x6.build_target_z(panel)
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")

    # bars_since_high_xs_rank if missing
    if "bars_since_high_xs_rank" not in panel.columns:
        panel["bars_since_high_xs_rank"] = (panel.groupby("open_time")["bars_since_high"]
                                            .rank(pct=True).astype("float32"))

    aggT_cols = feat_groups["aggT"]
    cx_cols = feat_groups["crossX_7"]
    v3_cols = x6.V3_EXTRAS
    BASE = x6.BASE
    COHORT = x6.COHORT_EXTRAS

    folds = x6.get_folds(panel)
    HL_70 = sorted(panel["symbol"].unique())
    print(f"\nHL-70: {len(HL_70)} syms, {len(folds)} folds")

    variants = [
        ("V5_full",              BASE + COHORT + aggT_cols + cx_cols + v3_cols),
        ("V5_minus_aggT",        BASE + COHORT + cx_cols + v3_cols),
        ("V5_minus_crossX_v3",   BASE + COHORT + aggT_cols),
        ("V5_minus_v3",          BASE + COHORT + aggT_cols + cx_cols),
        ("V5_minus_crossX",      BASE + COHORT + aggT_cols + v3_cols),
        ("BASE_cohort_only",     BASE + COHORT),
    ]
    results = []
    for v_name, feats in variants:
        feats = [f for f in dict.fromkeys(feats) if f in panel.columns]
        tf = time.time()
        print(f"\n[{v_name}] ({len(feats)} feats)")
        try:
            apd = x6.train_per_sym_ridge(panel, folds, feats, label=v_name)
            pred_path = CACHE / f"x53_{v_name}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            print(f"  trained {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]")
        except Exception as e:
            print(f"  TRAIN ERR: {e}"); results.append({"variant": v_name, "n_feats": len(feats), "error": str(e)}); continue
        m = x6.run_sleeve_on_preds(pred_path, f"x53_{v_name}")
        row = {"variant": v_name, "n_feats": len(feats), "train_ic": round(ic, 4), **m}
        results.append(row)
        if "sharpe" in m:
            print(f"  sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}")

    keys = ["variant", "n_feats", "train_ic", "sharpe", "ci_lo", "ci_hi",
            "totPnL", "maxDD", "folds_pos", "concentration", "error"]
    out_csv = OUT / "X53_aggT_ablation_hl70.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    main()
