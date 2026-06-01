"""X54 — V5_minus_v3 on canonical HL-50 to determine if universal improvement.

X53 found V5_minus_v3 (BASE+cohort+aggT+7cx) = +1.67 on HL-70.
Now test on canonical HL-50 (= panel_v2 50 syms).

If beats V5+5cx canonical HL-50 (+1.66) → universal improvement (always remove v3).
If <= +1.66 → universe-dependent optimum (HL-70 wants more features, HL-50 wants fewer).

Also test V5_minus_v3 with 5 crossX (subset, no new bn_spot features) for fair comparison.
"""
from __future__ import annotations
import csv, sys, time, warnings, importlib.util
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


def main():
    # Load panel (HL-70 with 7 crossX + full aggT + v3 — same panel as X51/X52)
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_hl70_v5_full.parquet")
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"] != "BTCUSDT"]

    # Canonical HL-50 = panel_v2 syms minus BTC
    canonical_syms = sorted(set(pd.read_parquet(
        REPO / "outputs/vBTC_features/panel_variants_with_funding_v2.parquet",
        columns=["symbol"])["symbol"].unique()) - {"BTCUSDT"})
    sub = panel[panel["symbol"].isin(canonical_syms)].copy()
    folds = x6.get_folds(x6.build_target_z(sub))
    sub = x6.build_target_z(sub)
    print(f"Canonical HL-50: {sub['symbol'].nunique()} syms × {len(sub):,} rows × {len(folds)} folds")

    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")
    if "bars_since_high_xs_rank" not in sub.columns:
        sub["bars_since_high_xs_rank"] = (sub.groupby("open_time")["bars_since_high"]
                                          .rank(pct=True).astype("float32"))

    cx_7 = ["bn_perp_okx_perp_z", "bn_perp_okx_spot_z", "okx_perp_spot_z",
            "bn_perp_cb_spot_z", "okx_cb_spot_z",
            "bn_spot_okx_spot_z", "bn_spot_cb_spot_z"]
    cx_5 = cx_7[:5]  # original 5
    aggT = ["signed_volume_4h", "tfi_4h", "aggr_ratio_4h", "buy_count_4h", "avg_trade_size_4h"]

    variants = [
        ("V5_minus_v3_7cx",  x6.BASE + x6.COHORT_EXTRAS + aggT + cx_7),
        ("V5_minus_v3_5cx",  x6.BASE + x6.COHORT_EXTRAS + aggT + cx_5),
        ("V5_full_7cx",      x6.BASE + x6.COHORT_EXTRAS + aggT + cx_7 + x6.V3_EXTRAS),
        ("V5_full_5cx",      x6.BASE + x6.COHORT_EXTRAS + aggT + cx_5 + x6.V3_EXTRAS),
    ]
    results = []
    for v_name, feats in variants:
        feats = [f for f in dict.fromkeys(feats) if f in sub.columns]
        tf = time.time()
        print(f"\n[{v_name}] ({len(feats)} feats)")
        try:
            apd = x6.train_per_sym_ridge(sub, folds, feats, label=v_name)
            pred_path = CACHE / f"x54_{v_name}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            print(f"  trained {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]")
        except Exception as e:
            print(f"  TRAIN ERR: {e}"); results.append({"variant": v_name, "error": str(e)}); continue
        m = x6.run_sleeve_on_preds(pred_path, f"x54_{v_name}")
        row = {"variant": v_name, "n_feats": len(feats), "train_ic": round(ic, 4), **m}
        results.append(row)
        if "sharpe" in m:
            print(f"  sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}")

    print(f"\n=== X54 results on canonical HL-50 ===")
    print(f"  Reference: V0 +2.01, V5+5cx (X29) +1.66, V5+7cx (X52) +1.13")
    for r in results:
        if "sharpe" in r:
            print(f"  {r['variant']:<20} ({r['n_feats']:>2} feats): Sharpe={r['sharpe']:+.2f} "
                  f"folds={r.get('folds_pos','?')} conc={r.get('concentration','?')}")

    keys = ["variant", "n_feats", "train_ic", "sharpe", "ci_lo", "ci_hi",
            "totPnL", "maxDD", "folds_pos", "concentration", "error"]
    out_csv = OUT / "X54_v5_minus_v3_canonical.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    main()
