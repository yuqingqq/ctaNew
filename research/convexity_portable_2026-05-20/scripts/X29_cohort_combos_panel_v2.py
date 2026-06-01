"""X29 — Re-run X26 cohort+extras combos using PANEL V2 (fixed aggT coverage).

CORRECTION: X26 used the OLD panel where 26 syms had 0% aggT coverage.
X18 created panel_v2 with 98.3% aggT coverage (all 51 syms).
This script re-tests the same 6 variants with panel_v2 to get clean numbers.

Variants (same as X26):
  V0: BASE + cohort (17 feats)             — should still be +2.01 (baseline)
  V1: BASE + cohort + aggT (22)            — may improve materially with full aggT
  V2: BASE + cohort + crossX (22)          — crossX coverage same as before
  V3: BASE + cohort + aggT + crossX (27)   — combo, may differ from X26
  V4: BASE + cohort + v3 (21)
  V5: BASE + cohort + ALL (31)
"""
from __future__ import annotations
import csv, sys, time, warnings, importlib.util, gc, resource
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
spec_b = importlib.util.spec_from_file_location("x6b",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6b_cohort_fill.py")
x6b = importlib.util.module_from_spec(spec_b); spec_b.loader.exec_module(x6b)

HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
HL_SYMS = set(HL_MAP[HL_MAP["on_hl"] == True]["symbol"].tolist())


def log_mem(label=""):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  [MEM {label}] peak_rss={rss_mb:.0f}MB", flush=True)


def load_panel_v2():
    """Load PANEL V2 (98.3% aggT coverage, all 51 syms)."""
    needed = (["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"]
              + x6.BASE + x6.AGGT_EXTRAS + x6.V3_EXTRAS)
    panel_path = REPO / "outputs/vBTC_features/panel_variants_with_funding_v2.parquet"
    panel = pd.read_parquet(panel_path, columns=list(set(needed)))
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"].isin(HL_SYMS) & (panel["symbol"] != "BTCUSDT")].copy()
    panel = x6b.build_cohort_fixed(panel)
    cross_path = REPO / "data/ml/cache/cross_exchange_features.parquet"
    cross_df = pd.read_parquet(cross_path)
    cross_df["open_time"] = pd.to_datetime(cross_df["open_time"], utc=True)
    cross_z_cols = [c for c in cross_df.columns if c.endswith("_basis_z")]
    panel = panel.merge(cross_df[["symbol", "open_time"] + cross_z_cols],
                        on=["symbol", "open_time"], how="left")
    panel = x6.build_target_z(panel)
    # X21 fix: do NOT add COHORT_EXTRAS to HEAVY_TAIL
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")
    return panel, cross_z_cols


def main():
    t0 = time.time()
    print("=== X29 re-run cohort combos with PANEL V2 (full aggT coverage) ===\n", flush=True)
    log_mem("start")
    panel, cross_z_cols = load_panel_v2()
    folds = x6.get_folds(panel)
    print(f"  panel V2: {len(panel):,} rows × {panel['symbol'].nunique()} syms", flush=True)
    # Verify coverage
    for c in ["aggr_ratio_4h"]:
        cov = panel[c].notna().mean() * 100
        per_sym = panel.groupby("symbol")[c].apply(lambda x: x.notna().mean() > 0.5).sum()
        print(f"  {c}: {cov:.1f}% overall, {per_sym}/{panel['symbol'].nunique()} syms >50%")
    log_mem("after_panel")

    variants = [
        ("V0_BASE_cohort_v2",            x6.BASE + x6.COHORT_EXTRAS,
         "baseline (BASE+cohort)"),
        ("V1_BASE_cohort_aggT_v2",       x6.BASE + x6.COHORT_EXTRAS + x6.AGGT_EXTRAS,
         "BASE + cohort + aggT (full coverage)"),
        ("V2_BASE_cohort_crossX_v2",     x6.BASE + x6.COHORT_EXTRAS + cross_z_cols,
         "BASE + cohort + crossX"),
        ("V3_BASE_cohort_aggT_crossX_v2", x6.BASE + x6.COHORT_EXTRAS + x6.AGGT_EXTRAS + cross_z_cols,
         "BASE + cohort + aggT + crossX"),
        ("V4_BASE_cohort_v3_v2",          x6.BASE + x6.COHORT_EXTRAS + x6.V3_EXTRAS,
         "BASE + cohort + v3_idio"),
        ("V5_BASE_cohort_ALL_v2",         x6.BASE + x6.COHORT_EXTRAS + x6.AGGT_EXTRAS + cross_z_cols + x6.V3_EXTRAS,
         "BASE + cohort + ALL"),
    ]

    results = []
    for v_name, feats, desc in variants:
        feats = list(dict.fromkeys(feats))
        tf = time.time()
        log_mem(f"before {v_name}")
        print(f"\n[{v_name}] {desc} ({len(feats)} feats)", flush=True)
        try:
            apd = x6.train_per_sym_ridge(panel, folds, feats, label=v_name)
            pred_path = CACHE / f"x29_{v_name}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            print(f"  trained: {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]", flush=True)
        except Exception as e:
            print(f"  TRAIN ERR: {e}"); import traceback; traceback.print_exc()
            results.append({"variant": v_name, "n_feats": len(feats), "error": str(e)}); continue
        m = x6.run_sleeve_on_preds(pred_path, f"x29_{v_name}")
        row = {"variant": v_name, "desc": desc, "n_feats": len(feats),
               "train_ic": round(ic, 4), "train_time_s": round(time.time()-tf, 0), **m}
        results.append(row)
        if "sharpe" in m:
            print(f"  sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}", flush=True)
        del apd; gc.collect()
        log_mem(f"after {v_name}")

    keys = ["variant", "desc", "n_feats", "train_ic", "sharpe", "ci_lo", "ci_hi",
            "totPnL", "maxDD", "folds_pos", "concentration", "train_time_s", "error"]
    out_csv = OUT / "X29_cohort_combos_panel_v2.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved {len(results)} → {out_csv} [{time.time()-t0:.0f}s]")
    print(f"\n=== X29 vs X26 (old panel) comparison ===")
    print(f"  variant                             X26(old)  X29(v2)  Δ")
    refs_x26 = {"V0": 2.01, "V1": 1.62, "V2": 1.90, "V3": 0.60, "V4": None, "V5": None}
    for r in results:
        if "sharpe" in r:
            key = r["variant"].split("_")[0]
            ref = refs_x26.get(key)
            delta_str = f"  Δ{(r['sharpe']-ref):+.2f}" if ref is not None else ""
            ref_str = f"{ref:+.2f}" if ref is not None else "n/a"
            print(f"  {r['variant']:<35} {ref_str:>8}  {r['sharpe']:+.2f}{delta_str}")


if __name__ == "__main__":
    main()
