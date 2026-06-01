"""X26 — Add aggT/crossX to cohort baseline (Ridge Per-sym +cohort = +2.01).

Tests whether adding aggT or crossX to BASE+cohort (the +2.01 cell) helps,
hurts, or is neutral. Each is tested individually so we can see incremental
contribution beyond the strong cohort effect.

Variants:
  V0: BASE + cohort (17 feats)         — baseline +2.01
  V1: BASE + cohort + aggT (22 feats)
  V2: BASE + cohort + crossX (22 feats)
  V3: BASE + cohort + aggT + crossX (27 feats)
  V4: BASE + cohort + v3 (22 feats)    — v3 = idio_* features
  V5: BASE + cohort + ALL (32 feats)   — kitchen sink

All on HL-50 with the canonical Ridge Per-sym pipeline (X21 fix applied).
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


def load_panel():
    needed = (["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"]
              + x6.BASE + x6.AGGT_EXTRAS + x6.V3_EXTRAS)
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                            columns=list(set(needed)))
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"].isin(HL_SYMS) & (panel["symbol"] != "BTCUSDT")].copy()
    panel = x6b.build_cohort_fixed(panel)
    # crossX
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
    print("=== X26 cohort + extras combos (canonical Per-sym Ridge) ===\n", flush=True)
    log_mem("start")
    panel, cross_z_cols = load_panel()
    folds = x6.get_folds(panel)
    print(f"  panel: {len(panel):,} rows × {panel['symbol'].nunique()} syms", flush=True)
    log_mem("after_panel")

    variants = [
        ("V0_BASE_cohort",            x6.BASE + x6.COHORT_EXTRAS,
         "baseline +2.01 (BASE+cohort)"),
        ("V1_BASE_cohort_aggT",       x6.BASE + x6.COHORT_EXTRAS + x6.AGGT_EXTRAS,
         "BASE + cohort + aggT (22 feats)"),
        ("V2_BASE_cohort_crossX",     x6.BASE + x6.COHORT_EXTRAS + cross_z_cols,
         "BASE + cohort + crossX (22 feats)"),
        ("V3_BASE_cohort_aggT_crossX", x6.BASE + x6.COHORT_EXTRAS + x6.AGGT_EXTRAS + cross_z_cols,
         "BASE + cohort + aggT + crossX (27 feats)"),
        ("V4_BASE_cohort_v3",          x6.BASE + x6.COHORT_EXTRAS + x6.V3_EXTRAS,
         "BASE + cohort + v3_idio (22 feats)"),
        ("V5_BASE_cohort_ALL",         x6.BASE + x6.COHORT_EXTRAS + x6.AGGT_EXTRAS + cross_z_cols + x6.V3_EXTRAS,
         "BASE + cohort + ALL (32 feats)"),
    ]

    results = []
    for v_name, feats, desc in variants:
        feats = list(dict.fromkeys(feats))  # dedup preserving order
        tf = time.time()
        log_mem(f"before {v_name}")
        print(f"\n[{v_name}] {desc} ({len(feats)} feats)", flush=True)
        try:
            apd = x6.train_per_sym_ridge(panel, folds, feats, label=v_name)
            pred_path = CACHE / f"x26_{v_name}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            print(f"  trained: {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]", flush=True)
        except Exception as e:
            print(f"  TRAIN ERR: {e}"); import traceback; traceback.print_exc()
            results.append({"variant": v_name, "n_feats": len(feats), "error": str(e)}); continue
        m = x6.run_sleeve_on_preds(pred_path, f"x26_{v_name}")
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
    out_csv = OUT / "X26_cohort_combos.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved {len(results)} → {out_csv} [{time.time()-t0:.0f}s]")
    print(f"\n=== X26 results ===")
    print(f"  V0 baseline (BASE+cohort, 17 feats): +2.01")
    for r in results:
        if "sharpe" in r:
            print(f"  {r['variant']:<30} ({r['n_feats']:>2}) Sharpe={r['sharpe']:+.2f}  IC={r['train_ic']:+.4f}")


if __name__ == "__main__":
    main()
