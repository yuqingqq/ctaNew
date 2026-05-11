"""Phase 1: per-symbol leave-one-in test for NEW_SYMBOLS additions.

For each candidate s ∈ NEW_SYMBOLS, run multi-OOS with ORIG25 ∪ {s} (26 syms)
and compare conv+PM stacked Sharpe to ORIG25 baseline (+2.75).

Categorize each candidate:
  Compatible    (Δsh > -0.10):  add without meaningful damage
  Marginal      (-0.10 to -0.30):  include only if capacity benefit justifies
  Reject        (< -0.30):  excludes structurally
"""
from __future__ import annotations
import sys, time, warnings, json
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import list_universe
from ml.research.alpha_v9_universe_expand import build_wide_panel_for, run_config

HORIZON = 48
TOP_K = 7
OUT_DIR = REPO / "outputs/universe_leave_one_in"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_SYMBOLS = ["ETCUSDT", "HBARUSDT", "ICPUSDT", "LDOUSDT", "TRBUSDT",
               "AAVEUSDT", "MKRUSDT", "AXSUSDT", "GMXUSDT",
               "1000PEPEUSDT", "1000SHIBUSDT", "TONUSDT", "ORDIUSDT", "WIFUSDT"]


def main():
    universe_full = sorted(list_universe(min_days=200))
    orig25 = sorted([s for s in universe_full if s not in set(NEW_SYMBOLS)])
    print(f"ORIG25 baseline ({len(orig25)} syms)")
    print(f"Testing leave-one-in for {len(NEW_SYMBOLS)} candidates")
    print()

    # ORIG25 reference run
    panel_orig25 = build_wide_panel_for(orig25)
    r_base = run_config(panel_orig25, top_k=TOP_K, label="ORIG25 (baseline)")
    print(f"\n  ORIG25 baseline Sharpe: {r_base['stacked_sharpe']:+.3f}\n")

    results = [r_base]
    for s in NEW_SYMBOLS:
        u = sorted(orig25 + [s])
        print(f"\n--- ORIG25 + {{{s}}} ({len(u)} syms) ---")
        panel = build_wide_panel_for(u)
        r = run_config(panel, top_k=TOP_K, label=f"ORIG25+{s}")
        results.append(r)

    # Print headline table
    print("\n" + "=" * 110)
    print("PHASE 1: LEAVE-ONE-IN COMPATIBILITY  (conv+PM stacked, 10-fold multi-OOS)")
    print("=" * 110)
    base_sh = r_base["stacked_sharpe"]
    print(f"  {'config':<22}  {'N':>3}  {'baseline_Sh':>11}  {'stacked_Sh':>10}  "
          f"{'Δ vs ORIG':>10}  {'CI_lo':>7}  {'CI_hi':>7}  {'category':>11}")
    rows = []
    for r in results:
        delta = r["stacked_sharpe"] - base_sh
        if r["label"].startswith("ORIG25 (baseline)"):
            cat = "—"
        elif delta > -0.10:
            cat = "compatible"
        elif delta > -0.30:
            cat = "marginal"
        else:
            cat = "REJECT"
        print(f"  {r['label']:<22}  {r['n_universe']:>3}  "
              f"{r['baseline_sharpe']:>+11.2f}  {r['stacked_sharpe']:>+10.2f}  "
              f"{delta:>+10.2f}  "
              f"{r['stacked_ci'][0]:>+7.2f}  {r['stacked_ci'][1]:>+7.2f}  "
              f"{cat:>11}")
        rows.append({**r, "delta_vs_orig25": delta, "category": cat})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "leave_one_in.csv", index=False)
    with open(OUT_DIR / "leave_one_in.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\n  saved → {OUT_DIR}")

    # Categorized summary
    print("\n" + "=" * 80)
    print("SUMMARY BY CATEGORY")
    print("=" * 80)
    df_new = df[df["label"] != "ORIG25 (baseline)"]
    for cat in ["compatible", "marginal", "REJECT"]:
        names = df_new[df_new["category"] == cat]["label"].str.replace("ORIG25+", "").tolist()
        print(f"  {cat:>12} ({len(names)}): {names}")


if __name__ == "__main__":
    main()
