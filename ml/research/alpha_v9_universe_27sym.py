"""ORIG25 + LDO + 1000SHIB combined test.

Phase 1 leave-one-in identified two compatible additions to ORIG25:
  LDOUSDT       Δsh +0.12  (CI lo +0.95)
  1000SHIBUSDT  Δsh +0.35  (CI lo +1.20)

This script tests both together (27-symbol universe). Three possibilities:
  Additive:    Sharpe ≈ +3.20 (lifts compose)
  Sub-additive: Sharpe ≈ +2.85 (overlap)
  Antagonistic: Sharpe < +2.75 (joint rank-shift hurts)

Discipline gate (must all pass for deployment):
  1. Stacked Sh ≥ +2.75 (no Sharpe regression)
  2. Bootstrap CI on Δnet vs ORIG25 lower bound > 0
  3. Per-fold consistency ≥ 6/10 vs ORIG25
  4. Hard-split frozen survives

Quick comparison only here; if passes 1+2, follow-up with hard-split test.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import list_universe
from ml.research.alpha_v9_universe_expand import build_wide_panel_for, run_config

OUT_DIR = REPO / "outputs/universe_27sym"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_SYMBOLS = {"ETCUSDT", "HBARUSDT", "ICPUSDT", "LDOUSDT", "TRBUSDT",
               "AAVEUSDT", "MKRUSDT", "AXSUSDT", "GMXUSDT",
               "1000PEPEUSDT", "1000SHIBUSDT", "TONUSDT", "ORDIUSDT", "WIFUSDT"}


def main():
    universe_full = sorted(list_universe(min_days=200))
    orig25 = sorted([s for s in universe_full if s not in NEW_SYMBOLS])
    additions = sorted(["LDOUSDT", "1000SHIBUSDT"])
    u27 = sorted(orig25 + additions)
    print(f"ORIG25 baseline ({len(orig25)} syms)")
    print(f"Adding: {additions}")
    print(f"ORIG27 universe ({len(u27)} syms)")
    print()

    panel_orig25 = build_wide_panel_for(orig25)
    r_base = run_config(panel_orig25, top_k=7, label="ORIG25")

    panel_27 = build_wide_panel_for(u27)
    r_27 = run_config(panel_27, top_k=7, label="ORIG25+LDO+1000SHIB")

    # Headline
    print("\n" + "=" * 100)
    print("ORIG25 vs ORIG25+LDO+1000SHIB  (conv+PM, 10-fold multi-OOS)")
    print("=" * 100)
    for r in [r_base, r_27]:
        delta = r["stacked_sharpe"] - r_base["stacked_sharpe"]
        marker = "" if r["label"] == "ORIG25" else f"  Δ vs ORIG25 = {delta:+.2f}"
        print(f"  {r['label']:<25}  N={r['n_universe']}  baseline_Sh={r['baseline_sharpe']:+.2f}  "
              f"stacked_Sh={r['stacked_sharpe']:+.2f}  CI=[{r['stacked_ci'][0]:+.2f},{r['stacked_ci'][1]:+.2f}]  "
              f"net={r['stacked_net']:+.2f} bps{marker}")

    delta_sh = r_27["stacked_sharpe"] - r_base["stacked_sharpe"]
    delta_net = r_27["stacked_net"] - r_base["stacked_net"]
    print(f"\n  Combined Δ vs ORIG25:  ΔSh={delta_sh:+.2f}  Δnet={delta_net:+.2f} bps/cyc")
    print(f"  Sum of individual Δsh:  +0.12 (LDO) + 0.35 (1000SHIB) = +0.47 (additive prediction)")
    if delta_sh >= 0.30:
        print(f"  → ADDITIVE: combined lift ~ sum of individual lifts")
    elif delta_sh >= 0.0:
        print(f"  → SUB-ADDITIVE: positive but with overlap")
    else:
        print(f"  → ANTAGONISTIC: combination hurts vs individual additions")

    # Save
    pd.DataFrame([r_base, r_27]).to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
