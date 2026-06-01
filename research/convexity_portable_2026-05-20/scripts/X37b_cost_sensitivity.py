"""X37b — Cost sensitivity using x6.run_sleeve_on_preds (proper API).

Tests V0 and V5 at multiple cost levels:
  costs ∈ {1, 2, 3, 4.5 baseline, 6, 9, 12} bps/leg
"""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

CACHE = REPO / "research/convexity_portable_2026-05-20/results/_cache"

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
import phase_ah_sleeve as sleeve


def main():
    print("=== X37b cost sensitivity test ===\n")
    cases = [
        ("V0_BASE_cohort", CACHE / "x29_V0_BASE_cohort_v2_preds.parquet"),
        ("V5_BASE_cohort_ALL", CACHE / "x29_V5_BASE_cohort_ALL_v2_preds.parquet"),
    ]
    costs = [1.0, 2.0, 3.0, 4.5, 6.0, 9.0, 12.0]

    for label, path in cases:
        if not path.exists():
            print(f"\n{label}: pred file missing"); continue
        print(f"\n[{label}]")
        print(f"  {'cost (bps/leg)':<16} {'Sharpe':>8} {'totPnL':>10} {'folds+':>8} {'conc':>8}")
        for c in costs:
            # Override sleeve cost constants
            sleeve.COST_PER_LEG = c
            sleeve.COST_PER_UNIT_ABS_DELTA = 0.5 * c
            m = x6.run_sleeve_on_preds(path, f"x37b_{label}_cost{c}")
            sh = m.get("sharpe")
            pnl = m.get("totPnL")
            fp = m.get("folds_pos", "?")
            conc = m.get("concentration", "?")
            sh_str = f"{sh:+.2f}" if isinstance(sh, (int, float)) else "?"
            pnl_str = f"{pnl}" if pnl is not None else "?"
            print(f"  {c:<14.1f}   {sh_str:>8}  {pnl_str:>10}  {str(fp):>8}  {str(conc):>8}", flush=True)


if __name__ == "__main__":
    main()
