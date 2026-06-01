"""X37 — Cost sensitivity test for V0 and V5.

Reruns sleeve eval at multiple cost assumptions to see sensitivity:
  costs ∈ {1, 2, 3, 4.5 (baseline), 6, 9, 12} bps per leg

Production at HL: ~3 bps taker, ~1 bps maker.
4.5 bps is conservative baseline used in our tests.

Uses phase_ah_sleeve.py's COST_PER_LEG parameter override.
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
# Load sleeve module
import phase_ah_sleeve as sleeve


def run_sleeve_with_cost(pred_path, cost_bps, label):
    """Override sleeve cost and run."""
    orig_cost = sleeve.COST_PER_LEG if hasattr(sleeve, "COST_PER_LEG") else None
    sleeve.COST_PER_LEG = cost_bps
    sleeve.COST_PER_UNIT_ABS_DELTA = 0.5 * cost_bps  # must be recalculated
    sleeve.APD_PATH = pred_path
    sleeve.OUT = REPO / "research/convexity_portable_2026-05-20/results" / f"_x37_cost{cost_bps}_{label}"
    sleeve.OUT.mkdir(parents=True, exist_ok=True)
    try:
        sleeve.main()
        sum_path = sleeve.OUT / "summary.json"
        import json
        if sum_path.exists():
            s = json.loads(sum_path.read_text())
            return {"sharpe": float(s.get("sharpe", 0)),
                     "totPnL": int(s.get("totPnL", 0)),
                     "folds_pos": s.get("folds_pos", "?")}
    except Exception as e:
        print(f"  ERR: {e}")
        return {}
    finally:
        if orig_cost is not None:
            sleeve.COST_PER_LEG = orig_cost
    return {}


def main():
    print("=== X37 cost sensitivity test ===\n")

    cases = [
        ("V0 BASE+cohort", CACHE / "x29_V0_BASE_cohort_v2_preds.parquet"),
        ("V5 BASE+cohort+ALL", CACHE / "x29_V5_BASE_cohort_ALL_v2_preds.parquet"),
    ]
    costs = [1.0, 2.0, 3.0, 4.5, 6.0, 9.0, 12.0]

    for label, path in cases:
        if not path.exists():
            print(f"\n{label}: prediction file missing"); continue
        print(f"\n[{label}] cost sweep:")
        print(f"  {'cost (bps/leg)':<16} {'Sharpe':>8} {'totPnL':>10} {'folds+':>8}")
        for c in costs:
            r = run_sleeve_with_cost(path, c, label.replace(" ", "_"))
            sh = r.get("sharpe")
            pnl = r.get("totPnL")
            fp = r.get("folds_pos", "?")
            sh_str = f"{sh:+.2f}" if isinstance(sh, (int, float)) else "?"
            pnl_str = f"{pnl}" if pnl is not None else "?"
            print(f"  {c:<14.1f}   {sh_str:>8}  {pnl_str:>10}  {fp:>8}")


if __name__ == "__main__":
    main()
