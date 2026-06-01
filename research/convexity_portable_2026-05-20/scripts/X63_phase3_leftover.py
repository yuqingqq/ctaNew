"""X63 — Phase 3 leftover diagnostics for V5_minus_v3_7cx + X66 regime ensemble.

Sections:
  A. Embargo sensitivity (re-train V5_mv3 with 0d, 1d, 3d, 7d embargo)
  B. Half-sample test (train on first half, test on second half)
  C. Sym-dropout simulation (random drops of 5, 10, 15 syms)
  D. X66 cost sensitivity (regime ensemble at cost {1,2,3,4.5,6,9,12} bps)
"""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
import phase_ah_sleeve as sleeve


def main():
    print("=" * 70)
    print("X63 Phase 3 leftover diagnostics on V5_minus_v3_7cx + X66 ensemble")
    print("=" * 70)

    # === D. X66 ensemble cost sensitivity (most important) ===
    print("\n--- D. X66 V5_mv3 sideways + V0 bull (thr=0.20) cost sensitivity ---")
    pred_path = CACHE / "x66_V5_sideways_V0_bull_t0.2_preds.parquet"
    if not pred_path.exists():
        print(f"  {pred_path} not found; skip D")
    else:
        print(f"  {'cost':<6} {'Sharpe':>8} {'folds+':>8} {'conc':>8}")
        for c in [1.0, 2.0, 3.0, 4.5, 6.0, 9.0, 12.0]:
            sleeve.COST_PER_LEG = c
            sleeve.COST_PER_UNIT_ABS_DELTA = 0.5 * c
            m = x6.run_sleeve_on_preds(pred_path, f"x63d_x66_c{c}")
            sh = m.get("sharpe")
            fp = m.get("folds_pos", "?")
            conc = m.get("concentration", "?")
            sh_str = f"{sh:+.2f}" if isinstance(sh, (int, float)) else "?"
            print(f"  {c:<6.1f} {sh_str:>8} {str(fp):>8} {str(conc):>8}", flush=True)
        # Reset cost
        sleeve.COST_PER_LEG = 4.5
        sleeve.COST_PER_UNIT_ABS_DELTA = 0.5 * 4.5

    # === C. Sym-dropout simulation on V5_mv3 ===
    print("\n--- C. Sym-dropout simulation on V5_mv3 HL-50 ---")
    base_apd = pd.read_parquet(CACHE / "x54_V5_minus_v3_7cx_preds.parquet")
    syms = sorted(base_apd["symbol"].unique())
    print(f"  Baseline: 50 syms, V5_mv3 = +1.74")
    np.random.seed(20260521)
    print(f"\n  {'n_drop':<8} {'mean Sharpe':>12} {'std':>8} {'min':>8} {'max':>8}")
    for n_drop in [3, 5, 10, 15]:
        sharpes = []
        for trial in range(10):
            drop_syms = np.random.choice(syms, n_drop, replace=False).tolist()
            apd_d = base_apd[~base_apd["symbol"].isin(drop_syms)]
            tmp = CACHE / f"x63c_drop{n_drop}_t{trial}_preds.parquet"
            apd_d.to_parquet(tmp, index=False)
            m = x6.run_sleeve_on_preds(tmp, f"x63c_d{n_drop}_t{trial}")
            sh = m.get("sharpe", 0) or 0
            sharpes.append(sh)
        arr = np.array(sharpes)
        print(f"  {n_drop:<8} {arr.mean():>+12.2f} {arr.std():>+8.2f} "
              f"{arr.min():>+8.2f} {arr.max():>+8.2f}", flush=True)

    # === B. Half-sample test (compute Sharpe on first half vs second half folds) ===
    print("\n--- B. Half-sample test on V5_mv3 HL-50 ---")
    base_apd = pd.read_parquet(CACHE / "x54_V5_minus_v3_7cx_preds.parquet")
    folds = sorted(base_apd["fold"].unique())
    first_half = folds[:len(folds)//2]
    second_half = folds[len(folds)//2:]
    print(f"  First half folds: {first_half}, Second half folds: {second_half}")
    for label, fset in [("first_half", first_half), ("second_half", second_half)]:
        apd_h = base_apd[base_apd["fold"].isin(fset)]
        tmp = CACHE / f"x63b_{label}_preds.parquet"
        apd_h.to_parquet(tmp, index=False)
        m = x6.run_sleeve_on_preds(tmp, f"x63b_{label}")
        sh = m.get("sharpe", 0) or 0
        fp = m.get("folds_pos", "?")
        conc = m.get("concentration", "?")
        print(f"  {label:<15}: Sharpe={sh:+.2f} folds={fp} conc={conc}", flush=True)


if __name__ == "__main__":
    main()
