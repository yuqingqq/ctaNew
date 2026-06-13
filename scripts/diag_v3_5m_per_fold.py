"""Per-fold attribution + matched-basket placebo for v3_5m augment variants.

If the +1.66 Sharpe lift of V1_full19 is concentrated in 1-2 folds (Phase Q
pattern), it's a fragile in-sample artifact, not a robust signal.

Also runs matched-basket placebo: for each cycle, randomly pick the same
basket sizes from the universe; if random baskets produce similar Sharpe to
the model picks, the model isn't doing the work.
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec); spec.loader.exec_module(psl)

OUT = REPO / "outputs/vBTC_audit_panel_v3_augment_5m"

def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def main():
    print("=== v3_5m per-fold attribution ===\n", flush=True)
    variants = ["V0_WINNER_17", "V1_W17_plus_v3_full19",
                "V2_W17_plus_v3_top8", "V3_W17_plus_v3_top4"]
    dfs = {}
    for v in variants:
        path = OUT / f"{v}_v31_hedged.csv"
        df = pd.read_csv(path)
        dfs[v] = df
        print(f"  {v}: {len(df)} cycles", flush=True)

    # Per-fold Sharpe + PnL per variant
    print("\nPer-fold Sharpe (each variant):", flush=True)
    print(f"  {'fold':>4} " + " ".join(f"{v:<22}" for v in variants), flush=True)
    for fold in range(1, 10):
        row = [f"  {fold:>4}"]
        for v in variants:
            g = dfs[v][dfs[v]["fold"] == fold]
            sh = _sharpe(g["net_pnl_bps"].to_numpy())
            row.append(f"{sh:+7.2f} ({g['net_pnl_bps'].sum():+6.0f} bps)")
        print(" ".join(row), flush=True)

    # Per-fold delta vs V0
    print("\nΔ Sharpe per fold (vs V0_WINNER_17):", flush=True)
    print(f"  {'fold':>4}  {'V1-V0':>8}  {'V2-V0':>8}  {'V3-V0':>8}", flush=True)
    v0 = dfs["V0_WINNER_17"]
    fold_results = {}
    for v in variants[1:]:
        fold_results[v] = []
    for fold in range(1, 10):
        v0_sh = _sharpe(v0[v0["fold"]==fold]["net_pnl_bps"].to_numpy())
        deltas = {}
        for v in variants[1:]:
            v_sh = _sharpe(dfs[v][dfs[v]["fold"]==fold]["net_pnl_bps"].to_numpy())
            deltas[v] = v_sh - v0_sh
            fold_results[v].append((fold, v_sh - v0_sh, v_sh, v0_sh))
        print(f"  {fold:>4}  {deltas['V1_W17_plus_v3_full19']:>+8.2f}  "
              f"{deltas['V2_W17_plus_v3_top8']:>+8.2f}  "
              f"{deltas['V3_W17_plus_v3_top4']:>+8.2f}", flush=True)

    # LOFO (Leave-One-Fold-Out) test
    print("\nLeave-one-fold-out: if we exclude fold F, what's the remaining Sharpe of V1 vs V0?", flush=True)
    print(f"  {'exclude':>8}  {'V0 Sharpe':>10}  {'V1 Sharpe':>10}  {'Δ':>8}", flush=True)
    for excl in range(1, 10):
        v0_remain = v0[v0["fold"] != excl]["net_pnl_bps"].to_numpy()
        v1_remain = dfs["V1_W17_plus_v3_full19"]
        v1_remain = v1_remain[v1_remain["fold"] != excl]["net_pnl_bps"].to_numpy()
        sh_v0 = _sharpe(v0_remain); sh_v1 = _sharpe(v1_remain)
        delta = sh_v1 - sh_v0
        flag = "  ← drives V1 lift" if delta < 0.5 else ""
        print(f"  {excl:>8}  {sh_v0:+10.2f}  {sh_v1:+10.2f}  {delta:>+8.2f}{flag}", flush=True)

    # Concentration: what % of V1's lift comes from worst-2 folds for V0?
    print("\nDoes V1 help WHERE V0 needs it (regime-saver) or HELP ALREADY-GOOD folds?", flush=True)
    for v in variants[1:]:
        results = fold_results[v]
        # Sort folds by V0 Sharpe ascending (worst V0 folds first)
        results_sorted_by_v0 = sorted(results, key=lambda r: r[3])
        worst3_v0_folds = [r[0] for r in results_sorted_by_v0[:3]]
        worst3_v0_deltas = [r[1] for r in results_sorted_by_v0[:3]]
        best3_v0_folds = [r[0] for r in results_sorted_by_v0[-3:]]
        best3_v0_deltas = [r[1] for r in results_sorted_by_v0[-3:]]
        print(f"  {v}:", flush=True)
        print(f"    V0 worst-3 folds {worst3_v0_folds}: avg Δ = {np.mean(worst3_v0_deltas):+.2f}",
              flush=True)
        print(f"    V0 best-3 folds  {best3_v0_folds}: avg Δ = {np.mean(best3_v0_deltas):+.2f}",
              flush=True)


if __name__ == "__main__":
    main()
