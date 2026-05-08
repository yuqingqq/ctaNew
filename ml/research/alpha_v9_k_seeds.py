"""2D sweep: K (portfolio size) × n_seeds (ensemble width).

Both were chosen / pinned BEFORE the cost-formula fix in commit df519ec.
At post-fix cost (2× higher than what selection saw), the optima may shift:

  - Higher K = more diversification = lower per-cycle variance, but
    dilutes alpha. With cost 2× higher, lower variance might win.
  - More seeds = lower ensemble noise. Pure compute cost; modest Sharpe lift.

Both run under conv_gate p=0.30 (validated production baseline). All folds
share the SAME set of 20 trained models — for n_seeds=N we average the
first N. Same panel, same folds, same evaluation.
"""
from __future__ import annotations
import json
import sys
import time
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import (
    _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v4_xs import portfolio_pnl_turnover_aware, block_bootstrap_ci
from ml.research.alpha_v8_h48_audit import build_wide_panel
from ml.research.alpha_v9_conviction_v2 import evaluate_portfolio

HORIZON = 48
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
GATE_PCTILE = 0.30
N_UNIVERSE = 25  # ORIG25
SEEDS_20 = (42, 7, 123, 99, 314,
             271, 1729, 2718, 1618, 6022,
             333, 555, 777, 999, 111,
             222, 444, 666, 888, 1234)
K_VALUES = [3, 5, 7, 10, 12, 15]
N_SEEDS_VALUES = [5, 10, 20]
OUT_DIR = REPO / "outputs/h48_k_seeds"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0


def main():
    panel = build_wide_panel()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    print(f"Multi-OOS folds: {len(folds)}")

    # Pre-train all 20 seeds per fold, cache predictions
    fold_preds = {}  # fid → np.ndarray (20 × n_test_rows)
    fold_tests = {}
    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            continue
        avail = [c for c in v6_clean if c in panel.columns]
        Xt = tr[avail].to_numpy(dtype=np.float32)
        yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
        Xc = ca[avail].to_numpy(dtype=np.float32)
        yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
        Xtest = test[avail].to_numpy(dtype=np.float32)

        all_preds = []
        for seed in SEEDS_20:
            m = _train(Xt, yt_, Xc, yc_, seed=seed)
            all_preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
        fold_preds[fold["fid"]] = np.array(all_preds)
        fold_tests[fold["fid"]] = test
        print(f"  fold {fold['fid']}: trained 20 seeds ({time.time() - t0:.0f}s)")

    # 2D sweep under conv_gate p=0.30
    print(f"\nEvaluating K × n_seeds matrix under conv_gate p={GATE_PCTILE}...")
    cell_records: dict[tuple, list] = {(k, ns): [] for k in K_VALUES for ns in N_SEEDS_VALUES}
    for fid, preds_all in fold_preds.items():
        test = fold_tests[fid]
        for ns in N_SEEDS_VALUES:
            yt_pred = preds_all[:ns].mean(axis=0)
            for k in K_VALUES:
                # Conv gate uses dispersion of mean preds; for K we change top_frac
                df = evaluate_portfolio(
                    test, yt_pred,
                    use_gate=True, gate_pctile=GATE_PCTILE,
                    use_magweight=False, top_k=k,
                )
                for _, r in df.iterrows():
                    cell_records[(k, ns)].append({
                        "fold": fid, "time": r["time"],
                        "gross": r["spread_ret_bps"],
                        "cost": r["cost_bps"], "net": r["net_bps"],
                        "long_turn": r["long_turnover"], "skipped": r["skipped"],
                    })

    # Display Sharpe matrix
    print("\n" + "=" * 110)
    print(f"K × N_SEEDS MATRIX — Sharpe (post-fix cost {COST_PER_LEG} bps/leg, conv_gate p={GATE_PCTILE})")
    print("=" * 110)
    header = f"  {'K':>4}"
    for ns in N_SEEDS_VALUES: header += f"   {f'n={ns}':>10}"
    print(header)
    sharpe_matrix = {}
    for k in K_VALUES:
        row = f"  {k:>4d}"
        for ns in N_SEEDS_VALUES:
            recs = cell_records[(k, ns)]
            if not recs:
                row += f"   {'NO DATA':>10}"
                continue
            df = pd.DataFrame(recs)
            sh = sharpe_est(df["net"].values)
            sharpe_matrix[(k, ns)] = sh
            row += f"   {sh:>+10.2f}"
        print(row)

    # Detail rows for each combo (sorted by Sharpe)
    print("\n" + "-" * 110)
    print(f"  {'K':>3} {'n_seed':>7} {'cycles':>7} {'%trade':>7} {'gross':>7} {'cost':>6} "
          f"{'net':>7} {'L_turn':>7} {'Sharpe':>7} {'95% CI':>15}")
    sorted_combos = sorted(sharpe_matrix.items(), key=lambda x: -x[1])
    summary = {}
    for (k, ns), sh in sorted_combos:
        recs = cell_records[(k, ns)]
        df = pd.DataFrame(recs)
        traded = df[df["skipped"] == 0]
        sh, lo, hi = block_bootstrap_ci(df["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)
        pct_trade = 100 * len(traded) / len(df)
        print(f"  {k:>3d} {ns:>7d} {len(df):>7d} {pct_trade:>6.1f}% "
              f"{traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>5.2f}  "
              f"{df['net'].mean():>+6.2f}  "
              f"{traded['long_turn'].mean() if len(traded) > 0 else 0:>6.0%}  "
              f"{sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]")
        summary[f"K={k}_n={ns}"] = {
            "K": k, "n_seeds": ns,
            "n_cycles": int(len(df)), "pct_trade": float(pct_trade),
            "gross": float(traded["gross"].mean() if len(traded) > 0 else 0),
            "cost": float(traded["cost"].mean() if len(traded) > 0 else 0),
            "net": float(df["net"].mean()),
            "long_turn": float(traded["long_turn"].mean() if len(traded) > 0 else 0),
            "sharpe": float(sh), "ci": [float(lo), float(hi)],
        }

    # Mark current production
    print(f"\n  Current production: K=7, n_seeds=5 → Sharpe {sharpe_matrix.get((7, 5), 'N/A')}")

    with open(OUT_DIR / "alpha_v9_k_seeds_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
