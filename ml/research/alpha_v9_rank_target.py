"""Test rank-target regression vs baseline z-score target (h=48 K=7 ORIG25).

Hypothesis: the eval metric is per-bar rank IC (cross-sectional ordering),
but the model trains on per-symbol z-scored alpha (L2 regression). This is
a metric mismatch — model spends capacity on extreme magnitudes that the
strategy doesn't trade on.

Test: replace `demeaned_target = (alpha - rmean_sym) / rstd_sym` with a
per-bar rank-percentile transformed to N(0,1) via van der Waerden:

    rank_pct = rank_within_bar(alpha_realized) / (n_bar + 1)
    rank_target = norm.ppf(rank_pct)

This target is automatically standardized per bar, equally weights long
and short tails, and trains the model directly on what the strategy uses.

Same panel, same multi-OOS folds, same β-neutral execution, same post-fix
cost (4.5 bps one-way). Paired per-cycle delta vs baseline.
"""
from __future__ import annotations
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import (
    ENSEMBLE_SEEDS, _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v4_xs import portfolio_pnl_turnover_aware, block_bootstrap_ci
from ml.research.alpha_v8_h48_audit import build_wide_panel

HORIZON = 48
TOP_K = 7
TOP_FRAC = TOP_K / 25.0
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
OUT_DIR = REPO / "outputs/h48_rank_target"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_rank_target(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `rank_target` column = van der Waerden of per-bar rank of alpha_realized.

    Per-bar (per open_time) rank → percentile → inverse normal CDF.
    Result: per bar, target distribution is approximately N(0,1), equal
    cross-sectional moments → no scale heterogeneity across bars.

    Excludes the current row from its own bar's rank? No — alpha_realized
    is forward-looking but the rank is computed using ALL symbols' alphas
    at the same bar, which is what the strategy uses (rank predictions
    cross-sectionally at each bar). This is the correct target.
    """
    p = panel.copy()
    p["__rank__"] = p.groupby("open_time")["alpha_realized"].rank(method="average")
    p["__n__"] = p.groupby("open_time")["alpha_realized"].transform("size")
    p["__pct__"] = p["__rank__"] / (p["__n__"] + 1.0)
    # Clip to (0, 1) just in case of edge artifacts, then inverse-normal.
    pct = np.clip(p["__pct__"].values, 1e-6, 1 - 1e-6)
    p["rank_target"] = stats.norm.ppf(pct).astype(np.float32)
    p = p.drop(columns=["__rank__", "__n__", "__pct__"])
    return p


def main():
    panel = build_wide_panel()
    panel = add_rank_target(panel)
    # Drop rows where either target is NaN (rank target should always exist
    # if alpha_realized is non-null and there are >=2 symbols at that bar).
    panel = panel.dropna(subset=["demeaned_target", "rank_target"])
    print(f"Panel rows: {len(panel):,}  unique bars: {panel['open_time'].nunique():,}")

    folds = _multi_oos_splits(panel)
    print(f"Multi-OOS folds: {len(folds)}")

    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    pairs = []

    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            print(f"  fold {fold['fid']}: skipped (insufficient data)")
            continue

        avail = [c for c in v6_clean if c in panel.columns]
        Xt = tr[avail].to_numpy(dtype=np.float32)
        Xc = ca[avail].to_numpy(dtype=np.float32)
        Xtest = test[avail].to_numpy(dtype=np.float32)

        results = {}
        for tag, tcol in [("baseline", "demeaned_target"), ("rank", "rank_target")]:
            yt_ = tr[tcol].to_numpy(dtype=np.float32)
            yc_ = ca[tcol].to_numpy(dtype=np.float32)
            models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
            yt_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                                for m in models], axis=0)
            r = portfolio_pnl_turnover_aware(
                test, yt_pred, top_frac=TOP_FRAC,
                cost_bps_per_leg=COST_PER_LEG, sample_every=HORIZON, beta_neutral=True,
            )
            results[tag] = r["df"][["time", "net_bps", "spread_ret_bps", "rank_ic"]].rename(
                columns={c: f"{tag}_{c}" for c in ["net_bps", "spread_ret_bps", "rank_ic"]})

        merged = results["baseline"].merge(results["rank"], on="time", how="inner")
        merged["fold"] = fold["fid"]
        pairs.append(merged)
        print(f"  fold {fold['fid']:>2}: {len(merged)} cycles  "
              f"base_net={merged['baseline_net_bps'].mean():+.2f}  "
              f"rank_net={merged['rank_net_bps'].mean():+.2f}  "
              f"base_IC={merged['baseline_rank_ic'].mean():+.4f}  "
              f"rank_IC={merged['rank_rank_ic'].mean():+.4f}  "
              f"({time.time()-t0:.0f}s)")

    paired = pd.concat(pairs, ignore_index=True)
    paired["delta_net"] = paired["rank_net_bps"] - paired["baseline_net_bps"]

    base = paired["baseline_net_bps"].to_numpy()
    rank = paired["rank_net_bps"].to_numpy()
    delta = paired["delta_net"].to_numpy()

    sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0
    base_sh, base_lo, base_hi = block_bootstrap_ci(base, statistic=sharpe_est,
                                                     block_size=7, n_boot=2000)
    rank_sh, rank_lo, rank_hi = block_bootstrap_ci(rank, statistic=sharpe_est,
                                                     block_size=7, n_boot=2000)
    delta_sh = delta.mean() / delta.std() * np.sqrt(CYCLES_PER_YEAR) if delta.std() > 0 else 0
    t = delta.mean() / (delta.std() / np.sqrt(len(delta)))
    p = 1 - stats.norm.cdf(abs(t))

    print("\n" + "=" * 100)
    print(f"MULTI-OOS PAIRED VALIDATION — RANK-TARGET vs BASELINE Z-SCORE")
    print(f"  h={HORIZON} K={TOP_K} ORIG25, β-neutral, {COST_PER_LEG} bps/leg one-way taker")
    print(f"  {paired['fold'].nunique()} folds, {len(delta)} cycles")
    print("=" * 100)
    print(f"  Baseline (z-score target):   Sharpe {base_sh:+.2f}  "
          f"[{base_lo:+.2f}, {base_hi:+.2f}]   "
          f"net {base.mean():+.2f} bps/cyc   "
          f"IC {paired['baseline_rank_ic'].mean():+.4f}")
    print(f"  Rank target (van der Waerden): Sharpe {rank_sh:+.2f}  "
          f"[{rank_lo:+.2f}, {rank_hi:+.2f}]   "
          f"net {rank.mean():+.2f} bps/cyc   "
          f"IC {paired['rank_rank_ic'].mean():+.4f}")
    print(f"  Delta (rank-base):           Sharpe Δ{delta_sh:+.2f}   "
          f"net Δ{delta.mean():+.3f} bps/cyc   "
          f"t={t:+.2f}  one-sided p={p:.4f}   "
          f"rank-wins {(delta > 0).mean()*100:.1f}% of cycles")

    paired.to_csv(OUT_DIR / "alpha_v9_rank_target_pairs.csv", index=False)
    summary = {
        "n_cycles": len(delta), "n_folds_used": int(paired["fold"].nunique()),
        "baseline_sharpe": float(base_sh), "baseline_ci": [float(base_lo), float(base_hi)],
        "rank_sharpe": float(rank_sh), "rank_ci": [float(rank_lo), float(rank_hi)],
        "delta_sharpe": float(delta_sh), "delta_mean_bps": float(delta.mean()),
        "delta_t_stat": float(t), "delta_p_value": float(p),
        "rank_wins_pct": float((delta > 0).mean() * 100),
        "baseline_rank_ic": float(paired["baseline_rank_ic"].mean()),
        "rank_target_rank_ic": float(paired["rank_rank_ic"].mean()),
        "cost_bps_per_leg": COST_PER_LEG,
    }
    with open(OUT_DIR / "alpha_v9_rank_target_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
