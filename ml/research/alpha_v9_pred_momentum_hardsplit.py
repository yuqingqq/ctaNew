"""PM_M2_b1 — hard-split frozen test.

Train LGBM ensemble ONCE on the first half of multi-OOS folds' training
data, freeze it, and apply baseline + PM_M2_b1 to the second half's test
folds without retraining. Mirrors the protocol used in
`alpha_v9_gate_validate.py V3` for conv_gate.

Pass criterion: PM_M2_b1 must show ΔSharpe > 0 with frozen model. If gate
only works because of per-fold retraining (model produces sharper, more
persistent predictions when calibrated to recent regime), hard-split kills
it. If gate is structural (filters noise blips universally), it survives.
"""
from __future__ import annotations
import json, sys, time, warnings
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
from ml.research.alpha_v9_pred_momentum import portfolio_pnl_pred_momentum_bn

HORIZON = 48
TOP_K = 7
TOP_FRAC = TOP_K / 25.0
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
OUT_DIR = REPO / "outputs/pred_momentum_hardsplit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _sharpe(x: np.ndarray) -> float:
    if len(x) == 0 or x.std() == 0:
        return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def main():
    panel = build_wide_panel()
    folds = _multi_oos_splits(panel)
    print(f"Multi-OOS folds: {len(folds)}")

    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    avail_feats = [c for c in v6_clean if c in panel.columns]

    n_train_folds = max(3, len(folds) // 2)
    train_folds = folds[:n_train_folds]
    test_folds = folds[n_train_folds:]
    print(f"  train folds (model freeze): {[f['fid'] for f in train_folds]}")
    print(f"  test folds (frozen eval):   {[f['fid'] for f in test_folds]}")

    # Train ONCE on combined data up to last train fold's test_end
    train_cutoff = train_folds[-1]["test_end"]
    panel_train = panel[panel["open_time"] < train_cutoff]
    panel_train_filt = panel_train[panel_train["autocorr_pctile_7d"] >= THRESHOLD]
    n_train = len(panel_train_filt)
    train_split = int(n_train * 0.85)
    Xt = panel_train_filt[avail_feats].iloc[:train_split].to_numpy(dtype=np.float32)
    yt_ = panel_train_filt["demeaned_target"].iloc[:train_split].to_numpy(dtype=np.float32)
    Xc = panel_train_filt[avail_feats].iloc[train_split:].to_numpy(dtype=np.float32)
    yc_ = panel_train_filt["demeaned_target"].iloc[train_split:].to_numpy(dtype=np.float32)
    print(f"  frozen training: {train_split:,} train rows, {n_train - train_split:,} cal rows")
    t0 = time.time()
    frozen_models = [_train(Xt, yt_, Xc, yc_, seed=s) for s in ENSEMBLE_SEEDS]
    print(f"  trained {len(frozen_models)} models in {time.time()-t0:.0f}s")

    base_all = []
    pm_all = []
    fold_records = []
    for fold in test_folds:
        _, _, test = _slice(panel, fold)
        Xtest = test[avail_feats].to_numpy(dtype=np.float32)
        pred_test = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                              for m in frozen_models], axis=0)
        r_base = portfolio_pnl_turnover_aware(
            test, pred_test, top_frac=TOP_FRAC,
            cost_bps_per_leg=COST_PER_LEG, sample_every=HORIZON, beta_neutral=True,
        )
        r_pm = portfolio_pnl_pred_momentum_bn(
            test, pred_test, top_k=TOP_K, M_cycles=2, band_mult=1.0,
            cost_bps_per_leg=COST_PER_LEG, sample_every=HORIZON,
        )
        if r_base.get("n_bars", 0) == 0 or r_pm.get("n_bars", 0) == 0:
            continue
        b_df = r_base["df"][["time", "net_bps"]].rename(columns={"net_bps": "base_net"})
        p_df = r_pm["df"][["time", "net_bps", "n_long", "n_short", "cost_bps", "long_turnover"]].rename(
            columns={"net_bps": "pm_net", "cost_bps": "pm_cost", "long_turnover": "pm_lto",
                     "n_long": "pm_nl", "n_short": "pm_ns"}
        )
        merged = b_df.merge(p_df, on="time", how="inner")
        merged["fold"] = fold["fid"]
        merged["delta"] = merged["pm_net"] - merged["base_net"]
        base_all.extend(merged["base_net"].tolist())
        pm_all.extend(merged["pm_net"].tolist())
        fold_records.append(merged)
        print(f"  fold {fold['fid']:>2}: {len(merged):>3} cyc  "
              f"base_net={merged['base_net'].mean():+.2f}  pm_net={merged['pm_net'].mean():+.2f}  "
              f"Δ={merged['delta'].mean():+.2f} (wins {(merged['delta']>0).mean()*100:.0f}%)  "
              f"K_avg={(merged['pm_nl'].mean() + merged['pm_ns'].mean())/2:.1f}")

    base_arr = np.array(base_all)
    pm_arr = np.array(pm_all)
    delta = pm_arr - base_arr

    print("\n" + "=" * 100)
    print(f"HARD-SPLIT FROZEN TEST  ({len(delta)} cycles, frozen model, NO retraining)")
    print("=" * 100)

    base_sh, base_lo, base_hi = block_bootstrap_ci(base_arr, statistic=_sharpe,
                                                     block_size=7, n_boot=2000)
    pm_sh, pm_lo, pm_hi = block_bootstrap_ci(pm_arr, statistic=_sharpe,
                                              block_size=7, n_boot=2000)
    delta_sh = _sharpe(delta)

    # Bootstrap CI on Δnet
    rng = np.random.default_rng(42)
    block = 7
    n_blocks = int(np.ceil(len(delta) / block))
    n_boot = 2000
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, len(delta) - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:len(delta)]
        boot_means[i] = delta[idx].mean()
    delta_lo, delta_hi = np.percentile(boot_means, [2.5, 97.5])

    t_stat = delta.mean() / (delta.std() / np.sqrt(len(delta))) if delta.std() > 0 else 0
    p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    paired = pd.concat(fold_records, ignore_index=True)
    per_fold = paired.groupby("fold").agg(
        n=("delta", "size"),
        base_net=("base_net", "mean"),
        pm_net=("pm_net", "mean"),
        delta=("delta", "mean"),
    ).reset_index()
    per_fold["fold_dsh"] = paired.groupby("fold").apply(
        lambda x: (_sharpe(x["pm_net"].to_numpy()) - _sharpe(x["base_net"].to_numpy()))
    ).values
    n_pos_net = int((per_fold["delta"] > 0).sum())
    n_pos_sh = int((per_fold["fold_dsh"] > 0).sum())

    print(f"  Baseline (frozen)  Sharpe {base_sh:+.2f}  [{base_lo:+.2f}, {base_hi:+.2f}]   net {base_arr.mean():+.2f} bps/cyc")
    print(f"  PM_M2_b1 (frozen)  Sharpe {pm_sh:+.2f}  [{pm_lo:+.2f}, {pm_hi:+.2f}]   net {pm_arr.mean():+.2f} bps/cyc")
    print(f"  Δ                  Sharpe {delta_sh:+.2f}   net {delta.mean():+.3f} bps/cyc   "
          f"95% CI [{delta_lo:+.3f}, {delta_hi:+.3f}] bps   t={t_stat:+.2f} (p={p_val:.4f})")
    print(f"  Per-fold           net-positive: {n_pos_net}/{len(per_fold)}   "
          f"Sharpe-positive: {n_pos_sh}/{len(per_fold)}   "
          f"PM-wins-cycles: {(delta>0).mean()*100:.1f}%")
    print(f"  Per-fold Δnet (bps): {per_fold['delta'].apply(lambda v: f'{v:+.2f}').tolist()}")
    print(f"  Per-fold Δsh:        {per_fold['fold_dsh'].apply(lambda v: f'{v:+.2f}').tolist()}")

    # Frozen-survives-test verdict
    survives = (pm_sh - base_sh > 0.20) and delta.mean() > 0
    print(f"\n  Frozen test verdict: {'SURVIVES' if survives else 'FAILS'} (ΔSharpe > +0.20 AND Δnet > 0)")

    paired.to_csv(OUT_DIR / "hardsplit_pairs.csv", index=False)
    per_fold.to_csv(OUT_DIR / "hardsplit_per_fold.csv", index=False)
    summary = {
        "n_cycles": len(delta), "n_test_folds": len(per_fold),
        "base_sharpe": base_sh, "pm_sharpe": pm_sh,
        "base_ci": [base_lo, base_hi], "pm_ci": [pm_lo, pm_hi],
        "delta_sharpe": delta_sh, "delta_mean_bps": float(delta.mean()),
        "delta_mean_ci": [float(delta_lo), float(delta_hi)],
        "delta_t": float(t_stat), "delta_p": float(p_val),
        "n_folds_net_positive": n_pos_net,
        "n_folds_sharpe_positive": n_pos_sh,
        "survives": bool(survives),
    }
    with open(OUT_DIR / "hardsplit_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
