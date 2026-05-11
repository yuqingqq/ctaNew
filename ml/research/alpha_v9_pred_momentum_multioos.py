"""Pred-momentum entry gate — full multi-OOS paired validation.

Discipline gate (per memory protocol):
  1. Multi-OOS Sharpe lift > +0.20 — *necessary but not sufficient*
  2. Block-bootstrap CI on Δnet_bps (paired, block_size=7) — bound > 0 ?
  3. Per-fold consistency: ≥ 6/10 folds with positive Δnet
  4. Hard-split frozen test (separate run) — does it survive no-retrain?

Tested variants:
  PM_M2_b1  strict — new entries must have been in top-K at t-1
  PM_M2_b2  loose  — new entries must have been in top-2K at t-1
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
OUT_DIR = REPO / "outputs/pred_momentum_multioos"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    ("PM_M2_b1", 2, 1.0),
    ("PM_M2_b2", 2, 2.0),
]


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

    # Per-cycle paired records
    pairs_by_variant: dict[str, list[pd.DataFrame]] = {v[0]: [] for v in VARIANTS}
    fold_summaries = []

    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            print(f"  fold {fold['fid']:>2}: skipped (insufficient data)")
            continue

        Xt = tr[avail_feats].to_numpy(dtype=np.float32)
        yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
        Xc = ca[avail_feats].to_numpy(dtype=np.float32)
        yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
        models = [_train(Xt, yt_, Xc, yc_, seed=s) for s in ENSEMBLE_SEEDS]
        Xtest = test[avail_feats].to_numpy(dtype=np.float32)
        pred_test = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                              for m in models], axis=0)

        r_base = portfolio_pnl_turnover_aware(
            test, pred_test, top_frac=TOP_FRAC,
            cost_bps_per_leg=COST_PER_LEG, sample_every=HORIZON, beta_neutral=True,
        )
        if r_base.get("n_bars", 0) == 0:
            print(f"  fold {fold['fid']:>2}: empty baseline"); continue

        base_df = r_base["df"][["time", "net_bps", "spread_ret_bps", "cost_bps",
                                "long_turnover", "short_turnover"]].rename(
            columns={c: f"base_{c}" for c in
                     ["net_bps", "spread_ret_bps", "cost_bps", "long_turnover", "short_turnover"]}
        )

        line = f"  fold {fold['fid']:>2}: {len(base_df):>3} cyc  base_net={base_df['base_net_bps'].mean():+.2f} bps"

        for label, M, b in VARIANTS:
            r_pm = portfolio_pnl_pred_momentum_bn(
                test, pred_test, top_k=TOP_K, M_cycles=M, band_mult=b,
                cost_bps_per_leg=COST_PER_LEG, sample_every=HORIZON,
            )
            if r_pm.get("n_bars", 0) == 0:
                continue
            pm_df = r_pm["df"][["time", "net_bps", "spread_ret_bps", "cost_bps",
                                "long_turnover", "short_turnover", "n_long", "n_short"]].rename(
                columns={c: f"pm_{c}" for c in
                         ["net_bps", "spread_ret_bps", "cost_bps", "long_turnover", "short_turnover", "n_long", "n_short"]}
            )
            merged = base_df.merge(pm_df, on="time", how="inner")
            merged["fold"] = fold["fid"]
            merged["variant"] = label
            merged["delta_net"] = merged["pm_net_bps"] - merged["base_net_bps"]
            pairs_by_variant[label].append(merged)
            d = merged["delta_net"].to_numpy()
            line += f"  | {label} pm_net={merged['pm_net_bps'].mean():+.2f}  Δ={d.mean():+.2f} (wins {(d>0).mean()*100:.0f}%)"

        fold_summaries.append({"fid": fold["fid"], "n": len(base_df)})
        print(f"{line}  ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 110)
    print(f"MULTI-OOS PAIRED VALIDATION  (h={HORIZON} K={TOP_K}, β-neutral, 4.5 bps/leg)")
    print("=" * 110)

    summary_records = []
    for label, _, _ in VARIANTS:
        if not pairs_by_variant[label]:
            print(f"  {label}: no pairs")
            continue
        paired = pd.concat(pairs_by_variant[label], ignore_index=True)
        base = paired["base_net_bps"].to_numpy()
        pm = paired["pm_net_bps"].to_numpy()
        delta = paired["delta_net"].to_numpy()

        base_sh, base_lo, base_hi = block_bootstrap_ci(base, statistic=_sharpe,
                                                       block_size=7, n_boot=2000)
        pm_sh, pm_lo, pm_hi = block_bootstrap_ci(pm, statistic=_sharpe,
                                                 block_size=7, n_boot=2000)
        delta_sh = _sharpe(delta)
        # Bootstrap CI on Δnet mean (per-cycle paired)
        delta_mean_lo, delta_mean_hi = (np.percentile(
            [_block_resample(delta, 7).mean() for _ in range(2000)],
            [2.5, 97.5]
        ))
        # Paired t-test on cycle-level delta
        t_stat = delta.mean() / (delta.std() / np.sqrt(len(delta))) if delta.std() > 0 else 0
        p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        # Per-fold breakdown
        per_fold = paired.groupby("fold").agg(
            n=("delta_net", "size"),
            base_net=("base_net_bps", "mean"),
            pm_net=("pm_net_bps", "mean"),
            delta=("delta_net", "mean"),
            cost_drop=("base_cost_bps", lambda s: paired.loc[s.index, "base_cost_bps"].mean()
                       - paired.loc[s.index, "pm_cost_bps"].mean()),
            K_avg=("pm_n_long", lambda s: (paired.loc[s.index, "pm_n_long"].mean()
                   + paired.loc[s.index, "pm_n_short"].mean()) / 2),
        ).reset_index()
        per_fold["fold_sharpe_base"] = paired.groupby("fold")["base_net_bps"].apply(
            lambda x: x.mean()/x.std()*np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0
        ).values
        per_fold["fold_sharpe_pm"] = paired.groupby("fold")["pm_net_bps"].apply(
            lambda x: x.mean()/x.std()*np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0
        ).values
        per_fold["fold_dsh"] = per_fold["fold_sharpe_pm"] - per_fold["fold_sharpe_base"]

        n_pos_net = int((per_fold["delta"] > 0).sum())
        n_pos_sh = int((per_fold["fold_dsh"] > 0).sum())
        n_folds_used = len(per_fold)

        print(f"\n  {label}  ({len(delta)} cycles across {n_folds_used} folds)")
        print(f"    Baseline   Sharpe {base_sh:+.2f}  [{base_lo:+.2f}, {base_hi:+.2f}]   net {base.mean():+.2f} bps/cyc")
        print(f"    {label:<10} Sharpe {pm_sh:+.2f}  [{pm_lo:+.2f}, {pm_hi:+.2f}]   net {pm.mean():+.2f} bps/cyc   "
              f"L_to {paired['pm_long_turnover'].mean()*100:.0f}%   K_avg {(paired['pm_n_long'].mean() + paired['pm_n_short'].mean())/2:.1f}")
        print(f"    Δ          Sharpe {delta_sh:+.2f}   net {delta.mean():+.3f} bps/cyc   "
              f"95% CI [{delta_mean_lo:+.3f}, {delta_mean_hi:+.3f}] bps   "
              f"t={t_stat:+.2f} (p={p_val:.4f})")
        print(f"    Per-fold   net-positive: {n_pos_net}/{n_folds_used}   "
              f"Sharpe-positive: {n_pos_sh}/{n_folds_used}   "
              f"PM-wins-cycles: {(delta>0).mean()*100:.1f}%")
        print(f"    Per-fold Δnet (bps): {per_fold['delta'].apply(lambda v: f'{v:+.2f}').tolist()}")
        print(f"    Per-fold Δsh:        {per_fold['fold_dsh'].apply(lambda v: f'{v:+.2f}').tolist()}")

        # Discipline gate verdict
        gate1 = pm_sh - base_sh > 0.20
        gate2 = delta_mean_lo > 0
        gate3 = n_pos_sh >= 6 if n_folds_used >= 9 else n_pos_sh / n_folds_used >= 0.6
        verdict = "PASSES gates 1+2+3" if (gate1 and gate2 and gate3) else "FAILS gates"
        print(f"    Discipline gates: ΔSharpe>+0.20 [{gate1}]  CI_lo>0 [{gate2}]  ≥60% folds Sharpe-pos [{gate3}]  → {verdict}")

        summary_records.append({
            "variant": label, "n_cycles": len(delta), "n_folds": n_folds_used,
            "base_sharpe": base_sh, "pm_sharpe": pm_sh,
            "base_ci": [base_lo, base_hi], "pm_ci": [pm_lo, pm_hi],
            "delta_sharpe": delta_sh, "delta_mean_bps": float(delta.mean()),
            "delta_mean_ci": [float(delta_mean_lo), float(delta_mean_hi)],
            "delta_t": float(t_stat), "delta_p": float(p_val),
            "n_folds_net_positive": n_pos_net,
            "n_folds_sharpe_positive": n_pos_sh,
            "pm_wins_cycles_pct": float((delta > 0).mean() * 100),
            "gate_passes": gate1 and gate2 and gate3,
            "per_fold_delta_net": per_fold["delta"].tolist(),
            "per_fold_delta_sharpe": per_fold["fold_dsh"].tolist(),
        })

        paired.to_csv(OUT_DIR / f"{label}_pairs.csv", index=False)
        per_fold.to_csv(OUT_DIR / f"{label}_per_fold.csv", index=False)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary_records, f, indent=2, default=str)
    print(f"\n  saved → {OUT_DIR}")


def _block_resample(x: np.ndarray, block_size: int) -> np.ndarray:
    n = len(x)
    n_blocks = int(np.ceil(n / block_size))
    starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
    out = np.concatenate([x[s:s+block_size] for s in starts])
    return out[:n]


if __name__ == "__main__":
    main()
