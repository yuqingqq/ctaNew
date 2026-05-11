"""Re-validate conv+PM with the live-model evaluator.

Issues 1+2 (HIGH severity, found 2026-05-09): the original `evaluate_stacked`
reset positions on conv-skip and PM-empty-leg, charging full re-entry cost
at next trade and recording 0 PnL during skip cycles. Live `paper_bot.py`
holds prior positions through these states, paying delta-only turnover
and accruing real MtM.

This script runs both execution models side-by-side on:
  1. Multi-OOS (10 folds, 1800 cycles)
  2. Hard-split frozen (folds 5-9 with model trained on folds 0-4)

For each, the conv+PM stacked Sharpe is compared between:
  research model: original behavior (reset on skip)
  live model:     hold-through (matches paper_bot)

Headline output: which Sharpe number to use for production validation.
"""
from __future__ import annotations
import sys, time, warnings, json
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import (
    ENSEMBLE_SEEDS, _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v8_h48_audit import build_wide_panel
from ml.research.alpha_v9_pred_momentum_stack import evaluate_stacked

HORIZON = 48
TOP_K = 7
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
OUT_DIR = REPO / "outputs/revalidate_live_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def main():
    panel = build_wide_panel()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    avail_feats = [c for c in v6_clean if c in panel.columns]

    # ====================================================================
    # PART 1: Multi-OOS comparison
    # ====================================================================
    print("=" * 100)
    print("PART 1: Multi-OOS conv+PM with live vs research execution models")
    print("=" * 100)

    cycles = {
        "baseline_research":  [],
        "baseline_live":       [],
        "convPM_research":    [],
        "convPM_live":          [],
    }

    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            print(f"  fold {fold['fid']}: skip"); continue
        Xt = tr[avail_feats].to_numpy(np.float32); yt_ = tr["demeaned_target"].to_numpy(np.float32)
        Xc = ca[avail_feats].to_numpy(np.float32); yc_ = ca["demeaned_target"].to_numpy(np.float32)
        models = [_train(Xt, yt_, Xc, yc_, seed=s) for s in ENSEMBLE_SEEDS]
        Xtest = test[avail_feats].to_numpy(np.float32)
        pred_test = np.mean([m.predict(Xtest, num_iteration=m.best_iteration) for m in models], axis=0)

        line = f"  fold {fold['fid']:>2}: "
        for variant_label, use_conv, use_pm, exec_model in [
            ("baseline_research",  False, False, "research"),
            ("baseline_live",       False, False, "live"),
            ("convPM_research",    True,  True,  "research"),
            ("convPM_live",          True,  True,  "live"),
        ]:
            df_eval = evaluate_stacked(
                test, pred_test,
                use_conv_gate=use_conv, use_pm_gate=use_pm,
                execution_model=exec_model,
            )
            for _, row in df_eval.iterrows():
                cycles[variant_label].append({
                    "fold": fold["fid"], "time": row["time"],
                    "net": row["net_bps"], "skipped": row["skipped"],
                })
            net_arr = df_eval["net_bps"].to_numpy()
            line += f"{variant_label[-12:]}={net_arr.mean():+.2f}({_sharpe(net_arr):+.1f}) "
        print(line + f"({time.time()-t0:.0f}s)")

    print("\n" + "=" * 100)
    print(f"MULTI-OOS RESULTS (1800 cycles, 10 folds)")
    print("=" * 100)
    print(f"  {'variant':<25}  {'mean_net':>9}  {'Sharpe':>7}  {'CI_lo':>7}  {'CI_hi':>7}")
    for label in ["baseline_research", "baseline_live", "convPM_research", "convPM_live"]:
        df_v = pd.DataFrame(cycles[label])
        if df_v.empty: continue
        net = df_v["net"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        print(f"  {label:<25}  {net.mean():>+9.2f}  {sh:>+7.2f}  {lo:>+7.2f}  {hi:>+7.2f}")

    # Δ live vs research at conv+PM stacked
    convPM_research = np.array([r["net"] for r in cycles["convPM_research"]])
    convPM_live = np.array([r["net"] for r in cycles["convPM_live"]])
    n_min = min(len(convPM_research), len(convPM_live))
    delta = convPM_live[:n_min] - convPM_research[:n_min]
    print(f"\n  Δ (live − research) on conv+PM:")
    print(f"    Mean Δnet:  {delta.mean():+.2f} bps/cyc  (live should be ≥0 from saved cost)")
    print(f"    Δsharpe:    {_sharpe(convPM_live) - _sharpe(convPM_research):+.2f}")
    print(f"    Cost diff:  research over-counts re-entry cost on every conv-skip cycle (~24%)")

    # ====================================================================
    # PART 2: Hard-split frozen comparison
    # ====================================================================
    print("\n" + "=" * 100)
    print("PART 2: Hard-split frozen with live vs research execution models")
    print("=" * 100)
    n_train_folds = max(3, len(folds) // 2)
    train_folds = folds[:n_train_folds]
    test_folds = folds[n_train_folds:]
    train_cutoff = train_folds[-1]["test_end"]
    panel_train = panel[panel["open_time"] < train_cutoff]
    panel_train_filt = panel_train[panel_train["autocorr_pctile_7d"] >= THRESHOLD]
    n_train = len(panel_train_filt)
    train_split = int(n_train * 0.85)
    Xt = panel_train_filt[avail_feats].iloc[:train_split].to_numpy(np.float32)
    yt_ = panel_train_filt["demeaned_target"].iloc[:train_split].to_numpy(np.float32)
    Xc = panel_train_filt[avail_feats].iloc[train_split:].to_numpy(np.float32)
    yc_ = panel_train_filt["demeaned_target"].iloc[train_split:].to_numpy(np.float32)
    print(f"  frozen training: {train_split:,} train rows, {n_train - train_split:,} cal rows")
    t0 = time.time()
    frozen_models = [_train(Xt, yt_, Xc, yc_, seed=s) for s in ENSEMBLE_SEEDS]
    print(f"  trained {len(frozen_models)} models in {time.time()-t0:.0f}s")

    hs_cycles = {
        "baseline_research":  [],
        "baseline_live":       [],
        "convPM_research":    [],
        "convPM_live":          [],
    }
    for fold in test_folds:
        _, _, test = _slice(panel, fold)
        Xtest = test[avail_feats].to_numpy(np.float32)
        pred_test = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                              for m in frozen_models], axis=0)
        for variant_label, use_conv, use_pm, exec_model in [
            ("baseline_research", False, False, "research"),
            ("baseline_live",      False, False, "live"),
            ("convPM_research",   True,  True,  "research"),
            ("convPM_live",         True,  True,  "live"),
        ]:
            df_eval = evaluate_stacked(
                test, pred_test,
                use_conv_gate=use_conv, use_pm_gate=use_pm,
                execution_model=exec_model,
            )
            for _, row in df_eval.iterrows():
                hs_cycles[variant_label].append({
                    "fold": fold["fid"], "time": row["time"],
                    "net": row["net_bps"], "skipped": row["skipped"],
                })

    print(f"\n  HARD-SPLIT FROZEN RESULTS")
    print(f"  {'variant':<25}  {'mean_net':>9}  {'Sharpe':>7}  {'CI_lo':>7}  {'CI_hi':>7}")
    for label in ["baseline_research", "baseline_live", "convPM_research", "convPM_live"]:
        df_v = pd.DataFrame(hs_cycles[label])
        if df_v.empty: continue
        net = df_v["net"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        print(f"  {label:<25}  {net.mean():>+9.2f}  {sh:>+7.2f}  {lo:>+7.2f}  {hi:>+7.2f}")

    # Hard-split paired Δ vs baseline (LIVE model)
    hs_base_live = np.array([r["net"] for r in hs_cycles["baseline_live"]])
    hs_convPM_live = np.array([r["net"] for r in hs_cycles["convPM_live"]])
    n_min_hs = min(len(hs_base_live), len(hs_convPM_live))
    delta_hs = hs_convPM_live[:n_min_hs] - hs_base_live[:n_min_hs]
    rng = np.random.default_rng(42)
    block = 7; n_blocks = int(np.ceil(len(delta_hs) / block)); n_boot = 2000
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, len(delta_hs) - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:len(delta_hs)]
        boot_means[i] = delta_hs[idx].mean()
    d_lo, d_hi = np.percentile(boot_means, [2.5, 97.5])
    print(f"\n  Hard-split conv+PM live Δ vs baseline_live:")
    print(f"    Δnet={delta_hs.mean():+.3f} bps  CI=[{d_lo:+.3f}, {d_hi:+.3f}]  "
          f"Δsh={_sharpe(hs_convPM_live) - _sharpe(hs_base_live):+.2f}")

    # Save
    for label, c in cycles.items():
        if c: pd.DataFrame(c).to_csv(OUT_DIR / f"multioos_{label}.csv", index=False)
    for label, c in hs_cycles.items():
        if c: pd.DataFrame(c).to_csv(OUT_DIR / f"hardsplit_{label}.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
