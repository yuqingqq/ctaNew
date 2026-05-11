"""Hard-split frozen test for conv+PM stacked (LGBM-only deployment config).

Tier 1A.2 follow-up: PM_M2_b1 alone passed hard-split (ΔSharpe +2.01).
The composed conv+PM hasn't been hard-split tested directly. This script
reapplies the frozen-ensemble protocol from alpha_v9_pred_momentum_hardsplit
but evaluates all four variants (baseline, conv_p30, PM_M2_b1, conv+PM)
on the frozen model.

Pass criterion: conv+PM ΔSharpe > +0.20 over baseline with frozen model AND
conv+PM Δnet > 0. If gate works structurally (filters noise universally
without depending on per-fold retraining), passes. If gate's edge depends
on the model being retrained per fold, fails.
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
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v8_h48_audit import build_wide_panel
from ml.research.alpha_v9_pred_momentum_stack import evaluate_stacked

HORIZON = 48
TOP_K = 7
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
OUT_DIR = REPO / "outputs/pm_stack_hardsplit"
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

    variants = [
        ("baseline", False, False),
        ("conv_p30", True,  False),
        ("PM_M2_b1", False, True),
        ("conv+PM",  True,  True),
    ]
    cycles_by_var: dict[str, list] = {v[0]: [] for v in variants}

    print(f"\n  {'fold':>4} {'cycles':>6}", end="")
    for v, _, _ in variants:
        print(f"  {v[:10]:>10}", end="")
    print()

    for fold in test_folds:
        _, _, test = _slice(panel, fold)
        Xtest = test[avail_feats].to_numpy(dtype=np.float32)
        pred_test = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                              for m in frozen_models], axis=0)
        line = f"  {fold['fid']:>4}"
        n = None
        for name, use_conv, use_pm in variants:
            df_eval = evaluate_stacked(test, pred_test,
                                        use_conv_gate=use_conv, use_pm_gate=use_pm)
            if df_eval.empty:
                continue
            n = len(df_eval)
            for _, row in df_eval.iterrows():
                cycles_by_var[name].append({
                    "fold": fold["fid"], "time": row["time"],
                    "net": row["net_bps"], "skipped": row["skipped"],
                    "n_long": row["n_long"], "n_short": row["n_short"],
                    "cost": row["cost_bps"],
                })
            mean_net = df_eval["net_bps"].mean()
            sh = _sharpe(df_eval["net_bps"].to_numpy())
            line += f"  {mean_net:+5.2f}/{sh:+4.1f}"
        if n: line = f"  {fold['fid']:>4} {n:>6}" + line[len(f"  {fold['fid']:>4}"):]
        print(line)

    # ===== Headline =====
    print("\n" + "=" * 100)
    print(f"HARD-SPLIT FROZEN — conv+PM stack  (frozen LGBM, NO retraining)")
    print("=" * 100)

    rows = []
    nets = {}
    for name, _, _ in variants:
        df_v = pd.DataFrame(cycles_by_var[name])
        if df_v.empty: continue
        net = df_v["net"].to_numpy()
        nets[name] = net
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        rows.append({
            "variant": name, "n": len(net),
            "net_bps": net.mean(),
            "cost_bps": df_v["cost"].mean(),
            "skip_pct": df_v["skipped"].mean() * 100,
            "K_avg": (df_v["n_long"].mean() + df_v["n_short"].mean()) / 2,
            "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format="%+.2f"))

    # ===== Paired Δ vs frozen baseline =====
    print("\n  Paired Δ vs frozen baseline:")
    base_net = nets["baseline"]
    paired_records = []
    for name in ["conv_p30", "PM_M2_b1", "conv+PM"]:
        if name not in nets: continue
        n_min = min(len(base_net), len(nets[name]))
        delta = nets[name][:n_min] - base_net[:n_min]
        rng = np.random.default_rng(42)
        n_boot = 2000; block = 7
        n_blocks = int(np.ceil(len(delta) / block))
        boot_means = np.empty(n_boot)
        for i in range(n_boot):
            starts = rng.integers(0, len(delta) - block + 1, size=n_blocks)
            idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:len(delta)]
            boot_means[i] = delta[idx].mean()
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        d_sh = _sharpe(nets[name][:n_min]) - _sharpe(base_net[:n_min])
        t_stat = delta.mean() / (delta.std() / np.sqrt(len(delta))) if delta.std() > 0 else 0
        sig = "✓" if lo > 0 else (" " if hi > 0 else "✗")
        print(f"    {name:<10}  Δnet={delta.mean():+.3f} bps  CI=[{lo:+.3f}, {hi:+.3f}]{sig}  "
              f"Δsh={d_sh:+.2f}  t={t_stat:+.2f}")
        paired_records.append({
            "variant": name, "delta_net": float(delta.mean()),
            "delta_ci": [float(lo), float(hi)], "delta_sh": float(d_sh),
            "t_stat": float(t_stat), "ci_lo_positive": lo > 0,
        })

    # Per-fold breakdown
    print("\n  Per-fold Δsh vs baseline (frozen):")
    print(f"    {'fold':>4} {'cyc':>4}", end="")
    for n in ["conv_p30", "PM_M2_b1", "conv+PM"]:
        print(f"  {n:>10}", end="")
    print()
    for fid in sorted(set(r["fold"] for r in cycles_by_var["baseline"])):
        line = f"    {fid:>4}"
        cyc_count = None
        for n in ["conv_p30", "PM_M2_b1", "conv+PM"]:
            base_f = [r["net"] for r in cycles_by_var["baseline"] if r["fold"] == fid]
            v_f = [r["net"] for r in cycles_by_var[n] if r["fold"] == fid]
            n_min_f = min(len(base_f), len(v_f))
            if n_min_f == 0: continue
            cyc_count = n_min_f
            d_sh = _sharpe(np.array(v_f[:n_min_f])) - _sharpe(np.array(base_f[:n_min_f]))
            line += f"  {d_sh:+9.2f}"
        line = f"    {fid:>4} {cyc_count if cyc_count else '?':>4}" + line[len(f"    {fid:>4}"):]
        print(line)

    # Verdict
    convpm_pass = nets.get("conv+PM") is not None and \
                  (_sharpe(nets["conv+PM"]) - _sharpe(nets["baseline"]) > 0.20) and \
                  (nets["conv+PM"].mean() > nets["baseline"].mean())
    print(f"\n  Frozen-test verdict (conv+PM): {'SURVIVES' if convpm_pass else 'FAILS'}")
    print(f"    (criteria: conv+PM ΔSharpe > +0.20 AND Δnet > 0 vs frozen baseline)")

    summary.to_csv(OUT_DIR / "hardsplit_summary.csv", index=False)
    for name, _, _ in variants:
        if cycles_by_var[name]:
            pd.DataFrame(cycles_by_var[name]).to_csv(
                OUT_DIR / f"{name}_hardsplit_cycles.csv", index=False
            )
    with open(OUT_DIR / "paired.json", "w") as f:
        json.dump(paired_records, f, indent=2, default=str)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
