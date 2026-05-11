"""Pred-disp size-overlay test (port from xyz validated +33% DD reduction).

Sweep continuous size-overlay (lo, hi) parameters on top of conv+PM stack.
For each non-skipped cycle, scale legs by dispersion-percentile within
trailing 252 cycle history.

Target metrics:
  - Sharpe: must be ≥ baseline (xyz showed zero Sharpe cost)
  - Max drawdown: target 25-35% reduction (xyz showed 33%)
  - Losing months (Dec 2025, Apr 2026 in v6): target partial absorption

Baseline: conv+PM with no overlay (live-model, validated +2.47 multi-OOS).

Variants tested:
  (lo=0.30, hi=1.00) — linear from conv-threshold to full size
  (lo=0.50, hi=1.00) — floor at 50% (more conservative)
  (lo=0.30, hi=0.70) — cap at 70% (always reduced size)
  (lo=0.50, hi=0.80) — narrow band, mid-conservative
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
OUT_DIR = REPO / "outputs/disp_overlay"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _max_dd_bps(net_arr):
    """Max drawdown in bps on the cumulative net curve."""
    cum = np.cumsum(net_arr)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(dd.min())


def main():
    panel = build_wide_panel()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    avail_feats = [c for c in v6_clean if c in panel.columns]

    variants = [
        ("baseline_no_overlay",  None,  None),
        ("overlay_0.30-1.00",    0.30,  1.00),
        ("overlay_0.50-1.00",    0.50,  1.00),
        ("overlay_0.30-0.70",    0.30,  0.70),
        ("overlay_0.50-0.80",    0.50,  0.80),
    ]
    cycles_by_var = {v[0]: [] for v in variants}

    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            continue
        Xt = tr[avail_feats].to_numpy(np.float32); yt_ = tr["demeaned_target"].to_numpy(np.float32)
        Xc = ca[avail_feats].to_numpy(np.float32); yc_ = ca["demeaned_target"].to_numpy(np.float32)
        models = [_train(Xt, yt_, Xc, yc_, seed=s) for s in ENSEMBLE_SEEDS]
        Xtest = test[avail_feats].to_numpy(np.float32)
        pred_test = np.mean([m.predict(Xtest, num_iteration=m.best_iteration) for m in models], axis=0)

        line = f"  fold {fold['fid']:>2}: "
        for label, lo, hi in variants:
            df_eval = evaluate_stacked(
                test, pred_test,
                use_conv_gate=True, use_pm_gate=True,
                disp_overlay_lo=lo, disp_overlay_hi=hi,
            )
            for _, row in df_eval.iterrows():
                cycles_by_var[label].append({
                    "fold": fold["fid"], "time": row["time"],
                    "net": row["net_bps"], "skipped": row["skipped"],
                    "n_long": row["n_long"], "n_short": row["n_short"],
                    "cost": row["cost_bps"],
                    "gross_L": row["gross_L"], "gross_S": row["gross_S"],
                })
            net_arr = df_eval["net_bps"].to_numpy()
            tag = label[-15:]
            line += f"{tag}={net_arr.mean():+.2f}({_sharpe(net_arr):+.1f}) "
        print(line + f"({time.time()-t0:.0f}s)")

    # ===== Headline =====
    print("\n" + "=" * 110)
    print("PRED-DISP SIZE-OVERLAY  (multi-OOS, conv+PM stack, live-model)")
    print("=" * 110)
    print(f"  {'variant':<22}  {'n':>4}  {'mean_net':>9}  {'Sharpe':>7}  {'CI_lo':>7}  {'CI_hi':>7}  "
          f"{'max_DD':>8}  {'gross_avg':>9}")
    rows = []
    for label, lo, hi in variants:
        df_v = pd.DataFrame(cycles_by_var[label])
        if df_v.empty: continue
        net = df_v["net"].to_numpy()
        sh, ci_lo, ci_hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        max_dd = _max_dd_bps(net)
        gross_avg = (df_v["gross_L"].mean() + df_v["gross_S"].mean()) / 2
        rows.append({
            "variant": label, "lo": lo, "hi": hi, "n": len(net),
            "net_bps": net.mean(),
            "sharpe": sh, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "max_dd_bps": max_dd,
            "gross_avg": gross_avg,
        })
        print(f"  {label:<22}  {len(net):>4}  {net.mean():>+9.2f}  {sh:>+7.2f}  "
              f"{ci_lo:>+7.2f}  {ci_hi:>+7.2f}  {max_dd:>+8.0f}  {gross_avg:>+9.2f}")

    # ===== Δ vs baseline (no overlay) =====
    print("\n  Δ vs baseline_no_overlay:")
    base_net = np.array([r["net"] for r in cycles_by_var["baseline_no_overlay"]])
    base_sharpe = _sharpe(base_net)
    base_dd = _max_dd_bps(base_net)
    for label, lo, hi in variants[1:]:
        df_v = pd.DataFrame(cycles_by_var[label])
        v_net = df_v["net"].to_numpy()
        v_sharpe = _sharpe(v_net)
        v_dd = _max_dd_bps(v_net)
        n_min = min(len(base_net), len(v_net))
        delta = v_net[:n_min] - base_net[:n_min]
        rng = np.random.default_rng(42)
        block = 7; n_blocks = int(np.ceil(len(delta) / block)); n_boot = 2000
        boot_means = np.empty(n_boot)
        for i in range(n_boot):
            starts = rng.integers(0, len(delta) - block + 1, size=n_blocks)
            idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:len(delta)]
            boot_means[i] = delta[idx].mean()
        d_lo, d_hi = np.percentile(boot_means, [2.5, 97.5])
        dd_change_pct = 100 * (v_dd - base_dd) / abs(base_dd) if base_dd != 0 else 0
        print(f"    {label:<22}  Δsh={v_sharpe-base_sharpe:+.2f}  "
              f"Δnet={delta.mean():+.3f} bps  CI=[{d_lo:+.3f}, {d_hi:+.3f}]  "
              f"max_DD: {base_dd:+.0f} → {v_dd:+.0f} ({dd_change_pct:+.0f}%)")

    # ===== Monthly: focus on losing months Dec 2025, Apr 2026 =====
    print("\n  Monthly net per cycle (focus on losing months from baseline):")
    for label, _, _ in variants:
        df_v = pd.DataFrame(cycles_by_var[label])
        df_v["time"] = pd.to_datetime(df_v["time"])
        df_v["month"] = df_v["time"].dt.to_period("M")
        mly = df_v.groupby("month")["net"].agg(["mean", "sum", "size"])
        # Show 2025-12 and 2026-04 specifically + worst month
        target_months = ["2025-12", "2026-04"]
        line_parts = [f"    {label:<22}: "]
        for m_str in target_months:
            try:
                m = pd.Period(m_str, freq="M")
                if m in mly.index:
                    line_parts.append(f"{m_str}={mly.loc[m, 'mean']:+.2f} bps")
                else:
                    line_parts.append(f"{m_str}=n/a")
            except Exception:
                pass
        print("  ".join(line_parts))

    # Save
    pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False)
    for label, _, _ in variants:
        if cycles_by_var[label]:
            pd.DataFrame(cycles_by_var[label]).to_csv(
                OUT_DIR / f"{label.replace('.', 'p').replace('-', '_to_')}_cycles.csv", index=False
            )
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
