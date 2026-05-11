"""conv+PM gate stack on the production hybrid (LGBM + Ridge_positioning).

Deploy-blocker Tier 1A: prior tests of PM_M2_b1 / conv+PM ran on PURE LGBM
predictions. Production model is `0.9 × z(lgbm_pred) + 0.1 × z(ridge_pos_pred)`.
This script reapplies the validated stack on the actual deployment predictor.

Question: does the gate's lift survive when the predictions already include
the Ridge head's positioning info? Three possibilities:
  - Pass: Ridge head and PM gate use orthogonal info → stacked Sharpe ~ +3.0-3.2
  - Fail-orthogonal: gate disrupts Ridge contribution → similar to LGBM-only +2.75
  - Fail-overlap: Ridge captures persistence info already → stacked Sharpe < +2.5

Reuses `predict_fold` (LGBM ensemble + Ridge fit + z-score blend) from
`alpha_v9_hybrid_validate.py` and `evaluate_stacked` (4-variant gate eval)
from `alpha_v9_pred_momentum_stack.py`.
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
from ml.research.alpha_v4_xs_1d import _multi_oos_splits
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v9_hybrid_validate import predict_fold, POS_FEATURES
from ml.research.alpha_v9_positioning_pack import build_panel
from ml.research.alpha_v9_pred_momentum_stack import evaluate_stacked

HORIZON = 48
TOP_K = 7
COST_PER_LEG = 4.5
CYCLES_PER_YEAR = (288 * 365) / HORIZON
GATE_PCTILE = 0.30
RIDGE_BLEND_W = 0.10
OUT_DIR = REPO / "outputs/pm_hybrid_stack"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _sharpe(x: np.ndarray) -> float:
    if len(x) == 0 or x.std() == 0:
        return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def main():
    panel = build_panel()  # includes positioning features
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    print(f"Multi-OOS folds: {len(folds)}")
    print(f"Hybrid blend: {(1-RIDGE_BLEND_W):.2f} × z(LGBM) + {RIDGE_BLEND_W:.2f} × z(Ridge_pos)")
    print(f"Positioning features: {POS_FEATURES}")

    variants = [
        ("baseline", False, False),
        ("conv_p30", True,  False),
        ("PM_M2_b1", False, True),
        ("conv+PM",  True,  True),
    ]
    cycles_by_var: dict[str, list] = {v[0]: [] for v in variants}
    cycles_by_var_lgbm_only: dict[str, list] = {v[0]: [] for v in variants}

    for fold in folds:
        t0 = time.time()
        test, lgbm_z, ridge_z = predict_fold(panel, fold, v6_clean, POS_FEATURES)
        if test is None:
            print(f"  fold {fold['fid']:>2}: skipped"); continue

        # Hybrid prediction: 0.9 z(LGBM) + 0.1 z(Ridge_pos)
        hybrid_pred = (1 - RIDGE_BLEND_W) * lgbm_z + RIDGE_BLEND_W * ridge_z

        # Also evaluate LGBM-only as baseline reproduction (sanity check)
        line = f"  fold {fold['fid']:>2}:  hybrid: "
        for name, use_conv, use_pm in variants:
            df_eval = evaluate_stacked(test, hybrid_pred,
                                        use_conv_gate=use_conv, use_pm_gate=use_pm)
            if df_eval.empty:
                continue
            for _, row in df_eval.iterrows():
                cycles_by_var[name].append({
                    "fold": fold["fid"], "time": row["time"],
                    "spread": row["spread_ret_bps"], "cost": row["cost_bps"],
                    "net": row["net_bps"], "skipped": row["skipped"],
                    "n_long": row["n_long"], "n_short": row["n_short"],
                })
            net_arr = df_eval["net_bps"].to_numpy()
            line += f"{name}={net_arr.mean():+.2f}({_sharpe(net_arr):+.1f})  "
        print(line + f"({time.time()-t0:.0f}s)")

        # LGBM-only on same fold for direct comparison
        for name, use_conv, use_pm in variants:
            df_eval = evaluate_stacked(test, lgbm_z,
                                        use_conv_gate=use_conv, use_pm_gate=use_pm)
            if df_eval.empty:
                continue
            for _, row in df_eval.iterrows():
                cycles_by_var_lgbm_only[name].append({
                    "fold": fold["fid"], "time": row["time"],
                    "net": row["net_bps"], "skipped": row["skipped"],
                })

    # ===== Headline =====
    print("\n" + "=" * 110)
    print(f"HYBRID (0.9 LGBM + 0.1 Ridge_pos) — conv+PM stack  (h={HORIZON} K={TOP_K} 4.5 bps/leg β-neutral)")
    print("=" * 110)

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
            "spread_bps": df_v["spread"].mean(),
            "cost_bps": df_v["cost"].mean(),
            "skip_pct": df_v["skipped"].mean() * 100,
            "K_avg": (df_v["n_long"].mean() + df_v["n_short"].mean()) / 2,
            "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format="%+.2f"))

    # ===== Paired Δ vs hybrid baseline =====
    print("\n  Paired Δ vs HYBRID baseline:")
    base_net = nets["baseline"]
    for name in ["conv_p30", "PM_M2_b1", "conv+PM"]:
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
        print(f"    {name:<10}  Δnet={delta.mean():+.3f} bps  CI95=[{lo:+.3f}, {hi:+.3f}]  "
              f"Δsh={d_sh:+.2f}  t={t_stat:+.2f}")

    # ===== Compositionality check =====
    if all(k in nets for k in ["baseline", "conv_p30", "PM_M2_b1", "conv+PM"]):
        n_min = min(len(nets[k]) for k in nets)
        d_conv = nets["conv_p30"][:n_min] - nets["baseline"][:n_min]
        d_pm = nets["PM_M2_b1"][:n_min] - nets["baseline"][:n_min]
        d_stk = nets["conv+PM"][:n_min] - nets["baseline"][:n_min]
        additive = d_conv.mean() + d_pm.mean()
        actual = d_stk.mean()
        pct = (100 * actual / additive) if additive != 0 else 0
        print(f"\n  Compositionality (hybrid):")
        print(f"    Δnet conv alone:  {d_conv.mean():+.3f} bps")
        print(f"    Δnet PM alone:    {d_pm.mean():+.3f} bps")
        print(f"    Sum if additive:  {additive:+.3f} bps")
        print(f"    Δnet conv+PM:     {actual:+.3f} bps  ({pct:+.0f}% of additive)")

    # ===== Comparison: hybrid vs LGBM-only at each variant =====
    print("\n" + "=" * 110)
    print("HYBRID vs LGBM-ONLY  (does Ridge head add to each gate variant?)")
    print("=" * 110)
    print(f"  {'variant':<10}  {'lgbm_only_net':>14} {'lgbm_only_sh':>13}  "
          f"{'hybrid_net':>11} {'hybrid_sh':>10}  {'Δsh':>6}  {'Δnet':>6}")
    for name, _, _ in variants:
        df_h = pd.DataFrame(cycles_by_var[name])
        df_l = pd.DataFrame(cycles_by_var_lgbm_only[name])
        if df_h.empty or df_l.empty: continue
        h_net = df_h["net"].mean(); h_sh = _sharpe(df_h["net"].to_numpy())
        l_net = df_l["net"].mean(); l_sh = _sharpe(df_l["net"].to_numpy())
        print(f"  {name:<10}  {l_net:>+14.2f} {l_sh:>+13.2f}  "
              f"{h_net:>+11.2f} {h_sh:>+10.2f}  {h_sh - l_sh:>+6.2f}  {h_net - l_net:>+6.2f}")

    # Save
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    for name, _, _ in variants:
        if cycles_by_var[name]:
            pd.DataFrame(cycles_by_var[name]).to_csv(OUT_DIR / f"{name}_hybrid_cycles.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
