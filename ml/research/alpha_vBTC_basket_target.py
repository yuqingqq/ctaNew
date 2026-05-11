"""Tier 1 retrain: v6_clean architecture on 51-name universe with basket-residual target.

Diagnostic from BTC-target Phase 4 showed the model was fitting BTC-regime
patterns that don't generalize. The structural fix is to switch the target
from `btc_target` (BTC-residual, contains alt-common factor contamination)
to `alpha_realized` (basket-residual, alt-common factor already removed).

This run keeps everything else identical to v6_clean:
  - 28-feature v6_clean set (no BTC-regime features)
  - LGBM (same hyperparameters)
  - conv_gate + PM_M2_b1
  - h=48
  - cross-sectional long top-K vs short bot-K (no BTC leg)

Decision criteria vs v6_clean baseline +2.47 Sharpe:
  - ≥+2.0  PASS, proceed to full ensemble validation
  - +1.0 to +2.0  MARGINAL, investigate cluster-decoupling
  - <+1.0  unexpected — investigate panel quality
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v9_pred_momentum_stack import evaluate_stacked

FAST_SEEDS = (42,)
FAST_N_FOLDS = 3

PANEL_PATH = REPO / "outputs/vBTC_features/panel_with_btc_features.parquet"
OUT_DIR = REPO / "outputs/vBTC_basket_target"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
V6_CLEAN_BASELINE = 2.47


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def main():
    print(f"Loading panel from {PANEL_PATH}...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    print(f"  {len(panel):,} rows × {panel.shape[1]} cols, {panel['symbol'].nunique()} syms", flush=True)

    feat_set = list(XS_FEATURE_COLS_V6_CLEAN)
    missing = [c for c in feat_set if c not in panel.columns]
    if missing:
        raise RuntimeError(f"Missing features: {missing}")
    print(f"  feature set: {len(feat_set)} v6_clean features (no BTC-regime features)", flush=True)

    all_folds = _multi_oos_splits(panel)
    n_total = len(all_folds)
    fold_idx = [n_total // 5, n_total // 2, 4 * n_total // 5][:FAST_N_FOLDS]
    folds = [all_folds[i] for i in fold_idx if i < n_total]
    print(f"  multi-OOS folds: {len(folds)} of {n_total} (fast mode, indices {fold_idx})", flush=True)

    cycles = []   # cross-sectional only
    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            print(f"  fold {fold['fid']}: insufficient data — skipped", flush=True)
            continue

        Xt = tr[feat_set].to_numpy(np.float32)
        Xc = ca[feat_set].to_numpy(np.float32)
        Xtest = test[feat_set].to_numpy(np.float32)

        # Train on basket-residual target (z-scored — same column v6_clean uses)
        yt = tr["demeaned_target"].to_numpy(np.float32)
        yc = ca["demeaned_target"].to_numpy(np.float32)
        mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
        if mask_t.sum() < 1000 or mask_c.sum() < 200:
            continue

        models = [_train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s)
                  for s in FAST_SEEDS]
        pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration) for m in models], axis=0)

        df = evaluate_stacked(test, pred, use_conv_gate=True, use_pm_gate=True)
        for _, r in df.iterrows():
            cycles.append({"fold": fold["fid"], "time": r["time"],
                           "net": r["net_bps"], "cost": r["cost_bps"],
                           "skipped": r["skipped"]})

        df_t = pd.DataFrame(cycles)
        df_t = df_t[df_t["fold"] == fold["fid"]]
        n = df_t["net"].to_numpy() if not df_t.empty else np.array([])
        sh = _sharpe(n) if len(n) else 0.0
        mn = n.mean() if len(n) else 0.0
        print(f"  fold {fold['fid']:>2}: basket_xs={mn:+.2f}({sh:+.1f})  "
              f"({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 100, flush=True)
    print("TIER 1 — basket-residual on 51-name universe + 28 v6_clean features", flush=True)
    print("=" * 100, flush=True)

    df_v = pd.DataFrame(cycles)
    if df_v.empty:
        print("  no cycles — exiting", flush=True)
        return

    net = df_v["net"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
    max_dd = _max_dd(net)
    cost = df_v["cost"].mean()

    print(f"  {'config':<22}  {'n':>4}  {'mean_net':>9}  {'cost':>7}  "
          f"{'Sharpe':>7}  {'CI_lo':>7}  {'CI_hi':>7}  {'max_DD':>8}", flush=True)
    print(f"  {'basket_xs':<22}  {len(net):>4}  {net.mean():>+9.2f}  {cost:>+7.2f}  "
          f"{sh:>+7.2f}  {lo:>+7.2f}  {hi:>+7.2f}  {max_dd:>+8.0f}", flush=True)

    # Per-fold breakdown for sign-stability check
    print(f"\n  Per-fold Sharpe (sign-stability indicator):", flush=True)
    for fid in sorted(df_v["fold"].unique()):
        n_f = df_v[df_v["fold"] == fid]["net"].to_numpy()
        if len(n_f) >= 3:
            print(f"    fold {fid}: n={len(n_f):>3}  mean={n_f.mean():+6.2f}  Sharpe={_sharpe(n_f):+5.2f}",
                  flush=True)

    print(f"\n  Decision (vs v6_clean baseline Sharpe={V6_CLEAN_BASELINE}):", flush=True)
    if sh >= 2.0:
        verdict = "PASS — proceed to full ensemble validation"
    elif sh >= 1.0:
        verdict = "MARGINAL — investigate cluster decoupling on 51 names"
    else:
        verdict = "UNEXPECTED FAIL — investigate panel construction"
    print(f"    Sharpe={sh:+.2f}  Δsh vs v6_clean={sh-V6_CLEAN_BASELINE:+.2f}  → {verdict}", flush=True)

    pd.DataFrame([{"config": "basket_xs", "n": len(net), "mean_net": net.mean(),
                   "cost": cost, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                   "max_dd": max_dd}]).to_csv(OUT_DIR / "summary.csv", index=False)
    df_v.to_csv(OUT_DIR / "basket_xs_cycles.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
