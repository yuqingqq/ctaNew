"""Full-mode validation of N=12 K=4 (and 2 alternates) with multi-seed ensemble.

Calibration: train seed=42 on folds 0-4, compute per-symbol IC, rank.
Production:  train 5 seeds on folds 5-9, ensemble predictions, evaluate
             on filtered universes for top configs from grid.

Configs evaluated:
  N=12 K=4 — grid winner
  N=15 K=3 — close second, slightly more diversified
  N=20 K=4 — wider universe, more capacity
  N=25 K=5 — broader baseline for comparison
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

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants.parquet"
IC_PATH = REPO / "outputs/vBTC_universe_noleak/calibration_ic.csv"
OUT_DIR = REPO / "outputs/vBTC_full_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON

PROD_FOLDS = [5, 6, 7, 8, 9]
SEEDS = (42, 1337, 7, 19, 2718)
CONFIGS = [
    ("N=12 K=4 (grid winner)", 12, 4),
    ("N=15 K=3",                15, 3),
    ("N=20 K=4",                20, 4),
    ("N=25 K=5 (broader)",      25, 5),
]


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def main():
    print(f"Loading data...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    df_ic = pd.read_csv(IC_PATH).sort_values("ic", ascending=False)
    sorted_syms = df_ic["symbol"].tolist()
    print(f"  Panel: {len(panel):,} rows", flush=True)
    print(f"  Top 25 by calibration IC: {sorted_syms[:25]}", flush=True)

    feat_set = list(XS_FEATURE_COLS_V6_CLEAN)
    folds_all = _multi_oos_splits(panel)

    # === Train ensemble on production folds ===
    print(f"\n=== Training {len(SEEDS)}-seed ensemble on {len(PROD_FOLDS)} production folds ===",
          flush=True)
    fold_data = {}   # {fid: (test_df, ensemble_pred)}
    for fid in PROD_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        train, cal, test = _slice(panel, folds_all[fid])
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200: continue
        Xt = tr[feat_set].to_numpy(np.float32)
        Xc = ca[feat_set].to_numpy(np.float32)
        Xtest = test[feat_set].to_numpy(np.float32)
        yt = tr["target_A"].to_numpy(np.float32)
        yc = ca["target_A"].to_numpy(np.float32)
        mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
        if mask_t.sum() < 1000 or mask_c.sum() < 200: continue
        models = [_train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s) for s in SEEDS]
        preds = np.array([m.predict(Xtest, num_iteration=m.best_iteration) for m in models])
        ensemble = preds.mean(axis=0)
        fold_data[fid] = (test.copy(), ensemble)
        print(f"  fold {fid}: ensembled {len(SEEDS)} seeds, {len(test):,} test rows "
              f"({time.time()-t0:.0f}s)", flush=True)

    # === Evaluate each config ===
    print(f"\n=== Configurations (no-leak, 5-seed ensemble, 5 production folds) ===", flush=True)
    print(f"  {'config':<28}  {'avg_IC':>7}  {'n':>4}  {'mean':>7}  {'cost':>6}  "
          f"{'Sharpe':>7}  {'CI_lo':>7}  {'CI_hi':>7}  {'maxDD':>7}", flush=True)

    summary = []
    for label, n, k in CONFIGS:
        universe = set(sorted_syms[:n])
        avg_ic = df_ic.iloc[:n]["ic"].mean()
        cycles = []
        for fid, (test_df, pred) in fold_data.items():
            keep_mask = test_df["symbol"].isin(universe).to_numpy()
            df = evaluate_stacked(test_df[keep_mask].copy(), pred[keep_mask],
                                   use_conv_gate=True, use_pm_gate=True, top_k=k)
            for _, r in df.iterrows():
                cycles.append({"fold": fid, "time": r["time"],
                                "net": r["net_bps"], "cost": r["cost_bps"]})
        df_v = pd.DataFrame(cycles)
        if df_v.empty: continue
        net = df_v["net"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        cost = df_v["cost"].mean()
        max_dd = _max_dd(net)
        per_fold_sh = {}
        for fid in sorted(df_v["fold"].unique()):
            n_f = df_v[df_v["fold"] == fid]["net"].to_numpy()
            per_fold_sh[fid] = _sharpe(n_f) if len(n_f) >= 3 else 0.0
        summary.append({"config": label, "N": n, "K": k, "avg_ic": avg_ic, "n_cycles": len(net),
                          "mean_net": net.mean(), "cost": cost,
                          "sharpe": sh, "ci_lo": lo, "ci_hi": hi, "max_dd": max_dd,
                          **{f"sh_f{f}": v for f, v in per_fold_sh.items()}})
        print(f"  {label:<28}  {avg_ic:>+7.4f}  {len(net):>4}  "
              f"{net.mean():>+7.2f}  {cost:>+6.2f}  "
              f"{sh:>+7.2f}  {lo:>+7.2f}  {hi:>+7.2f}  {max_dd:>+7.0f}", flush=True)

    # Per-fold detail for top config
    print(f"\n=== Per-fold breakdown ===", flush=True)
    print(f"  {'config':<28}  " + " ".join(f"{'fold' + str(f):>9}" for f in PROD_FOLDS), flush=True)
    for r in summary:
        cells = [f"{r.get(f'sh_f{f}', 0):+5.2f}" for f in PROD_FOLDS]
        print(f"  {r['config']:<28}  " + " ".join(f"{c:>9}" for c in cells), flush=True)

    pd.DataFrame(summary).to_csv(OUT_DIR / "validation_summary.csv", index=False)
    for r in summary:
        # Save cycles per config
        pass  # already in outputs/vBTC_k_sweep
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
