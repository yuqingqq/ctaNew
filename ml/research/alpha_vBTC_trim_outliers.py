"""Test outlier trimming on training data.

Hypothesis: extreme tail events (e.g., Oct 2025 dump) distort LGBM training,
causing fold-specific overfit and poor generalization. Trimming should help.

Configurations:
  baseline      no trimming
  trim_5sigma   drop train rows with |target_A| > 5
  trim_3sigma   drop train rows with |target_A| > 3
  trim_10sigma  drop train rows with |target_A| > 10 (very loose)

For each: 5-seed LGBM ensemble on N=15 K=3 universe, 5 production folds.
Test data is NOT trimmed — we evaluate on real-world conditions.
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
OUT_DIR = REPO / "outputs/vBTC_trim_outliers"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
PROD_FOLDS = [5, 6, 7, 8, 9]
SEEDS = (42, 1337, 7, 19, 2718)
TOP_N = 15
K = 3

CONFIGS = [
    ("baseline",      None),
    ("trim_10sigma",  10.0),
    ("trim_5sigma",    5.0),
    ("trim_3sigma",    3.0),
]


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _per_cycle_ic(test_df: pd.DataFrame, pred: np.ndarray, target_col: str) -> float:
    df = test_df.copy()
    df["pred"] = pred
    ics = []
    for t, g in df.groupby("open_time"):
        sub = g[[target_col, "pred"]].dropna()
        if len(sub) < 5: continue
        ic = sub["pred"].rank().corr(sub[target_col].rank())
        if not pd.isna(ic): ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0


def main():
    print(f"Loading data...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    df_ic_calib = pd.read_csv(IC_PATH).sort_values("ic", ascending=False)
    universe = set(df_ic_calib["symbol"].head(TOP_N).tolist())
    print(f"  Universe (top {TOP_N}): {sorted(universe)}", flush=True)

    feat_set = list(XS_FEATURE_COLS_V6_CLEAN)
    folds_all = _multi_oos_splits(panel)

    config_results = {}
    for label, sigma_thresh in CONFIGS:
        print(f"\n=== Config: {label} (sigma_thresh={sigma_thresh}) ===", flush=True)
        cycles = []
        ic_per_fold = {}
        trim_stats = []

        for fid in PROD_FOLDS:
            if fid >= len(folds_all): continue
            t0 = time.time()
            train, cal, test = _slice(panel, folds_all[fid])
            tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
            ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
            if len(tr) < 1000 or len(ca) < 200: continue

            # Trim train + cal if threshold given
            tr_n0, ca_n0 = len(tr), len(ca)
            if sigma_thresh is not None:
                tr = tr[tr["target_A"].abs() <= sigma_thresh]
                ca = ca[ca["target_A"].abs() <= sigma_thresh]
            tr_n1, ca_n1 = len(tr), len(ca)
            trim_stats.append((fid, tr_n0, tr_n1, ca_n0, ca_n1))

            Xt = tr[feat_set].to_numpy(np.float32)
            Xc = ca[feat_set].to_numpy(np.float32)
            Xtest = test[feat_set].to_numpy(np.float32)
            yt = tr["target_A"].to_numpy(np.float32)
            yc = ca["target_A"].to_numpy(np.float32)
            mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
            if mask_t.sum() < 1000 or mask_c.sum() < 200: continue

            preds = []
            iters = []
            for s in SEEDS:
                m = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s)
                preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
                iters.append(m.best_iteration)
            ensemble = np.mean(preds, axis=0)

            keep_mask = test["symbol"].isin(universe).to_numpy()
            test_f = test[keep_mask].copy()
            pred_f = ensemble[keep_mask]
            ic_per_fold[fid] = _per_cycle_ic(test_f, pred_f, "alpha_A")

            df_eval = evaluate_stacked(test_f, pred_f, use_conv_gate=True, use_pm_gate=True, top_k=K)
            for _, r in df_eval.iterrows():
                cycles.append({"fold": fid, "time": r["time"],
                                "net": r["net_bps"], "cost": r["cost_bps"]})
            n_f = pd.DataFrame(cycles); n_f = n_f[n_f["fold"] == fid]["net"].to_numpy()
            print(f"  fold {fid}: iters={iters} (trim: {tr_n0:,}→{tr_n1:,} train, "
                  f"{ca_n0:,}→{ca_n1:,} cal)  Sh={_sharpe(n_f):+.2f}  IC={ic_per_fold[fid]:+.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

        df_v = pd.DataFrame(cycles)
        if df_v.empty: continue
        net = df_v["net"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        config_results[label] = {
            "n": len(net), "mean_net": net.mean(), "cost": df_v["cost"].mean(),
            "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
            "avg_ic": np.mean(list(ic_per_fold.values())),
            "per_fold_sh": {fid: _sharpe(df_v[df_v["fold"] == fid]["net"].to_numpy())
                              for fid in sorted(df_v["fold"].unique())},
        }

    # Summary
    print(f"\n{'=' * 100}", flush=True)
    print(f"OUTLIER TRIM COMPARISON (N={TOP_N}, K={K}, 5-seed ensemble)", flush=True)
    print(f"{'=' * 100}", flush=True)
    print(f"  {'config':<14}  {'avg_IC':>8}  {'Sharpe':>7}  {'CI_lo':>7}  {'CI_hi':>7}  "
          f"{'mean':>7}  {'cost':>6}", flush=True)
    for label, _ in CONFIGS:
        if label not in config_results: continue
        r = config_results[label]
        print(f"  {label:<14}  {r['avg_ic']:>+8.4f}  {r['sharpe']:>+7.2f}  "
              f"{r['ci_lo']:>+7.2f}  {r['ci_hi']:>+7.2f}  "
              f"{r['mean_net']:>+7.2f}  {r['cost']:>+6.2f}", flush=True)

    print(f"\n  Per-fold Sharpe:", flush=True)
    for label, _ in CONFIGS:
        if label not in config_results: continue
        cells = " ".join(f"{config_results[label]['per_fold_sh'].get(f, 0):+5.2f}" for f in PROD_FOLDS)
        print(f"  {label:<14}  " + cells, flush=True)

    pd.DataFrame([{"config": l, **r} for l, r in config_results.items()]
                  ).to_csv(OUT_DIR / "trim_results.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
