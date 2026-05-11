"""Test Ridge variants on outlier-trimmed training data.

Combines two interventions:
  1. Outlier trimming (drop |target_A| > sigma)
  2. Linear models (Ridge_raw + Ridge_ranked)

Hypothesis: extreme outliers (Oct 2025 dump etc.) dominate squared-error loss,
so Ridge fails on raw data. After trimming, Ridge should improve.

Configs: 3 trim levels × 2 Ridge variants = 6 cells.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v9_pred_momentum_stack import evaluate_stacked

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants.parquet"
IC_PATH = REPO / "outputs/vBTC_universe_noleak/calibration_ic.csv"
OUT_DIR = REPO / "outputs/vBTC_ridge_trimmed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
PROD_FOLDS = [5, 6, 7, 8, 9]
TOP_N = 15
K = 3

TRIM_LEVELS = [
    ("baseline",      None),
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


def rank_transform(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    out = df.copy()
    for f in feat_cols:
        if f == "sym_id" or f not in df.columns: continue
        out[f] = df.groupby("open_time")[f].rank(pct=True)
    for f in feat_cols:
        if f in out.columns: out[f] = out[f].fillna(0.5)
    return out


def main():
    print(f"Loading data...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    df_ic = pd.read_csv(IC_PATH).sort_values("ic", ascending=False)
    universe = set(df_ic["symbol"].head(TOP_N).tolist())
    print(f"  Universe: {sorted(universe)}", flush=True)

    feat_set = list(XS_FEATURE_COLS_V6_CLEAN)
    folds_all = _multi_oos_splits(panel)

    # 6 configs: 3 trim × 2 ridge variants
    config_results = {}

    for trim_label, sigma in TRIM_LEVELS:
        for ridge_label in ["Ridge_raw", "Ridge_ranked"]:
            label = f"{ridge_label}_{trim_label}"
            print(f"\n=== {label} ===", flush=True)
            cycles = []
            ics = []

            for fid in PROD_FOLDS:
                if fid >= len(folds_all): continue
                t0 = time.time()
                train, cal, test = _slice(panel, folds_all[fid])
                tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
                ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
                tr_full = pd.concat([tr, ca], axis=0)
                if len(tr_full) < 1000: continue

                # Trim outliers
                tr_n0 = len(tr_full)
                if sigma is not None:
                    tr_full = tr_full[tr_full["target_A"].abs() <= sigma]
                tr_n1 = len(tr_full)

                if ridge_label == "Ridge_ranked":
                    tr_in = rank_transform(tr_full, feat_set)
                    test_in = rank_transform(test, feat_set)
                    Xtr = tr_in[feat_set].fillna(0.5).to_numpy(np.float32)
                    Xtest = test_in[feat_set].fillna(0.5).to_numpy(np.float32)
                    fit_data = (Xtr, Xtest, None)
                else:  # Ridge_raw with StandardScaler
                    Xtr = tr_full[feat_set].fillna(0).to_numpy(np.float32)
                    Xtest = test[feat_set].fillna(0).to_numpy(np.float32)
                    scaler = StandardScaler().fit(Xtr)
                    Xtr = scaler.transform(Xtr)
                    Xtest = scaler.transform(Xtest)
                    fit_data = (Xtr, Xtest, scaler)

                ytr = tr_full["target_A"].to_numpy(np.float32)
                mask = ~np.isnan(ytr)
                Xtr, Xtest, _ = fit_data

                m = Ridge(alpha=1.0).fit(Xtr[mask], ytr[mask])
                pred = m.predict(Xtest)

                keep_mask = test["symbol"].isin(universe).to_numpy()
                test_f = test[keep_mask].copy()
                pred_f = pred[keep_mask]
                ic = _per_cycle_ic(test_f, pred_f, "alpha_A")
                ics.append(ic)
                df_eval = evaluate_stacked(test_f, pred_f, use_conv_gate=True, use_pm_gate=True, top_k=K)
                for _, r in df_eval.iterrows():
                    cycles.append({"fold": fid, "time": r["time"],
                                    "net": r["net_bps"], "cost": r["cost_bps"]})
                n_f = pd.DataFrame(cycles)
                n_f = n_f[n_f["fold"] == fid]["net"].to_numpy() if "fold" in n_f.columns else np.array([])
                sh_fold = _sharpe(n_f) if len(n_f) else 0
                print(f"  fold {fid}: trim {tr_n0:,}→{tr_n1:,}  IC={ic:+.4f}  "
                      f"Sh={sh_fold:+.2f}  ({time.time()-t0:.0f}s)", flush=True)

            df_v = pd.DataFrame(cycles)
            if df_v.empty: continue
            net = df_v["net"].to_numpy()
            sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
            config_results[label] = {
                "n": len(net), "mean_net": net.mean(), "cost": df_v["cost"].mean(),
                "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                "avg_ic": np.mean(ics) if ics else 0,
                "per_fold_sh": {fid: _sharpe(df_v[df_v["fold"] == fid]["net"].to_numpy())
                                  for fid in sorted(df_v["fold"].unique())},
            }

    # Summary
    print(f"\n{'=' * 100}", flush=True)
    print(f"RIDGE × OUTLIER-TRIM GRID (N={TOP_N}, K={K})", flush=True)
    print(f"{'=' * 100}", flush=True)
    print(f"  {'config':<28}  {'avg_IC':>8}  {'Sharpe':>7}  {'CI_lo':>7}  {'CI_hi':>7}  "
          f"{'mean':>7}", flush=True)
    for label, r in config_results.items():
        print(f"  {label:<28}  {r['avg_ic']:>+8.4f}  {r['sharpe']:>+7.2f}  "
              f"{r['ci_lo']:>+7.2f}  {r['ci_hi']:>+7.2f}  {r['mean_net']:>+7.2f}",
              flush=True)

    print(f"\n  Per-fold Sharpe:", flush=True)
    for label, r in config_results.items():
        cells = " ".join(f"{r['per_fold_sh'].get(f, 0):+5.2f}" for f in PROD_FOLDS)
        print(f"  {label:<28}  " + cells, flush=True)

    pd.DataFrame([{"config": l, **r} for l, r in config_results.items()]
                  ).to_csv(OUT_DIR / "ridge_trim_grid.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
