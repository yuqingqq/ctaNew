"""Path 3: test model with only top-10 features from audit.

Compares:
  baseline_28        — current v6_clean 28 features (Sharpe +1.97)
  top10              — only top-10 features by combined score
  top15              — top-15 (intermediate)

If top-10 holds Sharpe ~+1.97, then the 18 dropped features were noise.
Then we can confidently add funding rate / new features on top of the lean set.
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
OUT_DIR = REPO / "outputs/vBTC_top10_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
PROD_FOLDS = [5, 6, 7, 8, 9]
SEEDS = (42, 1337, 7, 19, 2718)
TOP_N = 15
K = 3

# Top-10 features by combined score from feature_audit
TOP10 = [
    "obv_z_1d",
    "mfi",
    "bk_ema_slope_4h",
    "dom_change_288b_vs_bk",
    "return_1d",
    "dom_z_7d_vs_bk",
    "ema_slope_20_1h",
    "return_1d_xs_rank",
    "vwap_slope_96",
    "corr_change_3d_vs_bk",
    "sym_id",   # pooling identifier
]

# Top-15 (adds 5 more in score order)
TOP15 = TOP10[:-1] + [
    "bk_ret_48b",
    "idio_vol_1d_vs_bk_xs_rank",
    "dom_level_vs_bk",
    "ema_slope_20_1h_xs_rank",
    "idio_ret_48b_vs_bk",
    "sym_id",
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


def run_config(panel: pd.DataFrame, folds: list, feat_set: list,
                universe: set, label: str) -> dict:
    print(f"\n=== Config: {label} ({len(feat_set)} features) ===", flush=True)
    cycles = []
    ics = []
    iter_log = []

    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
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

        preds = []
        seed_iters = []
        for s in SEEDS:
            m = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s)
            preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
            seed_iters.append(m.best_iteration)
        ensemble = np.mean(preds, axis=0)

        keep_mask = test["symbol"].isin(universe).to_numpy()
        test_f = test[keep_mask].copy()
        pred_f = ensemble[keep_mask]
        ic = _per_cycle_ic(test_f, pred_f, "alpha_A")
        ics.append(ic)
        df_eval = evaluate_stacked(test_f, pred_f, use_conv_gate=True, use_pm_gate=True, top_k=K)
        for _, r in df_eval.iterrows():
            cycles.append({"fold": fold["fid"], "time": r["time"],
                            "net": r["net_bps"], "cost": r["cost_bps"]})
        n_f = pd.DataFrame(cycles)
        n_f = n_f[n_f["fold"] == fold["fid"]]["net"].to_numpy() if "fold" in n_f.columns else np.array([])
        sh = _sharpe(n_f) if len(n_f) else 0
        print(f"  fold {fold['fid']}: iters={seed_iters} Sh={sh:+.2f} IC={ic:+.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        iter_log.append((fold["fid"], seed_iters))

    df_v = pd.DataFrame(cycles)
    if df_v.empty: return {}
    net = df_v["net"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
    cost = df_v["cost"].mean()
    per_fold = {fid: _sharpe(df_v[df_v["fold"] == fid]["net"].to_numpy())
                  for fid in sorted(df_v["fold"].unique())}
    return {"label": label, "n_features": len(feat_set), "n_cycles": len(net),
            "mean_net": net.mean(), "cost": cost,
            "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
            "avg_ic": np.mean(ics) if ics else 0,
            "per_fold": per_fold}


def main():
    print(f"Loading data...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    df_ic_calib = pd.read_csv(IC_PATH).sort_values("ic", ascending=False)
    universe = set(df_ic_calib["symbol"].head(TOP_N).tolist())
    print(f"  Universe: {sorted(universe)}", flush=True)

    folds_all = _multi_oos_splits(panel)
    folds = [folds_all[i] for i in PROD_FOLDS if i < len(folds_all)]

    configs = [
        ("baseline_28", list(XS_FEATURE_COLS_V6_CLEAN)),
        ("top15",       TOP15),
        ("top10",       TOP10),
    ]
    results = []
    for label, feats in configs:
        r = run_config(panel, folds, feats, universe, label)
        if r: results.append(r)

    print(f"\n{'=' * 100}", flush=True)
    print(f"FEATURE SET COMPARISON (N={TOP_N}, K={K}, 5-seed ensemble)", flush=True)
    print(f"{'=' * 100}", flush=True)
    print(f"  {'config':<14} {'n_feat':>6}  {'avg_IC':>8}  {'Sharpe':>7}  {'CI_lo':>7}  "
          f"{'CI_hi':>7}  {'mean':>6}", flush=True)
    for r in results:
        print(f"  {r['label']:<14} {r['n_features']:>6}  {r['avg_ic']:>+8.4f}  "
              f"{r['sharpe']:>+7.2f}  {r['ci_lo']:>+7.2f}  {r['ci_hi']:>+7.2f}  "
              f"{r['mean_net']:>+6.2f}", flush=True)

    print(f"\n  Per-fold Sharpe:", flush=True)
    for r in results:
        cells = " ".join(f"{r['per_fold'].get(f, 0):+5.2f}" for f in PROD_FOLDS)
        print(f"  {r['label']:<14}  " + cells, flush=True)

    pd.DataFrame(results).to_csv(OUT_DIR / "top10_comparison.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
