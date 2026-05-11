"""A.1: no-leak rolling-IC universe filter.

Splits 10 folds into:
  - Calibration: folds 0-4 (compute per-symbol IC on these test sets)
  - Production:  folds 5-9 (filter universe based on calibration IC, evaluate)

The filter set is FIXED across all production folds (no rolling re-fit).
A more sophisticated version would use rolling IC; this is the simplest
honest test: select symbols using only data the production-period
evaluator wouldn't have seen at training time.

Total LGBM trainings: 10 (5 calib + 5 prod). ~30-40 min.
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
OUT_DIR = REPO / "outputs/vBTC_universe_noleak"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
SEEDS = (42,)

# Filter criteria (must match leaky version for direct comparison)
IC_MIN = 0.02
STAB_MIN = 0.8   # 4 of 5 calibration folds must agree on sign

CALIB_FOLDS = [0, 1, 2, 3, 4]
PROD_FOLDS  = [5, 6, 7, 8, 9]


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def _spearman(a, b):
    sub = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(sub) < 50: return np.nan
    return float(sub["a"].rank().corr(sub["b"].rank()))


def train_and_predict(panel, fold, feat_set):
    train, cal, test = _slice(panel, fold)
    tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
    ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
    if len(tr) < 1000 or len(ca) < 200: return None, None
    Xt = tr[feat_set].to_numpy(np.float32)
    Xc = ca[feat_set].to_numpy(np.float32)
    Xtest = test[feat_set].to_numpy(np.float32)
    yt = tr["target_A"].to_numpy(np.float32)
    yc = ca["target_A"].to_numpy(np.float32)
    mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
    if mask_t.sum() < 1000 or mask_c.sum() < 200: return None, None
    models = [_train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s) for s in SEEDS]
    pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration) for m in models], axis=0)
    return test, pred


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    print(f"  {len(panel):,} rows × {panel['symbol'].nunique()} syms", flush=True)

    feat_set = list(XS_FEATURE_COLS_V6_CLEAN)
    folds_all = _multi_oos_splits(panel)
    print(f"  Total folds: {len(folds_all)}", flush=True)
    print(f"  Calibration: {CALIB_FOLDS}  Production: {PROD_FOLDS}", flush=True)

    # === Step 1: Calibration — compute per-symbol IC on folds 0-4 ===
    print(f"\n=== STEP 1 — calibration trainings ===", flush=True)
    calib_data = []
    for fid in CALIB_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        test, pred = train_and_predict(panel, folds_all[fid], feat_set)
        if test is None: continue
        df = pd.DataFrame({"symbol": test["symbol"].to_numpy(),
                           "pred": pred, "alpha": test["alpha_A"].to_numpy()})
        calib_data.append(df.assign(fold=fid))
        print(f"  fold {fid}: {len(df):,} test rows  ({time.time()-t0:.0f}s)", flush=True)
    calib_df = pd.concat(calib_data, ignore_index=True).dropna()

    # Per-symbol IC across calibration period
    rows = []
    for s, g in calib_df.groupby("symbol"):
        if len(g) < 100: continue
        ic = _spearman(g["pred"], g["alpha"])
        per_fold = []
        for fid in g["fold"].unique():
            gf = g[g["fold"] == fid]
            if len(gf) >= 50:
                per_fold.append(_spearman(gf["pred"], gf["alpha"]))
        signs = [1 if i > 0 else -1 if i < 0 else 0 for i in per_fold if not pd.isna(i)]
        sign_stab = max(signs.count(1), signs.count(-1)) / max(len(signs), 1) if signs else 0
        rows.append({"symbol": s, "ic": ic, "sign_stab": sign_stab,
                     "n_calib_folds": len(per_fold)})
    df_ic = pd.DataFrame(rows).sort_values("ic", ascending=False)
    keep_set = set(df_ic[(df_ic["ic"] >= IC_MIN) & (df_ic["sign_stab"] >= STAB_MIN)]["symbol"])

    print(f"\n  Calibration IC for all symbols (sorted):", flush=True)
    print(f"  {'symbol':<14} {'IC':>7}  {'sign_stab':>10}  {'kept':>5}", flush=True)
    for _, r in df_ic.iterrows():
        kept = "YES" if r["symbol"] in keep_set else ""
        print(f"  {r['symbol']:<14} {r['ic']:>+7.4f}  {r['sign_stab']:>10.2f}  {kept:>5}", flush=True)
    print(f"\n  Keep set ({len(keep_set)}): {sorted(keep_set)}", flush=True)
    df_ic.to_csv(OUT_DIR / "calibration_ic.csv", index=False)

    # === Step 2: Production — evaluate baseline + filtered on folds 5-9 ===
    print(f"\n=== STEP 2 — production evaluation ===", flush=True)
    cycles_baseline = []
    cycles_filtered = []
    for fid in PROD_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        test, pred = train_and_predict(panel, folds_all[fid], feat_set)
        if test is None: continue
        # Baseline
        df_b = evaluate_stacked(test, pred, use_conv_gate=True, use_pm_gate=True)
        for _, r in df_b.iterrows():
            cycles_baseline.append({"fold": fid, "time": r["time"],
                                    "net": r["net_bps"], "cost": r["cost_bps"]})
        # Filtered
        keep_mask = test["symbol"].isin(keep_set).to_numpy()
        test_f = test[keep_mask].copy(); pred_f = pred[keep_mask]
        df_f = evaluate_stacked(test_f, pred_f, use_conv_gate=True, use_pm_gate=True)
        for _, r in df_f.iterrows():
            cycles_filtered.append({"fold": fid, "time": r["time"],
                                    "net": r["net_bps"], "cost": r["cost_bps"]})

        df_b = pd.DataFrame(cycles_baseline)
        df_f = pd.DataFrame(cycles_filtered)
        n_b = df_b[df_b["fold"] == fid]["net"].to_numpy() if "fold" in df_b.columns else np.array([])
        n_fl = df_f[df_f["fold"] == fid]["net"].to_numpy() if "fold" in df_f.columns else np.array([])
        b_mean = n_b.mean() if len(n_b) else 0.0
        f_mean = n_fl.mean() if len(n_fl) else 0.0
        print(f"  fold {fid}: baseline n={len(n_b)} mean={b_mean:+.2f} Sh={_sharpe(n_b) if len(n_b) else 0:+.2f}  "
              f"filtered n={len(n_fl)} mean={f_mean:+.2f} Sh={_sharpe(n_fl) if len(n_fl) else 0:+.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    # Aggregate
    print(f"\n{'=' * 100}", flush=True)
    print(f"NO-LEAK A.1 — universe filter from calib folds 0-4, evaluation on prod folds 5-9", flush=True)
    print(f"{'=' * 100}", flush=True)
    for label, cycles in [("baseline (51 names)", cycles_baseline),
                            (f"filtered ({len(keep_set)} names)", cycles_filtered)]:
        df_v = pd.DataFrame(cycles)
        if df_v.empty: continue
        net = df_v["net"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        cost = df_v["cost"].mean()
        max_dd = _max_dd(net)
        print(f"\n  {label}:", flush=True)
        print(f"    n={len(net)}  mean={net.mean():+.2f}  cost={cost:+.2f}  "
              f"Sharpe={sh:+.2f} CI=[{lo:+.2f},{hi:+.2f}]  max_DD={max_dd:+.0f}", flush=True)
        for fid in sorted(df_v["fold"].unique()):
            n_fold = df_v[df_v["fold"] == fid]["net"].to_numpy()
            if len(n_fold) >= 3:
                print(f"    fold {fid}: Sharpe={_sharpe(n_fold):+5.2f}  mean={n_fold.mean():+6.2f}", flush=True)

    pd.DataFrame(cycles_baseline).to_csv(OUT_DIR / "baseline_prod_cycles.csv", index=False)
    pd.DataFrame(cycles_filtered).to_csv(OUT_DIR / "filtered_prod_cycles.csv", index=False)
    pd.Series(sorted(keep_set)).to_csv(OUT_DIR / "keep_set.csv", index=False, header=["symbol"])
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
