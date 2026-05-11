"""Path A: train on full 51-name panel, but filter trading universe at eval
time to only the symbols with sign-stable positive IC from the diagnostic.

This tests whether removing the negative-IC and unstable-IC symbols from
trade decisions recovers Sharpe, without changing the model or the target.

Two configurations:
  X_baseline  — train+eval on all 51 names (Sharpe -0.13 from variant A)
  X_filtered  — train on all 51, eval ranking restricted to keep_set
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
IC_PATH = REPO / "outputs/vBTC_per_symbol_ic/per_symbol_ic.csv"
OUT_DIR = REPO / "outputs/vBTC_universe_filtered"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
SEEDS = (42,)

# Filtering criteria
IC_MIN = 0.02
STAB_MIN = 1.0


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def main():
    print(f"Loading per-symbol IC diagnostic...", flush=True)
    df_ic = pd.read_csv(IC_PATH)
    keep = df_ic[(df_ic["ic"] >= IC_MIN) & (df_ic["sign_stab"] >= STAB_MIN)]
    keep_set = set(keep["symbol"].tolist())
    print(f"  Keep set ({len(keep_set)} of {len(df_ic)}): IC≥{IC_MIN}, sign_stab≥{STAB_MIN}", flush=True)
    for _, r in keep.sort_values("ic", ascending=False).iterrows():
        print(f"    {r['symbol']:<14} IC={r['ic']:+.4f}", flush=True)

    print(f"\nLoading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    print(f"  {len(panel):,} rows × {panel['symbol'].nunique()} syms", flush=True)

    feat_set = list(XS_FEATURE_COLS_V6_CLEAN)
    all_folds = _multi_oos_splits(panel)
    fold_idx = [len(all_folds) // 5, len(all_folds) // 2, 4 * len(all_folds) // 5]
    folds = [all_folds[i] for i in fold_idx if i < len(all_folds)]
    print(f"  Folds: {len(folds)} of {len(all_folds)}", flush=True)

    cycles_baseline = []
    cycles_filtered = []

    for fold in folds:
        t0 = time.time()
        print(f"\nFold {fold['fid']}...", flush=True)
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

        models = [_train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s)
                  for s in SEEDS]
        pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration) for m in models], axis=0)
        test_with_pred = test.copy()
        test_with_pred["pred"] = pred

        # Baseline: all 51 names
        df_base = evaluate_stacked(test, pred, use_conv_gate=True, use_pm_gate=True)
        for _, r in df_base.iterrows():
            cycles_baseline.append({"fold": fold["fid"], "time": r["time"],
                                    "net": r["net_bps"], "cost": r["cost_bps"]})

        # Filtered: only keep_set
        keep_mask = test_with_pred["symbol"].isin(keep_set).to_numpy()
        test_filt = test_with_pred[keep_mask].copy()
        pred_filt = test_filt["pred"].to_numpy()
        df_filt = evaluate_stacked(test_filt, pred_filt, use_conv_gate=True, use_pm_gate=True)
        for _, r in df_filt.iterrows():
            cycles_filtered.append({"fold": fold["fid"], "time": r["time"],
                                    "net": r["net_bps"], "cost": r["cost_bps"]})

        # Per-fold sharpe
        n_b = pd.DataFrame(cycles_baseline)
        n_b = n_b[n_b["fold"] == fold["fid"]]["net"].to_numpy()
        n_f = pd.DataFrame(cycles_filtered)
        n_f = n_f[n_f["fold"] == fold["fid"]]["net"].to_numpy()
        print(f"  baseline: mean={n_b.mean():+.2f} Sharpe={_sharpe(n_b):+.2f}  "
              f"filtered: mean={n_f.mean():+.2f} Sharpe={_sharpe(n_f):+.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    # Aggregates
    print(f"\n{'=' * 100}", flush=True)
    print(f"BASELINE vs FILTERED (universe filtered to {len(keep_set)} sign-stable +IC names)", flush=True)
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
        print(f"    n={len(net)}  mean_net={net.mean():+.2f}  cost={cost:+.2f}", flush=True)
        print(f"    Sharpe={sh:+.2f}  CI=[{lo:+.2f}, {hi:+.2f}]  max_DD={max_dd:+.0f}", flush=True)
        print(f"    Per-fold:", flush=True)
        for fid in sorted(df_v["fold"].unique()):
            n_f = df_v[df_v["fold"] == fid]["net"].to_numpy()
            if len(n_f) >= 3:
                print(f"      fold {fid}: Sharpe={_sharpe(n_f):+5.2f}  mean={n_f.mean():+6.2f}", flush=True)

    pd.DataFrame(cycles_baseline).to_csv(OUT_DIR / "baseline_cycles.csv", index=False)
    pd.DataFrame(cycles_filtered).to_csv(OUT_DIR / "filtered_cycles.csv", index=False)
    pd.Series(sorted(keep_set)).to_csv(OUT_DIR / "keep_set.csv", index=False, header=["symbol"])
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
