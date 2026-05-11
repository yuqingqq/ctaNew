"""Tier 1 train+eval on a chosen basket variant. Use --variant {A,B,C}."""
from __future__ import annotations
import argparse, sys, time, warnings
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
OUT_DIR = REPO / "outputs/vBTC_basket_variants"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
FAST_SEEDS = (42,)
FAST_N_FOLDS = 3


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def main(variant: str):
    assert variant in ("A", "B", "C", "D")
    target_col = f"target_{variant}"
    tag = {"A": "equal-weight", "B": "inverse-vol", "C": "trimmed", "D": "cluster-residual"}[variant]

    print(f"=== VARIANT {variant} ({tag}) — target column: {target_col} ===", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    print(f"  {len(panel):,} rows × {panel['symbol'].nunique()} syms", flush=True)

    feat_set = list(XS_FEATURE_COLS_V6_CLEAN)
    all_folds = _multi_oos_splits(panel)
    fold_idx = [len(all_folds) // 5, len(all_folds) // 2, 4 * len(all_folds) // 5]
    folds = [all_folds[i] for i in fold_idx if i < len(all_folds)]

    cycles = []
    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200: continue

        Xt = tr[feat_set].to_numpy(np.float32)
        Xc = ca[feat_set].to_numpy(np.float32)
        Xtest = test[feat_set].to_numpy(np.float32)
        yt = tr[target_col].to_numpy(np.float32)
        yc = ca[target_col].to_numpy(np.float32)
        mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
        if mask_t.sum() < 1000 or mask_c.sum() < 200: continue

        models = [_train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s)
                  for s in FAST_SEEDS]
        pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration) for m in models], axis=0)

        df = evaluate_stacked(test, pred, use_conv_gate=True, use_pm_gate=True)
        for _, r in df.iterrows():
            cycles.append({"fold": fold["fid"], "time": r["time"],
                           "net": r["net_bps"], "cost": r["cost_bps"]})

        df_t = pd.DataFrame(cycles)
        df_t = df_t[df_t["fold"] == fold["fid"]]
        n = df_t["net"].to_numpy() if not df_t.empty else np.array([])
        sh = _sharpe(n) if len(n) else 0.0
        mn = n.mean() if len(n) else 0.0
        print(f"  fold {fold['fid']:>2}: variant_{variant}={mn:+.2f}({sh:+.1f})  "
              f"({time.time()-t0:.0f}s)", flush=True)

    df_v = pd.DataFrame(cycles)
    if df_v.empty:
        print("  no cycles, exiting", flush=True)
        return
    net = df_v["net"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
    max_dd = _max_dd(net)
    cost = df_v["cost"].mean()
    print(f"\n  AGGREGATE: n={len(net)}  mean_net={net.mean():+.2f}  cost={cost:+.2f}  "
          f"Sharpe={sh:+.2f} CI=[{lo:+.2f},{hi:+.2f}] max_DD={max_dd:+.0f}", flush=True)
    print(f"  Per-fold:", flush=True)
    for fid in sorted(df_v["fold"].unique()):
        n_f = df_v[df_v["fold"] == fid]["net"].to_numpy()
        if len(n_f) >= 3:
            print(f"    fold {fid}: Sharpe={_sharpe(n_f):+5.2f}  mean={n_f.mean():+6.2f}",
                  flush=True)

    pd.DataFrame([{"variant": variant, "tag": tag, "n": len(net), "mean_net": net.mean(),
                   "cost": cost, "sharpe": sh, "ci_lo": lo, "ci_hi": hi, "max_dd": max_dd}
                  ]).to_csv(OUT_DIR / f"variant_{variant}_summary.csv", index=False)
    df_v.to_csv(OUT_DIR / f"variant_{variant}_cycles.csv", index=False)
    print(f"  saved → {OUT_DIR}/variant_{variant}_*", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["A", "B", "C", "D"])
    args = p.parse_args()
    main(args.variant)
