"""Sequential evaluation of 4 basket-variant targets in one Python process.

Loads panel_variants.parquet once, then trains+evaluates Tier 1 on:
  A: equal-weight basket residual
  B: inverse-vol weighted basket residual
  C: trimmed equal-weight basket residual
  D: cluster-residual (data-driven hierarchical clusters)

Runs ~10-30 sec per variant (after one-time panel load + multi_oos build).
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
OUT_DIR = REPO / "outputs/vBTC_basket_variants"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
SEEDS = (42,)
N_FOLDS = 3
VARIANTS = [
    ("A", "equal-weight",     "target_A"),
    ("B", "inverse-vol",      "target_B"),
    ("C", "trimmed",          "target_C"),
    ("D", "cluster-residual", "target_D"),
]


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def run_variant(panel: pd.DataFrame, folds: list, variant: str, tag: str, target_col: str,
                feat_set: list) -> dict:
    print(f"\n=== VARIANT {variant} ({tag}) — target {target_col} ===", flush=True)
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
                  for s in SEEDS]
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
    if df_v.empty: return None
    net = df_v["net"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
    max_dd = _max_dd(net)
    cost = df_v["cost"].mean()
    per_fold = {}
    for fid in sorted(df_v["fold"].unique()):
        n_f = df_v[df_v["fold"] == fid]["net"].to_numpy()
        per_fold[fid] = (n_f.mean(), _sharpe(n_f))
    print(f"  AGGREGATE: n={len(net)}  mean_net={net.mean():+.2f}  cost={cost:+.2f}  "
          f"Sharpe={sh:+.2f} CI=[{lo:+.2f},{hi:+.2f}] max_DD={max_dd:+.0f}", flush=True)

    df_v.to_csv(OUT_DIR / f"variant_{variant}_cycles.csv", index=False)
    pd.DataFrame([{"variant": variant, "tag": tag, "n": len(net), "mean_net": net.mean(),
                   "cost": cost, "sharpe": sh, "ci_lo": lo, "ci_hi": hi, "max_dd": max_dd}
                  ]).to_csv(OUT_DIR / f"variant_{variant}_summary.csv", index=False)

    return {"variant": variant, "tag": tag, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
            "mean_net": net.mean(), "cost": cost, "max_dd": max_dd,
            "per_fold": per_fold, "n": len(net)}


def main():
    print(f"Loading panel...", flush=True)
    t0 = time.time()
    panel = pd.read_parquet(PANEL_PATH)
    print(f"  {len(panel):,} rows × {panel['symbol'].nunique()} syms in {time.time()-t0:.0f}s", flush=True)

    feat_set = list(XS_FEATURE_COLS_V6_CLEAN)
    all_folds = _multi_oos_splits(panel)
    fold_idx = [len(all_folds) // 5, len(all_folds) // 2, 4 * len(all_folds) // 5]
    folds = [all_folds[i] for i in fold_idx if i < len(all_folds)]
    print(f"  Folds: {len(folds)} of {len(all_folds)} (indices {fold_idx})", flush=True)

    results = []
    for variant, tag, tcol in VARIANTS:
        r = run_variant(panel, folds, variant, tag, tcol, feat_set)
        if r: results.append(r)

    # === Comparison table ===
    print(f"\n{'=' * 100}", flush=True)
    print(f"COMPARISON — basket-variant targets (3 folds × 1 seed, fast mode)", flush=True)
    print(f"{'=' * 100}", flush=True)
    print(f"  {'variant':<30} {'Sharpe':>7}  {'CI_lo':>7}  {'CI_hi':>7}  "
          f"{'mean':>+7}  {'cost':>+6}  {'maxDD':>+8}", flush=True)
    for r in results:
        print(f"  {r['variant']}-{r['tag']:<28} {r['sharpe']:>+7.2f}  {r['ci_lo']:>+7.2f}  "
              f"{r['ci_hi']:>+7.2f}  {r['mean_net']:>+7.2f}  {r['cost']:>+6.2f}  "
              f"{r['max_dd']:>+8.0f}", flush=True)

    print(f"\n  Per-fold breakdown:", flush=True)
    fold_ids = sorted(results[0]["per_fold"].keys()) if results else []
    print(f"  {'variant':<30} " + " ".join(f"{'fold' + str(f):>10}" for f in fold_ids), flush=True)
    for r in results:
        cells = [f"{r['per_fold'][f][1]:+5.2f}" for f in fold_ids]
        print(f"  {r['variant']}-{r['tag']:<28} " + " ".join(f"{c:>10}" for c in cells), flush=True)

    pd.DataFrame(results).to_csv(OUT_DIR / "comparison.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
