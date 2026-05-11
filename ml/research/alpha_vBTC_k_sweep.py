"""K-sweep × universe-size sweep on the no-leak filtered universe.

For each (universe_size N, K):
  - Universe = top-N symbols by calibration IC (must satisfy 2K+1 <= N)
  - Train + predict on production folds (5-9)
  - Evaluate with that K (long top-K, short bot-K) on the N-name universe
  - Report Sharpe + CI

Maps the trade-off between universe quality (smaller=higher avg IC) and
K (smaller=more concentrated, larger=more diversified).
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
KEEP_PATH = REPO / "outputs/vBTC_universe_noleak/keep_set.csv"
IC_PATH = REPO / "outputs/vBTC_universe_noleak/calibration_ic.csv"
OUT_DIR = REPO / "outputs/vBTC_k_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
SEEDS = (42,)
PROD_FOLDS = [5, 6, 7, 8, 9]
N_VALUES = [25, 20, 15, 12, 10, 8]   # universe sizes
K_VALUES = [2, 3, 4, 5, 6, 7, 8]      # number of long/short pairs


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
    # Pre-rank symbols by calibration IC so universe_top_N is sym list
    sorted_syms = df_ic["symbol"].tolist()
    print(f"  Panel: {len(panel):,} rows  Calibration IC available for {len(sorted_syms)} syms", flush=True)
    print(f"  Top 25 by IC: {sorted_syms[:25]}", flush=True)

    feat_set = list(XS_FEATURE_COLS_V6_CLEAN)
    folds_all = _multi_oos_splits(panel)

    # Train + predict once per production fold; reuse predictions across (N, K)
    print(f"\n=== Training on production folds ===", flush=True)
    fold_data = {}
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
        pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration) for m in models], axis=0)
        fold_data[fid] = (test.copy(), pred)   # full test set; subset later by universe
        print(f"  fold {fid}: trained, {len(test):,} test rows ({time.time()-t0:.0f}s)", flush=True)

    # === Grid: N × K ===
    print(f"\n=== Universe size × K grid (no-leak filter, top-N by calibration IC) ===", flush=True)
    print(f"  {'N':>3} {'K':>3}  {'avg_IC':>7}  {'n':>4}  {'mean':>7}  {'cost':>6}  "
          f"{'Sharpe':>7}  {'CI_lo':>7}  {'CI_hi':>7}  {'maxDD':>7}", flush=True)

    summary = []
    for n in N_VALUES:
        universe = set(sorted_syms[:n])
        avg_ic = df_ic.iloc[:n]["ic"].mean()
        for k in K_VALUES:
            if 2 * k + 1 > n: continue  # K too big for universe
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
            summary.append({"N": n, "K": k, "avg_ic": avg_ic, "n": len(net),
                              "mean_net": net.mean(), "cost": cost,
                              "sharpe": sh, "ci_lo": lo, "ci_hi": hi, "max_dd": max_dd})
            print(f"  N={n:>2} K={k:>2}  {avg_ic:>+7.4f}  {len(net):>4}  "
                  f"{net.mean():>+7.2f}  {cost:>+6.2f}  "
                  f"{sh:>+7.2f}  {lo:>+7.2f}  {hi:>+7.2f}  {max_dd:>+7.0f}", flush=True)

    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(OUT_DIR / "k_universe_grid.csv", index=False)
    if not df_sum.empty:
        # Best by Sharpe
        best = df_sum.sort_values("sharpe", ascending=False).head(5)
        print(f"\n=== TOP 5 (N, K) by Sharpe ===", flush=True)
        for _, r in best.iterrows():
            print(f"  N={int(r['N']):>2} K={int(r['K']):>2}  Sharpe={r['sharpe']:+.2f}  "
                  f"CI=[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]  avg_IC={r['avg_ic']:+.4f}",
                  flush=True)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
