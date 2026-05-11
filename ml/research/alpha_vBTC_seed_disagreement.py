"""Deep diagnostic: per-seed Sharpe and prediction disagreement.

Question: when we ensemble 5 seeds, the Sharpe drops vs single-seed.
Why? Two possibilities:
  H1. Individual seeds give wildly different predictions (high seed variance).
      Ensembling correctly averages them but the resulting "consensus" signal
      is weaker because each seed's signal differs.
  H2. Individual seeds all give similar predictions, but the strategy has
      high cycle-level variance, so single-seed Sharpe reading is noisy
      and ensembling reveals the true (lower) Sharpe.

This diagnostic measures both:
  - Per-seed Sharpe (5 individual values per config)
  - Cross-seed prediction correlation (per fold)
  - Per-seed feature importance to spot if seeds learn different patterns
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
OUT_DIR = REPO / "outputs/vBTC_seed_disagreement"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
PROD_FOLDS = [5, 6, 7, 8, 9]
SEEDS = (42, 1337, 7, 19, 2718)
TOP_N = 15
K = 3


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def main():
    print(f"Loading data...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    df_ic = pd.read_csv(IC_PATH).sort_values("ic", ascending=False)
    universe = set(df_ic["symbol"].head(TOP_N).tolist())
    print(f"  Universe (top {TOP_N}): {sorted(universe)}", flush=True)

    feat_set = list(XS_FEATURE_COLS_V6_CLEAN)
    folds_all = _multi_oos_splits(panel)

    # Train each seed individually, save predictions + feature importance
    print(f"\n=== Training {len(SEEDS)} seeds × {len(PROD_FOLDS)} folds (individually) ===",
          flush=True)
    fold_seed_data = {}   # {fid: {seed: (test_df, pred, gain_importance, best_iter)}}
    for fid in PROD_FOLDS:
        if fid >= len(folds_all): continue
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
        seed_dict = {}
        for s in SEEDS:
            t0 = time.time()
            m = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s)
            pred = m.predict(Xtest, num_iteration=m.best_iteration)
            gain = m.feature_importance(importance_type="gain")
            gain_norm = gain / max(gain.sum(), 1e-9)
            seed_dict[s] = (pred, gain_norm, m.best_iteration, time.time() - t0)
        fold_seed_data[fid] = (test, seed_dict)
        iter_str = " ".join(f"{seed_dict[s][2]:>3}" for s in SEEDS)
        print(f"  fold {fid}: best_iters per seed: {iter_str}", flush=True)

    # === Analysis 1: Per-seed Sharpe on filtered universe ===
    print(f"\n=== Per-seed Sharpe (N={TOP_N}, K={K}) ===", flush=True)
    per_seed_sh = {s: {} for s in SEEDS}
    for fid, (test, seed_dict) in fold_seed_data.items():
        keep_mask = test["symbol"].isin(universe).to_numpy()
        test_f = test[keep_mask]
        for s, (pred, _, _, _) in seed_dict.items():
            pred_f = pred[keep_mask]
            df = evaluate_stacked(test_f.copy(), pred_f, use_conv_gate=True, use_pm_gate=True, top_k=K)
            if df.empty:
                per_seed_sh[s][fid] = (0, 0)
                continue
            net = df["net_bps"].to_numpy()
            per_seed_sh[s][fid] = (net.mean(), _sharpe(net))

    # Print per-seed per-fold Sharpe table
    print(f"  {'seed':>5}  " + " ".join(f"{'fold' + str(f):>10}" for f in PROD_FOLDS) + "    aggr_Sh", flush=True)
    for s in SEEDS:
        cells = [per_seed_sh[s].get(f, (0, 0))[1] for f in PROD_FOLDS]
        # compute aggregate sharpe by combining all cycles
        all_net = []
        for fid in PROD_FOLDS:
            test, seed_dict = fold_seed_data.get(fid, (None, None))
            if test is None: continue
            keep_mask = test["symbol"].isin(universe).to_numpy()
            test_f = test[keep_mask]
            pred_f = seed_dict[s][0][keep_mask]
            df = evaluate_stacked(test_f.copy(), pred_f, use_conv_gate=True, use_pm_gate=True, top_k=K)
            if not df.empty:
                all_net.extend(df["net_bps"].tolist())
        agg_sh = _sharpe(np.array(all_net)) if all_net else 0
        cells_str = " ".join(f"{c:+10.2f}" for c in cells)
        print(f"  {s:>5}  {cells_str}    {agg_sh:+7.2f}", flush=True)

    # Ensemble
    print(f"\n=== Ensemble Sharpe ===", flush=True)
    ens_per_fold = {}
    all_net_ens = []
    for fid, (test, seed_dict) in fold_seed_data.items():
        keep_mask = test["symbol"].isin(universe).to_numpy()
        test_f = test[keep_mask]
        ens_pred = np.mean([seed_dict[s][0] for s in SEEDS], axis=0)
        pred_f = ens_pred[keep_mask]
        df = evaluate_stacked(test_f.copy(), pred_f, use_conv_gate=True, use_pm_gate=True, top_k=K)
        if df.empty: continue
        net = df["net_bps"].to_numpy()
        ens_per_fold[fid] = (net.mean(), _sharpe(net))
        all_net_ens.extend(net.tolist())
    cells = [ens_per_fold.get(f, (0, 0))[1] for f in PROD_FOLDS]
    cells_str = " ".join(f"{c:+10.2f}" for c in cells)
    agg_sh = _sharpe(np.array(all_net_ens))
    print(f"  ensemble  {cells_str}    {agg_sh:+7.2f}", flush=True)

    # === Analysis 2: Cross-seed prediction correlation ===
    print(f"\n=== Cross-seed prediction Spearman correlation per fold ===", flush=True)
    print(f"  Average pairwise rank correlation across 5 seeds per fold", flush=True)
    print(f"  High = seeds agree, low = seeds disagree (ensemble averages noise)", flush=True)
    for fid, (test, seed_dict) in fold_seed_data.items():
        keep_mask = test["symbol"].isin(universe).to_numpy()
        preds = np.array([seed_dict[s][0][keep_mask] for s in SEEDS])
        # pairwise corr
        from scipy.stats import spearmanr
        corrs = []
        for i in range(len(SEEDS)):
            for j in range(i + 1, len(SEEDS)):
                c, _ = spearmanr(preds[i], preds[j], nan_policy="omit")
                if not np.isnan(c): corrs.append(c)
        avg_corr = np.mean(corrs) if corrs else 0
        print(f"  fold {fid}: avg pairwise rank corr = {avg_corr:+.3f}", flush=True)

    # === Analysis 3: Feature importance variance ===
    print(f"\n=== Feature importance variance across seeds (avg gain by feature) ===", flush=True)
    for fid, (test, seed_dict) in fold_seed_data.items():
        gains = np.array([seed_dict[s][1] for s in SEEDS])
        mean_g = gains.mean(axis=0)
        std_g = gains.std(axis=0)
        # Top 8 features by mean gain
        order = np.argsort(-mean_g)[:8]
        print(f"\n  fold {fid} top-8 features (mean ± std gain across seeds):", flush=True)
        for i in order:
            f_name = feat_set[i]
            print(f"    {f_name:<32}  {mean_g[i]*100:>5.1f}% ± {std_g[i]*100:>4.1f}%", flush=True)

    # Save raw data
    rows = []
    for fid in PROD_FOLDS:
        if fid not in fold_seed_data: continue
        for s in SEEDS:
            mn, sh = per_seed_sh[s].get(fid, (0, 0))
            rows.append({"fold": fid, "seed": s, "mean_net": mn, "sharpe": sh,
                          "best_iter": fold_seed_data[fid][1][s][2]})
    pd.DataFrame(rows).to_csv(OUT_DIR / "per_seed_per_fold.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
