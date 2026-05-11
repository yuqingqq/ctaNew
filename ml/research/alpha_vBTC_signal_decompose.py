"""Decompose where alpha comes from: features vs model vs strategy.

For N=15 K=3 on production folds 5-9:

  M0  random predictions          — chance-level baseline
  M1  best single feature ranked  — weakest feature-only model
  M2  combined feature mean rank  — equal-weight features
  M3  LGBM ensemble (current)     — full model
  M4  ORACLE: actual realized α   — upper bound

For each prediction source per fold:
  - IC of pred vs realized alpha (cross-sectional rank correlation)
  - Sharpe of long-top-K / short-bot-K strategy
  - Per-cycle dispersion vs PnL relationship

Tells us:
  - if M3 IC == M0 IC → model adds nothing
  - if M2 IC ≈ M3 IC → simple combination as good as LGBM
  - if M1 IC ≈ M3 IC → single feature does it
  - if M3 IC < M4 IC × 0.3 → model captures <30% of available alpha
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
from ml.research.alpha_v9_pred_momentum_stack import evaluate_stacked

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants.parquet"
IC_PATH = REPO / "outputs/vBTC_universe_noleak/calibration_ic.csv"
OUT_DIR = REPO / "outputs/vBTC_signal_decompose"
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


def _spearman_per_cycle(test_df: pd.DataFrame, pred: np.ndarray, target_col: str) -> list:
    """For each open_time, compute cross-sectional rank correlation of pred vs target."""
    df = test_df.copy()
    df["pred"] = pred
    ics = []
    for t, g in df.groupby("open_time"):
        sub = g[[target_col, "pred"]].dropna()
        if len(sub) < 5: continue
        ic = sub["pred"].rank().corr(sub[target_col].rank())
        if not pd.isna(ic): ics.append(ic)
    return ics


def main():
    print(f"Loading data...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    df_ic = pd.read_csv(IC_PATH).sort_values("ic", ascending=False)
    universe = sorted(df_ic["symbol"].head(TOP_N).tolist())
    print(f"  Universe (top {TOP_N}): {universe}", flush=True)

    feat_set = list(XS_FEATURE_COLS_V6_CLEAN)
    folds_all = _multi_oos_splits(panel)

    # Train LGBM ensemble on production folds
    print(f"\n=== Training {len(SEEDS)}-seed LGBM ensemble ===", flush=True)
    fold_data = {}   # {fid: (test_df_filtered, lgbm_pred, single_feat_pred, mean_rank_pred, oracle_pred)}
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
        lgbm_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration) for m in models], axis=0)

        # Filter universe
        keep_mask = test["symbol"].isin(set(universe)).to_numpy()
        test_f = test[keep_mask].copy()
        lgbm_pred_f = lgbm_pred[keep_mask]

        # M0: random predictions
        rng = np.random.RandomState(fid)
        random_pred = rng.randn(len(test_f))

        # M1: best single feature (use return_1d_xs_rank — top stable feature)
        single_feat_pred = -test_f["return_1d_xs_rank"].to_numpy()  # negate: model showed -IC for return_1d

        # M2: equal-weight combination of feature ranks (sign-flipped per feature based on full-sample IC)
        # Compute per-feature direction from train period
        train_for_signs = tr.dropna(subset=["target_A"])
        ic_signs = {}
        for f in feat_set:
            if f == "sym_id": continue
            if f not in train_for_signs.columns: continue
            sub = train_for_signs[[f, "target_A"]].dropna().sample(n=min(50000, len(train_for_signs)), random_state=42)
            if len(sub) < 1000: continue
            r = sub[f].rank().corr(sub["target_A"].rank())
            ic_signs[f] = 1 if r > 0 else -1 if r < 0 else 0
        # Build mean-rank ensemble feature
        ranks = np.zeros(len(test_f))
        n_used = 0
        test_f_pd = test_f.copy()
        for f, sign in ic_signs.items():
            if f not in test_f_pd.columns: continue
            r = test_f_pd[f].rank() * sign
            ranks += r.fillna(r.mean()).to_numpy()
            n_used += 1
        mean_rank_pred = ranks / max(n_used, 1)

        # M4: oracle (use actual alpha as predictor)
        oracle_pred = test_f["alpha_A"].to_numpy()
        # Replace NaN with 0 (won't be picked as top/bot then)
        oracle_pred = np.nan_to_num(oracle_pred, nan=0.0)

        fold_data[fid] = (test_f, lgbm_pred_f, random_pred, single_feat_pred, mean_rank_pred, oracle_pred)
        print(f"  fold {fid}: trained ({time.time()-t0:.0f}s)", flush=True)

    # === Compute IC + Sharpe per (fold, model) ===
    print(f"\n=== IC + Sharpe per fold per prediction source (N={TOP_N}, K={K}) ===", flush=True)
    print(f"  {'fold':>4}  {'model':<12}  {'mean_IC':>8}  {'Sharpe':>7}  "
          f"{'mean_bps':>8}", flush=True)
    rows = []
    for fid, (test_f, lgbm_pred, random_pred, single_pred, mean_rank_pred, oracle_pred) in fold_data.items():
        for label, pred in [("random", random_pred),
                             ("single_feat", single_pred),
                             ("mean_rank", mean_rank_pred),
                             ("LGBM_ens", lgbm_pred),
                             ("oracle", oracle_pred)]:
            ics = _spearman_per_cycle(test_f, pred, "alpha_A")
            mean_ic = np.mean(ics) if ics else 0.0
            df_eval = evaluate_stacked(test_f.copy(), pred, use_conv_gate=True, use_pm_gate=True, top_k=K)
            if df_eval.empty:
                sharpe, mean_net = 0.0, 0.0
            else:
                net = df_eval["net_bps"].to_numpy()
                sharpe = _sharpe(net)
                mean_net = net.mean()
            rows.append({"fold": fid, "model": label, "mean_ic": mean_ic,
                          "sharpe": sharpe, "mean_net_bps": mean_net,
                          "n_cycles": len(df_eval) if not df_eval.empty else 0})
            print(f"  {fid:>4}  {label:<12}  {mean_ic:>+8.4f}  {sharpe:>+7.2f}  "
                  f"{mean_net:>+8.2f}", flush=True)
        print()

    df_out = pd.DataFrame(rows)
    # Aggregate Sharpe across folds for each model
    print(f"=== Aggregate (across folds) ===", flush=True)
    print(f"  {'model':<12}  {'mean_IC':>8}  {'avg_Sh':>7}  {'mean_bps':>8}  {'fold_Sh_range':>15}",
          flush=True)
    for label in ["random", "single_feat", "mean_rank", "LGBM_ens", "oracle"]:
        sub = df_out[df_out["model"] == label]
        avg_ic = sub["mean_ic"].mean()
        avg_sh = sub["sharpe"].mean()
        avg_net = sub["mean_net_bps"].mean()
        sh_range = f"[{sub['sharpe'].min():+.2f},{sub['sharpe'].max():+.2f}]"
        print(f"  {label:<12}  {avg_ic:>+8.4f}  {avg_sh:>+7.2f}  {avg_net:>+8.2f}  {sh_range:>15}",
              flush=True)

    df_out.to_csv(OUT_DIR / "signal_decompose.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
