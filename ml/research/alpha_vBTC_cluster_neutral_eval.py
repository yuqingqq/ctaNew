"""Cluster-neutral execution test on filtered universe.

Question: does cluster-aware K selection (long top-1 from each cluster,
short bot-1 from each cluster) improve over vanilla cross-sectional
(long top-K from full pool) on the filtered universe?

Hypothesis: clusters within the filtered universe may still have
unintended factor exposure when picks concentrate. Forcing cluster
balance should reduce that. But it also forces less-confident picks
into the book, which could hurt.

Comparison:
  V — vanilla cross-sectional (current Path A baseline, K=7)
  N — cluster-neutral (K_per_cluster, summing to similar total)

Both run on the SAME filtered universe (23 names, sign-stable +IC).
Same training data, same model. Only execution differs.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from collections import deque

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train
from ml.research.alpha_v4_xs import block_bootstrap_ci

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants.parquet"
CLUSTER_PATH = REPO / "outputs/vBTC_features/cluster_map.csv"
IC_PATH = REPO / "outputs/vBTC_per_symbol_ic/per_symbol_ic.csv"
OUT_DIR = REPO / "outputs/vBTC_cluster_neutral"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
SEEDS = (42,)
TOP_K = 7   # baseline K
COST_PER_LEG = 6.0   # bps per leg (matches v6_clean)
GATE_LOOKBACK = 252
GATE_PCTILE = 0.30


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def evaluate_simple(test_df: pd.DataFrame, pred: np.ndarray, *, mode: str,
                     cluster_map: dict, top_k_total: int = TOP_K,
                     k_per_cluster: int = 1) -> pd.DataFrame:
    """Cycle-by-cycle evaluator. mode='vanilla' picks top-K_total/bot-K_total
    from full pool. mode='cluster' picks K_per_cluster long+short from each
    cluster present in this cycle."""
    df = test_df.copy()
    df["pred"] = pred
    df["cluster"] = df["symbol"].map(cluster_map)
    times = sorted(df["open_time"].unique())
    if len(times) < 2: return pd.DataFrame()
    sample_every = HORIZON
    keep_times = set(times[::sample_every])
    df = df[df["open_time"].isin(keep_times)]

    bars = []
    dispersion_history = deque(maxlen=GATE_LOOKBACK)
    cur_long, cur_short = set(), set()

    for t, g in df.groupby("open_time"):
        if mode == "vanilla":
            if len(g) < 2 * top_k_total + 1: continue
            sym_arr = g["symbol"].to_numpy()
            pred_arr = g["pred"].to_numpy()
            idx_top = np.argpartition(-pred_arr, top_k_total - 1)[:top_k_total]
            idx_bot = np.argpartition(pred_arr, top_k_total - 1)[:top_k_total]
            new_long = set(sym_arr[idx_top])
            new_short = set(sym_arr[idx_bot])
            dispersion = float(pred_arr[idx_top].mean() - pred_arr[idx_bot].mean())
        else:  # cluster mode
            new_long, new_short = set(), set()
            cluster_disps = []
            for c, gc in g.groupby("cluster"):
                if len(gc) < 2 * k_per_cluster + 1: continue
                sym_arr = gc["symbol"].to_numpy()
                pred_arr = gc["pred"].to_numpy()
                k = min(k_per_cluster, len(gc) // 2)
                idx_top = np.argpartition(-pred_arr, k - 1)[:k]
                idx_bot = np.argpartition(pred_arr, k - 1)[:k]
                new_long |= set(sym_arr[idx_top])
                new_short |= set(sym_arr[idx_bot])
                cluster_disps.append(float(pred_arr[idx_top].mean() - pred_arr[idx_bot].mean()))
            if not new_long: continue
            dispersion = float(np.mean(cluster_disps)) if cluster_disps else 0.0

        # Conv gate
        skip = False
        if len(dispersion_history) >= 30:
            thr = float(np.quantile(list(dispersion_history), GATE_PCTILE))
            if dispersion < thr: skip = True
        dispersion_history.append(dispersion)

        if skip:
            # Hold-through MtM
            if cur_long or cur_short:
                long_g = g[g["symbol"].isin(cur_long)]
                short_g = g[g["symbol"].isin(cur_short)]
                long_ret = long_g["return_pct"].mean() if not long_g.empty else 0.0
                short_ret = short_g["return_pct"].mean() if not short_g.empty else 0.0
                spread = (long_ret - short_ret) * 1e4
                bars.append({"time": t, "net_bps": spread, "cost_bps": 0.0, "skipped": 1})
            else:
                bars.append({"time": t, "net_bps": 0.0, "cost_bps": 0.0, "skipped": 1})
            continue

        # Trade with new positions
        long_g = g[g["symbol"].isin(new_long)]
        short_g = g[g["symbol"].isin(new_short)]
        long_ret = long_g["return_pct"].mean() if not long_g.empty else 0.0
        short_ret = short_g["return_pct"].mean() if not short_g.empty else 0.0
        spread = (long_ret - short_ret) * 1e4
        # Turnover cost (rough): churn rate × cost_per_leg × 2 sides
        churn_long = len(new_long.symmetric_difference(cur_long)) / max(len(new_long | cur_long), 1)
        churn_short = len(new_short.symmetric_difference(cur_short)) / max(len(new_short | cur_short), 1)
        cost = (churn_long + churn_short) * COST_PER_LEG
        net = spread - cost
        bars.append({"time": t, "net_bps": net, "cost_bps": cost, "skipped": 0})
        cur_long, cur_short = new_long, new_short

    return pd.DataFrame(bars)


def main():
    print(f"Loading data...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    cluster_map = pd.read_csv(CLUSTER_PATH).set_index("symbol")["cluster"].to_dict()
    df_ic = pd.read_csv(IC_PATH)
    keep_set = set(df_ic[(df_ic["ic"] >= 0.02) & (df_ic["sign_stab"] >= 1.0)]["symbol"])
    print(f"  Panel: {len(panel):,} rows", flush=True)
    print(f"  Keep set: {len(keep_set)} names", flush=True)
    print(f"  Clusters in keep set: " +
          str({c: sum(1 for s in keep_set if cluster_map.get(s) == c)
                for c in sorted({cluster_map.get(s) for s in keep_set})}), flush=True)

    # Use leaky filter (same as Path A baseline) for direct comparison
    feat_set = list(XS_FEATURE_COLS_V6_CLEAN)
    folds_all = _multi_oos_splits(panel)
    fold_idx = [len(folds_all) // 5, len(folds_all) // 2, 4 * len(folds_all) // 5]
    folds = [folds_all[i] for i in fold_idx if i < len(folds_all)]

    cycles_v, cycles_n = [], []
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
        models = [_train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s) for s in SEEDS]
        pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration) for m in models], axis=0)

        # Filter to keep_set
        keep_mask = test["symbol"].isin(keep_set).to_numpy()
        test_f = test[keep_mask].copy()
        pred_f = pred[keep_mask]

        df_v = evaluate_simple(test_f, pred_f, mode="vanilla", cluster_map=cluster_map)
        df_n = evaluate_simple(test_f, pred_f, mode="cluster", cluster_map=cluster_map,
                                k_per_cluster=1)
        for _, r in df_v.iterrows():
            cycles_v.append({"fold": fold["fid"], "time": r["time"],
                              "net": r["net_bps"], "cost": r["cost_bps"]})
        for _, r in df_n.iterrows():
            cycles_n.append({"fold": fold["fid"], "time": r["time"],
                              "net": r["net_bps"], "cost": r["cost_bps"]})
        n_v = pd.DataFrame(cycles_v); n_v = n_v[n_v["fold"] == fold["fid"]]["net"].to_numpy()
        n_n = pd.DataFrame(cycles_n); n_n = n_n[n_n["fold"] == fold["fid"]]["net"].to_numpy()
        print(f"  fold {fold['fid']}: vanilla mean={n_v.mean():+.2f} Sh={_sharpe(n_v):+.2f}  "
              f"cluster mean={n_n.mean():+.2f} Sh={_sharpe(n_n):+.2f}  ({time.time()-t0:.0f}s)",
              flush=True)

    print(f"\n{'=' * 90}", flush=True)
    print(f"VANILLA vs CLUSTER-NEUTRAL — both on filtered ({len(keep_set)}) universe", flush=True)
    print(f"{'=' * 90}", flush=True)
    for label, cycles in [("vanilla (top-7/bot-7 from pool)", cycles_v),
                            ("cluster-neutral (K=1 per cluster)", cycles_n)]:
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
            n_f = df_v[df_v["fold"] == fid]["net"].to_numpy()
            if len(n_f) >= 3:
                print(f"    fold {fid}: Sharpe={_sharpe(n_f):+5.2f}", flush=True)

    pd.DataFrame(cycles_v).to_csv(OUT_DIR / "vanilla_cycles.csv", index=False)
    pd.DataFrame(cycles_n).to_csv(OUT_DIR / "cluster_cycles.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
