"""Phase D: clustered selection backtest on FULL39.

Compares cluster-bucketed conv+PM stack to:
  - ORIG25 global K=7 (+2.75 validated)
  - FULL39 global K=7 (+0.88 failed)
  - FULL39 global K=11 (+0.62 failed)

Sweeps K_c ∈ {1, 2}.

Key question: does cluster-bucketing produce Sharpe > +2.5 on FULL39?
"""
from __future__ import annotations
import sys, time, warnings, json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN, list_universe
from ml.research.alpha_v4_xs_1d import (
    ENSEMBLE_SEEDS, _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v9_universe_expand import build_wide_panel_for
from ml.research.alpha_v9_pred_momentum_stack import evaluate_stacked
from ml.research.portfolio_clustered import (
    load_clusters, get_sym_to_cluster,
    pm_gate_filter_clustered, _compute_band_per_cluster,
)

HORIZON = 48
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
GATE_LOOKBACK = 252
GATE_PCTILE = 0.30
PM_M = 2
PM_BAND = 1.0
OUT_DIR = REPO / "outputs/clustered_backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_SYMBOLS = {"ETCUSDT", "HBARUSDT", "ICPUSDT", "LDOUSDT", "TRBUSDT",
               "AAVEUSDT", "MKRUSDT", "AXSUSDT", "GMXUSDT",
               "1000PEPEUSDT", "1000SHIBUSDT", "TONUSDT", "ORDIUSDT", "WIFUSDT"}


def _bn_scale(top_df, bot_df):
    if top_df.empty or bot_df.empty:
        return 1.0, 1.0
    beta_L = float(top_df["beta_short_vs_bk"].mean())
    beta_S = float(bot_df["beta_short_vs_bk"].mean())
    if beta_L < 0.1 or beta_S < 0.1 or (beta_L + beta_S) < 0.3:
        return 1.0, 1.0
    denom = beta_L + beta_S
    return (float(np.clip(2.0 * beta_S / denom, 0.5, 1.5)),
            float(np.clip(2.0 * beta_L / denom, 0.5, 1.5)))


def evaluate_clustered(
    test: pd.DataFrame, yt: np.ndarray, *,
    clusters: dict[str, list[str]],
    k_per_cluster: int = 1,
    use_conv_gate: bool = True, use_pm_gate: bool = True,
    cost_bps_per_leg: float = COST_PER_LEG,
    sample_every: int = HORIZON,
    gate_pctile: float = GATE_PCTILE,
    gate_lookback: int = GATE_LOOKBACK,
    pm_m: int = PM_M, pm_band: float = PM_BAND,
) -> pd.DataFrame:
    """Stacked gate evaluator with cluster-bucketed selection."""
    cols = ["open_time", "symbol", "return_pct", "alpha_realized",
            "basket_fwd", "beta_short_vs_bk"]
    df = test[cols].copy()
    df["pred"] = yt
    times = sorted(df["open_time"].unique())
    if not times:
        return pd.DataFrame()
    if sample_every > 1:
        keep_times = set(times[::sample_every])
        df = df[df["open_time"].isin(keep_times)]

    K_TARGET = k_per_cluster * len(clusters)  # for per-name weighting
    band_k = max(k_per_cluster, int(round(pm_band * k_per_cluster)))

    history_per_cluster = {c: [] for c in clusters}
    dispersion_history: deque = deque(maxlen=gate_lookback)
    prev_long_pc = {c: set() for c in clusters}
    prev_short_pc = {c: set() for c in clusters}

    bars = []
    cur_long, cur_short = set(), set()
    prev_long_w, prev_short_w = {}, {}

    for t, g in df.groupby("open_time"):
        if len(g) < 4:
            continue

        # Conv gate: dispersion on TOP_GLOBAL minus BOT_GLOBAL of K_TARGET each
        # (still global signal, not cluster-local)
        sorted_g = g.sort_values("pred")
        if len(g) < 2 * K_TARGET + 1:
            continue
        bot_global = sorted_g.head(K_TARGET)
        top_global = sorted_g.tail(K_TARGET)
        dispersion = float(top_global["pred"].mean() - bot_global["pred"].mean())
        skip = False
        if use_conv_gate and len(dispersion_history) >= 30:
            thr = float(np.quantile(list(dispersion_history), gate_pctile))
            if dispersion < thr:
                skip = True
        dispersion_history.append(dispersion)

        # Always update PM history (per cluster) regardless of skip
        current_band = _compute_band_per_cluster(g, clusters, band_k_per_cluster=band_k)
        for c in clusters:
            history_per_cluster[c].append(current_band[c])
            if len(history_per_cluster[c]) > pm_m:
                history_per_cluster[c] = history_per_cluster[c][-pm_m:]

        if skip:
            bars.append({
                "time": t, "spread_ret_bps": 0.0,
                "long_turnover": 0.0, "short_turnover": 0.0,
                "cost_bps": 0.0, "net_bps": 0.0,
                "n_long": 0, "n_short": 0, "skipped": 1,
            })
            cur_long, cur_short = set(), set()
            prev_long_w = {}; prev_short_w = {}
            prev_long_pc = {c: set() for c in clusters}
            prev_short_pc = {c: set() for c in clusters}
            continue

        # Cluster-bucketed selection with PM gate
        if use_pm_gate:
            # Use history excluding current cycle
            history_for_filter = {c: history_per_cluster[c][:-1] for c in clusters}
            top_df, bot_df, scale_L, scale_S, info = pm_gate_filter_clustered(
                g, clusters, history_per_cluster=history_for_filter,
                prev_long_per_cluster=prev_long_pc, prev_short_per_cluster=prev_short_pc,
                k_per_cluster_long=k_per_cluster, k_per_cluster_short=k_per_cluster,
                M_cycles=pm_m, band_mult=pm_band,
            )
        else:
            from ml.research.portfolio_clustered import select_portfolio_clustered
            top_df, bot_df, scale_L, scale_S, info = select_portfolio_clustered(
                g, clusters, k_per_cluster_long=k_per_cluster,
                k_per_cluster_short=k_per_cluster,
            )

        if top_df.empty or bot_df.empty:
            cur_long, cur_short = set(top_df["symbol"]), set(bot_df["symbol"])
            prev_long_w = {s: 1.0 / K_TARGET for s in cur_long}
            prev_short_w = {s: 1.0 / K_TARGET for s in cur_short}
            bars.append({
                "time": t, "spread_ret_bps": 0.0,
                "long_turnover": 0.0, "short_turnover": 0.0,
                "cost_bps": 0.0, "net_bps": 0.0,
                "n_long": len(top_df), "n_short": len(bot_df), "skipped": 0,
            })
            continue

        # Per-name weight = 1/K_TARGET (variable leg gross)
        long_w = {s: scale_L / K_TARGET for s in top_df["symbol"]}
        short_w = {s: scale_S / K_TARGET for s in bot_df["symbol"]}
        gross_L = scale_L * len(top_df) / K_TARGET
        gross_S = scale_S * len(bot_df) / K_TARGET

        long_ret = gross_L * top_df["return_pct"].mean()
        short_ret = gross_S * bot_df["return_pct"].mean()
        spread_ret = long_ret - short_ret

        if not prev_long_w:
            long_to = sum(long_w.values())
            short_to = sum(short_w.values())
        else:
            all_l = set(long_w) | set(prev_long_w)
            long_to = sum(abs(long_w.get(s, 0) - prev_long_w.get(s, 0)) for s in all_l)
            all_s = set(short_w) | set(prev_short_w)
            short_to = sum(abs(short_w.get(s, 0) - prev_short_w.get(s, 0)) for s in all_s)
        cost_bps = cost_bps_per_leg * (long_to + short_to)

        bars.append({
            "time": t, "spread_ret_bps": spread_ret * 1e4,
            "long_turnover": long_to, "short_turnover": short_to,
            "cost_bps": cost_bps, "net_bps": spread_ret * 1e4 - cost_bps,
            "n_long": len(top_df), "n_short": len(bot_df), "skipped": 0,
        })

        # Update prev tracking
        cur_long = set(top_df["symbol"]); cur_short = set(bot_df["symbol"])
        prev_long_w, prev_short_w = long_w, short_w
        # Update prev_long_pc / prev_short_pc per cluster
        sym_to_c = get_sym_to_cluster(clusters)
        prev_long_pc = {c: set() for c in clusters}
        prev_short_pc = {c: set() for c in clusters}
        for s in cur_long:
            c = sym_to_c.get(s)
            if c: prev_long_pc[c].add(s)
        for s in cur_short:
            c = sym_to_c.get(s)
            if c: prev_short_pc[c].add(s)

    return pd.DataFrame(bars)


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def main():
    universe_full = sorted(list_universe(min_days=200))
    orig25 = sorted([s for s in universe_full if s not in NEW_SYMBOLS])
    clusters = load_clusters()
    print(f"Loaded {len(clusters)} clusters covering {sum(len(s) for s in clusters.values())} names")
    print(f"FULL39 universe: {len(universe_full)} names")
    print()

    panel_full = build_wide_panel_for(universe_full)
    folds = _multi_oos_splits(panel_full)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    avail_feats = [c for c in v6_clean if c in panel_full.columns]

    configs = [
        # (label, k_per_cluster, use_conv, use_pm)
        ("FULL clust K_c=1 baseline", 1, False, False),
        ("FULL clust K_c=1 + conv",   1, True,  False),
        ("FULL clust K_c=1 + PM",      1, False, True),
        ("FULL clust K_c=1 + conv+PM", 1, True,  True),
        ("FULL clust K_c=2 + conv+PM", 2, True,  True),
    ]
    cycles = {label: [] for label, _, _, _ in configs}

    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel_full, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200: continue
        Xt = tr[avail_feats].to_numpy(np.float32); yt_ = tr["demeaned_target"].to_numpy(np.float32)
        Xc = ca[avail_feats].to_numpy(np.float32); yc_ = ca["demeaned_target"].to_numpy(np.float32)
        models = [_train(Xt, yt_, Xc, yc_, seed=s) for s in ENSEMBLE_SEEDS]
        Xtest = test[avail_feats].to_numpy(np.float32)
        pred_test = np.mean([m.predict(Xtest, num_iteration=m.best_iteration) for m in models], axis=0)

        line = f"  fold {fold['fid']:>2}: "
        for label, kc, use_conv, use_pm in configs:
            df_eval = evaluate_clustered(
                test, pred_test, clusters=clusters, k_per_cluster=kc,
                use_conv_gate=use_conv, use_pm_gate=use_pm,
            )
            for _, row in df_eval.iterrows():
                cycles[label].append({
                    "fold": fold["fid"], "time": row["time"],
                    "net": row["net_bps"], "skipped": row["skipped"],
                    "n_long": row["n_long"], "n_short": row["n_short"],
                    "cost": row["cost_bps"],
                })
            net_arr = df_eval["net_bps"].to_numpy()
            line += f"{label[-12:]}={net_arr.mean():+.2f}({_sharpe(net_arr):+.1f}) "
        print(line + f"({time.time()-t0:.0f}s)")

    print("\n" + "=" * 110)
    print(f"PHASE D RESULTS: cluster-bucketed selection on FULL39")
    print("=" * 110)
    print(f"  {'config':<32}  {'n':>4}  {'mean_net':>8}  {'cost':>5}  {'K_avg':>5}  "
          f"{'Sharpe':>7}  {'CI_lo':>7}  {'CI_hi':>7}")
    for label, _, _, _ in configs:
        df_v = pd.DataFrame(cycles[label])
        if df_v.empty: continue
        net = df_v["net"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        active = df_v[df_v["skipped"] == 0]
        K_avg = (active["n_long"].mean() + active["n_short"].mean()) / 2 if not active.empty else 0
        print(f"  {label:<32}  {len(net):>4}  {net.mean():>+8.2f}  {df_v['cost'].mean():>5.2f}  {K_avg:>5.2f}  "
              f"{sh:>+7.2f}  {lo:>+7.2f}  {hi:>+7.2f}")

    # Compare to validated ORIG25 (+2.75)
    print(f"\n  Validated reference: ORIG25 global K=7 + conv+PM = +2.75")
    print(f"  Failed reference:    FULL39 global K=7 + conv+PM = +0.88")
    print(f"  Failed reference:    FULL39 global K=11 + conv+PM = +0.62")

    # Save per-cycle data
    for label, _, _, _ in configs:
        if cycles[label]:
            pd.DataFrame(cycles[label]).to_csv(
                OUT_DIR / f"{label.replace(' ', '_').replace('+', 'p')}_cycles.csv", index=False
            )
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
