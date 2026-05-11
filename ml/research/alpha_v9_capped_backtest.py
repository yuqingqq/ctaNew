"""Phase D-2: cluster-CAPPED selection backtest on FULL39.

Pivot from forced cluster-bucketed (K_c=1 forces a pick per cluster, including
weak ones) to cluster-CAPPED (top-K global ranking, max_per_cluster cap).

Target: beat ORIG25 +2.75 OR at least beat FULL39 global K=7 +0.88.

Sweeps max_per_cluster ∈ {2, 3} on FULL39 K=7.
"""
from __future__ import annotations
import sys, time, warnings
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
from ml.research.portfolio_clustered import (
    load_clusters, select_portfolio_capped, pm_gate_filter_capped,
)

HORIZON = 48
TOP_K = 7
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
GATE_LOOKBACK = 252
GATE_PCTILE = 0.30
PM_M = 2
PM_BAND = 1.0
OUT_DIR = REPO / "outputs/capped_backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_SYMBOLS = {"ETCUSDT", "HBARUSDT", "ICPUSDT", "LDOUSDT", "TRBUSDT",
               "AAVEUSDT", "MKRUSDT", "AXSUSDT", "GMXUSDT",
               "1000PEPEUSDT", "1000SHIBUSDT", "TONUSDT", "ORDIUSDT", "WIFUSDT"}


def evaluate_capped(
    test: pd.DataFrame, yt: np.ndarray, *,
    clusters: dict[str, list[str]],
    top_k: int = TOP_K,
    max_per_cluster: int = 2,
    use_conv_gate: bool = True, use_pm_gate: bool = True,
    cost_bps_per_leg: float = COST_PER_LEG,
    sample_every: int = HORIZON,
) -> pd.DataFrame:
    cols = ["open_time", "symbol", "return_pct", "alpha_realized",
            "basket_fwd", "beta_short_vs_bk"]
    df = test[cols].copy()
    df["pred"] = yt
    times = sorted(df["open_time"].unique())
    if not times: return pd.DataFrame()
    if sample_every > 1:
        keep_times = set(times[::sample_every])
        df = df[df["open_time"].isin(keep_times)]

    band_k = max(top_k, int(round(PM_BAND * top_k)))
    history = []
    dispersion_history = deque(maxlen=GATE_LOOKBACK)
    cur_long, cur_short = set(), set()
    prev_long_w, prev_short_w = {}, {}

    bars = []
    for t, g in df.groupby("open_time"):
        if len(g) < 2 * top_k + 1: continue
        sg = g.sort_values("pred")
        bot_global = sg.head(top_k)
        top_global = sg.tail(top_k)
        dispersion = float(top_global["pred"].mean() - bot_global["pred"].mean())
        skip = False
        if use_conv_gate and len(dispersion_history) >= 30:
            thr = float(np.quantile(list(dispersion_history), GATE_PCTILE))
            if dispersion < thr:
                skip = True
        dispersion_history.append(dispersion)

        # Update history every cycle
        bk = min(band_k, len(g))
        history.append({
            "long": sg.tail(bk)["symbol"].tolist(),
            "short": sg.head(bk)["symbol"].tolist(),
        })
        if len(history) > PM_M:
            history = history[-PM_M:]

        if skip:
            bars.append({
                "time": t, "spread_ret_bps": 0.0,
                "long_turnover": 0.0, "short_turnover": 0.0,
                "cost_bps": 0.0, "net_bps": 0.0,
                "n_long": 0, "n_short": 0, "skipped": 1,
            })
            cur_long, cur_short = set(), set()
            prev_long_w = {}; prev_short_w = {}
            continue

        if use_pm_gate:
            top_df, bot_df, scale_L, scale_S, info = pm_gate_filter_capped(
                g, clusters, history=history[:-1],
                prev_long_syms=cur_long, prev_short_syms=cur_short,
                top_k=top_k, max_per_cluster=max_per_cluster,
                M_cycles=PM_M, band_mult=PM_BAND,
            )
        else:
            top_df, bot_df, scale_L, scale_S, info = select_portfolio_capped(
                g, clusters, top_k=top_k, max_per_cluster=max_per_cluster,
            )

        if top_df.empty or bot_df.empty:
            cur_long = set(top_df["symbol"]); cur_short = set(bot_df["symbol"])
            prev_long_w = {s: 1.0 / top_k for s in cur_long}
            prev_short_w = {s: 1.0 / top_k for s in cur_short}
            bars.append({
                "time": t, "spread_ret_bps": 0.0,
                "long_turnover": 0.0, "short_turnover": 0.0,
                "cost_bps": 0.0, "net_bps": 0.0,
                "n_long": len(top_df), "n_short": len(bot_df), "skipped": 0,
            })
            continue

        long_w = {s: scale_L / top_k for s in top_df["symbol"]}
        short_w = {s: scale_S / top_k for s in bot_df["symbol"]}
        gross_L = scale_L * len(top_df) / top_k
        gross_S = scale_S * len(bot_df) / top_k

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
        cur_long = set(top_df["symbol"]); cur_short = set(bot_df["symbol"])
        prev_long_w, prev_short_w = long_w, short_w

    return pd.DataFrame(bars)


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def main():
    universe_full = sorted(list_universe(min_days=200))
    clusters = load_clusters()
    print(f"Universe: {len(universe_full)} (FULL39); {len(clusters)} clusters")
    panel_full = build_wide_panel_for(universe_full)
    folds = _multi_oos_splits(panel_full)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    avail_feats = [c for c in v6_clean if c in panel_full.columns]

    configs = [
        # (label, max_per_cluster, use_conv, use_pm)
        ("FULL cap=2 baseline",       2, False, False),
        ("FULL cap=2 + conv+PM",      2, True,  True),
        ("FULL cap=3 + conv+PM",      3, True,  True),
        ("FULL cap=4 + conv+PM",      4, True,  True),  # near-unconstrained
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
        for label, mpc, use_conv, use_pm in configs:
            df_eval = evaluate_capped(
                test, pred_test, clusters=clusters, top_k=TOP_K,
                max_per_cluster=mpc, use_conv_gate=use_conv, use_pm_gate=use_pm,
            )
            for _, row in df_eval.iterrows():
                cycles[label].append({
                    "fold": fold["fid"], "time": row["time"],
                    "net": row["net_bps"], "skipped": row["skipped"],
                    "n_long": row["n_long"], "n_short": row["n_short"],
                    "cost": row["cost_bps"],
                })
            net_arr = df_eval["net_bps"].to_numpy()
            line += f"{label[5:18]}={net_arr.mean():+.2f}({_sharpe(net_arr):+.1f}) "
        print(line + f"({time.time()-t0:.0f}s)")

    print("\n" + "=" * 110)
    print(f"PHASE D-2: cluster-CAPPED selection on FULL39")
    print("=" * 110)
    print(f"  {'config':<28}  {'n':>4}  {'mean_net':>8}  {'cost':>5}  {'K_avg':>5}  "
          f"{'Sharpe':>7}  {'CI_lo':>7}  {'CI_hi':>7}")
    for label, _, _, _ in configs:
        df_v = pd.DataFrame(cycles[label])
        if df_v.empty: continue
        net = df_v["net"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        active = df_v[df_v["skipped"] == 0]
        K_avg = (active["n_long"].mean() + active["n_short"].mean()) / 2 if not active.empty else 0
        print(f"  {label:<28}  {len(net):>4}  {net.mean():>+8.2f}  {df_v['cost'].mean():>5.2f}  {K_avg:>5.2f}  "
              f"{sh:>+7.2f}  {lo:>+7.2f}  {hi:>+7.2f}")

    print(f"\n  References:")
    print(f"    ORIG25 global K=7 + conv+PM = +2.75  ← target to beat")
    print(f"    FULL39 global K=7 + conv+PM = +0.88")
    print(f"    FULL39 global K=11 + conv+PM = +0.62")
    print(f"    FULL clust K_c=1 + conv+PM = -2.12  (forced bucketing fails)")

    for label, _, _, _ in configs:
        if cycles[label]:
            pd.DataFrame(cycles[label]).to_csv(
                OUT_DIR / f"{label.replace(' ', '_').replace('+', 'p')}_cycles.csv", index=False
            )
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
