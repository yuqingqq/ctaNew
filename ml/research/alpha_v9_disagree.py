"""Use cross-seed disagreement as a free uncertainty signal.

We already train 5 LGBM seeds and average them. The std across seeds at
inference is per-name uncertainty information that's currently thrown away.

Three variants composable with the validated conviction_gate (p=0.30):

  A. Disagree gate (cycle-level):
       cycle_disagreement = mean over names of std-across-seeds
       skip cycle if disagreement > 70th-pctile of trailing 252 cycles.
       Symmetric to conviction_gate (which skips low-dispersion cycles);
       this skips high-disagreement cycles.

  B. Compose conviction_gate + disagree_gate:
       skip if EITHER gate flags.

  C. Per-name filter (within-bar):
       restrict candidate pool to names with std < bar-median(std), then
       pick top-K from filtered pool. Drops names where the model is
       internally torn.

Multi-OOS framework, post-fix cost. Two baselines: sharp K=7 (raw) and
conviction_gate p=0.30 (validated winner). All variants paired against both.
"""
from __future__ import annotations
import json
import sys
import time
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import (
    ENSEMBLE_SEEDS, _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v8_h48_audit import build_wide_panel

HORIZON = 48
TOP_K = 7
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
GATE_LOOKBACK = 252
GATE_PCTILE_CONV = 0.30   # conviction_gate winner
GATE_PCTILE_DIS = 0.70    # disagree_gate: skip TOP 30% disagreement (= bottom of inverse)
OUT_DIR = REPO / "outputs/h48_disagree"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0


def _bn_scales(top_g, bot_g):
    beta_L = top_g["beta_short_vs_bk"].mean()
    beta_S = bot_g["beta_short_vs_bk"].mean()
    if beta_L < 0.1 or beta_S < 0.1 or (beta_L + beta_S) < 0.3:
        return 1.0, 1.0
    denom = beta_L + beta_S
    return (float(np.clip(2.0 * beta_S / denom, 0.5, 1.5)),
            float(np.clip(2.0 * beta_L / denom, 0.5, 1.5)))


def evaluate_with_disagree(
    test: pd.DataFrame, yt_mean: np.ndarray, yt_std: np.ndarray, *,
    use_conv_gate: bool = False, conv_pctile: float = GATE_PCTILE_CONV,
    use_dis_gate: bool = False, dis_pctile: float = GATE_PCTILE_DIS,
    use_name_filter: bool = False,
    top_k: int = TOP_K, cost_bps_per_leg: float = COST_PER_LEG,
    sample_every: int = HORIZON, gate_lookback: int = GATE_LOOKBACK,
) -> pd.DataFrame:
    cols = ["open_time", "symbol", "return_pct", "alpha_realized",
            "basket_fwd", "beta_short_vs_bk"]
    df = test[cols].copy()
    df["pred"] = yt_mean
    df["pred_std"] = yt_std
    times = sorted(df["open_time"].unique())
    if not times:
        return pd.DataFrame()
    if sample_every > 1:
        keep_times = set(times[::sample_every])
        df = df[df["open_time"].isin(keep_times)]

    bars = []
    prev_long_w: dict[str, float] = {}
    prev_short_w: dict[str, float] = {}
    conv_history: deque = deque(maxlen=gate_lookback)
    dis_history: deque = deque(maxlen=gate_lookback)

    for t, g in df.groupby("open_time"):
        if len(g) < 2 * top_k + 1:
            continue

        # Two cycle-wide signals
        sorted_g = g.sort_values("pred")
        bot = sorted_g.head(top_k)
        top = sorted_g.tail(top_k)
        dispersion = top["pred"].mean() - bot["pred"].mean()
        cycle_disagreement = g["pred_std"].mean()

        # Gate decisions
        skip = False
        if use_conv_gate and len(conv_history) >= 30:
            thr = np.quantile(list(conv_history), conv_pctile)
            if dispersion < thr:
                skip = True
        if use_dis_gate and len(dis_history) >= 30:
            thr = np.quantile(list(dis_history), dis_pctile)
            if cycle_disagreement > thr:
                skip = True
        conv_history.append(dispersion)
        dis_history.append(cycle_disagreement)

        if skip:
            bars.append({
                "time": t, "spread_ret_bps": 0.0, "long_turnover": 0.0,
                "short_turnover": 0.0, "cost_bps": 0.0, "net_bps": 0.0,
                "skipped": 1, "n_long": 0, "n_short": 0,
                "dispersion": dispersion, "disagreement": cycle_disagreement,
            })
            continue

        # Optional name filter
        if use_name_filter:
            median_std = g["pred_std"].median()
            pool = g[g["pred_std"] < median_std]
            if len(pool) < 2 * top_k + 1:
                # Fall back to full universe if filter is too aggressive
                pool = g
            sorted_g = pool.sort_values("pred")
            bot = sorted_g.head(top_k)
            top = sorted_g.tail(top_k)

        scale_L, scale_S = _bn_scales(top, bot)
        n_l = len(top); n_s = len(bot)
        long_w = {s: scale_L / n_l for s in top["symbol"]}
        short_w = {s: scale_S / n_s for s in bot["symbol"]}
        long_ret = scale_L * top["return_pct"].mean()
        short_ret = scale_S * bot["return_pct"].mean()
        spread_ret = long_ret - short_ret

        if not prev_long_w:
            long_to, short_to = scale_L, scale_S
        else:
            all_l = set(long_w) | set(prev_long_w)
            long_to = sum(abs(long_w.get(s, 0) - prev_long_w.get(s, 0)) for s in all_l)
            all_s = set(short_w) | set(prev_short_w)
            short_to = sum(abs(short_w.get(s, 0) - prev_short_w.get(s, 0)) for s in all_s)
        bar_cost_bps = cost_bps_per_leg * (long_to + short_to)
        net_bps = (spread_ret * 1e4) - bar_cost_bps

        bars.append({
            "time": t, "spread_ret_bps": spread_ret * 1e4,
            "long_turnover": long_to, "short_turnover": short_to,
            "cost_bps": bar_cost_bps, "net_bps": net_bps,
            "skipped": 0, "n_long": len(top), "n_short": len(bot),
            "dispersion": dispersion, "disagreement": cycle_disagreement,
        })
        prev_long_w, prev_short_w = long_w, short_w

    return pd.DataFrame(bars)


def main():
    panel = build_wide_panel()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    print(f"Multi-OOS folds: {len(folds)}")

    variants = [
        ("sharp",                    False, False, False),
        ("conv_gate",                 True, False, False),  # validated winner
        ("dis_gate",                 False,  True, False),
        ("compose_both",              True,  True, False),
        ("conv_gate_namefilter",      True, False,  True),
        ("compose_both_namefilter",   True,  True,  True),
    ]
    cycles: dict[str, list] = {v[0]: [] for v in variants}

    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            continue
        avail = [c for c in v6_clean if c in panel.columns]
        Xt = tr[avail].to_numpy(dtype=np.float32)
        yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
        Xc = ca[avail].to_numpy(dtype=np.float32)
        yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
        models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
        Xtest = test[avail].to_numpy(dtype=np.float32)
        # 5-seed predictions stacked: shape (n_seeds, n_rows)
        preds_per_seed = np.array([m.predict(Xtest, num_iteration=m.best_iteration)
                                     for m in models])
        yt_mean = preds_per_seed.mean(axis=0)
        yt_std = preds_per_seed.std(axis=0)

        for name, conv, dis, nf in variants:
            df = evaluate_with_disagree(
                test, yt_mean, yt_std,
                use_conv_gate=conv, use_dis_gate=dis, use_name_filter=nf,
            )
            for _, r in df.iterrows():
                cycles[name].append({
                    "fold": fold["fid"], "time": r["time"],
                    "gross": r["spread_ret_bps"], "cost": r["cost_bps"],
                    "net": r["net_bps"], "long_turn": r["long_turnover"],
                    "skipped": r["skipped"],
                })
        print(f"  fold {fold['fid']:>2}: {time.time() - t0:.0f}s  "
              f"pred_std mean={yt_std.mean():.4f}, max={yt_std.max():.4f}")

    print("\n" + "=" * 120)
    print(f"DISAGREEMENT SWEEP (h={HORIZON} K={TOP_K}, β-neutral, {COST_PER_LEG} bps/leg, post-fix cost)")
    print("=" * 120)
    print(f"  {'variant':<28} {'n_cyc':>5} {'%trade':>7} {'gross':>7} {'cost':>6} "
          f"{'net':>7} {'L_turn':>7} {'Sharpe':>7} {'95% CI':>15} "
          f"{'Δsharp_Sh':>10} {'Δgate_Sh':>10}")

    base_recs = pd.DataFrame(cycles["sharp"])
    base_arr = base_recs["net"].to_numpy()
    gate_recs = pd.DataFrame(cycles["conv_gate"])
    gate_arr = gate_recs["net"].to_numpy()

    summary = {}
    for name, _, _, _ in variants:
        df = pd.DataFrame(cycles[name])
        if df.empty: continue
        traded = df[df["skipped"] == 0]
        pct_trade = 100 * len(traded) / len(df) if len(df) > 0 else 0
        sh, lo, hi = block_bootstrap_ci(df["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)

        # Paired delta vs sharp
        m_sh = base_recs[["fold", "time", "net"]].rename(columns={"net": "base"}).merge(
            df[["fold", "time", "net"]], on=["fold", "time"], how="inner")
        d_vs_sharp = sharpe_est((m_sh["net"] - m_sh["base"]).to_numpy())

        # Paired delta vs conv_gate
        m_g = gate_recs[["fold", "time", "net"]].rename(columns={"net": "base_g"}).merge(
            df[["fold", "time", "net"]], on=["fold", "time"], how="inner")
        d_vs_gate = sharpe_est((m_g["net"] - m_g["base_g"]).to_numpy())

        print(f"  {name:<28} {len(df):>5d} {pct_trade:>6.1f}% "
              f"{traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>5.2f}  "
              f"{df['net'].mean():>+6.2f}  "
              f"{traded['long_turn'].mean() if len(traded) > 0 else 0:>6.0%}  "
              f"{sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  "
              f"{d_vs_sharp:>+9.2f}  {d_vs_gate:>+9.2f}")
        summary[name] = {
            "n_cycles": int(len(df)), "pct_trade": float(pct_trade),
            "gross_traded": float(traded["gross"].mean() if len(traded) > 0 else 0),
            "cost_traded": float(traded["cost"].mean() if len(traded) > 0 else 0),
            "net_overall": float(df["net"].mean()),
            "sharpe": float(sh), "ci": [float(lo), float(hi)],
            "delta_sharpe_vs_sharp": float(d_vs_sharp),
            "delta_sharpe_vs_convgate": float(d_vs_gate),
        }

    with open(OUT_DIR / "alpha_v9_disagree_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    for name, _, _, _ in variants:
        pd.DataFrame(cycles[name]).to_csv(OUT_DIR / f"{name}_cycles.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
