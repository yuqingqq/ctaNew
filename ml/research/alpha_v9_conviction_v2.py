"""Follow-up sweep: gate_pctile sensitivity + gate+magweight composition.

From v9_conviction findings:
  - conviction_gate at 30th-pctile: ΔSharpe +1.85 (p=0.056) ← winner
  - magweight: wash (gross+ cost+ cancel)
  - var_k: significant negative

This script:
  1. Sweeps gate_pctile ∈ {0.20, 0.30, 0.40, 0.50}
  2. Tests gate + magweight composition (skip low-conviction cycles AND
     within-leg magnitude-weight when trading) at each gate_pctile

Same multi-OOS framework, same panel, same 5-seed predictions, post-fix
cost. All variants paired against sharp K=7 baseline.
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
OUT_DIR = REPO / "outputs/h48_conviction_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _bn_scales(top_g, bot_g):
    beta_L = top_g["beta_short_vs_bk"].mean()
    beta_S = bot_g["beta_short_vs_bk"].mean()
    if beta_L < 0.1 or beta_S < 0.1 or (beta_L + beta_S) < 0.3:
        return 1.0, 1.0
    denom = beta_L + beta_S
    return (float(np.clip(2.0 * beta_S / denom, 0.5, 1.5)),
            float(np.clip(2.0 * beta_L / denom, 0.5, 1.5)))


def evaluate_portfolio(
    test: pd.DataFrame, yt: np.ndarray, *,
    use_gate: bool, gate_pctile: float,
    use_magweight: bool, top_k: int = TOP_K,
    cost_bps_per_leg: float = COST_PER_LEG,
    sample_every: int = HORIZON,
    gate_lookback: int = GATE_LOOKBACK,
) -> pd.DataFrame:
    """Returns per-cycle DataFrame with net_bps, gross, cost, etc."""
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

    bars = []
    prev_long_w: dict[str, float] = {}
    prev_short_w: dict[str, float] = {}
    dispersion_history: deque = deque(maxlen=gate_lookback)

    for t, g in df.groupby("open_time"):
        if len(g) < 2 * top_k + 1:
            continue
        sorted_g = g.sort_values("pred")
        bot = sorted_g.head(top_k)
        top = sorted_g.tail(top_k)
        median_pred = sorted_g["pred"].median()
        dispersion = top["pred"].mean() - bot["pred"].mean()

        # Gate decision (always update history)
        skip = False
        if use_gate and len(dispersion_history) >= 30:
            thr = np.quantile(list(dispersion_history), gate_pctile)
            if dispersion < thr:
                skip = True
        dispersion_history.append(dispersion)

        if skip:
            bars.append({
                "time": t, "spread_ret_bps": 0.0, "spread_alpha_bps": 0.0,
                "long_turnover": 0.0, "short_turnover": 0.0,
                "cost_bps": 0.0, "net_bps": 0.0, "skipped": 1,
                "n_long": 0, "n_short": 0, "dispersion": dispersion,
            })
            continue

        scale_L, scale_S = _bn_scales(top, bot)
        if use_magweight:
            top_dev = (top["pred"] - median_pred).abs()
            bot_dev = (bot["pred"] - median_pred).abs()
            wsum_l = top_dev.sum()
            wsum_s = bot_dev.sum()
            if wsum_l <= 0 or wsum_s <= 0:
                long_w = {s: scale_L / top_k for s in top["symbol"]}
                short_w = {s: scale_S / top_k for s in bot["symbol"]}
                long_ret = scale_L * top["return_pct"].mean()
                short_ret = scale_S * bot["return_pct"].mean()
                long_alpha = scale_L * top["alpha_realized"].mean()
                short_alpha = scale_S * bot["alpha_realized"].mean()
            else:
                long_w = {s: scale_L * d / wsum_l
                           for s, d in zip(top["symbol"], top_dev)}
                short_w = {s: scale_S * d / wsum_s
                            for s, d in zip(bot["symbol"], bot_dev)}
                # Weighted-leg returns: sum_i w_i × r_i (sum already scaled
                # by scale_L because weights sum to scale_L)
                long_ret = sum(long_w[s] * r for s, r in
                                zip(top["symbol"], top["return_pct"]))
                short_ret = sum(short_w[s] * r for s, r in
                                 zip(bot["symbol"], bot["return_pct"]))
                long_alpha = sum(long_w[s] * a for s, a in
                                  zip(top["symbol"], top["alpha_realized"]))
                short_alpha = sum(short_w[s] * a for s, a in
                                   zip(bot["symbol"], bot["alpha_realized"]))
        else:
            long_w = {s: scale_L / top_k for s in top["symbol"]}
            short_w = {s: scale_S / top_k for s in bot["symbol"]}
            long_ret = scale_L * top["return_pct"].mean()
            short_ret = scale_S * bot["return_pct"].mean()
            long_alpha = scale_L * top["alpha_realized"].mean()
            short_alpha = scale_S * bot["alpha_realized"].mean()

        spread_ret = long_ret - short_ret
        spread_alpha = long_alpha - short_alpha

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
            "spread_alpha_bps": spread_alpha * 1e4,
            "long_turnover": long_to, "short_turnover": short_to,
            "cost_bps": bar_cost_bps, "net_bps": net_bps,
            "skipped": 0, "n_long": len(top), "n_short": len(bot),
            "dispersion": dispersion,
        })
        prev_long_w, prev_short_w = long_w, short_w

    return pd.DataFrame(bars)


def main():
    panel = build_wide_panel()
    folds = _multi_oos_splits(panel)
    print(f"Multi-OOS folds: {len(folds)}")

    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)

    variants = [
        ("sharp",                False, 0.30, False),  # baseline
        ("gate_p20",              True, 0.20, False),
        ("gate_p30",              True, 0.30, False),
        ("gate_p40",              True, 0.40, False),
        ("gate_p50",              True, 0.50, False),
        ("magweight",            False, 0.30, True),
        ("gate_p20_mag",          True, 0.20, True),
        ("gate_p30_mag",          True, 0.30, True),
        ("gate_p40_mag",          True, 0.40, True),
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
        yt_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                            for m in models], axis=0)

        for name, use_gate, gate_p, use_mag in variants:
            df = evaluate_portfolio(
                test, yt_pred,
                use_gate=use_gate, gate_pctile=gate_p,
                use_magweight=use_mag,
            )
            for _, row in df.iterrows():
                cycles[name].append({
                    "fold": fold["fid"], "time": row["time"],
                    "gross": row["spread_ret_bps"],
                    "cost": row["cost_bps"], "net": row["net_bps"],
                    "long_turn": row["long_turnover"],
                    "skipped": row["skipped"],
                })
        print(f"  fold {fold['fid']:>2}: {time.time() - t0:.0f}s")

    # Summary table
    print("\n" + "=" * 120)
    print(f"CONVICTION-GATE SWEEP + MAGWEIGHT COMPOSITION  (h={HORIZON} K={TOP_K}, β-neutral, "
          f"{COST_PER_LEG} bps/leg, post-fix cost)")
    print("=" * 120)
    print(f"  {'variant':<18} {'n_cyc':>5} {'%trade':>7} {'gross':>7} {'cost':>7} "
          f"{'net':>7} {'L_turn':>7} {'Sharpe':>7} {'95% CI':>16} "
          f"{'ΔSharpe':>9} {'Δnet':>7} {'p_one':>7}")

    base_records = pd.DataFrame(cycles["sharp"])
    base_net = base_records["net"].to_numpy()
    sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0

    summary_dict = {}
    for name, _, _, _ in variants:
        df = pd.DataFrame(cycles[name])
        if df.empty: continue
        traded = df[df["skipped"] == 0]
        pct_trade = 100 * len(traded) / len(df) if len(df) > 0 else 0
        sh, lo, hi = block_bootstrap_ci(df["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)
        if name == "sharp":
            d_sh, d_net, p = 0.0, 0.0, 0.5
        else:
            m = base_records[["fold", "time", "net"]].rename(columns={"net": "base"}).merge(
                df[["fold", "time", "net"]], on=["fold", "time"], how="inner")
            delta = (m["net"] - m["base"]).to_numpy()
            d_sh = sharpe_est(delta)
            d_net = delta.mean()
            t = delta.mean() / (delta.std() / np.sqrt(len(delta))) if delta.std() > 0 else 0
            p = 1 - stats.norm.cdf(abs(t))
        print(f"  {name:<18} {len(df):>5d} {pct_trade:>6.1f}% "
              f"{traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>6.2f}  "
              f"{df['net'].mean():>+6.2f}  "
              f"{traded['long_turn'].mean() if len(traded) > 0 else 0:>6.0%}  "
              f"{sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  "
              f"{d_sh:>+8.2f}  {d_net:>+6.3f}  {p:>6.4f}")
        summary_dict[name] = {
            "n_cycles": int(len(df)), "pct_trade": float(pct_trade),
            "gross_traded": float(traded["gross"].mean() if len(traded) > 0 else 0),
            "cost_traded": float(traded["cost"].mean() if len(traded) > 0 else 0),
            "net_overall": float(df["net"].mean()),
            "long_turn_traded": float(traded["long_turn"].mean() if len(traded) > 0 else 0),
            "sharpe": float(sh), "ci": [float(lo), float(hi)],
            "delta_sharpe_vs_sharp": float(d_sh),
            "delta_net_vs_sharp": float(d_net),
            "p_one_sided": float(p),
        }

    with open(OUT_DIR / "alpha_v9_conviction_v2_summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)
    for name, _, _, _ in variants:
        pd.DataFrame(cycles[name]).to_csv(OUT_DIR / f"{name}_cycles.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
