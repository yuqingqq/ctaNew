"""Test conviction-aware position rules vs equal-weight K=7 baseline.

Three variants share the SAME trained model (z-score target, post-fix
cost framework). Only the position rule changes:

  A. Magnitude-weighted top-7/bot-7
       weights ∝ |pred_i - bar_median| within each leg, normalized to
       sum to scale_L (or scale_S) under β-neutral.
       → bigger position on outlier, smaller on borderline pick.

  B. Conviction gate
       compute dispersion = top_K_mean(pred) - bot_K_mean(pred). If below
       PIT trailing 30th-percentile threshold, skip cycle. Else equal-weight.
       → no trade when model has nothing to say.

  C. Variable K via threshold
       long = names with pred > median + σ_bar × mult
       short = names with pred < median - σ_bar × mult
       (skip if either side empty). Trades fewer names in compressed bars,
       more in dispersed bars.

All three composable with the existing β-neutral framework. Multi-OOS
paired comparison vs sharp K=7 baseline.
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
from ml.research.alpha_v4_xs import portfolio_pnl_turnover_aware, block_bootstrap_ci
from ml.research.alpha_v8_h48_audit import build_wide_panel

HORIZON = 48
TOP_K = 7
TOP_FRAC = TOP_K / 25.0
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
GATE_LOOKBACK = 252  # ~6 weeks of cycles for trailing percentile
GATE_PCTILE = 0.30   # skip cycles in bottom 30% of dispersion
VAR_K_MULT = 0.5     # |pred - median| > 0.5σ_bar
OUT_DIR = REPO / "outputs/h48_conviction"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _bn_scales(top_g: pd.DataFrame, bot_g: pd.DataFrame) -> tuple[float, float, bool]:
    """β-neutral leg scaling — matches portfolio_pnl_turnover_aware."""
    beta_L = top_g["beta_short_vs_bk"].mean()
    beta_S = bot_g["beta_short_vs_bk"].mean()
    if beta_L < 0.1 or beta_S < 0.1 or (beta_L + beta_S) < 0.3:
        return 1.0, 1.0, True
    denom = beta_L + beta_S
    return (float(np.clip(2.0 * beta_S / denom, 0.5, 1.5)),
            float(np.clip(2.0 * beta_L / denom, 0.5, 1.5)), False)


def portfolio_pnl_general(
    test: pd.DataFrame, yt: np.ndarray, *, mode: str,
    top_k: int = 7, cost_bps_per_leg: float = 4.5,
    sample_every: int = 48, gate_lookback: int = GATE_LOOKBACK,
    gate_pctile: float = GATE_PCTILE, var_k_mult: float = VAR_K_MULT,
) -> dict:
    """Unified portfolio P&L supporting modes: 'sharp', 'magweight',
    'conviction_gate', 'var_k'. β-neutral always on. Same cost formula
    as portfolio_pnl_turnover_aware (post-fix: no 0.5× factor).
    """
    cols = ["open_time", "symbol", "return_pct", "alpha_realized",
            "basket_fwd", "beta_short_vs_bk"]
    df = test[cols].copy()
    df["pred"] = yt
    times = sorted(df["open_time"].unique())
    if not times:
        return {"n_bars": 0}
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
        bar_std_pred = sorted_g["pred"].std()
        dispersion = top["pred"].mean() - bot["pred"].mean()

        # Decide whether to trade
        skip = False
        if mode == "conviction_gate":
            if len(dispersion_history) >= 30:
                thr = np.quantile(list(dispersion_history), gate_pctile)
                if dispersion < thr:
                    skip = True
            dispersion_history.append(dispersion)
        else:
            dispersion_history.append(dispersion)

        # Build long/short pools
        if mode == "var_k":
            cutoff = bar_std_pred * var_k_mult
            long_pool = sorted_g[sorted_g["pred"] > median_pred + cutoff]
            short_pool = sorted_g[sorted_g["pred"] < median_pred - cutoff]
            if len(long_pool) == 0 or len(short_pool) == 0:
                skip = True
            else:
                top, bot = long_pool, short_pool
        # else sharp/magweight/conviction_gate use top/bot from K=7

        if skip:
            # No trade this cycle: positions unchanged. Carry prev_w (if any)
            # No turnover, no cost. P&L is zero (we're flat per cycle since
            # this is non-overlapping label evaluation).
            bars.append({
                "time": t, "n": len(g), "n_long": 0, "n_short": 0,
                "spread_ret_bps": 0.0, "spread_alpha_bps": 0.0, "rank_ic": np.nan,
                "long_turnover": 0.0, "short_turnover": 0.0,
                "cost_bps": 0.0, "net_bps": 0.0,
                "scale_L": 0.0, "scale_S": 0.0, "skipped": 1, "dispersion": dispersion,
            })
            # IMPORTANT: skipping means closing prev positions at this point?
            # In a true forward harness, skipping = hold previous positions.
            # In this NON-OVERLAPPING evaluation, each cycle is independent
            # and prev_w only matters for turnover accounting. We keep
            # prev_long_w / prev_short_w unchanged so a future trade still
            # accounts for the LAST traded portfolio.
            continue

        scale_L, scale_S, degen_beta = _bn_scales(top, bot)

        if mode == "magweight":
            # Within each leg: weight ∝ |pred - median|. Normalize so sum = scale.
            top_dev = (top["pred"] - median_pred).abs()
            bot_dev = (bot["pred"] - median_pred).abs()
            wsum_l = top_dev.sum()
            wsum_s = bot_dev.sum()
            if wsum_l <= 0 or wsum_s <= 0:
                # Degenerate (all preds equal) — fall back to equal-weight
                long_w = {s: scale_L / top_k for s in top["symbol"]}
                short_w = {s: scale_S / top_k for s in bot["symbol"]}
            else:
                long_w = {s: scale_L * d / wsum_l
                          for s, d in zip(top["symbol"], top_dev)}
                short_w = {s: scale_S * d / wsum_s
                           for s, d in zip(bot["symbol"], bot_dev)}
            long_ret = sum(long_w[s] * r for s, r in
                           zip(top["symbol"], top["return_pct"])) / scale_L
            short_ret = sum(short_w[s] * r for s, r in
                            zip(bot["symbol"], bot["return_pct"])) / scale_S
            long_alpha = sum(long_w[s] * a for s, a in
                              zip(top["symbol"], top["alpha_realized"])) / scale_L
            short_alpha = sum(short_w[s] * a for s, a in
                               zip(bot["symbol"], bot["alpha_realized"])) / scale_S
            # Recover scaled returns per leg
            long_ret = long_ret * scale_L
            short_ret = short_ret * scale_S
            long_alpha = long_alpha * scale_L
            short_alpha = short_alpha * scale_S
        else:
            # sharp / conviction_gate (when not skipped) / var_k all use equal weight
            n_l = len(top)
            n_s = len(bot)
            long_w = {s: scale_L / n_l for s in top["symbol"]}
            short_w = {s: scale_S / n_s for s in bot["symbol"]}
            long_ret = scale_L * top["return_pct"].mean()
            short_ret = scale_S * bot["return_pct"].mean()
            long_alpha = scale_L * top["alpha_realized"].mean()
            short_alpha = scale_S * bot["alpha_realized"].mean()

        spread_ret = long_ret - short_ret
        spread_alpha = long_alpha - short_alpha
        ic = g["pred"].rank().corr(g["alpha_realized"].rank())

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
            "time": t, "n": len(g), "n_long": len(top), "n_short": len(bot),
            "spread_ret_bps": spread_ret * 1e4,
            "spread_alpha_bps": spread_alpha * 1e4,
            "rank_ic": ic,
            "long_turnover": long_to, "short_turnover": short_to,
            "cost_bps": bar_cost_bps, "net_bps": net_bps,
            "scale_L": scale_L, "scale_S": scale_S, "skipped": 0, "dispersion": dispersion,
        })
        prev_long_w, prev_short_w = long_w, short_w

    bdf = pd.DataFrame(bars)
    return {"n_bars": len(bdf), "df": bdf}


def main():
    panel = build_wide_panel()
    folds = _multi_oos_splits(panel)
    print(f"Multi-OOS folds: {len(folds)}")

    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    modes = ["sharp", "magweight", "conviction_gate", "var_k"]
    cycle_records: dict[str, list] = {m: [] for m in modes}

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

        for mode in modes:
            r = portfolio_pnl_general(
                test, yt_pred, mode=mode, top_k=TOP_K,
                cost_bps_per_leg=COST_PER_LEG, sample_every=HORIZON,
            )
            if r.get("n_bars", 0) > 0:
                for _, row in r["df"].iterrows():
                    cycle_records[mode].append({
                        "fold": fold["fid"], "time": row["time"],
                        "spread": row["spread_ret_bps"],
                        "alpha": row["spread_alpha_bps"],
                        "cost": row["cost_bps"],
                        "net": row["net_bps"],
                        "long_turn": row["long_turnover"],
                        "short_turn": row["short_turnover"],
                        "skipped": row["skipped"],
                        "n_long": row["n_long"], "n_short": row["n_short"],
                    })
        print(f"  fold {fold['fid']:>2}: {time.time() - t0:.0f}s")

    # Summarize
    print("\n" + "=" * 110)
    print(f"CONVICTION SWEEP (h={HORIZON} K={TOP_K}, β-neutral, {COST_PER_LEG} bps/leg, post-fix cost)")
    print("=" * 110)
    print(f"  {'mode':<22} {'n_cyc':>7} {'%trade':>7} {'gross':>8} {'cost':>7} "
          f"{'net':>8} {'L_turn':>7} {'#L':>4} {'Sharpe':>8} {'95% CI':>16}")

    summary_dict = {}
    for mode in modes:
        records = cycle_records[mode]
        if not records:
            print(f"  {mode:<22}  NO DATA")
            continue
        df = pd.DataFrame(records)
        traded = df[df["skipped"] == 0]
        pct_trade = 100 * len(traded) / len(df) if len(df) > 0 else 0

        sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0
        sh, lo, hi = block_bootstrap_ci(df["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)
        print(f"  {mode:<22} {len(df):>7d} {pct_trade:>6.1f}% "
              f"{traded['spread'].mean() if len(traded) > 0 else 0:>+7.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>6.2f}  "
              f"{df['net'].mean():>+7.2f}  "
              f"{traded['long_turn'].mean() if len(traded) > 0 else 0:>6.0%} "
              f"{traded['n_long'].mean() if len(traded) > 0 else 0:>4.1f}  "
              f"{sh:>+7.2f}  [{lo:>+5.2f},{hi:>+5.2f}]")
        summary_dict[mode] = {
            "n_cycles": int(len(df)), "pct_trade": float(pct_trade),
            "gross_traded": float(traded["spread"].mean() if len(traded) > 0 else 0),
            "cost_traded": float(traded["cost"].mean() if len(traded) > 0 else 0),
            "net_overall": float(df["net"].mean()),
            "long_turn_traded": float(traded["long_turn"].mean() if len(traded) > 0 else 0),
            "n_long_traded": float(traded["n_long"].mean() if len(traded) > 0 else 0),
            "sharpe": float(sh), "ci": [float(lo), float(hi)],
        }

    # Paired delta vs sharp baseline (per-time alignment)
    print("\n  --- PAIRED Δ vs sharp baseline (per cycle) ---")
    base_df = pd.DataFrame(cycle_records["sharp"])[["fold", "time", "net"]].rename(
        columns={"net": "base_net"})
    for mode in modes:
        if mode == "sharp": continue
        var_df = pd.DataFrame(cycle_records[mode])[["fold", "time", "net"]].rename(
            columns={"net": f"{mode}_net"})
        m = base_df.merge(var_df, on=["fold", "time"], how="inner")
        delta = (m[f"{mode}_net"] - m["base_net"]).to_numpy()
        sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0
        d_sh = sharpe_est(delta)
        t = delta.mean() / (delta.std() / np.sqrt(len(delta))) if delta.std() > 0 else 0
        p = 1 - stats.norm.cdf(abs(t))
        wins = (delta > 0).mean() * 100
        print(f"  {mode:<22} ΔSharpe={d_sh:+.2f}  Δnet={delta.mean():+.3f} bps/cyc  "
              f"t={t:+.2f}  p={p:.4f}  wins {wins:.1f}%")
        summary_dict[f"{mode}_vs_sharp"] = {
            "delta_sharpe": float(d_sh), "delta_net_bps": float(delta.mean()),
            "t_stat": float(t), "p_value": float(p), "wins_pct": float(wins),
        }

    with open(OUT_DIR / "alpha_v9_conviction_summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)
    pd.DataFrame(cycle_records["sharp"]).to_csv(OUT_DIR / "sharp_cycles.csv", index=False)
    pd.DataFrame(cycle_records["magweight"]).to_csv(OUT_DIR / "magweight_cycles.csv", index=False)
    pd.DataFrame(cycle_records["conviction_gate"]).to_csv(OUT_DIR / "gate_cycles.csv", index=False)
    pd.DataFrame(cycle_records["var_k"]).to_csv(OUT_DIR / "var_k_cycles.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
