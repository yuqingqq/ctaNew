"""Validate M=2 hysteresis under the proper multi-OOS framework.

Mirrors `alpha_v8_h48_audit.py` exactly (same panel, same folds, same
beta-neutral execution, same 4.5 bps/leg HL VIP-0 cost) — but evaluates
sharp-boundary baseline vs M=2 hysteresis on the SAME predictions and
SAME OOS cycles. Paired comparison.

If the +1.27 Sharpe lift from the 2-fold quick test holds here, hysteresis
is a deployable upgrade. If it shrinks below ~+0.3 Sharpe in the bootstrap CI,
treat it as noise.
"""
from __future__ import annotations
import json
import sys
import time
import warnings
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
COST_PER_LEG = 4.5  # HL VIP-0 taker — matches +3.63 baseline
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
EXIT_BUFFER = 2  # M=2 sweet spot from quick test
OUT_DIR = REPO / "outputs/h48_hysteresis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def portfolio_pnl_hysteresis_bn(test: pd.DataFrame, yt: np.ndarray, *,
                                  top_k: int, exit_buffer: int,
                                  cost_bps_per_leg: float, sample_every: int,
                                  beta_neutral: bool = False) -> dict:
    """Hysteresis variant of portfolio_pnl_turnover_aware. Names enter when
    rank ≤ top_k, exit only when rank > top_k + exit_buffer.

    Beta-neutral: same scaling/clipping/degenerate-fallback as the baseline
    function so the two are pair-comparable cycle-by-cycle.
    """
    cols = ["open_time", "symbol", "return_pct", "alpha_realized", "basket_fwd"]
    if beta_neutral:
        cols.append("beta_short_vs_bk")
    df = test[cols].copy()
    df["pred"] = yt
    times = sorted(df["open_time"].unique())
    if not times:
        return {"n_bars": 0}
    if sample_every > 1:
        keep_times = set(times[::sample_every])
        df = df[df["open_time"].isin(keep_times)]

    bars = []
    cur_long: set = set()
    cur_short: set = set()
    prev_long_w: dict[str, float] = {}
    prev_short_w: dict[str, float] = {}

    for t, g in df.groupby("open_time"):
        n = len(g)
        if n < 2 * top_k + exit_buffer:
            continue
        sorted_g = g.sort_values("pred").reset_index(drop=True)
        sorted_g["rank_top"] = n - 1 - sorted_g.index
        sorted_g["rank_bot"] = sorted_g.index

        # Long leg hysteresis
        new_long = set(cur_long)
        for s in list(new_long):
            r = sorted_g[sorted_g["symbol"] == s]
            if r.empty or r["rank_top"].iloc[0] > top_k + exit_buffer - 1:
                new_long.discard(s)
        for s in sorted_g[sorted_g["rank_top"] < top_k]["symbol"].tolist():
            if len(new_long) >= top_k: break
            new_long.add(s)
        if len(new_long) > top_k:
            ranked = sorted_g[sorted_g["symbol"].isin(new_long)].sort_values("rank_top")
            new_long = set(ranked.head(top_k)["symbol"])

        # Short leg hysteresis
        new_short = set(cur_short)
        for s in list(new_short):
            r = sorted_g[sorted_g["symbol"] == s]
            if r.empty or r["rank_bot"].iloc[0] > top_k + exit_buffer - 1:
                new_short.discard(s)
        for s in sorted_g[sorted_g["rank_bot"] < top_k]["symbol"].tolist():
            if len(new_short) >= top_k: break
            new_short.add(s)
        if len(new_short) > top_k:
            ranked = sorted_g[sorted_g["symbol"].isin(new_short)].sort_values("rank_bot")
            new_short = set(ranked.head(top_k)["symbol"])

        if not new_long or not new_short:
            cur_long, cur_short = new_long, new_short
            continue

        long_g = sorted_g[sorted_g["symbol"].isin(new_long)]
        short_g = sorted_g[sorted_g["symbol"].isin(new_short)]

        if beta_neutral:
            beta_L = long_g["beta_short_vs_bk"].mean()
            beta_S = short_g["beta_short_vs_bk"].mean()
            if beta_L < 0.1 or beta_S < 0.1 or (beta_L + beta_S) < 0.3:
                scale_L, scale_S = 1.0, 1.0
                degen_beta = True
            else:
                denom = beta_L + beta_S
                scale_L = float(np.clip(2.0 * beta_S / denom, 0.5, 1.5))
                scale_S = float(np.clip(2.0 * beta_L / denom, 0.5, 1.5))
                degen_beta = False
        else:
            scale_L, scale_S, degen_beta = 1.0, 1.0, False

        long_ret = scale_L * long_g["return_pct"].mean()
        short_ret = scale_S * short_g["return_pct"].mean()
        long_alpha = scale_L * long_g["alpha_realized"].mean()
        short_alpha = scale_S * short_g["alpha_realized"].mean()
        spread_ret = long_ret - short_ret
        spread_alpha = long_alpha - short_alpha
        ic = g["pred"].rank().corr(g["alpha_realized"].rank())

        long_w = {s: scale_L / top_k for s in new_long}
        short_w = {s: scale_S / top_k for s in new_short}
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
            "time": t, "n": n, "n_long": len(new_long), "n_short": len(new_short),
            "spread_ret_bps": spread_ret * 1e4,
            "spread_alpha_bps": spread_alpha * 1e4,
            "rank_ic": ic,
            "long_turnover": long_to, "short_turnover": short_to,
            "cost_bps": bar_cost_bps, "net_bps": net_bps,
            "scale_L": scale_L, "scale_S": scale_S,
            "degen_beta": int(degen_beta),
        })
        cur_long, cur_short = new_long, new_short
        prev_long_w, prev_short_w = long_w, short_w

    bdf = pd.DataFrame(bars)
    if bdf.empty:
        return {"n_bars": 0}
    return {
        "n_bars": len(bdf),
        "spread_ret_bps_mean": bdf["spread_ret_bps"].mean(),
        "cost_bps_mean": bdf["cost_bps"].mean(),
        "net_bps_mean": bdf["net_bps"].mean(),
        "long_turnover_mean": bdf["long_turnover"].mean(),
        "short_turnover_mean": bdf["short_turnover"].mean(),
        "df": bdf,
    }


def main():
    panel = build_wide_panel()
    print(f"Multi-OOS folds...")
    folds = _multi_oos_splits(panel)
    print(f"  {len(folds)} folds")

    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    pairs = []

    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            print(f"  fold {fold['fid']}: skipped (insufficient data)")
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

        # Baseline (sharp, beta-neutral)
        r_base = portfolio_pnl_turnover_aware(
            test, yt_pred, top_frac=TOP_FRAC,
            cost_bps_per_leg=COST_PER_LEG, sample_every=HORIZON, beta_neutral=True,
        )
        # M=2 hysteresis (beta-neutral)
        r_hyst = portfolio_pnl_hysteresis_bn(
            test, yt_pred, top_k=TOP_K, exit_buffer=EXIT_BUFFER,
            cost_bps_per_leg=COST_PER_LEG, sample_every=HORIZON, beta_neutral=True,
        )

        if r_base.get("n_bars", 0) == 0 or r_hyst.get("n_bars", 0) == 0:
            print(f"  fold {fold['fid']}: empty result")
            continue

        base_df = r_base["df"][["time", "net_bps", "spread_ret_bps", "cost_bps",
                                "long_turnover", "short_turnover"]].rename(
            columns={c: f"base_{c}" for c in
                     ["net_bps", "spread_ret_bps", "cost_bps", "long_turnover", "short_turnover"]})
        hyst_df = r_hyst["df"][["time", "net_bps", "spread_ret_bps", "cost_bps",
                                "long_turnover", "short_turnover"]].rename(
            columns={c: f"hyst_{c}" for c in
                     ["net_bps", "spread_ret_bps", "cost_bps", "long_turnover", "short_turnover"]})
        merged = base_df.merge(hyst_df, on="time", how="inner")
        merged["fold"] = fold["fid"]
        pairs.append(merged)
        print(f"  fold {fold['fid']:>2}: {len(merged)} cycles  "
              f"base_net={merged['base_net_bps'].mean():+.2f} bps  "
              f"hyst_net={merged['hyst_net_bps'].mean():+.2f} bps  "
              f"({time.time()-t0:.0f}s)")

    paired = pd.concat(pairs, ignore_index=True)
    paired["delta_net"] = paired["hyst_net_bps"] - paired["base_net_bps"]
    base = paired["base_net_bps"].to_numpy()
    hyst = paired["hyst_net_bps"].to_numpy()
    delta = paired["delta_net"].to_numpy()

    sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0
    base_sh, base_lo, base_hi = block_bootstrap_ci(base, statistic=sharpe_est,
                                                     block_size=7, n_boot=2000)
    hyst_sh, hyst_lo, hyst_hi = block_bootstrap_ci(hyst, statistic=sharpe_est,
                                                     block_size=7, n_boot=2000)
    delta_sh = delta.mean() / delta.std() * np.sqrt(CYCLES_PER_YEAR) if delta.std() > 0 else 0
    t = delta.mean() / (delta.std() / np.sqrt(len(delta)))
    p = 1 - stats.norm.cdf(abs(t))

    print("\n" + "=" * 100)
    print(f"MULTI-OOS PAIRED VALIDATION (h={HORIZON} K={TOP_K} M={EXIT_BUFFER}, "
          f"{len(folds)} folds, {len(delta)} cycles, β-neutral, 4.5 bps/leg)")
    print("=" * 100)
    print(f"  Baseline (sharp K=7):    Sharpe {base_sh:+.2f}  "
          f"[{base_lo:+.2f}, {base_hi:+.2f}]   "
          f"net {base.mean():+.2f} bps/cyc   "
          f"L_turn {paired['base_long_turnover'].mean()*100:.0f}%   "
          f"S_turn {paired['base_short_turnover'].mean()*100:.0f}%")
    print(f"  Hysteresis M=2:          Sharpe {hyst_sh:+.2f}  "
          f"[{hyst_lo:+.2f}, {hyst_hi:+.2f}]   "
          f"net {hyst.mean():+.2f} bps/cyc   "
          f"L_turn {paired['hyst_long_turnover'].mean()*100:.0f}%   "
          f"S_turn {paired['hyst_short_turnover'].mean()*100:.0f}%")
    print(f"  Delta (hyst-base):       Sharpe {delta_sh:+.2f}   "
          f"net {delta.mean():+.3f} bps/cyc   "
          f"t={t:+.2f}  one-sided p={p:.4f}   "
          f"hyst-wins {(delta>0).mean()*100:.1f}% of cycles")

    paired.to_csv(OUT_DIR / "alpha_v8_hysteresis_multioos_pairs.csv", index=False)
    summary = {
        "n_cycles": len(delta), "n_folds_used": int(paired["fold"].nunique()),
        "baseline_sharpe": float(base_sh), "baseline_ci": [float(base_lo), float(base_hi)],
        "hysteresis_sharpe": float(hyst_sh), "hysteresis_ci": [float(hyst_lo), float(hyst_hi)],
        "delta_sharpe": float(delta_sh), "delta_mean_bps": float(delta.mean()),
        "delta_t_stat": float(t), "delta_p_value": float(p),
        "hyst_wins_pct": float((delta > 0).mean() * 100),
        "exit_buffer_M": EXIT_BUFFER, "cost_bps_per_leg": COST_PER_LEG,
    }
    with open(OUT_DIR / "alpha_v8_hysteresis_multioos_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
