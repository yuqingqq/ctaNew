"""Test soft drawdown brake — reduce gross during losing streaks.

The April 2026 loss diagnostic showed:
  - Rank IC was fine (+0.068) — model wasn't broken
  - Realized spread was compressed (+4.23 bps vs typical +9-13)
  - One catastrophic cycle (-96 bps, IC -0.47) drove ~40% of the month's loss
  - Bad streak of 7 consecutive negative cycles

Prior DD brake (failed -0.68 ΔSharpe in audit) used 22-cycle lookback,
-50 bps threshold, FULL skip. It triggered too aggressively and trapped
in skip state.

Soft variants (this test):
  A. Soft brake: scale 0.5x at -30 bps, 0.25x at -100 bps (10-cyc lookback)
  B. Tight brake: scale 0.5x at -25 bps (5-cyc lookback) — fast response
  C. Spread-based brake: scale by trailing realized spread / target ratio
  D. Hybrid: soft brake + spread-based brake combined

All composed with conv_gate p=0.30 (validated production rule).
Multi-OOS framework includes the recent fold 9 (April 2026) so we can
directly measure how much loss it would have prevented.
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
GATE_PCTILE = 0.30
OUT_DIR = REPO / "outputs/h48_soft_brake"
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


def evaluate_with_soft_brake(
    test, yt_pred, *,
    use_conv_gate: bool = True,
    brake_mode: str = "none",  # 'none', 'soft', 'tight', 'spread', 'hybrid'
    pnl_lookback: int = 10,
    pnl_half_thr: float = -30.0,
    pnl_quarter_thr: float = -100.0,
    spread_lookback: int = 10,
    spread_target: float = 9.0,  # bps; below this = compressed
    spread_floor: float = 4.0,   # bps; below this = scale 0.25
    top_k: int = TOP_K,
):
    cols = ["open_time", "symbol", "return_pct", "alpha_realized",
            "basket_fwd", "beta_short_vs_bk"]
    df = test[cols].copy()
    df["pred"] = yt_pred
    times = sorted(df["open_time"].unique())
    keep_times = set(times[::HORIZON])
    df = df[df["open_time"].isin(keep_times)]

    bars = []
    prev_long_w: dict = {}
    prev_short_w: dict = {}
    conv_history = deque(maxlen=GATE_LOOKBACK)
    pnl_history = deque(maxlen=pnl_lookback)
    spread_history = deque(maxlen=spread_lookback)

    for t, g in df.groupby("open_time"):
        if len(g) < 2 * top_k + 1:
            continue
        sorted_g = g.sort_values("pred")
        bot = sorted_g.head(top_k)
        top = sorted_g.tail(top_k)
        dispersion = top["pred"].mean() - bot["pred"].mean()
        skip = False
        if use_conv_gate and len(conv_history) >= 30:
            thr = np.quantile(list(conv_history), GATE_PCTILE)
            if dispersion < thr:
                skip = True
        conv_history.append(dispersion)

        # Compute brake scale based on history
        brake_scale = 1.0
        if brake_mode != "none":
            if brake_mode in ("soft", "tight", "hybrid") and len(pnl_history) >= 3:
                recent_sum = sum(pnl_history)
                if brake_mode == "tight":
                    # Tight: fast response, half-size only
                    if recent_sum < pnl_half_thr:
                        brake_scale = min(brake_scale, 0.5)
                else:  # soft, hybrid
                    if recent_sum < pnl_quarter_thr:
                        brake_scale = min(brake_scale, 0.25)
                    elif recent_sum < pnl_half_thr:
                        brake_scale = min(brake_scale, 0.5)
            if brake_mode in ("spread", "hybrid") and len(spread_history) >= 5:
                recent_spread = np.mean(list(spread_history))
                if recent_spread < spread_floor:
                    brake_scale = min(brake_scale, 0.25)
                elif recent_spread < spread_target:
                    # Linear interp from 1.0 at target to 0.5 at floor
                    raw = 0.5 + 0.5 * (recent_spread - spread_floor) / (spread_target - spread_floor)
                    brake_scale = min(brake_scale, max(0.5, raw))

        if skip:
            bars.append({"time": t, "spread_ret_bps": 0.0, "long_turnover": 0.0,
                          "short_turnover": 0.0, "cost_bps": 0.0, "net_bps": 0.0,
                          "skipped": 1, "brake_scale": brake_scale})
            pnl_history.append(0.0)
            continue

        scale_L, scale_S = _bn_scales(top, bot)
        scale_L_eff = scale_L * brake_scale
        scale_S_eff = scale_S * brake_scale
        n_l, n_s = len(top), len(bot)
        long_w = {s: scale_L_eff / n_l for s in top["symbol"]}
        short_w = {s: scale_S_eff / n_s for s in bot["symbol"]}
        long_ret = scale_L_eff * top["return_pct"].mean()
        short_ret = scale_S_eff * bot["return_pct"].mean()
        spread_ret = long_ret - short_ret  # already scaled

        # Track unscaled spread for spread-based brake
        unscaled_spread_bps = (top["return_pct"].mean() - bot["return_pct"].mean()) * 1e4
        spread_history.append(unscaled_spread_bps)

        if not prev_long_w:
            long_to, short_to = scale_L_eff, scale_S_eff
        else:
            all_l = set(long_w) | set(prev_long_w)
            long_to = sum(abs(long_w.get(s, 0) - prev_long_w.get(s, 0)) for s in all_l)
            all_s = set(short_w) | set(prev_short_w)
            short_to = sum(abs(short_w.get(s, 0) - prev_short_w.get(s, 0)) for s in all_s)
        cost_bps = COST_PER_LEG * (long_to + short_to)
        net_bps = (spread_ret * 1e4) - cost_bps

        bars.append({"time": t, "spread_ret_bps": spread_ret * 1e4,
                      "long_turnover": long_to, "short_turnover": short_to,
                      "cost_bps": cost_bps, "net_bps": net_bps,
                      "skipped": 0, "brake_scale": brake_scale,
                      "unscaled_spread": unscaled_spread_bps})
        prev_long_w, prev_short_w = long_w, short_w
        pnl_history.append(net_bps)

    return pd.DataFrame(bars)


def main():
    panel = build_wide_panel()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    print(f"Multi-OOS folds: {len(folds)}")

    variants = [
        ("baseline_gate (production)", "none",   {}),
        ("soft_brake (10cyc -30/-100, 0.5x/0.25x)", "soft",
            {"pnl_lookback": 10, "pnl_half_thr": -30, "pnl_quarter_thr": -100}),
        ("soft_brake (10cyc -50/-150)", "soft",
            {"pnl_lookback": 10, "pnl_half_thr": -50, "pnl_quarter_thr": -150}),
        ("tight_brake (5cyc -25, 0.5x)", "tight",
            {"pnl_lookback": 5, "pnl_half_thr": -25}),
        ("tight_brake (3cyc -20, 0.5x)", "tight",
            {"pnl_lookback": 3, "pnl_half_thr": -20}),
        ("spread_brake (10cyc, target 9)", "spread",
            {"spread_lookback": 10, "spread_target": 9.0, "spread_floor": 4.0}),
        ("spread_brake (5cyc, target 9)", "spread",
            {"spread_lookback": 5, "spread_target": 9.0, "spread_floor": 4.0}),
        ("hybrid (10cyc soft + spread)", "hybrid",
            {"pnl_lookback": 10, "pnl_half_thr": -30, "pnl_quarter_thr": -100,
              "spread_lookback": 10, "spread_target": 9.0, "spread_floor": 4.0}),
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

        for name, mode, params in variants:
            df = evaluate_with_soft_brake(
                test, yt_pred, brake_mode=mode, **params,
            )
            for _, r in df.iterrows():
                cycles[name].append({
                    "fold": fold["fid"], "time": r["time"],
                    "spread": r["spread_ret_bps"], "cost": r["cost_bps"],
                    "net": r["net_bps"], "long_turn": r["long_turnover"],
                    "skipped": r["skipped"], "brake_scale": r["brake_scale"],
                })
        print(f"  fold {fold['fid']}: {time.time() - t0:.0f}s")

    print("\n" + "=" * 130)
    print(f"SOFT BRAKE SWEEP (h={HORIZON} K={TOP_K}, β-neutral, "
          f"{COST_PER_LEG} bps/leg, post-fix cost, conv_gate p={GATE_PCTILE})")
    print("=" * 130)
    print(f"  {'variant':<42} {'n_cyc':>5} {'%trade':>7} {'mean_scale':>10} {'gross':>7} "
          f"{'cost':>6} {'net':>7} {'Sharpe':>7} {'95% CI':>15} {'Δgate':>7}")

    base_recs = pd.DataFrame(cycles["baseline_gate (production)"])
    summary = {}
    fold9_summary = {}

    for name, *_ in variants:
        df = pd.DataFrame(cycles[name])
        if df.empty: continue
        traded = df[df["skipped"] == 0]
        sh, lo, hi = block_bootstrap_ci(df["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)
        m = base_recs[["fold", "time", "net"]].rename(columns={"net": "base"}).merge(
            df[["fold", "time", "net"]], on=["fold", "time"], how="inner")
        d_g = sharpe_est((m["net"] - m["base"]).to_numpy())
        print(f"  {name:<42} {len(df):>5d} {100*len(traded)/len(df):>6.1f}% "
              f"{traded['brake_scale'].mean() if len(traded) > 0 else 0:>9.3f}  "
              f"{traded['spread'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>5.2f}  "
              f"{df['net'].mean():>+6.2f}  "
              f"{sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  {d_g:>+6.2f}")
        summary[name] = {
            "n_cycles": int(len(df)), "pct_trade": float(100*len(traded)/len(df)),
            "mean_scale": float(traded['brake_scale'].mean() if len(traded) > 0 else 0),
            "net": float(df["net"].mean()),
            "sharpe": float(sh), "ci": [float(lo), float(hi)],
            "delta_sharpe_vs_gate": float(d_g),
        }
        # Per-fold breakdown for fold 9 specifically
        f9 = df[df["fold"] == 9]
        if len(f9) > 0:
            fold9_summary[name] = {
                "fold9_net": float(f9["net"].mean()),
                "fold9_sharpe": float(sharpe_est(f9["net"].values)),
                "fold9_skip_pct": float(100 * f9["skipped"].mean()),
                "fold9_brake_mean": float(f9[f9["skipped"]==0]["brake_scale"].mean()
                                            if (f9["skipped"]==0).any() else 0),
            }

    print(f"\n  --- FOLD 9 (April 2026) breakdown — does the brake help here? ---")
    print(f"  {'variant':<42} {'fold9_net':>10} {'fold9_Sh':>9} {'%skip':>7} {'brake':>7}")
    for name in [v[0] for v in variants]:
        if name not in fold9_summary: continue
        r = fold9_summary[name]
        print(f"  {name:<42} {r['fold9_net']:>+9.2f}  {r['fold9_sharpe']:>+8.2f}  "
              f"{r['fold9_skip_pct']:>6.1f}%  {r['fold9_brake_mean']:>6.3f}")

    print(f"\n  --- FOLD 6 (Dec 2025) breakdown — another bad fold ---")
    print(f"  {'variant':<42} {'fold6_net':>10} {'fold6_Sh':>9}")
    for name in [v[0] for v in variants]:
        df = pd.DataFrame(cycles[name])
        f6 = df[df["fold"] == 6]
        if len(f6) > 0:
            print(f"  {name:<42} {f6['net'].mean():>+9.2f}  "
                  f"{sharpe_est(f6['net'].values):>+8.2f}")

    print(f"\n  --- FOLD 3 (Sep 2025) breakdown — another bad fold ---")
    print(f"  {'variant':<42} {'fold3_net':>10} {'fold3_Sh':>9}")
    for name in [v[0] for v in variants]:
        df = pd.DataFrame(cycles[name])
        f3 = df[df["fold"] == 3]
        if len(f3) > 0:
            print(f"  {name:<42} {f3['net'].mean():>+9.2f}  "
                  f"{sharpe_est(f3['net'].values):>+8.2f}")

    with open(OUT_DIR / "alpha_v9_soft_brake_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
