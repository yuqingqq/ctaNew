"""Cycle-level vol-target gross exposure test.

Scale portfolio gross exposure by inverse predicted basket realized vol:

    scale_t = clip(target_vol / predicted_vol_t, clip_lo, clip_hi)

where:
  predicted_vol_t = realized vol of basket over trailing 24h (288 5min bars)
  target_vol      = trailing 252-cycle median of predicted_vol (PIT)

Both long and short weights scaled symmetrically: long_w' = scale × (1/K),
short_w' = scale × (1/K). Cost scales with effective notional: cost =
4.5 × Σ|Δw'| where Δw' = scale_t × w_t - scale_{t-1} × w_{t-1}.

Hypothesis: per-cycle return variance is dominated by gross moves, and
basket vol has persistence → predictable. If μ_per_cycle is approximately
constant across vol regimes, vol-targeting homogenizes σ_per_cycle and
lifts Sharpe via denominator reduction without touching the alpha numerator.

Different from:
  - per-name vol-target (memory: -1.19 ΔSharpe). That sized INDIVIDUAL
    names; high-vol names HAVE more alpha so down-weighting them hurts.
  - hi_vol gate (failed validation). That was a binary skip; this is
    continuous sizing of gross exposure.
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
from ml.research.alpha_v9_risk_overlay import build_panel_with_market_state

HORIZON = 48
TOP_K = 7
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
GATE_LOOKBACK = 252
GATE_PCTILE = 0.30
VOL_TARGET_LOOKBACK = 252
OUT_DIR = REPO / "outputs/h48_vol_target"
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


def evaluate_with_vol_target(
    test: pd.DataFrame, yt_pred: np.ndarray, *,
    use_conv_gate: bool = True,
    use_vol_target: bool = False,
    vol_target_lookback: int = VOL_TARGET_LOOKBACK,
    vol_target_pctile: float = 0.50,  # median as target
    clip_lo: float = 0.5,
    clip_hi: float = 1.5,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    cols = ["open_time", "symbol", "return_pct", "alpha_realized",
            "basket_fwd", "beta_short_vs_bk", "bk_vol_24h"]
    df = test[cols].copy()
    df["pred"] = yt_pred
    times = sorted(df["open_time"].unique())
    keep_times = set(times[::HORIZON])
    df = df[df["open_time"].isin(keep_times)]

    bars = []
    prev_long_w: dict[str, float] = {}
    prev_short_w: dict[str, float] = {}
    conv_history: deque = deque(maxlen=GATE_LOOKBACK)
    vol_history: deque = deque(maxlen=vol_target_lookback)

    for t, g in df.groupby("open_time"):
        if len(g) < 2 * top_k + 1:
            continue
        sorted_g = g.sort_values("pred")
        bot = sorted_g.head(top_k)
        top = sorted_g.tail(top_k)
        dispersion = top["pred"].mean() - bot["pred"].mean()
        bk_vol_now = g["bk_vol_24h"].iloc[0]

        # Conv gate
        skip = False
        if use_conv_gate and len(conv_history) >= 30:
            thr = np.quantile(list(conv_history), GATE_PCTILE)
            if dispersion < thr:
                skip = True
        conv_history.append(dispersion)

        if skip:
            bars.append({"time": t, "spread_ret_bps": 0.0, "long_turnover": 0.0,
                          "short_turnover": 0.0, "cost_bps": 0.0, "net_bps": 0.0,
                          "skipped": 1, "vol_pred": bk_vol_now, "scale": 0.0,
                          "gross_exp": 0.0})
            if not pd.isna(bk_vol_now):
                vol_history.append(bk_vol_now)
            continue

        # Vol target scale
        vol_scale = 1.0
        if use_vol_target and not pd.isna(bk_vol_now) and len(vol_history) >= 30:
            target_vol = np.quantile(list(vol_history), vol_target_pctile)
            raw = target_vol / max(bk_vol_now, 1e-8)
            vol_scale = float(np.clip(raw, clip_lo, clip_hi))
        if not pd.isna(bk_vol_now):
            vol_history.append(bk_vol_now)

        # β-neutral leg scaling
        scale_L, scale_S = _bn_scales(top, bot)
        # Apply vol target on top of β-neutral
        scale_L_eff = scale_L * vol_scale
        scale_S_eff = scale_S * vol_scale

        n_l = len(top); n_s = len(bot)
        long_w = {s: scale_L_eff / n_l for s in top["symbol"]}
        short_w = {s: scale_S_eff / n_s for s in bot["symbol"]}
        # Returns scale with effective leg notional
        long_ret = scale_L_eff * top["return_pct"].mean()
        short_ret = scale_S_eff * bot["return_pct"].mean()
        spread_ret = long_ret - short_ret

        if not prev_long_w:
            long_to, short_to = scale_L_eff, scale_S_eff
        else:
            all_l = set(long_w) | set(prev_long_w)
            long_to = sum(abs(long_w.get(s, 0) - prev_long_w.get(s, 0)) for s in all_l)
            all_s = set(short_w) | set(prev_short_w)
            short_to = sum(abs(short_w.get(s, 0) - prev_short_w.get(s, 0)) for s in all_s)
        bar_cost_bps = COST_PER_LEG * (long_to + short_to)
        net_bps = (spread_ret * 1e4) - bar_cost_bps

        bars.append({"time": t, "spread_ret_bps": spread_ret * 1e4,
                      "long_turnover": long_to, "short_turnover": short_to,
                      "cost_bps": bar_cost_bps, "net_bps": net_bps, "skipped": 0,
                      "vol_pred": bk_vol_now, "scale": vol_scale,
                      "gross_exp": scale_L_eff + scale_S_eff})
        prev_long_w, prev_short_w = long_w, short_w

    return pd.DataFrame(bars)


def main():
    panel = build_panel_with_market_state()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    print(f"Multi-OOS folds: {len(folds)}")

    variants = [
        ("baseline_gate (production)", False, None, None, None),
        ("vol_target_clip_05_15",       True, 0.50, 0.5, 1.5),
        ("vol_target_clip_03_20",       True, 0.50, 0.3, 2.0),
        ("vol_target_clip_07_13",       True, 0.50, 0.7, 1.3),
        ("vol_target_p25_05_15",        True, 0.25, 0.5, 1.5),  # use 25th-pctile (smaller target)
        ("vol_target_p75_05_15",        True, 0.75, 0.5, 1.5),  # use 75th-pctile (larger target)
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

        for name, use_vt, pctile, lo, hi in variants:
            df = evaluate_with_vol_target(
                test, yt_pred,
                use_conv_gate=True,
                use_vol_target=use_vt,
                vol_target_pctile=pctile if pctile else 0.5,
                clip_lo=lo if lo else 0.5,
                clip_hi=hi if hi else 1.5,
            )
            for _, r in df.iterrows():
                cycles[name].append({
                    "fold": fold["fid"], "time": r["time"],
                    "gross": r["spread_ret_bps"], "cost": r["cost_bps"],
                    "net": r["net_bps"], "long_turn": r["long_turnover"],
                    "skipped": r["skipped"], "scale": r.get("scale", 1.0),
                    "gross_exp": r.get("gross_exp", 2.0),
                    "vol_pred": r.get("vol_pred", np.nan),
                })
        print(f"  fold {fold['fid']}: {time.time() - t0:.0f}s")

    print("\n" + "=" * 130)
    print(f"VOL-TARGET GROSS EXPOSURE TEST (h={HORIZON} K={TOP_K} ORIG25, β-neutral, "
          f"{COST_PER_LEG} bps/leg, post-fix cost, conv_gate p={GATE_PCTILE})")
    print("=" * 130)
    print(f"  {'variant':<32} {'%trade':>7} {'gross':>7} {'cost':>6} {'net':>7} "
          f"{'L_turn':>7} {'mean_scale':>10} {'mean_gross_exp':>14} "
          f"{'Sharpe':>7} {'95% CI':>15} {'Δgate':>7}")

    base_recs = pd.DataFrame(cycles["baseline_gate (production)"])
    summary = {}
    for name, *_ in variants:
        df = pd.DataFrame(cycles[name])
        if df.empty: continue
        traded = df[df["skipped"] == 0]
        sh, lo_ci, hi_ci = block_bootstrap_ci(df["net"].values, statistic=sharpe_est,
                                                 block_size=7, n_boot=2000)
        m = base_recs[["fold", "time", "net"]].rename(columns={"net": "base"}).merge(
            df[["fold", "time", "net"]], on=["fold", "time"], how="inner")
        d_g = sharpe_est((m["net"] - m["base"]).to_numpy())
        print(f"  {name:<32} {100*len(traded)/len(df):>6.1f}% "
              f"{traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>5.2f}  "
              f"{df['net'].mean():>+6.2f}  "
              f"{traded['long_turn'].mean() if len(traded) > 0 else 0:>6.0%}  "
              f"{traded['scale'].mean() if len(traded) > 0 else 0:>9.3f}  "
              f"{traded['gross_exp'].mean() if len(traded) > 0 else 0:>13.3f}  "
              f"{sh:>+6.2f}  [{lo_ci:>+5.2f},{hi_ci:>+5.2f}]  {d_g:>+6.2f}")
        summary[name] = {
            "n_cycles": int(len(df)), "pct_trade": float(100*len(traded)/len(df)),
            "net": float(df["net"].mean()),
            "sharpe": float(sh), "ci": [float(lo_ci), float(hi_ci)],
            "delta_sharpe_vs_gate": float(d_g),
            "mean_scale": float(traded['scale'].mean() if len(traded) > 0 else 0),
            "mean_gross_exp": float(traded['gross_exp'].mean() if len(traded) > 0 else 0),
        }

    # Diagnostic: scale distribution and autocorrelation for clip_05_15
    print(f"\n  --- DIAGNOSTIC: scale distribution (vol_target_clip_05_15) ---")
    df = pd.DataFrame(cycles["vol_target_clip_05_15"])
    traded = df[df["skipped"] == 0]
    if len(traded) > 0:
        print(f"  scale percentiles: p10={traded['scale'].quantile(0.1):.3f}, "
              f"p50={traded['scale'].quantile(0.5):.3f}, "
              f"p90={traded['scale'].quantile(0.9):.3f}")
        print(f"  scale autocorr (lag 1): {traded['scale'].autocorr(lag=1):.3f}")
        print(f"  fraction at clip boundary: "
              f"low={(traded['scale'] <= 0.51).mean()*100:.1f}%, "
              f"high={(traded['scale'] >= 1.49).mean()*100:.1f}%")
        print(f"  net std on traded: {traded['net'].std():.2f} (vs baseline traded std)")
        base_traded = base_recs[base_recs["skipped"] == 0]
        print(f"  baseline net std on traded: {base_traded['net'].std():.2f}")

    with open(OUT_DIR / "alpha_v9_vol_target_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
