"""Funding-as-gate + time-of-day gate test.

(A) Funding gate: skip cycle when cross-universe mean(|funding rate|) is
    in top 30% of trailing 252-cycle distribution. Different mechanism
    from funding-as-feature (which already failed); tests whether
    extreme-funding REGIMES are unprofitable for cross-sectional alpha.

(B) Time-of-day gate: bin cycles by UTC start hour. Funding settles at
    00/08/16 UTC; cycles starting at those hours include settlement
    flow. Test:
      - funding-hour-only:      keep only cycles starting at {0, 8, 16}
      - non-funding-hour-only:  keep only cycles starting at {4, 12, 20}
      - per-hour diagnostic:    Sharpe by cycle start hour

All variants composed with conv_gate p=0.30 (validated production).
Multi-OOS framework, post-fix cost surface.
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
GATE_PCTILE_CONV = 0.30
GATE_PCTILE_FUNDING_HI = 0.70  # skip top 30% funding-stress
CACHE_DIR = REPO / "data/ml/cache"
OUT_DIR = REPO / "outputs/h48_funding_session"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0


def load_funding(symbol: str) -> pd.Series:
    """Load cached funding rate; index by calc_time, ffilled later."""
    p = CACHE_DIR / f"funding_{symbol}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    s = df.set_index("calc_time")["funding_rate"]
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def add_market_funding(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute market-wide |funding| series and merge to panel by open_time."""
    universe = sorted(panel["symbol"].unique())
    funding_series = {}
    for s in universe:
        f = load_funding(s)
        if f is not None:
            funding_series[s] = f
    if not funding_series:
        return panel.assign(funding_mkt_abs=np.nan)

    # Construct a wide df indexed by funding settlement times
    fdf = pd.DataFrame(funding_series)
    # Cross-universe mean of |funding rate|
    market_abs = fdf.abs().mean(axis=1)
    # Resample to 5min and forward-fill (funding is constant between settlements)
    bar_idx = pd.DatetimeIndex(sorted(panel["open_time"].unique()))
    market_abs_5min = market_abs.reindex(market_abs.index.union(bar_idx)).sort_index()
    market_abs_5min = market_abs_5min.ffill().reindex(bar_idx)
    market_df = market_abs_5min.rename("funding_mkt_abs").reset_index()
    market_df = market_df.rename(columns={"index": "open_time"})
    panel = panel.merge(market_df, on="open_time", how="left")
    return panel


def _bn_scales(top_g, bot_g):
    beta_L = top_g["beta_short_vs_bk"].mean()
    beta_S = bot_g["beta_short_vs_bk"].mean()
    if beta_L < 0.1 or beta_S < 0.1 or (beta_L + beta_S) < 0.3:
        return 1.0, 1.0
    denom = beta_L + beta_S
    return (float(np.clip(2.0 * beta_S / denom, 0.5, 1.5)),
            float(np.clip(2.0 * beta_L / denom, 0.5, 1.5)))


def evaluate_with_funding_session(
    test: pd.DataFrame, yt_pred: np.ndarray, *,
    use_conv_gate: bool = True,
    use_funding_gate: bool = False,
    funding_pctile: float = GATE_PCTILE_FUNDING_HI,
    hour_filter: set | None = None,  # only trade cycles starting at these UTC hours
    top_k: int = TOP_K,
) -> pd.DataFrame:
    cols = ["open_time", "symbol", "return_pct", "alpha_realized",
            "basket_fwd", "beta_short_vs_bk", "funding_mkt_abs"]
    df = test[cols].copy()
    df["pred"] = yt_pred
    times = sorted(df["open_time"].unique())
    keep_times = set(times[::HORIZON])
    df = df[df["open_time"].isin(keep_times)]

    bars = []
    prev_long_w: dict[str, float] = {}
    prev_short_w: dict[str, float] = {}
    conv_history: deque = deque(maxlen=GATE_LOOKBACK)
    funding_history: deque = deque(maxlen=GATE_LOOKBACK)

    for t, g in df.groupby("open_time"):
        if len(g) < 2 * top_k + 1:
            continue
        sorted_g = g.sort_values("pred")
        bot = sorted_g.head(top_k)
        top = sorted_g.tail(top_k)
        dispersion = top["pred"].mean() - bot["pred"].mean()
        funding_now = g["funding_mkt_abs"].iloc[0]
        cycle_hour = pd.Timestamp(t).hour

        skip = False
        if use_conv_gate and len(conv_history) >= 30:
            thr = np.quantile(list(conv_history), GATE_PCTILE_CONV)
            if dispersion < thr:
                skip = True
        conv_history.append(dispersion)

        if use_funding_gate and not pd.isna(funding_now) and len(funding_history) >= 30:
            thr = np.quantile(list(funding_history), funding_pctile)
            if funding_now > thr:
                skip = True
        if not pd.isna(funding_now):
            funding_history.append(funding_now)

        if hour_filter is not None and cycle_hour not in hour_filter:
            skip = True

        if skip:
            bars.append({"time": t, "spread_ret_bps": 0.0, "long_turnover": 0.0,
                          "short_turnover": 0.0, "cost_bps": 0.0, "net_bps": 0.0,
                          "skipped": 1, "hour": cycle_hour, "funding": funding_now})
            continue

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
        bar_cost_bps = COST_PER_LEG * (long_to + short_to)
        net_bps = (spread_ret * 1e4) - bar_cost_bps

        bars.append({"time": t, "spread_ret_bps": spread_ret * 1e4,
                      "long_turnover": long_to, "short_turnover": short_to,
                      "cost_bps": bar_cost_bps, "net_bps": net_bps, "skipped": 0,
                      "hour": cycle_hour, "funding": funding_now})
        prev_long_w, prev_short_w = long_w, short_w

    return pd.DataFrame(bars)


def main():
    panel = build_wide_panel()
    panel = add_market_funding(panel)
    print(f"  funding_mkt_abs non-null: {panel['funding_mkt_abs'].notna().sum():,} of {len(panel):,}")
    print(f"  funding_mkt_abs distribution: mean={panel['funding_mkt_abs'].mean()*100:.4f}%, "
          f"p90={panel['funding_mkt_abs'].quantile(0.9)*100:.4f}%")
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    print(f"Multi-OOS folds: {len(folds)}")

    variants = [
        ("sharp",                    False, False, None),
        ("conv_gate",                 True, False, None),  # production
        ("conv_gate_fund_p70",        True,  True, None),
        ("conv_gate_funding_hours",   True, False, {0, 8, 16}),
        ("conv_gate_nonfund_hours",   True, False, {4, 12, 20}),
        ("conv_gate_asia_session",    True, False, {0, 4}),
        ("conv_gate_eu_session",      True, False, {8, 12}),
        ("conv_gate_us_session",      True, False, {16, 20}),
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

        for name, conv, fund, hour_filt in variants:
            df = evaluate_with_funding_session(
                test, yt_pred,
                use_conv_gate=conv, use_funding_gate=fund, hour_filter=hour_filt,
            )
            for _, r in df.iterrows():
                cycles[name].append({
                    "fold": fold["fid"], "time": r["time"],
                    "gross": r["spread_ret_bps"], "cost": r["cost_bps"],
                    "net": r["net_bps"], "long_turn": r["long_turnover"],
                    "skipped": r["skipped"], "hour": r["hour"],
                    "funding": r["funding"],
                })
        print(f"  fold {fold['fid']}: {time.time() - t0:.0f}s")

    # Summary
    print("\n" + "=" * 110)
    print(f"FUNDING + SESSION GATES (h={HORIZON} K={TOP_K} ORIG25, β-neutral, "
          f"{COST_PER_LEG} bps/leg, post-fix cost)")
    print("=" * 110)
    print(f"  {'variant':<28} {'n_cyc':>5} {'%trade':>7} {'gross':>7} {'cost':>6} "
          f"{'net':>7} {'L_turn':>7} {'Sharpe':>7} {'95% CI':>15} {'Δgate':>7}")

    gate_recs = pd.DataFrame(cycles["conv_gate"])

    summary = {}
    for name, *_ in variants:
        df = pd.DataFrame(cycles[name])
        if df.empty: continue
        traded = df[df["skipped"] == 0]
        pct_trade = 100 * len(traded) / len(df) if len(df) > 0 else 0
        sh, lo, hi = block_bootstrap_ci(df["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)
        m_g = gate_recs[["fold", "time", "net"]].rename(columns={"net": "base_g"}).merge(
            df[["fold", "time", "net"]], on=["fold", "time"], how="inner")
        d_g = sharpe_est((m_g["net"] - m_g["base_g"]).to_numpy())
        print(f"  {name:<28} {len(df):>5d} {pct_trade:>6.1f}% "
              f"{traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>5.2f}  "
              f"{df['net'].mean():>+6.2f}  "
              f"{traded['long_turn'].mean() if len(traded) > 0 else 0:>6.0%}  "
              f"{sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  {d_g:>+6.2f}")
        summary[name] = {
            "n_cycles": int(len(df)), "pct_trade": float(pct_trade),
            "net": float(df["net"].mean()),
            "sharpe": float(sh), "ci": [float(lo), float(hi)],
            "delta_sharpe_vs_gate": float(d_g),
        }

    # Per-hour diagnostic on production conv_gate
    print("\n  --- PER-HOUR SHARPE DIAGNOSTIC (conv_gate production) ---")
    print(f"  {'hour':>5} {'cycles':>7} {'%trade':>7} {'gross':>7} {'net':>7} {'Sharpe':>7}")
    g_df = pd.DataFrame(cycles["conv_gate"])
    for hour in sorted(g_df["hour"].unique()):
        h = g_df[g_df["hour"] == hour]
        traded = h[h["skipped"] == 0]
        if h.empty: continue
        sh = sharpe_est(h["net"].to_numpy())
        print(f"  {hour:>5d} {len(h):>7d} {100*len(traded)/len(h):>6.1f}% "
              f"{traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{h['net'].mean():>+6.2f}  {sh:>+6.2f}")

    with open(OUT_DIR / "alpha_v9_funding_session_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
