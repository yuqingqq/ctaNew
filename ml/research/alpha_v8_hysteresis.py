"""Test hysteresis on crypto v6_clean h=48 K=7.

Hypothesis (from xyz work): hysteresis on top-K boundary cuts turnover by
~50% with minimal alpha loss → net Sharpe lift via cost reduction.

Crypto-specific question: does this transfer from xyz?
- xyz daily: turnover 45% → 26% with M=2 hysteresis, Sharpe +1.11 → +2.02
- crypto h=48: turnover currently ~47% per memory; can hysteresis halve it?
- Crypto LGBMs have very short trees (5-50 iters), models already
  parsimonious — hysteresis should be orthogonal to that

Test: K=7 fixed, sweep M ∈ {0 (sharp baseline), 1, 2, 3, 5}.
"""
from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs import (
    _train, _walk_forward_splits, _slice, block_bootstrap_ci,
)
from ml.research.alpha_v8_multihorizon import (
    build_panel_with_multi_horizon_labels, train_horizon_ensemble,
    predict_ensemble, ENSEMBLE_SEEDS, REGIME_CUTOFF, COST_BPS_PER_LEG,
    PNL_HORIZON, N_FOLDS,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

K = 7
TOP_FRAC_K7 = K / 25


# ---- portfolio with hysteresis -----------------------------------------

def portfolio_pnl_hysteresis(test: pd.DataFrame, yt: np.ndarray, *,
                              top_k: int, exit_buffer: int,
                              cost_bps_per_leg: float, sample_every: int) -> dict:
    """Same as portfolio_pnl_turnover_aware but with hysteresis.
    Names enter when rank in top_k, exit only when rank > top_k + exit_buffer."""
    cols = ["open_time", "symbol", "return_pct", "alpha_realized", "basket_fwd"]
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
        # Rank from top: 0 = highest pred (best long candidate)
        sorted_g["rank_top"] = n - 1 - sorted_g.index
        # Rank from bottom: 0 = lowest pred (best short candidate)
        sorted_g["rank_bot"] = sorted_g.index

        # Long leg: hysteresis update
        new_long = set(cur_long)
        # Exit: drop names whose rank dropped below top_k + exit_buffer
        for s in list(new_long):
            r = sorted_g[sorted_g["symbol"] == s]
            if r.empty or r["rank_top"].iloc[0] > top_k + exit_buffer - 1:
                new_long.discard(s)
        # Enter: add candidates from top_k that aren't already in
        candidates_l = sorted_g[sorted_g["rank_top"] < top_k]["symbol"].tolist()
        for s in candidates_l:
            if len(new_long) >= top_k:
                break
            new_long.add(s)
        # Trim if exceeded (rare)
        if len(new_long) > top_k:
            ranked = sorted_g[sorted_g["symbol"].isin(new_long)].sort_values("rank_top")
            new_long = set(ranked.head(top_k)["symbol"])

        # Short leg: same logic, mirrored
        new_short = set(cur_short)
        for s in list(new_short):
            r = sorted_g[sorted_g["symbol"] == s]
            if r.empty or r["rank_bot"].iloc[0] > top_k + exit_buffer - 1:
                new_short.discard(s)
        candidates_s = sorted_g[sorted_g["rank_bot"] < top_k]["symbol"].tolist()
        for s in candidates_s:
            if len(new_short) >= top_k:
                break
            new_short.add(s)
        if len(new_short) > top_k:
            ranked = sorted_g[sorted_g["symbol"].isin(new_short)].sort_values("rank_bot")
            new_short = set(ranked.head(top_k)["symbol"])

        if not new_long or not new_short:
            cur_long, cur_short = new_long, new_short
            continue

        # P&L
        long_g = sorted_g[sorted_g["symbol"].isin(new_long)]
        short_g = sorted_g[sorted_g["symbol"].isin(new_short)]
        long_ret = long_g["return_pct"].mean()
        short_ret = short_g["return_pct"].mean()
        long_alpha = long_g["alpha_realized"].mean()
        short_alpha = short_g["alpha_realized"].mean()
        spread_ret = long_ret - short_ret
        spread_alpha = long_alpha - short_alpha
        ic = sorted_g["pred"].rank().corr(sorted_g["alpha_realized"].rank())

        # Turnover (equal-weight per leg → 1/k per name)
        long_w = {s: 1.0 / top_k for s in new_long}
        short_w = {s: 1.0 / top_k for s in new_short}
        if not prev_long_w:
            long_to, short_to = 1.0, 1.0  # initial entry
        else:
            all_l = set(long_w) | set(prev_long_w)
            long_to = 0.5 * sum(abs(long_w.get(s, 0) - prev_long_w.get(s, 0)) for s in all_l)
            all_s = set(short_w) | set(prev_short_w)
            short_to = 0.5 * sum(abs(short_w.get(s, 0) - prev_short_w.get(s, 0)) for s in all_s)
        bar_cost_bps = cost_bps_per_leg * (long_to + short_to)
        net_bps = (spread_ret * 1e4) - bar_cost_bps

        bars.append({
            "time": t, "n": n, "n_long": len(new_long), "n_short": len(new_short),
            "spread_ret_bps": spread_ret * 1e4,
            "spread_alpha_bps": spread_alpha * 1e4,
            "rank_ic": ic,
            "long_turnover": long_to, "short_turnover": short_to,
            "cost_bps": bar_cost_bps, "net_bps": net_bps,
        })
        cur_long, cur_short = new_long, new_short
        prev_long_w, prev_short_w = long_w, short_w

    bdf = pd.DataFrame(bars)
    if bdf.empty:
        return {"n_bars": 0}
    return {
        "n_bars": len(bdf),
        "spread_ret_bps_mean": bdf["spread_ret_bps"].mean(),
        "spread_alpha_bps_mean": bdf["spread_alpha_bps"].mean(),
        "cost_bps_mean": bdf["cost_bps"].mean(),
        "net_bps_mean": bdf["net_bps"].mean(),
        "long_turnover_mean": bdf["long_turnover"].mean(),
        "short_turnover_mean": bdf["short_turnover"].mean(),
        "df": bdf,
    }


# ---- main ---------------------------------------------------------------

def main() -> None:
    from features_ml.cross_sectional import list_universe
    symbols = list_universe(min_days=200)
    if os.environ.get("UNIVERSE", "ORIG25") == "ORIG25":
        from live.train_v6_clean_artifact import NEW_SYMBOLS
        symbols = [s for s in symbols if s not in NEW_SYMBOLS]
    log.info("universe: %d symbols", len(symbols))

    panel = build_panel_with_multi_horizon_labels(symbols)
    folds = _walk_forward_splits(panel, n_folds=N_FOLDS,
                                  train_days=120, cal_days=20,
                                  test_days=30, embargo_days=2)
    log.info("walk-forward folds: %d", len(folds))

    M_VALUES = [0, 1, 2, 3, 5]
    cycle_records: dict[int, list] = {m: [] for m in M_VALUES}

    for fold in folds:
        train, cal, test = _slice(panel, fold)
        if len(train) < 1000 or len(test) < 100:
            continue
        log.info("fold %d: train=%d cal=%d test=%d (%s → %s)",
                 fold["fid"], len(train), len(cal), len(test),
                 fold["test_start"].date(), fold["test_end"].date())

        # Single-horizon h=48 ensemble (5 seeds)
        models = train_horizon_ensemble(
            train, cal, XS_FEATURE_COLS_V6_CLEAN,
            target_col=f"demeaned_target_h{PNL_HORIZON}",
        )
        if not models:
            log.warning("  fold %d: training failed", fold["fid"])
            continue
        X_test = test[XS_FEATURE_COLS_V6_CLEAN].values.astype(np.float32)
        preds = predict_ensemble(models, X_test)
        log.info("  trained ensemble; pred std=%.4f", preds.std())

        for M in M_VALUES:
            if M == 0:
                # Sharp boundary baseline (use existing function for parity)
                from ml.research.alpha_v4_xs import portfolio_pnl_turnover_aware
                out = portfolio_pnl_turnover_aware(
                    test, preds, top_frac=TOP_FRAC_K7,
                    cost_bps_per_leg=COST_BPS_PER_LEG,
                    sample_every=PNL_HORIZON, beta_neutral=False,
                )
            else:
                out = portfolio_pnl_hysteresis(
                    test, preds, top_k=K, exit_buffer=M,
                    cost_bps_per_leg=COST_BPS_PER_LEG,
                    sample_every=PNL_HORIZON,
                )
            if out.get("n_bars", 0) > 0:
                for _, row in out["df"].iterrows():
                    cycle_records[M].append({
                        "fold": fold["fid"],
                        "spread": row["spread_ret_bps"],
                        "alpha": row["spread_alpha_bps"],
                        "net": row["net_bps"],
                        "long_turn": row["long_turnover"],
                        "short_turn": row["short_turnover"],
                    })

    # Summarize
    log.info("\n=== HYSTERESIS SWEEP (turnover-aware, %d-fold WF, h=48 K=7) ===",
             N_FOLDS)
    log.info("  %-15s %5s %12s %12s %14s %14s %12s %16s",
             "M", "n_cyc", "spread/cyc", "net/cyc", "L_turn",
             "S_turn", "Sharpe", "95% CI")
    cycle_per_year = 365 * 24 * 60 / (PNL_HORIZON * 5)  # 4h cycles per year

    for M in M_VALUES:
        records = cycle_records[M]
        if not records:
            log.info("  M=%-12d  NO DATA", M); continue
        df = pd.DataFrame(records)
        sharpe_estimator = lambda x: x.mean() / x.std() * np.sqrt(cycle_per_year) \
                                       if x.std() > 0 else 0
        sh, lo, hi = block_bootstrap_ci(df["net"].values,
                                         statistic=sharpe_estimator,
                                         block_size=7, n_boot=2000)
        label = "M=0 (sharp baseline)" if M == 0 else f"M={M} hysteresis"
        log.info("  %-15s %5d %+10.2fbps %+10.2fbps %12.0f%% %12.0f%% %+10.2f  [%+.2f, %+.2f]",
                 label, len(df), df["spread"].mean(), df["net"].mean(),
                 100 * df["long_turn"].mean(),
                 100 * df["short_turn"].mean(),
                 sh, lo, hi)


if __name__ == "__main__":
    main()
