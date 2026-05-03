"""Cross-sectional alpha v4: rank-based long-short portfolio across N symbols.

Hypothesis: predicting alpha vs a 25-symbol basket (instead of one ref) and
trading top-quintile long / bottom-quintile short multiplies sample size and
provides built-in market neutrality without paying 2× cost per trade.

Setup:
  - Universe: 20-25 USDM perps (BTC, ETH, SOL + 22 more liquid names)
  - Features: existing kline + regime + basket-relative (features_ml/cross_sectional.py)
  - Target: alpha_s = my_fwd_s - β_s × basket_fwd
  - Pooled training across all symbols and bars
  - At inference: predict alpha_s for every (s, t), rank cross-sectionally, hold top-K long / bottom-K short

Cost model per portfolio bar:
  - Long basket: 1 × naked round-trip cost (fee + slip + spread on each leg of K names; weighted)
  - Short basket: 1 × naked round-trip cost
  - Total ~2 × 12 bps = 24 bps per held bar (assume 4h hold)

Evaluation:
  - Spread alpha = avg(top-K alpha) - avg(bottom-K alpha)
  - Net per portfolio bar = spread_alpha - 24 bps
  - Sharpe of portfolio returns
  - Per-bar IC (cross-sectional rank correlation between predictions and realized)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features_ml.cross_sectional import (
    XS_FEATURE_COLS, assemble_universe, list_universe, make_xs_alpha_labels,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ENSEMBLE_SEEDS = (42, 7, 123, 99, 314)
HORIZON = 48
REGIME_CUTOFF = 0.50  # was 0.33; multi-OOS lift +0.58 Sharpe (2026-05-03)
HOLDOUT_DAYS = 90

# Cost: 12 bps per leg RT (fee 5×2 + slip 1×2 + spread ~1×2 → ~14 bps; conservative)
NAKED_COST_BPS_PER_LEG = 12.0


def _train(X_train, y_train, X_cal, y_cal, *, seed):
    params = dict(
        objective="regression", metric="rmse", learning_rate=0.03,
        num_leaves=63, max_depth=8, min_data_in_leaf=100,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        lambda_l2=3.0, verbose=-1,
        seed=seed, feature_fraction_seed=seed, bagging_seed=seed,
        data_random_seed=seed,
    )
    dtr = lgb.Dataset(X_train, y_train, free_raw_data=False)
    dc = lgb.Dataset(X_cal, y_cal, reference=dtr, free_raw_data=False)
    return lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dc],
                     callbacks=[lgb.early_stopping(stopping_rounds=80),
                                lgb.log_evaluation(period=0)])


def _stack_xs_panel(feats_by_sym: dict, labels_by_sym: dict, *, cols: list) -> pd.DataFrame:
    """Stack per-symbol features+labels into a long-format DataFrame: (time, symbol, ...)."""
    frames = []
    for s, f in feats_by_sym.items():
        lab = labels_by_sym[s]
        df = f.join(lab, how="inner")
        df = df.dropna(subset=cols + ["demeaned_target", "return_pct", "basket_fwd"])
        df["symbol"] = s
        df = df.reset_index().rename(columns={"index": "open_time"})
        if "open_time" not in df.columns:
            df = df.reset_index()
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    if "open_time" not in panel.columns:
        # Index lost during reset_index when index was named differently
        raise RuntimeError("panel missing open_time")
    return panel


def _walk_forward_splits(panel: pd.DataFrame, *, n_folds: int = 5,
                          train_days: int = 50, cal_days: int = 10,
                          test_days: int = 20, embargo_days: float = 1.0):
    data_start = panel["open_time"].min()
    data_end = panel["open_time"].max()
    embargo = pd.Timedelta(days=embargo_days)
    cursor = data_start
    folds = []
    for fid in range(n_folds):
        train_start = cursor
        train_end = train_start + pd.Timedelta(days=train_days)
        cal_start = train_end
        cal_end = cal_start + pd.Timedelta(days=cal_days)
        test_start = cal_end + embargo
        test_end = test_start + pd.Timedelta(days=test_days)
        if test_end > data_end:
            break
        folds.append({
            "fid": fid, "train_start": train_start, "train_end": train_end,
            "cal_start": cal_start, "cal_end": cal_end,
            "test_start": test_start, "test_end": test_end, "embargo": embargo,
        })
        cursor = test_end + embargo
    return folds


def _holdout_split(panel: pd.DataFrame, *, holdout_days: int = HOLDOUT_DAYS):
    data_end = panel["open_time"].max()
    holdout_start = data_end - pd.Timedelta(days=holdout_days)
    cal_start = holdout_start - pd.Timedelta(days=11)
    cal_end = cal_start + pd.Timedelta(days=10)
    return [{
        "fid": 0, "train_start": panel["open_time"].min(), "train_end": cal_start,
        "cal_start": cal_start, "cal_end": cal_end,
        "test_start": holdout_start, "test_end": data_end,
        "embargo": pd.Timedelta(days=1),
    }]


def _slice(panel, fold):
    """Train: index < cal_start with exit_time embargoed; cal/test by date range.
    No look-ahead via exit_time check."""
    test_left = fold["test_start"] - fold["embargo"]
    test_right = fold["test_end"] + fold["embargo"]
    train = panel[panel["open_time"] < fold["cal_start"]]
    if "exit_time" in train.columns:
        overlap = (train["exit_time"] >= test_left) & (train["open_time"] < test_right)
        train = train.loc[~overlap]
    cal = panel[(panel["open_time"] >= fold["cal_start"]) & (panel["open_time"] < fold["cal_end"])]
    if "exit_time" in cal.columns:
        cal = cal[cal["exit_time"] < fold["test_start"]]
    test = panel[(panel["open_time"] >= fold["test_start"]) & (panel["open_time"] < fold["test_end"])]
    return train, cal, test


def block_bootstrap_ci(
    series: np.ndarray, *, statistic, n_boot: int = 2000,
    block_size: int = 7, seed: int = 42, ci: float = 0.95,
) -> tuple[float, float, float]:
    """Block-bootstrap CI for a statistic over a 1d series.

    Block size preserves short-range autocorrelation (default 7 cycles ≈ 1
    week at h=288). Returns (point_estimate, lo, hi) for a (ci)-coverage CI.
    """
    rng = np.random.default_rng(seed)
    n = len(series)
    if n < 2 * block_size:
        s = float(statistic(series))
        return s, np.nan, np.nan
    n_blocks = int(np.ceil(n / block_size))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_size)[None, :]).ravel()[:n]
        boots[i] = float(statistic(series[idx]))
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(statistic(series)), float(lo), float(hi)


def portfolio_pnl_turnover_aware(
    test: pd.DataFrame, yt: np.ndarray, *, top_frac: float = 0.2,
    cost_bps_per_leg: float = NAKED_COST_BPS_PER_LEG,
    sample_every: int,
    beta_neutral: bool = False,
) -> dict:
    """Turnover-aware NON-OVERLAPPING LABEL EVALUATION (not a full backtest).

    What this is:
      - Sample test bars at non-overlapping cadence (sample_every == HORIZON
        so each cycle's `return_pct` is the realized h-forward spread).
      - At each cycle: re-rank, pick top-K / bot-K, compute spread of
        h-forward returns of the held names, charge fee proportional to
        rebalance turnover.

    What this is NOT:
      - A continuous mark-to-market equity curve.
      - A compounded-return backtest.
      - A funding-charge accounting (perp funding rates not modelled here).
      - A liquidation-aware simulation (no margin, no PnL drawdown logic).
      - A maker-fill simulator (taker-rate cost only).

    Cost: at each rebalance, `cost_bps_per_leg × (long_to + short_to)` where
    turnover ∈ [0, 1] is L1-distance/2 between successive weight vectors.
    First rebalance pays full entry on both legs.

    Beta-neutral: if True, scale leg notionals so dollar-beta of long leg
    matches dollar-beta of short leg (keeping target gross ≈ 2). Scales
    clipped to [0.5, 1.5]; rebalances with degenerate β (either leg < 0.1,
    or sum < 0.3) fall back to equal-weight and are flagged in `degen_beta`.
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
    prev_long_w: dict[str, float] = {}
    prev_short_w: dict[str, float] = {}
    for t, g in df.groupby("open_time"):
        if len(g) < 5:
            continue
        n_side = max(1, int(np.floor(top_frac * len(g))))
        sorted_g = g.sort_values("pred")
        bot = sorted_g.head(n_side)
        top = sorted_g.tail(n_side)
        if beta_neutral:
            beta_L = top["beta_short_vs_bk"].mean()
            beta_S = bot["beta_short_vs_bk"].mean()
            # Guardrails: skip rebalance if either leg has near-zero or
            # negative average beta — scaling becomes unstable / negative.
            if beta_L < 0.1 or beta_S < 0.1 or (beta_L + beta_S) < 0.3:
                # Fall back to equal-weight for this rebalance; flag it.
                scale_L, scale_S = 1.0, 1.0
                degen_beta = True
            else:
                denom = beta_L + beta_S
                raw_scale_L = 2.0 * beta_S / denom
                raw_scale_S = 2.0 * beta_L / denom
                # Clip extreme tilts to [0.5, 1.5] so a single weird leg can't
                # blow up gross exposure.
                scale_L = float(np.clip(raw_scale_L, 0.5, 1.5))
                scale_S = float(np.clip(raw_scale_S, 0.5, 1.5))
                degen_beta = False
        else:
            scale_L, scale_S = 1.0, 1.0
            degen_beta = False
        long_ret = scale_L * top["return_pct"].mean()
        short_ret = scale_S * bot["return_pct"].mean()
        long_alpha = scale_L * top["alpha_realized"].mean()
        short_alpha = scale_S * bot["alpha_realized"].mean()
        spread_ret = long_ret - short_ret
        spread_alpha = long_alpha - short_alpha
        ic = g["pred"].rank().corr(g["alpha_realized"].rank())

        long_w = {s: scale_L / n_side for s in top["symbol"]}
        short_w = {s: scale_S / n_side for s in bot["symbol"]}
        if not prev_long_w:
            long_to, short_to = scale_L, scale_S  # full entry
        else:
            all_l = set(long_w) | set(prev_long_w)
            long_to = 0.5 * sum(abs(long_w.get(s, 0) - prev_long_w.get(s, 0)) for s in all_l)
            all_s = set(short_w) | set(prev_short_w)
            short_to = 0.5 * sum(abs(short_w.get(s, 0) - prev_short_w.get(s, 0)) for s in all_s)
        bar_cost_bps = cost_bps_per_leg * (long_to + short_to)
        net_bps = (spread_ret * 1e4) - bar_cost_bps
        bars.append({
            "time": t, "n": len(g), "n_side": n_side,
            "spread_ret_bps": spread_ret * 1e4,
            "spread_alpha_bps": spread_alpha * 1e4,
            "rank_ic": ic,
            "long_turnover": long_to, "short_turnover": short_to,
            "cost_bps": bar_cost_bps, "net_bps": net_bps,
            "scale_L": scale_L, "scale_S": scale_S,
            "gross_exposure": scale_L + scale_S,
            "degen_beta": int(degen_beta),
        })
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
        "spread_ret_std": bdf["spread_ret_bps"].std(),
        "rank_ic_mean": bdf["rank_ic"].mean(),
        "win_rate_net": (bdf["net_bps"] > 0).mean(),
        "long_turnover_mean": bdf["long_turnover"].mean(),
        "short_turnover_mean": bdf["short_turnover"].mean(),
        "scale_L_mean": bdf["scale_L"].mean(),
        "scale_S_mean": bdf["scale_S"].mean(),
        "scale_L_min": bdf["scale_L"].min(),
        "scale_L_max": bdf["scale_L"].max(),
        "scale_S_min": bdf["scale_S"].min(),
        "scale_S_max": bdf["scale_S"].max(),
        "gross_exposure_mean": bdf["gross_exposure"].mean(),
        "degen_beta_frac": bdf["degen_beta"].mean(),
        "sharpe_per_bar": bdf["spread_ret_bps"].mean() / bdf["spread_ret_bps"].std()
                          if bdf["spread_ret_bps"].std() > 0 else np.nan,
        "df": bdf,
    }


def _portfolio_pnl(test: pd.DataFrame, yt: np.ndarray, *, top_frac: float = 0.2,
                    cost_bps_per_leg: float = NAKED_COST_BPS_PER_LEG) -> dict:
    """Compute long-short top-K / bottom-K portfolio P&L per bar.

    Per bar t: rank predictions across symbols available; long the top `top_frac`
    of names, short the bottom `top_frac`. Equal weight within each side.
    Realized P&L per bar = avg(top return) - avg(bottom return) - 2 × cost.

    Returns aggregated stats over all bars in `test`.
    """
    df = test[["open_time", "symbol", "return_pct", "alpha_realized", "basket_fwd"]].copy()
    df["pred"] = yt
    bars = []
    bar_count_per_side = max(1, int(round(top_frac * df.groupby("open_time")["symbol"].count().mean())))
    for t, g in df.groupby("open_time"):
        if len(g) < 5: continue
        n_side = max(1, int(np.floor(top_frac * len(g))))
        sorted_g = g.sort_values("pred")
        bot = sorted_g.head(n_side)
        top = sorted_g.tail(n_side)
        long_ret = top["return_pct"].mean()
        short_ret = bot["return_pct"].mean()
        long_alpha = top["alpha_realized"].mean()
        short_alpha = bot["alpha_realized"].mean()
        spread_ret = long_ret - short_ret
        spread_alpha = long_alpha - short_alpha
        # Per-bar cross-sectional IC of predictions vs realized alpha
        ic = g["pred"].rank().corr(g["alpha_realized"].rank())
        bars.append({
            "time": t, "n": len(g), "n_side": n_side,
            "long_ret_bps": long_ret * 1e4,
            "short_ret_bps": short_ret * 1e4,
            "spread_ret_bps": spread_ret * 1e4,
            "spread_alpha_bps": spread_alpha * 1e4,
            "rank_ic": ic,
        })
    bdf = pd.DataFrame(bars)
    if bdf.empty:
        return {"n_bars": 0}
    cost_total = 2 * cost_bps_per_leg  # long leg RT + short leg RT
    bdf["net_bps"] = bdf["spread_ret_bps"] - cost_total
    return {
        "n_bars": len(bdf),
        "spread_ret_bps_mean": bdf["spread_ret_bps"].mean(),
        "spread_alpha_bps_mean": bdf["spread_alpha_bps"].mean(),
        "net_bps_mean": bdf["net_bps"].mean(),
        "spread_ret_std": bdf["spread_ret_bps"].std(),
        "rank_ic_mean": bdf["rank_ic"].mean(),
        "win_rate_net": (bdf["net_bps"] > 0).mean(),
        # Sharpe (per-bar; not annualized — bars overlap given horizon)
        "sharpe_per_bar": bdf["spread_ret_bps"].mean() / bdf["spread_ret_bps"].std() if bdf["spread_ret_bps"].std() > 0 else np.nan,
        "df": bdf,  # for further analysis
    }


def main():
    universe = list_universe(min_days=200)
    log.info("universe: %d symbols: %s", len(universe), universe)

    pkg = assemble_universe(universe, horizon=HORIZON)
    feats_by_sym = pkg["feats_by_sym"]
    basket_close = pkg["basket_close"]
    labels_by_sym = make_xs_alpha_labels(feats_by_sym, basket_close, HORIZON)

    panel = _stack_xs_panel(feats_by_sym, labels_by_sym, cols=XS_FEATURE_COLS)
    log.info("panel: %d rows, %d unique bars, %d unique symbols",
              len(panel), panel["open_time"].nunique(), panel["symbol"].nunique())
    panel = panel.dropna(subset=["autocorr_pctile_7d"])

    print("=" * 70)
    print("CROSS-SECTIONAL ALPHA v4 — top-quintile / bottom-quintile portfolio")
    print("=" * 70)

    for mode, fold_fn in [("walk-forward", lambda: _walk_forward_splits(panel)),
                           ("OOS holdout",   lambda: _holdout_split(panel))]:
        print(f"\n--- {mode} ---")
        folds = fold_fn()
        bar_summaries = []
        for fold in folds:
            train, cal, test = _slice(panel, fold)
            train_f = train[train["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
            cal_f = cal[cal["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
            # Don't filter test by regime — we want every bar's cross-section
            test_f = test
            if len(train_f) < 1000 or len(cal_f) < 200: continue

            X_train = train_f[XS_FEATURE_COLS].to_numpy()
            y_train = train_f["demeaned_target"].to_numpy()
            X_cal = cal_f[XS_FEATURE_COLS].to_numpy()
            y_cal = cal_f["demeaned_target"].to_numpy()

            models = []
            for seed in ENSEMBLE_SEEDS:
                m = _train(X_train, y_train, X_cal, y_cal, seed=seed)
                models.append(m)

            yt = np.mean([m.predict(test_f[XS_FEATURE_COLS].to_numpy(),
                                       num_iteration=m.best_iteration) for m in models], axis=0)

            result = _portfolio_pnl(test_f, yt, top_frac=0.2)
            ta = portfolio_pnl_turnover_aware(test_f, yt, top_frac=0.2, sample_every=HORIZON)
            ta_bn = portfolio_pnl_turnover_aware(test_f, yt, top_frac=0.2, sample_every=HORIZON, beta_neutral=True)
            if result.get("n_bars", 0) == 0: continue
            print(f"  fold {fold['fid']} ({fold['test_start'].date()} → {fold['test_end'].date()}):  "
                   f"n={result['n_bars']}, spread_ret={result['spread_ret_bps_mean']:+.2f}, "
                   f"alpha={result['spread_alpha_bps_mean']:+.2f}, "
                   f"net_perbar={result['net_bps_mean']:+.2f}, "
                   f"net_TA={ta['net_bps_mean']:+.2f} (to={ta['long_turnover_mean']:.2f}), "
                   f"net_TA_BN={ta_bn['net_bps_mean']:+.2f} (alpha_BN={ta_bn['spread_alpha_bps_mean']:+.2f}, ret_BN={ta_bn['spread_ret_bps_mean']:+.2f}), "
                   f"ic={result['rank_ic_mean']:+.4f}")
            bar_summaries.append({**result, "ta": ta, "ta_bn": ta_bn})

            # Save final-fold importance for the holdout
            if mode == "OOS holdout":
                gains = np.mean([m.feature_importance(importance_type="gain") for m in models], axis=0)
                share = gains / gains.sum()
                imp = pd.Series(dict(zip(XS_FEATURE_COLS, share))).sort_values(ascending=False)
                print(f"\n  v4 feature importance (OOS gain share):")
                for f, v in imp.items():
                    print(f"    {f:<28}: {v*100:5.2f}%")

        if bar_summaries:
            spread = np.mean([b["spread_ret_bps_mean"] for b in bar_summaries])
            alpha = np.mean([b["spread_alpha_bps_mean"] for b in bar_summaries])
            net = np.mean([b["net_bps_mean"] for b in bar_summaries])
            ic = np.mean([b["rank_ic_mean"] for b in bar_summaries])
            net_ta = np.mean([b["ta"]["net_bps_mean"] for b in bar_summaries])
            cost_ta = np.mean([b["ta"]["cost_bps_mean"] for b in bar_summaries])
            to_ta = np.mean([b["ta"]["long_turnover_mean"] for b in bar_summaries])
            net_ta_bn = np.mean([b["ta_bn"]["net_bps_mean"] for b in bar_summaries])
            ret_ta_bn = np.mean([b["ta_bn"]["spread_ret_bps_mean"] for b in bar_summaries])
            alpha_ta_bn = np.mean([b["ta_bn"]["spread_alpha_bps_mean"] for b in bar_summaries])
            print(f"\n  AVG across folds: spread_ret={spread:+.2f}, spread_alpha={alpha:+.2f}, "
                   f"net_perbar={net:+.2f}, net_TA={net_ta:+.2f} (cost={cost_ta:.2f}, to={to_ta:.2f}), "
                   f"net_TA_BN={net_ta_bn:+.2f} (ret={ret_ta_bn:+.2f}, alpha={alpha_ta_bn:+.2f}), "
                   f"rank_ic={ic:+.4f}")

            # Fee-tier sensitivity sweep (β-neutral execution), time-normalized.
            # Per-tier Sharpe uses pooled per-cycle NET series (cost varies
            # per cycle via turnover, so net std differs from gross std).
            cycles_per_day = 288 / HORIZON
            cycles_per_year = cycles_per_day * 365
            scale_min = min([b["ta_bn"]["scale_L_min"] for b in bar_summaries] +
                              [b["ta_bn"]["scale_S_min"] for b in bar_summaries])
            scale_max = max([b["ta_bn"]["scale_L_max"] for b in bar_summaries] +
                              [b["ta_bn"]["scale_S_max"] for b in bar_summaries])
            gross = np.mean([b["ta_bn"]["gross_exposure_mean"] for b in bar_summaries])
            degen = np.mean([b["ta_bn"]["degen_beta_frac"] for b in bar_summaries])
            # Pool per-cycle records across folds for honest tier Sharpe.
            pooled = pd.concat([b["ta_bn"]["df"] for b in bar_summaries], ignore_index=True)
            tot_to = pooled["long_turnover"] + pooled["short_turnover"]
            tot_to_mean = tot_to.mean()
            print(f"\n  β-neutral diagnostics: gross={gross:.2f}, scale∈[{scale_min:.2f},{scale_max:.2f}], "
                   f"degen_β_rebalances={degen*100:.1f}%")
            print(f"  Fee sensitivity (β-neutral, h={HORIZON}, n_cycles_pooled={len(pooled)}, "
                   f"cycles/day={cycles_per_day:.2f}, total_to/cycle_mean={tot_to_mean:.2f}):")
            print(f"  Note: net/yr% is arithmetic (bps/cyc × cyc/yr / 1e4), not compounded. "
                   f"Sharpe uses per-cycle net std × sqrt(cycles/yr).")
            print(f"    {'Tier':<32} {'fee/leg':>8} {'cost/cyc':>9} {'net/cyc':>8} {'net_std':>8} {'net/day':>9} {'net/yr%':>9} {'Sharpe':>8}")
            for tier_name, fee_per_leg in [
                ("VIP-0 taker", 12.0),
                ("VIP-1 taker", 10.0),
                ("VIP-3 taker", 6.0),
                ("maker tilt 50% on VIP-0", 7.5),
                ("VIP-3 + maker tilt", 3.0),
                ("VIP-9 maker", 1.0),
            ]:
                # Per-cycle net for THIS tier from pooled bdf
                net_series = pooled["spread_ret_bps"] - fee_per_leg * tot_to
                net_cyc = net_series.mean()
                net_std = net_series.std()
                cost_cyc = fee_per_leg * tot_to_mean
                net_day = net_cyc * cycles_per_day
                net_yr_pct = net_day * 365 / 100
                sharpe_cyc = (net_cyc / net_std) if net_std > 0 else np.nan
                sharpe_yr = sharpe_cyc * np.sqrt(cycles_per_year)
                print(f"    {tier_name:<32} {fee_per_leg:7.1f} {cost_cyc:8.2f} {net_cyc:+7.2f} {net_std:8.2f} {net_day:+8.1f} {net_yr_pct:+8.1f} {sharpe_yr:+7.2f}")

            # Bootstrap 95% CI on OOS Sharpe and net/cycle for the
            # deployment-relevant tiers. Block size 7 cycles ≈ 1 week at
            # h=288 (1.2d at h=48). Skip for WF (folds aren't pooled
            # contiguously, so block bootstrap is misleading).
            if mode == "OOS holdout":
                print(f"\n  Bootstrap 95% CI (block-bootstrap, block=7 cycles, n_boot=2000) — OOS:")
                print(f"    {'Tier':<32} {'fee/leg':>8} {'net/cyc CI':>20} {'Sharpe_yr CI':>20}")
                for tier_name, fee_per_leg in [
                    ("VIP-0 taker", 12.0),
                    ("VIP-3 taker", 6.0),
                    ("VIP-3 + maker tilt", 3.0),
                    ("VIP-9 maker", 1.0),
                ]:
                    ns = (pooled["spread_ret_bps"] - fee_per_leg * tot_to).to_numpy()
                    _, lo_n, hi_n = block_bootstrap_ci(ns, statistic=np.mean, block_size=7)
                    def _sharpe(arr):
                        s = arr.std()
                        return (arr.mean() / s * np.sqrt(cycles_per_year)) if s > 0 else 0.0
                    pt_s, lo_s, hi_s = block_bootstrap_ci(ns, statistic=_sharpe, block_size=7)
                    print(f"    {tier_name:<32} {fee_per_leg:7.1f}  [{lo_n:+6.2f}, {hi_n:+6.2f}]      [{lo_s:+5.2f}, {hi_s:+5.2f}]")


if __name__ == "__main__":
    main()
