"""Cross-sectional alpha v4 at 1d horizon (h=288).

Sister to `alpha_v4_xs.py`. Hypothesis (HANDOFF.md Option A): residual mean-
reversion at 1d is documented at 30-50 bps in academic literature (Lou & Polk
2014); 4h was too noisy / too efficient. The horizon audit
(`alpha_v4_horizon_audit.py`) confirmed mean |IC| jumps 0.034 → 0.049 and
`dom_level_vs_bk` IC more than doubles at h=288.

Key differences from h=48:
  - HORIZON = 288 (1d forward)
  - CV folds enlarged 3x: train 120d, cal 20d, test 40d (each h=288 label
    needs 1d of forward data, so a 50d fold has too few non-overlapping samples)
  - Embargo 2d (was 1d) — avoids label overlap leakage between train and test
  - Portfolio sampling: every 12 bars (1h) instead of every bar. With h=288
    on 5m bars, adjacent bars share 287/288 of their forward window — per-bar
    sampling massively double-counts cost.
  - Cost: only charged on actual position turnover between sampled bars.
    A position re-entered identically pays no cost.

Other knobs (REGIME_CUTOFF, NAKED_COST_BPS_PER_LEG, ENSEMBLE_SEEDS, top_frac)
are pinned at the v4 defaults for fair comparison.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import os

from features_ml.cross_sectional import (
    XS_FEATURE_COLS, XS_FEATURE_COLS_V5, XS_FEATURE_COLS_V5_LEAN,
    XS_FEATURE_COLS_V6, XS_FEATURE_COLS_V6_CLEAN,
    XS_FEATURE_COLS_V7, XS_FEATURE_COLS_V7_LEAN,
    XS_RANK_SOURCES, add_xs_rank_features,
    assemble_universe, list_universe, make_xs_alpha_labels,
)
from ml.research.alpha_v4_xs import portfolio_pnl_turnover_aware, block_bootstrap_ci

# Feature set toggle: FEATURE_SET ∈ {v4 (default), v5, v5_lean, v6, v7}.
FEATURE_SET = os.environ.get("FEATURE_SET", "v4").lower()
# Phase 1.4 universe trim: set TRIM_UNIVERSE=1 to apply IS-only trim at OOS.
TRIM_UNIVERSE = os.environ.get("TRIM_UNIVERSE", "0") == "1"
# Multi-OOS validation: set MULTI_OOS=1 to use expanding-window walk-forward
# with many non-overlapping 30-day test windows instead of one 90-day holdout.
MULTI_OOS = os.environ.get("MULTI_OOS", "0") == "1"
if FEATURE_SET == "v5":
    ACTIVE_FEATURE_COLS = XS_FEATURE_COLS_V5
elif FEATURE_SET == "v5_lean":
    ACTIVE_FEATURE_COLS = XS_FEATURE_COLS_V5_LEAN
elif FEATURE_SET == "v6":
    ACTIVE_FEATURE_COLS = XS_FEATURE_COLS_V6
elif FEATURE_SET == "v6_clean":
    ACTIVE_FEATURE_COLS = XS_FEATURE_COLS_V6_CLEAN
elif FEATURE_SET == "v7":
    ACTIVE_FEATURE_COLS = XS_FEATURE_COLS_V7
elif FEATURE_SET == "v7_lean":
    ACTIVE_FEATURE_COLS = XS_FEATURE_COLS_V7_LEAN
else:
    ACTIVE_FEATURE_COLS = XS_FEATURE_COLS

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ENSEMBLE_SEEDS = (42, 7, 123, 99, 314)
HORIZON = 288  # 1d forward
REGIME_CUTOFF = 0.50  # was 0.33; multi-OOS lift +0.58 Sharpe (2026-05-03)
HOLDOUT_DAYS = 90
SAMPLE_EVERY_BARS = HORIZON  # rebalance every h bars (non-overlapping). return_pct is h-fwd, so cycle return = panel's spread.
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
    frames = []
    for s, f in feats_by_sym.items():
        lab = labels_by_sym[s]
        df = f.join(lab, how="inner")
        df = df.dropna(subset=cols + ["demeaned_target", "return_pct", "basket_fwd"])
        df["symbol"] = s
        df = df.reset_index().rename(columns={"index": "open_time"})
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    if "open_time" not in panel.columns:
        raise RuntimeError("panel missing open_time")
    return panel


def _walk_forward_splits(panel, *, n_folds: int = 4,
                          train_days: int = 120, cal_days: int = 20,
                          test_days: int = 40, embargo_days: float = 2.0):
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


def _holdout_split(panel, *, holdout_days: int = HOLDOUT_DAYS):
    data_end = panel["open_time"].max()
    holdout_start = data_end - pd.Timedelta(days=holdout_days)
    cal_start = holdout_start - pd.Timedelta(days=22)
    cal_end = cal_start + pd.Timedelta(days=20)
    return [{
        "fid": 0, "train_start": panel["open_time"].min(), "train_end": cal_start,
        "cal_start": cal_start, "cal_end": cal_end,
        "test_start": holdout_start, "test_end": data_end,
        "embargo": pd.Timedelta(days=2),
    }]


def _multi_oos_splits(panel, *, min_train_days: int = 60, cal_days: int = 20,
                       test_days: int = 30, embargo_days: float = 2.0):
    """Expanding-window walk-forward producing many non-overlapping OOS windows.

    Train is ANCHORED to data_start (expanding). Cal is the cal_days bars
    before the embargo. Test is test_days starting after embargo.
    Successive folds' test windows are non-overlapping (with embargo gap).

    For h=288 with 400 days of data, this produces ~10 folds × 30 days =
    ~300 cycles of OOS evaluation, vs the single-holdout 90 cycles.

    Anti-leakage: train uses everything before cal_start, then _slice purges
    rows whose exit_time spills into [test_left, test_right). Same protocol
    as the existing single OOS holdout.
    """
    data_start = panel["open_time"].min()
    data_end = panel["open_time"].max()
    embargo = pd.Timedelta(days=embargo_days)
    # Earliest test_start needs min_train_days + cal_days + embargo of data behind
    earliest_test_start = data_start + pd.Timedelta(days=min_train_days + cal_days) + embargo
    test_start = earliest_test_start
    folds = []
    fid = 0
    while test_start + pd.Timedelta(days=test_days) <= data_end:
        cal_end = test_start - embargo
        cal_start = cal_end - pd.Timedelta(days=cal_days)
        train_start = data_start  # anchored expanding train
        train_end = cal_start
        test_end = test_start + pd.Timedelta(days=test_days)
        folds.append({
            "fid": fid, "train_start": train_start, "train_end": train_end,
            "cal_start": cal_start, "cal_end": cal_end,
            "test_start": test_start, "test_end": test_end, "embargo": embargo,
        })
        test_start = test_end + embargo
        fid += 1
    return folds


def _slice(panel, fold):
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




def main():
    universe = list_universe(min_days=200)
    log.info("universe: %d symbols", len(universe))

    pkg = assemble_universe(universe, horizon=HORIZON)
    feats_by_sym = pkg["feats_by_sym"]
    basket_close = pkg["basket_close"]
    labels_by_sym = make_xs_alpha_labels(feats_by_sym, basket_close, HORIZON)

    # If feature set includes xs_rank features, compute them post-stack
    # (they need cross-sectional info, not available per-symbol).
    rank_cols = [c for c in ACTIVE_FEATURE_COLS if c.endswith("_xs_rank")]
    src_cols_for_stack = [c for c in ACTIVE_FEATURE_COLS if not c.endswith("_xs_rank")]
    # Need source features available for ranking even if not in final cols
    source_features = list({s for s, d in XS_RANK_SOURCES.items() if d in rank_cols})
    stack_cols = list(set(src_cols_for_stack + source_features))
    panel = _stack_xs_panel(feats_by_sym, labels_by_sym, cols=stack_cols)
    if rank_cols:
        panel = add_xs_rank_features(panel,
            sources={s: d for s, d in XS_RANK_SOURCES.items() if d in rank_cols})
        # Now drop rows where rank features are NaN (universe too small at edge)
        panel = panel.dropna(subset=rank_cols)
    log.info("panel: %d rows, %d unique bars, %d unique symbols (feature_set=%s, n_features=%d)",
              len(panel), panel["open_time"].nunique(), panel["symbol"].nunique(),
              FEATURE_SET, len(ACTIVE_FEATURE_COLS))
    panel = panel.dropna(subset=["autocorr_pctile_7d"])

    print("=" * 80)
    print(f"CROSS-SECTIONAL ALPHA v4 at h={HORIZON} (1d) — top-quintile L/S")
    print(f"Portfolio sampling every {SAMPLE_EVERY_BARS} bars; turnover-aware cost")
    print("=" * 80)

    if MULTI_OOS:
        # Skip walk-forward; use expanding-window multi-OOS as the headline.
        modes = [("OOS holdout (multi-window)",
                   lambda: _multi_oos_splits(panel))]
    else:
        modes = [("walk-forward", lambda: _walk_forward_splits(panel)),
                 ("OOS holdout", lambda: _holdout_split(panel))]
    for mode, fold_fn in modes:
        print(f"\n--- {mode} ---")
        folds = fold_fn()
        bar_summaries = []
        for fold in folds:
            train, cal, test = _slice(panel, fold)
            train_f = train[train["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
            cal_f = cal[cal["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
            test_f = test
            if len(train_f) < 1000 or len(cal_f) < 200:
                log.info("  fold %d skipped: train=%d, cal=%d", fold["fid"], len(train_f), len(cal_f))
                continue

            X_train = train_f[ACTIVE_FEATURE_COLS].to_numpy()
            y_train = train_f["demeaned_target"].to_numpy()
            X_cal = cal_f[ACTIVE_FEATURE_COLS].to_numpy()
            y_cal = cal_f["demeaned_target"].to_numpy()

            models = []
            for seed in ENSEMBLE_SEEDS:
                m = _train(X_train, y_train, X_cal, y_cal, seed=seed)
                models.append(m)

            yt = np.mean([m.predict(test_f[ACTIVE_FEATURE_COLS].to_numpy(),
                                       num_iteration=m.best_iteration) for m in models], axis=0)

            result = portfolio_pnl_turnover_aware(test_f, yt, top_frac=0.2,
                                                   sample_every=SAMPLE_EVERY_BARS)
            result_bn = portfolio_pnl_turnover_aware(test_f, yt, top_frac=0.2,
                                                     sample_every=SAMPLE_EVERY_BARS,
                                                     beta_neutral=True)
            if result.get("n_bars", 0) == 0:
                continue
            print(f"  fold {fold['fid']} ({fold['test_start'].date()} → {fold['test_end'].date()}):  "
                   f"n={result['n_bars']}, spread_ret={result['spread_ret_bps_mean']:+.2f}, "
                   f"alpha={result['spread_alpha_bps_mean']:+.2f}, "
                   f"cost={result['cost_bps_mean']:.2f}, "
                   f"net={result['net_bps_mean']:+.2f}, "
                   f"net_BN={result_bn['net_bps_mean']:+.2f} (ret_BN={result_bn['spread_ret_bps_mean']:+.2f}, alpha_BN={result_bn['spread_alpha_bps_mean']:+.2f}), "
                   f"ic={result['rank_ic_mean']:+.4f}, to={result['long_turnover_mean']:.2f}")
            # Save predictions + test frame for top-K sweep at end of OOS branch.
            bar_summaries.append({**result, "bn": result_bn,
                                   "_test_f": test_f, "_yt": yt})

            if mode.startswith("OOS holdout"):
                gains = np.mean([m.feature_importance(importance_type="gain") for m in models], axis=0)
                share = gains / gains.sum()
                imp = pd.Series(dict(zip(ACTIVE_FEATURE_COLS, share))).sort_values(ascending=False)
                print(f"\n  v4-1d feature importance (OOS gain share):")
                for f, v in imp.items():
                    print(f"    {f:<28}: {v*100:5.2f}%")

        if bar_summaries:
            spread = np.mean([b["spread_ret_bps_mean"] for b in bar_summaries])
            alpha = np.mean([b["spread_alpha_bps_mean"] for b in bar_summaries])
            cost = np.mean([b["cost_bps_mean"] for b in bar_summaries])
            net = np.mean([b["net_bps_mean"] for b in bar_summaries])
            ic = np.mean([b["rank_ic_mean"] for b in bar_summaries])
            l_to = np.mean([b["long_turnover_mean"] for b in bar_summaries])
            net_bn = np.mean([b["bn"]["net_bps_mean"] for b in bar_summaries])
            ret_bn = np.mean([b["bn"]["spread_ret_bps_mean"] for b in bar_summaries])
            alpha_bn = np.mean([b["bn"]["spread_alpha_bps_mean"] for b in bar_summaries])
            print(f"\n  AVG across folds: spread_ret={spread:+.2f}, spread_alpha={alpha:+.2f}, "
                   f"cost={cost:.2f}, net={net:+.2f}, "
                   f"net_BN={net_bn:+.2f} (ret_BN={ret_bn:+.2f}, alpha_BN={alpha_bn:+.2f}), "
                   f"rank_ic={ic:+.4f}, to={l_to:.2f}")

            # Fee-tier sensitivity sweep (β-neutral, time-normalized).
            # Per-tier Sharpe uses pooled per-cycle NET series (cost varies
            # per cycle via turnover, so net std differs from gross std).
            cycles_per_day = 288 / HORIZON
            cycles_per_year = cycles_per_day * 365
            gross = np.mean([b["bn"]["gross_exposure_mean"] for b in bar_summaries])
            scale_min = min([b["bn"]["scale_L_min"] for b in bar_summaries] +
                              [b["bn"]["scale_S_min"] for b in bar_summaries])
            scale_max = max([b["bn"]["scale_L_max"] for b in bar_summaries] +
                              [b["bn"]["scale_S_max"] for b in bar_summaries])
            degen = np.mean([b["bn"]["degen_beta_frac"] for b in bar_summaries])
            pooled = pd.concat([b["bn"]["df"] for b in bar_summaries], ignore_index=True)
            tot_to = pooled["long_turnover"] + pooled["short_turnover"]
            tot_to_mean = tot_to.mean()
            print(f"\n  β-neutral diagnostics: gross={gross:.2f}, scale∈[{scale_min:.2f},{scale_max:.2f}], "
                   f"degen_β_rebalances={degen*100:.1f}%")
            print(f"  Fee sensitivity (β-neutral, h={HORIZON}, n_cycles_pooled={len(pooled)}, "
                   f"cycles/day={cycles_per_day:.2f}, total_to/cycle_mean={tot_to_mean:.2f}):")
            print(f"  Note: net/yr% is arithmetic (bps/cyc × cyc/yr / 1e4), not compounded. "
                   f"Sharpe uses per-cycle net std × sqrt(cycles/yr).")
            print(f"    {'Tier':<32} {'fee/leg':>8} {'cost/cyc':>9} {'net/cyc':>8} {'net_std':>8} {'net/day':>9} {'net/yr%':>9} {'Sharpe':>8}")
            # Tiers below: ONE-WAY taker fee per trade (HL VIP-0 baseline = 4.5).
            for tier_name, fee_per_leg in [
                ("HL VIP-0 taker (4.5)", 4.5),
                ("HYPE Bronze taker (-10%)", 4.05),
                ("Bronze + referral (-14%)", 3.87),
                ("HYPE Silver taker (-20%)", 3.6),
                ("HYPE Gold taker (-30%)", 3.15),
                ("HL VIP-0 maker (1.5)", 1.5),
                ("HYPE Diamond maker (~0.75)", 0.75),
            ]:
                net_series = pooled["spread_ret_bps"] - fee_per_leg * tot_to
                net_cyc = net_series.mean()
                net_std = net_series.std()
                cost_cyc = fee_per_leg * tot_to_mean
                net_day = net_cyc * cycles_per_day
                net_yr_pct = net_day * 365 / 100
                sharpe_cyc = (net_cyc / net_std) if net_std > 0 else np.nan
                sharpe_yr = sharpe_cyc * np.sqrt(cycles_per_year)
                print(f"    {tier_name:<32} {fee_per_leg:7.1f} {cost_cyc:8.2f} {net_cyc:+7.2f} {net_std:8.2f} {net_day:+8.1f} {net_yr_pct:+8.1f} {sharpe_yr:+7.2f}")

            # Bootstrap 95% CI on OOS Sharpe and net/cycle for deployment tiers.
            # block=7 cycles ≈ 1 week at h=288 — preserves short-range autocorr.
            if mode.startswith("OOS holdout"):
                print(f"\n  Bootstrap 95% CI (block-bootstrap, block=7 cycles, n_boot=2000) — OOS:")
                print(f"    {'Tier':<32} {'fee/leg':>8} {'net/cyc CI':>20} {'Sharpe_yr CI':>20}")
                for tier_name, fee_per_leg in [
                    ("HL VIP-0 taker (4.5)", 4.5),
                    ("HYPE Silver taker (3.6)", 3.6),
                    ("HL VIP-0 maker (1.5)", 1.5),
                    ("HYPE Diamond maker (0.75)", 0.75),
                ]:
                    ns = (pooled["spread_ret_bps"] - fee_per_leg * tot_to).to_numpy()
                    _, lo_n, hi_n = block_bootstrap_ci(ns, statistic=np.mean, block_size=7)
                    def _sharpe(arr, cpy=cycles_per_year):
                        s = arr.std()
                        return (arr.mean() / s * np.sqrt(cpy)) if s > 0 else 0.0
                    pt_s, lo_s, hi_s = block_bootstrap_ci(ns, statistic=_sharpe, block_size=7)
                    print(f"    {tier_name:<32} {fee_per_leg:7.1f}  [{lo_n:+6.2f}, {hi_n:+6.2f}]      [{lo_s:+5.2f}, {hi_s:+5.2f}]")

                # Phase 1.1: Top-K reduction sweep. Reuse the same predictions
                # and test_f, just vary top_frac. Compares per-cycle alpha,
                # net at VIP-3+maker, and Sharpe (with bootstrap CI) for K∈{1,2,3,5}.
                # Hypothesis: diagnostic Section D shows top-1 long-only alpha
                # is +21.7 bps vs top-5 +7.8 — concentrating may extract more.
                print(f"\n  Phase 1.1: Top-K sweep (β-neutral, h={HORIZON}, OOS):")
                print(f"    {'K':>3} {'top_frac':>9} {'spread_ret':>11} {'alpha':>8} {'cost':>7} {'net@3bps':>10} {'Sharpe_yr':>10} {'Sharpe CI':>20}")
                for top_frac in [0.04, 0.08, 0.12, 0.20, 0.28, 0.36, 0.48]:
                    pooled_k_dfs = []
                    for b in bar_summaries:
                        r = portfolio_pnl_turnover_aware(
                            b["_test_f"], b["_yt"],
                            top_frac=top_frac, sample_every=SAMPLE_EVERY_BARS,
                            beta_neutral=True)
                        if r.get("n_bars", 0) > 0:
                            pooled_k_dfs.append(r["df"])
                    if not pooled_k_dfs:
                        continue
                    pk = pd.concat(pooled_k_dfs, ignore_index=True)
                    n_k = max(1, int(np.floor(top_frac * 25)))
                    tot_to_k = pk["long_turnover"] + pk["short_turnover"]
                    spread_k = pk["spread_ret_bps"].mean()
                    alpha_k = pk["spread_alpha_bps"].mean()
                    # Net at VIP-3+maker (3 bps/leg RT)
                    fee = 3.0
                    cost_k = fee * tot_to_k.mean()
                    net_series_k = pk["spread_ret_bps"] - fee * tot_to_k
                    net_k = net_series_k.mean()
                    sharpe_k = (net_k / net_series_k.std()) * np.sqrt(cycles_per_year) \
                                if net_series_k.std() > 0 else np.nan
                    ns_k = net_series_k.to_numpy()
                    def _sharpe_yr(arr, cpy=cycles_per_year):
                        s = arr.std()
                        return (arr.mean() / s * np.sqrt(cpy)) if s > 0 else 0.0
                    _, lo_sk, hi_sk = block_bootstrap_ci(ns_k, statistic=_sharpe_yr, block_size=7)
                    print(f"    {n_k:>3} {top_frac:>9.2f} {spread_k:>+11.2f} {alpha_k:>+8.2f} "
                           f"{cost_k:>7.2f} {net_k:>+10.2f} {sharpe_k:>+10.2f} "
                           f"[{lo_sk:+5.2f}, {hi_sk:+5.2f}]")

                # Phase 1.4: In-sample universe trim. The trim rule is defined
                # on TRAINING+CAL predictions only (no OOS peek):
                #   keep symbols whose IS Spearman IC of pred vs alpha_realized
                #   is > 0.
                # Then re-run portfolio over the trimmed universe at OOS only.
                if TRIM_UNIVERSE and len(bar_summaries) >= 1:
                    b = bar_summaries[0]
                    test_f_full = b["_test_f"]
                    yt_full = b["_yt"]
                    # Get train + cal IC for trim rules
                    yt_train = np.mean([m.predict(train_f[ACTIVE_FEATURE_COLS].to_numpy(),
                                                    num_iteration=m.best_iteration)
                                          for m in models], axis=0)
                    yt_cal = np.mean([m.predict(cal_f[ACTIVE_FEATURE_COLS].to_numpy(),
                                                  num_iteration=m.best_iteration)
                                        for m in models], axis=0)
                    train_pred = train_f.assign(pred=yt_train)
                    cal_pred = cal_f.assign(pred=yt_cal)
                    train_ic = train_pred.groupby("symbol").apply(
                        lambda g: g["pred"].rank().corr(g["alpha_realized"].rank())
                                  if len(g) >= 200 else np.nan
                    ).dropna().sort_values(ascending=False)
                    cal_ic = cal_pred.groupby("symbol").apply(
                        lambda g: g["pred"].rank().corr(g["alpha_realized"].rank())
                                  if len(g) >= 30 else np.nan
                    ).dropna().sort_values(ascending=False)

                    n_drop = max(1, int(round(0.25 * len(train_ic))))
                    rng = np.random.default_rng(42)

                    rules = {}
                    rules["A: drop bot-quartile by TRAIN IC"] = train_ic.tail(n_drop).index.tolist()
                    rules["B: random drop (control)"] = list(rng.choice(
                        train_ic.index.tolist(), size=n_drop, replace=False))
                    rules["C: drop TOP-quartile by TRAIN IC (inverse of A)"] = train_ic.head(n_drop).index.tolist()
                    rules["D: drop bot-quartile by CAL IC"] = cal_ic.tail(n_drop).index.tolist()

                    print(f"\n  Phase 1.4: Universe trim robustness check (kept {len(train_ic) - n_drop}/25, K=7≈40%):")
                    print(f"    {'Rule':<48} {'net@3bps':>10} {'Sharpe':>8} {'Sharpe CI':>20} {'dropped':<60}")
                    for rule_name, drop_syms in rules.items():
                        keep_syms = [s for s in train_ic.index if s not in drop_syms]
                        test_trim = test_f_full[test_f_full["symbol"].isin(keep_syms)]
                        yt_trim = yt_full[test_f_full["symbol"].isin(keep_syms).to_numpy()]
                        # K=7 (≈40% of 19 = ~7-8)
                        top_frac_eq = 7 / max(1, len(keep_syms))
                        r = portfolio_pnl_turnover_aware(
                            test_trim, yt_trim,
                            top_frac=top_frac_eq, sample_every=SAMPLE_EVERY_BARS,
                            beta_neutral=True)
                        if r.get("n_bars", 0) == 0:
                            continue
                        pk = r["df"]
                        tot_to_k = pk["long_turnover"] + pk["short_turnover"]
                        fee = 3.0
                        net_series_k = pk["spread_ret_bps"] - fee * tot_to_k
                        net_k = net_series_k.mean()
                        sharpe_k = (net_k / net_series_k.std()) * np.sqrt(cycles_per_year) \
                                    if net_series_k.std() > 0 else np.nan
                        ns_k = net_series_k.to_numpy()
                        def _sharpe_yr3(arr, cpy=cycles_per_year):
                            s = arr.std()
                            return (arr.mean() / s * np.sqrt(cpy)) if s > 0 else 0.0
                        _, lo_sk, hi_sk = block_bootstrap_ci(ns_k, statistic=_sharpe_yr3, block_size=7)
                        print(f"    {rule_name:<48} {net_k:>+10.2f} {sharpe_k:>+8.2f} "
                               f"[{lo_sk:+5.2f}, {hi_sk:+5.2f}]   {','.join(sorted(drop_syms))}")
                    # Also baseline (no trim) for comparison
                    test_full = test_f_full
                    yt_full_arr = yt_full
                    r0 = portfolio_pnl_turnover_aware(
                        test_full, yt_full_arr, top_frac=7/25,
                        sample_every=SAMPLE_EVERY_BARS, beta_neutral=True)
                    pk0 = r0["df"]
                    tot_to_0 = pk0["long_turnover"] + pk0["short_turnover"]
                    ns_0 = (pk0["spread_ret_bps"] - 3.0 * tot_to_0).to_numpy()
                    net_0 = ns_0.mean()
                    sharpe_0 = (net_0 / ns_0.std()) * np.sqrt(cycles_per_year) if ns_0.std() > 0 else np.nan
                    def _sh(arr, cpy=cycles_per_year):
                        s = arr.std()
                        return (arr.mean() / s * np.sqrt(cpy)) if s > 0 else 0.0
                    _, lo0, hi0 = block_bootstrap_ci(ns_0, statistic=_sh, block_size=7)
                    print(f"    {'BASELINE: no trim, K=7/25':<48} {net_0:>+10.2f} {sharpe_0:>+8.2f} "
                           f"[{lo0:+5.2f}, {hi0:+5.2f}]")


if __name__ == "__main__":
    main()
