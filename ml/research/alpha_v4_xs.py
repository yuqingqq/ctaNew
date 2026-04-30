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
REGIME_CUTOFF = 0.33
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
            if result.get("n_bars", 0) == 0: continue
            print(f"  fold {fold['fid']} ({fold['test_start'].date()} → {fold['test_end'].date()}):  "
                   f"n_bars={result['n_bars']}, spread_ret={result['spread_ret_bps_mean']:+.2f}, "
                   f"spread_alpha={result['spread_alpha_bps_mean']:+.2f}, "
                   f"net={result['net_bps_mean']:+.2f}, ic={result['rank_ic_mean']:+.4f}, "
                   f"win={result['win_rate_net']*100:.1f}%, sharpe={result['sharpe_per_bar']:.3f}")
            bar_summaries.append(result)

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
            print(f"\n  AVG across folds: spread_ret={spread:+.2f}, spread_alpha={alpha:+.2f}, "
                   f"net={net:+.2f}, rank_ic={ic:+.4f}")


if __name__ == "__main__":
    main()
