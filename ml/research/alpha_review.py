"""Forensic review: decompose alpha-residual strategy PnL into:
    1. Alpha capture (predicted alpha vs realized alpha)
    2. Market exposure (passive beta carry on the position direction)
    3. Costs

For each OOS trade, compute:
    side[t]            ∈ {+1, -1}  from sign(predicted_alpha)
    my_fwd[t]          realized symbol return
    ref_fwd[t]         realized reference (BTC/ETH) return
    beta[t]            point-in-time rolling beta
    alpha_realized[t]  = my_fwd - beta * ref_fwd
    market_term[t]     = beta * ref_fwd
    gross[t]           = side * my_fwd          (what we trade)
    alpha_pnl[t]       = side * alpha_realized  (alpha capture)
    market_pnl[t]      = side * beta * ref_fwd  (passive market via long/short ETH)

Report per symbol:
    - mean alpha_pnl   (true alpha capture)
    - mean market_pnl  (passive market — could be positive if model long during up market)
    - mean gross       = alpha_pnl + market_pnl
    - mean cost
    - mean net
    - sign agreement: P(side > 0 AND market_term > 0) — does model go long when market up?
    - long_frac
    - alpha IC: corr(predicted_alpha, alpha_realized)
    - return IC: corr(predicted_alpha, my_fwd)

If alpha_pnl >> market_pnl → real alpha
If alpha_pnl ≈ 0 and market_pnl drives total → model captures market direction, NOT alpha
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.cost_model import CostConfig
from ml.cv import FoldSpec, split_features_by_fold
from ml.research.trend_pooled_v2 import _build_symbol_features

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TR_FEATURES = [
    "atr_zscore_1d", "return_1d", "efficiency_96", "adx_15m",
    "bars_since_high", "atr_pct", "realized_vol_1h",
    "dist_resistance_20", "dist_resistance_50",
    "volume_ma_20", "volume_ma_50",
    "ema_slope_20_1h", "bb_squeeze_20", "vpin",
    "hour_cos", "hour_sin",
    "tfi_smooth", "signed_volume",
]
ENSEMBLE_SEEDS = (42, 7, 123, 99, 314)
THRESHOLD_Q = 0.95
HORIZON = 48
REGIME_CUTOFF = 0.33
VOL_WIN = 288
HOLDOUT_DAYS = 90
BETA_WINDOW = 288

REF_OF = {"BTCUSDT": "ETHUSDT", "ETHUSDT": "BTCUSDT", "SOLUSDT": "BTCUSDT"}


def _make_alpha_label_with_components(my_feats, ref_feats, horizon):
    """Build alpha labels AND keep all the intermediate components for forensic decomposition."""
    my_close = my_feats["close"]
    ref_close = ref_feats["close"].reindex(my_close.index).ffill()
    my_fwd = my_close.pct_change(horizon).shift(-horizon)
    exit_time = my_close.index.to_series().shift(-horizon)

    my_ret = my_close.pct_change()
    ref_ret = ref_close.pct_change()
    cov = (my_ret * ref_ret).rolling(BETA_WINDOW).mean() - \
          my_ret.rolling(BETA_WINDOW).mean() * ref_ret.rolling(BETA_WINDOW).mean()
    var = ref_ret.rolling(BETA_WINDOW).var().replace(0, np.nan)
    beta = (cov / var).clip(-3, 3).shift(1)
    ref_fwd = ref_close.pct_change(horizon).shift(-horizon)
    alpha = my_fwd - beta * ref_fwd

    rmean = alpha.expanding(min_periods=288).mean().shift(horizon)
    rstd = alpha.rolling(VOL_WIN * 7, min_periods=VOL_WIN).std().shift(horizon)
    target = (alpha - rmean) / rstd.replace(0, np.nan)
    return pd.DataFrame({
        "return_pct": my_fwd,
        "ref_fwd": ref_fwd,
        "beta": beta,
        "alpha_realized": alpha,
        "demeaned_target": target,
        "exit_time": exit_time,
    }).dropna(subset=["demeaned_target", "return_pct", "exit_time"])


def _train(X_train, y_train, X_cal, y_cal, *, seed):
    params = dict(
        objective="regression", metric="rmse", learning_rate=0.03,
        num_leaves=63, max_depth=8, min_data_in_leaf=50,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        lambda_l2=3.0, verbose=-1,
        seed=seed, feature_fraction_seed=seed, bagging_seed=seed,
        data_random_seed=seed,
    )
    dtr = lgb.Dataset(X_train, y_train, free_raw_data=False)
    dc = lgb.Dataset(X_cal, y_cal, reference=dtr, free_raw_data=False)
    return lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dc],
                     callbacks=[lgb.early_stopping(stopping_rounds=50),
                                lgb.log_evaluation(period=0)])


def main():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    sym_data = {s: _build_symbol_features(s) for s in symbols}
    ref_data = {s: _build_symbol_features(REF_OF[s]) for s in symbols}
    labels_by_sym = {
        s: _make_alpha_label_with_components(sym_data[s][0], ref_data[s][0], HORIZON)
        for s in symbols
    }

    # Define IS / OOS splits
    feats0 = sym_data[symbols[0]][0]
    data_end = feats0.index.max()
    holdout_start = data_end - pd.Timedelta(days=HOLDOUT_DAYS)
    cal_start = holdout_start - pd.Timedelta(days=11)
    cal_end = cal_start + pd.Timedelta(days=10)
    is_train_end = cal_start
    cols_avail = [c for c in TR_FEATURES if c in feats0.columns]

    # ===== Alpha distribution audit =====
    print("=" * 70)
    print("ALPHA TARGET DISTRIBUTION AUDIT (full data)")
    print("=" * 70)
    for s in symbols:
        labels = labels_by_sym[s]
        my = labels["return_pct"]
        alpha = labels["alpha_realized"]
        beta = labels["beta"]
        ref = labels["ref_fwd"]
        print(f"\n{s} (ref={REF_OF[s]}):")
        print(f"  my_fwd:           mean={my.mean()*1e4:+.2f} bps, std={my.std()*1e4:.1f} bps")
        print(f"  ref_fwd:          mean={ref.mean()*1e4:+.2f} bps, std={ref.std()*1e4:.1f} bps")
        print(f"  beta:             mean={beta.mean():.3f}, std={beta.std():.3f}")
        print(f"  alpha_realized:   mean={alpha.mean()*1e4:+.2f} bps, std={alpha.std()*1e4:.1f} bps")
        print(f"  alpha/my_fwd std ratio: {alpha.std()/my.std():.3f}  (should be < 1; lower = more variance reduction)")
        # Correlation
        c = my.corr(ref)
        print(f"  corr(my_fwd, ref_fwd):    {c:+.3f}")

    # ===== Train pooled model on alpha target, evaluate per symbol on OOS =====
    print("\n" + "=" * 70)
    print("TRAIN POOLED ON ALPHA, EVALUATE OOS PER SYMBOL")
    print("=" * 70)

    train_dfs, cal_dfs = [], []
    for s in symbols:
        feats_s, _ = sym_data[s]
        joined = feats_s.join(labels_by_sym[s], how="inner")
        train_purged = joined[joined.index < is_train_end]
        if "exit_time" in train_purged.columns:
            overlap = train_purged["exit_time"] >= is_train_end
            train_purged = train_purged.loc[~overlap]
        cal_chunk = joined[(joined.index >= cal_start) & (joined.index < cal_end)]
        if "exit_time" in cal_chunk.columns:
            overlap = cal_chunk["exit_time"] >= holdout_start
            cal_chunk = cal_chunk.loc[~overlap]
        train_dfs.append(train_purged)
        cal_dfs.append(cal_chunk)
    train = pd.concat(train_dfs, ignore_index=True).dropna(subset=cols_avail + ["demeaned_target", "return_pct"])
    cal = pd.concat(cal_dfs, ignore_index=True).dropna(subset=cols_avail + ["demeaned_target", "return_pct"])
    train_f = train[train["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
    cal_f = cal[cal["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
    log.info("after regime gate: train=%d, cal=%d", len(train_f), len(cal_f))

    cal_preds, models = [], []
    for seed in ENSEMBLE_SEEDS:
        m = _train(train_f[cols_avail].to_numpy(),
                    train_f["demeaned_target"].to_numpy(),
                    cal_f[cols_avail].to_numpy(),
                    cal_f["demeaned_target"].to_numpy(), seed=seed)
        models.append(m)
        cal_preds.append(m.predict(cal_f[cols_avail].to_numpy(), num_iteration=m.best_iteration))
    yc = np.mean(cal_preds, axis=0)
    thr = float(np.quantile(np.abs(yc), THRESHOLD_Q))
    log.info("trigger threshold: %.4f", thr)

    cost = CostConfig(flat_slippage_bps=1.0)
    fee = cost.fee_taker * 2; slip = 2 * cost.flat_slippage_bps / 1e4

    print(f"\nDecomposition: net = alpha_pnl + market_pnl - cost")
    print(f"  alpha_pnl  = side × alpha_realized      (true alpha capture)")
    print(f"  market_pnl = side × beta × ref_fwd       (passive market via position direction)")
    print(f"  net        = side × my_fwd - cost\n")

    rows = []
    for s in symbols:
        feats_s, spread_bps = sym_data[s]
        joined = feats_s.join(labels_by_sym[s], how="inner")
        oos = joined[(joined.index >= holdout_start) & (joined.index < data_end)]
        oos = oos.dropna(subset=cols_avail + ["demeaned_target", "return_pct", "atr_pct",
                                                "alpha_realized", "ref_fwd", "beta"])
        oos_f = oos[oos["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
        if len(oos_f) < 20: continue

        preds = [m.predict(oos_f[cols_avail].to_numpy(), num_iteration=m.best_iteration) for m in models]
        yt = np.mean(preds, axis=0)
        triggered = np.abs(yt) >= thr
        n = int(triggered.sum())
        if n == 0: continue
        side = np.sign(yt[triggered]); side[side == 0] = 1
        atr = np.clip(oos_f["atr_pct"].to_numpy()[triggered], 1e-4, 1e-1)
        inv_vol = np.clip((1.0 / atr) / (1.0 / atr).mean(), 0.3, 3.0)

        # PnL decomposition
        my_fwd_t = oos_f["return_pct"].to_numpy()[triggered]
        ref_fwd_t = oos_f["ref_fwd"].to_numpy()[triggered]
        beta_t = oos_f["beta"].to_numpy()[triggered]
        alpha_realized_t = oos_f["alpha_realized"].to_numpy()[triggered]

        gross = side * my_fwd_t * inv_vol
        alpha_pnl = side * alpha_realized_t * inv_vol
        market_pnl = side * beta_t * ref_fwd_t * inv_vol

        idx = oos_f.index[triggered]
        sp_e = spread_bps.reindex(idx).fillna(0.0).to_numpy()
        sp_x = spread_bps.reindex(oos_f["exit_time"].iloc[triggered]).fillna(0.0).to_numpy()
        spread_term = 0.5 * (sp_e + sp_x) / 1e4
        cost_per_trade = (fee + slip + spread_term) * inv_vol
        net = gross - cost_per_trade

        # ICs
        all_pred = yt
        all_alpha_real = oos_f["alpha_realized"].to_numpy()
        all_my_ret = oos_f["return_pct"].to_numpy()
        ic_alpha = np.corrcoef(all_pred, all_alpha_real)[0, 1]
        ic_my = np.corrcoef(all_pred, all_my_ret)[0, 1]

        # Per-trade summaries
        rows.append({
            "symbol": s,
            "n": n,
            "long_frac": (side > 0).mean(),
            "trigger_rate": triggered.mean(),
            "mean_my_fwd_when_long_bps":  (my_fwd_t[side > 0].mean() * 1e4) if (side > 0).any() else np.nan,
            "mean_ref_fwd_when_long_bps": (ref_fwd_t[side > 0].mean() * 1e4) if (side > 0).any() else np.nan,
            "mean_beta_when_triggered":   beta_t.mean(),
            "alpha_pnl_bps":   alpha_pnl.mean() * 1e4,
            "market_pnl_bps":  market_pnl.mean() * 1e4,
            "gross_pnl_bps":   gross.mean() * 1e4,
            "net_pnl_bps":     net.mean() * 1e4,
            "ic_pred_vs_alpha":  ic_alpha,
            "ic_pred_vs_myret":  ic_my,
        })

    print("=" * 70)
    print("PER-SYMBOL OOS DECOMPOSITION")
    print("=" * 70)
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))

    print("\n--- Interpretation ---")
    for _, r in df.iterrows():
        s = r["symbol"]
        print(f"\n{s}:")
        print(f"  Total net per trade: {r['net_pnl_bps']:+.2f} bps")
        print(f"     = alpha capture {r['alpha_pnl_bps']:+.2f} + market exposure {r['market_pnl_bps']:+.2f} "
              f"- cost {r['gross_pnl_bps'] - r['net_pnl_bps']:.2f}")
        if abs(r['alpha_pnl_bps']) > abs(r['market_pnl_bps']):
            verdict = "ALPHA-driven"
        elif abs(r['market_pnl_bps']) > 2 * abs(r['alpha_pnl_bps']):
            verdict = "MARKET-driven (NOT real alpha capture)"
        else:
            verdict = "MIXED (alpha + market)"
        print(f"  Verdict: {verdict}")
        print(f"  IC pred vs alpha:  {r['ic_pred_vs_alpha']:+.4f}  (should be positive if model predicts alpha)")
        print(f"  IC pred vs my_ret: {r['ic_pred_vs_myret']:+.4f}")


if __name__ == "__main__":
    main()
