"""Test two-anchor orthogonal residualization (BTC + basket-resid-to-BTC).

Current label:
    α = my_fwd - β_basket × basket_fwd
    target = (α - rmean_sym) / rstd_sym

Proposed label (two-anchor orthogonal):
    BTC_fwd already orthogonal (it's the "market" factor for crypto)
    basket_resid = basket - β_(basket→BTC) × BTC          (orthogonalize basket to BTC)
    α = my_fwd - β_(my→BTC) × BTC_fwd - β_(my→basket_resid) × basket_resid_fwd
    target = (α - rmean_sym) / rstd_sym

All betas: rolling 1d (288-bar) PIT, shifted by 1 bar.

Hypothesis: BTC is the dominant crypto factor (~70%+ of variance for most
alts). Equal-weight basket dilutes BTC's specific exposure 1/25, so the
single-basket residual still has BTC variance leaking in. A clean BTC
anchor should produce a sharper alpha-residual target.

Train fresh model on new target, compare to baseline under same multi-OOS
framework + post-fix cost.
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

from features_ml.cross_sectional import (
    XS_FEATURE_COLS_V6_CLEAN, XS_RANK_SOURCES,
    add_basket_features, add_engineered_flow_features, add_xs_rank_features,
    build_basket, build_kline_features, list_universe,
)
from ml.research.alpha_v4_xs_1d import (
    ENSEMBLE_SEEDS, _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v4_xs import portfolio_pnl_turnover_aware, block_bootstrap_ci

HORIZON = 48
TOP_K = 7
TOP_FRAC = TOP_K / 25.0
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
BETA_WINDOW = 288  # 1 day
NEW_SYMBOLS = {"ETCUSDT", "HBARUSDT", "ICPUSDT", "LDOUSDT", "TRBUSDT",
                "AAVEUSDT", "MKRUSDT", "AXSUSDT", "GMXUSDT",
                "1000PEPEUSDT", "1000SHIBUSDT", "TONUSDT", "ORDIUSDT", "WIFUSDT"}
OUT_DIR = REPO / "outputs/h48_multi_anchor"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _rolling_beta(y: pd.Series, x: pd.Series, *, window: int) -> pd.Series:
    """Rolling beta of y vs x; returns shifted by 1 bar (PIT)."""
    cov = (y * x).rolling(window).mean() - y.rolling(window).mean() * x.rolling(window).mean()
    var = x.rolling(window).var().replace(0, np.nan)
    return (cov / var).clip(-5, 5).shift(1)


def build_panels_with_both_labels():
    """Build wide panel with BOTH current (single-basket) and new (two-anchor) labels."""
    universe_full = sorted(list_universe(min_days=200))
    orig25 = sorted([s for s in universe_full if s not in NEW_SYMBOLS])
    print(f"Building panel for {len(orig25)} ORIG25 syms…")

    feats = {s: build_kline_features(s) for s in orig25}
    closes = pd.DataFrame({s: feats[s]["close"] for s in orig25}).sort_index()
    basket_ret, basket_close = build_basket(closes)
    btc_close = closes["BTCUSDT"].copy()
    btc_ret = btc_close.pct_change()

    # β_(basket→BTC), then basket_resid = basket - β × BTC (PIT, on contemporaneous returns)
    beta_bk_to_btc = _rolling_beta(basket_ret, btc_ret, window=BETA_WINDOW)
    bk_resid_ret = basket_ret - beta_bk_to_btc * btc_ret  # PIT
    bk_resid_close = (1 + bk_resid_ret.fillna(0)).cumprod()

    # Forward returns (h-bar)
    btc_fwd = btc_close.pct_change(HORIZON).shift(-HORIZON)
    bk_fwd = basket_close.pct_change(HORIZON).shift(-HORIZON)
    bk_resid_fwd = bk_fwd - beta_bk_to_btc * btc_fwd

    sym_to_id = {s: i for i, s in enumerate(orig25)}

    enriched = {}
    labels_baseline = {}
    labels_2anchor = {}
    for s in orig25:
        f = feats[s].reindex(closes.index)
        f = add_basket_features(f, basket_close, basket_ret)
        f = add_engineered_flow_features(f)
        f["sym_id"] = sym_to_id[s]
        if f.index.tz is None:
            f.index = f.index.tz_localize("UTC")
        enriched[s] = f

        my_close = f["close"]
        my_ret = my_close.pct_change()
        my_fwd = my_close.pct_change(HORIZON).shift(-HORIZON)

        # --- Baseline: single-basket label (matches make_xs_alpha_labels) ---
        beta_bk = f["beta_short_vs_bk"]  # already PIT-shifted
        alpha_base = my_fwd - beta_bk * bk_fwd

        # --- Two-anchor: BTC + basket_resid ---
        beta_my_btc = _rolling_beta(my_ret, btc_ret, window=BETA_WINDOW)
        my_resid_to_btc = my_ret - beta_my_btc * btc_ret
        beta_my_bkr = _rolling_beta(my_resid_to_btc, bk_resid_ret, window=BETA_WINDOW)
        alpha_2a = my_fwd - beta_my_btc * btc_fwd - beta_my_bkr * bk_resid_fwd

        # Per-symbol z-score (matches existing convention)
        def _zscore(alpha):
            rmean = alpha.expanding(min_periods=288).mean().shift(HORIZON)
            rstd = alpha.rolling(288 * 7, min_periods=288).std().shift(HORIZON)
            return (alpha - rmean) / rstd.replace(0, np.nan)

        target_base = _zscore(alpha_base)
        target_2a = _zscore(alpha_2a)

        if f.index.tz is None:
            f.index = f.index.tz_localize("UTC")
        idx = f.index
        labels_baseline[s] = pd.DataFrame({
            "return_pct": my_fwd, "basket_fwd": bk_fwd,
            "alpha_realized": alpha_base, "demeaned_target": target_base,
            "exit_time": idx.to_series().shift(-HORIZON),
        }, index=idx)
        labels_2anchor[s] = pd.DataFrame({
            "return_pct": my_fwd, "basket_fwd": bk_fwd,
            "alpha_realized_2a": alpha_2a, "demeaned_target_2a": target_2a,
        }, index=idx)

    # Stack
    rank_cols = [c for c in XS_FEATURE_COLS_V6_CLEAN if c.endswith("_xs_rank")]
    src_cols = list({src for src, dst in XS_RANK_SOURCES.items() if dst in rank_cols})
    needed = list(set(list(XS_FEATURE_COLS_V6_CLEAN)
                        + ["sym_id", "autocorr_pctile_7d", "beta_short_vs_bk"]
                        + src_cols) - set(rank_cols))

    frames = []
    for s, f in enriched.items():
        avail = [c for c in needed if c in f.columns]
        df = f[avail].join(labels_baseline[s], how="inner")
        df = df.join(labels_2anchor[s][["alpha_realized_2a", "demeaned_target_2a"]], how="inner")
        df["symbol"] = s
        df = df.reset_index().rename(columns={"index": "open_time"})
        for c in df.select_dtypes("float64").columns:
            df[c] = df[c].astype("float32")
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True, sort=False)
    panel = add_xs_rank_features(panel, sources=XS_RANK_SOURCES)
    for c in rank_cols:
        if c in panel.columns:
            panel[c] = panel[c].astype("float32")
    panel = panel.dropna(subset=list(XS_FEATURE_COLS_V6_CLEAN)
                            + ["autocorr_pctile_7d", "demeaned_target",
                                "demeaned_target_2a", "return_pct"])
    print(f"  panel: {len(panel):,} rows  unique bars {panel['open_time'].nunique():,}")
    print(f"  baseline target std: {panel['demeaned_target'].std():.4f}")
    print(f"  2-anchor target std: {panel['demeaned_target_2a'].std():.4f}")
    print(f"  alpha base std: {panel['alpha_realized'].std()*1e4:.2f} bps")
    print(f"  alpha 2a std:   {panel['alpha_realized_2a'].std()*1e4:.2f} bps")
    print(f"  corr(targets):  {panel[['demeaned_target','demeaned_target_2a']].corr().iloc[0,1]:.4f}")
    return panel


def main():
    panel = build_panels_with_both_labels()
    folds = _multi_oos_splits(panel)
    print(f"Multi-OOS folds: {len(folds)}")

    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    pairs = []

    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            continue

        avail = [c for c in v6_clean if c in panel.columns]
        Xt = tr[avail].to_numpy(dtype=np.float32)
        Xc = ca[avail].to_numpy(dtype=np.float32)
        Xtest = test[avail].to_numpy(dtype=np.float32)

        results = {}
        for tag, tcol in [("baseline", "demeaned_target"), ("2anchor", "demeaned_target_2a")]:
            yt_ = tr[tcol].to_numpy(dtype=np.float32)
            yc_ = ca[tcol].to_numpy(dtype=np.float32)
            models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
            yt_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                                for m in models], axis=0)
            r = portfolio_pnl_turnover_aware(
                test, yt_pred, top_frac=TOP_FRAC,
                cost_bps_per_leg=COST_PER_LEG, sample_every=HORIZON, beta_neutral=True,
            )
            results[tag] = r["df"][["time", "net_bps", "spread_ret_bps", "rank_ic", "cost_bps"]].rename(
                columns={c: f"{tag}_{c}" for c in ["net_bps", "spread_ret_bps", "rank_ic", "cost_bps"]})

        merged = results["baseline"].merge(results["2anchor"], on="time", how="inner")
        merged["fold"] = fold["fid"]
        pairs.append(merged)
        print(f"  fold {fold['fid']:>2}: {len(merged)} cycles  "
              f"base_net={merged['baseline_net_bps'].mean():+.2f}  "
              f"2a_net={merged['2anchor_net_bps'].mean():+.2f}  "
              f"base_IC={merged['baseline_rank_ic'].mean():+.4f}  "
              f"2a_IC={merged['2anchor_rank_ic'].mean():+.4f}  "
              f"({time.time() - t0:.0f}s)")

    paired = pd.concat(pairs, ignore_index=True)
    paired["delta_net"] = paired["2anchor_net_bps"] - paired["baseline_net_bps"]

    base = paired["baseline_net_bps"].to_numpy()
    twoa = paired["2anchor_net_bps"].to_numpy()
    delta = paired["delta_net"].to_numpy()

    sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0
    base_sh, base_lo, base_hi = block_bootstrap_ci(base, statistic=sharpe_est, block_size=7, n_boot=2000)
    twoa_sh, twoa_lo, twoa_hi = block_bootstrap_ci(twoa, statistic=sharpe_est, block_size=7, n_boot=2000)
    d_sh = sharpe_est(delta)
    t = delta.mean() / (delta.std() / np.sqrt(len(delta))) if delta.std() > 0 else 0
    p = 1 - stats.norm.cdf(abs(t))

    print("\n" + "=" * 100)
    print("MULTI-OOS PAIRED — TWO-ANCHOR ORTHOGONAL vs SINGLE-BASKET LABEL")
    print(f"  h={HORIZON} K={TOP_K} ORIG25, β-neutral, {COST_PER_LEG} bps/leg, post-fix cost")
    print(f"  {paired['fold'].nunique()} folds, {len(delta)} cycles")
    print("=" * 100)
    print(f"  Baseline (single-basket):    Sharpe {base_sh:+.2f}  "
          f"[{base_lo:+.2f}, {base_hi:+.2f}]   net {base.mean():+.2f} bps/cyc   "
          f"IC {paired['baseline_rank_ic'].mean():+.4f}   "
          f"gross {paired['baseline_spread_ret_bps'].mean():+.2f}")
    print(f"  2-anchor (BTC+basket_resid): Sharpe {twoa_sh:+.2f}  "
          f"[{twoa_lo:+.2f}, {twoa_hi:+.2f}]   net {twoa.mean():+.2f} bps/cyc   "
          f"IC {paired['2anchor_rank_ic'].mean():+.4f}   "
          f"gross {paired['2anchor_spread_ret_bps'].mean():+.2f}")
    print(f"  Delta (2a-base):  ΔSharpe={d_sh:+.2f}  Δnet={delta.mean():+.3f} bps/cyc  "
          f"t={t:+.2f}  one-sided p={p:.4f}  2a-wins={(delta > 0).mean()*100:.1f}%")

    paired.to_csv(OUT_DIR / "alpha_v9_multi_anchor_pairs.csv", index=False)
    summary = {
        "n_cycles": len(delta), "n_folds_used": int(paired["fold"].nunique()),
        "baseline_sharpe": float(base_sh), "baseline_ci": [float(base_lo), float(base_hi)],
        "twoanchor_sharpe": float(twoa_sh), "twoanchor_ci": [float(twoa_lo), float(twoa_hi)],
        "delta_sharpe": float(d_sh), "delta_net_bps": float(delta.mean()),
        "delta_t_stat": float(t), "delta_p_value": float(p),
        "twoanchor_wins_pct": float((delta > 0).mean() * 100),
        "baseline_rank_ic": float(paired["baseline_rank_ic"].mean()),
        "twoanchor_rank_ic": float(paired["2anchor_rank_ic"].mean()),
    }
    with open(OUT_DIR / "alpha_v9_multi_anchor_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
