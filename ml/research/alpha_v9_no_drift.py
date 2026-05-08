"""Test removing per-symbol expanding-mean centering from target.

Current label:
    rmean_s(t) = alpha_s.expanding(288).mean().shift(horizon)
    rstd_s(t)  = alpha_s.rolling(288*7, 288).std().shift(horizon)
    target = (alpha - rmean_s) / rstd_s

The XS strategy demeans cross-sectionally at trade time (it ranks per-bar
across symbols). The per-symbol expanding mean is a slow drift adjustment
that may be redundant — if BTC has a +30 bps/cycle drift in alpha, that
drift gets ranked AGAINST other symbols' drifts, so it's already factored
out cross-sectionally. Subtracting it from the target adds a slow-changing
bias the model has to learn to undo.

Test:
    target_nodrift = alpha / rstd_s   (just standardize, no recentering)

Multi-OOS paired vs current target. Cheap test (one-line label change).
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
NEW_SYMBOLS = {"ETCUSDT", "HBARUSDT", "ICPUSDT", "LDOUSDT", "TRBUSDT",
                "AAVEUSDT", "MKRUSDT", "AXSUSDT", "GMXUSDT",
                "1000PEPEUSDT", "1000SHIBUSDT", "TONUSDT", "ORDIUSDT", "WIFUSDT"}
OUT_DIR = REPO / "outputs/h48_no_drift"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_panel_with_both_targets():
    """Wide panel with current target + nodrift target."""
    universe_full = sorted(list_universe(min_days=200))
    orig25 = sorted([s for s in universe_full if s not in NEW_SYMBOLS])
    print(f"Building panel for {len(orig25)} ORIG25 syms…")

    feats = {s: build_kline_features(s) for s in orig25}
    closes = pd.DataFrame({s: feats[s]["close"] for s in orig25}).sort_index()
    basket_ret, basket_close = build_basket(closes)
    bk_fwd = basket_close.pct_change(HORIZON).shift(-HORIZON)
    sym_to_id = {s: i for i, s in enumerate(orig25)}

    enriched = {}
    labels = {}
    for s in orig25:
        f = feats[s].reindex(closes.index)
        f = add_basket_features(f, basket_close, basket_ret)
        f = add_engineered_flow_features(f)
        f["sym_id"] = sym_to_id[s]
        if f.index.tz is None:
            f.index = f.index.tz_localize("UTC")
        enriched[s] = f

        my_close = f["close"]
        my_fwd = my_close.pct_change(HORIZON).shift(-HORIZON)
        beta = f["beta_short_vs_bk"]
        alpha = my_fwd - beta * bk_fwd

        rmean = alpha.expanding(min_periods=288).mean().shift(HORIZON)
        rstd = alpha.rolling(288 * 7, min_periods=288).std().shift(HORIZON)
        target_curr = (alpha - rmean) / rstd.replace(0, np.nan)
        target_nodrift = alpha / rstd.replace(0, np.nan)

        labels[s] = pd.DataFrame({
            "return_pct": my_fwd, "basket_fwd": bk_fwd,
            "alpha_realized": alpha,
            "demeaned_target": target_curr,
            "nodrift_target": target_nodrift,
            "exit_time": my_close.index.to_series().shift(-HORIZON),
        }, index=my_close.index)

    rank_cols = [c for c in XS_FEATURE_COLS_V6_CLEAN if c.endswith("_xs_rank")]
    src_cols = list({src for src, dst in XS_RANK_SOURCES.items() if dst in rank_cols})
    needed = list(set(list(XS_FEATURE_COLS_V6_CLEAN)
                        + ["sym_id", "autocorr_pctile_7d", "beta_short_vs_bk"]
                        + src_cols) - set(rank_cols))
    frames = []
    for s, f in enriched.items():
        avail = [c for c in needed if c in f.columns]
        df = f[avail].join(labels[s], how="inner")
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
                                "nodrift_target", "return_pct"])
    print(f"  panel: {len(panel):,} rows")
    print(f"  current target std:  {panel['demeaned_target'].std():.4f}, mean {panel['demeaned_target'].mean():+.4f}")
    print(f"  nodrift target std:  {panel['nodrift_target'].std():.4f}, mean {panel['nodrift_target'].mean():+.4f}")
    print(f"  corr(targets):       {panel[['demeaned_target','nodrift_target']].corr().iloc[0,1]:.4f}")
    return panel


def main():
    panel = build_panel_with_both_targets()
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
        for tag, tcol in [("baseline", "demeaned_target"), ("nodrift", "nodrift_target")]:
            yt_ = tr[tcol].to_numpy(dtype=np.float32)
            yc_ = ca[tcol].to_numpy(dtype=np.float32)
            models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
            yt_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                                for m in models], axis=0)
            r = portfolio_pnl_turnover_aware(
                test, yt_pred, top_frac=TOP_FRAC,
                cost_bps_per_leg=COST_PER_LEG, sample_every=HORIZON, beta_neutral=True,
            )
            results[tag] = r["df"][["time", "net_bps", "spread_ret_bps", "rank_ic"]].rename(
                columns={c: f"{tag}_{c}" for c in ["net_bps", "spread_ret_bps", "rank_ic"]})

        merged = results["baseline"].merge(results["nodrift"], on="time", how="inner")
        merged["fold"] = fold["fid"]
        pairs.append(merged)
        print(f"  fold {fold['fid']:>2}: base_net={merged['baseline_net_bps'].mean():+.2f}  "
              f"nodrift_net={merged['nodrift_net_bps'].mean():+.2f}  "
              f"base_IC={merged['baseline_rank_ic'].mean():+.4f}  "
              f"nodrift_IC={merged['nodrift_rank_ic'].mean():+.4f}  ({time.time()-t0:.0f}s)")

    paired = pd.concat(pairs, ignore_index=True)
    paired["delta_net"] = paired["nodrift_net_bps"] - paired["baseline_net_bps"]
    base = paired["baseline_net_bps"].to_numpy()
    nodrift = paired["nodrift_net_bps"].to_numpy()
    delta = paired["delta_net"].to_numpy()

    sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0
    base_sh, base_lo, base_hi = block_bootstrap_ci(base, statistic=sharpe_est, block_size=7, n_boot=2000)
    nd_sh, nd_lo, nd_hi = block_bootstrap_ci(nodrift, statistic=sharpe_est, block_size=7, n_boot=2000)
    d_sh = sharpe_est(delta)
    t = delta.mean() / (delta.std() / np.sqrt(len(delta))) if delta.std() > 0 else 0
    p = 1 - stats.norm.cdf(abs(t))

    print("\n" + "=" * 100)
    print("MULTI-OOS PAIRED — DROP EXPANDING-MEAN CENTERING")
    print(f"  h={HORIZON} K={TOP_K} ORIG25, β-neutral, {COST_PER_LEG} bps/leg, post-fix cost")
    print("=" * 100)
    print(f"  Baseline (with rmean):    Sharpe {base_sh:+.2f}  [{base_lo:+.2f}, {base_hi:+.2f}]   "
          f"net {base.mean():+.2f} bps/cyc   IC {paired['baseline_rank_ic'].mean():+.4f}   "
          f"gross {paired['baseline_spread_ret_bps'].mean():+.2f}")
    print(f"  No-drift (alpha/rstd):    Sharpe {nd_sh:+.2f}  [{nd_lo:+.2f}, {nd_hi:+.2f}]   "
          f"net {nodrift.mean():+.2f} bps/cyc   IC {paired['nodrift_rank_ic'].mean():+.4f}   "
          f"gross {paired['nodrift_spread_ret_bps'].mean():+.2f}")
    print(f"  Delta (nodrift-base):  ΔSharpe={d_sh:+.2f}  Δnet={delta.mean():+.3f} bps/cyc  "
          f"t={t:+.2f}  p={p:.4f}  nd-wins={(delta>0).mean()*100:.1f}%")

    paired.to_csv(OUT_DIR / "alpha_v9_no_drift_pairs.csv", index=False)
    summary = {
        "n_cycles": len(delta),
        "baseline_sharpe": float(base_sh), "baseline_ci": [float(base_lo), float(base_hi)],
        "nodrift_sharpe": float(nd_sh), "nodrift_ci": [float(nd_lo), float(nd_hi)],
        "delta_sharpe": float(d_sh), "delta_net_bps": float(delta.mean()),
        "delta_p_value": float(p),
    }
    with open(OUT_DIR / "alpha_v9_no_drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
