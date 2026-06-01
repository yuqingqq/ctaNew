"""Current-stack vBTC probe for Binance futures metrics features.

This is intentionally a research sidecar, not a production-path edit. It tests
whether cached Binance /metrics data (open interest + long/short ratios) adds
incremental signal to the corrected 4h vBTC stack.

Feature hygiene:
  - metrics are aligned to panel open_time with backward/ffill alignment
  - all raw metrics-derived features are shifted one 5m bar after alignment
  - cross-sectional ranks are computed per open_time from PIT raw features
  - final evaluation reuses alpha_vBTC_final_simulation's folds, PIT listing
    eligibility, rolling-IC universe, conv_gate, PM, flat_real, and DD overlay
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ml.research.alpha_vBTC_final_simulation import (  # noqa: E402
    ALL_FOLDS,
    IC_UPDATE_DAYS,
    IC_WINDOW_DAYS,
    KLINES_DIR,
    MIN_HISTORY_DAYS,
    OOS_FOLDS,
    PANEL_PATH,
    SEEDS,
    THRESHOLD,
    WINNER_21,
    _max_dd,
    _sharpe,
    apply_dd_tier_aggressive,
    build_rolling_ic_universe_pit,
    evaluate_flat_real,
    get_listing_dates_from_klines,
)
from ml.research.alpha_v4_xs import block_bootstrap_ci  # noqa: E402
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train  # noqa: E402


CACHE_DIR = REPO / "data/ml/cache"
OUT_DIR = REPO / "outputs/vBTC_metrics_feature_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_METRIC_FEATURES = [
    "metrics_oi_z_24h",
    "metrics_oi_change_4h",
    "metrics_oi_change_24h",
    "metrics_top_ls_z_24h",
    "metrics_top_ls_change_4h",
    "metrics_top_ls_change_24h",
    "metrics_taker_ls_z_24h",
    "metrics_taker_ls_change_4h",
    "metrics_taker_ls_change_24h",
    "metrics_price_oi_divergence_24h",
]
RANK_METRIC_FEATURES = [f"{c}_xs_rank" for c in RAW_METRIC_FEATURES]


def _safe_tz_utc(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s)
    if hasattr(out.dtype, "tz") and out.dtype.tz is not None:
        return out.dt.tz_convert("UTC")
    return out.dt.tz_localize("UTC")


def _load_metrics(symbol: str) -> pd.DataFrame | None:
    path = CACHE_DIR / f"metrics_{symbol}.parquet"
    if not path.exists():
        return None
    m = pd.read_parquet(path).sort_index()
    if not isinstance(m.index, pd.DatetimeIndex):
        return None
    if m.index.tz is None:
        m.index = m.index.tz_localize("UTC")
    else:
        m.index = m.index.tz_convert("UTC")
    return m[~m.index.duplicated(keep="last")]


def _rolling_z(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    mu = x.rolling(window, min_periods=min_periods).mean()
    sd = x.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    return ((x - mu) / sd).clip(-5, 5)


def _build_symbol_metrics_features(symbol: str, g: pd.DataFrame) -> pd.DataFrame:
    """Return PIT metrics features indexed like one symbol's panel rows."""
    idx = pd.DatetimeIndex(g["open_time"])
    m = _load_metrics(symbol)
    out = pd.DataFrame(index=idx, columns=RAW_METRIC_FEATURES, dtype="float32")
    if m is None or m.empty:
        return out

    raw = pd.DataFrame(index=m.index)
    oi = m["sum_open_interest_value"].astype("float64")
    top_ls = np.log(
        m["sum_toptrader_long_short_ratio"].astype("float64").replace([0, np.inf], np.nan)
    ).clip(-3, 3)
    taker_ls = np.log(
        m["sum_taker_long_short_vol_ratio"].astype("float64").replace([0, np.inf], np.nan)
    ).clip(-3, 3)

    raw["metrics_oi_z_24h"] = _rolling_z(oi, 288, 72)
    raw["metrics_oi_change_4h"] = oi.pct_change(48).clip(-2, 2)
    raw["metrics_oi_change_24h"] = oi.pct_change(288).clip(-2, 2)
    raw["metrics_top_ls_z_24h"] = _rolling_z(top_ls, 288, 72)
    raw["metrics_top_ls_change_4h"] = (top_ls - top_ls.shift(48)).clip(-3, 3)
    raw["metrics_top_ls_change_24h"] = (top_ls - top_ls.shift(288)).clip(-3, 3)
    raw["metrics_taker_ls_z_24h"] = _rolling_z(taker_ls, 288, 72)
    raw["metrics_taker_ls_change_4h"] = (taker_ls - taker_ls.shift(48)).clip(-3, 3)
    raw["metrics_taker_ls_change_24h"] = (taker_ls - taker_ls.shift(288)).clip(-3, 3)

    # Price/OI divergence: positive when price and OI move opposite directions.
    # Some current vBTC panels omit raw close; in that case leave this single
    # feature as NaN and let the feature-set coverage report show it.
    if "close" in g.columns:
        close = pd.Series(g["close"].to_numpy(), index=idx).sort_index()
        px_on_metrics = close.reindex(close.index.union(raw.index)).sort_index().ffill().reindex(raw.index)
        px_chg = px_on_metrics.pct_change(288)
        raw["metrics_price_oi_divergence_24h"] = (
            np.tanh(px_chg * 10) * -np.tanh(raw["metrics_oi_change_24h"] * 5)
        ).clip(-1, 1)
    else:
        raw["metrics_price_oi_divergence_24h"] = np.nan

    aligned = raw.reindex(raw.index.union(idx)).sort_index().ffill().reindex(idx)
    aligned = aligned.shift(1)  # do not consume a just-published bar at the same timestamp
    for col in RAW_METRIC_FEATURES:
        out[col] = aligned[col].astype("float32")
    return out


def add_metrics_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["open_time"] = _safe_tz_utc(out["open_time"])
    frames = []
    loaded = 0
    for symbol, g in out.groupby("symbol", sort=False):
        g2 = g.sort_values("open_time").copy()
        feats = _build_symbol_metrics_features(symbol, g2)
        if feats.notna().any().any():
            loaded += 1
        for col in RAW_METRIC_FEATURES:
            g2[col] = feats[col].to_numpy()
        frames.append(g2)
    out = pd.concat(frames, ignore_index=True, sort=False)

    for src, dst in zip(RAW_METRIC_FEATURES, RANK_METRIC_FEATURES):
        out[dst] = out.groupby("open_time")[src].rank(pct=True).astype("float32")

    coverage = {
        col: {
            "nonnull": int(out[col].notna().sum()),
            "nonnull_pct": float(out[col].notna().mean() * 100),
        }
        for col in RAW_METRIC_FEATURES + RANK_METRIC_FEATURES
    }
    print(f"  metrics files loaded for {loaded}/{out['symbol'].nunique()} symbols", flush=True)
    return out, coverage


def train_fold(panel: pd.DataFrame, fold: dict, feat_set: list[str], eligible_syms: set[str]):
    train, cal, test = _slice(panel, fold)
    tr = train[(train["autocorr_pctile_7d"] >= THRESHOLD) & (train["symbol"].isin(eligible_syms))]
    ca = cal[(cal["autocorr_pctile_7d"] >= THRESHOLD) & (cal["symbol"].isin(eligible_syms))]
    test_r = test[test["symbol"].isin(eligible_syms)].copy()
    if len(tr) < 1000 or len(ca) < 200 or len(test_r) < 100:
        return None, None
    Xt = tr[feat_set].to_numpy(np.float32)
    Xc = ca[feat_set].to_numpy(np.float32)
    Xtest = test_r[feat_set].to_numpy(np.float32)
    yt = tr["target_A"].to_numpy(np.float32)
    yc = ca["target_A"].to_numpy(np.float32)
    mask_t = ~np.isnan(yt)
    mask_c = ~np.isnan(yc)
    if mask_t.sum() < 1000 or mask_c.sum() < 200:
        return None, None
    preds = []
    for seed in SEEDS:
        model = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=seed)
        preds.append(model.predict(Xtest, num_iteration=model.best_iteration))
    return test_r, np.mean(preds, axis=0)


def evaluate_feature_set(panel: pd.DataFrame, feat_set: list[str], label: str,
                         folds_all: list[dict], eligibility_at) -> tuple[pd.DataFrame, dict]:
    all_preds = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all):
            continue
        t0 = time.time()
        eligible = eligibility_at(folds_all[fid]["cal_start"])
        td, pred = train_fold(panel, folds_all[fid], feat_set, eligible)
        if td is None:
            continue
        cols = ["symbol", "open_time", "alpha_A", "return_pct"]
        if "exit_time" in td.columns:
            cols.append("exit_time")
        df = td[cols].copy()
        df["pred"] = pred
        df["fold"] = fid
        all_preds.append(df)
        print(f"    {label} fold {fid}: eligible={len(eligible)} n_test={len(td):,} "
              f"({time.time() - t0:.0f}s)", flush=True)
    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    ts = apd["open_time"]
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        ts_naive = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        ts_naive = ts
    apd["t_int"] = ts_naive.astype("datetime64[ms]").astype("int64").to_numpy()

    oos_pred = apd[apd["fold"].isin(OOS_FOLDS)].copy()
    # Preserve pandas Timestamp objects with timezone. numpy datetime64 values
    # from Series.unique() do not reliably compare equal to groupby Timestamp
    # keys inside evaluate_flat_real, which makes universe lookups miss.
    oos_times = oos_pred["open_time"].drop_duplicates().sort_values().tolist()[::48]
    rolling_universe, _ = build_rolling_ic_universe_pit(
        apd, oos_times, IC_WINDOW_DAYS, IC_UPDATE_DAYS, eligibility_at
    )
    test_cols = ["symbol", "open_time", "pred", "return_pct", "alpha_A"]
    if "exit_time" in oos_pred.columns:
        test_cols.append("exit_time")
    cycles = evaluate_flat_real(oos_pred[test_cols].copy(), rolling_universe)
    cycles["time"] = pd.to_datetime(cycles["time"])
    for fid in OOS_FOLDS:
        fold_t = set(apd[apd["fold"] == fid]["open_time"].unique())
        cycles.loc[cycles["time"].isin(fold_t), "fold"] = fid

    net_raw = cycles["net_raw_bps"].to_numpy()
    net_overlay, sizes = apply_dd_tier_aggressive(net_raw)
    cycles["net_with_overlay_bps"] = net_overlay
    cycles["size_multiplier"] = sizes
    cycles.to_csv(OUT_DIR / f"{label}_cycles.csv", index=False)

    sh, lo, hi = block_bootstrap_ci(net_raw, statistic=_sharpe, block_size=7, n_boot=2000)
    sho, loo, hio = block_bootstrap_ci(net_overlay, statistic=_sharpe, block_size=7, n_boot=2000)
    summary = {
        "label": label,
        "n_features": len(feat_set),
        "features": feat_set,
        "raw": {
            "sharpe": sh,
            "ci": [lo, hi],
            "mean_bps": float(net_raw.mean()),
            "total_pnl": float(net_raw.sum()),
            "max_dd": _max_dd(net_raw),
        },
        "overlay": {
            "sharpe": sho,
            "ci": [loo, hio],
            "mean_bps": float(net_overlay.mean()),
            "total_pnl": float(net_overlay.sum()),
            "max_dd": _max_dd(net_overlay),
        },
    }
    return cycles, summary


def paired_delta(base: pd.DataFrame, var: pd.DataFrame) -> dict:
    m = base[["time", "net_raw_bps"]].rename(columns={"net_raw_bps": "base"}).merge(
        var[["time", "net_raw_bps"]].rename(columns={"net_raw_bps": "variant"}),
        on="time",
        how="inner",
    )
    delta = (m["variant"] - m["base"]).to_numpy()
    return {
        "n": int(len(delta)),
        "mean_delta_bps": float(delta.mean()),
        "delta_sharpe": _sharpe(delta),
        "wins_pct": float((delta > 0).mean() * 100),
    }


def main():
    print("=== vBTC metrics feature probe (current 4h stack) ===", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    panel["open_time"] = _safe_tz_utc(panel["open_time"])
    print(f"  base panel: {len(panel):,} rows, {panel['symbol'].nunique()} symbols", flush=True)

    panel, coverage = add_metrics_features(panel)
    folds_all = _multi_oos_splits(panel)

    listings = get_listing_dates_from_klines()
    panel_first_obs = panel.groupby("symbol")["open_time"].min()
    for sym, t in panel_first_obs.items():
        if sym not in listings:
            ts = pd.Timestamp(t)
            listings[sym] = ts.tz_convert("UTC") if ts.tz is not None else ts.tz_localize("UTC")
    panel_syms = set(panel["symbol"].unique())

    def eligibility_at(timestamp):
        ts = pd.Timestamp(timestamp)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) is not None and listings[s] <= cutoff}

    base_features = [f for f in WINNER_21 if f in panel.columns]
    feature_sets = {
        "baseline_winner21": base_features,
        "metrics_rank_10": base_features + [f for f in RANK_METRIC_FEATURES if f in panel.columns],
        "metrics_raw_rank_20": (
            base_features
            + [f for f in RAW_METRIC_FEATURES if f in panel.columns]
            + [f for f in RANK_METRIC_FEATURES if f in panel.columns]
        ),
    }

    summaries = {"coverage": coverage, "variants": {}, "paired_delta_vs_baseline": {}}
    cycles_by_label = {}
    for label, feats in feature_sets.items():
        print(f"\n  --- {label}: {len(feats)} features ---", flush=True)
        cycles, summary = evaluate_feature_set(panel, feats, label, folds_all, eligibility_at)
        cycles_by_label[label] = cycles
        summaries["variants"][label] = summary
        print(f"    raw Sharpe {summary['raw']['sharpe']:+.2f} "
              f"CI [{summary['raw']['ci'][0]:+.2f},{summary['raw']['ci'][1]:+.2f}] "
              f"PnL {summary['raw']['total_pnl']:+.0f} maxDD {summary['raw']['max_dd']:+.0f}",
              flush=True)

    base = cycles_by_label["baseline_winner21"]
    for label, cycles in cycles_by_label.items():
        if label == "baseline_winner21":
            continue
        summaries["paired_delta_vs_baseline"][label] = paired_delta(base, cycles)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\n  saved -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
