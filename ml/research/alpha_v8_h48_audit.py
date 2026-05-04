"""Definitive audit: aggTrade trade-flow features at h=48 — TESTED NEGATIVE.

This is the consolidated reproduction of the May 4 2026 audit that
established aggregated trade-flow features do NOT add deployable Sharpe
to v6_clean at h=48 K=7 ORIG25.

The audit ran in three stages, each progressively more rigorous:

Stage 1: Per-symbol univariate IC at h=48
  - 5min-bar primitives (signed_volume, tfi, vpin, etc.) → 1 of 20 passed
    gates, marginal (avg_trade_size only, |IC| ≈ 0.018).
  - 4h-aggregated primitives (signed_volume_4h, tfi_4h, etc.) → 6 of 29
    passed gates with |IC| 0.018-0.026. Suggested microstructure has
    real signal at the right aggregation timescale.

Stage 2: Portfolio impact (multi-OOS at h=48 K=7 on full ORIG25)
  - Adding flows to v6_clean: ΔSharpe ≈ +0.04 (essentially flat)
  - Replacing kline-flow with aggTrade-flow: ΔSharpe varied by panel
    (+1.18 narrow / -0.31 wide) — wide vs narrow had ZERO test-cycle
    overlap due to fold-boundary shifts from the 22k-row truncation,
    making the comparison invalid.

Stage 3: Unified paired test (both configs trained on SAME wide panel,
  evaluated on SAME 1620 OOS cycles, paired per-cycle delta)
  - Δ = -0.54 bps/cycle, paired t = -0.54 (p=0.29), 6/9 folds favor swap
  - **Indistinguishable from zero, slight negative point estimate.**

Mechanism (verified by redundancy diagnostic):
  - aggTrade signed_volume / tfi / aggr_ratio / buy_count are 0.34-0.84
    correlated with kline OBV / VWAP / MFI / volume_ma. The kline
    versions approximate aggressor-side flow from price direction; the
    aggTrade versions observe it directly. Higher fidelity but
    information-equivalent at h=48 horizon.

Run with:
    python3 -m ml.research.alpha_v8_h48_audit

Outputs all artifacts to outputs/h48_features/ for inspection.
Total runtime: ~25 minutes on the dev server.
"""
from __future__ import annotations
import gc, json, sys, time, warnings
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
    build_basket, build_kline_features, list_universe, make_xs_alpha_labels,
)
from ml.research.alpha_v4_xs_1d import (
    ENSEMBLE_SEEDS, _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v4_xs import portfolio_pnl_turnover_aware, block_bootstrap_ci

HORIZON = 48
TOP_FRAC = 7 / 25.0
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
CACHE_DIR = REPO / "data/ml/cache"
OUT_DIR = REPO / "outputs/h48_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_SYMBOLS = {"ETCUSDT", "HBARUSDT", "ICPUSDT", "LDOUSDT", "TRBUSDT",
               "AAVEUSDT", "MKRUSDT", "AXSUSDT", "GMXUSDT",
               "1000PEPEUSDT", "1000SHIBUSDT", "TONUSDT", "ORDIUSDT", "WIFUSDT"}

KLINE_FLOW_TO_DROP = ["obv_z_1d", "obv_signal", "mfi", "vwap_zscore",
                       "vwap_slope_96", "price_volume_corr_10",
                       "price_volume_corr_20", "volume_ma_50",
                       "obv_z_1d_xs_rank", "vwap_zscore_xs_rank"]
AGGTRADE_4H_TO_ADD = ["signed_volume_4h", "tfi_4h", "aggr_ratio_4h",
                       "buy_count_4h", "avg_trade_size_4h"]


def aggregate_4h_flow(flow: pd.DataFrame, w: int = 48) -> pd.DataFrame:
    """4h-aggregated trade-flow features from a flow_<SYM>.parquet cache."""
    sv = flow["signed_volume"].rolling(w, min_periods=max(2, w // 4)).sum()
    tv = (flow["buy_volume"] + flow["sell_volume"]).rolling(w, min_periods=max(2, w // 4)).sum()
    bc = flow["buy_count"].rolling(w, min_periods=max(2, w // 4)).sum()
    sc = flow["sell_count"].rolling(w, min_periods=max(2, w // 4)).sum()
    out = pd.DataFrame(index=flow.index)
    out["signed_volume_4h"] = sv
    out["tfi_4h"] = sv / tv.replace(0, np.nan)
    out["aggr_ratio_4h"] = (bc - sc) / (bc + sc).replace(0, np.nan)
    out["buy_count_4h"] = bc
    out["avg_trade_size_4h"] = tv / (bc + sc).replace(0, np.nan)
    return out


def build_wide_panel():
    """Wide panel: dropna only on v6_clean features. Keeps ~22k early rows
    where flow features are NaN (LGBM handles natively). Production-realistic
    training data."""
    universe_full = sorted(list_universe(min_days=200))
    orig25 = sorted([s for s in universe_full if s not in NEW_SYMBOLS])
    print(f"Building wide panel for {len(orig25)} ORIG25 syms…")
    feats = {s: build_kline_features(s) for s in orig25}
    closes = pd.DataFrame({s: feats[s]["close"] for s in orig25}).sort_index()
    basket_ret, basket_close = build_basket(closes)
    sym_to_id = {s: i for i, s in enumerate(orig25)}

    enriched = {}
    for s in orig25:
        f = feats[s].reindex(closes.index)
        f = add_basket_features(f, basket_close, basket_ret)
        f = add_engineered_flow_features(f)
        f["sym_id"] = sym_to_id[s]
        if f.index.tz is None:
            f.index = f.index.tz_localize("UTC")
        cache = CACHE_DIR / f"flow_{s}.parquet"
        if cache.exists():
            flow = pd.read_parquet(cache)
            if flow.index.tz is None:
                flow.index = flow.index.tz_localize("UTC")
            f = f.join(aggregate_4h_flow(flow), how="left")
        enriched[s] = f
    labels = make_xs_alpha_labels(enriched, basket_close, HORIZON)

    rank_cols = [c for c in XS_FEATURE_COLS_V6_CLEAN if c.endswith("_xs_rank")]
    src_cols = list({s for s, d in XS_RANK_SOURCES.items() if d in rank_cols})
    needed = list(set(list(XS_FEATURE_COLS_V6_CLEAN)
                       + ["sym_id", "autocorr_pctile_7d", "beta_short_vs_bk"]
                       + src_cols + AGGTRADE_4H_TO_ADD) - set(rank_cols))

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
                          + ["autocorr_pctile_7d", "demeaned_target", "return_pct"])
    print(f"  panel: {len(panel):,} rows  ({panel['signed_volume_4h'].isna().sum():,} have NaN flow)")
    return panel


def unified_paired_test(panel):
    """Train both v6_clean and v6_clean_v2 on the SAME panel + folds.
    Compute paired per-cycle delta. This is the production-relevant test."""
    print("\n" + "=" * 100)
    print("STAGE 3: Unified paired test (production-realistic)")
    print("  Both configs trained on same wide panel, evaluated on same 1620 OOS cycles.")
    print("=" * 100)

    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    swap = [f for f in v6_clean if f not in KLINE_FLOW_TO_DROP] + AGGTRADE_4H_TO_ADD

    folds = _multi_oos_splits(panel)
    pairs = []
    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal  [cal  ["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            continue
        results = {}
        for name, feats in [("baseline", v6_clean), ("swap", swap)]:
            avail = [c for c in feats if c in panel.columns]
            Xt = tr[avail].to_numpy(dtype=np.float32)
            yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
            Xc = ca[avail].to_numpy(dtype=np.float32)
            yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
            models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
            Xtest = test[avail].to_numpy(dtype=np.float32)
            yt_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration) for m in models], axis=0)
            r = portfolio_pnl_turnover_aware(
                test, yt_pred, top_frac=TOP_FRAC,
                cost_bps_per_leg=COST_PER_LEG, sample_every=HORIZON, beta_neutral=True,
            )
            results[name] = r["df"][["time", "net_bps"]].rename(
                columns={"net_bps": f"{name}_net"})
        merged = results["baseline"].merge(results["swap"], on="time", how="outer")
        merged["fold"] = fold["fid"]
        pairs.append(merged)
        print(f"  fold {fold['fid']}: {len(merged)} cycles ({time.time()-t0:.0f}s)")

    paired = pd.concat(pairs, ignore_index=True)
    paired["delta_net"] = paired["swap_net"] - paired["baseline_net"]
    base = paired["baseline_net"].dropna().to_numpy()
    swap_arr = paired["swap_net"].dropna().to_numpy()
    delta = paired["delta_net"].dropna().to_numpy()

    base_sharpe = base.mean() / base.std() * np.sqrt(CYCLES_PER_YEAR)
    swap_sharpe = swap_arr.mean() / swap_arr.std() * np.sqrt(CYCLES_PER_YEAR)
    t = delta.mean() / (delta.std() / np.sqrt(len(delta)))
    p = 1 - stats.norm.cdf(abs(t))

    print(f"\nResult (N={len(delta)} OOS cycles):")
    print(f"  v6_clean (current):       Sharpe {base_sharpe:+.2f}, mean {base.mean():+.2f} bps/cycle")
    print(f"  v6_clean_v2 (swap):       Sharpe {swap_sharpe:+.2f}, mean {swap_arr.mean():+.2f} bps/cycle")
    print(f"  paired Δ (swap-base):     {delta.mean():+.3f} bps/cycle")
    print(f"  paired t-stat:            {t:+.2f}  (one-sided p={p:.4f})")
    print(f"  swap-wins rate:           {(delta > 0).mean()*100:.1f}% of cycles")

    paired.to_csv(OUT_DIR / "alpha_v8_h48_paired.csv", index=False)
    summary = {"baseline_sharpe": float(base_sharpe), "swap_sharpe": float(swap_sharpe),
                "paired_delta_mean_bps": float(delta.mean()), "paired_t": float(t),
                "p_value_one_sided": float(p), "n_cycles": len(delta)}
    with open(OUT_DIR / "alpha_v8_h48_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    panel = build_wide_panel()
    summary = unified_paired_test(panel)

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    if abs(summary["paired_t"]) < 1.0:
        print(f"  TESTED NEGATIVE — paired t={summary['paired_t']:+.2f} indistinguishable from zero.")
        print(f"  v6_clean_v2 is information-equivalent to v6_clean within noise.")
        print(f"  Do NOT deploy. aggTrade direction is dead at h=48.")
    elif summary["paired_t"] >= 1.0:
        print(f"  Directional positive (t={summary['paired_t']:+.2f}). Consider shadow mode.")
    else:
        print(f"  Negative (t={summary['paired_t']:+.2f}). Do NOT deploy.")


if __name__ == "__main__":
    main()
