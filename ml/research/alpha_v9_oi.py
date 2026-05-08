"""Test OI (open-interest) and positioning features.

Adds 5 features to v6_clean (28 → 33 panel features, with the new ones
xs_rank-style for v6_clean architectural compatibility):

  oi_z_24h_xs_rank          — per-symbol OI z-score, then per-bar rank
  oi_change_24h_xs_rank     — per-symbol 24h OI delta, then per-bar rank
  price_oi_divergence_xs_rank — sign(price_change_24h) - sign(oi_change_24h),
                                ranked
  taker_ls_ratio_xs_rank    — taker buy/sell volume ratio, per-bar rank
  top_trader_ls_ratio_xs_rank — top trader long/short ratio, per-bar rank

All computed point-in-time (no forward leakage). Multi-OOS paired vs
v6_clean baseline. Mechanism hypothesis: positioning data (OI + L/S
ratios) is genuinely orthogonal to price/volume — same price move
with different OI direction means opposite mean-reversion implication.
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
    build_basket, build_kline_features, list_universe, make_xs_alpha_labels,
)
from ml.research.alpha_v4_xs_1d import (
    ENSEMBLE_SEEDS, _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v4_xs import portfolio_pnl_turnover_aware, block_bootstrap_ci
from ml.research.alpha_v9_conviction_v2 import evaluate_portfolio

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
CACHE_DIR = REPO / "data/ml/cache"
OUT_DIR = REPO / "outputs/h48_oi"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0


def load_metrics(symbol: str) -> pd.DataFrame | None:
    """Load cached metrics, return None if missing."""
    p = CACHE_DIR / f"metrics_{symbol}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return df


def build_oi_features(metrics: pd.DataFrame, kline_close: pd.Series) -> pd.DataFrame:
    """Compute per-symbol OI/positioning features. PIT, no forward leakage.

    Inputs:
      metrics: from metrics_loader, columns include sum_open_interest,
               sum_open_interest_value, sum_taker_long_short_vol_ratio,
               sum_toptrader_long_short_ratio
      kline_close: 5min close prices for the symbol (for price-OI divergence)
    """
    m = metrics.copy()
    if m.index.tz is None:
        m.index = m.index.tz_localize("UTC")
    out = pd.DataFrame(index=m.index)

    # OI in USD (already accounts for price; cleaner cross-symbol comparable)
    oi = m["sum_open_interest_value"]
    # 24h z-score (288 bars × 1 = 288 5min bars)
    rmean = oi.rolling(288, min_periods=72).mean()
    rstd = oi.rolling(288, min_periods=72).std().replace(0, np.nan)
    out["oi_z_24h"] = ((oi - rmean) / rstd).clip(-5, 5)

    # 24h OI change as percentage
    out["oi_change_24h"] = (oi.pct_change(288)).clip(-2, 2)

    # Price-OI divergence: rank gap between sign of price and sign of OI change
    # Indexed at 5min freq matching metrics, so reindex kline close
    px = kline_close.reindex(m.index, method="ffill")
    px_chg = px.pct_change(288)
    # Sign of price change × sign of OI change → quadrants:
    #   (+,+) bullish positioning   (1)
    #   (+,-) short squeeze         (2)  ← interesting
    #   (-,+) fresh shorts          (3)  ← interesting
    #   (-,-) bullish unwind        (4)
    # Encode as continuous: tanh(price_chg) × -tanh(oi_chg)
    # so values: positive when (price up & oi down) or (price down & oi up) — divergence
    #            negative when same-direction (convergence/positioning consistent)
    out["price_oi_divergence"] = (np.tanh(px_chg * 10) * -np.tanh(out["oi_change_24h"] * 5)).clip(-1, 1)

    # Taker-side L/S volume ratio (already a ratio; transform via log)
    tk = m["sum_taker_long_short_vol_ratio"]
    out["taker_ls_log"] = np.log(tk.replace([0, np.inf], np.nan)).clip(-3, 3)

    # Top trader L/S ratio
    tt = m["sum_toptrader_long_short_ratio"]
    out["top_trader_ls_log"] = np.log(tt.replace([0, np.inf], np.nan)).clip(-3, 3)

    # Shift everything by 1 bar to be PIT (use prior bar's value)
    out = out.shift(1)
    return out


def build_panel_with_oi():
    universe_full = sorted(list_universe(min_days=200))
    orig25 = sorted([s for s in universe_full if s not in NEW_SYMBOLS])
    print(f"Building panel for {len(orig25)} ORIG25 syms…")
    feats = {s: build_kline_features(s) for s in orig25}
    closes = pd.DataFrame({s: feats[s]["close"] for s in orig25}).sort_index()
    basket_ret, basket_close = build_basket(closes)
    sym_to_id = {s: i for i, s in enumerate(orig25)}

    enriched = {}
    n_oi_loaded = 0
    for s in orig25:
        f = feats[s].reindex(closes.index)
        f = add_basket_features(f, basket_close, basket_ret)
        f = add_engineered_flow_features(f)
        f["sym_id"] = sym_to_id[s]
        if f.index.tz is None:
            f.index = f.index.tz_localize("UTC")

        # Add OI features
        m = load_metrics(s)
        if m is not None:
            oi_feats = build_oi_features(m, f["close"])
            # Reindex to f's index (5min)
            oi_feats = oi_feats.reindex(f.index, method="ffill")
            for col in ["oi_z_24h", "oi_change_24h", "price_oi_divergence",
                         "taker_ls_log", "top_trader_ls_log"]:
                f[col] = oi_feats[col]
            n_oi_loaded += 1
        else:
            print(f"  WARNING: no metrics for {s}")
            for col in ["oi_z_24h", "oi_change_24h", "price_oi_divergence",
                         "taker_ls_log", "top_trader_ls_log"]:
                f[col] = np.nan
        enriched[s] = f
    print(f"  OI metrics loaded for {n_oi_loaded}/{len(orig25)} symbols")

    labels = make_xs_alpha_labels(enriched, basket_close, HORIZON)

    rank_cols_v6 = [c for c in XS_FEATURE_COLS_V6_CLEAN if c.endswith("_xs_rank")]
    src_cols_v6 = list({src for src, dst in XS_RANK_SOURCES.items() if dst in rank_cols_v6})
    needed = list(set(list(XS_FEATURE_COLS_V6_CLEAN)
                        + ["sym_id", "autocorr_pctile_7d", "beta_short_vs_bk"]
                        + src_cols_v6
                        + ["oi_z_24h", "oi_change_24h", "price_oi_divergence",
                            "taker_ls_log", "top_trader_ls_log"]) - set(rank_cols_v6))

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

    # Add v6_clean xs_rank features as before
    panel = add_xs_rank_features(panel, sources=XS_RANK_SOURCES)
    for c in rank_cols_v6:
        if c in panel.columns:
            panel[c] = panel[c].astype("float32")

    # Build OI-derived xs_rank features
    OI_RANK_SOURCES = {
        "oi_z_24h": "oi_z_24h_xs_rank",
        "oi_change_24h": "oi_change_24h_xs_rank",
        "price_oi_divergence": "price_oi_divergence_xs_rank",
        "taker_ls_log": "taker_ls_log_xs_rank",
        "top_trader_ls_log": "top_trader_ls_log_xs_rank",
    }
    panel = add_xs_rank_features(panel, sources=OI_RANK_SOURCES)
    for c in OI_RANK_SOURCES.values():
        if c in panel.columns:
            panel[c] = panel[c].astype("float32")

    panel = panel.dropna(subset=list(XS_FEATURE_COLS_V6_CLEAN)
                            + ["autocorr_pctile_7d", "demeaned_target", "return_pct"])
    print(f"  panel: {len(panel):,} rows  bars: {panel['open_time'].nunique():,}")
    print(f"  OI feature non-null counts:")
    for c in ["oi_z_24h", "oi_change_24h", "price_oi_divergence",
                "taker_ls_log", "top_trader_ls_log"]:
        print(f"    {c}: {panel[c].notna().sum():,} ({100*panel[c].notna().mean():.1f}%)")
    return panel


def main():
    panel = build_panel_with_oi()
    folds = _multi_oos_splits(panel)
    print(f"Multi-OOS folds: {len(folds)}")

    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    OI_FEATURES = [
        "oi_z_24h_xs_rank",
        "oi_change_24h_xs_rank",
        "price_oi_divergence_xs_rank",
        "taker_ls_log_xs_rank",
        "top_trader_ls_log_xs_rank",
    ]
    feature_sets = {
        "baseline": v6_clean,
        "oi_only_3": v6_clean + ["oi_z_24h_xs_rank", "oi_change_24h_xs_rank",
                                   "price_oi_divergence_xs_rank"],
        "oi_full_5": v6_clean + OI_FEATURES,
    }

    cycles: dict[str, list] = {k: [] for k in feature_sets}
    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            continue
        for tag, feats in feature_sets.items():
            avail = [c for c in feats if c in panel.columns]
            Xt = tr[avail].to_numpy(dtype=np.float32)
            yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
            Xc = ca[avail].to_numpy(dtype=np.float32)
            yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
            Xtest = test[avail].to_numpy(dtype=np.float32)
            models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
            yt_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                                for m in models], axis=0)
            # Evaluate WITH and WITHOUT conv_gate (production rule)
            for gate_label, use_gate in [("sharp", False), ("gate", True)]:
                df = evaluate_portfolio(
                    test, yt_pred,
                    use_gate=use_gate, gate_pctile=0.30,
                    use_magweight=False, top_k=TOP_K,
                )
                key = f"{tag}_{gate_label}"
                if key not in cycles:
                    cycles[key] = []
                for _, r in df.iterrows():
                    cycles[key].append({
                        "fold": fold["fid"], "time": r["time"],
                        "gross": r["spread_ret_bps"], "cost": r["cost_bps"],
                        "net": r["net_bps"], "long_turn": r["long_turnover"],
                        "skipped": r["skipped"],
                    })
        print(f"  fold {fold['fid']}: {time.time() - t0:.0f}s")

    # Summarize
    print("\n" + "=" * 110)
    print(f"OI FEATURE TEST — multi-OOS Sharpe (h={HORIZON} K={TOP_K} ORIG25, {COST_PER_LEG} bps/leg)")
    print("=" * 110)
    print(f"  {'config':<26} {'gross':>7} {'cost':>6} {'net':>7} {'L_turn':>7} "
          f"{'Sharpe':>7} {'95% CI':>15}")

    summary = {}
    for key in sorted(cycles.keys()):
        recs = pd.DataFrame(cycles[key])
        if recs.empty: continue
        traded = recs[recs["skipped"] == 0]
        sh, lo, hi = block_bootstrap_ci(recs["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)
        print(f"  {key:<26} "
              f"{traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>5.2f}  "
              f"{recs['net'].mean():>+6.2f}  "
              f"{traded['long_turn'].mean() if len(traded) > 0 else 0:>6.0%}  "
              f"{sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]")
        summary[key] = {
            "n_cycles": int(len(recs)), "net": float(recs["net"].mean()),
            "sharpe": float(sh), "ci": [float(lo), float(hi)],
        }

    # Paired delta vs baseline_gate (production)
    print(f"\n  --- PAIRED Δ vs baseline_gate (production) ---")
    base_recs = pd.DataFrame(cycles["baseline_gate"])
    for key in ["oi_only_3_gate", "oi_full_5_gate", "oi_only_3_sharp", "oi_full_5_sharp"]:
        if key not in cycles: continue
        v_recs = pd.DataFrame(cycles[key])
        m = base_recs[["fold", "time", "net"]].rename(columns={"net": "base"}).merge(
            v_recs[["fold", "time", "net"]], on=["fold", "time"], how="inner")
        delta = (m["net"] - m["base"]).to_numpy()
        d_sh = sharpe_est(delta)
        t = delta.mean() / (delta.std() / np.sqrt(len(delta))) if delta.std() > 0 else 0
        p = 1 - stats.norm.cdf(abs(t))
        print(f"  {key:<26} ΔSharpe={d_sh:+.2f}  Δnet={delta.mean():+.3f}  t={t:+.2f}  p={p:.4f}  "
              f"wins {(delta > 0).mean()*100:.1f}%")
        summary[f"delta_{key}"] = {
            "delta_sharpe": float(d_sh), "delta_net": float(delta.mean()),
            "t_stat": float(t), "p_value": float(p),
            "wins_pct": float((delta > 0).mean() * 100),
        }

    with open(OUT_DIR / "alpha_v9_oi_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
