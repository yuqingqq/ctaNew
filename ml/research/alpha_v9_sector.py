"""Sector-conditional residualization test.

Current label:
    alpha_s = my_fwd - β_basket_s × basket_fwd     (basket = all 25)

Proposed:
    alpha_s = my_fwd - β_sector_s × sector_basket_fwd_{sector(s)}

where sector(s) is symbol s's sector membership and sector_basket excludes
self (leave-one-out).

Sector partition (3 sectors):
    L1     (17): BTC, ETH, BNB, SOL, ADA, AVAX, DOT, ATOM, NEAR, APT, SUI,
                  INJ, TIA, SEI, BCH, LTC, FIL
    Apps   (5):  ARB, OP, LINK, UNI, RUNE
    Other  (3):  DOGE, WLD, XRP

Hypothesis: an L1 symbol's residual against L1 peers is a cleaner signal
than its residual against the full 25-basket (which includes structurally-
different DeFi/L2/meme symbols). Better-residualized labels → better
predictions → potentially higher Sharpe.

Tests under post-fix cost surface, same multi-OOS framework, conv_gate
p=0.30 production rule. Multi-anchor (which failed -2.22) was different:
those used GLOBAL multi-anchor; this uses GROUP-LOCAL anchor per symbol.
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
    build_basket, build_kline_features, list_universe, BETA_WINDOW,
)
from ml.research.alpha_v4_xs_1d import (
    ENSEMBLE_SEEDS, _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v9_conviction_v2 import evaluate_portfolio

HORIZON = 48
TOP_K = 7
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
NEW_SYMBOLS = {"ETCUSDT", "HBARUSDT", "ICPUSDT", "LDOUSDT", "TRBUSDT",
                "AAVEUSDT", "MKRUSDT", "AXSUSDT", "GMXUSDT",
                "1000PEPEUSDT", "1000SHIBUSDT", "TONUSDT", "ORDIUSDT", "WIFUSDT"}
OUT_DIR = REPO / "outputs/h48_sector"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_MAP = {
    # L1: 17 symbols
    "BTCUSDT": "L1", "ETHUSDT": "L1", "BNBUSDT": "L1", "SOLUSDT": "L1",
    "ADAUSDT": "L1", "AVAXUSDT": "L1", "DOTUSDT": "L1", "ATOMUSDT": "L1",
    "NEARUSDT": "L1", "APTUSDT": "L1", "SUIUSDT": "L1", "INJUSDT": "L1",
    "TIAUSDT": "L1", "SEIUSDT": "L1", "BCHUSDT": "L1", "LTCUSDT": "L1",
    "FILUSDT": "L1",
    # Apps (L2s + DeFi): 5 symbols
    "ARBUSDT": "Apps", "OPUSDT": "Apps", "LINKUSDT": "Apps",
    "UNIUSDT": "Apps", "RUNEUSDT": "Apps",
    # Other: 3 symbols
    "DOGEUSDT": "Other", "WLDUSDT": "Other", "XRPUSDT": "Other",
}


def _rolling_beta(y: pd.Series, x: pd.Series, *, window: int = BETA_WINDOW) -> pd.Series:
    cov = (y * x).rolling(window).mean() - y.rolling(window).mean() * x.rolling(window).mean()
    var = x.rolling(window).var().replace(0, np.nan)
    return (cov / var).clip(-5, 5).shift(1)


def build_sector_baskets(closes: pd.DataFrame) -> dict:
    """For each symbol, build leave-one-out sector basket return + close."""
    rets = closes.pct_change()
    sector_baskets = {}
    for sym in closes.columns:
        sec = SECTOR_MAP.get(sym)
        if sec is None:
            continue
        peers = [s for s, sc in SECTOR_MAP.items() if sc == sec and s != sym
                  and s in closes.columns]
        if not peers:
            # No peers in same sector — fall back to whole basket
            peer_rets = rets[[c for c in closes.columns if c != sym]]
        else:
            peer_rets = rets[peers]
        sec_ret = peer_rets.mean(axis=1, skipna=True)
        sec_close = (1.0 + sec_ret.fillna(0.0)).cumprod()
        sector_baskets[sym] = (sec_ret, sec_close)
    return sector_baskets


def make_sector_labels(feats_by_sym: dict, basket_close: pd.Series,
                        sector_baskets: dict, horizon: int) -> dict:
    """Build per-symbol sector-residualized labels."""
    bk_fwd = basket_close.pct_change(horizon).shift(-horizon)  # whole-basket fwd (kept for return_pct etc)
    labels = {}
    for s, f in feats_by_sym.items():
        if s not in sector_baskets:
            continue
        my_close = f["close"]
        my_ret = my_close.pct_change()
        my_fwd = my_close.pct_change(horizon).shift(-horizon)
        sec_ret, sec_close = sector_baskets[s]
        sec_fwd = sec_close.pct_change(horizon).shift(-horizon)

        # β to sector basket (PIT)
        beta_sec = _rolling_beta(my_ret, sec_ret, window=BETA_WINDOW)
        alpha_sector = my_fwd - beta_sec * sec_fwd

        # Per-symbol z-score (consistent with v6_clean convention)
        rmean = alpha_sector.expanding(min_periods=288).mean().shift(horizon)
        rstd = alpha_sector.rolling(288 * 7, min_periods=288).std().shift(horizon)
        target_sector = (alpha_sector - rmean) / rstd.replace(0, np.nan)

        exit_time = my_close.index.to_series().shift(-horizon)
        labels[s] = pd.DataFrame({
            "return_pct": my_fwd, "basket_fwd": bk_fwd,
            "alpha_realized": alpha_sector,
            "demeaned_target_sector": target_sector,
            "exit_time": exit_time,
        }, index=my_close.index)
    return labels


def build_panel_with_sector_labels():
    universe_full = sorted(list_universe(min_days=200))
    orig25 = sorted([s for s in universe_full if s not in NEW_SYMBOLS])
    print(f"Building panel for {len(orig25)} ORIG25 syms…")
    feats = {s: build_kline_features(s) for s in orig25}
    closes = pd.DataFrame({s: feats[s]["close"] for s in orig25}).sort_index()
    basket_ret, basket_close = build_basket(closes)
    sector_baskets = build_sector_baskets(closes)
    sym_to_id = {s: i for i, s in enumerate(orig25)}

    # Print sector composition
    sector_counts = {}
    for s in orig25:
        sec = SECTOR_MAP.get(s, "Unknown")
        sector_counts[sec] = sector_counts.get(sec, 0) + 1
    print(f"  Sector composition: {sector_counts}")

    enriched = {}
    for s in orig25:
        f = feats[s].reindex(closes.index)
        f = add_basket_features(f, basket_close, basket_ret)
        f = add_engineered_flow_features(f)
        f["sym_id"] = sym_to_id[s]
        if f.index.tz is None:
            f.index = f.index.tz_localize("UTC")
        enriched[s] = f

    # Baseline (whole-basket) labels
    from features_ml.cross_sectional import make_xs_alpha_labels
    labels_baseline = make_xs_alpha_labels(enriched, basket_close, HORIZON)
    # Sector labels
    labels_sector = make_sector_labels(enriched, basket_close, sector_baskets, HORIZON)

    rank_cols_v6 = [c for c in XS_FEATURE_COLS_V6_CLEAN if c.endswith("_xs_rank")]
    src_cols_v6 = list({src for src, dst in XS_RANK_SOURCES.items() if dst in rank_cols_v6})
    needed = list(set(list(XS_FEATURE_COLS_V6_CLEAN)
                        + ["sym_id", "autocorr_pctile_7d", "beta_short_vs_bk"]
                        + src_cols_v6) - set(rank_cols_v6))

    frames = []
    for s, f in enriched.items():
        avail = [c for c in needed if c in f.columns]
        df = f[avail].join(labels_baseline[s], how="inner")
        if s in labels_sector:
            df = df.join(labels_sector[s][["demeaned_target_sector", "alpha_realized"]].rename(
                columns={"alpha_realized": "alpha_realized_sector"}),
                how="inner")
        df["symbol"] = s
        df["sector"] = SECTOR_MAP.get(s, "Unknown")
        df = df.reset_index().rename(columns={"index": "open_time"})
        for c in df.select_dtypes("float64").columns:
            df[c] = df[c].astype("float32")
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True, sort=False)
    panel = add_xs_rank_features(panel, sources=XS_RANK_SOURCES)
    for c in rank_cols_v6:
        if c in panel.columns:
            panel[c] = panel[c].astype("float32")
    panel = panel.dropna(subset=list(XS_FEATURE_COLS_V6_CLEAN)
                            + ["autocorr_pctile_7d", "demeaned_target",
                                "demeaned_target_sector", "return_pct"])
    print(f"  panel: {len(panel):,} rows  bars: {panel['open_time'].nunique():,}")
    print(f"  baseline target std:  {panel['demeaned_target'].std():.4f}")
    print(f"  sector target std:    {panel['demeaned_target_sector'].std():.4f}")
    print(f"  alpha base std (bps): {panel['alpha_realized'].std()*1e4:.2f}")
    print(f"  alpha sector std:     {panel['alpha_realized_sector'].std()*1e4:.2f}")
    print(f"  corr(targets):        {panel[['demeaned_target','demeaned_target_sector']].corr().iloc[0,1]:.4f}")
    return panel


def main():
    panel = build_panel_with_sector_labels()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    print(f"Multi-OOS folds: {len(folds)}")

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
        for tag, tcol in [("baseline", "demeaned_target"),
                            ("sector", "demeaned_target_sector")]:
            yt_ = tr[tcol].to_numpy(dtype=np.float32)
            yc_ = ca[tcol].to_numpy(dtype=np.float32)
            models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
            yt_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                                for m in models], axis=0)
            for gate_label, use_gate in [("sharp", False), ("gate", True)]:
                df = evaluate_portfolio(
                    test, yt_pred,
                    use_gate=use_gate, gate_pctile=0.30,
                    use_magweight=False, top_k=TOP_K,
                )
                key = f"{tag}_{gate_label}"
                if key not in results:
                    results[key] = []
                for _, r in df.iterrows():
                    results[key].append({
                        "fold": fold["fid"], "time": r["time"],
                        "gross": r["spread_ret_bps"], "cost": r["cost_bps"],
                        "net": r["net_bps"], "long_turn": r["long_turnover"],
                        "rank_ic": r.get("rank_ic", np.nan), "skipped": r["skipped"],
                    })
        pairs.append(results)
        print(f"  fold {fold['fid']}: {time.time() - t0:.0f}s")

    # Aggregate cycles across folds
    all_keys = ["baseline_sharp", "baseline_gate", "sector_sharp", "sector_gate"]
    agg: dict[str, list] = {k: [] for k in all_keys}
    for fold_results in pairs:
        for k in all_keys:
            agg[k].extend(fold_results.get(k, []))

    sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0

    print("\n" + "=" * 110)
    print(f"SECTOR-RESIDUAL TEST (h={HORIZON} K={TOP_K} ORIG25, β-neutral, "
          f"{COST_PER_LEG} bps/leg, post-fix cost)")
    print("=" * 110)
    print(f"  {'config':<22} {'n_cyc':>5} {'%trade':>7} {'gross':>7} {'cost':>6} "
          f"{'net':>7} {'L_turn':>7} {'Sharpe':>7} {'95% CI':>15}")

    summary = {}
    for key in all_keys:
        df = pd.DataFrame(agg[key])
        if df.empty: continue
        traded = df[df["skipped"] == 0]
        sh, lo, hi = block_bootstrap_ci(df["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)
        print(f"  {key:<22} {len(df):>5d} {100*len(traded)/len(df):>6.1f}% "
              f"{traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>5.2f}  "
              f"{df['net'].mean():>+6.2f}  "
              f"{traded['long_turn'].mean() if len(traded) > 0 else 0:>6.0%}  "
              f"{sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]")
        summary[key] = {
            "n_cycles": int(len(df)), "net": float(df["net"].mean()),
            "sharpe": float(sh), "ci": [float(lo), float(hi)],
        }

    # Paired delta vs production (baseline_gate)
    print(f"\n  --- PAIRED Δ vs production (baseline_gate) ---")
    base = pd.DataFrame(agg["baseline_gate"])
    for key in ["sector_gate", "sector_sharp", "baseline_sharp"]:
        v = pd.DataFrame(agg[key])
        m = base[["fold", "time", "net"]].rename(columns={"net": "base"}).merge(
            v[["fold", "time", "net"]], on=["fold", "time"], how="inner")
        delta = (m["net"] - m["base"]).to_numpy()
        d_sh = sharpe_est(delta)
        t = delta.mean() / (delta.std() / np.sqrt(len(delta))) if delta.std() > 0 else 0
        p = 1 - stats.norm.cdf(abs(t))
        print(f"  {key:<22} ΔSharpe={d_sh:+.2f}  Δnet={delta.mean():+.3f}  t={t:+.2f}  p={p:.4f}  "
              f"wins {(delta > 0).mean()*100:.1f}%")
        summary[f"delta_{key}"] = {
            "delta_sharpe": float(d_sh), "delta_net": float(delta.mean()),
            "t": float(t), "p": float(p),
            "wins_pct": float((delta > 0).mean() * 100),
        }

    with open(OUT_DIR / "alpha_v9_sector_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
