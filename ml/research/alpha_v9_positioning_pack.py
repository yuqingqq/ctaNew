"""Test coherent positioning pack: funding-z + LS-ratio-z + OI-change.

Each component tested individually before:
  - funding rates: -0.99 to -1.94 ΔSharpe, linear Δ IC -0.0003
  - LS ratios:    Δ IC ≈ -0.0009 in OI(5) test
  - OI change:    -3.06 ΔSharpe, linear Δ IC -0.0004

This test: combine the THREE in a focused pack, retrain v6_clean + 3 features,
check both linear oracle and LGBM multi-OOS. Does the coherent positioning
representation extract nonlinear interactions LGBM missed before?

Three feature variants tested:
  - raw 3 features (continuous)
  - xs_rank versions (per-bar percentile)
  - both (raw + rank, 6 features total)
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

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
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v9_conviction_v2 import evaluate_portfolio

HORIZON = 48
TOP_K = 7
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
GATE_PCTILE = 0.30
NEW_SYMBOLS = {"ETCUSDT", "HBARUSDT", "ICPUSDT", "LDOUSDT", "TRBUSDT",
                "AAVEUSDT", "MKRUSDT", "AXSUSDT", "GMXUSDT",
                "1000PEPEUSDT", "1000SHIBUSDT", "TONUSDT", "ORDIUSDT", "WIFUSDT"}
CACHE_DIR = REPO / "data/ml/cache"
OUT_DIR = REPO / "outputs/h48_positioning_pack"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0


def load_funding(sym):
    p = CACHE_DIR / f"funding_{sym}.parquet"
    if not p.exists(): return None
    df = pd.read_parquet(p).set_index("calc_time")["funding_rate"]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[~df.index.duplicated(keep="last")].sort_index()


def load_metrics(sym):
    p = CACHE_DIR / f"metrics_{sym}.parquet"
    if not p.exists(): return None
    return pd.read_parquet(p)


def build_positioning_features(sym, kline_idx):
    """Compute funding_z, ls_ratio_z, oi_change for one symbol on 5min cadence.
    All shifted by 1 bar (PIT)."""
    out = pd.DataFrame(index=kline_idx)

    # 1. Funding-z: z-score of funding rate over trailing 24 settlements (8 days)
    fund = load_funding(sym)
    if fund is not None:
        # Reindex to 5min and ffill (funding is constant between settlements)
        fund_5min = fund.reindex(fund.index.union(kline_idx)).sort_index().ffill().reindex(kline_idx)
        # Trailing 24 settlements ~ 24 × 8h = 192h = 2304 5min bars
        rmean = fund_5min.rolling(2304, min_periods=288).mean()
        rstd = fund_5min.rolling(2304, min_periods=288).std().replace(0, np.nan)
        out["funding_z_24h"] = ((fund_5min - rmean) / rstd).clip(-5, 5)
    else:
        out["funding_z_24h"] = np.nan

    # 2. LS-ratio-z: top trader L/S ratio z-score over trailing 24h
    metrics = load_metrics(sym)
    if metrics is not None:
        ls = metrics["sum_toptrader_long_short_ratio"].copy()
        if ls.index.tz is None:
            ls.index = ls.index.tz_localize("UTC")
        ls_5min = ls.reindex(ls.index.union(kline_idx)).sort_index().ffill().reindex(kline_idx)
        rmean = ls_5min.rolling(288, min_periods=72).mean()
        rstd = ls_5min.rolling(288, min_periods=72).std().replace(0, np.nan)
        out["ls_ratio_z_24h"] = ((ls_5min - rmean) / rstd).clip(-5, 5)

        # 3. OI-change: 24h OI value pct change
        oi = metrics["sum_open_interest_value"].copy()
        if oi.index.tz is None:
            oi.index = oi.index.tz_localize("UTC")
        oi_5min = oi.reindex(oi.index.union(kline_idx)).sort_index().ffill().reindex(kline_idx)
        out["oi_change_24h"] = oi_5min.pct_change(288).clip(-2, 2)
    else:
        out["ls_ratio_z_24h"] = np.nan
        out["oi_change_24h"] = np.nan

    # PIT shift
    return out.shift(1)


def build_panel():
    universe_full = sorted(list_universe(min_days=200))
    orig25 = sorted([s for s in universe_full if s not in NEW_SYMBOLS])
    print(f"Building panel for {len(orig25)} ORIG25 syms…")
    feats = {s: build_kline_features(s) for s in orig25}
    closes = pd.DataFrame({s: feats[s]["close"] for s in orig25}).sort_index()
    basket_ret, basket_close = build_basket(closes)
    sym_to_id = {s: i for i, s in enumerate(orig25)}

    enriched = {}
    n_pos = 0
    for s in orig25:
        f = feats[s].reindex(closes.index)
        f = add_basket_features(f, basket_close, basket_ret)
        f = add_engineered_flow_features(f)
        f["sym_id"] = sym_to_id[s]
        if f.index.tz is None:
            f.index = f.index.tz_localize("UTC")
        pos = build_positioning_features(s, f.index)
        for c in ["funding_z_24h", "ls_ratio_z_24h", "oi_change_24h"]:
            f[c] = pos[c]
            if pos[c].notna().any():
                pass
        if pos["funding_z_24h"].notna().any():
            n_pos += 1
        enriched[s] = f
    print(f"  positioning features loaded for {n_pos}/{len(orig25)} symbols")

    labels = make_xs_alpha_labels(enriched, basket_close, HORIZON)
    rank_cols_v6 = [c for c in XS_FEATURE_COLS_V6_CLEAN if c.endswith("_xs_rank")]
    src_cols_v6 = list({src for src, dst in XS_RANK_SOURCES.items() if dst in rank_cols_v6})
    pos_cols = ["funding_z_24h", "ls_ratio_z_24h", "oi_change_24h"]
    needed = list(set(list(XS_FEATURE_COLS_V6_CLEAN)
                        + ["sym_id", "autocorr_pctile_7d", "beta_short_vs_bk"]
                        + src_cols_v6 + pos_cols) - set(rank_cols_v6))

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
    POS_RANK_SOURCES = {
        "funding_z_24h": "funding_z_24h_xs_rank",
        "ls_ratio_z_24h": "ls_ratio_z_24h_xs_rank",
        "oi_change_24h": "oi_change_24h_xs_rank",
    }
    panel = add_xs_rank_features(panel, sources=POS_RANK_SOURCES)
    rank_all = list(rank_cols_v6) + list(POS_RANK_SOURCES.values())
    for c in rank_all:
        if c in panel.columns:
            panel[c] = panel[c].astype("float32")
    panel = panel.dropna(subset=list(XS_FEATURE_COLS_V6_CLEAN)
                            + ["autocorr_pctile_7d", "demeaned_target", "return_pct"])
    print(f"  panel: {len(panel):,} rows  bars: {panel['open_time'].nunique():,}")
    for c in pos_cols:
        cnt = panel[c].notna().sum()
        print(f"    {c}: {cnt:,} ({100*cnt/len(panel):.1f}%) non-null, "
              f"std {panel[c].std():.4f}")
    return panel


def linear_oracle_per_bar_ic(X_tr, y_tr, X_te, y_te, bar_te):
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(np.nan_to_num(X_tr, nan=0.0))
    Xs_te = scaler.transform(np.nan_to_num(X_te, nan=0.0))
    Xs_tr = np.nan_to_num(Xs_tr, nan=0.0); Xs_te = np.nan_to_num(Xs_te, nan=0.0)
    m = Ridge(alpha=1.0, fit_intercept=True)
    m.fit(Xs_tr, y_tr)
    pred = m.predict(Xs_te)
    df = pd.DataFrame({"pred": pred, "y": y_te, "bar": bar_te})
    ic = df.groupby("bar").apply(
        lambda g: g["pred"].rank().corr(g["y"].rank()) if len(g) >= 3 else np.nan
    ).mean()
    return float(ic)


def main():
    panel = build_panel()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    POS_3 = ["funding_z_24h", "ls_ratio_z_24h", "oi_change_24h"]
    POS_3_RANK = ["funding_z_24h_xs_rank", "ls_ratio_z_24h_xs_rank", "oi_change_24h_xs_rank"]
    POS_6 = POS_3 + POS_3_RANK
    print(f"Multi-OOS folds: {len(folds)}")

    feature_sets = {
        "baseline_v6_clean": v6_clean,
        "v6_clean + pos_raw_3": v6_clean + POS_3,
        "v6_clean + pos_rank_3": v6_clean + POS_3_RANK,
        "v6_clean + pos_all_6": v6_clean + POS_6,
    }

    cycles: dict[str, list] = {k: [] for k in feature_sets}
    linear_ics: dict[str, list] = {k: [] for k in feature_sets}

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

            # Linear oracle (ridge on stacked tr+ca)
            X_full = np.vstack([Xt, Xc]).astype(np.float64)
            y_full = np.concatenate([yt_, yc_]).astype(np.float64)
            bar_te = test["open_time"].astype("int64").values
            ic_ridge = linear_oracle_per_bar_ic(
                X_full, y_full, Xtest.astype(np.float64),
                test["demeaned_target"].to_numpy(dtype=np.float64), bar_te,
            )
            linear_ics[tag].append(ic_ridge)

            # LGBM ensemble
            models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
            yt_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                                for m in models], axis=0)
            df = evaluate_portfolio(test, yt_pred, use_gate=True, gate_pctile=GATE_PCTILE,
                                     use_magweight=False, top_k=TOP_K)
            for _, r in df.iterrows():
                cycles[tag].append({
                    "fold": fold["fid"], "time": r["time"],
                    "gross": r["spread_ret_bps"], "cost": r["cost_bps"],
                    "net": r["net_bps"], "skipped": r["skipped"],
                    "long_turn": r["long_turnover"],
                })
        print(f"  fold {fold['fid']}: {time.time() - t0:.0f}s")

    print("\n" + "=" * 110)
    print(f"POSITIONING PACK TEST (h={HORIZON} K={TOP_K} ORIG25, β-neutral, "
          f"{COST_PER_LEG} bps/leg, post-fix cost, conv_gate p={GATE_PCTILE})")
    print("=" * 110)

    print(f"\n  --- LINEAR ORACLE (Ridge OLS rank IC, per-fold mean) ---")
    print(f"  {'config':<28} {'mean IC':>10} {'Δ vs baseline':>15}")
    base_ic = np.mean(linear_ics["baseline_v6_clean"])
    for tag in feature_sets.keys():
        ic = np.mean(linear_ics[tag])
        d = ic - base_ic
        print(f"  {tag:<28} {ic:>+9.4f}  {d:>+13.4f}")

    print(f"\n  --- LGBM PORTFOLIO MULTI-OOS ---")
    print(f"  {'config':<28} {'cycles':>7} {'gross':>7} {'cost':>6} {'net':>7} "
          f"{'Sharpe':>7} {'95% CI':>15} {'Δgate':>7}")
    base_recs = pd.DataFrame(cycles["baseline_v6_clean"])
    summary = {}
    for tag in feature_sets.keys():
        df = pd.DataFrame(cycles[tag])
        traded = df[df["skipped"] == 0]
        sh, lo, hi = block_bootstrap_ci(df["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)
        m = base_recs[["fold", "time", "net"]].rename(columns={"net": "base"}).merge(
            df[["fold", "time", "net"]], on=["fold", "time"], how="inner")
        d_g = sharpe_est((m["net"] - m["base"]).to_numpy())
        print(f"  {tag:<28} {len(df):>7d} "
              f"{traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>5.2f}  "
              f"{df['net'].mean():>+6.2f}  "
              f"{sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  {d_g:>+6.2f}")
        summary[tag] = {
            "linear_oracle_ic": float(np.mean(linear_ics[tag])),
            "delta_ic_vs_baseline": float(np.mean(linear_ics[tag]) - base_ic),
            "lgbm_sharpe": float(sh), "lgbm_ci": [float(lo), float(hi)],
            "delta_sharpe_vs_baseline": float(d_g),
            "n_cycles": int(len(df)),
            "net": float(df["net"].mean()),
        }

    with open(OUT_DIR / "alpha_v9_positioning_pack_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
