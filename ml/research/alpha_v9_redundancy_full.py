"""Complete redundancy diagnostic for the 6 packs categorized as redundant.

Adds full standalone+marginal+corr+verdict to packs that previously had
only direct LGBM or linear oracle testing:

  G. Funding-rate-as-feature (3): funding_rate level, funding_z_7d, funding_streak_pos
  H. OI-pack-3 (3): oi_z_24h, oi_change_24h, price_oi_divergence
  I. OI-pack-5 (5): adds taker_ls_log, top_trader_ls_log
  J. AggTrade-flow-5: signed_volume_4h, tfi_4h, aggr_ratio_4h, buy_count_4h, avg_trade_size_4h

Plus already-diagnosed packs for sanity check (D and E from prior).

Verdicts:
  standalone_IC > +0.010 AND marginal_d_ic > +0.001 → ORTHOGONAL
  standalone_IC > +0.010 AND marginal_d_ic ≤ +0.001 → REDUNDANT (info captured by v6_clean)
  standalone_IC ≤ +0.010 AND marginal_d_ic ≤ +0.001 → NO INFO
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice
from ml.research.alpha_v8_h48_audit import aggregate_4h_flow

THRESHOLD = 0.50
HORIZON = 48
NEW_SYMBOLS = {"ETCUSDT", "HBARUSDT", "ICPUSDT", "LDOUSDT", "TRBUSDT",
                "AAVEUSDT", "MKRUSDT", "AXSUSDT", "GMXUSDT",
                "1000PEPEUSDT", "1000SHIBUSDT", "TONUSDT", "ORDIUSDT", "WIFUSDT"}
CACHE_DIR = REPO / "data/ml/cache"
OUT_DIR = REPO / "outputs/h48_redundancy_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_predict_ridge(X_tr, y_tr, X_te, alpha=1.0):
    sc = StandardScaler()
    Xs = sc.fit_transform(np.nan_to_num(X_tr, nan=0.0))
    Xte = sc.transform(np.nan_to_num(X_te, nan=0.0))
    Xs = np.nan_to_num(Xs, nan=0.0); Xte = np.nan_to_num(Xte, nan=0.0)
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(Xs, y_tr)
    return m.predict(Xte)


def per_bar_ic(pred, y, bar_ids):
    df = pd.DataFrame({"p": pred, "y": y, "b": bar_ids})
    return df.groupby("b").apply(
        lambda g: g["p"].rank().corr(g["y"].rank()) if len(g) >= 3 else np.nan
    ).mean()


def load_funding(s):
    p = CACHE_DIR / f"funding_{s}.parquet"
    if not p.exists(): return None
    df = pd.read_parquet(p).set_index("calc_time")["funding_rate"]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[~df.index.duplicated(keep="last")].sort_index()


def load_metrics(s):
    p = CACHE_DIR / f"metrics_{s}.parquet"
    if not p.exists(): return None
    return pd.read_parquet(p)


def build_panel():
    universe_full = sorted(list_universe(min_days=200))
    orig25 = sorted([s for s in universe_full if s not in NEW_SYMBOLS])
    print(f"Building panel for {len(orig25)} ORIG25 syms…", flush=True)
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

        # Pack G features: funding-rate-as-feature
        fund = load_funding(s)
        if fund is not None:
            f5m = fund.reindex(fund.index.union(f.index)).sort_index().ffill().reindex(f.index)
            f["funding_rate_level"] = f5m.shift(1).clip(-0.005, 0.005)
            # 7-day z (24 settlements)
            window = 7 * 288
            rmean = f5m.rolling(window, min_periods=window // 4).mean()
            rstd = f5m.rolling(window, min_periods=window // 4).std().replace(0, np.nan)
            f["funding_z_7d"] = ((f5m - rmean) / rstd).clip(-5, 5).shift(1)
            # Streak: rolling sign-counter
            sign = np.sign(f5m.fillna(0))
            f["funding_streak_pos"] = (sign.rolling(window, min_periods=window // 4)
                                          .sum().clip(-21, 21)).shift(1)
        else:
            for c in ["funding_rate_level", "funding_z_7d", "funding_streak_pos"]:
                f[c] = np.nan

        # Pack H/I features: OI + L/S ratios
        m = load_metrics(s)
        if m is not None:
            if m.index.tz is None:
                m.index = m.index.tz_localize("UTC")
            oi = m["sum_open_interest_value"].copy()
            oi5m = oi.reindex(oi.index.union(f.index)).sort_index().ffill().reindex(f.index)
            rmean = oi5m.rolling(288, min_periods=72).mean()
            rstd = oi5m.rolling(288, min_periods=72).std().replace(0, np.nan)
            f["oi_z_24h"] = ((oi5m - rmean) / rstd).clip(-5, 5).shift(1)
            f["oi_change_24h"] = oi5m.pct_change(288).clip(-2, 2).shift(1)
            # Price-OI divergence
            px = f["close"]
            px_chg = px.pct_change(288)
            f["price_oi_divergence"] = (np.tanh(px_chg * 10)
                                          * -np.tanh(f["oi_change_24h"] * 5)).clip(-1, 1)
            # Taker L/S volume ratio
            tk = m["sum_taker_long_short_vol_ratio"].copy()
            tk5m = tk.reindex(tk.index.union(f.index)).sort_index().ffill().reindex(f.index)
            f["taker_ls_log"] = np.log(tk5m.replace([0, np.inf], np.nan)).clip(-3, 3).shift(1)
            # Top trader L/S ratio
            tt = m["sum_toptrader_long_short_ratio"].copy()
            tt5m = tt.reindex(tt.index.union(f.index)).sort_index().ffill().reindex(f.index)
            f["top_trader_ls_log"] = np.log(tt5m.replace([0, np.inf], np.nan)).clip(-3, 3).shift(1)
        else:
            for c in ["oi_z_24h", "oi_change_24h", "price_oi_divergence",
                       "taker_ls_log", "top_trader_ls_log"]:
                f[c] = np.nan

        # Pack J features: aggTrade flow (4h-aggregated)
        flow_p = CACHE_DIR / f"flow_{s}.parquet"
        if flow_p.exists():
            flow = pd.read_parquet(flow_p)
            if flow.index.tz is None:
                flow.index = flow.index.tz_localize("UTC")
            agg = aggregate_4h_flow(flow)
            agg5m = agg.reindex(f.index, method="ffill")
            for c in agg.columns:
                f[c] = agg5m[c].shift(1)
        else:
            for c in ["signed_volume_4h", "tfi_4h", "aggr_ratio_4h", "buy_count_4h", "avg_trade_size_4h"]:
                f[c] = np.nan

        enriched[s] = f

    labels = make_xs_alpha_labels(enriched, basket_close, HORIZON)
    rank_cols_v6 = [c for c in XS_FEATURE_COLS_V6_CLEAN if c.endswith("_xs_rank")]
    src_cols_v6 = list({src for src, dst in XS_RANK_SOURCES.items() if dst in rank_cols_v6})
    new_cols = ["funding_rate_level", "funding_z_7d", "funding_streak_pos",
                 "oi_z_24h", "oi_change_24h", "price_oi_divergence",
                 "taker_ls_log", "top_trader_ls_log",
                 "signed_volume_4h", "tfi_4h", "aggr_ratio_4h",
                 "buy_count_4h", "avg_trade_size_4h"]
    needed = list(set(list(XS_FEATURE_COLS_V6_CLEAN)
                        + ["sym_id", "autocorr_pctile_7d", "beta_short_vs_bk"]
                        + src_cols_v6 + new_cols) - set(rank_cols_v6))
    frames = []
    for s, ff in enriched.items():
        avail = [c for c in needed if c in ff.columns]
        df = ff[avail].join(labels[s], how="inner")
        df["symbol"] = s
        df = df.reset_index().rename(columns={"index": "open_time"})
        for c in df.select_dtypes("float64").columns:
            df[c] = df[c].astype("float32")
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True, sort=False)
    panel = add_xs_rank_features(panel, sources=XS_RANK_SOURCES)
    NEW_RANK = {c: f"{c}_xs_rank" for c in new_cols}
    panel = add_xs_rank_features(panel, sources=NEW_RANK)
    rank_all = list(rank_cols_v6) + list(NEW_RANK.values())
    for c in rank_all:
        if c in panel.columns:
            panel[c] = panel[c].astype("float32")
    panel = panel.dropna(subset=list(XS_FEATURE_COLS_V6_CLEAN)
                            + ["autocorr_pctile_7d", "demeaned_target", "return_pct"])
    print(f"  panel: {len(panel):,} rows  bars: {panel['open_time'].nunique():,}", flush=True)
    return panel


def main():
    panel = build_panel()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    PACKS = {
        "G: Funding-as-feature (level, z_7d, streak)": [
            "funding_rate_level_xs_rank", "funding_z_7d_xs_rank", "funding_streak_pos_xs_rank"
        ],
        "H: OI-pack-3 (oi_z, oi_chg, oi_divergence)": [
            "oi_z_24h_xs_rank", "oi_change_24h_xs_rank", "price_oi_divergence_xs_rank"
        ],
        "I: OI-pack-5 (H + taker_ls + top_trader_ls)": [
            "oi_z_24h_xs_rank", "oi_change_24h_xs_rank", "price_oi_divergence_xs_rank",
            "taker_ls_log_xs_rank", "top_trader_ls_log_xs_rank"
        ],
        "J: AggTrade-flow-5 (signed_vol, tfi, aggr, etc.)": [
            "signed_volume_4h_xs_rank", "tfi_4h_xs_rank", "aggr_ratio_4h_xs_rank",
            "buy_count_4h_xs_rank", "avg_trade_size_4h_xs_rank"
        ],
    }
    print(f"Folds: {len(folds)}, packs: {len(PACKS)}", flush=True)

    fold_results = {"baseline": []}
    for k in PACKS:
        fold_results[k] = {"standalone": [], "marginal": [], "pred_corr": []}

    for fold in folds:
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            continue
        bar_te = test["open_time"].astype("int64").values
        y_full = np.concatenate([tr["demeaned_target"].to_numpy(dtype=np.float64),
                                  ca["demeaned_target"].to_numpy(dtype=np.float64)])
        y_te = test["demeaned_target"].to_numpy(dtype=np.float64)

        # Baseline v6 prediction
        avail_v6 = [c for c in v6_clean if c in panel.columns]
        X_full_v6 = np.vstack([tr[avail_v6].to_numpy(dtype=np.float64),
                                ca[avail_v6].to_numpy(dtype=np.float64)])
        X_te_v6 = test[avail_v6].to_numpy(dtype=np.float64)
        pred_v6 = fit_predict_ridge(X_full_v6, y_full, X_te_v6)
        ic_v6 = per_bar_ic(pred_v6, y_te, bar_te)
        fold_results["baseline"].append(ic_v6)

        for label, cols in PACKS.items():
            avail = [c for c in cols if c in panel.columns]
            if not avail:
                continue
            # Standalone: Ridge on PACK ALONE
            X_full_p = np.vstack([tr[avail].to_numpy(dtype=np.float64),
                                    ca[avail].to_numpy(dtype=np.float64)])
            X_te_p = test[avail].to_numpy(dtype=np.float64)
            pred_pack = fit_predict_ridge(X_full_p, y_full, X_te_p)
            ic_pack = per_bar_ic(pred_pack, y_te, bar_te)
            # Marginal: v6 + pack
            X_full_combo = np.vstack([tr[avail_v6 + avail].to_numpy(dtype=np.float64),
                                        ca[avail_v6 + avail].to_numpy(dtype=np.float64)])
            X_te_combo = test[avail_v6 + avail].to_numpy(dtype=np.float64)
            pred_combo = fit_predict_ridge(X_full_combo, y_full, X_te_combo)
            ic_combo = per_bar_ic(pred_combo, y_te, bar_te)
            pred_corr = np.corrcoef(pred_v6, pred_pack)[0, 1]

            fold_results[label]["standalone"].append(ic_pack)
            fold_results[label]["marginal"].append(ic_combo - ic_v6)
            fold_results[label]["pred_corr"].append(pred_corr)

    print("\n" + "=" * 110, flush=True)
    print("REDUNDANCY DIAGNOSTIC — packs G, H, I, J", flush=True)
    print("=" * 110, flush=True)
    base_ic = np.mean(fold_results["baseline"])
    print(f"  baseline (v6_clean) IC: {base_ic:+.4f}", flush=True)
    print()
    print(f"  {'pack':<55} {'standalone':>12} {'marginal_ΔIC':>14} {'pred_corr':>11} {'verdict':>14}", flush=True)
    summary = {"baseline_ic": float(base_ic), "packs": {}}
    for label in PACKS:
        r = fold_results[label]
        if not r["standalone"]:
            print(f"  {label:<55}  NO DATA")
            continue
        si = np.mean(r["standalone"])
        mi = np.mean(r["marginal"])
        pc = np.mean(r["pred_corr"])
        if si > 0.010 and mi > 0.001:
            verdict = "ORTHOGONAL"
        elif si > 0.010:
            verdict = "REDUNDANT"
        elif si > 0.005:
            verdict = "WEAK-REDUNDANT" if mi <= 0.001 else "WEAK-ORTHOGONAL"
        else:
            verdict = "NO INFO"
        print(f"  {label:<55} {si:+.4f}        {mi:+.4f}       {pc:+.3f}    {verdict:<14}", flush=True)
        summary["packs"][label] = {
            "standalone_ic": float(si),
            "marginal_d_ic": float(mi),
            "pred_corr": float(pc),
            "verdict": verdict,
        }

    with open(OUT_DIR / "alpha_v9_redundancy_full_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
