"""Test second positioning pack through the linear-oracle gate.

Discipline: before adding any ridge head, the candidate pack must pass:
  Linear oracle Δ IC ≥ +0.001 vs baseline v6_clean

Pack 1 (already validated): funding_z + LS_z + OI_change_24h
   → Linear oracle Δ IC = +0.0027, hybrid Sharpe lift +0.43

Pack 2 (this test): mirror-dynamics of pack 1
  - funding_change_24h: 24h change in funding rate (dynamic, not level)
  - ls_ratio_change_24h: 24h change in top trader L/S ratio (dynamic)
  - oi_z_24h: 24h z-score of OI level (level, not change)

Hypothesis: dynamics of two and level of one are orthogonal to the
levels-of-two-and-change-of-one used in pack 1. Same data source,
different temporal aspect.

If Δ IC > +0.001 → multi-OOS hybrid test
If Δ IC ≤ +0.001 → reject, demonstrates discipline working
"""
from __future__ import annotations
import json
import sys
import time
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
OUT_DIR = REPO / "outputs/h48_pack2_oracle"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0


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


def build_pack2_features(sym, kline_idx):
    """Compute funding_change, ls_change, oi_level (z-scored) — pack 2 mirror."""
    out = pd.DataFrame(index=kline_idx)

    # 1. funding_change_24h: 24h change in funding rate (vs 1d ago)
    f = load_funding(sym)
    if f is not None:
        f5m = f.reindex(f.index.union(kline_idx)).sort_index().ffill().reindex(kline_idx)
        # 24h change = funding(t) - funding(t-24h). Funding settles every 8h, so
        # 24h change captures direction over 3 settlements.
        out["funding_change_24h"] = (f5m - f5m.shift(288)).clip(-0.005, 0.005)
    else:
        out["funding_change_24h"] = np.nan

    m = load_metrics(sym)
    if m is not None:
        # 2. ls_ratio_change_24h: 24h change in top trader L/S ratio
        ls = m["sum_toptrader_long_short_ratio"].copy()
        if ls.index.tz is None:
            ls.index = ls.index.tz_localize("UTC")
        ls5m = ls.reindex(ls.index.union(kline_idx)).sort_index().ffill().reindex(kline_idx)
        out["ls_ratio_change_24h"] = (ls5m - ls5m.shift(288)).clip(-3, 3)

        # 3. oi_z_24h: 24h z-score of OI value
        oi = m["sum_open_interest_value"].copy()
        if oi.index.tz is None:
            oi.index = oi.index.tz_localize("UTC")
        oi5m = oi.reindex(oi.index.union(kline_idx)).sort_index().ffill().reindex(kline_idx)
        rmean = oi5m.rolling(288, min_periods=72).mean()
        rstd = oi5m.rolling(288, min_periods=72).std().replace(0, np.nan)
        out["oi_z_24h"] = ((oi5m - rmean) / rstd).clip(-5, 5)
    else:
        out["ls_ratio_change_24h"] = np.nan
        out["oi_z_24h"] = np.nan

    return out.shift(1)


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
        p2 = build_pack2_features(s, f.index)
        for c in ["funding_change_24h", "ls_ratio_change_24h", "oi_z_24h"]:
            f[c] = p2[c]
        enriched[s] = f

    labels = make_xs_alpha_labels(enriched, basket_close, HORIZON)
    rank_cols_v6 = [c for c in XS_FEATURE_COLS_V6_CLEAN if c.endswith("_xs_rank")]
    src_cols_v6 = list({src for src, dst in XS_RANK_SOURCES.items() if dst in rank_cols_v6})
    pack2_cols = ["funding_change_24h", "ls_ratio_change_24h", "oi_z_24h"]
    needed = list(set(list(XS_FEATURE_COLS_V6_CLEAN)
                        + ["sym_id", "autocorr_pctile_7d", "beta_short_vs_bk"]
                        + src_cols_v6 + pack2_cols) - set(rank_cols_v6))

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
    PACK2_RANK = {
        "funding_change_24h": "funding_change_24h_xs_rank",
        "ls_ratio_change_24h": "ls_ratio_change_24h_xs_rank",
        "oi_z_24h": "oi_z_24h_xs_rank",
    }
    panel = add_xs_rank_features(panel, sources=PACK2_RANK)
    rank_all = list(rank_cols_v6) + list(PACK2_RANK.values())
    for c in rank_all:
        if c in panel.columns:
            panel[c] = panel[c].astype("float32")
    panel = panel.dropna(subset=list(XS_FEATURE_COLS_V6_CLEAN)
                            + ["autocorr_pctile_7d", "demeaned_target", "return_pct"])
    print(f"  panel: {len(panel):,} rows  bars: {panel['open_time'].nunique():,}", flush=True)
    for c in pack2_cols:
        cnt = panel[c].notna().sum()
        print(f"    {c}: {cnt:,} non-null, std {panel[c].std():.4f}", flush=True)
    return panel


def per_bar_ic(pred, y, bar_ids):
    df = pd.DataFrame({"p": pred, "y": y, "b": bar_ids})
    return df.groupby("b").apply(
        lambda g: g["p"].rank().corr(g["y"].rank()) if len(g) >= 3 else np.nan
    ).mean()


def fit_predict_ridge(X_tr, y_tr, X_te, alpha=1.0):
    sc = StandardScaler()
    Xs = sc.fit_transform(np.nan_to_num(X_tr, nan=0.0))
    Xte = sc.transform(np.nan_to_num(X_te, nan=0.0))
    Xs = np.nan_to_num(Xs, nan=0.0); Xte = np.nan_to_num(Xte, nan=0.0)
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(Xs, y_tr)
    return m.predict(Xte)


def z(p):
    s = p.std()
    return (p - p.mean()) / (s if s > 1e-8 else 1.0)


def main():
    panel = build_panel()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    PACK2_RAW = ["funding_change_24h", "ls_ratio_change_24h", "oi_z_24h"]
    PACK2_RANK = ["funding_change_24h_xs_rank", "ls_ratio_change_24h_xs_rank", "oi_z_24h_xs_rank"]
    print(f"Folds: {len(folds)}", flush=True)

    # ===== STAGE 1: LINEAR ORACLE GATE =====
    print("\n" + "=" * 80, flush=True)
    print("STAGE 1: LINEAR ORACLE GATE", flush=True)
    print("Required: Δ IC vs baseline > +0.001 (gate threshold)", flush=True)
    print("=" * 80, flush=True)

    fold_ics = {"baseline": [], "pack2_raw": [], "pack2_rank": []}
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

        for tag, cols in [("baseline", v6_clean),
                            ("pack2_raw", v6_clean + PACK2_RAW),
                            ("pack2_rank", v6_clean + PACK2_RANK)]:
            avail = [c for c in cols if c in panel.columns]
            X_full = np.vstack([tr[avail].to_numpy(dtype=np.float64),
                                  ca[avail].to_numpy(dtype=np.float64)])
            X_te = test[avail].to_numpy(dtype=np.float64)
            pred = fit_predict_ridge(X_full, y_full, X_te, alpha=1.0)
            fold_ics[tag].append(per_bar_ic(pred, y_te, bar_te))

    base_ic = np.mean(fold_ics["baseline"])
    p2_raw_ic = np.mean(fold_ics["pack2_raw"])
    p2_rank_ic = np.mean(fold_ics["pack2_rank"])
    d_raw = p2_raw_ic - base_ic
    d_rank = p2_rank_ic - base_ic

    print(f"  baseline (v6_clean) OLS rank IC:           {base_ic:+.4f}", flush=True)
    print(f"  v6_clean + pack2_raw (3) OLS rank IC:      {p2_raw_ic:+.4f}   Δ {d_raw:+.4f}", flush=True)
    print(f"  v6_clean + pack2_rank (3) OLS rank IC:     {p2_rank_ic:+.4f}   Δ {d_rank:+.4f}", flush=True)

    print(f"\n  GATE THRESHOLD: Δ IC > +0.001", flush=True)
    if max(d_raw, d_rank) > 0.001:
        winning_form = "raw" if d_raw > d_rank else "rank"
        winning_ic = max(d_raw, d_rank)
        print(f"  → PASS ({winning_form} form, Δ IC = {winning_ic:+.4f}). Proceeding to hybrid test.", flush=True)
        proceed_features = PACK2_RAW if winning_form == "raw" else PACK2_RANK
    else:
        print(f"  → FAIL. Δ IC < +0.001 in both forms. Rejecting pack 2.", flush=True)
        print(f"  This demonstrates the discipline: even mirror-dynamics of a", flush=True)
        print(f"  validated pack don't automatically pass. Pack 2 doesn't carry", flush=True)
        print(f"  enough orthogonal info to justify a ridge head.", flush=True)
        summary = {
            "baseline_ic": float(base_ic),
            "pack2_raw_ic": float(p2_raw_ic),
            "pack2_rank_ic": float(p2_rank_ic),
            "delta_ic_raw": float(d_raw),
            "delta_ic_rank": float(d_rank),
            "gate_passed": False,
            "verdict": "rejected_at_gate",
        }
        with open(OUT_DIR / "alpha_v9_pack2_oracle_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  saved → {OUT_DIR}", flush=True)
        return

    # ===== STAGE 2: MULTI-OOS HYBRID TEST (only if gate passed) =====
    print("\n" + "=" * 80, flush=True)
    print("STAGE 2: MULTI-OOS HYBRID TEST (gate passed, proceeding)", flush=True)
    print("=" * 80, flush=True)

    fold_data = {}
    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            continue
        avail_v6 = [c for c in v6_clean if c in panel.columns]
        Xt = tr[avail_v6].to_numpy(dtype=np.float32)
        yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
        Xc = ca[avail_v6].to_numpy(dtype=np.float32)
        yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
        Xtest = test[avail_v6].to_numpy(dtype=np.float32)
        models = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
        lgbm_pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                              for m in models], axis=0)

        avail_pack2 = [c for c in proceed_features if c in panel.columns]
        X_full = np.vstack([tr[avail_pack2].to_numpy(dtype=np.float64),
                              ca[avail_pack2].to_numpy(dtype=np.float64)])
        y_full = np.concatenate([yt_.astype(np.float64), yc_.astype(np.float64)])
        X_te = test[avail_pack2].to_numpy(dtype=np.float64)
        ridge_pack2 = fit_predict_ridge(X_full, y_full, X_te)

        fold_data[fold["fid"]] = {
            "test": test,
            "lgbm_z": z(lgbm_pred),
            "ridge_pack2_z": z(ridge_pack2),
        }
        print(f"  fold {fold['fid']}: {time.time() - t0:.0f}s", flush=True)

    def evaluate_blend(blend):
        recs = []
        for fid, fd in fold_data.items():
            blended = np.zeros_like(fd["lgbm_z"])
            for k, w in blend.items():
                blended = blended + w * fd[k]
            df = evaluate_portfolio(fd["test"], blended, use_gate=True,
                                     gate_pctile=GATE_PCTILE, use_magweight=False, top_k=TOP_K)
            for _, r in df.iterrows():
                recs.append({"fold": fid, "time": r["time"], "net": r["net_bps"],
                              "skipped": r["skipped"], "gross": r["spread_ret_bps"],
                              "cost": r["cost_bps"]})
        return pd.DataFrame(recs)

    print(f"\n  variant                               gross   cost     net  Sharpe          95% CI   Δgate", flush=True)
    base_sh = sharpe_est(evaluate_blend({"lgbm_z": 1.0})["net"].values)

    for label, blend in [
        ("LGBM-only baseline",                      {"lgbm_z": 1.0}),
        ("LGBM + ridge_pack2 (w=0.05)",              {"lgbm_z": 0.95, "ridge_pack2_z": 0.05}),
        ("LGBM + ridge_pack2 (w=0.10)",              {"lgbm_z": 0.90, "ridge_pack2_z": 0.10}),
        ("LGBM + ridge_pack2 (w=0.15)",              {"lgbm_z": 0.85, "ridge_pack2_z": 0.15}),
    ]:
        df = evaluate_blend(blend)
        traded = df[df["skipped"] == 0]
        sh, lo, hi = block_bootstrap_ci(df["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)
        d = sh - base_sh
        print(f"  {label:<38} "
              f"{traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>5.2f}  "
              f"{df['net'].mean():>+6.2f}  "
              f"{sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  {d:>+6.2f}", flush=True)


if __name__ == "__main__":
    main()
