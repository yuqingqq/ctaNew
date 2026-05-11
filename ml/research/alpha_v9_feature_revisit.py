"""Revisit previously-failed features through the hybrid linear-oracle gate.

Hypothesis: features that failed when added directly to LGBM may still
carry small orthogonal linear information that a separate ridge head can
extract. Linear oracle is the diagnostic: if Δ IC > +0.001, the
information IS there — only the model couldn't extract it.

Candidate packs to test (each is 3 features):

  A. Stage 2 additives (3): historical "candidates" that failed direct LGBM
       with -1.5 to -2.0 ΔSharpe. Per memory.
       - dom_z_1d_vs_bk: 1d basket-relative dominance (vs 7d in v6_clean)
       - realized_vol_4h: short-window realized vol
       - idio_vol_4h: short-window basket-residual vol

  B. Multi-horizon returns (3): different timescales than v6_clean's return_1d
       - return_2h: 24-bar return
       - return_8h: 96-bar return
       - return_36h: 432-bar return

  C. Funding derivatives (3): different from pack 1's funding-LEVEL
       - funding_vol_7d: rolling std of funding rate (regime indicator)
       - funding_momentum_7d: 7-day change (different from pack 2's 24h)
       - funding_streak_abs: |consecutive same-sign settlements|

  D. Multi-horizon dominance (3): different from v6_clean's dom_z_7d
       - dom_z_3d_vs_bk
       - dom_z_14d_vs_bk
       - dom_change_864b_vs_bk (3d change)

Each pack must clear: Δ IC > +0.001 (linear oracle gate).
If passes, run multi-OOS hybrid as a ridge head at w=0.10.
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
    BETA_WINDOW,
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
OUT_DIR = REPO / "outputs/h48_feature_revisit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0


def load_funding(s):
    p = CACHE_DIR / f"funding_{s}.parquet"
    if not p.exists(): return None
    df = pd.read_parquet(p).set_index("calc_time")["funding_rate"]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[~df.index.duplicated(keep="last")].sort_index()


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

        my_close = f["close"]
        my_ret = my_close.pct_change()
        bk_close_idx = basket_close.reindex(f.index).ffill()
        bk_ret_idx = basket_ret.reindex(f.index).fillna(0)

        # ---- Pack A: stage 2 additives (3) ----
        spread = np.log(my_close / bk_close_idx)
        rmean_1d = spread.rolling(288, min_periods=72).mean()
        rstd_1d = spread.rolling(288, min_periods=72).std().replace(0, np.nan)
        f["dom_z_1d_vs_bk"] = ((spread - rmean_1d) / rstd_1d).clip(-5, 5)
        f["realized_vol_4h"] = my_ret.rolling(48, min_periods=12).std()
        # idio vol 4h: short-window vol of basket-residual returns
        beta = f.get("beta_short_vs_bk", pd.Series(np.nan, index=f.index))
        idio_ret = my_ret - beta * bk_ret_idx
        f["idio_vol_4h"] = idio_ret.rolling(48, min_periods=12).std()

        # ---- Pack B: multi-horizon returns (3) ----
        f["return_2h"] = my_close.pct_change(24)
        f["return_8h"] = my_close.pct_change(96)
        f["return_36h"] = my_close.pct_change(432)

        # ---- Pack C: funding derivatives ----
        fund = load_funding(s)
        if fund is not None:
            fund_5min = fund.reindex(fund.index.union(f.index)).sort_index().ffill().reindex(f.index)
            # 7d window of funding values: 21 settlements at 8h cadence
            window_5m = 7 * 288
            f["funding_vol_7d"] = fund_5min.rolling(window_5m, min_periods=window_5m // 4).std().clip(0, 0.005)
            f["funding_momentum_7d"] = (fund_5min - fund_5min.shift(window_5m)).clip(-0.002, 0.002)
            # streak: rolling sign-counter
            sign = np.sign(fund_5min.fillna(0))
            f["funding_streak_abs"] = sign.rolling(window_5m, min_periods=window_5m // 4).sum().abs().clip(0, 21)
        else:
            f["funding_vol_7d"] = np.nan
            f["funding_momentum_7d"] = np.nan
            f["funding_streak_abs"] = np.nan

        # ---- Pack D: multi-horizon dominance (3) ----
        rmean_3d = spread.rolling(864, min_periods=216).mean()
        rstd_3d = spread.rolling(864, min_periods=216).std().replace(0, np.nan)
        f["dom_z_3d_vs_bk"] = ((spread - rmean_3d) / rstd_3d).clip(-5, 5)
        rmean_14d = spread.rolling(4032, min_periods=1008).mean()
        rstd_14d = spread.rolling(4032, min_periods=1008).std().replace(0, np.nan)
        f["dom_z_14d_vs_bk"] = ((spread - rmean_14d) / rstd_14d).clip(-5, 5)
        f["dom_change_864b_vs_bk"] = spread - spread.shift(864)

        # PIT shift (use prior bar's value)
        new_cols = ["dom_z_1d_vs_bk", "realized_vol_4h", "idio_vol_4h",
                     "return_2h", "return_8h", "return_36h",
                     "funding_vol_7d", "funding_momentum_7d", "funding_streak_abs",
                     "dom_z_3d_vs_bk", "dom_z_14d_vs_bk", "dom_change_864b_vs_bk"]
        for c in new_cols:
            f[c] = f[c].shift(1)
        enriched[s] = f

    labels = make_xs_alpha_labels(enriched, basket_close, HORIZON)
    rank_cols_v6 = [c for c in XS_FEATURE_COLS_V6_CLEAN if c.endswith("_xs_rank")]
    src_cols_v6 = list({src for src, dst in XS_RANK_SOURCES.items() if dst in rank_cols_v6})
    new_cols = ["dom_z_1d_vs_bk", "realized_vol_4h", "idio_vol_4h",
                 "return_2h", "return_8h", "return_36h",
                 "funding_vol_7d", "funding_momentum_7d", "funding_streak_abs",
                 "dom_z_3d_vs_bk", "dom_z_14d_vs_bk", "dom_change_864b_vs_bk"]
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


def z(p):
    s = p.std()
    return (p - p.mean()) / (s if s > 1e-8 else 1.0)


def main():
    panel = build_panel()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    PACKS = {
        "A_stage2_additives": ["dom_z_1d_vs_bk_xs_rank", "realized_vol_4h_xs_rank", "idio_vol_4h_xs_rank"],
        "B_mh_returns":        ["return_2h_xs_rank", "return_8h_xs_rank", "return_36h_xs_rank"],
        "C_funding_derivs":    ["funding_vol_7d_xs_rank", "funding_momentum_7d_xs_rank", "funding_streak_abs_xs_rank"],
        "D_mh_dominance":      ["dom_z_3d_vs_bk_xs_rank", "dom_z_14d_vs_bk_xs_rank", "dom_change_864b_vs_bk_xs_rank"],
    }
    print(f"Folds: {len(folds)}, packs to test: {len(PACKS)}", flush=True)

    # ===== STAGE 1: LINEAR ORACLE GATE =====
    print("\n" + "=" * 90, flush=True)
    print("STAGE 1: LINEAR ORACLE GATE (Δ IC > +0.001 required to proceed)", flush=True)
    print("=" * 90, flush=True)
    fold_ics = {"baseline": []}
    for k in PACKS:
        fold_ics[k] = []

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

        # Baseline
        avail = [c for c in v6_clean if c in panel.columns]
        X_full = np.vstack([tr[avail].to_numpy(dtype=np.float64),
                              ca[avail].to_numpy(dtype=np.float64)])
        X_te = test[avail].to_numpy(dtype=np.float64)
        pred = fit_predict_ridge(X_full, y_full, X_te, alpha=1.0)
        fold_ics["baseline"].append(per_bar_ic(pred, y_te, bar_te))

        # Each pack
        for label, cols in PACKS.items():
            cols_combined = v6_clean + cols
            avail = [c for c in cols_combined if c in panel.columns]
            X_full = np.vstack([tr[avail].to_numpy(dtype=np.float64),
                                  ca[avail].to_numpy(dtype=np.float64)])
            X_te = test[avail].to_numpy(dtype=np.float64)
            pred = fit_predict_ridge(X_full, y_full, X_te, alpha=1.0)
            fold_ics[label].append(per_bar_ic(pred, y_te, bar_te))

    base_ic = np.mean(fold_ics["baseline"])
    print(f"  baseline (v6_clean) OLS rank IC:           {base_ic:+.4f}", flush=True)
    print()
    print(f"  {'pack':<25} {'IC':>10} {'Δ vs baseline':>15} {'gate (>+0.001)':>20}", flush=True)
    passed = []
    for label in PACKS:
        ic = np.mean(fold_ics[label])
        d = ic - base_ic
        verdict = "PASS" if d > 0.001 else "FAIL"
        print(f"  {label:<25} {ic:+.4f}     {d:+.4f}        {verdict}", flush=True)
        if d > 0.001:
            passed.append(label)

    if not passed:
        print(f"\n  → All packs FAILED linear oracle gate. No multi-OOS test needed.", flush=True)
        print(f"  This confirms previously-rejected features genuinely lack orthogonal", flush=True)
        print(f"  linear information; the prior LGBM failures were correctly diagnosing", flush=True)
        print(f"  redundancy, not just model incompetence.", flush=True)
        summary = {
            "baseline_ic": float(base_ic),
            "packs": {k: {"ic": float(np.mean(fold_ics[k])),
                          "delta_ic": float(np.mean(fold_ics[k]) - base_ic),
                          "gate_passed": bool((np.mean(fold_ics[k]) - base_ic) > 0.001)}
                       for k in PACKS},
            "passed_packs": passed,
        }
        with open(OUT_DIR / "alpha_v9_feature_revisit_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  saved → {OUT_DIR}", flush=True)
        return

    # ===== STAGE 2: MULTI-OOS HYBRID FOR PACKS THAT PASSED =====
    print("\n" + "=" * 90, flush=True)
    print(f"STAGE 2: MULTI-OOS HYBRID for {len(passed)} pack(s) that passed gate", flush=True)
    print("=" * 90, flush=True)

    # Cache LGBM predictions and ridge predictions per fold
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

        y_full = np.concatenate([yt_.astype(np.float64), yc_.astype(np.float64)])
        ridge_preds = {}
        for label in passed:
            cols = PACKS[label]
            avail_p = [c for c in cols if c in panel.columns]
            X_full = np.vstack([tr[avail_p].to_numpy(dtype=np.float64),
                                  ca[avail_p].to_numpy(dtype=np.float64)])
            X_te = test[avail_p].to_numpy(dtype=np.float64)
            ridge_preds[label] = z(fit_predict_ridge(X_full, y_full, X_te))

        fold_data[fold["fid"]] = {
            "test": test, "lgbm_z": z(lgbm_pred),
            "ridge_z": ridge_preds,
        }
        print(f"  fold {fold['fid']}: {time.time() - t0:.0f}s", flush=True)

    def evaluate_blend(blend):
        recs = []
        for fid, fd in fold_data.items():
            blended = np.zeros_like(fd["lgbm_z"])
            for k, w in blend.items():
                if k == "lgbm_z":
                    blended = blended + w * fd["lgbm_z"]
                else:
                    blended = blended + w * fd["ridge_z"][k]
            df = evaluate_portfolio(fd["test"], blended, use_gate=True,
                                     gate_pctile=GATE_PCTILE, use_magweight=False, top_k=TOP_K)
            for _, r in df.iterrows():
                recs.append({"fold": fid, "time": r["time"], "net": r["net_bps"],
                              "skipped": r["skipped"], "gross": r["spread_ret_bps"],
                              "cost": r["cost_bps"]})
        return pd.DataFrame(recs)

    print(f"\n  variant                              gross   cost     net  Sharpe          95% CI   Δgate", flush=True)
    base_sh = sharpe_est(evaluate_blend({"lgbm_z": 1.0})["net"].values)
    summary = {"baseline_lgbm": {"sharpe": float(base_sh), "delta": 0.0}}
    for label in passed:
        for w in [0.05, 0.10, 0.15]:
            df = evaluate_blend({"lgbm_z": 1.0 - w, label: w})
            traded = df[df["skipped"] == 0]
            sh, lo, hi = block_bootstrap_ci(df["net"].values, statistic=sharpe_est,
                                              block_size=7, n_boot=2000)
            d = sh - base_sh
            print(f"  LGBM + ridge_{label}@{w:.2f}        "
                  f"{traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
                  f"{traded['cost'].mean() if len(traded) > 0 else 0:>5.2f}  "
                  f"{df['net'].mean():>+6.2f}  "
                  f"{sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  {d:>+6.2f}", flush=True)
            summary[f"{label}_w{w:.2f}"] = {
                "sharpe": float(sh), "ci": [float(lo), float(hi)],
                "delta_vs_baseline": float(d),
                "n_cycles": int(len(df)),
            }

    with open(OUT_DIR / "alpha_v9_feature_revisit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
