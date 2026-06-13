"""Step 15: Ridge R7 — R3 expanded with target-horizon features that audit
showed are safe for linear (monotonic or u_shape, NOT inverted_u).

Additions to R3 (which already has 16 W17 + 6 squared = 22 features):

  Monotonic features (add as-is, 5):
    + xs_alpha_iqr_12b        (|s|=0.051, monotonic, regime indicator)
    + xs_alpha_mean_48b       (|s|=0.050, monotonic, alt-beta tilt)
    + idio_ret_to_btc_48b     (|s|=0.023, monotonic, 4h backward residual = target-horizon)
    + idio_max_abs_12b        (|s|=0.008, monotonic, jump magnitude)
    + idio_vol_to_btc_1d      (|s|=0.006, monotonic, 1d idio vol)

  U-shape features (add base + squared, 4):
    + idio_kurt_1d   + idio_kurt_1d_sq
    + idio_ret_to_btc_12b   + idio_ret_to_btc_12b_sq
    + beta_to_btc    + beta_to_btc_sq
    + idio_skew_1d   + idio_skew_1d_sq

  DEADLY (NOT added — inverted_u kills linear):
    btc_ret_12b/48b/288b, btc_ema_slope_4h,
    name_factor_loading_1d, name_idio_share_1d

R7 total = 22 (R3) + 5 (mono) + 4×2 (u-shape) = 35 features

Run through V3.1 + IC-signed ranking (the best variant from Step 13).
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec); spec.loader.exec_module(psl)
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice
from ml.research.alpha_v4_xs import block_bootstrap_ci

TARGETS = REPO / "linear_model/data/targets.parquet"
PANEL_BASE = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT = REPO / "linear_model/results"

BASE_16 = ["return_1d","atr_pct","dom_level_vs_bk","dom_change_288b_vs_bk",
           "bk_ema_slope_4h","corr_change_3d_vs_bk","obv_z_1d","vwap_slope_96",
           "bars_since_high_xs_rank","idio_vol_1d_vs_bk_xs_rank",
           "funding_rate","funding_rate_z_7d","corr_to_btc_1d",
           "idio_vol_to_btc_1h","beta_to_btc_change_5d","funding_rate_1d_change"]

U_SHAPE_R3 = ["beta_to_btc_change_5d", "dom_change_288b_vs_bk",
              "corr_to_btc_1d", "corr_change_3d_vs_bk",
              "dom_level_vs_bk", "return_1d"]

NEW_MONOTONIC = ["xs_alpha_iqr_12b", "xs_alpha_mean_48b",
                 "idio_ret_to_btc_48b", "idio_max_abs_12b",
                 "idio_vol_to_btc_1d"]
NEW_USHAPE = ["idio_kurt_1d", "idio_ret_to_btc_12b",
              "beta_to_btc", "idio_skew_1d"]

ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
SEEDS = (42, 1337, 7, 19, 2718)
AUTO_THRESH = 0.5
ALL_FOLDS = list(range(10))
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
CAPITAL = 100.0
TRAILING_IC_DAYS = 90


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def folds_positive(df_v):
    return sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)


def get_listings():
    L = {}
    for d in KLINES_DIR.iterdir():
        if not d.is_dir(): continue
        m5 = d / "5m"
        if not m5.exists(): continue
        f = sorted(m5.glob("*.parquet"))
        if not f: continue
        try: L[d.name] = pd.Timestamp(f[0].stem, tz="UTC")
        except Exception: pass
    return L


def winsorize_zscore(s, train_s, p_lo=0.01, p_hi=0.99):
    s_train = train_s.dropna()
    lo, hi = s_train.quantile(p_lo), s_train.quantile(p_hi)
    s_w = s_train.clip(lower=lo, upper=hi)
    mu, sd = s_w.mean(), s_w.std()
    if sd < 1e-8: sd = 1.0
    return ((s.clip(lower=lo, upper=hi) - mu) / sd).astype("float32")


def build_features(panel, train_panel):
    """R7: R3 features + new monotonic + new u-shape (base+sq)."""
    X = pd.DataFrame({"symbol": panel["symbol"], "open_time": panel["open_time"]})
    # 16 base
    for f in BASE_16:
        X[f] = winsorize_zscore(panel[f], train_panel[f])
    # 6 R3 squared
    for f in U_SHAPE_R3:
        X[f + "_sq"] = (X[f] ** 2).astype("float32")
    # 5 new monotonic (just winsorize+zscore)
    for f in NEW_MONOTONIC:
        X[f] = winsorize_zscore(panel[f], train_panel[f])
    # 4 new u-shape (base + squared)
    for f in NEW_USHAPE:
        X[f] = winsorize_zscore(panel[f], train_panel[f])
        X[f + "_sq"] = (X[f] ** 2).astype("float32")
    feat_cols = [c for c in X.columns if c not in ("symbol", "open_time")]
    X[feat_cols] = X[feat_cols].fillna(0)
    return X, feat_cols


def train(panel, folds_all, feat_cols):
    all_preds = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t_fold = time.time()
        train_, cal, test = _slice(panel, folds_all[fid])
        tr = train_[train_["autocorr_pctile_7d"] >= AUTO_THRESH].dropna(subset=["target_z"])
        te = test.dropna(subset=["target_z"]).copy()
        if len(tr) < 1000 or len(te) < 100: continue
        Xt = tr[feat_cols].to_numpy(np.float32)
        Xte = te[feat_cols].to_numpy(np.float32)
        yt = tr["target_z"].to_numpy(np.float32)
        mt = ~np.isnan(yt)
        fold_preds = []
        alphas = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, mt.sum(), size=mt.sum())
            m = RidgeCV(alphas=ALPHAS, scoring="r2", cv=None, fit_intercept=True)
            m.fit(Xt[mt][idx], yt[mt][idx])
            fold_preds.append(m.predict(Xte).astype(np.float32))
            alphas.append(float(m.alpha_))
        pred = np.mean(fold_preds, axis=0)
        df_pred = te[["symbol","open_time","exit_time","alpha_beta",
                       "target_z","sigma_idio_ref","return_pct"]].copy()
        df_pred["pred_z"] = pred
        df_pred["fold"] = fid
        all_preds.append(df_pred)
        cyc_ic = df_pred.dropna(subset=["alpha_beta"]).groupby("open_time").apply(
            lambda g: g["pred_z"].rank().corr(g["alpha_beta"].rank())
            if len(g) >= 5 else np.nan).dropna()
        print(f"  fold {fid}: IC={cyc_ic.mean():+.4f}, n={len(te):,}, "
              f"α={alphas}, {time.time()-t_fold:.0f}s", flush=True)
    return pd.concat(all_preds, ignore_index=True).sort_values(["open_time","symbol"])


def compute_trailing_ic(apd, sampled_t, win_days=90):
    apd_s = apd[apd["open_time"].isin(set(sampled_t))].sort_values(
        ["symbol","open_time"]).reset_index(drop=True)
    cycles_per_day = 6
    win_cycles = win_days * cycles_per_day
    ic_records = []
    for sym, g in apd_s.groupby("symbol"):
        g = g.sort_values("open_time").reset_index(drop=True)
        pred = g["pred_z"].to_numpy(); alpha = g["alpha_beta"].to_numpy()
        n = len(g)
        ics = np.full(n, np.nan)
        for i in range(50, n):
            lo = max(0, i - win_cycles)
            p, a = pred[lo:i], alpha[lo:i]
            mask = ~np.isnan(p) & ~np.isnan(a)
            if mask.sum() < 50: continue
            pr = pd.Series(p[mask]).rank().to_numpy()
            ar = pd.Series(a[mask]).rank().to_numpy()
            if pr.std() < 1e-6 or ar.std() < 1e-6: continue
            ics[i] = np.corrcoef(pr, ar)[0,1]
        for j, t in enumerate(g["open_time"]):
            ic_records.append({"symbol":sym, "open_time":t, "trail_ic": ics[j]})
    return pd.DataFrame(ic_records).fillna(0)


def aggregate(records, alpha_wide):
    sleeve_queue = deque(maxlen=psl.N_SLEEVES)
    prev_weights = {}
    rows = []
    for _, rec in records.iterrows():
        t = rec["time"]; fold = rec["fold"]
        if rec["traded"]:
            sleeve_queue.append({"longs":list(rec["long_basket"]),
                                  "shorts":list(rec["short_basket"])})
        else:
            sleeve_queue.append({"longs":[],"shorts":[]})
        tw = defaultdict(float)
        sw = 1.0 / psl.N_SLEEVES
        for sl in sleeve_queue:
            nL, nS = len(sl["longs"]), len(sl["shorts"])
            if nL == 0 or nS == 0: continue
            for s in sl["longs"]: tw[s] += sw * (1.0/nL)
            for s in sl["shorts"]: tw[s] -= sw * (1.0/nS)
        gross = 0.0
        if t in alpha_wide.index:
            a = alpha_wide.loc[t]
            for sym, w in prev_weights.items():
                if sym in a.index and not pd.isna(a[sym]):
                    gross += w * a[sym] * 1e4
        syms = set(tw.keys()) | set(prev_weights.keys())
        abs_d = sum(abs(tw.get(s,0)-prev_weights.get(s,0)) for s in syms)
        cost = abs_d * psl.COST_PER_UNIT_ABS_DELTA
        rows.append({"time":t,"fold":fold,"gross_pnl_bps":gross,"cost_bps":cost,
                     "net_pnl_bps":gross-cost,"turnover":abs_d})
        prev_weights = dict(tw)
    return pd.DataFrame(rows)


def main():
    print("=== Step 15: Ridge R7 expanded ===\n", flush=True)
    t0 = time.time()

    tgt = pd.read_parquet(TARGETS)
    tgt["open_time"] = pd.to_datetime(tgt["open_time"], utc=True)
    cols_load = list(set(BASE_16 + NEW_MONOTONIC + NEW_USHAPE))
    base = pd.read_parquet(PANEL_BASE,
                            columns=["symbol","open_time"] + cols_load)
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
    panel = tgt.merge(base, on=["symbol","open_time"], how="left")
    print(f"Panel: {len(panel):,} rows", flush=True)

    folds_all = _multi_oos_splits(panel)
    train_mask = panel["open_time"].between(
        _slice(panel, folds_all[0])[0].open_time.min(),
        _slice(panel, folds_all[0])[0].open_time.max())
    train_panel = panel[train_mask]

    X, feat_cols = build_features(panel, train_panel)
    print(f"R7 features: {len(feat_cols)} (16 base + 6 R3-sq + 5 mono + 4 ushape + 4 ushape-sq)",
          flush=True)
    panel_x = panel.drop(columns=[c for c in cols_load if c in panel.columns])
    panel_x = panel_x.merge(X, on=["symbol","open_time"], how="left")

    apd = train(panel_x, folds_all, feat_cols)
    apd.to_parquet(OUT / "ridge_r7_preds.parquet", index=False)
    cyc_ic = apd.dropna(subset=["alpha_beta"]).groupby("open_time").apply(
        lambda g: g["pred_z"].rank().corr(g["alpha_beta"].rank())
        if len(g) >= 5 else np.nan).dropna()
    print(f"\nR7 overall per-cycle IC: {cyc_ic.mean():+.4f}", flush=True)
    print(f"  vs R3 (22 feats): +0.0133", flush=True)
    print(f"  vs LGBM:          +0.0162", flush=True)

    # Run through V3.1 with both baseline and IC-signed
    listings = get_listings()
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    apd["alpha_A"] = apd["alpha_beta"]
    panel_syms = set(apd["symbol"].unique())
    for s, t in apd.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t

    def elig_pit(b):
        ts = b if isinstance(b, pd.Timestamp) else pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]

    # Trailing IC
    print("\nComputing trailing per-symbol IC...", flush=True)
    df_ic = compute_trailing_ic(apd, sampled_t, TRAILING_IC_DAYS)
    print(f"  trail_ic mean={df_ic['trail_ic'].mean():+.4f}, "
          f"std={df_ic['trail_ic'].std():.4f}", flush=True)

    apd_full = apd.merge(df_ic, on=["symbol","open_time"], how="left")
    apd_full["trail_ic"] = apd_full["trail_ic"].fillna(0)
    apd_full["pred_A"] = apd_full["pred_z"]
    apd_full["pred_B"] = apd_full["pred_z"] * apd_full["trail_ic"]

    # Universe (use base pred_z)
    apd_full["pred"] = apd_full["pred_z"]
    universe = psl.build_rolling_ic_universe(apd_full, sampled_t, psl.TOP_N, elig_pit)
    alpha_wide = apd_full.pivot_table(index="open_time", columns="symbol",
                                        values="alpha_A", aggfunc="first").sort_index()
    print(f"  Universe built", flush=True)

    results = []
    for label, col in [("A_baseline","pred_A"), ("B_IC_signed","pred_B")]:
        print(f"\n  ===== {label} =====", flush=True)
        apd_v = apd_full.copy()
        apd_v["pred"] = apd_v[col]
        records = psl.run_production_protocol_save_sleeves(apd_v, universe)
        df_v = aggregate(records, alpha_wide)
        net = df_v["net_pnl_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe,
                                          block_size=7, n_boot=1000)
        end_eq = CAPITAL + net.sum()/1e4*CAPITAL
        cyc_ic_v = apd_full.dropna(subset=["alpha_beta"]).groupby("open_time").apply(
            lambda g: g[col].rank().corr(g["alpha_beta"].rank())
            if len(g) >= 5 else np.nan).dropna()
        r = {"label":label, "sharpe":sh, "ci_lo":lo, "ci_hi":hi,
             "ic":float(cyc_ic_v.mean()), "end_eq":end_eq,
             "gross":float(df_v["gross_pnl_bps"].mean()),
             "cost":float(df_v["cost_bps"].mean()),
             "folds_pos":folds_positive(df_v),
             "n_traded":int(records["traded"].sum()),
             "maxDD": _max_dd(net)}
        results.append(r)
        print(f"    Sharpe={sh:+.2f} [{lo:+.2f},{hi:+.2f}], IC={r['ic']:+.4f}, "
              f"end-eq=${end_eq:.2f}, gross={r['gross']:+.2f}, "
              f"cost={r['cost']:+.2f}, folds+={r['folds_pos']}/9", flush=True)
        df_v.to_csv(OUT / f"r7_backtest_{label}.csv", index=False)

    print("\n" + "="*90, flush=True)
    print("  R7 EXPANDED RIDGE vs prior variants", flush=True)
    print("="*90, flush=True)
    print(f"  {'variant':<25} {'Sharpe':>8} {'IC':>9}", flush=True)
    for r in results:
        print(f"  R7 {r['label']:<22} {r['sharpe']:+8.2f} {r['ic']:+9.4f}", flush=True)
    print(f"\n  Prior comparisons:", flush=True)
    print(f"    R3 baseline (V3.1):                      +0.03  +0.0133", flush=True)
    print(f"    R3 + IC-signed (best Step 13):          +0.15  +0.0132", flush=True)
    print(f"    Original Ridge (with sym dummies):      -1.62  +0.0135", flush=True)
    print(f"    LGBM clean-PIT production:              +0.74  +0.0157", flush=True)
    print(f"\n  Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
