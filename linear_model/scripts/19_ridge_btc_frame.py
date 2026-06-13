"""Step 19: Ridge with PURE BTC-frame features (no basket-frame).

R3_BTC = R3 minus 5 basket features + 5 BTC-frame replacements + audit each.

Basket → BTC replacements (sourced from panel_btc_only_clean.parquet):
  dom_level_vs_bk           → dom_btc_z_1d
  dom_change_288b_vs_bk     → dom_btc_change_288b
  bk_ema_slope_4h           → btc_ema_slope_4h (warning: was inverted_u in Step 14)
  corr_change_3d_vs_bk      → corr_to_btc_change_3d
  idio_vol_1d_vs_bk_xs_rank → idio_vol_to_btc_1d (already in R7)

For each replacement: audit shape, apply transform if needed (squared for u_shape).
Run through V3.1 + IC-signed and compare to R3 (+0.15) and R7 (+1.60).
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from scipy import stats

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec); spec.loader.exec_module(psl)
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice
from ml.research.alpha_v4_xs import block_bootstrap_ci

TARGETS = REPO / "linear_model/data/targets.parquet"
PANEL_BASE = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
PANEL_BTC  = REPO / "outputs/vBTC_features_btc_only/panel_btc_only_clean.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT = REPO / "linear_model/results"

# 11 W17 features that are BTC-frame or frame-neutral (kept)
FRAME_NEUTRAL = ["return_1d","atr_pct","obv_z_1d","vwap_slope_96",
                 "bars_since_high_xs_rank","funding_rate","funding_rate_z_7d",
                 "funding_rate_1d_change",
                 "corr_to_btc_1d", "idio_vol_to_btc_1h", "beta_to_btc_change_5d"]

# 5 basket-frame replaced by BTC-frame
BTC_REPLACEMENTS = ["dom_btc_z_1d", "dom_btc_change_288b",
                    "btc_ema_slope_4h", "corr_to_btc_change_3d",
                    "idio_vol_to_btc_1d"]

# R3's 6 U-shape features get squared terms (only the ones still present)
# Was: beta_to_btc_change_5d, dom_change_288b_vs_bk, corr_to_btc_1d,
#      corr_change_3d_vs_bk, dom_level_vs_bk, return_1d
# In R3_BTC: beta_to_btc_change_5d, corr_to_btc_1d, return_1d are kept
# The 3 basket ones are replaced — need to re-audit shapes of replacements
KEEP_USHAPE_R3 = ["beta_to_btc_change_5d", "corr_to_btc_1d", "return_1d"]

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


def shape_diag(decile_targets):
    d = decile_targets.values
    if len(d) < 10: return "insufficient"
    rho = stats.spearmanr(range(10), d).statistic
    mid = np.mean(d[3:7])
    tails = np.mean(d[[0,1,2,7,8,9]])
    if rho > 0.7:   return "monotonic_up"
    elif rho < -0.7: return "monotonic_down"
    elif mid > tails + abs(np.std(d)) * 0.5: return "inverted_u (BAD)"
    elif mid < tails - abs(np.std(d)) * 0.5: return "u_shape (sq fix)"
    else: return "noisy"


def winsorize_zscore(s, train_s, p_lo=0.01, p_hi=0.99):
    s_train = train_s.dropna()
    lo, hi = s_train.quantile(p_lo), s_train.quantile(p_hi)
    s_w = s_train.clip(lower=lo, upper=hi)
    mu, sd = s_w.mean(), s_w.std()
    if sd < 1e-8: sd = 1.0
    return ((s.clip(lower=lo, upper=hi) - mu) / sd).astype("float32")


def audit_feature(s, target, n_min=1000):
    valid = ~s.isna() & ~target.isna()
    ss, yy = s[valid].clip(s.quantile(0.005), s.quantile(0.995)), target[valid]
    if len(ss) < n_min: return None
    pr = stats.pearsonr(ss, yy).statistic
    sr = stats.spearmanr(ss, yy).statistic
    try:
        q = pd.qcut(ss, 10, labels=False, duplicates="drop")
        dec = yy.groupby(q).mean()
        shape = shape_diag(dec)
        d0, d9 = float(dec.iloc[0])*1e4, float(dec.iloc[-1])*1e4
    except Exception:
        shape = "insufficient"; d0=d9=np.nan
    return {"pearson": pr, "spearman": sr, "shape": shape,
            "d0_bps": d0, "d9_bps": d9,
            "skew": stats.skew(ss), "kurt": stats.kurtosis(ss)}


def compute_trailing_ic(apd, sampled_t, win_days=90):
    apd_s = apd[apd["open_time"].isin(set(sampled_t))].sort_values(
        ["symbol","open_time"]).reset_index(drop=True)
    cycles_per_day = 6
    win_cycles = win_days * cycles_per_day
    rows = []
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
            rows.append({"symbol":sym, "open_time":t, "trail_ic":ics[j]})
    return pd.DataFrame(rows).fillna(0)


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
    print("=== Step 19: Ridge R3_BTC — pure BTC-frame features ===\n", flush=True)
    t0 = time.time()

    tgt = pd.read_parquet(TARGETS)
    tgt["open_time"] = pd.to_datetime(tgt["open_time"], utc=True)
    # Frame-neutral W17 + R3 U-shape kept
    keep_load = list(set(FRAME_NEUTRAL + KEEP_USHAPE_R3))
    base = pd.read_parquet(PANEL_BASE,
                            columns=["symbol","open_time"] + keep_load +
                                     ["btc_ema_slope_4h", "idio_vol_to_btc_1d"])
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
    # BTC-frame replacements from btc_only panel
    btc_load = ["dom_btc_z_1d", "dom_btc_change_288b", "corr_to_btc_change_3d"]
    btc_panel = pd.read_parquet(PANEL_BTC,
                                  columns=["symbol","open_time"] + btc_load)
    btc_panel["open_time"] = pd.to_datetime(btc_panel["open_time"], utc=True)
    base = base.merge(btc_panel, on=["symbol","open_time"], how="left")
    panel = tgt.merge(base, on=["symbol","open_time"], how="left")
    print(f"Panel: {len(panel):,} rows × {panel.shape[1]} cols", flush=True)

    folds_all = _multi_oos_splits(panel)
    train0, _, _ = _slice(panel, folds_all[0])
    train0 = train0[train0["autocorr_pctile_7d"] >= AUTO_THRESH].dropna(subset=["target_z"])
    target_bps = train0["alpha_beta"] * 1e4

    # ===== Audit BTC replacements =====
    print("\nAuditing 5 BTC-frame replacements:", flush=True)
    print(f"  {'feature':<28} {'pearson':>9} {'spearman':>10} "
          f"{'d0_bps':>9} {'d9_bps':>9} {'shape':<24}", flush=True)
    audit_results = {}
    for f in BTC_REPLACEMENTS:
        if f not in panel.columns:
            print(f"    {f}: NOT FOUND in panel"); continue
        s = train0[f]
        a = audit_feature(s, target_bps)
        if a is None:
            print(f"    {f}: insufficient data"); continue
        audit_results[f] = a
        print(f"  {f:<28} {a['pearson']:>+9.4f} {a['spearman']:>+10.4f} "
              f"{a['d0_bps']:>+9.1f} {a['d9_bps']:>+9.1f} {a['shape']:<24}",
              flush=True)

    # Decide which BTC replacements to keep + transforms
    print("\nDecisions:", flush=True)
    btc_keep = []
    btc_squared = []
    for f, a in audit_results.items():
        if "inverted_u" in a["shape"]:
            print(f"  {f}: SKIP (inverted_u — deadly for linear)")
        elif "u_shape" in a["shape"]:
            print(f"  {f}: KEEP + squared (u_shape needs polynomial)")
            btc_keep.append(f); btc_squared.append(f)
        else:
            print(f"  {f}: KEEP as-is ({a['shape']})")
            btc_keep.append(f)

    # ===== Build feature matrix =====
    print(f"\nBuilding feature matrix...", flush=True)
    X = pd.DataFrame({"symbol": panel["symbol"], "open_time": panel["open_time"],
                      "alpha_beta": panel["alpha_beta"],
                      "target_z": panel["target_z"],
                      "autocorr_pctile_7d": panel["autocorr_pctile_7d"]})
    train_panel = panel[panel["open_time"].between(
        _slice(panel, folds_all[0])[0].open_time.min(),
        _slice(panel, folds_all[0])[0].open_time.max())]

    # 11 frame-neutral W17
    for f in FRAME_NEUTRAL:
        X[f] = winsorize_zscore(panel[f], train_panel[f])
    # R3 squared for U-shape ones still present
    for f in KEEP_USHAPE_R3:
        X[f + "_sq"] = (X[f] ** 2).astype("float32")
    # BTC replacements (kept)
    for f in btc_keep:
        X[f] = winsorize_zscore(panel[f], train_panel[f])
    # BTC squared for u-shape replacements
    for f in btc_squared:
        X[f + "_sq"] = (X[f] ** 2).astype("float32")

    feat_cols = [c for c in X.columns if c not in
                  ("symbol","open_time","alpha_beta","target_z","autocorr_pctile_7d")]
    X[feat_cols] = X[feat_cols].fillna(0)
    print(f"  R3_BTC features: {len(feat_cols)}", flush=True)
    print(f"    11 frame-neutral W17 + {len(KEEP_USHAPE_R3)} R3-squared + "
          f"{len(btc_keep)} BTC + {len(btc_squared)} BTC-squared", flush=True)

    # ===== Train Ridge =====
    print(f"\nTraining Ridge across {len(ALL_FOLDS)} folds × {len(SEEDS)} seeds...",
          flush=True)
    all_preds = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t_fold = time.time()
        train, cal, test = _slice(X, folds_all[fid])
        tr = train[train["autocorr_pctile_7d"] >= AUTO_THRESH].dropna(subset=["target_z"])
        te = test.dropna(subset=["target_z"]).copy()
        if len(tr) < 1000 or len(te) < 100: continue
        Xt = tr[feat_cols].to_numpy(np.float32)
        Xte = te[feat_cols].to_numpy(np.float32)
        yt = tr["target_z"].to_numpy(np.float32)
        mt = ~np.isnan(yt)
        fold_preds = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, mt.sum(), size=mt.sum())
            m = RidgeCV(alphas=ALPHAS, scoring="r2", cv=None, fit_intercept=True)
            m.fit(Xt[mt][idx], yt[mt][idx])
            fold_preds.append(m.predict(Xte).astype(np.float32))
        pred = np.mean(fold_preds, axis=0)
        df_pred = te[["symbol","open_time","alpha_beta"]].copy()
        # exit_time needed
        df_pred = df_pred.merge(panel[["symbol","open_time","exit_time","return_pct"]],
                                  on=["symbol","open_time"], how="left")
        df_pred["pred_z"] = pred
        df_pred["fold"] = fid
        all_preds.append(df_pred)
        cyc_ic = df_pred.dropna(subset=["alpha_beta"]).groupby("open_time").apply(
            lambda g: g["pred_z"].rank().corr(g["alpha_beta"].rank())
            if len(g) >= 5 else np.nan).dropna()
        print(f"  fold {fid}: IC={cyc_ic.mean():+.4f}, n={len(te):,}, "
              f"{time.time()-t_fold:.0f}s", flush=True)

    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time","symbol"])
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    apd["alpha_A"] = apd["alpha_beta"]
    apd.to_parquet(OUT / "ridge_r3_btc_preds.parquet", index=False)
    cyc_ic = apd.dropna(subset=["alpha_beta"]).groupby("open_time").apply(
        lambda g: g["pred_z"].rank().corr(g["alpha_beta"].rank())
        if len(g) >= 5 else np.nan).dropna()
    print(f"\nR3_BTC overall per-cycle IC: {cyc_ic.mean():+.4f}", flush=True)
    print(f"  vs R3 (with basket):              +0.0133", flush=True)
    print(f"  vs R7 (R3 + 9 new):               +0.0154", flush=True)
    print(f"  vs LGBM:                          +0.0162", flush=True)

    # ===== Backtest A and B variants =====
    listings = get_listings()
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

    print(f"\nComputing trailing IC...", flush=True)
    df_ic = compute_trailing_ic(apd, sampled_t, TRAILING_IC_DAYS)
    apd_full = apd.merge(df_ic, on=["symbol","open_time"], how="left")
    apd_full["trail_ic"] = apd_full["trail_ic"].fillna(0)
    apd_full["pred_A"] = apd_full["pred_z"]
    apd_full["pred_B"] = apd_full["pred_z"] * apd_full["trail_ic"]
    apd_full["pred"] = apd_full["pred_z"]
    universe = psl.build_rolling_ic_universe(apd_full, sampled_t, psl.TOP_N, elig_pit)
    alpha_wide = apd_full.pivot_table(index="open_time", columns="symbol",
                                        values="alpha_A", aggfunc="first").sort_index()

    results = []
    for label, col in [("A_baseline","pred_A"), ("B_IC_signed","pred_B")]:
        print(f"\n  ===== R3_BTC {label} =====", flush=True)
        apd_v = apd_full.copy(); apd_v["pred"] = apd_v[col]
        records = psl.run_production_protocol_save_sleeves(apd_v, universe)
        df_v = aggregate(records, alpha_wide)
        net = df_v["net_pnl_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe,
                                          block_size=7, n_boot=1000)
        end_eq = CAPITAL + net.sum()/1e4*CAPITAL
        r = {"label":label, "sharpe":sh, "ci_lo":lo, "ci_hi":hi,
             "end_eq":end_eq, "gross":float(df_v["gross_pnl_bps"].mean()),
             "cost":float(df_v["cost_bps"].mean()),
             "folds_pos":folds_positive(df_v), "maxDD":_max_dd(net),
             "n_traded":int(records["traded"].sum())}
        results.append(r)
        print(f"    Sharpe={sh:+.2f} [{lo:+.2f},{hi:+.2f}], end-eq=${end_eq:.2f}, "
              f"gross={r['gross']:+.2f}, cost={r['cost']:+.2f}, folds+={r['folds_pos']}/9",
              flush=True)
        df_v.to_csv(OUT / f"r3_btc_backtest_{label}.csv", index=False)

    print("\n" + "="*80, flush=True)
    print("  R3_BTC (pure BTC-frame) vs R3 (with basket features)", flush=True)
    print("="*80, flush=True)
    print(f"  {'variant':<25} {'Sharpe':>8}", flush=True)
    for r in results:
        print(f"  R3_BTC {r['label']:<18} {r['sharpe']:+8.2f}", flush=True)
    print(f"  R3 baseline (with basket):  +0.03", flush=True)
    print(f"  R3 + IC-signed (with basket): +0.15", flush=True)
    print(f"  R7 + IC-signed (R3 + 9 new): +1.60", flush=True)
    print(f"  LGBM production:             +0.74", flush=True)
    print(f"\n  Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
