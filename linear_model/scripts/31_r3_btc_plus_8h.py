"""Step 31: Pure BTC-frame + short-horizon features.

Combines:
  - R3_BTC structure (basket features REPLACED by BTC analogs)
  - + return_8h (short-horizon momentum, |sp| 0.039 monotonic)
  - + vol_zscore_4h_over_7d (4h volume z, |sp| 0.022 monotonic)

= 11 frame-neutral W17 + 3 R3-squared U-shape kept
+ 4 BTC replacements (dom_btc_z_1d, dom_btc_change_288b, corr_to_btc_change_3d,
   idio_vol_to_btc_1d) + 2 BTC-squared
+ 2 new short-horizon
= 22 features, FULLY BTC-frame at target-aligned horizons

Validate with corrected pipeline + LOFO + P2 placebo.
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
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
PANEL_BTC = REPO / "outputs/vBTC_features_btc_only/panel_btc_only_clean.parquet"
STEP29 = REPO / "linear_model/results/step29_features.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT = REPO / "linear_model/results"

# Frame-neutral W17 (no basket, no sym_id)
FRAME_NEUTRAL = ["return_1d","atr_pct","obv_z_1d","vwap_slope_96",
                 "bars_since_high_xs_rank","funding_rate","funding_rate_z_7d",
                 "funding_rate_1d_change",
                 "corr_to_btc_1d", "idio_vol_to_btc_1h", "beta_to_btc_change_5d"]

# R3 squared for U-shape that survive
KEEP_USHAPE_R3 = ["beta_to_btc_change_5d", "corr_to_btc_1d", "return_1d"]

# BTC replacements (from Step 19 audit — these have valid monotonic/u-shape)
BTC_KEEP = ["dom_btc_z_1d", "dom_btc_change_288b", "corr_to_btc_change_3d",
            "idio_vol_to_btc_1d"]
BTC_USHAPE = ["dom_btc_change_288b", "corr_to_btc_change_3d"]

# NEW short-horizon
NEW_SHORT = ["return_8h", "vol_zscore_4h_over_7d"]

ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
SEEDS = (42, 1337, 7, 19, 2718)
AUTO_THRESH = 0.5
ALL_FOLDS = list(range(10))
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
CAPITAL = 100.0
TRAILING_IC_DAYS = 90
HOLD_BARS = 288
N_PLACEBO = 100


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


def aggregate_hold_through(records, alpha_wide):
    sleeve_queue = deque(maxlen=psl.N_SLEEVES)
    prev_weights = {}
    bar_freq = pd.Timedelta(minutes=5)
    rows = []
    for _, rec in records.iterrows():
        t = rec["time"]; fold = rec["fold"]
        if rec["traded"]:
            sleeve_queue.append({"entry_time":t, "longs":list(rec["long_basket"]),
                                  "shorts":list(rec["short_basket"])})
        max_age = HOLD_BARS * bar_freq
        sleeve_queue = deque(
            [s for s in sleeve_queue if (t - s["entry_time"]) < max_age],
            maxlen=psl.N_SLEEVES)
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
    print("=== Step 31: Pure BTC-frame + short-horizon ===\n", flush=True)
    t0 = time.time()
    listings = get_listings()

    tgt = pd.read_parquet(TARGETS)
    tgt["open_time"] = pd.to_datetime(tgt["open_time"], utc=True)
    cols_base = list(set(FRAME_NEUTRAL + KEEP_USHAPE_R3))
    base = pd.read_parquet(PANEL, columns=["symbol","open_time"] + cols_base)
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
    btc_panel = pd.read_parquet(PANEL_BTC,
                                  columns=["symbol","open_time"] + BTC_KEEP)
    btc_panel["open_time"] = pd.to_datetime(btc_panel["open_time"], utc=True)
    step29 = pd.read_parquet(STEP29, columns=["symbol","open_time"] + NEW_SHORT)
    step29["open_time"] = pd.to_datetime(step29["open_time"], utc=True)
    panel = tgt.merge(base, on=["symbol","open_time"], how="left")
    panel = panel.merge(btc_panel, on=["symbol","open_time"], how="left")
    panel = panel.merge(step29, on=["symbol","open_time"], how="left")
    print(f"Panel: {len(panel):,} rows", flush=True)

    folds_all = _multi_oos_splits(panel)
    train_mask = panel["open_time"].between(
        _slice(panel, folds_all[0])[0].open_time.min(),
        _slice(panel, folds_all[0])[0].open_time.max())
    train_panel = panel[train_mask]

    # Build features
    X = pd.DataFrame({"symbol": panel["symbol"], "open_time": panel["open_time"],
                      "alpha_beta": panel["alpha_beta"],
                      "target_z": panel["target_z"],
                      "autocorr_pctile_7d": panel["autocorr_pctile_7d"]})
    for f in FRAME_NEUTRAL:
        X[f] = winsorize_zscore(panel[f], train_panel[f])
    for f in KEEP_USHAPE_R3:
        X[f + "_sq"] = (X[f] ** 2).astype("float32")
    for f in BTC_KEEP:
        X[f] = winsorize_zscore(panel[f], train_panel[f])
    for f in BTC_USHAPE:
        X[f + "_sq"] = (X[f] ** 2).astype("float32")
    for f in NEW_SHORT:
        X[f] = winsorize_zscore(panel[f], train_panel[f])

    feat_cols = [c for c in X.columns if c not in
                  ("symbol","open_time","alpha_beta","target_z","autocorr_pctile_7d")]
    X[feat_cols] = X[feat_cols].fillna(0)
    print(f"R3_BTC_8h_VZ features: {len(feat_cols)}", flush=True)
    print(f"  Composition: 11 frame-neutral + 3 R3-sq + 4 BTC + 2 BTC-sq + 2 short",
          flush=True)

    # Train
    print("\nTraining...", flush=True)
    all_preds = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t_fold = time.time()
        train_, cal, test = _slice(X, folds_all[fid])
        tr = train_[train_["autocorr_pctile_7d"] >= AUTO_THRESH].dropna(subset=["target_z"])
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
    apd.to_parquet(OUT / "ridge_r3_btc_8h_vz_preds.parquet", index=False)
    cyc_ic = apd.dropna(subset=["alpha_beta"]).groupby("open_time").apply(
        lambda g: g["pred_z"].rank().corr(g["alpha_beta"].rank())
        if len(g) >= 5 else np.nan).dropna()
    print(f"\nR3_BTC_8h_VZ IC: {cyc_ic.mean():+.4f}", flush=True)

    # Backtest
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

    df_ic = compute_trailing_ic(apd, sampled_t, TRAILING_IC_DAYS)
    apd_full = apd.merge(df_ic, on=["symbol","open_time"], how="left")
    apd_full["trail_ic"] = apd_full["trail_ic"].fillna(0)
    apd_full["pred_B"] = apd_full["pred_z"] * apd_full["trail_ic"]
    apd_full["pred"] = apd_full["pred_z"]
    universe = psl.build_rolling_ic_universe(apd_full, sampled_t, psl.TOP_N, elig_pit)
    alpha_wide = apd_full.pivot_table(index="open_time", columns="symbol",
                                        values="alpha_A", aggfunc="first").sort_index()

    print(f"\n{'='*90}", flush=True)
    print(f"  BACKTEST", flush=True)
    print(f"{'='*90}", flush=True)
    sh_main = None; df_v_main = None
    for label, col in [("A_baseline","pred_z"), ("B_IC_signed","pred_B")]:
        apd_v = apd_full.copy(); apd_v["pred"] = apd_v[col]
        records = psl.run_production_protocol_save_sleeves(apd_v, universe)
        df_v = aggregate_hold_through(records, alpha_wide)
        net = df_v["net_pnl_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe,
                                          block_size=7, n_boot=1000)
        end_eq = CAPITAL + net.sum()/1e4*CAPITAL
        print(f"  R3_BTC_8h_VZ {label}: Sharpe={sh:+.2f} [{lo:+.2f},{hi:+.2f}], "
              f"end-eq=${end_eq:.2f}, gross={df_v['gross_pnl_bps'].mean():+.2f}, "
              f"folds+={folds_positive(df_v)}/9", flush=True)
        df_v.to_csv(OUT / f"r3_btc_8h_vz_{label}.csv", index=False)
        if label == "B_IC_signed":
            sh_main = sh; df_v_main = df_v

    # LOFO
    print(f"\n  LOFO on B_IC_signed (Sharpe = {sh_main:+.2f}):", flush=True)
    print(f"    {'exclude':>9}  {'rem Sharpe':>11}  {'Delta':>7}  {'fold pnl':>10}",
          flush=True)
    for excl in range(1, 10):
        rem = df_v_main[df_v_main["fold"] != excl]["net_pnl_bps"].to_numpy()
        fold_pnl = df_v_main[df_v_main["fold"] == excl]["net_pnl_bps"].sum()
        sh_rem = _sharpe(rem)
        delta = sh_rem - sh_main
        flag = "  ← drives lift" if delta < -0.4 else ""
        print(f"    {excl:>9}  {sh_rem:>+11.2f}  {delta:>+7.2f}  {fold_pnl:>+10.0f}{flag}",
              flush=True)

    # P2 placebo
    print(f"\n--- P2 placebo ({N_PLACEBO} seeds) ---", flush=True)
    p2 = []
    for seed in range(N_PLACEBO):
        records_p = psl.run_production_protocol_save_sleeves(
            apd_full, universe, placebo_seed=seed)
        df_v_p = aggregate_hold_through(records_p, alpha_wide)
        p2.append(_sharpe(df_v_p["net_pnl_bps"].to_numpy()))
        if (seed+1) % 25 == 0:
            print(f"  seed {seed+1}/{N_PLACEBO}: mean={np.mean(p2):+.3f}", flush=True)
    p2 = np.array(p2)
    p5,p25,p50,p75,p95 = np.percentile(p2, [5,25,50,75,95])
    print(f"\n  P2 mean: {p2.mean():+.3f}  std: {p2.std():.3f}", flush=True)
    print(f"  p5/p25/p50/p75/p95: {p5:+.2f}/{p25:+.2f}/{p50:+.2f}/"
          f"{p75:+.2f}/{p95:+.2f}", flush=True)
    print(f"  R3_BTC_8h_VZ rank: {(p2 < sh_main).mean()*100:.1f}%", flush=True)
    print(f"  Edge over p95: {sh_main - p95:+.2f}", flush=True)
    pd.DataFrame({"placebo":p2}).to_csv(OUT / "r3_btc_8h_vz_placebo.csv", index=False)

    print(f"\nReferences:", flush=True)
    print(f"  R3 corrected + IC-signed (basket+R3):       +0.86 (folds 1+2)", flush=True)
    print(f"  R3_BTC + IC-signed (no basket, R3 base):    +1.92 (fold 6 fragile)",
          flush=True)
    print(f"  R3_8h_VZ + IC-signed (basket+R3+2 new):     +0.34", flush=True)
    print(f"  R3_BTC_8h_VZ + IC-signed (THIS):            {sh_main:+.2f}",
          flush=True)
    print(f"  LGBM production:                            +0.74", flush=True)
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
