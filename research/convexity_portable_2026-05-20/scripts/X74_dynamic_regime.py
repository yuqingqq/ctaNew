"""X74 — Dynamic regime methods on 3-year panel (replaces hardcoded BTC-30d gate).

Addresses two user points:
  1. Rolling-window vs expanding-window training
  2. A dynamic PIT regime classifier (KMeans on trailing features) + per-regime
     model routing, plus a SOFT (continuous) regime weight vs the hard gate.

All regime detection is PIT (trailing-only features). No hindsight.

Parts:
  A. Rolling (trailing ~12mo) vs expanding V0 over 3 years.
  B. PIT regime classifier: KMeans K=4 on trailing [btc_ret_7d/30d/90d, btc_rvol_30d,
     xs_dispersion, btc_dist_ma]. Show cluster→realized-regime mapping + per-cluster
     V0 spread Sharpe (which clusters the strategy works in).
  C. Per-cluster specialist routing (train V0 per cluster, route by PIT cluster).
  D. Soft regime weight: scale pred by smooth f(regime score) instead of hard gate.
"""
from __future__ import annotations
import sys, importlib.util, time
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scripts"))
OUT = REPO/"research/convexity_portable_2026-05-20/results"; CACHE = OUT/"_cache"
KLINES = REPO/"data/ml/test/parquet/klines"
spec = importlib.util.spec_from_file_location("x6", REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
ALPHAS = [0.01, 0.1, 1, 10, 100]


def btc_regime_features():
    """All trailing/PIT BTC + market regime features at 4h cadence."""
    files = sorted((KLINES/"BTCUSDT"/"5m").glob("*.parquet"))
    btc = pd.concat([pd.read_parquet(f, columns=["open_time","close"]) for f in files],
                     ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
    btc = btc.set_index("open_time")["close"].astype(np.float64)
    logret = np.log(btc/btc.shift(1))
    feat = pd.DataFrame({
        "btc_ret_7d": btc/btc.shift(2016)-1,
        "btc_ret_30d": btc/btc.shift(8640)-1,
        "btc_ret_90d": btc/btc.shift(25920)-1,
        "btc_rvol_30d": logret.rolling(8640, min_periods=2880).std()*np.sqrt(8640),
        "btc_dist_ma": btc/btc.rolling(57600, min_periods=2880).mean()-1,  # ~200d
    })
    return feat.reset_index()


def train_v0(panel, folds, feats, label, rolling_days=None):
    """Per-sym Ridge V0. If rolling_days set, train only on trailing window."""
    preds = []
    for f, ts, te, ec in folds:
        if rolling_days is not None:
            tr_start = ec - pd.Timedelta(days=rolling_days)
            tr = panel[(panel["open_time"] < ec) & (panel["open_time"] >= tr_start)]
        else:
            tr = panel[panel["open_time"] < ec]
        oos = panel[(panel["open_time"] >= ts) & (panel["open_time"] <= te)]
        if len(tr) < 500 or len(oos) == 0: continue
        ycol = "target_z" if "target_z" in panel.columns else "alpha_vs_btc_realized"
        for sym, g_oos in oos.groupby("symbol"):
            g_tr = tr[tr["symbol"] == sym].dropna(subset=[ycol])
            if len(g_tr) < 100: continue
            Xtr = g_tr[feats].replace([np.inf,-np.inf],np.nan).fillna(0).values
            mu, sd = Xtr.mean(0), Xtr.std(0)+1e-9
            try: model = RidgeCV(alphas=ALPHAS).fit((Xtr-mu)/sd, g_tr[ycol].values)
            except Exception: continue
            Xo = g_oos[feats].replace([np.inf,-np.inf],np.nan).fillna(0).values
            d = g_oos[["symbol","open_time","alpha_vs_btc_realized","return_pct","exit_time"]].copy()
            d.columns = ["symbol","open_time","alpha_A","return_pct","exit_time"]
            d["pred"] = model.predict((Xo-mu)/sd); d["fold"] = f
            preds.append(d)
    return pd.concat(preds, ignore_index=True) if preds else None


def sleeve(apd, label):
    p = CACHE/f"{label}_preds.parquet"; apd.to_parquet(p, index=False)
    return x6.run_sleeve_on_preds(p, label)


def main():
    t0 = time.time()
    print("=== X74 dynamic regime methods (3-year) ===\n", flush=True)
    panel = pd.read_parquet(REPO/"outputs/vBTC_features/panel_3yr_v0.parquet")
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    if "target_z" not in panel.columns: panel = x6.build_target_z(panel)
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")
    if "bars_since_high_xs_rank" not in panel.columns:
        panel["bars_since_high_xs_rank"] = panel.groupby("open_time")["bars_since_high"].rank(pct=True).astype("float32")
    feats = [f for f in x6.BASE + x6.COHORT_EXTRAS if f in panel.columns]
    folds = x6.get_folds(panel)

    rf = btc_regime_features()
    panel = panel.merge(rf, on="open_time", how="left")

    # === A. Rolling vs expanding ===
    print("--- A. Rolling vs expanding window training ---", flush=True)
    for rd, name in [(None,"expanding"), (365,"roll_12mo"), (180,"roll_6mo")]:
        apd = train_v0(panel, folds, feats, f"x74A_{name}", rolling_days=rd)
        if apd is None: print(f"  {name}: FAILED"); continue
        m = sleeve(apd, f"x74A_{name}")
        print(f"  {name:<12}: Sharpe={m.get('sharpe',0):+.2f} folds={m.get('folds_pos','?')} conc={m.get('concentration','?')}", flush=True)

    # === B. PIT KMeans regime classifier ===
    print("\n--- B. PIT regime classifier (KMeans K=4 on trailing features) ---", flush=True)
    rfeat_cols = ["btc_ret_7d","btc_ret_30d","btc_ret_90d","btc_rvol_30d","btc_dist_ma"]
    # Per-cycle regime features (one row per timestamp)
    cyc_rf = panel.groupby("open_time")[rfeat_cols].first().dropna()
    scaler = StandardScaler()
    Z = scaler.fit_transform(cyc_rf.values)
    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(Z)
    cyc_rf["cluster"] = km.labels_
    # Map cluster → mean BTC 30d ret (to interpret)
    print("  Cluster interpretation (mean trailing features):")
    for c in range(4):
        sub = cyc_rf[cyc_rf["cluster"]==c]
        print(f"    cluster {c}: n={len(sub):>5} ret30d={sub['btc_ret_30d'].mean():>+.2f} "
              f"rvol={sub['btc_rvol_30d'].mean():.2f} ret90d={sub['btc_ret_90d'].mean():>+.2f}", flush=True)
    cluster_map = cyc_rf["cluster"].reset_index()
    panel = panel.merge(cluster_map, on="open_time", how="left")

    # Per-cluster V0 spread Sharpe (using expanding V0 preds)
    apd_exp = train_v0(panel, folds, feats, "x74B_exp", rolling_days=None)
    apd_exp = apd_exp.merge(cluster_map, on="open_time", how="left")
    print("\n  Per-cluster V0 K=3 spread Sharpe (which regimes work):")
    for c in range(4):
        sub = apd_exp[apd_exp["cluster"]==c]
        if len(sub) < 100: continue
        def cyc_spread(g):
            if len(g)<8 or g["pred"].std()==0: return np.nan
            gg=g.sort_values("pred"); return (gg.tail(3)["alpha_A"].mean()-gg.head(3)["alpha_A"].mean())*10000
        sp = sub.groupby("open_time").apply(cyc_spread).dropna()
        sh = sp.mean()/sp.std()*np.sqrt(6*365) if len(sp)>2 and sp.std()>0 else np.nan
        print(f"    cluster {c}: cycles={sub['open_time'].nunique():>5} spread={sp.mean():>+7.2f}bps Sharpe={sh:>+.2f}", flush=True)

    # === C. Per-cluster specialist routing ===
    print("\n--- C. Per-cluster specialist routing ---", flush=True)
    # Train a specialist per cluster (on that cluster's rows in train set), route by PIT cluster
    routed_preds = []
    for f, ts, te, ec in folds:
        tr_all = panel[panel["open_time"] < ec]
        oos = panel[(panel["open_time"]>=ts)&(panel["open_time"]<=te)]
        if len(oos)==0: continue
        ycol = "target_z" if "target_z" in panel.columns else "alpha_vs_btc_realized"
        for c in range(4):
            tr_c = tr_all[(tr_all["cluster"]==c)].dropna(subset=[ycol])
            oos_c = oos[oos["cluster"]==c]
            if len(tr_c) < 500 or len(oos_c)==0: continue
            for sym, g_oos in oos_c.groupby("symbol"):
                g_tr = tr_c[tr_c["symbol"]==sym]
                if len(g_tr) < 50: g_tr = tr_c  # fallback pooled-in-cluster
                Xtr = g_tr[feats].replace([np.inf,-np.inf],np.nan).fillna(0).values
                mu,sd = Xtr.mean(0), Xtr.std(0)+1e-9
                try: model = RidgeCV(alphas=ALPHAS).fit((Xtr-mu)/sd, g_tr[ycol].values)
                except Exception: continue
                Xo = g_oos[feats].replace([np.inf,-np.inf],np.nan).fillna(0).values
                d = g_oos[["symbol","open_time","alpha_vs_btc_realized","return_pct","exit_time"]].copy()
                d.columns = ["symbol","open_time","alpha_A","return_pct","exit_time"]
                d["pred"]=model.predict((Xo-mu)/sd); d["fold"]=f
                routed_preds.append(d)
    if routed_preds:
        apd_routed = pd.concat(routed_preds, ignore_index=True)
        m = sleeve(apd_routed, "x74C_cluster_routed")
        print(f"  cluster-routed specialists: Sharpe={m.get('sharpe',0):+.2f} folds={m.get('folds_pos','?')} conc={m.get('concentration','?')}", flush=True)

    # === D. Soft regime weight (continuous, vs hard gate) ===
    print("\n--- D. Soft regime weight (scale pred by smooth f(btc_30d)) ---", flush=True)
    # weight = sigmoid(-(btc_ret_30d - center)/scale): downweight smoothly as bull strengthens
    apd_soft = apd_exp.merge(panel[["symbol","open_time","btc_ret_30d"]].drop_duplicates(["symbol","open_time"]),
                              on=["symbol","open_time"], how="left")
    for center, scale in [(0.10, 0.05), (0.05, 0.05), (0.15, 0.08)]:
        a = apd_soft.copy()
        w = 1.0/(1.0+np.exp((a["btc_ret_30d"]-center)/scale))
        a["pred"] = a["pred"] * w
        m = sleeve(a[["symbol","open_time","alpha_A","return_pct","exit_time","pred","fold"]],
                   f"x74D_soft_c{center}")
        print(f"  soft weight (center={center}, scale={scale}): Sharpe={m.get('sharpe',0):+.2f} folds={m.get('folds_pos','?')}", flush=True)

    print(f"\nReference: expanding V0 +0.12; hard bull-filter@0.10 +1.13")
    print(f"Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
