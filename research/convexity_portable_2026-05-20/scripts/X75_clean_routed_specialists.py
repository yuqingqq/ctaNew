"""X75 — Clean cluster-routed regime specialists (proper x6 winsorized preprocessing).

Replaces X74's simplified train_v0 with the canonical x6.train_per_sym_ridge
(winsorization + rank-transform for heavy-tail + target_z.notna + exit_time purge).

Method (all PIT):
  1. PIT KMeans K=4 regime classifier on trailing BTC features (same as X74)
  2. For each cluster c: run x6.train_per_sym_ridge on panel[cluster==c] with GLOBAL
     folds → trains per-sym on cluster-c train rows, predicts cluster-c OOS rows
  3. Concatenate all clusters' OOS preds → routed predictions
  4. Sleeve, compare to proper single V0 (+0.12) and hard bull-filter (+1.13)

Also tests K=3 and K=5 for robustness.
"""
from __future__ import annotations
import sys, importlib.util, time
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scripts"))
OUT = REPO/"research/convexity_portable_2026-05-20/results"; CACHE = OUT/"_cache"
KLINES = REPO/"data/ml/test/parquet/klines"
spec = importlib.util.spec_from_file_location("x6", REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)


def btc_regime_features():
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
        "btc_dist_ma": btc/btc.rolling(57600, min_periods=2880).mean()-1,
    })
    return feat.reset_index()


def routed_predict(panel, folds, feats, K, label):
    """KMeans K regimes (PIT) + per-cluster x6.train_per_sym_ridge, concatenate."""
    rfeat = ["btc_ret_7d","btc_ret_30d","btc_ret_90d","btc_rvol_30d","btc_dist_ma"]
    cyc = panel.groupby("open_time")[rfeat].first().dropna()
    Z = StandardScaler().fit_transform(cyc.values)
    km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(Z)
    cyc["cluster"] = km.labels_
    cmap = cyc["cluster"].reset_index()
    p = panel.merge(cmap, on="open_time", how="left")
    parts = []
    for c in range(K):
        pc = p[p["cluster"] == c]
        if pc["open_time"].nunique() < 50: continue
        try:
            apd_c = x6.train_per_sym_ridge(pc, folds, feats, label=f"{label}_c{c}")
            parts.append(apd_c)
        except Exception as e:
            print(f"    cluster {c} train err: {e}")
    if not parts: return None
    return pd.concat(parts, ignore_index=True).sort_values(["open_time","symbol"])


def main():
    t0 = time.time()
    print("=== X75 clean cluster-routed specialists (proper preprocessing) ===\n", flush=True)
    panel = pd.read_parquet(REPO/"outputs/vBTC_features/panel_3yr_v0.parquet")
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    if "target_z" not in panel.columns: panel = x6.build_target_z(panel)
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")
    if "bars_since_high_xs_rank" not in panel.columns:
        panel["bars_since_high_xs_rank"] = panel.groupby("open_time")["bars_since_high"].rank(pct=True).astype("float32")
    panel = panel.merge(btc_regime_features(), on="open_time", how="left")
    feats = [f for f in x6.BASE + x6.COHORT_EXTRAS if f in panel.columns]
    folds = x6.get_folds(panel)
    print(f"Panel {len(panel):,} rows × {panel['symbol'].nunique()} syms; feats={len(feats)}\n")

    # Proper single V0 baseline (reproduce X70 +0.12)
    apd0 = x6.train_per_sym_ridge(panel, folds, feats, label="x75_singleV0")
    apd0.to_parquet(CACHE/"x75_singleV0_preds.parquet", index=False)
    m0 = x6.run_sleeve_on_preds(CACHE/"x75_singleV0_preds.parquet", "x75_singleV0")
    print(f"Single V0 (proper, expanding): Sharpe={m0.get('sharpe',0):+.2f} folds={m0.get('folds_pos','?')} conc={m0.get('concentration','?')}\n")

    print(f"{'K':<4} {'Sharpe':>8} {'folds+':>8} {'conc':>8} {'PnL':>9}")
    for K in [3, 4, 5]:
        tf = time.time()
        apd = routed_predict(panel, folds, feats, K, f"x75_K{K}")
        if apd is None: print(f"{K:<4} FAILED"); continue
        pth = CACHE/f"x75_routed_K{K}_preds.parquet"; apd.to_parquet(pth, index=False)
        m = x6.run_sleeve_on_preds(pth, f"x75_routed_K{K}")
        sh = m.get("sharpe",0) or 0
        print(f"{K:<4} {sh:>+8.2f} {str(m.get('folds_pos','?')):>8} {str(m.get('concentration','?')):>8} "
              f"{str(m.get('totPnL','?')):>9}  [{time.time()-tf:.0f}s]", flush=True)

    print(f"\nReference: proper single V0 +0.12; hard bull-filter@0.10 +1.13; X74 routed (simplified) +1.14")
    print(f"Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
