"""PROPER incremental test via the strategy's actual model shape: PER-SYMBOL RidgeCV walk-forward (each name's own
features predict its own forward residual-vs-BTC), then CROSS-SECTIONAL rank-IC(pred, alpha) per bar. This reproduces
the strategy's +0.03 rank-IC (pooled cross-sectional Ridge did not — it inverted). Compare V0 vs V0+imb_ewma vs
V0+[ewma,mean12,run], both eras. VALID only if the V0 baseline reproduces ~+0.02-0.03; then the delta is meaningful.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr
from live.bookdepth_persist import persist_feats
V0 = ["return_1d", "atr_pct", "vwap_slope_96", "bars_since_high", "autocorr_pctile_7d", "obv_z_1d", "corr_to_btc_1d",
      "beta_to_btc_change_5d", "idio_vol_to_btc_1h", "idio_vol_to_btc_1d", "funding_rate", "funding_rate_z_7d",
      "funding_rate_1d_change", "rvol_7d", "ret_3d", "btc_rvol_7d", "bars_since_high_xs_rank"]
TGT = "alpha_vs_btc_realized"

def load():
    rows = []
    for f in [x for x in glob.glob("/home/yuqing/ctaNew/data/ml/cache/l2_*.parquet") if "BTCUSDT" not in x]:
        sym = Path(f).stem[3:]
        d0 = pd.read_parquet(f)[["l2_imb1"]]; d0.index = pd.to_datetime(d0.index, utc=True) + pd.Timedelta("4h")
        pf = persist_feats(d0["l2_imb1"].sort_index())[["imb_ewma", "imb_mean12", "imb_run"]]
        pf["symbol"] = sym; pf["open_time"] = pf.index; rows.append(pf.reset_index(drop=True))
    m = pd.concat(rows, ignore_index=True)
    pan = pd.read_parquet("/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet",
                          columns=["symbol", "open_time", TGT] + V0)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    return m.merge(pan, on=["symbol", "open_time"], how="inner")

def persym_preds(df, feats, folds=3):
    """per-symbol RidgeCV, expanding walk-forward; returns preds over the test portion."""
    out = []
    for sym, g in df.groupby("symbol"):
        g = g.dropna(subset=[TGT] + feats).sort_values("open_time")
        if len(g) < 400: continue
        bars = g["open_time"].values; n = len(g)
        for k in range(1, folds + 1):
            lo = int(n * (0.4 + 0.2 * (k - 1))); hi = int(n * (0.4 + 0.2 * k))
            tr = g.iloc[:lo]; te = g.iloc[lo:hi]
            if len(tr) < 250 or len(te) < 20: continue
            mu = tr[feats].mean(); sd = tr[feats].std().replace(0, 1)
            Xtr = ((tr[feats] - mu) / sd).values; Xte = ((te[feats] - mu) / sd).values
            m = RidgeCV(alphas=[1, 10, 100, 1000]).fit(Xtr, tr[TGT].values)
            p = te[["open_time", TGT]].copy(); p["pred"] = m.predict(Xte); out.append(p)
    return pd.concat(out) if out else None

def rank_ic(P):
    ics = [spearmanr(g["pred"], g[TGT]).correlation for _, g in P.groupby("open_time") if len(g) >= 5]
    return np.nanmean(ics)

def perbar_ic(P):
    return P.groupby("open_time").apply(lambda g: spearmanr(g["pred"], g[TGT]).correlation if len(g) >= 5 else np.nan).dropna()

def delta_ci(Pbase, Padd):
    a = perbar_ic(Pbase); b = perbar_ic(Padd); j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    j["d"] = j["b"] - j["a"]; j["day"] = pd.to_datetime(j.index, utc=True).floor("1D")
    g = [x["d"].values for _, x in j.groupby("day")]
    rng = np.random.default_rng(3)
    out = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(3000)]
    return j["d"].mean(), tuple(np.percentile(out, [2.5, 97.5]))

def main():
    m = load()
    cut = pd.Timestamp("2025-10-01", tz="UTC"); eras = {"RECENT": m[m.open_time >= cut], "OOS": m[m.open_time < cut]}
    print(f"per-symbol RidgeCV walk-forward | RECENT {len(eras['RECENT'])} OOS {len(eras['OOS'])} rows")
    print("VALID only if V0 baseline reproduces ~+0.02-0.03 (the strategy's honest rank-IC)\n")
    ADD = ["imb_ewma", "imb_mean12", "imb_run"]
    print(f"  {'feature set':22s} | RECENT rank-IC | OOS rank-IC")
    P = {}
    for name, feats in [("V0 (baseline)", V0), ("V0 + imb_ewma", V0 + ["imb_ewma"]), ("V0 + ewma+mean12+run", V0 + ADD)]:
        Pr = persym_preds(eras["RECENT"], feats); Po = persym_preds(eras["OOS"], feats); P[name] = (Pr, Po)
        ric = rank_ic(Pr) if Pr is not None else np.nan; oic = rank_ic(Po) if Po is not None else np.nan
        print(f"  {name:22s} | {ric:+.4f}       | {oic:+.4f}")
    print("\n  DELTA (V0+ewma+mean12+run) - V0, day-clustered bootstrap CI:")
    for era, idx in [("RECENT", 0), ("OOS", 1)]:
        d, (lo, up) = delta_ci(P["V0 (baseline)"][idx], P["V0 + ewma+mean12+run"][idx])
        flag = "adds (CI>0)" if lo > 0 else ("hurts (CI<0)" if up < 0 else "within noise (CI~0)")
        print(f"    {era}: Δrank-IC {d:+.4f} [{lo:+.4f},{up:+.4f}] -> {flag}")
    print("PSABLDONE")

if __name__ == "__main__":
    main()
