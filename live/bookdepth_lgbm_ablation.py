"""RECHECK #1 (user: 'orderbook should add edge, did you use it wrong?'): my ablations were LINEAR RidgeCV. A tree
model can use non-linear/threshold/interaction effects a linear model can't. Test: pooled LGBM + sym_id (the v6 shape),
predicting z_res, V0 vs V0+imb (ewma/mean12/run + the raw imb1 level + fragility slope/asym), expanding walk-forward
within each era's covered period, rank-IC(pred, alpha) both eras + delta CI. If LGBM+L2 beats LGBM-V0 both eras, the
edge is there but needs a non-linear model. If not, the linear-null holds non-linearly too.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from live.bookdepth_persist import persist_feats
V0 = ["return_1d", "atr_pct", "vwap_slope_96", "bars_since_high", "autocorr_pctile_7d", "obv_z_1d", "corr_to_btc_1d",
      "beta_to_btc_change_5d", "idio_vol_to_btc_1h", "idio_vol_to_btc_1d", "funding_rate", "funding_rate_z_7d",
      "funding_rate_1d_change", "rvol_7d", "ret_3d", "btc_rvol_7d", "bars_since_high_xs_rank"]
L2FE = ["imb_ewma", "imb_mean12", "imb_run", "l2_imb1", "l2_slope", "l2_asym1", "l2_liq1"]
rng = np.random.default_rng(11)

def load():
    rows = []
    for f in [x for x in glob.glob("/home/yuqing/ctaNew/data/ml/cache/l2_*.parquet") if "BTCUSDT" not in x]:
        sym = Path(f).stem[3:]; d = pd.read_parquet(f)
        d.index = pd.to_datetime(d.index, utc=True) + pd.Timedelta("4h")
        pf = persist_feats(d["l2_imb1"].sort_index())[["imb_ewma", "imb_mean12", "imb_run"]]
        for c in ["l2_imb1", "l2_slope", "l2_asym1", "l2_liq1"]: pf[c] = d[c].reindex(pf.index)
        pf["symbol"] = sym; pf["open_time"] = pf.index; rows.append(pf.reset_index(drop=True))
    L = pd.concat(rows, ignore_index=True)
    pan = pd.read_parquet("/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet",
                          columns=["symbol", "open_time", "alpha_vs_btc_realized"] + V0)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    m = pan.merge(L, on=["symbol", "open_time"], how="inner")
    g = m.groupby("open_time"); sd = g["alpha_vs_btc_realized"].transform("std").replace(0, np.nan)
    m["z_res"] = ((m["alpha_vs_btc_realized"] - g["alpha_vs_btc_realized"].transform("mean")) / sd).clip(-10, 10)
    m["sym_id"] = m["symbol"].astype("category").cat.codes
    return m.dropna(subset=["z_res"])

def wf(sub, feats, folds=3):
    sub = sub.sort_values("open_time").reset_index(drop=True)
    bars = pd.DatetimeIndex(sub["open_time"]).unique().sort_values()
    sp = np.array_split(np.arange(len(bars)), folds + 1); preds = []
    fe = feats + ["sym_id"]
    for k in range(1, folds + 1):
        tre, tee = bars[sp[k - 1][-1]], bars[sp[k][-1]]
        tr = sub[sub.open_time <= tre]; te = sub[(sub.open_time > tre) & (sub.open_time <= tee)]
        if len(tr) < 5000 or len(te) < 500: continue
        m = LGBMRegressor(n_estimators=300, num_leaves=15, learning_rate=0.03, min_child_samples=100,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)
        m.fit(tr[fe], tr["z_res"].values, categorical_feature=["sym_id"])
        p = te[["open_time", "alpha_vs_btc_realized"]].copy(); p["pred"] = m.predict(te[fe]); preds.append(p)
    if not preds: return None
    P = pd.concat(preds)
    return P.groupby("open_time").apply(lambda g: spearmanr(g["pred"], g["alpha_vs_btc_realized"]).correlation if len(g) >= 5 else np.nan).dropna()

def main():
    m = load()
    cut = pd.Timestamp("2025-10-01", tz="UTC"); eras = {"RECENT": m[m.open_time >= cut], "OOS": m[m.open_time < cut]}
    print(f"pooled LGBM + sym_id (v6 shape) | RECENT {len(eras['RECENT'])} OOS {len(eras['OOS'])} rows\n")
    for era, sub in eras.items():
        ib = wf(sub, V0); ia = wf(sub, V0 + L2FE)
        print(f"### {era} ###")
        print(f"  LGBM V0            rank-IC {ib.mean():+.4f}")
        print(f"  LGBM V0 + L2 (7)   rank-IC {ia.mean():+.4f}")
        j = pd.concat([ib.rename("a"), ia.rename("b")], axis=1).dropna(); j["d"] = j["b"] - j["a"]
        j["day"] = pd.to_datetime(j.index, utc=True).floor("1D"); gg = [x["d"].values for _, x in j.groupby("day")]
        boot = [np.concatenate([gg[k] for k in rng.integers(0, len(gg), len(gg))]).mean() for _ in range(3000)]
        lo, up = np.percentile(boot, [2.5, 97.5])
        flag = "ADDS (CI>0)" if lo > 0 else ("HURTS (CI<0)" if up < 0 else "within noise")
        print(f"  Δ(+L2) {j['d'].mean():+.4f} [{lo:+.4f},{up:+.4f}] -> {flag}\n")
    print("LGBMABLDONE")

if __name__ == "__main__":
    main()
