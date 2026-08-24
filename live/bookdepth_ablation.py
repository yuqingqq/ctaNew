"""Rigorous redundancy test for sustained imbalance (answering 'is it PROPERLY tested?'). Two tests beyond the linear
partial-IC:
 (A) OVERLAP: regress imb_ewma on the 17 V0 features -> R2 (how much of imb_ewma V0 explains) + the top-|corr| V0
     features (WHICH ones absorb it). Both eras. Shows the mechanism instead of asserting it.
 (B) MODEL ABLATION (gold standard): pooled RidgeCV predicting target_z, EXPANDING walk-forward within each era,
     V0 vs V0+imb_ewma vs V0+[ewma,mean12,run]. Compare OOS cross-sectional rank-IC. If adding imb_ewma lifts rank-IC
     in BOTH eras it's NOT redundant; if no lift, redundant confirmed at the model level (not just linearly).
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeCV, LinearRegression
from live.bookdepth_persist import persist_feats
rng = np.random.default_rng(31)
V0 = ["return_1d", "atr_pct", "vwap_slope_96", "bars_since_high", "autocorr_pctile_7d", "obv_z_1d", "corr_to_btc_1d",
      "beta_to_btc_change_5d", "idio_vol_to_btc_1h", "idio_vol_to_btc_1d", "funding_rate", "funding_rate_z_7d",
      "funding_rate_1d_change", "rvol_7d", "ret_3d", "btc_rvol_7d", "bars_since_high_xs_rank"]
ADD = ["imb_ewma", "imb_mean12", "imb_run"]

def load():
    rows = []
    for f in [x for x in glob.glob("/home/yuqing/ctaNew/data/ml/cache/l2_*.parquet") if "BTCUSDT" not in x]:
        sym = Path(f).stem[3:]
        d0 = pd.read_parquet(f)[["l2_imb1"]]; d0.index = pd.to_datetime(d0.index, utc=True) + pd.Timedelta("4h")
        pf = persist_feats(d0["l2_imb1"].sort_index())[ADD]
        pf["symbol"] = sym; pf["open_time"] = pf.index; rows.append(pf.reset_index(drop=True))
    m = pd.concat(rows, ignore_index=True)
    pan = pd.read_parquet("/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet",
                          columns=["symbol", "open_time", "target_z"] + V0)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    m = m.merge(pan, on=["symbol", "open_time"], how="inner").dropna(subset=["target_z", "imb_ewma"] + V0)
    return m

def wf_ic(sub, feats, folds=4):
    """cross-sectional: rank features per BAR (the strategy is cross-sectional), pooled Ridge, expanding walk-forward."""
    sub = sub.sort_values("open_time").reset_index(drop=True).copy()
    R = sub.groupby("open_time")[feats].rank(pct=True) - 0.5          # per-bar cross-sectional rank (no leakage)
    sub[feats] = R.fillna(0.0)
    bars = pd.DatetimeIndex(sub["open_time"]).unique().sort_values()
    splits = np.array_split(np.arange(len(bars)), folds + 1)
    preds = []
    for k in range(1, folds + 1):
        tr_end = bars[splits[k - 1][-1]]; te_end = bars[splits[k][-1]]
        tr = sub[sub.open_time <= tr_end]; te = sub[(sub.open_time > tr_end) & (sub.open_time <= te_end)]
        if len(tr) < 3000 or len(te) < 500: continue
        m = RidgeCV(alphas=[1, 10, 100, 1000]).fit(tr[feats].values, tr["target_z"].values)
        p = te[["open_time", "target_z"]].copy(); p["pred"] = m.predict(te[feats].values); preds.append(p)
    if not preds: return np.nan
    P = pd.concat(preds)
    ic = P.groupby("open_time").apply(lambda g: g["pred"].corr(g["target_z"], method="spearman") if len(g) >= 8 else np.nan).dropna()
    return ic.mean()

def main():
    m = load()
    cut = pd.Timestamp("2025-10-01", tz="UTC"); eras = {"RECENT": m[m.open_time >= cut], "OOS": m[m.open_time < cut]}
    print(f"merged {len(m)} rows | RECENT {len(eras['RECENT'])} OOS {len(eras['OOS'])}\n")

    print("### (A) OVERLAP: how much of imb_ewma do the V0 features explain? ###")
    for era, sub in eras.items():
        X = ((sub[V0] - sub[V0].mean()) / sub[V0].std().replace(0, 1)).fillna(0).values
        y = ((sub["imb_ewma"] - sub["imb_ewma"].mean()) / sub["imb_ewma"].std()).fillna(0).values
        r2 = LinearRegression().fit(X, y).score(X, y)
        corrs = sub[V0 + ["imb_ewma"]].corr(method="spearman")["imb_ewma"].drop("imb_ewma").abs().sort_values(ascending=False)
        top = ", ".join(f"{k}:{corrs[k]:.2f}" for k in corrs.index[:5])
        print(f"  {era}: R2(imb_ewma ~ V0) = {r2:.3f} | top |corr| V0: {top}")

    print("\n### (B) MODEL ABLATION: pooled RidgeCV, expanding walk-forward, OOS rank-IC(pred, target_z) ###")
    print(f"  {'feature set':24s} | RECENT rank-IC | OOS rank-IC")
    for name, feats in [("V0 (baseline)", V0), ("V0 + imb_ewma", V0 + ["imb_ewma"]),
                        ("V0 + ewma+mean12+run", V0 + ADD)]:
        r = wf_ic(eras["RECENT"], feats); o = wf_ic(eras["OOS"], feats)
        print(f"  {name:24s} | {r:+.4f}       | {o:+.4f}")
    print("\n  redundant if 'V0+imb_ewma' rank-IC ~ 'V0' in BOTH eras; adds if it lifts both. ABLDONE")

if __name__ == "__main__":
    main()
