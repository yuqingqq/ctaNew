"""Is l2_liq1 INCREMENTAL to the V0 feature set, or redundant with its vol/size features? Partial cross-sectional
rank-IC: control target_z and l2_liq1 for ALL V0 features (cross-sectionally rank everything per bar, pooled-OLS
residualize both on V0 ranks), then correlate the residuals. If partial-IC >> 0 in BOTH eras, l2_liq1 adds beyond V0;
if it collapses toward 0, V0 already captures it (redundant). Tested overall AND among over-extended names (top-tercile
return_1d = the short pool, where the raw l2_liq1 signal lived). Day-clustered bootstrap CI. Uses the pilot cache;
this is the cheap redundancy gate before any bigger fetch.
"""
import glob
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
rng = np.random.default_rng(5)
V0 = ["return_1d", "atr_pct", "vwap_slope_96", "bars_since_high", "autocorr_pctile_7d", "obv_z_1d", "corr_to_btc_1d",
      "beta_to_btc_change_5d", "idio_vol_to_btc_1h", "idio_vol_to_btc_1d", "funding_rate", "funding_rate_z_7d",
      "funding_rate_1d_change", "rvol_7d", "ret_3d", "btc_rvol_7d", "bars_since_high_xs_rank"]
L2 = ["l2_liq1", "l2_slope", "l2_imbstd", "l2_asym1", "l2_imb1"]

def xrank(df, cols):
    """cross-sectional rank (0..1) within each bar."""
    return df.groupby("open_time")[cols].rank(pct=True)

def residualize(y, X):
    """pooled OLS residual of y on X (with intercept); X,y already NaN-filled."""
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta

def day_boot_corr(rx, ry, days):
    d = pd.DataFrame({"rx": rx, "ry": ry, "day": days})
    grps = [g for _, g in d.groupby("day")]
    if len(grps) < 5: return (np.nan, np.nan)
    out = []
    for _ in range(2000):
        s = pd.concat([grps[i] for i in rng.integers(0, len(grps), len(grps))])
        out.append(s["rx"].corr(s["ry"]))
    return tuple(np.nanpercentile(out, [2.5, 97.5]))

def partial_ic(sub, feat):
    sub = sub.dropna(subset=[feat, "target_z"] + V0).copy()
    if len(sub) < 200: return (np.nan, np.nan, np.nan, np.nan)
    R = xrank(sub, V0 + [feat, "target_z"]).fillna(0.5)
    Xv0 = R[V0].values
    ry = residualize(R["target_z"].values, Xv0)
    rx = residualize(R[feat].values, Xv0)
    raw = np.corrcoef(R[feat].values, R["target_z"].values)[0, 1]
    part = np.corrcoef(rx, ry)[0, 1]
    lo, up = day_boot_corr(rx, ry, sub["open_time"].dt.floor("1D").values)
    return raw, part, lo, up

def main():
    fr = []
    for f in glob.glob("/home/yuqing/ctaNew/data/ml/cache/l2_*.parquet"):
        d = pd.read_parquet(f).reset_index(); d["symbol"] = Path(f).stem[3:]; fr.append(d)
    L = pd.concat(fr, ignore_index=True)
    L["obs_bar"] = pd.to_datetime(L["obs_bar"], utc=True); L["open_time"] = L["obs_bar"] + pd.Timedelta("4h")
    pan = pd.read_parquet("/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet")
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    m = pan.merge(L[["symbol", "open_time"] + L2], on=["symbol", "open_time"], how="inner")
    cut = pd.Timestamp("2025-10-01", tz="UTC")
    eras = {"RECENT": m[m.open_time >= cut], "OOS": m[m.open_time < cut]}
    print(f"merged {len(m)} rows | {m.symbol.nunique()} syms | eras: RECENT {len(eras['RECENT'])} OOS {len(eras['OOS'])}")
    print("partial-IC = rank-IC(feature, target_z) CONTROLLING for all V0 features; if ~raw it adds, if ~0 it's redundant\n")
    for scope, filt in [("ALL names", lambda s: s),
                        ("over-extended (top-tercile return_1d = short pool)",
                         lambda s: s[s.groupby("open_time")["return_1d"].transform(lambda x: x >= x.quantile(2/3))])]:
        print(f"### scope: {scope} ###")
        print(f"{'feature':10s} | {'RECENT raw->partial [CI]':34s} | {'OOS raw->partial [CI]':34s} | adds both?")
        for feat in L2:
            cells = {}
            for era, sub in eras.items():
                raw, part, lo, up = partial_ic(filt(sub), feat)
                cells[era] = (raw, part, lo, up)
            (rr, rp, rl, ru) = cells["RECENT"]; (orr, op, ol, ou) = cells["OOS"]
            adds = "YES" if (np.sign(rp) == np.sign(op) and abs(rp) > 0.02 and abs(op) > 0.02
                             and (rl > 0 or ru < 0) and (ol > 0 or ou < 0)) else "no"
            print(f"{feat:10s} | {rr:+.3f}->{rp:+.3f} [{rl:+.3f},{ru:+.3f}] | {orr:+.3f}->{op:+.3f} [{ol:+.3f},{ou:+.3f}] | {adds}")
        print()
    print("INCDONE")

if __name__ == "__main__":
    main()
