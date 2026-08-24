"""Decisive: is abs(Δdepth) [the strong new OB vol signal the user identified] INCREMENTAL to the model's trailing-vol
features (rvol_7d, atr_pct, idio_vol), or redundant like imbstd? Raw + partial x-sec rank-IC vs forward 1d vol, both
eras. If partial CI>0 both eras, order-book size-change dynamics LEAD vol beyond klines vol = a genuinely useful new
feature for the risk/sizing side. (Compares imbstd for reference.)
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
rng = np.random.default_rng(19)
CTRL = ["rvol_7d", "atr_pct", "idio_vol_to_btc_1d"]

def build():
    rows = []
    for f in [x for x in glob.glob("/home/yuqing/ctaNew/data/ml/cache/l2_*.parquet") if "BTCUSDT" not in x]:
        sym = Path(f).stem[3:]; d = pd.read_parquet(f).sort_index(); d.index = pd.to_datetime(d.index, utc=True)
        seg = (d.index.to_series().diff() > pd.Timedelta("8h")).cumsum().values
        pf = pd.DataFrame(index=d.index)
        pf["abs_d_liq1"] = d["l2_liq1"].groupby(seg).diff().abs() if "l2_liq1" in d else np.nan
        pf["imbstd"] = d["l2_imbstd"] if "l2_imbstd" in d else np.nan
        pf.index = pf.index + pd.Timedelta("4h")
        pf["symbol"] = sym; pf["open_time"] = pf.index; rows.append(pf.reset_index(drop=True))
    L = pd.concat(rows, ignore_index=True)
    pan = pd.read_parquet("/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet",
                          columns=["symbol", "open_time", "return_pct"] + CTRL)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True); pan = pan.sort_values(["symbol", "open_time"])
    pan["fwd_vol1d"] = pan.groupby("symbol")["return_pct"].transform(lambda s: s.rolling(6).std().shift(-6))
    return pan.merge(L, on=["symbol", "open_time"], how="inner")

def xrank(df, cols): return df.groupby("open_time")[cols].rank(pct=True)
def resid(y, X): A = np.column_stack([np.ones(len(X)), X]); b, *_ = np.linalg.lstsq(A, y, rcond=None); return y - A @ b
def dayci(rx, ry, days):
    d = pd.DataFrame({"rx": rx, "ry": ry, "day": days}); g = [x for _, x in d.groupby("day")]
    if len(g) < 5: return (np.nan, np.nan)
    o = [pd.concat([g[i] for i in rng.integers(0, len(g), len(g))]).pipe(lambda s: s["rx"].corr(s["ry"])) for _ in range(2000)]
    return tuple(np.nanpercentile(o, [2.5, 97.5]))

def test(sub, feat):
    sub = sub.dropna(subset=[feat, "fwd_vol1d"] + CTRL).copy()
    if len(sub) < 500: return None
    R = xrank(sub, [feat, "fwd_vol1d"] + CTRL).fillna(0.5)
    raw = R[feat].corr(R["fwd_vol1d"])
    rx = resid(R[feat].values, R[CTRL].values); ry = resid(R["fwd_vol1d"].values, R[CTRL].values)
    lo, up = dayci(rx, ry, sub["open_time"].dt.floor("1D").values)
    return raw, np.corrcoef(rx, ry)[0, 1], lo, up

def main():
    m = build(); cut = pd.Timestamp("2025-10-01", tz="UTC")
    eras = {"RECENT": m[m.open_time >= cut], "OOS": m[m.open_time < cut]}
    print(f"merged {len(m)} | target = forward 1d vol | control = trailing vol {CTRL}\n")
    for feat in ["abs_d_liq1", "imbstd"]:
        print(f"### {feat} ###")
        for era, sub in eras.items():
            r = test(sub, feat)
            if r is None: print(f"  {era}: n/a"); continue
            raw, part, lo, up = r
            f = "INCREMENTAL (CI>0)" if lo > 0 else ("neg" if up < 0 else "redundant (CI~0)")
            print(f"  {era}: raw IC {raw:+.3f} -> partial {part:+.3f} [{lo:+.3f},{up:+.3f}] -> {f}")
        print()
    print("read: partial CI>0 BOTH eras = OB size-dynamics lead vol beyond klines = useful. VOLINCRDONE")

if __name__ == "__main__":
    main()
