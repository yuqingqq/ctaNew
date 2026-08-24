"""User's point: vol should come from SIZE/depth CHANGE; direction from imbalance/size-flow CHANGE (imbstd = std =
magnitude, not directional). Test the CHANGE features derivable from the 4h cache: d_liq1 (depth change), |d_liq1|
(depth turbulence), d_imb1 (imbalance change = coarse net directional flow), vs 3 targets — forward vol, raw return
(direction), alpha (residual direction) — both eras, x-sec rank-IC + day CI. Answers whether size-change is a vol
signal and imbalance-change a directional one. (Within-bar net flow / depth-std need re-enrichment from raw bookDepth.)
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
rng = np.random.default_rng(17)

def build():
    rows = []
    for f in [x for x in glob.glob("/home/yuqing/ctaNew/data/ml/cache/l2_*.parquet") if "BTCUSDT" not in x]:
        sym = Path(f).stem[3:]; d = pd.read_parquet(f).sort_index()
        d.index = pd.to_datetime(d.index, utc=True)
        # gap-aware bar-to-bar change (reset diff across >8h breaks)
        seg = (d.index.to_series().diff() > pd.Timedelta("8h")).cumsum().values
        liq = d["l2_liq1"] if "l2_liq1" in d else pd.Series(np.nan, index=d.index)
        imb = d["l2_imb1"] if "l2_imb1" in d else pd.Series(np.nan, index=d.index)
        pf = pd.DataFrame(index=d.index)
        pf["d_liq1"] = liq.groupby(seg).diff(); pf["abs_d_liq1"] = pf["d_liq1"].abs()
        pf["d_imb1"] = imb.groupby(seg).diff()
        pf.index = pf.index + pd.Timedelta("4h")                     # PIT
        pf["symbol"] = sym; pf["open_time"] = pf.index; rows.append(pf.reset_index(drop=True))
    L = pd.concat(rows, ignore_index=True)
    pan = pd.read_parquet("/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet",
                          columns=["symbol", "open_time", "return_pct", "alpha_vs_btc_realized"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True); pan = pan.sort_values(["symbol", "open_time"])
    pan["fwd_vol1d"] = pan.groupby("symbol")["return_pct"].transform(lambda s: s.rolling(6).std().shift(-6))
    return pan.merge(L, on=["symbol", "open_time"], how="inner")

def xic(df, feat, tgt): return df.groupby("open_time").apply(lambda g: g[feat].corr(g[tgt], method="spearman") if g[[feat, tgt]].dropna().shape[0] >= 8 else np.nan).dropna()
def dayci(ic):
    s = pd.DataFrame({"ic": ic.values}, index=pd.to_datetime(ic.index, utc=True)); s["d"] = s.index.floor("1D")
    g = [x["ic"].values for _, x in s.groupby("d")]
    if len(g) < 5: return (np.nan, np.nan)
    o = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(2500)]
    return tuple(np.nanpercentile(o, [2.5, 97.5]))

def main():
    m = build(); cut = pd.Timestamp("2025-10-01", tz="UTC")
    eras = {"RECENT": m[m.open_time >= cut], "OOS": m[m.open_time < cut]}
    print(f"merged {len(m)} | RECENT {len(eras['RECENT'])} OOS {len(eras['OOS'])}\n")
    for tgt, lab in [("fwd_vol1d", "FORWARD VOL"), ("return_pct", "RAW return (DIRECTION)"), ("alpha_vs_btc_realized", "ALPHA (residual dir)")]:
        print(f"### target = {lab} ###")
        for feat in ["d_liq1", "abs_d_liq1", "d_imb1"]:
            cells = {}
            for era, sub in eras.items():
                ic = xic(sub, feat, tgt); lo, up = dayci(ic); cells[era] = (ic.mean(), lo, up)
            (ra, rl, ru), (oa, ol, ou) = cells["RECENT"], cells["OOS"]
            both = "BOTH-ERA" if (np.sign(ra) == np.sign(oa) and (rl > 0 or ru < 0) and (ol > 0 or ou < 0) and abs(ra) > 0.02 and abs(oa) > 0.02) else "no"
            print(f"  {feat:10s} RECENT {ra:+.4f} [{rl:+.4f},{ru:+.4f}] | OOS {oa:+.4f} [{ol:+.4f},{ou:+.4f}] | {both}")
        print()
    print("read: size-change (d_liq1/abs) -> vol?  imbalance-change (d_imb1) -> direction? FLOWDONE")

if __name__ == "__main__":
    main()
