"""User: measure imbalance STRENGTH with RELATIVE values, not raw (the book is structurally bid-heavy, so raw imb1
mixes the baseline lean with the signal). Compare, for DIRECTION, both eras, full data:
  imb1      raw signed imbalance
  imb1_dev  imb1 - trailing-30bar mean         (deviation from the name's OWN typical lean)
  imb1_z    (imb1 - roll30 mean)/roll30 std    (RELATIVE STRENGTH: how many sigma from normal, per name)
Metrics: (a) cross-sectional rank-IC + day-CI, (b) DIRECTIONAL long-short spread (long top-tercile / short bottom-
tercile by the feature, 4h hold, net 8bps, daily Sharpe + CI) = would a directional book make money. vs return_pct
(raw dir) and alpha. Gap-aware (resets across >8h breaks).
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
rng = np.random.default_rng(29); COST = 0.0008

def build():
    rows = []
    for f in [x for x in glob.glob("/home/yuqing/ctaNew/data/ml/cache/l2_*.parquet") if "BTCUSDT" not in x]:
        sym = Path(f).stem[3:]; d = pd.read_parquet(f).sort_index()
        if "l2_imb1" not in d: continue
        d.index = pd.to_datetime(d.index, utc=True); seg = (d.index.to_series().diff() > pd.Timedelta("8h")).cumsum().values
        imb = d["l2_imb1"]
        mu = imb.groupby(seg).transform(lambda s: s.rolling(30, min_periods=10).mean())
        sd = imb.groupby(seg).transform(lambda s: s.rolling(30, min_periods=10).std())
        pf = pd.DataFrame({"imb1": imb, "imb1_dev": imb - mu, "imb1_z": (imb - mu) / sd.replace(0, np.nan)})
        pf.index = d.index + pd.Timedelta("4h")                    # PIT
        pf["symbol"] = sym; pf["open_time"] = pf.index; rows.append(pf.reset_index(drop=True))
    L = pd.concat(rows, ignore_index=True)
    pan = pd.read_parquet("/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet",
                          columns=["symbol", "open_time", "return_pct", "alpha_vs_btc_realized"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    return pan.merge(L, on=["symbol", "open_time"], how="inner")

def xic(df, feat, tgt): return df.groupby("open_time").apply(lambda g: g[feat].corr(g[tgt], method="spearman") if g[[feat, tgt]].dropna().shape[0] >= 8 else np.nan).dropna()
def dboot(s):
    d = pd.DataFrame({"v": s.values}, index=pd.to_datetime(s.index, utc=True)); d["day"] = d.index.floor("1D")
    g = [x["v"].values for _, x in d.groupby("day")]
    if len(g) < 5: return (np.nan, np.nan)
    o = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(2500)]
    return tuple(np.nanpercentile(o, [2.5, 97.5]))
def ls_spread(df, feat, tgt):
    def sp(g):
        gg = g[[feat, tgt]].dropna()
        if len(gg) < 12: return np.nan
        q = gg[feat].rank(method="first"); n = len(gg); top = gg[q > 2 * n / 3][tgt].mean(); bot = gg[q <= n / 3][tgt].mean()
        return top - bot
    return df.groupby("open_time").apply(sp).dropna()

def main():
    m = build(); cut = pd.Timestamp("2025-10-01", tz="UTC")
    eras = {"RECENT": m[m.open_time >= cut], "OOS": m[m.open_time < cut]}
    print(f"merged {len(m)} | RECENT {len(eras['RECENT'])} OOS {len(eras['OOS'])}")
    print("corr(imb1, imb1_z) = %.2f (relative vs raw)\n" % m["imb1"].corr(m["imb1_z"]))
    for tgt, lab in [("return_pct", "RAW direction"), ("alpha_vs_btc_realized", "ALPHA")]:
        print(f"### target = {lab} ###")
        print(f"{'feature':9s} | {'rank-IC RECENT [CI]':26s} {'OOS [CI]':26s} | LS-spread bps/4h RECENT / OOS (net 8bps)")
        for feat in ["imb1", "imb1_dev", "imb1_z"]:
            out = []
            for era, sub in eras.items():
                ic = xic(sub, feat, tgt); il, iu = dboot(ic)
                sp = ls_spread(sub, feat, tgt); spd = sp - COST; sl, su = dboot(spd)
                out.append((ic.mean(), il, iu, spd.mean() * 1e4, sl * 1e4, su * 1e4))
            (ra, rl, ru, rs, rsl, rsu), (oa, ol, ou, os_, osl, osu) = out
            print(f"{feat:9s} | {ra:+.4f} [{rl:+.4f},{ru:+.4f}] {oa:+.4f} [{ol:+.4f},{ou:+.4f}] | {rs:+.1f}[{rsl:+.1f},{rsu:+.1f}] / {os_:+.1f}[{osl:+.1f},{osu:+.1f}]")
        print()
    print("read: does RELATIVE (imb1_z/dev) beat RAW (imb1) for direction, both eras + tradeable spread? RELDONE")

if __name__ == "__main__":
    main()
