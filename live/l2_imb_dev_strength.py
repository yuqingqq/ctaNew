"""User's decomposition: imb1_dev (imb1 - trailing mean) = DIRECTION; |z| = |dev|/sd = STRENGTH (how unusual). Hypothesis:
the DIRECTION (imb1_dev) predicts forward return MORE when the deviation is HEAVY (high |z|) — a conditional/threshold
effect a linear IC misses. Test: bucket observations by |z| (deviation strength), and within each bucket measure the
directional IC = Spearman(imb1_dev, forward return), both eras, day-clustered CI. If |IC| grows with |z|, heavy
deviations are the signal. Also the directional long-short spread among the EXTREME-|z| events (would you trade it).
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
rng = np.random.default_rng(31); COST = 0.0008

def build():
    rows = []
    for f in [x for x in glob.glob("/home/yuqing/ctaNew/data/ml/cache/l2_*.parquet") if "BTCUSDT" not in x]:
        sym = Path(f).stem[3:]; d = pd.read_parquet(f).sort_index()
        if "l2_imb1" not in d: continue
        d.index = pd.to_datetime(d.index, utc=True); seg = (d.index.to_series().diff() > pd.Timedelta("8h")).cumsum().values
        imb = d["l2_imb1"]
        mu = imb.groupby(seg).transform(lambda s: s.rolling(30, min_periods=10).mean())
        sd = imb.groupby(seg).transform(lambda s: s.rolling(30, min_periods=10).std())
        dev = imb - mu; z = dev / sd.replace(0, np.nan)
        pf = pd.DataFrame({"dev": dev, "absz": z.abs(), "zsign": np.sign(dev)})
        pf.index = d.index + pd.Timedelta("4h"); pf["symbol"] = sym; pf["open_time"] = pf.index
        rows.append(pf.reset_index(drop=True))
    L = pd.concat(rows, ignore_index=True)
    pan = pd.read_parquet("/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet",
                          columns=["symbol", "open_time", "return_pct", "alpha_vs_btc_realized"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    return pan.merge(L, on=["symbol", "open_time"], how="inner")

def dayci_ic(sub, feat, tgt):
    """mean of per-day Spearman(feat,tgt) + day-clustered bootstrap CI (light: compute daily IC once, resample days)."""
    g = sub.dropna(subset=[feat, tgt]).copy(); g["day"] = g["open_time"].dt.floor("1D")
    perday = g.groupby("day").apply(lambda x: x[feat].corr(x[tgt], method="spearman") if len(x) >= 6 else np.nan).dropna().values
    if len(perday) < 5: return (np.nan, np.nan, np.nan)
    boot = [perday[rng.integers(0, len(perday), len(perday))].mean() for _ in range(2000)]
    return (perday.mean(), *np.nanpercentile(boot, [2.5, 97.5]))

def main():
    m = build(); cut = pd.Timestamp("2025-10-01", tz="UTC")
    eras = {"RECENT": m[m.open_time >= cut], "OOS": m[m.open_time < cut]}
    print(f"merged {len(m)} | RECENT {len(eras['RECENT'])} OOS {len(eras['OOS'])}")
    print("does DIRECTION (imb1_dev) predict forward return MORE at HIGH |z| (heavy deviation)?\n")
    for tgt, lab in [("return_pct", "RAW direction"), ("alpha_vs_btc_realized", "ALPHA")]:
        print(f"### target = {lab} : directional IC(imb1_dev, fwd) by |z| (deviation-strength) quintile ###")
        for era, sub in eras.items():
            sub = sub.dropna(subset=["absz"]).copy()
            sub["zb"] = pd.qcut(sub["absz"], 5, labels=["|z| Q1(small)", "Q2", "Q3", "Q4", "Q5(heavy)"], duplicates="drop")
            print(f"  {era}:")
            for b, g in sub.groupby("zb"):
                ic, lo, up = dayci_ic(g, "dev", tgt)
                mz = g["absz"].median()
                f = "sig" if (lo > 0 or up < 0) else "~0"
                print(f"    {str(b):14s} (median|z| {mz:.2f}, n={len(g):6d}) IC(dev,fwd) {ic:+.4f} [{lo:+.4f},{up:+.4f}] {f}")
        print()
    # extreme-|z| directional book: among top-|z| quintile, long +dev / short -dev, net of cost, both eras
    print("### EXTREME-|z| directional book (top quintile |z|, long +dev / short -dev, 4h, net 8bps) ###")
    for era, sub in eras.items():
        sub = sub.dropna(subset=["absz", "dev", "return_pct"]).copy()
        thr = sub["absz"].quantile(0.8); ext = sub[sub.absz >= thr]
        ret = np.sign(ext["dev"]) * ext["return_pct"] - COST     # directional: go the way dev points
        r = pd.DataFrame({"r": ret.values}, index=ext["open_time"].values); r.index = pd.to_datetime(r.index, utc=True)
        r["day"] = r.index.floor("1D"); dd = r.groupby("day")["r"].mean()
        sh = dd.mean() / dd.std() * np.sqrt(365) if dd.std() > 0 else np.nan
        print(f"  {era}: extreme-|z| n={len(ext)} | directional net {ret.mean()*1e4:+.2f}bps/4h | daily Sharpe {sh:+.2f}")
    print("read: if IC(dev) rises with |z| AND extreme book is +/CI>0 both eras, heavy deviation IS a signal. DEVSTRDONE")

if __name__ == "__main__":
    main()
