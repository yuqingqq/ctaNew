"""Sweep: baseline WINDOW (12/30/60/90 bars) x book DEPTH (imb1/2/3/5) for the relative-strength directional signal.
For each (depth, window): per-symbol gap-aware rolling mu/sd -> dev (direction) + |z| (strength); PIT +4h; measure
(a) directional rank-IC(dev, fwd return) both eras, (b) extreme-|z| directional book (top-quintile |z|, long +dev /
short -dev, net 8bps) daily Sharpe both eras. Flags any (depth,window) that is directional in BOTH eras. Reads the
saved pilot (l2_pilot_fetch.py) so it runs offline. 50 syms x 3mo/era.
"""
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
rng = np.random.default_rng(37); COST = 0.0008
SD = Path("/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad")
CUT = pd.Timestamp("2025-10-01", tz="UTC")

def prep(df, depth, w):
    out = []
    mp = max(5, w // 3)
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("obs_bar"); seg = (g["obs_bar"].diff() > pd.Timedelta("8h")).cumsum().values
        imb = g[depth]
        mu = imb.groupby(seg).transform(lambda s: s.rolling(w, min_periods=mp).mean())
        sd = imb.groupby(seg).transform(lambda s: s.rolling(w, min_periods=mp).std())
        gg = pd.DataFrame({"dev": (imb - mu).values, "absz": ((imb - mu) / sd.replace(0, np.nan)).abs().values})
        gg["symbol"] = sym; gg["open_time"] = g["obs_bar"].values + pd.Timedelta("4h")
        out.append(gg)
    return pd.concat(out, ignore_index=True)

def ic(sub, feat="dev", tgt="return_pct"):
    s = sub.groupby("open_time").apply(lambda g: g[feat].corr(g[tgt], method="spearman") if g[[feat, tgt]].dropna().shape[0] >= 6 else np.nan).dropna()
    if len(s) < 5: return (np.nan, np.nan, np.nan)
    d = pd.DataFrame({"v": s.values}, index=pd.to_datetime(s.index, utc=True)); d["day"] = d.index.floor("1D")
    gr = [x["v"].values for _, x in d.groupby("day")]
    b = [np.concatenate([gr[i] for i in rng.integers(0, len(gr), len(gr))]).mean() for _ in range(1200)]
    return (s.mean(), *np.nanpercentile(b, [2.5, 97.5]))

def ext_book(sub):
    sub = sub.dropna(subset=["absz", "dev", "return_pct"])
    if len(sub) < 200: return np.nan
    ext = sub[sub.absz >= sub.absz.quantile(0.8)]
    r = np.sign(ext["dev"]) * ext["return_pct"] - COST
    dd = pd.Series(r.values, index=pd.to_datetime(ext["open_time"].values, utc=True)).groupby(lambda t: t.date()).mean()
    return dd.mean() / dd.std() * np.sqrt(365) if dd.std() > 0 else np.nan

def main():
    P = pd.read_parquet(SD / "pilot_imbdepth.parquet")
    P["obs_bar"] = pd.to_datetime(P["obs_bar"], utc=True)
    depths = [c for c in ["imb1", "imb2", "imb3", "imb5"] if c in P.columns]
    pan = pd.read_parquet("/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet",
                          columns=["symbol", "open_time", "return_pct"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    print(f"pilot {len(P)} rows, {P.symbol.nunique()} syms, depths {depths}\n")
    print(f"{'depth':6s} {'win':>4s} | {'dir rank-IC(dev) REC':22s} {'OOS':22s} | ext-|z| book Sharpe REC/OOS | both-era?")
    for depth in depths:
        for w in [12, 30, 60, 90]:
            F = prep(P, depth, w); F["open_time"] = pd.to_datetime(F["open_time"], utc=True)
            m = pan.merge(F, on=["symbol", "open_time"], how="inner")
            rec, oos = m[m.open_time >= CUT], m[m.open_time < CUT]
            (ra, rl, ru) = ic(rec); (oa, ol, ou) = ic(oos)
            sr, so = ext_book(rec), ext_book(oos)
            icboth = (np.sign(ra) == np.sign(oa)) and (rl > 0 or ru < 0) and (ol > 0 or ou < 0)
            bkboth = (np.sign(sr) == np.sign(so)) and abs(sr) > 0.3 and abs(so) > 0.3
            flag = "IC-both" if icboth else ("BOOK-both" if bkboth else "no")
            print(f"{depth:6s} {w:>4d} | {ra:+.4f} [{rl:+.4f},{ru:+.4f}] {oa:+.4f} [{ol:+.4f},{ou:+.4f}] | {sr:+.2f} / {so:+.2f} | {flag}")
        print()
    print("read: any (depth,window) with same-sign CI-off-zero IC both eras OR extreme-book Sharpe same-sign both = a lead. SWEEPDONE")

if __name__ == "__main__":
    main()
