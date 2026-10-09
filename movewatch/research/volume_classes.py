"""Run: python3 research/volume_classes.py

Descriptive (all three windows already read for pump questions; decides nothing): does signal 2
(`vol_persist` = log10 of hour 1-4 volume / hour -3-0 volume after a pump alert; A2) work inside
a class of coin? Same windows and the same 15 classes as research/premium_classes.py.

Trades with the frozen A2 cuts (docs/SIGNAL_SEARCH_FREEZE.json), stress 1 with funding:
- FADE: short the heavy pumps (vol_persist > 0.35259), +4 h -> +24 h (heavy-fade S24);
- FOLLOW: buy the quiet pumps (vol_persist <= -0.03228), +4 h -> +72 h.

Rule, stated before the run: a class WORKS for a trade if the ranking (Spearman of vol_persist
against the coin return +4 h -> +72 h) is < 0 in all three windows, the trade's mean is > 0 in
all three windows (each n >= 100), and C's 95% CI (cluster-week bootstrap) excludes 0.
15 classes x 2 trades = 30 tries. For every class-trade with C's CI above 0, the C trades are
also split by year (a regime check). A pass is a forward-only candidate, not evidence."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/yuqing/ctaNew/movewatch")
import research.heavy_fade as hf
import research.premium_angles as pa
import research.premium_classes as pc
import research.signal_search as ss

MIN_N = pc.MIN_N


def cell(x: pd.DataFrame) -> dict:
    y = x[x["r_S72"].notna() & x["vol_persist"].notna()]
    ic = ss.spearman(y["vol_persist"].to_numpy(), y["r_S72"].to_numpy()) if len(y) >= MIN_N else np.nan
    fade, foll = x[x["vol_persist"] > hf.HEAVY], x[x["vol_persist"] <= hf.QUIET]
    return {"n": len(x), "ic": ic, "fade_rows": fade, "follow_rows": foll,
            "fade": hf.stats(pa.pnl(fade, "S24", "short"), fade["time"]),
            "follow": hf.stats(pa.pnl(foll, "S72", "long"), foll["time"])}


if __name__ == "__main__":
    sectors = pc.fetch_sectors()
    p = pc.frame(require="vol_persist")
    print(f"pumps with vol_persist: {p.groupby('win').size().to_dict()}; rank missing {p['rank'].isna().sum()}, age missing {p['age_d'].isna().sum()}")
    works, regime = [], []
    print(f"\n{'class':20} | {'ranking S / V / C':>24} | {'FADE short heavy 24h: S / V / C':>52} | {'FOLLOW buy quiet 72h: S / V / C':>52}")
    for name, m in {"(control) all pumps": p["vol_persist"].notna(), **pc.classes(p, sectors)}.items():
        cs = {w: cell(p[m.to_numpy() & (p["win"] == w).to_numpy()]) for w in ("S", "V", "C")}
        ics = " / ".join(f"{cs[w]['ic']:+.3f}" if np.isfinite(cs[w]["ic"]) else "  n/a " for w in ("S", "V", "C"))
        line = f"{name:20} | {ics:>24}"
        for t in ("fade", "follow"):
            cc = cs["C"][t]
            ci = f" [{100 * cc['ci'][0]:+.2f},{100 * cc['ci'][1]:+.2f}]" if cc["status"] == "OK" else ""
            line += " | " + " / ".join(pc.show(cs[w], t) for w in ("S", "V", "C")) + ci
            ok = (all(np.isfinite(cs[w]["ic"]) and cs[w]["ic"] < 0 for w in "SVC")
                  and all(cs[w][t]["n"] >= MIN_N and cs[w][t]["mean"] > 0 for w in "SVC")
                  and cc["status"] == "OK" and cc["ci"][0] > 0)
            if ok and not name.startswith("(control)"):
                works.append((name, t))
            if cc["status"] == "OK" and cc["ci"][0] > 0:
                x = cs["C"][f"{t}_rows"]
                v = pd.Series(pa.pnl(x, "S24" if t == "fade" else "S72", "short" if t == "fade" else "long"), index=x.index).dropna()
                yr = v.groupby(x.loc[v.index, "time"].dt.year).agg(["size", "mean", "median"])
                top = yr["size"].mul(yr["mean"]).idxmax()
                regime.append(f"{name:20} {t:6} C by year: " + " | ".join(f"{y} n{int(r['size'])} {100 * r['mean']:+.1f}% (med {100 * r['median']:+.1f}%)" for y, r in yr.iterrows())
                              + f" | {top} share of P&L {v[x.loc[v.index, 'time'].dt.year == top].sum() / v.sum():.0%}")
        print(line)
    print(f"\nWORKS (rule above): {works if works else 'none'} of 30 tries")
    print("\nC cells with CI above 0, by year:" + ("\n" + "\n".join(regime) if regime else " none"))

    # Added AFTER seeing the class table (post-hoc, forward-only): thin coins (< $20M/day) had both
    # legs positive in all three windows, so pool the two legs and check cost stress and years.
    import research.btc_hedge as bh
    thin = p[p["liq"] < 20e6]

    def legs(x: pd.DataFrame, stress: int) -> pd.DataFrame:
        f, q = x[x["vol_persist"] > hf.HEAVY], x[x["vol_persist"] <= hf.QUIET]
        cq = np.array([2 * (bh.FEE + stress * bh.slip(v)) for v in q["liq"].to_numpy(float)])
        nq = q["r_S72"].to_numpy(float) - cq - q["fund_S72"].to_numpy(float)
        return pd.concat([pd.DataFrame({"v": hf.net(f, "S24", stress), "time": f["time"].to_numpy(), "leg": "fade"}),
                          pd.DataFrame({"v": nq, "time": q["time"].to_numpy(), "leg": "follow"})], ignore_index=True) \
            .assign(time=lambda d: pd.to_datetime(d["time"], utc=True))

    print("\nPOST-HOC thin coins (< $20M/day), short heavy 24h + buy quiet 72h pooled:")
    for w in ("S", "V", "C"):
        for st in (1, 3):
            d = legs(thin[thin["win"] == w], st)
            s = hf.stats(d["v"].to_numpy(), d["time"])
            print(f"  {w} stress {st}: n{s['n']} mean {100 * s['mean']:+.2f}% [{100 * s['ci'][0]:+.2f},{100 * s['ci'][1]:+.2f}] "
                  f"median {100 * s['median']:+.2f}% worst {100 * s['worst']:+.0f}%")
    d = legs(thin[thin["win"] == "C"], 1)
    print("  C by year, mean %:", d.groupby([d["time"].dt.year, "leg"])["v"].mean().mul(100).round(2).unstack().to_dict("index"))
