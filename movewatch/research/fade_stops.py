"""Run: python3 research/fade_stops.py [--stress 3]

Descriptive (all windows already read for pump questions; decides nothing): does a stop-loss help
forward test F1's trade? Population as F1: non-meme heavy pumps (vol_persist > A2 heavy cut),
short +4 h -> +24 h, stress 1 with funding (as the research book).

Stops, from the rows research/heavy_fade.py built: cover at the first minute CLOSE at or above
entry x (1 + stop), so a jump through the level fills at that minute's close (overshoot
included); stops at +15 / +25 / +40%. Per window and per year: n, mean with its cluster-week
95% CI, median, win rate, p5, worst, the share stopped, and losses / wins. The stop exit is a
market buy into a spike, so `--stress 3` (slippage x3) is the fairer cost for the stopped legs."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/yuqing/ctaNew/movewatch")
import research.heavy_fade as hf
import research.premium_classes as pc

STRESS = 3 if "--stress" in sys.argv and sys.argv[sys.argv.index("--stress") + 1] == "3" else 1
CELLS = ("S24", "S24_stop15", "S24_stop25", "S24_stop40")


def row(lab: str, x: pd.DataFrame, cell: str) -> str:
    v = hf.net(x, cell, STRESS)
    ok = np.isfinite(v)
    v, t = v[ok], x["time"][ok]
    if len(v) < 30:
        return f"{lab:9} {cell:11} n {len(v):5d}"
    s = hf.stats(v, t)
    ci = f"[{100 * s['ci'][0]:+.2f}, {100 * s['ci'][1]:+.2f}]" if s["status"] == "OK" else "(few)"
    w, l = v[v > 0], v[v <= 0]
    stopped = x.loc[ok, f"stopped_{cell}"].astype(bool).mean() if cell != "S24" else 0.0
    return (f"{lab:9} {cell:11} n {len(v):5d} mean {100 * v.mean():+6.2f}% {ci:18} med {100 * np.median(v):+5.1f}% "
            f"win {100 * (v > 0).mean():3.0f}% p5 {100 * np.percentile(v, 5):+6.1f}% worst {100 * v.min():+5.0f}% "
            f"stopped {100 * stopped:4.1f}% losses/wins {-l.sum() / w.sum():.2f}")


if __name__ == "__main__":
    p = pc.frame(require="vol_persist")
    x = p[(p["vol_persist"] > hf.HEAVY) & (~p["meme"])].copy()
    print(f"F1 population (non-meme heavy pumps), short +4h -> +24h, stress {STRESS} + funding")
    for lab, sel in (("all", slice(None)), ("C 2020-25", x["win"] == "C"), ("V+S 25-26", x["win"].isin(["V", "S"]))):
        print()
        for c in CELLS:
            print(row(lab, x[sel] if not isinstance(sel, slice) else x, c))
    print("\nmean per trade by year:")
    yr = x["time"].dt.year
    print(f"{'year':6} " + " ".join(f"{c:>11}" for c in CELLS))
    for y in sorted(yr.unique()):
        g = x[yr == y]
        print(f"{y:<6} " + " ".join(f"{100 * np.nanmean(hf.net(g, c, STRESS)):+10.2f}%" for c in CELLS) + f"   n {len(g)}")
