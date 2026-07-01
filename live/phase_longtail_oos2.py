"""Generalized OOS + episode-concentration eval for arbitrary candidate tags.

Usage:
  python3 -m live.phase_longtail_oos2 <base_tag> <cand_tag> [<cand_tag> ...] [--split 2025-10-04]
Each tag is a dir under live/state/longtail/ containing state/cycles.csv.
For OOS runs (fullhist preds) --split separates the true hold-out (before split) from the
loop's in-sample window. For production-preds runs (single 8-month window) pass --split none
to skip the split and just get regime + folds_beat + episode-concentration.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
ROOT = Path("/home/yuqing/ctaNew/live/state/longtail")
ANN = np.sqrt(365); GT = -0.25

def load(tag):
    c = pd.read_csv(ROOT / tag / "state" / "cycles.csv")
    c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    return c.sort_values("open_time").set_index("open_time")
def dsh(s):
    d = (s.fillna(0)/1e4).resample("1D").sum(); return float(d.mean()/d.std()*ANN) if d.std()>0 else np.nan
def maxdd(s):
    eq = s.fillna(0).cumsum(); return float((eq-eq.cummax()).min())
def cvar5(s):
    x = s.dropna().to_numpy();
    return float(np.mean(np.sort(x)[:max(1,len(x)//20)])) if len(x)>=20 else np.nan
def grind(c): return c[(c["regime"]=="bear") & (c["btc_ret_30d"]>=GT)]
def deep(c):  return c[(c["regime"]=="bear") & (c["btc_ret_30d"]<GT)]
def bull(c):  return c[c["regime"]=="bull"]
def bearall(c): return c[c["regime"]=="bear"]

def block(name, cb, cc):
    """cb=baseline slice df, cc=candidate slice df"""
    def line(lbl, fb, fc):
        sb, sc = fb(cb)["pnl_bps"], fc(cc)["pnl_bps"]
        # per-month episode improvement
        mb = fb(cb).groupby(fb(cb).index.to_period("M"))["pnl_bps"].sum()
        mc = fc(cc).groupby(fc(cc).index.to_period("M"))["pnl_bps"].sum()
        com = mb.index.intersection(mc.index); d = (mc[com]-mb[com])
        imp = int((d>5).sum()); wor=int((d<-5).sum())
        print(f"    {lbl:9s} tot {sb.sum():+8.0f}->{sc.sum():+8.0f} (Δ{sc.sum()-sb.sum():+7.0f})  sh {dsh(sb):+.2f}->{dsh(sc):+.2f}  "
              f"maxDD {maxdd(sb):+7.0f}->{maxdd(sc):+7.0f}  CVaR5 {cvar5(sb):+6.0f}->{cvar5(sc):+6.0f}  | episodes {imp}+/{wor}- of {len(com)}")
    print(f"  === {name} ===")
    line("ALL", lambda c:c, lambda c:c)
    line("BULL", bull, bull)
    line("BEAR", bearall, bearall)
    line("GRIND", grind, grind)
    line("DEEP", deep, deep)

def main():
    args = sys.argv[1:]
    split = pd.Timestamp("2025-10-04", tz="UTC")
    if "--split" in args:
        i = args.index("--split"); v = args[i+1]; args = args[:i]+args[i+2:]
        split = None if v == "none" else pd.Timestamp(v, tz="UTC")
    base_tag, cand_tags = args[0], args[1:]
    b = load(base_tag)
    print(f"BASE={base_tag}  window {b.index.min().date()}->{b.index.max().date()} ({len(b)} cyc)\n")
    for t in cand_tags:
        c = load(t)
        print(f"########## {t} vs {base_tag} ##########")
        if split is not None and b.index.min() < split:
            for nm, lo, hi in [("OOS 2022..2025-09 HOLD-OUT", None, split), ("IS 2025-10..2026-05", split, None)]:
                def sl(x):
                    m = pd.Series(True, index=x.index)
                    if lo is not None: m &= x.index >= lo
                    if hi is not None: m &= x.index < hi
                    return x[m]
                block(nm, sl(b), sl(c))
        else:
            block("full window", b, c)
        print()

if __name__ == "__main__":
    main()
