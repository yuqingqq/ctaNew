"""Long-tail evaluator for the convexity v3 stack.

Reports overall + per-regime performance for a run's cycles.csv, focused on the TWO
long-tail failure modes flagged in the review:
  * BULL-SQUEEZE  : regime=='bull' — owns the entire baseline maxDD (May-2026 short squeeze,
                    return_1d shorts the recent gainers that then rip). We split bull into
                    the profitable early part vs the squeeze by sign of cumulative equity slope,
                    but the operative target metric is BULL maxDD + BULL total.
  * BEAR-GRIND    : regime=='bear' AND btc_ret_30d >= GRIND_THR (shallow selloff, no reversion) —
                    chronic negative-Sharpe bleed. Must be cut WITHOUT hurting BEAR-DEEP
                    (btc_ret_30d < GRIND_THR, the capitulation short alpha, t~3.2).

Usage:
  python3 -m live.phase_longtail_eval <state_dir> [--base <baseline_state_dir>]
  (state_dir must contain cycles.csv; --base enables folds-beat + delta columns)
"""
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

ANN = np.sqrt(365)
GRIND_THR = -0.25   # bear >= this = shallow grind (bad); < this = deep capitulation (good)

def load(state_dir):
    f = Path(state_dir) / "cycles.csv"
    if not f.exists():
        return None
    c = pd.read_csv(f)
    c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    c = c.sort_values("open_time").set_index("open_time")
    # month-of-history fold for folds-beat (walk-forward proxy: calendar month)
    c["fold"] = c.index.to_period("M").astype(str)
    return c

def dsh(s):
    d = (s.fillna(0) / 1e4).resample("1D").sum()
    return float(d.mean() / d.std() * ANN) if d.std() > 0 else np.nan

def maxdd(s):
    eq = s.fillna(0).cumsum()
    return float((eq - eq.cummax()).min())

def subset_masks(c):
    reg = c["regime"].astype(str)
    b30 = c["btc_ret_30d"]
    return {
        "ALL":        pd.Series(True, index=c.index),
        "BULL":       reg == "bull",
        "SIDE":       reg == "side",
        "BEAR":       reg == "bear",
        "BEAR_DEEP":  (reg == "bear") & (b30 < GRIND_THR),
        "BEAR_GRIND": (reg == "bear") & (b30 >= GRIND_THR),
    }

def row(name, s):
    if s.dropna().empty or len(s) == 0:
        return dict(subset=name, n=0)
    x = s.fillna(0)
    return dict(subset=name, n=int((s.notna()).sum()),
                total=round(float(x.sum()), 0), per_cyc=round(float(x.mean()), 1),
                sharpe=round(dsh(s), 3), maxDD=round(maxdd(s), 0),
                worst=round(float(x.min()), 0))

def report(c, base=None, tag=""):
    masks = subset_masks(c)
    rows = [row(k, c.loc[m, "pnl_bps"]) for k, m in masks.items()]
    df = pd.DataFrame(rows)
    print(f"\n===== {tag or 'run'} =====")
    print(df.to_string(index=False))
    if base is not None:
        # per-fold folds-beat on ALL pnl
        bf = base.groupby("fold")["pnl_bps"].sum()
        cf = c.groupby("fold")["pnl_bps"].sum()
        common = bf.index.intersection(cf.index)
        beat = int((cf[common] > bf[common]).sum())
        # deltas on the key metrics
        bm = subset_masks(base)
        print(f"\n  vs baseline: folds_beat {beat}/{len(common)}")
        for k in ["ALL", "BULL", "BEAR_GRIND", "BEAR_DEEP", "SIDE"]:
            cs, bs = c.loc[masks[k], "pnl_bps"], base.loc[bm[k], "pnl_bps"]
            d_tot = cs.fillna(0).sum() - bs.fillna(0).sum()
            d_sh = dsh(cs) - dsh(bs)
            d_dd = maxdd(cs) - maxdd(bs)
            print(f"    {k:11s} Δtot {d_tot:+8.0f}  Δsharpe {d_sh:+.3f}  ΔmaxDD {d_dd:+8.0f}")
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("state_dir")
    ap.add_argument("--base", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    c = load(a.state_dir)
    if c is None:
        print(f"no cycles.csv in {a.state_dir}"); sys.exit(1)
    base = load(a.base) if a.base else None
    report(c, base, a.tag or Path(a.state_dir).name)
