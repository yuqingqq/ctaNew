"""Analyze an ab_split experiment vs the wfund baseline: overall + per-month-fold robustness.
Usage: python3 live/opt_analyze.py --base live/state/opt_<exp> --label <exp>
Prints a verdict line and per-fold table; the runner appends stdout to insights.md.
"""
import sys, argparse, json
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/home/yuqing/ctaNew")
ANN = np.sqrt(6*365)
BASELINE = REPO/"live/state/ab_wfund175"   # SIZING_MODE=equal per-symbol two-book

def load_eq(base, policy):
    f = Path(base)/policy/"combine"/"twobook_equity.csv"
    if not f.exists(): return None
    d = pd.read_csv(f); d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d["mon"] = d["open_time"].dt.to_period("M")
    return d

def stats(d):
    x = d["pnl_bps_combined_fill0"].fillna(0)
    sh = x.mean()/x.std()*ANN if x.std()>0 else 0.0
    eq = x.cumsum(); dd = (eq - eq.cummax()).min()
    return sh, x.sum(), dd

def fold_sharpe(d):
    return d.groupby("mon")["pnl_bps_combined_fill0"].apply(lambda v: v.mean()/v.std()*ANN if v.std()>0 else 0.0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base"); ap.add_argument("--label", default="exp")
    a = ap.parse_args()
    print(f"\n################  {a.label}  ################")
    for pol in ("monthly","daily"):
        t = load_eq(a.base, pol); b = load_eq(BASELINE, pol)
        if t is None: print(f"[{pol}] MISSING treatment equity"); continue
        tsh,tpnl,tdd = stats(t); bsh,bpnl,bdd = stats(b)
        # per-month paired
        tf, bf = fold_sharpe(t), fold_sharpe(b)
        idx = sorted(set(tf.index)&set(bf.index))
        wins = sum(1 for m in idx if tf[m] >= bf[m])
        # paired monthly PnL diff
        tm = t.groupby("mon")["pnl_bps_combined_fill0"].sum(); bm = b.groupby("mon")["pnl_bps_combined_fill0"].sum()
        diff = (tm - bm).reindex(idx).fillna(0); pos = int((diff>0).sum())
        print(f"[{pol:7s}] Sharpe {tsh:+.3f} (base {bsh:+.3f}, Δ{tsh-bsh:+.3f}) | maxDD {tdd:+.0f} (base {bdd:+.0f}) | "
              f"totPnL {tpnl:+.0f} (base {bpnl:+.0f}) | fold-Sharpe-wins {wins}/{len(idx)} | PnL-diff-pos-months {pos}/{len(idx)}")
        if pol=="monthly":
            tab = pd.DataFrame({"treat_sh":tf.reindex(idx).round(2),"base_sh":bf.reindex(idx).round(2),
                                "pnl_diff":diff.round(0)})
            print("  per-month:\n   "+tab.to_string().replace("\n","\n   "))
    # verdict heuristic (monthly)
    t = load_eq(a.base,"monthly"); b = load_eq(BASELINE,"monthly")
    if t is not None:
        tsh,_,tdd = stats(t); bsh,_,bdd = stats(b)
        dd_better = tdd > bdd  # less negative
        print(f"  >>> VERDICT: Δsharpe {tsh-bsh:+.3f}, DD {'BETTER' if dd_better else 'worse'} "
              f"({tdd:+.0f} vs {bdd:+.0f}) — {'PROMISING, validate placebo+nested' if (tsh-bsh)>0.15 or (dd_better and tsh-bsh>-0.10) else 'no edge'}")

if __name__ == "__main__": main()
