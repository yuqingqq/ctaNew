"""Long-leg vs short-leg alpha by regime x era — VANILLA model book (pre-gate), clean books.

Answers "which side carries the alpha." LONG = top-1 by long-book pred, realized residual alpha (we
long -> +alpha). SHORT = bottom-2 by base-book pred, realized residual alpha (we short -> PnL =
-alpha). Per-leg net charges that leg's own turnover at pinned cost. NOT production-gated (same vanilla
frame as attribution_v4_regime.py) — the lens for where the signal is, not what production trades.
"""
import numpy as np, pandas as pd
import sys; sys.path.insert(0, "/home/yuqing/ctaNew/live")
from attribution_v4_regime import btc_reg, load, COST
import warnings; warnings.filterwarnings("ignore")

def legs(base, long, reg):
    lg = long.groupby("open_time"); rows=[]; prevL=None; prevS=set()
    for t, g in base.groupby("open_time"):
        if len(g) < 5 or t not in reg: continue
        try: gl = lg.get_group(t)
        except KeyError: continue
        Lp = gl.nlargest(1, "pred"); S = g.nsmallest(2, "pred")
        if len(Lp) < 1 or len(S) < 2: continue
        la = Lp.iloc[0]["alpha_A"]*1e4; sa = S["alpha_A"].mean()*1e4
        if not (np.isfinite(la) and np.isfinite(sa)): continue
        Ln, Ss = Lp.iloc[0]["symbol"], set(S["symbol"])
        # per-leg turnover (weight-1 leg convention for a like-for-like leg comparison): long swap=1.0,
        # short = frac of 2 names changed; cost = turn*0.5*COST
        lt = 1.0 if (prevL is None or Ln != prevL) else 0.0
        st = (len(Ss - prevS)/2.0) if prevS else 1.0
        rows.append((t, reg[t], la, la - lt*0.5*COST, -sa, -sa - st*0.5*COST, lt, st))
        prevL, prevS = Ln, Ss
    return pd.DataFrame(rows, columns=["t","reg","Lg","Ln_","Sg","Sn","lt","st"])

def sh(x):
    d = pd.to_datetime(x["t"]).dt.date
    def a(c):
        dr = x[[c]].groupby(d).sum()[c]; return dr.mean()/dr.std()*np.sqrt(365) if dr.std()>0 else np.nan
    return a

def main():
    reg = btc_reg()
    for era, bp, lp in (("OOS 2023-25","hl_v4base_oos_cleanfix","hl_v4long_oos_cleanfix"),
                        ("RECENT 2025-10+","hl_tgt_res_base_cleanfix","hl_tgt_res_long_cleanfix")):
        base, long = load(bp, lp); df = legs(base, long, reg); A = sh(df)
        print(f"\n===== {era}: LONG vs SHORT leg, residual alpha (vanilla book, pre-gate, net {COST}bps) =====")
        print(f"  {'regime':<9}{'n':>5} | {'LONG net':>9} {'L Sh':>6} {'L turn':>7} | {'SHORT net':>10} {'S Sh':>6} {'S turn':>7}")
        for rg in ["side","bear","bull","deepbull","ALL"]:
            s = df if rg=="ALL" else df[df.reg==rg]
            if len(s) < 3: continue
            a = sh(s)
            print(f"  {rg:<9}{len(s):>5} | {s['Ln_'].mean():>+9.1f} {a('Ln_'):>+6.2f} {s['lt'].mean()*100:>6.0f}% | "
                  f"{s['Sn'].mean():>+10.1f} {a('Sn'):>+6.2f} {s['st'].mean()*100:>6.0f}%")
    print("\n(LONG net = long-pick residual alpha − turnover; SHORT net = −short-pick residual alpha − turnover.")
    print(" Both want +. Compares where the tradeable edge lives.)\nLSDONE")

if __name__ == "__main__":
    main()
