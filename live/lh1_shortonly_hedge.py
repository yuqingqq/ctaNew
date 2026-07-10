"""LH1: replace the non-deep-bull long lottery with a basket beta-hedge (LH1 pre-reg, addendum 40).

BASELINE = 1L/2S (long top-1 long-pred, short bottom-2 base-pred). TREATMENT = short bottom-2 + long the
equal-weight BASKET (cross-sec mean residual alpha ≈ near-zero selection, low turnover). Non-deep-bull
cycles only (deep-bull mom1d held fixed → cancels). Book-level residual net Sharpe, both eras. Gate:
treatment ≥ baseline in BOTH eras + jackpot-preservation. Committed, vanilla frame (estimator law).
"""
import numpy as np, pandas as pd
import sys; sys.path.insert(0, "/home/yuqing/ctaNew/live")
from attribution_v4_regime import btc_reg, load, COST
import warnings; warnings.filterwarnings("ignore")
WL = WS = 0.5

def build(base, long, reg):
    lg = long.groupby("open_time"); rows=[]; prevL=None; prevS=set()
    for t, g in base.groupby("open_time"):
        if len(g) < 5 or reg.get(t) in (None, "deepbull"): continue   # non-deep-bull only
        try: gl = lg.get_group(t)
        except KeyError: continue
        Lp = gl.nlargest(1, "pred"); S = g.nsmallest(2, "pred")
        if len(Lp) < 1 or len(S) < 2: continue
        la = Lp.iloc[0]["alpha_A"]*1e4; sa = S["alpha_A"].mean()*1e4
        ba = g["alpha_A"].mean()*1e4     # equal-weight basket residual (the hedge's "alpha")
        if not (np.isfinite(la) and np.isfinite(sa) and np.isfinite(ba)): continue
        Ln, Ss = Lp.iloc[0]["symbol"], set(S["symbol"])
        lt = 1.0 if (prevL is None or Ln != prevL) else 0.0          # alt-long turnover
        st = (len(Ss - prevS)/2.0) if prevS else 1.0                 # short turnover (frac of 2)
        cost_short = st*0.5 * WS*COST/0.5                            # short leg cost (WS weight)
        # baseline: alt-long (turnover lt) + short ; treatment: basket-long (~0 turnover) + short
        base_net = WL*la - WS*sa - (lt*0.5*WL*COST/0.5) - (st*0.5*WS*COST/0.5)
        trt_net  = WL*ba - WS*sa - (0.0)                 - (st*0.5*WS*COST/0.5)  # basket ~0 turnover
        rows.append((t, base_net, trt_net, la, ba))
        prevL, prevS = Ln, Ss
    return pd.DataFrame(rows, columns=["t","base","trt","la","ba"])

def sh(x, c):
    d = pd.to_datetime(x["t"]).dt.date; dr = x[[c]].groupby(d).sum()[c]
    return dr.mean()/dr.std()*np.sqrt(365) if dr.std()>0 else np.nan
def mdd(x): e=np.cumsum(x); return float((e-np.maximum.accumulate(e)).min())
def jkp(x): x=np.sort(np.asarray(x,float)); return x[-max(1,len(x)//10):].sum()/x.sum()*100 if x.sum()!=0 else np.nan

def main():
    reg = btc_reg(); res={}
    for era, bp, lp in (("RECENT 2025-10+","hl_tgt_res_base_cleanfix","hl_tgt_res_long_cleanfix"),
                       ("OOS 2023-25","hl_v4base_oos_cleanfix","hl_v4long_oos_cleanfix")):
        base, long = load(bp, lp); d = build(base, long, reg)
        bS, tS = sh(d,"base"), sh(d,"trt")
        print(f"\n===== {era}: non-deep-bull book (n={len(d)}), residual net, pinned cost =====")
        print(f"  BASELINE (1L/2S alt-long): Sharpe {bS:+.2f}  net {d.base.mean():+.1f}  maxDD {mdd(d.base.values):+.0f}  jackpot {jkp(d.base.values):.0f}%")
        print(f"  TREATMENT (short+basket) : Sharpe {tS:+.2f}  net {d.trt.mean():+.1f}  maxDD {mdd(d.trt.values):+.0f}  jackpot {jkp(d.trt.values):.0f}%")
        print(f"  Δ Sharpe {tS-bS:+.2f} | alt-long resid mean {d.la.mean():+.1f} (the lottery) vs basket resid mean {d.ba.mean():+.1f} (the hedge)")
        res[era] = tS - bS
    both = all(v > 0 for v in res.values())
    one = any(v > 0 for v in res.values())
    print(f"\n  >> GATE (treatment ≥ baseline BOTH eras): {'PASS -> candidate' if both else ('FAIL — era-fragile (one-era only)' if one else 'FAIL — treatment worse')}")
    print(f"     recent Δ {res['RECENT 2025-10+']:+.2f} | OOS Δ {res['OOS 2023-25']:+.2f}")
    print("LH1DONE")

if __name__ == "__main__":
    main()
