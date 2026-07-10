"""Is the era-fragility a TAIL problem (hedgeable) or a SIGNAL problem (structural)?

The short leg grinds era-stably (median+, win>50% BOTH eras) but its Sharpe is era-fragile because the
squeeze LEFT tail (#4) is worse OOS. Test: cap the worst-k% short-cycle losses (simulating a squeeze
hedge / stop) and re-measure the short-leg Sharpe in BOTH eras. If tail-capped short is positive in
BOTH eras, the path to era-robustness is HEDGE THE SHORT TAIL (#4 -> DATA1), not time the era.
Vanilla book, clean, residual bps. Descriptive (a hedge feasibility probe, not a deployable overlay).
"""
import numpy as np, pandas as pd
import sys; sys.path.insert(0, "/home/yuqing/ctaNew/live")
from attribution_v4_regime import btc_reg, load
import warnings; warnings.filterwarnings("ignore")

def short_pnl(base, reg):
    rows=[]
    for t,g in base.groupby("open_time"):
        if len(g)<5 or reg.get(t) in (None,"deepbull"): continue   # non-deep-bull (traded 1L/2S)
        S=g.nsmallest(2,"pred"); sp=-S["alpha_A"].mean()*1e4
        if np.isfinite(sp): rows.append((t,sp))
    return pd.DataFrame(rows,columns=["t","S"])

def sh(x,c):
    d=pd.to_datetime(x["t"]).dt.date; dr=x[[c]].groupby(d).sum()[c]
    return dr.mean()/dr.std()*np.sqrt(365) if dr.std()>0 else np.nan

def main():
    reg=btc_reg(); out={}
    for era,bp in (("RECENT","hl_tgt_res_base_cleanfix"),("OOS","hl_v4base_oos_cleanfix")):
        base,_=load(bp,bp); d=short_pnl(base,reg); out[era]=d
        print(f"\n===== {era}: short-leg PnL, raw vs tail-capped (non-deep-bull, n={len(d)}) =====")
        raw=sh(d,"S")
        print(f"  RAW short: Sharpe {raw:+.2f}  mean {d.S.mean():+.1f}  median {d.S.median():+.1f}  win% {(d.S>0).mean()*100:.0f}  worst {d.S.min():+.0f}")
        for k in (2,5,10):
            fl=np.percentile(d.S, k)                    # floor = kth percentile (cap squeeze losses)
            d[f"c{k}"]=d.S.clip(lower=fl)
            print(f"  cap worst {k:>2d}% (floor {fl:+7.0f}): Sharpe {sh(d,f'c{k}'):+.2f}  mean {d[f'c{k}'].mean():+.1f}  "
                  f"(lost tail bps/cyc {d.S.mean()-d[f'c{k}'].mean():+.1f})")
    # joint read: is a single cap era-robust (positive BOTH eras)?
    print("\n  >> ERA-ROBUSTNESS of the tail-capped short (both eras positive?):")
    for k in (2,5,10):
        rr,oo=sh(out['RECENT'].assign(c=out['RECENT'].S.clip(lower=np.percentile(out['RECENT'].S,k))),"c"), \
              sh(out['OOS'].assign(c=out['OOS'].S.clip(lower=np.percentile(out['OOS'].S,k))),"c")
        print(f"     cap {k:>2d}%: recent {rr:+.2f} | OOS {oo:+.2f}  -> {'ERA-ROBUST (both +)' if rr>0 and oo>0 else 'not both-positive'}")
    print("SHORTHEDGEDONE")

if __name__=="__main__":
    main()
