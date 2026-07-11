"""event_legsplit (correction to addendum 50): decompose the 1L/2S book into LONG and SHORT legs to
test whether the "90%-from-8-days" event-concentration is the SHORT crash-wins (as addendum 50 implied)
or a LONG-leg artifact. Per-leg hit-rate + median/mean (grind vs lottery) and daily grind-vs-tails per leg.
Clean books, both eras, non-deep-bull, residual PnL (bps). NaN labels (clean-panel gap-fixes) dropped.
"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "live")
from attribution_v4_regime import btc_reg, load
import warnings; warnings.filterwarnings("ignore")

def legs(base, long, reg):
    lg = long.groupby("open_time"); Lp = []; Sp = []; rows = []
    for t, g in base.groupby("open_time"):
        if len(g) < 5 or reg.get(t) in (None, "deepbull"): continue
        try: gl = lg.get_group(t)
        except KeyError: continue
        L = gl.nlargest(1, "pred"); S = g.nsmallest(2, "pred")
        if len(L) < 1 or len(S) < 2: continue
        la = float(L.iloc[0]["alpha_A"] * 1e4)
        sl = [float(-r["alpha_A"] * 1e4) for _, r in S.iterrows()]
        if np.isfinite(la): Lp.append(la)
        Sp += [x for x in sl if np.isfinite(x)]
        rows.append((pd.Timestamp(t).normalize(), 0.5 * la, 0.5 * np.nanmean(sl)))
    daily = pd.DataFrame(rows, columns=["day", "long_c", "short_c"]).groupby("day").sum()
    return np.array(Lp), np.array(Sp), daily

def leg_stats(a, label):
    a = a[np.isfinite(a)]
    return (f"  {label}: hit={(a>0).mean()*100:4.1f}%  median={np.median(a):+6.1f}  mean={a.mean():+6.1f}  "
            f"p05={np.percentile(a,5):+6.0f}  p95={np.percentile(a,95):+6.0f}")

def daily_split(daily, col, label):
    s = daily[col].sort_values(ascending=False); tot = s.sum()
    top8 = s.head(8).sum(); rest = s.iloc[8:]
    verd = "grinds UP" if rest.mean() > 0 else "BLEEDS"
    return (f"  {label} daily: total {tot:+8.0f} | top-8 {top8:+7.0f} ({top8/tot*100 if tot else float('nan'):+4.0f}%) | "
            f"other {len(rest)}d {rest.sum():+7.0f} = {rest.mean():+5.1f}/day  [{verd}]")

def main():
    reg = btc_reg()
    for era, bp, lp in (("RECENT", "hl_tgt_res_base_cleanfix", "hl_tgt_res_long_cleanfix"),
                        ("OOS", "hl_v4base_oos_cleanfix", "hl_v4long_oos_cleanfix")):
        base, long = load(bp, lp); Lp, Sp, daily = legs(base, long, reg)
        print(f"\n===== {era} =====")
        print("per-LEG residual PnL (bps):")
        print(leg_stats(Lp, "LONG  (top-1 pred)  "))
        print(leg_stats(Sp, "SHORT (bottom-2 pred)"))
        print("book-contribution daily (WL=WS=0.5, gross of cost):")
        print(daily_split(daily, "short_c", "SHORT"))
        print(daily_split(daily, "long_c", "LONG "))
        b = (daily["long_c"] + daily["short_c"]).sort_values(ascending=False)
        rest = b.iloc[8:]
        print(f"  BOOK  daily: non-tail grind {rest.mean():+.1f}/day  [{'net-positive' if rest.mean()>0 else 'net-BLEED'}]")
    print("LEGSPLITDONE")

if __name__ == "__main__":
    main()
