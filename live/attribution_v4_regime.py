"""COMMITTED per-regime attribution of the v4 1L/2S book, LEAKED vs CLEAN books, at the PINNED cost.

Audit remediation (addendum 24/25): (1) the exact V4_LIMITATIONS per-regime cells came from an
UNCOMMITTED script — this is the committed replacement; (2) it charges the PINNED cost
`turn*0.5*COST` (COST=9bps), not the table's 0.25*9 half-charge; (3) it runs on BOTH the leaked OOS
books (hl_v4base_oos/hl_v4long_oos, corrupt 2025-02-28 label) and the clean-panel books
(*_cleanfix, gap labels NaN'd) to quantify the leak's inflation.

Book: long = top-1 by long-book pred; short = bottom-2 by base-book pred; weights 0.5/0.5. Frame =
residual alpha (alpha_A, the v4 target) and naked (return_pct) — both stored in the book from the
panel used to generate it (so the leaked book carries the corrupt label, the clean book the NaN).
Regime = btc_ret_30d (bear<-0.10 / side / bull>0.10) from local BTC 4h. Sharpe = daily-aggregated ×√365.
"""
import sys, glob
import numpy as np, pandas as pd
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
R = Path("/home/yuqing/ctaNew"); D = R/"live/state/convexity"
COST = 9.0; WL = WS = 0.5

def btc_reg():
    fs = sorted(glob.glob(str(R/"data/ml/test/parquet/klines/BTCUSDT/5m/*.parquet")))
    b = pd.concat([pd.read_parquet(f, columns=["open_time","close"]) for f in fs], ignore_index=True)
    b["open_time"] = pd.to_datetime(b["open_time"], utc=True)
    b = b.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")["close"]
    b4 = b[(b.index.hour % 4 == 0) & (b.index.minute == 0)]
    r30 = b4 / b4.shift(180) - 1
    return {t: ("bull" if v > 0.10 else "bear" if v < -0.10 else "side") for t, v in r30.items() if np.isfinite(v)}

def load(pfx):
    def L(p):
        d = pd.read_parquet(D/f"{p}/v0full_hl60.parquet", columns=["symbol","open_time","pred","alpha_A","return_pct"])
        d["open_time"] = pd.to_datetime(d["open_time"], utc=True); return d
    base = L(f"hl_v4base_oos{pfx}"); long = L(f"hl_v4long_oos{pfx}")
    return base, long

def attribute(base, long, reg):
    """per-cycle 1L/2S net (bps) in residual + naked frames, with pinned turnover cost, tagged by regime."""
    lg = long.groupby("open_time"); rows = []
    prevL = None; prevS = set()
    for t, g in base.groupby("open_time"):
        if len(g) < 5 or t not in reg: continue
        try: gl = lg.get_group(t)
        except KeyError: continue
        Lp = gl.nlargest(1, "pred")
        S = g.nsmallest(2, "pred")
        if len(Lp) < 1 or len(S) < 2: continue
        # frames: residual (alpha_A) and naked (return_pct), in bps
        la, lr = Lp.iloc[0]["alpha_A"]*1e4, Lp.iloc[0]["return_pct"]*1e4
        sa, sr = S["alpha_A"].mean()*1e4, S["return_pct"].mean()*1e4
        if not (np.isfinite(la) and np.isfinite(sa)): continue   # clean book: corrupt cycle -> NaN -> dropped
        Ln, Ss = Lp.iloc[0]["symbol"], set(S["symbol"])
        # pinned turnover: sum|Δw| ; long leg w=0.5 (swap->1.0), each short w=0.25 (swap->0.5)
        turn = (1.0 if (prevL is None or Ln != prevL) else 0.0) + 0.5*len(Ss - prevS if prevS else Ss)
        cost = turn * 0.5 * COST
        rows.append((t, reg[t], WL*la - WS*sa - cost, WL*lr - WS*sr - cost))
        prevL, prevS = Ln, Ss
    df = pd.DataFrame(rows, columns=["t","reg","net_resid","net_naked"])
    return df

def sh(x):  # daily-aggregate then annualize
    d = pd.to_datetime(x["t"]).dt.date
    dr = x[["net_resid","net_naked"]].groupby(d).sum()
    return {c: (dr[c].mean()/dr[c].std()*np.sqrt(365) if dr[c].std() > 0 else np.nan) for c in ["net_resid","net_naked"]}

def main():
    reg = btc_reg()
    for label, pfx in (("LEAKED (hl_v4base_oos)", ""), ("CLEAN  (_cleanfix)", "_cleanfix")):
        base, long = load(pfx); df = attribute(base, long, reg)
        S = sh(df)
        print(f"\n===== {label}: OOS per-regime 1L/2S net, PINNED 0.5x9 cost =====")
        print(f"  {'regime':<6} {'n':>5} | {'resid net':>10} {'resid Sh':>9} | {'naked net':>10} {'naked Sh':>9}")
        for rg in ["side","bear","bull"]:
            s = df[df.reg == rg]
            if not len(s): continue
            ss = sh(s)
            print(f"  {rg:<6} {len(s):>5} | {s.net_resid.mean():>+10.1f} {ss['net_resid']:>+9.2f} | {s.net_naked.mean():>+10.1f} {ss['net_naked']:>+9.2f}")
        print(f"  ALL    {len(df):>5} | {df.net_resid.mean():>+10.1f} {S['net_resid']:>+9.2f} | {df.net_naked.mean():>+10.1f} {S['net_naked']:>+9.2f}")
    print("\nATTRIBDONE")

if __name__ == "__main__":
    main()
