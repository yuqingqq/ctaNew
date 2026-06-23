"""Analyze the per-name concentration-cap sweep (CONC_CAP) vs baseline on the v2 modeled backtest.
Honest checks: (1) cap00 == base no-op; (2) aggregate Sharpe/maxDD/CVaR5/Dec-bleed; (3) PER-FOLD (monthly)
breakdown — does the cap help broadly or just one window?; (4) monotonicity across cap strength.
"""
import sys
import numpy as np, pandas as pd
ANN=np.sqrt(365)
BASE="live/state/v3loop"
RUNS=[("base","dd_repro"),("cap00","dd_cap00"),("cap50","dd_cap50"),("cap40","dd_cap40"),("cap30","dd_cap30")]

def load(d):
    c=pd.read_csv(f"{BASE}/{d}/state/cycles.csv"); c["open_time"]=pd.to_datetime(c["open_time"],utc=True)
    return c.sort_values("open_time").set_index("open_time")

def agg(c):
    dd=c["pnl_bps"].resample("1D").sum(); sh=dd.mean()/dd.std()*ANN
    eq=c["pnl_bps"].cumsum(); mdd=(eq-eq.cummax()).min()
    cv5=c["pnl_bps"][c["pnl_bps"]<=c["pnl_bps"].quantile(.05)].mean()
    dec=c.loc["2025-12-12":"2026-01-13","pnl_bps"].sum()
    return c["pnl_bps"].sum(),sh,mdd,c["pnl_bps"].min(),cv5,dec

cs={tag:load(d) for tag,d in RUNS}
print(f"{'config':8s} {'totPnL':>8} {'Sharpe':>7} {'maxDD':>8} {'worstcyc':>8} {'CVaR5':>7} {'DecBleed':>8}")
for tag,_ in RUNS:
    tp,sh,mdd,wc,cv5,dec=agg(cs[tag])
    print(f"{tag:8s} {tp:>+8.0f} {sh:>+7.2f} {mdd:>+8.0f} {wc:>+8.0f} {cv5:>+7.0f} {dec:>+8.0f}")

# no-op check
b=cs["base"]["pnl_bps"]; z=cs["cap00"]["pnl_bps"]
print(f"\nno-op check (cap00 vs base): max|Δpnl_bps| = {np.abs((b-z).dropna()).max():.6f}  (should be ~0)")

# per-month (fold proxy) Sharpe: base vs cap40
print("\n=== per-month daily Sharpe: base vs cap40 vs cap30 ===")
def msh(c):
    d=c["pnl_bps"].resample("1D").sum()
    return d.groupby(d.index.to_period("M")).apply(lambda x: x.mean()/x.std()*ANN if x.std()>0 else np.nan)
mb,m4,m3=msh(cs["base"]),msh(cs["cap40"]),msh(cs["cap30"])
M=pd.DataFrame({"base":mb,"cap40":m4,"cap30":m3})
print(M.round(2).to_string())
print(f"\nmonths cap40 beats base: {(M.cap40>M.base).sum()}/{M.base.notna().sum()}   cap30: {(M.cap30>M.base).sum()}/{M.base.notna().sum()}")
