"""Regime-tailoring via an OPPORTUNITY gate (full-history validated). Root cause of negative years: the cross-
sectional spread (long_ret+short_ret) collapses to ~cost in low-dispersion/grinding-trend regimes (2024 bull wash,
2022/2025). Fix: trade only when the recent OPPORTUNITY is large — combine (a) equity-curve trend (strategy trailing
PnL) and (b) trailing gross-spread, each vs its PIT trailing median. Validated on 2022-2026 vs matched placebo.
  python live/phase_regime_gate.py
"""
import numpy as np, pandas as pd
ROOT="/home/yuqing/ctaNew"; ANN=np.sqrt(365)
c=pd.read_csv(f"{ROOT}/live/state/v3loop/fullhist/cycles.csv"); c["open_time"]=pd.to_datetime(c["open_time"],utc=True)
c=c.sort_values("open_time").reset_index(drop=True); r=c["pnl_bps"].fillna(0)/1e4
def dsh(x): d=pd.Series(np.asarray(x),index=c["open_time"]).resample("1D").sum(); return d.mean()/d.std()*ANN if d.std()>0 else np.nan
c["gross_spread"]=c["long_ret_bps"]+c["short_ret_bps"]
s_eq=r.shift(1).rolling(360).sum(); s_sp=c["gross_spread"].shift(1).rolling(180).mean()
t_eq=s_eq.shift(1).rolling(360,min_periods=90).quantile(0.5); t_sp=s_sp.shift(1).rolling(360,min_periods=90).quantile(0.5)
on=((s_eq>t_eq)&(s_sp>t_sp)).fillna(False)
elig=(t_eq.notna()&t_sp.notna()).values; n=int(on.sum())
sh=dsh(r.where(on,0.0)); base=dsh(r)
rng=np.random.default_rng(11); idx=np.where(elig)[0]
ps=np.array([dsh(r.where(pd.Series((lambda m:(m.__setitem__(rng.choice(idx,n,replace=False),True) or m))(np.zeros(len(c),bool)),index=c.index),0.0)) for _ in range(300)])
print(f"baseline all-weather dSharpe {base:+.3f}")
print(f"COMBINED opportunity gate: dSharpe {sh:+.3f} (trades {100*n/len(c):.0f}% cycles) vs placebo mean {ps.mean():+.3f} p95 {np.quantile(ps,.95):+.3f} -> rank p{100*(ps<sh).mean():.0f}")
cc=c.copy(); cc["rf"]=r.where(on,0.0).values
for yr,g in cc.groupby(cc["open_time"].dt.year):
    d=pd.Series(g["rf"].values,index=g["open_time"]).resample("1D").sum(); s=d.mean()/d.std()*ANN if d.std()>0 else float('nan')
    print(f"  {yr}: gated {s:+.2f} (on {100*on[c['open_time'].dt.year==yr].mean():.0f}%)")
print("CAVEAT: lookbacks (60d eq / 30d spread / 360-cyc median) are choices -> nested-OOS validate before live.")
