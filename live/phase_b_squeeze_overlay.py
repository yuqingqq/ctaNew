"""Phase B (tail loop): can a FREE crowding signal TIME the bear short-squeeze de-gross better than random?
PIT-honest post-hoc overlay on the baseline run (no bot change, no replay):
  squeeze_risk[t] = -mean( lagged funding_rate_z_7d ) over the SELECTED SHORTS at t   (shorts crowded = neg funding
  = squeeze fuel -> high risk). De-gross rule: scale that cycle's pnl by MULT when squeeze_risk[t] > trailing
  percentile PCT (PIT, trailing window). Compare signal-targeted de-gross vs MATCHED PLACEBO (de-gross the same
  NUMBER of random cycles), N seeds, on DAILY Sharpe + tail (worst-1%, maxDD). Signal must rank > p95 of placebo to
  claim a real timing edge (else it's just generic de-gross, like the auto-sizer that tied random).
"""
import numpy as np, pandas as pd
ROOT="/home/yuqing/ctaNew"
LAG=2; WIN=180  # 8h funding lag (PIT); trailing 30d percentile window

c=pd.read_csv(f"{ROOT}/live/state/v3loop/iter5_tilt0/cycles.csv"); c["open_time"]=pd.to_datetime(c["open_time"],utc=True)
c=c.sort_values("open_time").reset_index(drop=True)
pred=pd.read_parquet(f"{ROOT}/live/state/v3loop/iter5_tilt0/predictions.parquet")
pred["open_time"]=pd.to_datetime(pred["open_time"],utc=True)
pan=pd.read_parquet(f"{ROOT}/outputs/vBTC_features/panel_expanded_v0.parquet",
                    columns=["symbol","open_time","funding_rate_z_7d"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fz_lag"]=pan.groupby("symbol")["funding_rate_z_7d"].shift(LAG)   # PIT

# per-cycle short-basket crowding = mean lagged funding_z of SELECTED shorts
sh=pred[pred["selected_short"]].merge(pan[["symbol","open_time","fz_lag"]],on=["symbol","open_time"],how="left")
crowd=sh.groupby("open_time")["fz_lag"].mean().rename("short_fz")
c=c.merge(crowd, on="open_time", how="left")
c["squeeze_risk"]=-c["short_fz"]      # higher = shorts more crowded = more squeeze risk

ANN=np.sqrt(365)
def metrics(pnl_bps):
    s=pd.Series(pnl_bps, index=c["open_time"])
    d=(s.fillna(0)/1e4).resample("1D").sum()
    sh_d=d.mean()/d.std()*ANN
    eq=s.fillna(0).cumsum(); dd=float((eq-eq.cummax()).min())
    x=s.dropna(); w1=float(x.nsmallest(max(1,len(x)//100)).sum())
    return sh_d, dd, w1

base_sh, base_dd, base_w1 = metrics(c["pnl_bps"].values)
print(f"baseline: dailySharpe {base_sh:+.3f}  maxDD {base_dd:.0f}  worst1% {base_w1:.0f}")

def apply_degross(mask, mult):
    p=c["pnl_bps"].values.copy(); p[mask.values]=p[mask.values]*mult; return p

# trailing-PIT percentile of squeeze_risk
sr=c["squeeze_risk"]
for PCT in [80, 85, 90]:
  for MULT in [0.5, 0.0]:
    thr=sr.shift(1).rolling(WIN, min_periods=60).quantile(PCT/100)
    fire=(sr > thr) & sr.notna() & thr.notna()
    n=int(fire.sum())
    if n<5: print(f"PCT{PCT} mult{MULT}: too few fires ({n})"); continue
    sh_s, dd_s, w1_s = metrics(apply_degross(fire, MULT))
    # matched placebo: de-gross n RANDOM cycles (from the same eligible set where thr is defined), 200 seeds
    elig=np.where(thr.notna().values)[0]
    rng=np.random.default_rng(0); shs=[]; w1s=[]
    for k in range(200):
        idx=rng.choice(elig, size=n, replace=False); m=pd.Series(False,index=c.index); m.iloc[idx]=True
        a,_,b=metrics(apply_degross(m, MULT)); shs.append(a); w1s.append(b)
    shs=np.array(shs); w1s=np.array(w1s)
    rank_sh=100*(shs<sh_s).mean(); rank_w1=100*(w1s<w1_s).mean()  # higher worst1% (less negative) = better
    print(f"PCT{PCT} mult{MULT} fires={n:3d}: Sharpe {sh_s:+.3f} (plac mean {shs.mean():+.3f} p95 {np.quantile(shs,.95):+.3f} -> rank p{rank_sh:.0f}) | "
          f"worst1% {w1_s:.0f} (plac mean {w1s.mean():.0f} -> rank p{rank_w1:.0f}) | maxDD {dd_s:.0f}")
