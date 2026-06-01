"""Robustness: short-side G4 placebo across MAX windows and pool definitions.
Also test the LONG side (low-MAX bounce) symmetrically. If neither side beats
random-from-pool at >=p95, the IC does not monetize -> NO-CANDIDATE."""
import pandas as pd, numpy as np
pd.options.mode.chained_assignment=None
GRID=48; K=5
RC='research/convexity_portable_2026-05-20/results/_cache/x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet'
pred=pd.read_parquet(RC,columns=['symbol','open_time','pred','alpha_A'])
pred['ot']=pd.to_datetime(pred['open_time'],utc=True)
pred=pred[(pred.ot.dt.hour%4==0)&(pred.ot.dt.minute==0)].copy()
pan=pd.read_parquet('outputs/vBTC_features/panel_hl70.parquet',columns=['symbol','open_time','return_pct'])
pan['ot']=pd.to_datetime(pan['open_time'],utc=True); pan=pan.sort_values(['symbol','ot'])
pan['trail4h']=pan.groupby('symbol')['return_pct'].shift(GRID)
pan4=pan[(pan.ot.dt.hour%4==0)&(pan.ot.dt.minute==0)].copy().sort_values(['symbol','ot'])
rng=np.random.default_rng(1)
for W in [3,6,12]:
    pan4['MAX']=pan4.groupby('symbol')['trail4h'].transform(lambda s:s.rolling(W,min_periods=3).max())
    m=pred.merge(pan4[['symbol','ot','MAX']],on=['symbol','ot'],how='inner').dropna(subset=['pred','alpha_A','MAX'])
    # SHORT pool = low-pred half; LONG pool = high-pred half
    sm_real=[]; lm_real=[]; srand=[[] for _ in range(200)]; lrand=[[] for _ in range(200)]
    for ot,gg in m.groupby('ot'):
        if len(gg)<4*K: continue
        sp=gg.sort_values('pred').head(max(2*K,len(gg)//2))   # short-eligible
        lp=gg.sort_values('pred').tail(max(2*K,len(gg)//2))   # long-eligible
        if len(sp)<2*K or len(lp)<2*K: continue
        sm_real.append(-sp.sort_values('MAX').tail(K)['alpha_A'].mean())  # short highest-MAX
        lm_real.append( lp.sort_values('MAX').head(K)['alpha_A'].mean())  # long lowest-MAX (bounce)
        for si in range(200):
            srand[si].append(-sp['alpha_A'].iloc[rng.choice(len(sp),K,replace=False)].mean())
            lrand[si].append( lp['alpha_A'].iloc[rng.choice(len(lp),K,replace=False)].mean())
    sr=np.array([np.mean(x) for x in srand]); lr=np.array([np.mean(x) for x in lrand])
    sreal=np.mean(sm_real); lreal=np.mean(lm_real)
    print(f"W={W:>2}: SHORT highest-MAX {sreal*1e4:+.2f}bps p{(sr<sreal).mean()*100:.0f} (rand {sr.mean()*1e4:+.2f}) | "
          f"LONG lowest-MAX {lreal*1e4:+.2f}bps p{(lr<lreal).mean()*100:.0f} (rand {lr.mean()*1e4:+.2f})")
