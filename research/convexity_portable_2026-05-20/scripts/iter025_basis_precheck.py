"""iter-025 spot-perp BASIS cross-sectional pre-check (fail-fast).
Idea: instantaneous spot-perp basis dislocation (perp rich/cheap vs spot) as a
cross-sectional alpha distinct from funding (the integral) and from price-momentum.
Free data (Binance spot+perp). Pre-built panel: outputs/vBTC_features_spot/spot_panel.parquet
(20 HL70 symbols, 2025-07..2026-04). Features: sp_basis_4h, sp_basis_z1d, sp_retdiff_4h, sp_taker_imb_1d.

Fail-fast order:
 (1) Univariate XS-IC vs fwd alpha-residual (alpha_A) on 4h grid.
 (2) Orthogonality to pred.
 (3) DECISIVE construction-layer marginal pre-check (iter-022/023 wall): within the
     pred-conditioned long/short pools, does tilting by basis pick better names than a
     matched-random-K from the SAME pool (200 seeds, >=p95)?
Note: only 20/70 HL70 syms have spot -> this is the FAVORABLE in-sample subset; a fail here
is decisive (a real build would need spot for all 70 + EXT transport, which we don't have).
"""
import pandas as pd, numpy as np
pd.options.mode.chained_assignment=None
K=5
RC='research/convexity_portable_2026-05-20/results/_cache/x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet'
pred=pd.read_parquet(RC,columns=['symbol','open_time','pred','alpha_A','return_pct'])
sp=pd.read_parquet('outputs/vBTC_features_spot/spot_panel.parquet')
bcols=['sp_basis_4h','sp_basis_z1d','sp_retdiff_4h','sp_taker_imb_1d']

# restrict to 4h entry grid FIRST (cheap), both sides on open_time
pred['ot']=pd.to_datetime(pred['open_time'],utc=True)
pred=pred[(pred.ot.dt.hour%4==0)&(pred.ot.dt.minute==0)].copy()
sp['ot']=pd.to_datetime(sp['open_time'],utc=True)
sp=sp[(sp.ot.dt.hour%4==0)&(sp.ot.dt.minute==0)].copy()
m=pred.merge(sp[['symbol','ot']+bcols],on=['symbol','ot'],how='inner')
m=m.dropna(subset=['pred','alpha_A','sp_basis_4h'])
print('merged 4h-grid rows',len(m),'cycles',m.ot.nunique(),'syms',m.symbol.nunique())

# rank-based IC per cycle, vectorized via groupby rank then corr
def xs_ic(df,sig,tgt='alpha_A'):
    d=df[['ot',sig,tgt]].dropna()
    rs=d.groupby('ot')[sig].rank(); rt=d.groupby('ot')[tgt].rank()
    tmp=pd.DataFrame({'ot':d['ot'].values,'rs':rs.values,'rt':rt.values})
    ics=tmp.groupby('ot').apply(lambda x:np.corrcoef(x.rs,x.rt)[0,1] if len(x)>3 and x.rs.std()>0 and x.rt.std()>0 else np.nan).dropna()
    return ics
print('\n=== (1) Univariate XS-IC vs fwd alpha_A ===')
for sig in bcols+['pred']:
    ics=xs_ic(m,sig)
    if len(ics)<10: print(sig,'few'); continue
    t=ics.mean()/(ics.std()/np.sqrt(len(ics)))
    print(f'  IC({sig:16s}) mean {ics.mean():+.4f} t {t:+.2f} n {len(ics)}')

print('\n=== (2) XS corr(basis,pred) ===')
for sig in bcols:
    ics=xs_ic(m,sig,'pred')
    print(f'  corr({sig:16s},pred) mean {ics.mean():+.4f}')

print('\n=== (3) DECISIVE construction-layer marginal (pred pools, 200 random seeds) ===')
# Hypothesis sign: perp-rich (high basis) -> perp overpriced -> underperforms -> SHORT; cheap -> LONG.
# Test BOTH the basis level and the basis_z. For each cycle:
#  long pool = top-half by pred; long_basis_tilt = within pool, LOWEST basis K (cheap perp).
#  short pool = bottom-half by pred; short_basis_tilt = within pool, HIGHEST basis K (rich perp).
rng=np.random.default_rng(0)
for sig in ['sp_basis_4h','sp_basis_z1d','sp_retdiff_4h']:
    mm=m.dropna(subset=[sig])
    long_real=[]; short_real=[]; long_rand=[[] for _ in range(200)]; short_rand=[[] for _ in range(200)]
    nc=0
    for ot,gg in mm.groupby('ot'):
        if len(gg)<4*K: continue
        nc+=1; gg=gg.copy()
        n=len(gg); half=max(2*K,n//2)
        lp=gg.sort_values('pred').tail(half)   # long-eligible (high pred)
        sp_=gg.sort_values('pred').head(half)  # short-eligible (low pred)
        # long: cheapest-perp (lowest basis) K ; PnL = +mean fwd alpha
        long_real.append(lp.sort_values(sig).head(K)['alpha_A'].mean())
        # short: richest-perp (highest basis) K ; PnL = -mean fwd alpha
        short_real.append(-sp_.sort_values(sig).tail(K)['alpha_A'].mean())
        for si in range(200):
            li=rng.choice(len(lp),K,replace=False); long_rand[si].append(lp['alpha_A'].iloc[li].mean())
            si2=rng.choice(len(sp_),K,replace=False); short_rand[si].append(-sp_['alpha_A'].iloc[si2].mean())
    if nc<10: print(f'  {sig}: too few cycles ({nc})'); continue
    lr=np.mean(long_real); sr=np.mean(short_real)
    lrand=np.array([np.mean(x) for x in long_rand]); srand=np.array([np.mean(x) for x in short_rand])
    lp_pct=(lrand<lr).mean()*100; sp_pct=(srand<sr).mean()*100
    print(f'  {sig:16s} cycles {nc}')
    print(f'     LONG cheap-perp  : {lr*1e4:+.2f} bps/cyc  rank p{lp_pct:.0f}  (rand {lrand.mean()*1e4:+.2f})')
    print(f'     SHORT rich-perp  : {sr*1e4:+.2f} bps/cyc  rank p{sp_pct:.0f}  (rand {srand.mean()*1e4:+.2f})')
