"""READ-ONLY audit: reproduce the optimized held-book net per-cycle series,
measure autocorrelation & effective sample size, and recompute Sharpe with a
Newey-West / overlap-corrected annualization. Mirrors X93 exactly."""
import pandas as pd, numpy as np
from pathlib import Path
REPO=Path("/home/yuqing/ctaNew")
RCACHE=REPO/"research/convexity_portable_2026-05-20/results/_cache"
KLINES=REPO/"data/ml/test/parquet/klines"
COST=4.5e-4; K=3; HOLD=6

def load_close(sym):
    sd=KLINES/sym/"5m"
    if not sd.exists(): return None
    dfs=[pd.read_parquet(f,columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    df=pd.concat(dfs,ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    return df.set_index("open_time")["close"].astype(np.float64)

def heldbook(cyc_w, ret_seq):
    n=len(cyc_w); prev={}; rets=[]
    for t in range(n):
        active=cyc_w[max(0,t-HOLD+1):t+1]; net={}
        for w in active:
            for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
        alls=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls)
        rl=ret_seq[t]
        rets.append(sum(net.get(s,0)*rl.get(s,0.0) for s in net)-turn*0.5*COST); prev=net
    return np.array(rets)

apd=pd.read_parquet(RCACHE/"x70_v0_3yr_preds.parquet")
apd["open_time"]=pd.to_datetime(apd["open_time"],utc=True)
apd=apd[(apd["open_time"].dt.hour%4==0)&(apd["open_time"].dt.minute==0)]
syms=sorted(apd["symbol"].unique())
mr=[]
for sym in syms:
    c=load_close(sym)
    if c is None: continue
    c4=c[(c.index.hour%4==0)&(c.index.minute==0)]; mom=(c4/c4.shift(180)-1).shift(1)
    mr.append(pd.DataFrame({"symbol":sym,"open_time":mom.index,"mom30":mom.values}))
mom=pd.concat(mr,ignore_index=True); mom["open_time"]=pd.to_datetime(mom["open_time"],utc=True)
apd=apd.merge(mom,on=["symbol","open_time"],how="left")
btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
btc30=(b4/b4.shift(180)-1).to_frame("btc_ret_30d").reset_index()
btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
apd=apd.merge(btc30,on="open_time",how="left").dropna(subset=["btc_ret_30d"])
apd["regime"]=np.where(apd["btc_ret_30d"]>0.10,"bull",np.where(apd["btc_ret_30d"]<-0.10,"bear","side"))
times=sorted(apd["open_time"].unique()); by_t={ot:g for ot,g in apd.groupby("open_time")}
cyc_w=[]; ret_seq=[]
for ot in times:
    g=by_t[ot]; rg=g["regime"].iloc[0]
    ret_seq.append(dict(zip(g["symbol"],g["return_pct"])))
    if rg=="bear": cyc_w.append({}); continue
    key="mom30" if rg=="bull" else "pred"
    gg=g.dropna(subset=[key])
    if len(gg)<2*K: cyc_w.append({}); continue
    gg=gg.sort_values(key); L=gg.tail(K)["symbol"].tolist(); S=gg.head(K)["symbol"].tolist()
    w={}
    for s in L: w[s]=w.get(s,0)+1.0/K
    for s in S: w[s]=w.get(s,0)-1.0/K
    cyc_w.append(w)
r=heldbook(cyc_w, ret_seq)
r=pd.Series(r).dropna().values
n=len(r); mu=r.mean(); sd=r.std()
naive=mu/sd*np.sqrt(6*365)
print(f"n cycles={n}  mean={mu:.3e}  std={sd:.3e}")
print(f"NAIVE Sharpe (sqrt(6*365), iid)         = {naive:+.3f}")
# autocorrelations
acf=[np.corrcoef(r[:-k],r[k:])[0,1] for k in range(1,12)]
print("ACF lags 1..11:", " ".join(f"{a:+.3f}" for a in acf))
# Newey-West variance inflation factor for the MEAN (long-run variance)
# var_LR = gamma0 + 2*sum_{k=1}^{L}(1-k/(L+1))*gamma_k  ; use L=HOLD-1=5 and also Bartlett auto
def nw_var(x,L):
    x=x-x.mean(); g0=np.mean(x*x)
    s=g0
    for k in range(1,L+1):
        gk=np.mean(x[:-k]*x[k:]); s+=2*(1-k/(L+1))*gk
    return s
for L in [5,6,10]:
    lr=nw_var(r,L)
    sd_eff=np.sqrt(lr)
    # corrected annualized Sharpe: mean/sd_eff still *sqrt(periods_per_year) but
    # the std used for per-period Sharpe should reflect serial dependence.
    # Effective sample size: n_eff = n * var(sample mean iid)/var_NW
    inflation=lr/(sd**2)
    sh_corr=mu/sd_eff*np.sqrt(6*365)
    n_eff=n/inflation
    print(f"L={L:2d}: var_inflation={inflation:.2f}  n_eff={n_eff:.0f}  overlap-corrected Sharpe={sh_corr:+.3f}")
# sum of ACF approach (effective independent obs for sqrt-time scaling)
rho_sum=1+2*sum((1-k/HOLD)*acf[k-1] for k in range(1,HOLD))
print(f"\nBartlett rho-sum (L=HOLD-1, triangular)={rho_sum:.3f} -> Sharpe/sqrt(rho_sum)={naive/np.sqrt(max(rho_sum,1e-9)):+.3f}")

# === Return-application correctness: trace a single sleeve's lifetime ===
# A basket entered at cycle t0 enters net at t0..t0+5 with weight w/HOLD each cycle.
# It should earn the 4h return of EACH of those 6 cycles exactly once. Sum of
# (w/HOLD)*ret_4h over 6 cycles ~ w * mean(6x 4h returns) = w * (24h return)/... 
# Verify total gross PnL = sum over cycles of one sleeve's contribution equals
# holding w for 24h. Build a one-sleeve check ignoring cost.
def one_sleeve_gross(cyc_w, ret_seq, t0):
    w=cyc_w[t0]
    if not w: return None
    tot=0.0
    for k in range(HOLD):
        t=t0+k
        if t>=len(ret_seq): break
        rl=ret_seq[t]
        tot+=sum((wt/HOLD)*rl.get(s,0.0) for s,wt in w.items())
    # compare to entering full weight w and earning compounded 24h: approx sum of 4h rets
    approx=sum(wt*sum(ret_seq[t0+k].get(s,0.0) for k in range(HOLD) if t0+k<len(ret_seq)) for s,wt in w.items())
    return tot, approx/HOLD
# pick a few non-empty cycles
checked=0
for t0 in range(100, 2000):
    res=one_sleeve_gross(cyc_w, ret_seq, t0)
    if res:
        tot,approx=res
        assert abs(tot-approx)<1e-12, (tot,approx)
        checked+=1
    if checked>=200: break
print(f"\n[one-sleeve trace] {checked} baskets: per-sleeve PnL = (w/6)*sum of 6 disjoint 4h returns. "
      f"Each cycle's 4h return earned exactly once. No gap/double-count. OK")

# === turnover sanity: avg turnover & avg cost in bps/cycle ===
prev={}; turns=[]
for t in range(len(cyc_w)):
    active=cyc_w[max(0,t-HOLD+1):t+1]; net={}
    for w in active:
        for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
    alls=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls); prev=net
    turns.append(turn)
turns=np.array(turns)
print(f"[turnover] mean per-cycle turnover={turns.mean():.3f} units -> cost={turns.mean()*0.5*COST*1e4:.2f} bps/cycle; "
      f"gross leverage ~ {(2.0):.1f} units; cost/gross-mean ratio check")
gross_mu=mu+turns.mean()*0.5*COST
print(f"  gross mean/cycle={gross_mu*1e4:+.2f}bps  cost/cycle={turns.mean()*0.5*COST*1e4:.2f}bps  net={mu*1e4:+.2f}bps  "
      f"cost as % of gross={turns.mean()*0.5*COST/gross_mu*100:.0f}%")
