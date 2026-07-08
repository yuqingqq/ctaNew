"""TIP-ACCURACY v2 — production-mirrored, significance-gated tip evaluator for candidate pred sets.

Upgrades over tip_accuracy.py (per ALPHA_TESTING_PLAN_REVIEW.md):
  1. Mirrors the ACTUAL selection rule: long = top-K_LONG by the LONG-book pred, short = bottom-K_SHORT
     by the SHORT-book pred (two-book, asymmetric K, default 1/2), within the ELIGIBLE universe
     (maturity >= 180d per cycle + hygiene from the bot's universe snapshot).
  2. Regime split from the bot's own hysteresis labels (cycles.csv) — a tip lift in bull is unusable.
  3. Significance on the PAIRED per-cycle diff (candidate tip - base tip): stride-6 non-overlap t,
     5-day-block bootstrap CI, and the concentration gates applied to the DIFF (top-3 share, halves).
  4. Placebo: same within-cycle shuffle of the target for both models -> diff must collapse.
  5. Dispersion-normalized tip (fraction of that cycle's cross-sectional spread) so high-vol months
     don't dominate the mean.
Verdict is a SCREEN (kill/proceed), never an adopt signal — adoption stays with the full frozen-stack
replay at fair fees + OOS.

Env: BASE_LONG, BASE_SHORT, CAND_LONG, CAND_SHORT (pred parquet paths; a two-book model passes its two
files, an rr_both-style model passes the same file twice), K_LONG=1, K_SHORT=2,
REGIME_CSV=live/state/longtail/v4_ab_fee45/ret/cycles.csv, LABEL=<name>.
"""
import os
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6
KL=int(os.environ.get("K_LONG","1")); KS=int(os.environ.get("K_SHORT","2"))
LABEL=os.environ.get("LABEL","candidate")
REGIME_CSV=os.environ.get("REGIME_CSV",f"{R}/live/state/longtail/v4_ab_fee45/ret/cycles.csv")

# ---- panel target + dispersion + eligibility ----
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
first_seen=pan.groupby("symbol")["open_time"].min()
uni=pd.read_csv(f"{os.path.dirname(REGIME_CSV)}/universe.csv")
hygiene_ok=set(uni[uni.in_universe]["symbol"])

def loadp(path,name):
    d=pd.read_parquet(path,columns=["symbol","open_time","pred"]).rename(columns={"pred":name})
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True); return d
d=(loadp(os.environ["BASE_LONG"],"bl").merge(loadp(os.environ["BASE_SHORT"],"bs"),on=["symbol","open_time"])
   .merge(loadp(os.environ["CAND_LONG"],"cl"),on=["symbol","open_time"])
   .merge(loadp(os.environ["CAND_SHORT"],"cs"),on=["symbol","open_time"])
   .merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"]))
d=d[d.open_time>=pd.Timestamp("2025-10-04",tz="UTC")].dropna(subset=["fwd","bl","bs","cl","cs"])
d=d[d.symbol.isin(hygiene_ok)]
d=d[(d.open_time-d.symbol.map(first_seen)).dt.days>=180]              # per-cycle maturity, PIT
reg=pd.read_csv(REGIME_CSV,parse_dates=["open_time"])[["open_time","regime"]]
reg["open_time"]=pd.to_datetime(reg["open_time"],utc=True)
d=d.merge(reg,on="open_time",how="left"); d["regime"]=d["regime"].fillna("side")

# ---- per-cycle tips (real + placebo), vectorized-ish loop ----
rng=np.random.RandomState(0)
rows=[]
for ot,g in d.groupby("open_time"):
    if len(g)<max(KL+KS,20): continue
    disp=g["fwd"].std()
    fsh=g["fwd"].sample(frac=1,random_state=rng).to_numpy()           # same shuffle for both models
    gb=g.assign(fsh=fsh)
    tb=g.nlargest(KL,"bl")["fwd"].mean()-g.nsmallest(KS,"bs")["fwd"].mean()
    tc=g.nlargest(KL,"cl")["fwd"].mean()-g.nsmallest(KS,"cs")["fwd"].mean()
    pb=gb.nlargest(KL,"bl")["fsh"].mean()-gb.nsmallest(KS,"bs")["fsh"].mean()
    pc=gb.nlargest(KL,"cl")["fsh"].mean()-gb.nsmallest(KS,"cs")["fsh"].mean()
    same_l=set(g.nlargest(KL,"bl").symbol)==set(g.nlargest(KL,"cl").symbol)
    same_s=set(g.nsmallest(KS,"bs").symbol)==set(g.nsmallest(KS,"cs").symbol)
    rows.append((ot,g["regime"].iloc[0],tb,tc,pb,pc,disp,same_l and same_s))
T=pd.DataFrame(rows,columns=["ot","regime","tb","tc","pb","pc","disp","same"]).set_index("ot").sort_index()
T["diff"]=T.tc-T.tb; T["pdiff"]=T.pc-T.pb
T["diff_n"]=T["diff"]/T["disp"].replace(0,np.nan)

def stride_t(s):
    u=np.array(sorted(s.index)); ss=s[s.index.isin(set(u[::H]))].dropna()
    return (ss.mean()/ss.std()*np.sqrt(len(ss)) if len(ss)>10 and ss.std()>0 else np.nan), len(ss)
def block_ci(s,nboot=3000):
    dd=s.resample("1D").sum(); dd=dd[dd!=0]
    nb=max(2,len(dd)//5); blocks=[dd.iloc[i*5:(i+1)*5].sum() for i in range(nb)]
    bs=[np.mean([blocks[j] for j in rng.randint(0,nb,nb)]) for _ in range(nboot)]
    return np.percentile(bs,2.5),np.percentile(bs,97.5)
def top3_share(s):
    tot=s.sum()
    if abs(tot)<1e-12: return np.nan
    same=s[np.sign(s)==np.sign(tot)]
    return float(same.abs().nlargest(3).sum()/abs(tot))
def halves(s):
    h1,h2=np.array_split(s.dropna().values,2)
    return np.sign(h1.mean())==np.sign(h2.mean()), h1.mean(), h2.mean()

print(f"=== TIP-ACCURACY v2: {LABEL}  (K={KL}/{KS}, two-book selection, eligible universe, {len(T)} cycles) ===")
print(f"picks identical to base: {T['same'].mean()*100:.0f}% of cycles\n")
hdr=(f"{'regime':7s} {'n':>5s} | {'base':>7s} {'cand':>7s} {'diff':>6s} {'d_norm':>7s} | "
     f"{'t(nol)':>6s} {'CI95(5d-blk)':>16s} | {'top3%':>5s} {'halves':>13s} {'plc_t':>6s} | verdict")
print(hdr)
verdicts={}
for regn in ["side","bear","bull","ALL"]:
    g=T if regn=="ALL" else T[T.regime==regn]
    if len(g)<30: print(f"{regn:7s} {len(g):5d} | (too few cycles)"); continue
    t6,n6=stride_t(g["diff"]); lo,hi=block_ci(g["diff"])
    tp,_=stride_t(g["pdiff"])
    t3=top3_share(g["diff"]); hs,h1,h2=halves(g["diff"])
    # CI (daily 5d-block bootstrap, full sample) is the PRIMARY significance test — the stride-6 t
    # discards 5/6 of cycles and is unstable on the heavy-tailed tip series (kept as diagnostic only).
    if hi<0: v="REJECT(worse)"
    elif lo>0 and t3<0.40 and hs and abs(tp if np.isfinite(tp) else 0)<2: v="PASS"
    elif lo>0: v="POS-but-fragile"      # significant mean but concentrated/unstable/placebo-warm
    else: v="NEUTRAL/underpowered"
    verdicts[regn]=v
    print(f"{regn:7s} {len(g):5d} | {g.tb.mean():+7.1f} {g.tc.mean():+7.1f} {g['diff'].mean():+6.1f} {g['diff_n'].mean():+7.3f} | "
          f"{t6:+6.2f} [{lo:+6.1f},{hi:+6.1f}] | {t3*100 if np.isfinite(t3) else float('nan'):5.0f} {str(hs):>5s}({h1:+.0f}/{h2:+.0f}) {tp:+6.2f} | {v}")
act=[verdicts.get("side","n/a"),verdicts.get("bear","n/a")]
overall="PASS" if "PASS" in act and "REJECT(worse)" not in act else ("REJECT" if "REJECT(worse)" in act else "NEUTRAL")
print(f"\nOVERALL (pred-active regimes side+bear): {overall}   [screen only — adoption requires full-stack replay at FEE_BPS_FILL=4.5 + OOS]")
print("TIPV2DONE")
