"""One-by-one review of positioning features (user request 2026-06-03):
(1) VALIDATE generation: coverage, distribution, NaN, PIT-lag sanity (no contemporaneous bleed).
(2) per-feature univariate IC (overall + high-vol + within-decliner for the flush).
(3) REDUNDANCY: max |corr| with V0 features (is it new info or a price-proxy?).
(4) MARGINAL: pooled walk-forward OOS IC of V0 vs V0+each-feature (does it ADD?).
-> tells us WHY #14 (all-together) hurt: which feats are noise / redundant / orthogonal-useful.
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
V0 = list(tt.V0)
POS = ["oi_chg4h_z","oi_flush_dec","global_ls","global_ls_z","smart_minus_retail","taker_ls","taker_ls_z"]
OOS = pd.Timestamp("2025-10-04", tz="UTC")
CUTS = [pd.Timestamp(t,tz="UTC") for t in ["2025-10-04","2025-12-01","2026-02-01","2026-04-01","2026-05-27"]]

# rebuild positioning features (same logic as gen) + a CONTEMPORANEOUS (unlagged) oi_chg for bleed check
rows=[]
for f in glob.glob(str(REPO/"data/ml/cache/metrics_*.parquet")):
    d=pd.read_parquet(f)
    if len(d)<800 or "sum_open_interest" not in d.columns: continue
    d=d.reset_index(); d.columns=["create_time"]+list(d.columns[1:]); d["create_time"]=pd.to_datetime(d["create_time"],utc=True)
    d=d.set_index("create_time").sort_index(); d=d[~d.index.duplicated(keep="last")]
    r4=lambda c: d[c].resample("4h").last()
    oi,tl,gl,tk=r4("sum_open_interest"),r4("sum_toptrader_long_short_ratio"),r4("count_long_short_ratio"),r4("sum_taker_long_short_vol_ratio")
    chg=oi.pct_change(1).replace([np.inf,-np.inf],np.nan).clip(-0.5,0.5)   # FIX: OI≈0 div-explosion
    tl=tl.clip(0,20); gl=gl.clip(0,20); tk=tk.clip(0,10)                    # FIX: ratio outliers (taker max was 20181)
    zz=lambda s,w=180:((s-s.rolling(w,min_periods=60).mean())/s.rolling(w,min_periods=60).std()).replace([np.inf,-np.inf],np.nan).clip(-10,10)
    rows.append(pd.DataFrame({"open_time":oi.index,"symbol":d["symbol"].iloc[0],
        "oi_chg4h_z":zz(chg).shift(1).values, "oi_chg4h_z_NOLAG":zz(chg).values,    # NOLAG = bleed check
        "global_ls":gl.shift(1).values,"global_ls_z":zz(gl).shift(1).values,
        "smart_minus_retail":(zz(tl)-zz(gl)).shift(1).values,
        "taker_ls":tk.shift(1).values,"taker_ls_z":zz(tk).shift(1).values}))
M=pd.concat(rows,ignore_index=True).drop_duplicates(["symbol","open_time"]); M["open_time"]=pd.to_datetime(M["open_time"],utc=True)

PAN=pd.read_parquet(tt.PANEL,columns=["symbol","open_time","exit_time","return_pct"]+V0)  # ret_3d already in V0
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)]
g=PAN.groupby("open_time"); sd=g["return_pct"].transform("std").replace(0,np.nan)
PAN["fwd"]=PAN["return_pct"]-g["return_pct"].transform("mean")
PAN["xs_z"]=((PAN["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)
D=PAN.merge(M,on=["symbol","open_time"],how="left").reset_index(drop=True)
D["oi_flush_dec"]=D["oi_chg4h_z"]*(D["ret_3d"]<0).astype(float)
Doos=D[D.open_time>=OOS]; hi=Doos[Doos.rvol_7d>Doos.rvol_7d.median()]

print("="*70); print("(1) VALIDATION: coverage / distribution / NaN")
for c in POS:
    s=Doos[c].dropna(); cov=Doos[c].notna().mean()
    print(f"  {c:20s}: cov={cov*100:4.0f}% mean={s.mean():+.3f} std={s.std():.3f} min={s.min():+.2f} max={s.max():+.2f}")
print("\n(1b) PIT BLEED CHECK (oi_chg4h_z lagged vs NOLAG):")
def ic(df,c): s=df[[c,"fwd"]].dropna(); return s[c].corr(s["fwd"],method="spearman") if len(s)>300 else np.nan
print(f"  oi_chg4h_z (lagged, used)  IC={ic(Doos,'oi_chg4h_z'):+.4f}  | oi_chg4h_z_NOLAG IC={ic(Doos,'oi_chg4h_z_NOLAG'):+.4f} (big gap = bleed correctly removed)")

print("\n"+"="*70); print("(2)+(3) per-feature: univariate IC + redundancy with V0")
for c in POS:
    icall, ichi = ic(Doos,c), ic(hi,c)
    red = max(abs(Doos[[c,v]].dropna().corr().iloc[0,1]) for v in V0 if Doos[[c,v]].dropna().shape[0]>300)
    nearest = max(V0, key=lambda v: abs(Doos[[c,v]].dropna().corr().iloc[0,1]) if Doos[[c,v]].dropna().shape[0]>300 else 0)
    print(f"  {c:20s}: IC_all={icall:+.4f} IC_hivol={ichi:+.4f} | max|corr|V0={red:.2f} ({nearest})")
# oi_flush within decliner+flush (the P=0.93 conditional)
dec=Doos[(Doos.ret_3d<=Doos.ret_3d.quantile(0.25))]
fl=dec[dec.oi_chg4h_z<=-2]; nf=dec[dec.oi_chg4h_z>-2]
print(f"  [flush conditional] decliner+flush fwd {fl.fwd.mean():+.5f} (n={len(fl)}) vs no-flush {nf.fwd.mean():+.5f} -> edge {fl.fwd.mean()-nf.fwd.mean():+.5f}")

print("\n"+"="*70); print("(4) MARGINAL pooled OOS IC: V0 vs V0+each feature (walk-forward)")
EMB=pd.Timedelta(days=1); P=D[D["xs_z"].notna()].copy()
def ladder(feats):
    ics=[]
    for i in range(len(CUTS)-1):
        c0,c1=CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr=P[P.exit_time<fc]; te=P[(P.open_time>=c0)&(P.open_time<c1)].copy()
        if len(tr)<5000 or len(te)==0: continue
        Xtr=tr[feats].fillna(0).to_numpy(); mu,sg=Xtr.mean(0),Xtr.std(0); sg[sg==0]=1
        m=RidgeCV(alphas=(0.1,1,10,100)).fit((Xtr-mu)/sg,tr["xs_z"])
        te["p"]=m.predict((te[feats].fillna(0).to_numpy()-mu)/sg)
        ics.append(te.groupby("open_time").apply(lambda x:x["p"].corr(x["xs_z"],method="spearman")).mean())
    return float(np.mean(ics))
base=ladder(V0); print(f"  V0 baseline pooled IC = {base:+.4f}")
for c in POS:
    print(f"  V0+{c:20s} = {ladder(V0+[c]):+.4f}  (Δ{ladder(V0+[c])-base:+.4f})")
print(f"  V0+ALL_POS = {ladder(V0+POS):+.4f}  (Δ{ladder(V0+POS)-base:+.4f})")
