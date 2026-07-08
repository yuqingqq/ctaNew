"""Tip-SNR diagnostic. (1) Is the model's tip signal or noise over time — rolling SNR of the traded tip, stability,
autocorrelation. (2) Can PIT features {r30, BTC-vol, xs-dispersion, trailing-tip-edge} predict the FORWARD tip
edge (sign) and its MAGNITUDE (|edge| = SNR/noise proxy), incrementally over BTC-r30 alone? If dispersion/vol add
R2 over r30 for |edge|, a systematic tip-distribution regime beats the heuristic. v4 preds, K_long=1/K_short=2. bps.
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
from sklearn.linear_model import LinearRegression
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; KL=1; KS=2
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
def lp(p,c):
    d=pd.read_parquet(f"{R}/live/state/convexity/{p}/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":c})
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True); return d
def build(b,l):
    return lp(b,"pb").merge(lp(l,"pl"),on=["symbol","open_time"]).merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"]).dropna(subset=["fwd"])
WIN={"OOS":("hl_v4base_oos","hl_v4long_oos"),"RECENT":("hl_tgt_res_base","hl_tgt_res_long")}
data={k:build(*v) for k,v in WIN.items()}
allg=pd.DatetimeIndex(sorted(set().union(*[set(d["open_time"].unique()) for d in data.values()])))
def fm(per):
    try:
        r=requests.get(f"https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/4h/BTCUSDT-4h-{per.strftime('%Y-%m')}.zip",timeout=20)
        if r.status_code!=200: return None
        z=zipfile.ZipFile(io.BytesIO(r.content)); raw=z.read(z.namelist()[0]).decode(); hdr=0 if raw.split(",",1)[0]=="open_time" else None
        x=pd.read_csv(io.StringIO(raw),header=hdr); x.columns=["open_time","o","h","l","close","v","ct","qv","n","tb","tbq","ig"][:x.shape[1]]
        vv=pd.to_numeric(x["open_time"],errors="coerce"); u="us" if vv.dropna().median()>1e15 else "ms"
        x["open_time"]=pd.to_datetime(vv,unit=u,utc=True); x["close"]=pd.to_numeric(x["close"],errors="coerce"); return x[["open_time","close"]]
    except Exception: return None
with ThreadPoolExecutor(max_workers=12) as ex:
    parts=[q for q in ex.map(fm,pd.period_range("2022-06",allg.max().to_period("M"),freq="M")) if q is not None]
btc=pd.concat(parts).dropna().drop_duplicates("open_time").set_index("open_time").sort_index()["close"]
btc=btc.reindex(pd.DatetimeIndex(sorted(set(btc.index)|set(allg)))).ffill()
r30=(btc/btc.shift(180)-1); btcret=np.log(btc/btc.shift(1)); btcvol=btcret.rolling(30).std()  # trailing 5d vol
def cyc_series(d):
    rows=[]
    for ot,g in d.groupby("open_time"):
        if len(g)<KL+KS: continue
        tip=g.nlargest(KL,"pl")["fwd"].mean()-g.nsmallest(KS,"pb")["fwd"].mean()
        disp=g["pb"].std()  # cross-sectional pred dispersion (PIT — pred known at t)
        rows.append((ot,tip,disp))
    s=pd.DataFrame(rows,columns=["ot","tip","disp"]).set_index("ot").sort_index()
    s["r30"]=s.index.map(r30.to_dict()); s["btcvol"]=s.index.map(btcvol.to_dict())
    s["trail_tip"]=s["tip"].shift(H).rolling(90).mean()  # exit-lagged trailing tip edge (PIT, like REGIME_GATE)
    return s.dropna()
for win,d in data.items():
    s=cyc_series(d)
    print(f"\n{'='*72}\n=== {win} ({len(s)} cyc) ===")
    # (1) rolling tip SNR over time
    roll=s["tip"].rolling(90); rsnr=(roll.mean()/roll.std()).dropna()
    print(f"(1) tip edge: mean {s['tip'].mean():+.1f}, per-cycle SNR {s['tip'].mean()/s['tip'].std()*np.sqrt(len(s)):+.2f}")
    print(f"    rolling-90 SNR: mean {rsnr.mean():+.2f}, %windows<0 {100*(rsnr<0).mean():.0f}%, min {rsnr.min():+.2f} max {rsnr.max():+.2f}")
    print(f"    tip autocorr(lag {H}, exit-lagged): {s['tip'].autocorr(H):+.3f}  (>0 = tip health persists => predictable)")
    # (2) predictability regressions
    X0=s[["r30"]].values; Xd=s[["r30","disp","btcvol"]].values; Xa=s[["r30","disp","btcvol","trail_tip"]].values
    def r2(X,y):
        m=LinearRegression().fit(X,y); return m.score(X,y)
    for tgt,lab in [(s["tip"].values,"SIGNED tip (which way)"),(s["tip"].abs().values,"|tip| MAGNITUDE (SNR/noise)")]:
        print(f"(2) predict {lab}: R2  r30={r2(X0,tgt):.4f}  +disp+vol={r2(Xd,tgt):.4f}  +trail_tip={r2(Xa,tgt):.4f}")
    # feature correlations with |tip|
    for f in ["r30","disp","btcvol","trail_tip"]:
        print(f"    corr({f}, |tip|)={s[f].corr(s['tip'].abs()):+.3f} | corr({f}, tip)={s[f].corr(s['tip']):+.3f}")
print("\n(R2 for |tip|: if +disp+vol >> r30, dispersion/vol predict tip MAGNITUDE => systematic SNR regime beats heuristic.)")
print("TIPSNRDONE")
