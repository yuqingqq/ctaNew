"""Set the v4 baseline: single per-symbol RidgeCV on V0_LEAN + resid_rev (both legs), RESIDUAL target
(xs_z(alpha_vs_btc_realized)). Simple gate-free book, gross, no tuning. Writes live/V4_PERFORMANCE.md.
Model preds = hl_tgt_res_long (V0_LEAN+RR, residual target) used for BOTH legs. base-only book no longer used.
"""
import io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; ANN=np.sqrt(365)
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
p=pd.read_parquet(f"{R}/live/state/convexity/hl_tgt_res_long/v0full_hl60.parquet",columns=["symbol","open_time","pred"])
p["open_time"]=pd.to_datetime(p["open_time"],utc=True)
d=p.merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"])
d=d[d.open_time>=pd.Timestamp("2025-10-04",tz="UTC")].dropna(subset=["fwd"])
grid=pd.DatetimeIndex(sorted(d["open_time"].unique()))
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
    parts=[q for q in ex.map(fm,pd.period_range("2024-06",grid.max().to_period("M"),freq="M")) if q is not None]
btc=pd.concat(parts).dropna().drop_duplicates("open_time").set_index("open_time").sort_index()["close"]
btc=btc.reindex(pd.DatetimeIndex(sorted(set(btc.index)|set(grid)))).ffill(); r30=(btc/btc.shift(180)-1)
reg={t:("bull" if v>0.10 else "bear" if v<-0.10 else "side") for t,v in r30.items()}
def metr(KL,KS):
    rows=[]
    for ot,g in d.groupby("open_time"):
        if len(g)<KL+KS: continue
        rows.append((ot,reg.get(ot,"side"),g.nlargest(KL,"pred")["fwd"].mean()-g.nsmallest(KS,"pred")["fwd"].mean()))
    x=pd.DataFrame(rows,columns=["ot","reg","ls"]).set_index("ot")
    dd=(x["ls"]/1e4).resample("1D").sum(); sh=dd.mean()/dd.std()*ANN if dd.std()>0 else np.nan
    eq=dd.cumsum(); mdd=float((eq-eq.cummax()).min()*1e4); br={rg:x[x.reg==rg]["ls"].mean() for rg in ["bull","side","bear"]}
    return dict(sh=sh,lsm=x["ls"].mean(),tot=x["ls"].sum(),mdd=mdd,pos=100*(x["ls"]>0).mean(),br=br)
L=["# Convexity v4 — performance tracker","",
"## v4 DEFAULT (locked 2026-07-06)",
"- **Model:** single per-symbol RidgeCV, **features = `V0_LEAN + resid_rev_2 + resid_rev_3`** (16), used for BOTH legs.",
"- **Target:** **residual** — `xs_z(alpha_vs_btc_realized)` (train on what we farm).",
"- Preds: `hl_tgt_res_long/v0full_hl60.parquet` (V0_LEAN+RR, residual target). base-only book retired.",
"- Baseline book below: fixed K, equal weight, **gross, NO gates/regime/hedge/sizing/conc-cap** — the zero-tuning reference.",
"",
"## Baseline (in-sample 2025-10-04+, gross, no gates)",
"| K L/S | dailySharpe | L/S mean/cyc | totPnL | maxDD | %pos | bull | side | bear |",
"|---|---|---|---|---|---|---|---|---|"]
for KL,KS in [(1,2),(2,2),(3,3)]:
    m=metr(KL,KS); tag=" **(canonical)**" if (KL,KS)==(1,2) else ""
    L.append(f"| {KL}/{KS}{tag} | {m['sh']:+.2f} | {m['lsm']:+.1f} | {m['tot']:+.0f} | {m['mdd']:+.0f} | {m['pos']:.0f}% | {m['br']['bull']:+.0f} | {m['br']['side']:+.0f} | {m['br']['bear']:+.0f} |")
L+=["",
"## How this baseline was chosen (fair comparisons, gate-free simple book)",
"- **resid_rev on BOTH legs (adopted):** rr_both beats base_both by +0.5–0.68 Sharpe & better maxDD, both time-halves",
"  positive, across K. Beats the old base-short/residrev-long split (split leaves ~0.38 Sharpe on the short leg).",
"  The two-book split was suboptimal; single V0_LEAN+RR model for both legs is better AND simpler.",
"- **Residual target (adopted on consistency):** train on the residual we farm. Honest evidence: helps at K=1/2",
"  (+0.53 Sharpe) but tied at K=2/2, slightly negative K=3/3, and WORSENS maxDD at every K. Chosen for consistency;",
"  its performance case is K=1/2-specific and concentrated — revisit at OOS.",
"",
"## Caveats (must clear before any live/production use)",
"- Gross, in-sample (2025-10+). NOT comparable to v3's net+gated headline.",
"- Residual-target edge is concentrated (median 0) — OOS (2022–2026) is the decisive test.",
"- Short-leg hedge role not captured by forward alpha (but rr_both maxDD argues it's not degraded).",
"",
"## Change log",
"| date | change | dailySharpe (K=1/2) | notes |",
"|---|---|---|---|",
"| 2026-07-06 | v4 baseline SET: residual target + V0_LEAN+resid_rev both legs | (see table) | zero-tuning reference; fair-comparison chosen |"]
open(f"{R}/live/V4_PERFORMANCE.md","w").write("\n".join(L)+"\n")
print("\n".join(L[:22])); print("\nwrote live/V4_PERFORMANCE.md"); print("V4SETDONE")
