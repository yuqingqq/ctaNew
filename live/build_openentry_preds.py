"""Diagnostic: rebuild baseline preds' realized returns using OPEN@hh:00 entry/exit (NO 5m delay) instead of
CLOSE@hh:05. Same signal (pred unchanged) — only the realized return swapped. Because the signal (resid_rev etc.)
embeds the hh:05 close, open@hh:00 entry is LOOK-AHEAD; this quantifies that gap.
Outputs hl_openentry/ + hl_residrev_openentry/ (return_pct, alpha_A swapped to open-based).
"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.convexity_paper_bot as bot
KL=bot.KLINES; H=48  # 48 5m-bars = 4h

def load_oc_4h(sym):
    sd=KL/sym/"5m"
    if not sd.exists(): return None
    df=pd.concat([pd.read_parquet(f,columns=["open_time","open","close"]) for f in sorted(sd.glob("*.parquet"))],ignore_index=True)
    df=df.drop_duplicates("open_time").sort_values("open_time"); df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    df=df.set_index("open_time")
    m=(df.index.hour%4==0)&(df.index.minute==0)
    return df[m][["open","close"]].astype(float)

def fwd(series): return series.shift(-1)/series - 1   # 4h-grid forward (next 4h bar)

btc=load_oc_4h("BTCUSDT")
btc_ret_open=fwd(btc["open"]).rename("btc_o"); btc_ret_close=fwd(btc["close"]).rename("btc_c")
# beta exactly as target_alpha: rolling(288,min72) cov/var of 5m LOG close returns, shift(1)  -> reuse panel value instead
PAN=pd.read_parquet(REPO/"outputs/vBTC_features/panel_expanded_v0.parquet",
                    columns=["symbol","open_time","alpha_vs_btc_realized","return_pct"])
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True)

def build(sym):
    oc=load_oc_4h(sym)
    if oc is None or len(oc)<50: return None
    ro=fwd(oc["open"]); rc=fwd(oc["close"])
    j=pd.concat([ro.rename("ro"),rc.rename("rc"),btc_ret_open,btc_ret_close],axis=1)
    p=PAN[PAN.symbol==sym].set_index("open_time")
    j=j.join(p[["alpha_vs_btc_realized","return_pct"]],how="inner").dropna(subset=["ro","rc"])
    if not len(j): return None
    # back out beta from the close-based panel alpha: alpha_c = rc - beta*btc_c  => beta = (rc - alpha_c)/btc_c
    beta=(j["rc"]-j["alpha_vs_btc_realized"])/j["btc_c"].replace(0,np.nan)
    j["alpha_open"]=j["ro"]-beta*j["btc_o"]
    return j[["ro","alpha_open"]].rename(columns={"ro":"ret_open"})

def swap(src_sub, dst_sub):
    d=pd.read_parquet(REPO/"live/state/convexity"/src_sub/"v0full_hl60.parquet"); d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    parts=[]
    for sym,g in d.groupby("symbol"):
        b=build(sym)
        if b is None: continue
        gg=g.merge(b.reset_index(),on="open_time",how="left").dropna(subset=["ret_open"])
        gg["return_pct"]=gg["ret_open"]; gg["alpha_A"]=gg["alpha_open"]
        parts.append(gg.drop(columns=["ret_open","alpha_open"]))
    out=pd.concat(parts,ignore_index=True)
    od=REPO/"live/state/convexity"/dst_sub; od.mkdir(parents=True,exist_ok=True)
    out.to_parquet(od/"v0full_hl60.parquet")
    import shutil; shutil.copy(REPO/"live/state/convexity/hl/fullflow_hl60.parquet", od/"fullflow_hl60.parquet")
    return d["symbol"].nunique(), out["symbol"].nunique(), len(out)

print("base:", swap("hl","hl_openentry"))
print("long:", swap("hl_residrev","hl_residrev_openentry"))
print("DONE")
