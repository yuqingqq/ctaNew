"""STAGE 1 of the full-universe test: fetch the MISSING USDT-perp 5m klines for 2025-01 -> 2026-06 from Binance Vision
(public daily/monthly archives; reachable from this box; INCLUDES delisted -> survivorship-free). Target = USDT perps
in Vision NOT already in our panel. Resumable (skips symbols already saved). Checkpoint-logs every 25 symbols + errors.
  out: data/ml/cache/fulluniv/<SYM>.parquet  (open_time, close, volume)
"""
import os, io, re, zipfile, time, sys
import concurrent.futures as cf
import requests, numpy as np, pandas as pd
ROOT="/home/yuqing/ctaNew"; OUT=f"{ROOT}/data/ml/cache/fulluniv"; os.makedirs(OUT,exist_ok=True)
BASE="https://data.binance.vision/data/futures/um"
MONTHS=[f"{y}-{m:02d}" for y in (2025,) for m in range(1,13)]+[f"2026-{m:02d}" for m in range(1,6)]
DAYS=[f"2026-06-{d:02d}" for d in range(1,6)]   # partial last month
def target_symbols():
    x=requests.get("https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/futures/um/daily/klines/",timeout=30).text
    usdt=[s for s in re.findall(r'<Prefix>data/futures/um/daily/klines/([^/]+)/</Prefix>',x) if s.endswith("USDT")]
    have=set(pd.read_parquet(f"{ROOT}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol"]).symbol.unique())
    return sorted(set(usdt)-have)
def _csv(zbytes):
    z=zipfile.ZipFile(io.BytesIO(zbytes)); n=z.namelist()[0]
    df=pd.read_csv(z.open(n),header=None,usecols=[0,4,5],names=["open_time","close","volume"])
    # open_time epoch ms (sometimes us in recent files) -> normalize
    ot=pd.to_numeric(df["open_time"],errors="coerce")
    unit="us" if ot.max()>1e15 else "ms"
    df["open_time"]=pd.to_datetime(ot,unit=unit,utc=True)
    return df.dropna()
def fetch_one(sym):
    fp=f"{OUT}/{sym}.parquet"
    if os.path.exists(fp): return sym,"skip-exists",0
    parts=[]
    for mo in MONTHS:
        u=f"{BASE}/monthly/klines/{sym}/5m/{sym}-5m-{mo}.zip"
        try:
            r=requests.get(u,timeout=30)
            if r.status_code==200: parts.append(_csv(r.content))
        except Exception: pass
    for d in DAYS:
        u=f"{BASE}/daily/klines/{sym}/5m/{sym}-5m-{d}.zip"
        try:
            r=requests.get(u,timeout=30)
            if r.status_code==200: parts.append(_csv(r.content))
        except Exception: pass
    if not parts: return sym,"no-data",0
    df=pd.concat(parts,ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df=df[(df.open_time>=pd.Timestamp("2025-01-01",tz="UTC"))&(df.open_time<pd.Timestamp("2026-06-05",tz="UTC"))]
    if len(df)<300: return sym,"too-short",len(df)
    df.to_parquet(fp); return sym,"ok",len(df)
def main():
    syms=target_symbols(); n=len(syms)
    print(f"START fetch: {n} target symbols (USDT perps not in panel), window 2025-01..2026-06",flush=True)
    ok=saved=err=skip=0; t0=time.time()
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        for i,(sym,status,nb) in enumerate(ex.map(fetch_one,syms),1):
            if status=="ok": ok+=1; saved+=1
            elif status=="skip-exists": skip+=1; saved+=1
            elif status in ("no-data","too-short"): err+=1
            if i%25==0 or i==n:
                print(f"PROGRESS {i}/{n}: saved {saved}, no-data/short {err}, elapsed {time.time()-t0:.0f}s",flush=True)
    print(f"DONE fetch: {saved} symbols saved to {OUT} ({ok} new), {err} skipped(no-data/short), {time.time()-t0:.0f}s",flush=True)
if __name__=="__main__": main()
