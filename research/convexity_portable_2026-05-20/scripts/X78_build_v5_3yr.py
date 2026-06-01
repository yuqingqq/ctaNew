"""X78 — Build full V5 panel over 3 years + test V5_mv3 (uncond + regime-routed).

Prereqs (run first):
  - X76 OKX+CB extended to 2023 (crossX inputs)
  - X77 aggTrades extended to 2023 + build_aggtrade_features (flow_<sym>.parquet over 3yr)
  - klines already extended (v3 from klines); panel_3yr_v0 has BASE+cohort+funding+target

Steps:
  A. Build 7-feature crossX over 3yr for candidate syms (from BN-perp/spot, OKX, CB 1h)
  B. Build aggT 4h features over 3yr (from flow_<sym>.parquet)
  C. Compute v3 idio over 3yr (from klines)
  D. Merge all into panel_3yr_v0 → panel_3yr_v5
  E. Test V5_mv3 (BASE+cohort+aggT+7cx, 29 feats):
       - unconditional (single, expanding)
       - regime-routed (KMeans K=5, per-cluster specialists)
     Compare to V0 3yr (+0.12 uncond, +1.07 routed K=5)
"""
from __future__ import annotations
import sys, time, importlib.util, warnings, gc
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scripts"))
CACHE = REPO/"data/ml/cache"; KLINES = REPO/"data/ml/test/parquet/klines"
OUT = REPO/"research/convexity_portable_2026-05-20/results"; RCACHE = OUT/"_cache"
HORIZON = 48
spec = importlib.util.spec_from_file_location("x6", REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)

CANDIDATES = [
    "ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT",
    "DOTUSDT","ATOMUSDT","LTCUSDT","BCHUSDT","NEARUSDT","UNIUSDT","TIAUSDT","SUIUSDT",
    "SEIUSDT","INJUSDT","ARBUSDT","APTUSDT","OPUSDT","AAVEUSDT","AXSUSDT","FILUSDT",
    "ETCUSDT","TRBUSDT","WLDUSDT","ICPUSDT","ONDOUSDT","PENDLEUSDT","LDOUSDT","JTOUSDT",
    "ENAUSDT","HBARUSDT","TONUSDT","STRKUSDT","WIFUSDT","ORDIUSDT","JUPUSDT","GMXUSDT",
    "TAOUSDT","RUNEUSDT","SUSDT","ZECUSDT",
]
CX7 = ["bn_perp_okx_perp_z","bn_perp_okx_spot_z","okx_perp_spot_z","bn_perp_cb_spot_z",
       "okx_cb_spot_z","bn_spot_okx_spot_z","bn_spot_cb_spot_z"]
AGGT = ["signed_volume_4h","tfi_4h","aggr_ratio_4h","buy_count_4h","avg_trade_size_4h"]
V3 = ["idio_max_abs_12b","idio_skew_1d","idio_kurt_1d","name_idio_share_1d"]


def load_1h(prefix, sym):
    fp = CACHE/f"{prefix}_{sym}_1h.parquet"
    if not fp.exists(): return None
    df = pd.read_parquet(fp)
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True).astype("datetime64[ns, UTC]")
        df = df.set_index("open_time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df["close"].astype(np.float32) if "close" in df.columns else None


def load_perp_hourly(sym):
    sd = KLINES/sym/"5m"
    if not sd.exists(): return None
    dfs = [pd.read_parquet(f, columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True).astype("datetime64[ns, UTC]")
    df = df.set_index("open_time").sort_index()
    return df[df.index.minute==0]["close"].astype(np.float32)


def basis_bps(a, b):
    mid = (a+b)/2
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((a-b)/mid*10000.0).astype(np.float32)


def build_crossX(sym):
    bn_perp = load_perp_hourly(sym)
    if bn_perp is None: return None
    idx = bn_perp.index
    al = {"bn_perp": bn_perp}
    for nm, s in [("bn_spot",load_1h("bn_spot",sym)),("okx_perp",load_1h("okx_swap",sym)),
                   ("okx_spot",load_1h("okx_spot",sym)),("cb_spot",load_1h("cb_spot",sym))]:
        if s is not None: al[nm] = s.reindex(idx).ffill()
    d = pd.DataFrame(al); out = pd.DataFrame(index=idx)
    if "okx_perp" in d: out["bn_perp_okx_perp"]=basis_bps(d["bn_perp"],d["okx_perp"])
    if "okx_spot" in d: out["bn_perp_okx_spot"]=basis_bps(d["bn_perp"],d["okx_spot"])
    if "okx_perp" in d and "okx_spot" in d: out["okx_perp_spot"]=basis_bps(d["okx_perp"],d["okx_spot"])
    if "cb_spot" in d: out["bn_perp_cb_spot"]=basis_bps(d["bn_perp"],d["cb_spot"])
    if "okx_spot" in d and "cb_spot" in d: out["okx_cb_spot"]=basis_bps(d["okx_spot"],d["cb_spot"])
    if "bn_spot" in d and "okx_spot" in d: out["bn_spot_okx_spot"]=basis_bps(d["bn_spot"],d["okx_spot"])
    if "bn_spot" in d and "cb_spot" in d: out["bn_spot_cb_spot"]=basis_bps(d["bn_spot"],d["cb_spot"])
    o4 = out[out.index.hour % 4 == 0]
    for c in list(o4.columns):
        roll = o4[c].rolling(180, min_periods=24).agg(["mean","std"])
        o4[c+"_z"] = ((o4[c]-roll["mean"])/roll["std"].replace(0,np.nan)).shift(1).astype(np.float32)
    o4["symbol"]=sym
    return o4.reset_index().rename(columns={"index":"open_time"})


def agg_4h_flow(flow, w=48):
    sv=flow["signed_volume"].rolling(w,min_periods=max(2,w//4)).sum()
    tv=(flow["buy_volume"]+flow["sell_volume"]).rolling(w,min_periods=max(2,w//4)).sum()
    bc=flow["buy_count"].rolling(w,min_periods=max(2,w//4)).sum()
    sc=flow["sell_count"].rolling(w,min_periods=max(2,w//4)).sum()
    out=pd.DataFrame(index=flow.index)
    out["signed_volume_4h"]=sv; out["tfi_4h"]=sv/tv.replace(0,np.nan)
    out["aggr_ratio_4h"]=(bc-sc)/(bc+sc).replace(0,np.nan)
    out["buy_count_4h"]=bc; out["avg_trade_size_4h"]=tv/(bc+sc).replace(0,np.nan)
    return out


def compute_v3(sym, btc_close):
    sd = KLINES/sym/"5m"
    if not sd.exists(): return None
    dfs=[pd.read_parquet(f,columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    df=pd.concat(dfs,ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True).astype("datetime64[ns, UTC]")
    df=df.set_index("open_time").sort_index(); mc=df["close"].astype(np.float32)
    ba=btc_close.reindex(mc.index).ffill()
    mr=np.log(mc/mc.shift(1)); br=np.log(ba/ba.shift(1))
    cov=mr.rolling(288,min_periods=72).cov(br); var=br.rolling(288,min_periods=72).var()
    beta=(cov/var.replace(0,np.nan)).shift(1); idio=mr-beta*br
    return pd.DataFrame({"symbol":sym,"open_time":mc.index,
        "idio_max_abs_12b":idio.rolling(12,min_periods=6).apply(lambda x:np.max(np.abs(x))).shift(1).astype(np.float32).values,
        "idio_skew_1d":idio.rolling(288,min_periods=72).skew().shift(1).astype(np.float32).values,
        "idio_kurt_1d":idio.rolling(288,min_periods=72).kurt().shift(1).astype(np.float32).values,
        "name_idio_share_1d":(idio.rolling(288,min_periods=72).var()/mr.rolling(288,min_periods=72).var().replace(0,np.nan)).shift(1).astype(np.float32).values})


def btc_regime_features():
    files=sorted((KLINES/"BTCUSDT"/"5m").glob("*.parquet"))
    btc=pd.concat([pd.read_parquet(f,columns=["open_time","close"]) for f in files],ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    btc["open_time"]=pd.to_datetime(btc["open_time"],utc=True); btc=btc.set_index("open_time")["close"].astype(np.float64)
    lr=np.log(btc/btc.shift(1))
    return pd.DataFrame({"btc_ret_7d":btc/btc.shift(2016)-1,"btc_ret_30d":btc/btc.shift(8640)-1,
        "btc_ret_90d":btc/btc.shift(25920)-1,"btc_rvol_30d":lr.rolling(8640,min_periods=2880).std()*np.sqrt(8640),
        "btc_dist_ma":btc/btc.rolling(57600,min_periods=2880).mean()-1}).reset_index()


def routed(panel, folds, feats, K, label):
    rf=["btc_ret_7d","btc_ret_30d","btc_ret_90d","btc_rvol_30d","btc_dist_ma"]
    cyc=panel.groupby("open_time")[rf].first().dropna()
    km=KMeans(n_clusters=K,random_state=42,n_init=10).fit(StandardScaler().fit_transform(cyc.values))
    cyc["cluster"]=km.labels_; cmap=cyc["cluster"].reset_index()
    p=panel.merge(cmap,on="open_time",how="left"); parts=[]
    for c in range(K):
        pc=p[p["cluster"]==c]
        if pc["open_time"].nunique()<50: continue
        try: parts.append(x6.train_per_sym_ridge(pc,folds,feats,label=f"{label}_c{c}"))
        except Exception as e: print(f"    c{c} err {e}")
    return pd.concat(parts,ignore_index=True).sort_values(["open_time","symbol"]) if parts else None


def main():
    t0=time.time()
    print("=== X78 build V5 3yr + test ===\n", flush=True)
    panel=pd.read_parquet(REPO/"outputs/vBTC_features/panel_3yr_v0.parquet")
    panel["open_time"]=pd.to_datetime(panel["open_time"],utc=True)
    panel["exit_time"]=pd.to_datetime(panel["exit_time"],utc=True)
    print(f"V0 panel: {len(panel):,} rows × {panel['symbol'].nunique()} syms")

    btc_close=load_perp_hourly("BTCUSDT")
    btc5=btc_close.reindex(pd.date_range(btc_close.index.min(),btc_close.index.max(),freq="5min",tz="UTC")).ffill()
    btc5.index=pd.DatetimeIndex(btc5.index).astype("datetime64[ns, UTC]")

    # A. crossX
    print("--- A. crossX 3yr ---", flush=True)
    cxs=[]
    for i,sym in enumerate(CANDIDATES,1):
        cx=build_crossX(sym)
        if cx is not None: cxs.append(cx)
        if i%10==0: print(f"  crossX {i}/{len(CANDIDATES)}", flush=True)
    cx_panel=pd.concat(cxs,ignore_index=True); cx_panel["open_time"]=pd.to_datetime(cx_panel["open_time"],utc=True)
    zc=[c for c in cx_panel.columns if c.endswith("_z")]
    panel=panel.merge(cx_panel[["symbol","open_time"]+zc],on=["symbol","open_time"],how="left")
    del cxs,cx_panel; gc.collect()
    print(f"  crossX merged ({len(zc)} feats)", flush=True)

    # B. aggT
    print("--- B. aggT 3yr ---", flush=True)
    arows=[]
    for i,sym in enumerate(CANDIDATES,1):
        fp=CACHE/f"flow_{sym}.parquet"
        if not fp.exists(): continue
        flow=pd.read_parquet(fp)
        if "signed_volume" not in flow.columns: continue
        if not isinstance(flow.index,pd.DatetimeIndex):
            if "open_time" in flow.columns: flow=flow.set_index("open_time")
        if flow.index.tz is None: flow.index=flow.index.tz_localize("UTC")
        a=agg_4h_flow(flow.sort_index()); a["symbol"]=sym
        a=a.reset_index().rename(columns={"index":"open_time",flow.index.name or "index":"open_time"})
        if "open_time" not in a.columns: a["open_time"]=a.iloc[:,0]
        arows.append(a[["symbol","open_time"]+AGGT])
        if i%10==0: print(f"  aggT {i}/{len(CANDIDATES)}", flush=True)
    if arows:
        ap=pd.concat(arows,ignore_index=True); ap["open_time"]=pd.to_datetime(ap["open_time"],utc=True)
        for c in AGGT:
            if c in panel.columns: panel=panel.drop(columns=[c])
        panel=panel.merge(ap,on=["symbol","open_time"],how="left")
        print(f"  aggT merged", flush=True)
    else:
        print("  WARNING: no flow files — aggT skipped (run build_aggtrade_features first)", flush=True)

    # C. v3
    print("--- C. v3 3yr ---", flush=True)
    v3s=[]
    for i,sym in enumerate(CANDIDATES,1):
        v=compute_v3(sym,btc5)
        if v is not None: v3s.append(v)
        if i%10==0: print(f"  v3 {i}/{len(CANDIDATES)}", flush=True)
    v3p=pd.concat(v3s,ignore_index=True); v3p["open_time"]=pd.to_datetime(v3p["open_time"],utc=True)
    panel=panel.merge(v3p,on=["symbol","open_time"],how="left"); del v3s,v3p; gc.collect()
    print(f"  v3 merged", flush=True)

    panel=panel.merge(btc_regime_features(),on="open_time",how="left")
    if "target_z" not in panel.columns: panel=x6.build_target_z(panel)
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")
    if "bars_since_high_xs_rank" not in panel.columns:
        panel["bars_since_high_xs_rank"]=panel.groupby("open_time")["bars_since_high"].rank(pct=True).astype("float32")
    panel.to_parquet(REPO/"outputs/vBTC_features/panel_3yr_v5.parquet",index=False)
    print(f"\n  saved panel_3yr_v5.parquet [{time.time()-t0:.0f}s]", flush=True)

    # E. Test
    folds=x6.get_folds(panel)
    have_aggt = all(c in panel.columns and panel[c].notna().any() for c in AGGT)
    v5_feats=[f for f in x6.BASE+x6.COHORT_EXTRAS+(AGGT if have_aggt else [])+zc if f in panel.columns]
    print(f"\n--- E. Test V5_mv3 ({len(v5_feats)} feats; aggT={'yes' if have_aggt else 'NO'}) ---", flush=True)

    apd0=x6.train_per_sym_ridge(panel,folds,v5_feats,label="x78_v5_single")
    apd0.to_parquet(RCACHE/"x78_v5_single_preds.parquet",index=False)
    m0=x6.run_sleeve_on_preds(RCACHE/"x78_v5_single_preds.parquet","x78_v5_single")
    print(f"  V5_mv3 unconditional: Sharpe={m0.get('sharpe',0):+.2f} folds={m0.get('folds_pos','?')} conc={m0.get('concentration','?')}", flush=True)

    for K in [4,5]:
        apd=routed(panel,folds,v5_feats,K,f"x78_v5_K{K}")
        if apd is None: print(f"  routed K={K} FAILED"); continue
        pth=RCACHE/f"x78_v5_routed_K{K}_preds.parquet"; apd.to_parquet(pth,index=False)
        m=x6.run_sleeve_on_preds(pth,f"x78_v5_routed_K{K}")
        print(f"  V5_mv3 routed K={K}: Sharpe={m.get('sharpe',0):+.2f} folds={m.get('folds_pos','?')} conc={m.get('concentration','?')}", flush=True)

    print(f"\nReference: V0 3yr uncond +0.12, routed K=5 +1.07")
    print(f"Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
