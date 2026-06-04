"""Execution-latency budget for the resid_rev fast edge, at 5m resolution (user Q).
For each 4h decision t: resid_rev book-B long(top-3 pred)/short(bottom-3) basket. Measure the L-S basket return
captured if you ENTER Δ minutes late (5m close at t+Δ) and EXIT at the normal next-4h mark. edge(Δ) vs edge(0)
tells us how many minutes of data-processing+execution we can spend and still keep the edge. Read-only, PIT selection."""
import sys, glob; from pathlib import Path
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
KL = REPO/"data/ml/test/parquet/klines"; OOS = pd.Timestamp("2025-10-04", tz="UTC")
DELTAS = [0,5,10,15,30,45,60,120,240]   # minutes late
# book-B (low-vol) membership + resid_rev preds
pan = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","rvol_7d"]); pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True)
pan = pan[(pan.open_time.dt.hour%4==0)&(pan.open_time.dt.minute==0)]
rv = pan[pan.open_time>=OOS].groupby("symbol")["rvol_7d"].mean().sort_values(); lov = set(rv.index[:94])
pr = pd.read_parquet("live/state/convexity/hl_residrev/v0full_hl60.parquet")[["symbol","open_time","pred"]]
pr["open_time"]=pd.to_datetime(pr["open_time"],utc=True); pr = pr[(pr.open_time>=OOS)&(pr.symbol.isin(lov))]
# per-cycle long/short baskets
picks = {}
for t, g in pr.groupby("open_time"):
    if len(g) < 6: continue
    gg = g.sort_values("pred")
    picks[t] = (gg["symbol"].tail(3).tolist(), gg["symbol"].head(3).tolist())
need_syms = set(s for L,S in picks.values() for s in L+S)
print(f"{len(picks)} cycles, {len(need_syms)} symbols involved; loading 5m closes...")
# load 5m closes for needed syms over OOS
close5 = {}
for sym in need_syms:
    fs = sorted(glob.glob(str(KL/sym/"5m"/"*.parquet")))
    fs = [f for f in fs if f.split("/")[-1][:4] >= "2025"]
    if not fs: continue
    d = pd.concat([pd.read_parquet(f, columns=["open_time","close"]) for f in fs], ignore_index=True)
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True); d=d.drop_duplicates("open_time").set_index("open_time").sort_index()
    close5[sym] = d["close"]
def px(sym, ts):
    s = close5.get(sym)
    if s is None: return np.nan
    i = s.index.searchsorted(ts, side="right")-1   # last 5m close at/before ts (PIT)
    return float(s.iloc[i]) if 0 <= i < len(s) else np.nan
# for each Δ, mean L-S basket return over [t+Δ, t+4h]
rows=[]
for d in DELTAS:
    ls=[]
    for t,(L,S) in picks.items():
        ent=t+pd.Timedelta(minutes=d); ex=t+pd.Timedelta(hours=4)
        lr=[px(s,ex)/px(s,ent)-1 for s in L]; sr=[px(s,ex)/px(s,ent)-1 for s in S]
        lr=[x for x in lr if np.isfinite(x)]; sr=[x for x in sr if np.isfinite(x)]
        if lr and sr: ls.append(np.mean(lr)-np.mean(sr))   # long-short basket return
    ls=np.array(ls); rows.append((d, ls.mean()*1e4, len(ls)))
R=pd.DataFrame(rows, columns=["delay_min","LS_edge_bps","n"])
base=R.loc[R.delay_min==0,"LS_edge_bps"].iloc[0]
R["pct_of_t0"]=R["LS_edge_bps"]/base*100
print("\n=== resid_rev L-S basket edge vs ENTRY DELAY (5m resolution) ===")
print(R.round(2).to_string(index=False))
print(f"\nedge at t+0 = {base:.1f} bps per 4h. Latency budget = where edge stays useful.")
