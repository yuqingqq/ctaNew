"""Add POSITIONING features (from metrics) to V0 and regenerate per-symbol two-book preds. (#170)
Features (all PIT, lag-safe to avoid the contemporaneous bleed that faked oi_chg +0.05):
  oi_chg4h_z   : z-score of 4h OI change through t-1 (the liquidation-flush indicator)
  oi_flush_dec : oi_chg4h_z * I(recent decliner)  -> the P=0.93 flush->bounce conditional
  global_ls(_z): retail long/short ratio (IC +0.017)   smart_minus_retail: positioning divergence (IC -0.015)
  taker_ls(_z) : aggressive taker buy/sell ratio (user: informational)
Covered syms (~50) get the features; uncovered get 0 (neutral) -> per-symbol RidgeCV ignores constant-0.
-> live/state/convexity/hl_pos175/{fullflow_hl60,v0full_hl60}.parquet
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
x6 = tt.x6; V0 = list(tt.V0)
EMB = pd.Timedelta(days=1); HL = 60.0
CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01",
        "2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]
OUT = REPO/"live/state/convexity/hl_pos175"; OUT.mkdir(parents=True, exist_ok=True)
POS = ["oi_chg4h_z","oi_flush_dec","global_ls","global_ls_z","smart_minus_retail","taker_ls","taker_ls_z"]

def build_positioning():
    rows = []
    for f in glob.glob(str(REPO/"data/ml/cache/metrics_*.parquet")):
        d = pd.read_parquet(f)
        if len(d) < 800 or "sum_open_interest" not in d.columns: continue
        d = d.reset_index(); d.columns = ["create_time"]+list(d.columns[1:])
        d["create_time"] = pd.to_datetime(d["create_time"], utc=True); d = d.set_index("create_time").sort_index()
        d = d[~d.index.duplicated(keep="last")]                       # dedupe (new ingested files had dup timestamps)
        r4 = lambda c: d[c].resample("4h").last()
        oi, tl, gl, tk = r4("sum_open_interest"), r4("sum_toptrader_long_short_ratio"), r4("count_long_short_ratio"), r4("sum_taker_long_short_vol_ratio")
        chg = oi.pct_change(1)
        zz = lambda s,w=180: (s-s.rolling(w,min_periods=60).mean())/s.rolling(w,min_periods=60).std()
        g = pd.DataFrame({"open_time": oi.index, "symbol": d["symbol"].iloc[0],
            "oi_chg4h_z": zz(chg).shift(1).values,           # PIT: through t-1
            "global_ls": gl.shift(1).values, "global_ls_z": zz(gl).shift(1).values,
            "smart_minus_retail": (zz(tl)-zz(gl)).shift(1).values,
            "taker_ls": tk.shift(1).values, "taker_ls_z": zz(tk).shift(1).values})
        rows.append(g)
    M = pd.concat(rows, ignore_index=True); M["open_time"] = pd.to_datetime(M["open_time"], utc=True)
    M = M.drop_duplicates(["symbol","open_time"]).reset_index(drop=True)
    return M

F, flowcols = tt.build_flow(); FLOWSYMS = set(F.symbol.unique())   # kept for two-book routing parity
fk = [c for c in ["fl_vpin","fl_vpin_1d","fl_tfi","fl_tfi_1d"] if c in flowcols]
_last = pd.read_parquet(tt.PANEL, columns=["open_time"]); _last["open_time"]=pd.to_datetime(_last["open_time"],utc=True)
CUTS = CUTS + [_last["open_time"].max().normalize()+pd.Timedelta(days=1)]
PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)  # ret_3d already in V0
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN = PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].merge(F, on=["symbol","open_time"], how="left")
PAN = PAN.merge(build_positioning(), on=["symbol","open_time"], how="left").reset_index(drop=True)
PAN["oi_flush_dec"] = PAN["oi_chg4h_z"] * (PAN["ret_3d"] < 0).astype(float)   # flush-on-decliner conditional
for c in POS: PAN[c] = PAN[c].fillna(0.0)                                      # uncovered -> neutral
ncov = (PAN.groupby("symbol")["oi_chg4h_z"].apply(lambda s:(s!=0).any())).sum()
g = PAN.groupby("open_time"); sd = g["return_pct"].transform("std").replace(0,np.nan)
PAN["xs_z"] = ((PAN["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)
PAN = PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
print(f"positioning features active on {ncov} symbols")

def gen(use_flow, outpath):
    rec = []
    for i in range(len(CUTS)-1):
        c0,c1 = CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr = PAN[(PAN.exit_time<fc)&PAN["xs_z"].notna()]; te = PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
        t_end = tr["open_time"].max()
        for sym,gg in tr.groupby("symbol"):
            if len(gg) < 300: continue
            uf = use_flow and (sym in FLOWSYMS) and gg[fk].notna().any().all()
            feats = V0 + POS + (fk if uf else [])
            try:
                s,h = x6.fit_preproc(gg, feats); X = x6.apply_preproc(gg, feats, s, h)
                w = np.exp(-((t_end-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, gg["xs_z"].to_numpy(), sample_weight=w)
                gte = te[te.symbol==sym]
                if len(gte):
                    rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,
                        "alpha_A":gte["alpha_vs_btc_realized"].values,"return_pct":gte["return_pct"].values,
                        "exit_time":gte["exit_time"].values,"pred":m.predict(x6.apply_preproc(gte,feats,s,h)),"fold":i}))
            except Exception: pass
    out = pd.concat(rec, ignore_index=True)
    for c in ("open_time","exit_time"): out[c]=pd.to_datetime(out[c],utc=True)
    out.to_parquet(outpath); return out["symbol"].nunique(), len(out)

a=gen(True, OUT/"fullflow_hl60.parquet"); b=gen(False, OUT/"v0full_hl60.parquet")
print(f"V0+positioning preds: flow {a}, price {b} -> {OUT}")
