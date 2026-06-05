"""Four-leg META-LABELING (#172): keep base Ridge pred as the alpha RANKER; train SEPARATE leg-specific
tradeability classifiers (A-long, A-short, B-long, B-short) predicting forward-alpha SIGN from candidate-level
features, then blend: pred_long = zcyc(pred) + LAM*zcyc(P_long_good); pred_short = zcyc(pred) - LAM*zcyc(P_short_good).
Principle: separate alpha-direction from tail-state. Candidate features (PIT): resid_rev_{2,3,6} (BTC-residual reversal),
danger (z atr + z idio_vol - z corr), pred, pred_rank, pred_disp, btc_rvol_7d (regime).
-> live/state/convexity/hl_meta/{fullflow_hl60,v0full_hl60}.parquet  (adds pred_long, pred_short cols)
Validate ONLY through full replay (AB_HLDIR=hl_meta), both cadences + bootstrap.
"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
OOS = pd.Timestamp("2025-10-04", tz="UTC")
CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01",
        "2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]
LAM = float(__import__("os").environ.get("META_LAM", "1.0"))
HLBASE = REPO/"live/state/convexity/hl"; OUT = REPO/"live/state/convexity/hl_meta"; OUT.mkdir(parents=True, exist_ok=True)
FEATS = ["resid_rev_2","resid_rev_3","resid_rev_6","danger","pred","pred_rank","pred_disp","btc_rvol_7d"]

# panel features for danger + regime + static book membership
PF = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","atr_pct","idio_vol_to_btc_1h","corr_to_btc_1d","rvol_7d","btc_rvol_7d"])
PF["open_time"] = pd.to_datetime(PF["open_time"], utc=True)
PF = PF[(PF.open_time.dt.hour%4==0)&(PF.open_time.dt.minute==0)]
rvmean = PF[PF.open_time>=OOS].groupby("symbol")["rvol_7d"].mean().sort_values(ascending=False)
HIVOL = set(rvmean.index[:80])   # static book A membership for meta training

_last = pd.read_parquet(tt.PANEL, columns=["open_time"]); _last["open_time"]=pd.to_datetime(_last["open_time"],utc=True)
CUTS_ALL = CUTS + [_last["open_time"].max().normalize()+pd.Timedelta(days=1)]

def zc(s, by):  # cross-sectional z within cycle
    g = s.groupby(by); return (s - g.transform("mean")) / g.transform("std").replace(0, np.nan)

def process(fname, book_syms, label):
    d = pd.read_parquet(HLBASE/fname)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True); d["exit_time"] = pd.to_datetime(d["exit_time"], utc=True)
    d = d.sort_values(["symbol","open_time"])
    # PIT resid-rev (trailing sum of PAST realized residual alpha; 4h label => no overlap)
    a = d.groupby("symbol")["alpha_A"]
    d["resid_rev_2"] = -a.transform(lambda s: s.shift(1).rolling(2).sum())
    d["resid_rev_3"] = -a.transform(lambda s: s.shift(1).rolling(3).sum())
    d["resid_rev_6"] = -a.transform(lambda s: s.shift(1).rolling(6).sum())
    d = d.merge(PF[["symbol","open_time","atr_pct","idio_vol_to_btc_1h","corr_to_btc_1d","btc_rvol_7d"]],
                on=["symbol","open_time"], how="left")
    d["danger"] = (zc(d["atr_pct"], d["open_time"]) + zc(d["idio_vol_to_btc_1h"], d["open_time"])
                   - zc(d["corr_to_btc_1d"], d["open_time"]))
    d["pred_rank"] = d.groupby("open_time")["pred"].rank(pct=True)
    d["pred_disp"] = d.groupby("open_time")["pred"].transform("std")
    for c in FEATS: d[c] = d[c].fillna(0.0)
    d["long_good"] = (d["alpha_A"] > 0).astype(int)
    d["short_good"] = (d["alpha_A"] < 0).astype(int)
    d["P_long"] = 0.5; d["P_short"] = 0.5
    bk = d[d.symbol.isin(book_syms)]   # train on static book membership only
    for i in range(len(CUTS_ALL)-1):
        c0, c1 = CUTS_ALL[i], CUTS_ALL[i+1]; fc = c0 - pd.Timedelta(days=1)
        tr = bk[(bk.exit_time < fc)]; te_idx = d[(d.open_time>=c0)&(d.open_time<c1)].index
        if len(tr) < 2000 or len(te_idx) == 0: continue
        Xtr = tr[FEATS].to_numpy(); mu, sg = Xtr.mean(0), Xtr.std(0); sg[sg==0] = 1
        Xte = (d.loc[te_idx, FEATS].to_numpy() - mu) / sg
        for side, P in [("long_good","P_long"), ("short_good","P_short")]:
            y = tr[side].to_numpy()
            if y.sum() < 50 or (len(y)-y.sum()) < 50: continue
            m = LogisticRegression(C=1.0, max_iter=200).fit((Xtr-mu)/sg, y)
            d.loc[te_idx, P] = m.predict_proba(Xte)[:, 1]
    d["pred_long"]  = zc(d["pred"], d["open_time"]).fillna(0) + LAM*zc(d["P_long"],  d["open_time"]).fillna(0)
    d["pred_short"] = zc(d["pred"], d["open_time"]).fillna(0) - LAM*zc(d["P_short"], d["open_time"]).fillna(0)
    keep = ["symbol","open_time","alpha_A","return_pct","exit_time","pred","pred_long","pred_short","fold","P_long","P_short"]
    d[keep].to_parquet(OUT/fname)
    print(f"{label}: {fname} rows={len(d)} | P_long mean {d.P_long.mean():.3f} P_short mean {d.P_short.mean():.3f}")

process("fullflow_hl60.parquet", HIVOL, "book A (flow)")
process("v0full_hl60.parquet", set(rvmean.index)-HIVOL, "book B (v0)")
print(f"meta-labels written -> {OUT}  (LAM={LAM})")
