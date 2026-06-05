"""#185 — walk-forward preds with CROSS-EXCHANGE PREMIUM feature added to V0 (base/short) and V0+resid_rev (long).
premium = log(other_venue/binance), XS-demeaned per 4h bar, PIT shift (default 1; XE_LAG=2 for latency robustness).
Conditionally per-symbol (only if that symbol has venue coverage, mirroring the flow `uf` pattern). fillna(0)=XS-neutral.

Outputs (v0full only; fullflow copied from baseline for harness membership):
  live/state/convexity/hl_xexch/v0full_hl60.parquet           (V0 + xexch)            short ranker
  live/state/convexity/hl_xexch_residrev/v0full_hl60.parquet  (V0 + resid_rev + xexch) long ranker
Env: XE_VENUES=okx,coinbase  XE_FEATS=level (or level,chg3)  XE_LAG=1
"""
import os, sys, shutil; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
import live.convexity_paper_bot as bot
x6 = tt.x6; V0 = list(tt.V0); EMB = pd.Timedelta(days=1); HL = 60.0
RR = ["resid_rev_2","resid_rev_3"]
XE = REPO/"data/ml/cache/xexch"
VENUES = os.environ.get("XE_VENUES","okx").split(",")
KINDS  = os.environ.get("XE_FEATS","level").split(",")
LAG    = int(os.environ.get("XE_LAG","1"))
CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01",
        "2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]

PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
PAN["open_time"]=pd.to_datetime(PAN["open_time"],utc=True); PAN["exit_time"]=pd.to_datetime(PAN["exit_time"],utc=True)
PAN = PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(["symbol","open_time"])
_last = PAN["open_time"].max().normalize()+pd.Timedelta(days=1); CUTS = CUTS + [_last]
a = PAN.groupby("symbol")["alpha_vs_btc_realized"]
PAN["resid_rev_2"] = -a.transform(lambda s: s.shift(1).rolling(2).sum())
PAN["resid_rev_3"] = -a.transform(lambda s: s.shift(1).rolling(3).sum())
for c in RR: PAN[c]=PAN[c].fillna(0.0)
g = PAN.groupby("open_time"); sd=g["return_pct"].transform("std").replace(0,np.nan)
PAN["xs_z"]=((PAN["return_pct"]-g["return_pct"].transform("mean"))/sd).clip(-10,10)

# ---- build cross-exchange premium features, merge into PAN ----
def venue_premium(venue):
    prem = {}
    for f in (XE/venue).glob("*.parquet"):
        s = f.stem
        cv = pd.read_parquet(f); cv["open_time"]=pd.to_datetime(cv["open_time"],utc=True)
        cv = cv.set_index("open_time")["close"].astype(float)
        bn = bot.load_close_4h(s)
        if bn is None or not len(bn): continue
        j = pd.concat([np.log(cv).rename("o"), np.log(bn).rename("b")], axis=1).dropna()
        if len(j) < 100: continue
        prem[s] = (j["o"]-j["b"]).rename(s)
    if not prem: return pd.DataFrame()
    P = pd.concat(prem.values(), axis=1); Pxs = P.sub(P.median(axis=1), axis=0)
    return Pxs

XFEATS = []
PREM_PARQUET = os.environ.get("XE_PREMIUM_PARQUET")   # #185 FIX: boundary-aligned premium precomputed (hh:00 open, all venues)
if PREM_PARQUET:
    ap = pd.read_parquet(PREM_PARQUET); ap["open_time"] = pd.to_datetime(ap["open_time"], utc=True)
    cols = [c for c in ap.columns if c.endswith("_level") and c.split("_")[0] in VENUES]
    PAN = PAN.merge(ap[["symbol","open_time"]+cols], on=["symbol","open_time"], how="left"); XFEATS += cols
    print(f"loaded ALIGNED premium {cols} from {PREM_PARQUET}")
else:
  for venue in VENUES:
    Pxs = venue_premium(venue)
    if Pxs.empty: print(f"[warn] no {venue} data"); continue
    feat_frames = {}
    if "level" in KINDS: feat_frames[f"{venue}_level"] = Pxs.shift(LAG)
    if "chg3"  in KINDS: feat_frames[f"{venue}_chg3"]  = (Pxs - Pxs.shift(3)).shift(LAG)
    if "chg1"  in KINDS: feat_frames[f"{venue}_chg1"]  = (Pxs - Pxs.shift(1)).shift(LAG)
    for fname, Fdf in feat_frames.items():
        long = Fdf.reset_index().melt(id_vars="open_time", var_name="symbol", value_name=fname)
        long["open_time"]=pd.to_datetime(long["open_time"],utc=True)
        PAN = PAN.merge(long, on=["symbol","open_time"], how="left"); XFEATS.append(fname)
# coverage mask per (symbol): does symbol have ANY non-null xexch in OOS?
cov = {f: set(PAN.loc[PAN[f].notna(),"symbol"].unique()) for f in XFEATS}
for f in XFEATS: PAN[f] = PAN[f].fillna(0.0)
PAN = PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)
print(f"xexch feats {XFEATS} (lag={LAG}); coverage "+", ".join(f"{f}:{len(cov[f])}" for f in XFEATS))

def gen_v0full(extra_base, outdir):
    """extra_base: list added to V0 for the SHORT/base model; long model = base + RR. Both get xexch if covered."""
    rec_b, rec_l = [], []
    for i in range(len(CUTS)-1):
        c0,c1 = CUTS[i],CUTS[i+1]; fc=c0-EMB
        tr = PAN[(PAN.exit_time<fc)&PAN["xs_z"].notna()]; te = PAN[(PAN.open_time>=c0)&(PAN.open_time<c1)]
        t_end = tr["open_time"].max()
        for sym,gg in tr.groupby("symbol"):
            if len(gg) < 300: continue
            xf = [f for f in XFEATS if sym in cov[f]]
            gte = te[te.symbol==sym]
            if not len(gte): continue
            w = np.exp(-((t_end-gg["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
            y = gg["xs_z"].to_numpy()
            for feats, rec in [(V0+xf, rec_b), (V0+RR+xf, rec_l)]:
                try:
                    s,h = x6.fit_preproc(gg, feats); X = x6.apply_preproc(gg, feats, s, h)
                    m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, y, sample_weight=w)
                    rec.append(pd.DataFrame({"symbol":sym,"open_time":gte["open_time"].values,
                        "alpha_A":gte["alpha_vs_btc_realized"].values,"return_pct":gte["return_pct"].values,
                        "exit_time":gte["exit_time"].values,"pred":m.predict(x6.apply_preproc(gte,feats,s,h)),"fold":i}))
                except Exception: pass
    base_dir, long_dir = outdir
    for rec, d in [(rec_b, base_dir), (rec_l, long_dir)]:
        d.mkdir(parents=True, exist_ok=True)
        out = pd.concat(rec, ignore_index=True)
        for c in ("open_time","exit_time"): out[c]=pd.to_datetime(out[c],utc=True)
        out.to_parquet(d/"v0full_hl60.parquet")
        shutil.copy(REPO/"live/state/convexity/hl/fullflow_hl60.parquet", d/"fullflow_hl60.parquet")  # membership only
    return len(rec_b), len(rec_l)

base_dir = REPO/"live/state/convexity/hl_xexch"; long_dir = REPO/"live/state/convexity/hl_xexch_residrev"
nb, nl = gen_v0full([], (base_dir, long_dir))
print(f"DONE base rows {nb} -> {base_dir} ; long rows {nl} -> {long_dir}")
