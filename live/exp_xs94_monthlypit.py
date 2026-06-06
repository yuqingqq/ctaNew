"""Apply MONTHLY-PIT rvol exclude (re-rank top-80 high-vol per fold, as-of each fold cut) to the regenerated
preds — the production-faithful universe (vs the single frozen 5/29 set my first experiment used). Validates
the +4.32 baseline, then this is the correct universe to re-run XS94 on.
"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
SPLIT_N = 80
CUTS = [pd.Timestamp(t, tz="UTC") for t in ["2025-10-04","2025-11-01","2025-12-01","2026-01-01",
        "2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27","2026-06-30"]]
# trailing-30d rvol per symbol per fold-cut (PIT)
pan = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","rvol_7d"]); pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True)
def excl_for(c0):
    lo = c0 - pd.Timedelta(days=30)
    rv = pan[(pan.open_time>=lo)&(pan.open_time<c0)].groupby("symbol")["rvol_7d"].mean().dropna()
    return set(rv.sort_values(ascending=False).index[:SPLIT_N])     # top-80 high-vol as-of c0

arm = sys.argv[1] if len(sys.argv)>1 else "baseline"
src = REPO/f"live/state/exp_xs94/{arm}"
for book,full in [("base","short_full"),("long","long_full")]:
    d = pd.read_parquet(src/f"{full}.parquet"); d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    keep = []
    for i in range(len(CUTS)-1):
        ex = excl_for(CUTS[i])
        w = d[(d.open_time>=CUTS[i])&(d.open_time<CUTS[i+1])]
        keep.append(w[~w["symbol"].isin(ex)])     # monthly-PIT exclude
    out = pd.concat(keep, ignore_index=True)
    out.to_parquet(src/f"{book}_mpit.parquet")
    print(f"[{arm}] {book}: monthly-PIT filtered {len(d)}->{len(out)} rows", flush=True)
print("DONE mpit filter")
