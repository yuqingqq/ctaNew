"""TRUE production-model OOS preds: the SAME lean-V0 RidgeCV walk-forward pipeline as
gen_lean_wf_preds.py (base/short book) + gen_residrev_hl.py (long book), but with the monthly
CUTS moved back to cover 2023-01 -> 2025-09 (genuinely held-out time never used to derive any
config lever). Removes the 'different weaker model' confound of fullhist_mpit — same model class,
same features (base book = V0_LEAN, no funding, so funding-coverage gaps don't affect it).
Only remaining caveat: survivorship (175 current symbols; delisted names absent).

Outputs (v0full/no-flow books, which the driver actually consumes):
  live/state/convexity/hl_lean175_oos/v0full_hl60.parquet   (base/short)
  live/state/convexity/hl_residrev_oos/v0full_hl60.parquet  (long)

Usage: python3 live/gen_oos_preds.py [START=2023-01-01] [END=2025-10-01]
"""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
x6 = tt.x6; V0 = list(tt.V0)
V0_LEAN = [f for f in V0 if not f.startswith("funding")]
EMB = pd.Timedelta(days=1); HL = 60.0
RR = ["resid_rev_2", "resid_rev_3"]

START = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2023-01-01", tz="UTC")
END   = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2025-10-01", tz="UTC")
CUTS = list(pd.date_range(START, END, freq="MS", tz="UTC"))
print(f"OOS CUTS: {CUTS[0].date()} .. {CUTS[-1].date()} ({len(CUTS)-1} folds)", flush=True)

PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True); PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
PAN = PAN[(PAN.open_time.dt.hour % 4 == 0) & (PAN.open_time.dt.minute == 0)].sort_values(["symbol","open_time"])
# resid-rev PIT features (long book) — trailing sum of PAST per-bar residual alpha
a = PAN.groupby("symbol")["alpha_vs_btc_realized"]
PAN["resid_rev_2"] = -a.transform(lambda s: s.shift(1).rolling(2).sum())
PAN["resid_rev_3"] = -a.transform(lambda s: s.shift(1).rolling(3).sum())
for c in RR: PAN[c] = PAN[c].fillna(0.0)
_g = PAN.groupby("open_time"); _sd = _g["return_pct"].transform("std").replace(0, np.nan)
PAN["xs_z"] = ((PAN["return_pct"] - _g["return_pct"].transform("mean")) / _sd).clip(-10, 10)
PAN = PAN.sort_values(["symbol","open_time"]).reset_index(drop=True)

def gen(feats, outpath):
    rec = []
    for i in range(len(CUTS)-1):
        c0, c1 = CUTS[i], CUTS[i+1]; fit_cut = c0 - EMB
        tr = PAN[(PAN.exit_time < fit_cut) & PAN["xs_z"].notna()]
        te = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1)]
        if not len(tr) or not len(te): continue
        t_end = tr["open_time"].max()
        for sym, g in tr.groupby("symbol"):
            if len(g) < 300: continue
            try:
                s, h = x6.fit_preproc(g, feats); X = x6.apply_preproc(g, feats, s, h)
                w = np.exp(-((t_end - g["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
                m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, g["xs_z"].to_numpy(), sample_weight=w)
                gte = te[te.symbol == sym]
                if len(gte):
                    rec.append(pd.DataFrame({"symbol": sym, "open_time": gte["open_time"].values,
                        "alpha_A": gte["alpha_vs_btc_realized"].values, "return_pct": gte["return_pct"].values,
                        "exit_time": gte["exit_time"].values, "pred": m.predict(x6.apply_preproc(gte, feats, s, h)), "fold": i}))
            except Exception: pass
        print(f"  fold {i} {c0.date()}: cum {len(rec)} sym-frames", flush=True)
    out = pd.concat(rec, ignore_index=True)
    for c in ("open_time","exit_time"): out[c] = pd.to_datetime(out[c], utc=True)
    outpath.parent.mkdir(parents=True, exist_ok=True); out.to_parquet(outpath)
    return out["symbol"].nunique(), len(out)

print("=== BASE/short book (V0_LEAN) ===", flush=True)
bs = gen(V0_LEAN, REPO/"live/state/convexity/hl_lean175_oos/v0full_hl60.parquet")
print("=== LONG book (V0_LEAN + resid_rev, matches hl_residrev_lean) ===", flush=True)
ls = gen(V0_LEAN + RR, REPO/"live/state/convexity/hl_residrev_oos/v0full_hl60.parquet")
print(f"DONE base {bs}, long {ls}", flush=True)
