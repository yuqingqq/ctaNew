"""v4 (RESIDUAL-target) OOS preds — exact mirror of gen_oos_preds.py (same WF pipeline, cuts, features,
HL, embargo, survivorship caveat) but the training label is xs_z(alpha_vs_btc_realized) instead of
xs_z(return_pct). Produces the v4 analogues of hl_lean175_oos / hl_residrev_oos so v4 can be scored OOS.

Outputs:
  live/state/convexity/hl_v4base_oos/v0full_hl60.parquet   (base/short, V0_LEAN, residual target)
  live/state/convexity/hl_v4long_oos/v0full_hl60.parquet   (long, V0_LEAN+resid_rev, residual target)
Usage: python3 live/gen_oos_v4.py [START=2023-01-01] [END=2025-10-01]
"""
import sys
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
print(f"v4 OOS CUTS: {CUTS[0].date()} .. {CUTS[-1].date()} ({len(CUTS)-1} folds)", flush=True)

PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","exit_time","return_pct","alpha_vs_btc_realized"]+V0)
PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True); PAN["exit_time"] = pd.to_datetime(PAN["exit_time"], utc=True)
PAN = PAN[(PAN.open_time.dt.hour % 4 == 0) & (PAN.open_time.dt.minute == 0)].sort_values(["symbol","open_time"])
a = PAN.groupby("symbol")["alpha_vs_btc_realized"]
PAN["resid_rev_2"] = -a.transform(lambda s: s.shift(1).rolling(2).sum())
PAN["resid_rev_3"] = -a.transform(lambda s: s.shift(1).rolling(3).sum())
for c in RR: PAN[c] = PAN[c].fillna(0.0)
_g = PAN.groupby("open_time"); _sd = _g["alpha_vs_btc_realized"].transform("std").replace(0, np.nan)
PAN["xs_z"] = ((PAN["alpha_vs_btc_realized"] - _g["alpha_vs_btc_realized"].transform("mean")) / _sd).clip(-10, 10)
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

print("=== v4 BASE/short book (V0_LEAN, residual target) ===", flush=True)
bs = gen(V0_LEAN, REPO/"live/state/convexity/hl_v4base_oos/v0full_hl60.parquet")
print("=== v4 LONG book (V0_LEAN + resid_rev, residual target) ===", flush=True)
ls = gen(V0_LEAN + RR, REPO/"live/state/convexity/hl_v4long_oos/v0full_hl60.parquet")
print(f"DONE base {bs}, long {ls}", flush=True)
