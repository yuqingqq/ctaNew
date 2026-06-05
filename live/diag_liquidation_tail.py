"""Does OI/liquidation-cascade data help the high-vol FAT TAIL? (user idea 2026-06-03)
Hypothesis: high-vol mean-reversion fails on LIQUIDATION CASCADES. OI-change distinguishes them:
  price falling + OI COLLAPSING  = forced deleveraging -> keeps falling (don't fade)
  price falling + OI RISING      = new shorts / overreaction -> bounces (fade works)
Decisive test: among fade-setups (recent decliners), does OI-change separate continued-falls from bounces,
especially in the fat tail? Uses cached metrics (50 syms). Read-only.
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt

# --- cascade proxies from metrics, on 4h grid, PIT ---
rows = []
for f in glob.glob(str(REPO/"data/ml/cache/metrics_*.parquet")):
    d = pd.read_parquet(f)
    if "sum_open_interest" not in d.columns or len(d) < 500: continue
    d = d.reset_index().rename(columns={d.index.name or "index":"create_time"})
    d["create_time"] = pd.to_datetime(d["create_time"], utc=True)
    d = d.set_index("create_time").sort_index()
    oi = d["sum_open_interest"].resample("4h").last()
    g = pd.DataFrame({"open_time": oi.index, "symbol": d["symbol"].iloc[0],
                      "oi": oi.values})
    g["oi_chg_1d"] = g["oi"].pct_change(6)        # 24h OI change (6 x 4h)
    g["oi_chg_4h"] = g["oi"].pct_change(1)
    ls = d["sum_toptrader_long_short_ratio"].resample("4h").last()
    tk = d["sum_taker_long_short_vol_ratio"].resample("4h").last()
    g["ls_ratio"] = ls.reindex(oi.index).values
    g["taker_ratio"] = tk.reindex(oi.index).values
    rows.append(g)
M = pd.concat(rows, ignore_index=True)
M["open_time"] = pd.to_datetime(M["open_time"], utc=True)

PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","return_pct","ret_3d","return_1d","rvol_7d"])
PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True)
PAN = PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)]
PAN = PAN[PAN.open_time >= pd.Timestamp("2025-10-04", tz="UTC")]
g = PAN.groupby("open_time"); PAN["fwd"] = PAN["return_pct"] - g["return_pct"].transform("mean")
D = PAN.merge(M[["symbol","open_time","oi_chg_1d","oi_chg_4h","ls_ratio","taker_ratio"]], on=["symbol","open_time"], how="inner")
print(f"metrics-covered rows in OOS: {len(D)} ({D.symbol.nunique()} syms)\n")

def ic(df, f):
    s = df[[f,"fwd"]].dropna(); return s[f].corr(s["fwd"], method="spearman") if len(s) > 200 else np.nan

print("=== 1. cascade-proxy univariate IC vs fwd demeaned return ===")
for f in ["oi_chg_1d","oi_chg_4h","ls_ratio","taker_ratio"]:
    print(f"  {f:12s}: all={ic(D,f):+.4f} | hi-vol(top-rvol-half)={ic(D[D.rvol_7d>D.rvol_7d.median()],f):+.4f}")

# --- KEY: among recent DECLINERS (the long-fade setups), does OI-change separate cascade vs bounce? ---
print("\n=== 2. KEY: among recent decliners (ret_3d in bottom 25%), does OI-change predict bounce vs continued fall? ===")
dec = D[D.ret_3d <= D.ret_3d.quantile(0.25)].dropna(subset=["oi_chg_1d"])
hi = dec[dec.oi_chg_1d >= dec.oi_chg_1d.median()]   # OI rising (new shorts / overreaction -> expect bounce)
lo = dec[dec.oi_chg_1d <  dec.oi_chg_1d.median()]   # OI falling (forced deleverage -> expect keep falling)
print(f"  decliners n={len(dec)}")
print(f"  OI-RISING decliners (overreaction?):  mean fwd {hi.fwd.mean():+.5f}  median {hi.fwd.median():+.5f}  win {(hi.fwd>0).mean():.3f}")
print(f"  OI-FALLING decliners (cascade?):      mean fwd {lo.fwd.mean():+.5f}  median {lo.fwd.median():+.5f}  win {(lo.fwd>0).mean():.3f}")
print(f"  >>> spread (rising - falling) = {hi.fwd.mean()-lo.fwd.mean():+.5f}  ({'CASCADE PROXY WORKS: OI-fall decliners keep falling' if hi.fwd.mean()>lo.fwd.mean() else 'no separation'})")

# --- 3. fat tail: in worst-decile fwd outcomes, is OI-change distinctively negative? ---
print("\n=== 3. FAT-TAIL: worst-10% fwd outcomes vs rest — OI-change signature ===")
tail = D[D.fwd <= D.fwd.quantile(0.10)].dropna(subset=["oi_chg_1d"]); rest = D[D.fwd > D.fwd.quantile(0.10)].dropna(subset=["oi_chg_1d"])
print(f"  worst-10% fwd: mean oi_chg_1d {tail.oi_chg_1d.mean():+.4f} | rest {rest.oi_chg_1d.mean():+.4f} | "
      f"ls_ratio {tail.ls_ratio.mean():.3f} vs {rest.ls_ratio.mean():.3f}")
print(f"  >>> {'tail has distinct OI signature -> predictable' if abs(tail.oi_chg_1d.mean()-rest.oi_chg_1d.mean())>0.005 else 'tail OI signature weak'}")
