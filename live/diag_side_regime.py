"""Diagnose WHY side regime is broken in H2.

Hypothesis (per user): the current market "feels bearish" even though the 30d-return
classifier calls it side (btc_30d in [-10%, +10%]). If H2 side cycles have systematically
lower BTC trend / higher correlation / lower XS dispersion than H1 side cycles, the V0
mean-reversion alpha would naturally underperform — it depends on dispersion + neutral
correlation, both of which compress in stealth-bear conditions.

Comparing H1 (Oct 4 → Jan 22) and H2 (Jan 22 → May 11) on side-regime cycles only:
  - btc_ret_30d distribution
  - pred_disp (XS std of preds at each cycle) — proxy for tradable dispersion
  - per-cycle V0 IC = corr(pred, realized 4h return) — does the model still rank?
  - BTC trend within the 30d window (slope sign) — stealth bear?
"""
import pandas as pd, numpy as np
from pathlib import Path
REPO = Path("/home/yuqing/ctaNew")
PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
KLINES = REPO/"data/ml/test/parquet/klines"

def load_close_4h(sym):
    sd = KLINES/sym/"5m"
    if not sd.exists(): return pd.Series(dtype="float64")
    dfs = [pd.read_parquet(f, columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    if not dfs: return pd.Series(dtype="float64")
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    c = df.set_index("open_time")["close"].astype(np.float64)
    return c[(c.index.hour%4==0)&(c.index.minute==0)]

# load preds
d = pd.read_parquet(PREDS)
d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
d = d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)]

# BTC trailing 30d return + recent trend
btc = load_close_4h("BTCUSDT")
btc30 = (btc/btc.shift(180)-1).rename("btc_ret_30d")
btc7  = (btc/btc.shift(42)-1).rename("btc_ret_7d")
btc1  = (btc/btc.shift(6)-1).rename("btc_ret_1d")
d = d.merge(btc30.reset_index(), on="open_time", how="left")
d = d.merge(btc7.reset_index(),  on="open_time", how="left")
d = d.merge(btc1.reset_index(),  on="open_time", how="left")
d["regime"] = np.where(d["btc_ret_30d"]>0.10,"bull",np.where(d["btc_ret_30d"]<-0.10,"bear","side"))

# per-cycle features
def per_cycle(g):
    return pd.Series(dict(
        n=len(g),
        pred_disp=g["pred"].std(),
        ic_4h=g["pred"].rank().corr(g["return_pct"].rank()) if len(g)>=5 else np.nan,
        median_ret=g["return_pct"].median(),
        std_ret=g["return_pct"].std(),
        btc_ret_30d=g["btc_ret_30d"].iloc[0],
        btc_ret_7d=g["btc_ret_7d"].iloc[0],
        btc_ret_1d=g["btc_ret_1d"].iloc[0],
        regime=g["regime"].iloc[0],
    ))

cyc = d.groupby("open_time").apply(per_cycle).reset_index()

# define windows
H1 = (pd.Timestamp("2025-10-04",tz="UTC"), pd.Timestamp("2026-01-22",tz="UTC"))
H2 = (pd.Timestamp("2026-01-22",tz="UTC"), pd.Timestamp("2026-05-11",tz="UTC"))

# side-only comparison
for label, (s,e) in [("H1", H1), ("H2", H2)]:
    sub = cyc[(cyc["open_time"]>=s) & (cyc["open_time"]<e) & (cyc["regime"]=="side")]
    print(f"\n=== {label} SIDE cycles ===  n={len(sub)}")
    print(f"  btc_30d:  mean {sub['btc_ret_30d'].mean():+.3f}  median {sub['btc_ret_30d'].median():+.3f}  std {sub['btc_ret_30d'].std():.3f}")
    print(f"  btc_7d:   mean {sub['btc_ret_7d'].mean():+.3f}  median {sub['btc_ret_7d'].median():+.3f}")
    print(f"  btc_1d:   mean {sub['btc_ret_1d'].mean():+.4f}  median {sub['btc_ret_1d'].median():+.4f}")
    print(f"  V0 IC(4h): mean {sub['ic_4h'].mean():+.4f}  median {sub['ic_4h'].median():+.4f}  >0%: {100*(sub['ic_4h']>0).mean():.1f}%")
    print(f"  pred_disp (XS std): mean {sub['pred_disp'].mean():.3f}  median {sub['pred_disp'].median():.3f}")
    print(f"  XS realized std: mean {sub['std_ret'].mean()*100:.2f}%  median {sub['std_ret'].median()*100:.2f}%")
    # stealth-bear classification: btc_30d in [-10,+10] AND btc_7d < 0 AND btc_1d < 0
    sb = sub[(sub["btc_ret_7d"]<0)&(sub["btc_ret_1d"]<0)]
    print(f"  'stealth bear' subset (7d<0 AND 1d<0): {len(sb)} of {len(sub)} = {100*len(sb)/max(1,len(sub)):.0f}%")

# overall stealth-bear test in H2
print("\n=== H2 side cycles, conditional on BTC trend direction ===")
h2s = cyc[(cyc["open_time"]>=H2[0]) & (cyc["open_time"]<H2[1]) & (cyc["regime"]=="side")]
for label, mask in [
    ("btc_7d > 0  (stealth bull)",     h2s["btc_ret_7d"]>0),
    ("btc_7d < 0  (stealth bear)",     h2s["btc_ret_7d"]<0),
    ("btc_7d in [-3%, +3%]  (true side)", (h2s["btc_ret_7d"]>-0.03)&(h2s["btc_ret_7d"]<0.03)),
]:
    sub = h2s[mask]
    if len(sub)<5: print(f"  {label}: too few ({len(sub)})"); continue
    print(f"  {label}: n={len(sub)}  IC mean {sub['ic_4h'].mean():+.4f}  pred_disp {sub['pred_disp'].mean():.3f}")

# show monthly side-IC trajectory
cyc["month"] = cyc["open_time"].dt.to_period("M").astype(str)
print("\n=== Monthly side-regime IC trajectory ===")
m = cyc[cyc["regime"]=="side"].groupby("month").agg(n=("ic_4h","count"),ic=("ic_4h","mean"),btc30=("btc_ret_30d","mean"),btc7=("btc_ret_7d","mean"),pdisp=("pred_disp","mean"))
print(m.round(4).to_string())
