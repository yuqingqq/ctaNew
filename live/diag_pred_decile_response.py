"""Mechanism diagnostic: at each cycle, bin all symbols by pred decile (1=lowest pred,
10=highest), then average forward residual within each decile. Plot the response curve
H1 vs H2. If the H1 curve is monotonically increasing across all deciles (mean-reversion
works both directions) but H2 is monotonic only at the bottom (shorts work) and flat/
declining at the top (longs broken), we can pinpoint exactly where the model's
prediction-to-realization map broke.

Also test: autocorrelation of residual returns at the per-symbol level. If the residual
return AC went from negative (mean-reversion) to positive (persistence) between H1 and
H2, that's the operative cause — the underlying market property the model relies on
has flipped sign.
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
REPO = Path("/home/yuqing/ctaNew")
PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"

print("loading preds + panel...")
d = pd.read_parquet(PREDS)
d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
d = d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)]

H1 = (pd.Timestamp("2025-10-04",tz="UTC"), pd.Timestamp("2026-01-22",tz="UTC"))
H2 = (pd.Timestamp("2026-01-22",tz="UTC"), pd.Timestamp("2026-05-26",tz="UTC"))

# ============ TEST A: pred-decile -> forward residual response curve ============
print("\n=== TEST A: per-pred-decile average forward residual (alpha_A) ===")
def decile_response(g):
    if len(g) < 20: return None
    g = g.copy()
    g["decile"] = pd.qcut(g["pred"], 10, labels=False, duplicates="drop")
    return g.groupby("decile")["alpha_A"].mean()

for label,(s,e) in [("H1",H1),("H2",H2)]:
    sub = d[(d["open_time"]>=s)&(d["open_time"]<e)]
    # per-cycle decile means → list of Series → DataFrame (cycles × deciles) → mean across cycles
    per_cycle_list = []
    for ot, g in sub.groupby("open_time"):
        r = decile_response(g)
        if r is not None: per_cycle_list.append(r)
    if not per_cycle_list: continue
    by_dec = pd.DataFrame(per_cycle_list)
    mean_per_decile = by_dec.mean() * 1e4   # bps
    print(f"\n  {label} (n_cycles={sub['open_time'].nunique()})  — bps per decile:")
    print(f"   decile:  {' '.join(f'd{int(i):>2}' for i in mean_per_decile.index)}")
    print(f"   alpha:   {' '.join(f'{v:+5.1f}' for v in mean_per_decile.values)}")
    # interpret: monotonic increasing = model rank works; flat top = long signal broken
    top = mean_per_decile.iloc[-3:].mean(); bot = mean_per_decile.iloc[:3].mean()
    print(f"   top 3 deciles avg: {top:+.2f} bps   bot 3 deciles avg: {bot:+.2f} bps   spread: {top-bot:+.2f}")

# ============ TEST B: per-sym residual autocorrelation H1 vs H2 ============
print("\n\n=== TEST B: residual (alpha_A) autocorrelation per sym, H1 vs H2 ===")
print("If H1 AC<0 (mean-reversion) and H2 AC~0 or >0 (persistence), the regime flipped.")
ac1_list, ac2_list = [], []
syms = d["symbol"].unique()
for sym in syms:
    s_data = d[d["symbol"]==sym].sort_values("open_time")
    a1 = s_data[(s_data["open_time"]>=H1[0])&(s_data["open_time"]<H1[1])]["alpha_A"]
    a2 = s_data[(s_data["open_time"]>=H2[0])&(s_data["open_time"]<H2[1])]["alpha_A"]
    if len(a1)>=50: ac1_list.append(a1.autocorr(lag=1))
    if len(a2)>=50: ac2_list.append(a2.autocorr(lag=1))
print(f"\n  H1 per-sym AC(lag=1) of residual: n={len(ac1_list)}, mean {np.nanmean(ac1_list):+.4f}, median {np.nanmedian(ac1_list):+.4f}")
print(f"  H2 per-sym AC(lag=1) of residual: n={len(ac2_list)}, mean {np.nanmean(ac2_list):+.4f}, median {np.nanmedian(ac2_list):+.4f}")
print(f"  %syms with H1 AC < 0: {100*np.mean(np.array(ac1_list)<0):.1f}%")
print(f"  %syms with H2 AC < 0: {100*np.mean(np.array(ac2_list)<0):.1f}%")

# ============ TEST C: per-sym corr_to_btc H1 vs H2 (does the market couple up?) ============
print("\n\n=== TEST C: per-sym BTC correlation H1 vs H2 (coupling test) ===")
p = pd.read_parquet(PANEL, columns=["symbol","open_time","corr_to_btc_1d"])
p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
p = p[(p["open_time"].dt.hour%4==0)&(p["open_time"].dt.minute==0)]
c1 = p[(p["open_time"]>=H1[0])&(p["open_time"]<H1[1])].groupby("symbol")["corr_to_btc_1d"].mean()
c2 = p[(p["open_time"]>=H2[0])&(p["open_time"]<H2[1])].groupby("symbol")["corr_to_btc_1d"].mean()
mm = pd.concat([c1.rename("corr_H1"), c2.rename("corr_H2")], axis=1).dropna()
mm["change"] = mm["corr_H2"] - mm["corr_H1"]
print(f"  syms n={len(mm)}")
print(f"  H1 mean corr_to_btc_1d (per-sym): {mm['corr_H1'].mean():.3f}  median: {mm['corr_H1'].median():.3f}")
print(f"  H2 mean corr_to_btc_1d (per-sym): {mm['corr_H2'].mean():.3f}  median: {mm['corr_H2'].median():.3f}")
print(f"  median change H2-H1: {mm['change'].median():+.3f}")
print(f"  %syms with HIGHER btc-correlation in H2: {100*(mm['change']>0).mean():.0f}%")

# ============ TEST D: cross-sectional dispersion of forward residuals H1 vs H2 ============
print("\n\n=== TEST D: cross-sectional std of forward residual (alpha_A) by cycle ===")
for label,(s,e) in [("H1",H1),("H2",H2)]:
    sub = d[(d["open_time"]>=s)&(d["open_time"]<e)]
    xs_std = sub.groupby("open_time")["alpha_A"].std()
    print(f"  {label}: mean cross-sec std of forward alpha = {xs_std.mean()*1e4:.1f} bps  median {xs_std.median()*1e4:.1f} bps  n_cycles={len(xs_std)}")
