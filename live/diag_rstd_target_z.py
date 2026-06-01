"""Deep diagnosis: how does the 7-day rolling rstd influence target_z, and how does
that influence pred_disp downstream?

Chain: realized vol -> rstd (7d rolling per-sym) -> target_z = alpha/rstd -> model pred -> pred_disp

Tests:
1. rstd trajectory over time (monthly mean across syms) — has it grown in 2026?
2. target_z |abs| trajectory over time — has it compressed?
3. correlation pred_disp vs rstd_mean (per cycle)
4. per-symbol view: which syms had the biggest rstd inflation?
5. counterfactual: what would target_z dispersion look like with 30/60/90d rstd?
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
REPO = Path("/home/yuqing/ctaNew")
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"

# Load panel with rmean/rstd/target_z + alpha_vs_btc_realized
print("loading panel + preds...")
p = pd.read_parquet(PANEL, columns=["symbol","open_time","alpha_vs_btc_realized","rmean","rstd","target_z"])
p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
p = p[(p["open_time"].dt.hour%4==0)&(p["open_time"].dt.minute==0)]
print(f"  panel: {len(p):,} rows × {p['symbol'].nunique()} syms × {p['open_time'].min().date()}→{p['open_time'].max().date()}")

# === 1) rstd trajectory monthly (cross-sec mean, median, p90) ===
p["month"] = p["open_time"].dt.to_period("M").astype(str)
print("\n=== 1) rstd monthly (cross-sec stats across syms) ===")
m = p.groupby("month")["rstd"].agg(["mean","median",lambda x:x.quantile(0.9)]).rename(columns={"<lambda_0>":"p90"}).round(5)
print(m.tail(20).to_string())

# === 2) target_z |abs| monthly (model's actual training signal magnitude) ===
print("\n=== 2) |target_z| monthly (cross-sec stats) ===")
p["abs_z"] = p["target_z"].abs()
m2 = p.groupby("month")["abs_z"].agg(["mean","median",lambda x:x.quantile(0.9)]).rename(columns={"<lambda_0>":"p90"}).round(3)
print(m2.tail(20).to_string())

# === 3) per-cycle: pred_disp vs rstd_mean ===
print("\n=== 3) per-cycle alignment pred_disp vs rstd_mean (H1 vs H2) ===")
preds = pd.read_parquet(PREDS, columns=["symbol","open_time","pred"])
preds["open_time"] = pd.to_datetime(preds["open_time"], utc=True)
preds = preds[(preds["open_time"].dt.hour%4==0)&(preds["open_time"].dt.minute==0)]
# per-cycle aggregates
cyc_pred = preds.groupby("open_time")["pred"].std().rename("pred_disp")
cyc_rstd = p.groupby("open_time")["rstd"].mean().rename("rstd_mean")
cyc_az   = p.groupby("open_time")["abs_z"].mean().rename("abs_z_mean")
cyc = pd.concat([cyc_pred, cyc_rstd, cyc_az], axis=1).dropna()
H1 = (pd.Timestamp("2025-10-04",tz="UTC"), pd.Timestamp("2026-01-22",tz="UTC"))
H2 = (pd.Timestamp("2026-01-22",tz="UTC"), pd.Timestamp("2026-05-26",tz="UTC"))
for label,(s,e) in [("H1",H1),("H2",H2)]:
    sub = cyc[(cyc.index>=s)&(cyc.index<e)]
    print(f"  {label} n={len(sub)}: rstd_mean {sub['rstd_mean'].mean():.5f}  "
          f"abs_z_mean {sub['abs_z_mean'].mean():.3f}  pred_disp {sub['pred_disp'].mean():.3f}  "
          f"corr(pred_disp, rstd_mean) {sub['pred_disp'].corr(sub['rstd_mean']):+.3f}")

# === 4) per-sym rstd inflation: H1 vs H2 ===
print("\n=== 4) per-sym rstd: H1 mean vs H2 mean, top inflation ===")
ph = p[(p["open_time"]>=H1[0])&(p["open_time"]<H1[1])].groupby("symbol")["rstd"].mean().rename("rstd_H1")
ph2= p[(p["open_time"]>=H2[0])&(p["open_time"]<H2[1])].groupby("symbol")["rstd"].mean().rename("rstd_H2")
mm = pd.concat([ph,ph2],axis=1).dropna()
mm["inflation"] = mm["rstd_H2"]/mm["rstd_H1"]
print(f"  syms: {len(mm)}  median inflation H2/H1 = {mm['inflation'].median():.2f}x")
print(f"  pct syms with H2 rstd > H1 rstd: {100*(mm['inflation']>1).mean():.0f}%")
print(f"  pct syms with H2 rstd > 2x H1 rstd: {100*(mm['inflation']>2).mean():.0f}%")
print(f"  TOP 10 most-inflated syms:")
print(mm.sort_values("inflation", ascending=False).head(10).round(4).to_string())

# === 5) counterfactual: recompute rstd with longer windows on a few syms ===
print("\n=== 5) counterfactual rstd with 30d / 60d / 90d windows (3 syms) ===")
HORIZON=48; bars_per_d=288
for win_days in [7, 30, 60, 90]:
    win_bars = bars_per_d * win_days
    for sym in ["BTCUSDT","ETHUSDT","SOLUSDT","DOGEUSDT","SUIUSDT"]:
        sp = p[p["symbol"]==sym].sort_values("open_time")
        if len(sp)<win_bars+HORIZON+10: continue
        ra = sp["alpha_vs_btc_realized"].rolling(win_bars, min_periods=bars_per_d).std().shift(HORIZON)
        rh1 = ra[(sp["open_time"]>=H1[0]) & (sp["open_time"]<H1[1])].mean()
        rh2 = ra[(sp["open_time"]>=H2[0]) & (sp["open_time"]<H2[1])].mean()
        if win_days==7:
            print(f"  {sym:<14} win={win_days:>2}d: rstd H1={rh1:.5f}  H2={rh2:.5f}  inflation={rh2/rh1 if rh1 else float('nan'):.2f}x")
        else:
            print(f"  {sym:<14} win={win_days:>2}d: rstd H1={rh1:.5f}  H2={rh2:.5f}  inflation={rh2/rh1 if rh1 else float('nan'):.2f}x")
    print()
