"""Per-feature cross-sectional dispersion H1 vs H2. Identifies which V0 features
have lost their cross-sectional spread — the likely operative cause of pred_disp
collapse (since pred = sum(coef * feature)).
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
REPO = Path("/home/yuqing/ctaNew")
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"

FEATS = ["return_1d","atr_pct","obv_z_1d","vwap_slope_96",
         "bars_since_high","bars_since_high_xs_rank","autocorr_pctile_7d",
         "corr_to_btc_1d","beta_to_btc_change_5d",
         "idio_vol_to_btc_1h","idio_vol_to_btc_1d",
         "funding_rate","funding_rate_z_7d","funding_rate_1d_change",
         "rvol_7d","ret_3d","btc_rvol_7d"]

p = pd.read_parquet(PANEL, columns=["symbol","open_time"]+FEATS)
p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
p = p[(p["open_time"].dt.hour%4==0)&(p["open_time"].dt.minute==0)]

H1 = (pd.Timestamp("2025-10-04",tz="UTC"), pd.Timestamp("2026-01-22",tz="UTC"))
H2 = (pd.Timestamp("2026-01-22",tz="UTC"), pd.Timestamp("2026-05-26",tz="UTC"))

h1 = p[(p["open_time"]>=H1[0])&(p["open_time"]<H1[1])]
h2 = p[(p["open_time"]>=H2[0])&(p["open_time"]<H2[1])]

# per-feature cross-sectional std (across syms) at each cycle, then averaged across cycles
print("=== Per-feature cross-sectional std (mean across cycles) H1 vs H2 ===\n")
print(f"{'feature':<28} {'H1 xs_std':>12} {'H2 xs_std':>12} {'ratio H2/H1':>13} {'H1 xs_iqr':>12} {'H2 xs_iqr':>12}")
print("-" * 96)
rows=[]
for f in FEATS:
    h1_xs = h1.groupby("open_time")[f].agg(['std', lambda x: x.quantile(0.75)-x.quantile(0.25)]).mean()
    h2_xs = h2.groupby("open_time")[f].agg(['std', lambda x: x.quantile(0.75)-x.quantile(0.25)]).mean()
    ratio = h2_xs.iloc[0]/h1_xs.iloc[0] if h1_xs.iloc[0] else float('nan')
    rows.append((f, h1_xs.iloc[0], h2_xs.iloc[0], ratio, h1_xs.iloc[1], h2_xs.iloc[1]))
df = pd.DataFrame(rows, columns=['feature','h1_std','h2_std','ratio','h1_iqr','h2_iqr']).sort_values('ratio')
for _, r in df.iterrows():
    print(f"{r['feature']:<28} {r['h1_std']:>12.4f} {r['h2_std']:>12.4f} {r['ratio']:>13.3f} {r['h1_iqr']:>12.4f} {r['h2_iqr']:>12.4f}")

print("\nFeatures with biggest XS-dispersion COMPRESSION (lowest ratio):")
print(df.head(5)[['feature','ratio']].to_string(index=False))
print("\nFeatures with biggest XS-dispersion EXPANSION (highest ratio):")
print(df.tail(5)[['feature','ratio']].to_string(index=False))
