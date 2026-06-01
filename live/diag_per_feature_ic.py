"""Per-feature univariate IC H1 vs H2.

For each cycle, compute the cross-sectional rank correlation between each feature value
and the forward residual (alpha_A). Average across cycles per period. If specific
features had IC in H1 but lost it (or flipped sign) in H2, those features are the
operative cause of long-side breakdown.

Note: this measures UNIVARIATE IC (feature → forward residual directly). The model's
actual prediction is a linear combination of all features, so a feature losing
univariate IC doesn't necessarily kill the model — but if MOST features lose IC, the
combined prediction will too. And if any feature has *flipped sign*, the model's
learned coefficient is now actively wrong.
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from scipy.stats import spearmanr
REPO = Path("/home/yuqing/ctaNew")
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"

FEATS = ["return_1d","atr_pct","obv_z_1d","vwap_slope_96",
         "bars_since_high","bars_since_high_xs_rank","autocorr_pctile_7d",
         "corr_to_btc_1d","beta_to_btc_change_5d",
         "idio_vol_to_btc_1h","idio_vol_to_btc_1d",
         "funding_rate","funding_rate_z_7d","funding_rate_1d_change",
         "rvol_7d","ret_3d","btc_rvol_7d"]

print("loading panel...")
p = pd.read_parquet(PANEL, columns=["symbol","open_time","alpha_vs_btc_realized"]+FEATS)
p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
p = p[(p["open_time"].dt.hour%4==0)&(p["open_time"].dt.minute==0)]
p = p.rename(columns={"alpha_vs_btc_realized":"fwd_alpha"})
print(f"  {len(p):,} rows × {p['symbol'].nunique()} syms")

H1 = (pd.Timestamp("2025-10-04",tz="UTC"), pd.Timestamp("2026-01-22",tz="UTC"))
H2 = (pd.Timestamp("2026-01-22",tz="UTC"), pd.Timestamp("2026-05-26",tz="UTC"))

def feature_xs_ic_per_period(df, feats):
    rows = []
    for f in feats:
        sub = df[["open_time","fwd_alpha",f]].dropna()
        # per-cycle Spearman corr (across syms) between feature value and forward alpha
        ics = []
        for ot, g in sub.groupby("open_time"):
            if len(g) < 10: continue
            r,_ = spearmanr(g[f], g["fwd_alpha"])
            if np.isfinite(r): ics.append(r)
        ic_mean = np.mean(ics) if ics else np.nan
        ic_pct_pos = 100*(np.array(ics)>0).mean() if ics else np.nan
        rows.append((f, ic_mean, ic_pct_pos, len(ics)))
    return pd.DataFrame(rows, columns=['feature','ic_xs_mean','pct_cycles_pos','n_cycles'])

for label,(s,e) in [("H1",H1),("H2",H2)]:
    sub = p[(p["open_time"]>=s)&(p["open_time"]<e)]
    print(f"\n=== {label} (n_cycles={sub['open_time'].nunique()}) per-feature univariate XS IC ===")
    df = feature_xs_ic_per_period(sub, FEATS).sort_values('ic_xs_mean')
    print(df.round(4).to_string(index=False))

# Side-by-side comparison
print(f"\n=== H1 vs H2 IC comparison (sorted by |Δ IC| descending) ===")
print(f"{'feature':<28} {'H1 IC':>10} {'H2 IC':>10} {'Δ (H2-H1)':>12} {'sign flip?':>12}")
print("-" * 76)
sub1 = p[(p["open_time"]>=H1[0])&(p["open_time"]<H1[1])]
sub2 = p[(p["open_time"]>=H2[0])&(p["open_time"]<H2[1])]
df1 = feature_xs_ic_per_period(sub1, FEATS).set_index('feature')
df2 = feature_xs_ic_per_period(sub2, FEATS).set_index('feature')
cmp = pd.DataFrame({'h1':df1['ic_xs_mean'], 'h2':df2['ic_xs_mean']})
cmp['delta'] = cmp['h2'] - cmp['h1']
cmp['abs_delta'] = cmp['delta'].abs()
cmp = cmp.sort_values('abs_delta', ascending=False)
for f,row in cmp.iterrows():
    flip = "YES" if (np.sign(row['h1'])!=np.sign(row['h2']) and abs(row['h1'])>0.005 and abs(row['h2'])>0.005) else ""
    print(f"{f:<28} {row['h1']:+10.4f} {row['h2']:+10.4f} {row['delta']:+12.4f} {flip:>12}")

print(f"\n=== Summary ===")
print(f"  H1 features with |IC| > 0.01: {(cmp['h1'].abs()>0.01).sum()}/17")
print(f"  H2 features with |IC| > 0.01: {(cmp['h2'].abs()>0.01).sum()}/17")
print(f"  Sign flips between H1 and H2 (both |IC|>0.005): {((np.sign(cmp['h1'])!=np.sign(cmp['h2'])) & (cmp['h1'].abs()>0.005) & (cmp['h2'].abs()>0.005)).sum()}")
print(f"  Mean |IC| H1: {cmp['h1'].abs().mean():.4f}")
print(f"  Mean |IC| H2: {cmp['h2'].abs().mean():.4f}")
