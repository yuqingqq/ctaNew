"""Diagnose the 0.019% dom_change_288b_vs_bk mismatch: benign time-grid /
listing-edge artifact in the positional recompute, or a real panel defect?"""
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/home/yuqing/ctaNew")
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
df = pd.read_parquet(PANEL, columns=["symbol", "open_time", "dom_level_vs_bk",
                                     "dom_change_288b_vs_bk"])
df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)

ref_std = float(df["dom_change_288b_vs_bk"].std())
rc = df.groupby("symbol", sort=False)["dom_level_vs_bk"].transform(lambda s: s - s.shift(288))
d = (rc - df["dom_change_288b_vs_bk"]).abs()
bad = df[d > 1e-3 * ref_std].copy()
bad["delta"] = d[d > 1e-3 * ref_std]
print(f"ref_std={ref_std:.4f}  n_bad={len(bad):,} ({len(bad)/len(df)*100:.4f}%)")

# per-symbol bar spacing: is the panel a continuous 5-min grid per symbol?
df["dt"] = df.groupby("symbol")["open_time"].diff()
gap = df[df["dt"] > pd.Timedelta("5min")]
print(f"\nrows where intra-symbol gap > 5min: {len(gap):,}")
print("gap size distribution (top):")
print(gap["dt"].value_counts().head(8).to_string())

# how many bad rows are within the first 288*2 bars of a symbol (listing edge)?
df["rk"] = df.groupby("symbol").cumcount()
bad2 = df.loc[bad.index]
print(f"\nbad rows with symbol-rank < 576 (listing edge): "
      f"{(bad2['rk'] < 576).sum():,} / {len(bad2):,}")
# how many bad rows are near an internal gap (within 288 bars after a gap)?
gap_syms = {}
for s, g in df.groupby("symbol"):
    gi = g.index[g["dt"] > pd.Timedelta("5min")].tolist()
    gap_syms[s] = gi
near_gap = 0
for idx, r in bad2.iterrows():
    gi = gap_syms.get(r["symbol"], [])
    if any(0 <= (idx - gx) < 288 for gx in gi):
        near_gap += 1
print(f"bad rows within 288 bars after an internal gap: {near_gap:,} / {len(bad2):,}")
print(f"bad rows explained by (listing-edge OR post-gap): "
      f"{((bad2['rk'] < 576) | pd.Series([any(0<=(idx-gx)<288 for gx in gap_syms.get(r['symbol'],[])) for idx,r in bad2.iterrows()], index=bad2.index)).sum():,} / {len(bad2):,}")
print("\nbad rows by symbol (top 10):")
print(bad["symbol"].value_counts().head(10).to_string())
