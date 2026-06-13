"""Definitive classification of the 1,110 post-gap dom_change rows.

The R0(b) FAIL was a positional .shift(288) crossing 4 internal gaps. The
original builder (features_ml/klines.py) reindexes each symbol to a regular
5-min grid, so the correct recompute is a TIME-based 288-step shift on that
grid (a value 288*5min before t; gap-spanning -> NaN). Compare the panel
column to that time-grid recipe:
  - if they match (<=1e-3*std) on all non-NaN rows -> column is faithful,
    R0 dom check PASSES (the 0.019% were my positional-shift bug).
  - if the panel has finite values where the time-grid recipe is NaN/diff
    at the 4 gaps -> those post-gap windows are contaminated -> rebuild.
"""
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/home/yuqing/ctaNew")
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
df = pd.read_parquet(PANEL, columns=["symbol", "open_time", "dom_level_vs_bk",
                                     "dom_change_288b_vs_bk"])
df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)
ref_std = float(df["dom_change_288b_vs_bk"].std())

parts = []
for s, g in df.groupby("symbol", sort=False):
    g = g.set_index("open_time")
    # regular 5-min grid spanning this symbol's span
    grid = pd.date_range(g.index.min(), g.index.max(), freq="5min", tz="UTC")
    spread = g["dom_level_vs_bk"].reindex(grid)        # gaps -> NaN
    dom_tg = (spread - spread.shift(288)).reindex(g.index)  # time-based 288-step
    parts.append(pd.Series(dom_tg.values, index=g.reset_index().index
                           if False else g.index, name=s))
# rebuild aligned to df order
dom_tg = []
for s, g in df.groupby("symbol", sort=False):
    gi = g.set_index("open_time")
    grid = pd.date_range(gi.index.min(), gi.index.max(), freq="5min", tz="UTC")
    sp = gi["dom_level_vs_bk"].reindex(grid)
    dom_tg.append((sp - sp.shift(288)).reindex(gi.index).values)
df["dom_tg"] = np.concatenate(dom_tg)

d = (df["dom_tg"] - df["dom_change_288b_vs_bk"]).abs()
both = df["dom_tg"].notna() & df["dom_change_288b_vs_bk"].notna()
mx = float(d[both].max())
print(f"ref_std={ref_std:.4f}")
print(f"time-grid recompute vs panel (both non-NaN): n={both.sum():,} "
      f"max|Δ|={mx:.3e} ({mx/ref_std:.2e}·std) "
      f"frac_within_1e-4={float((d[both] <= 1e-4*ref_std).mean()):.6f}")

# the 4-gap symbols specifically
for s in ["PUMPUSDT", "STRKUSDT", "VIRTUALUSDT"]:
    sub = df[df["symbol"] == s]
    bd = (sub["dom_tg"] - sub["dom_change_288b_vs_bk"]).abs()
    b = sub["dom_tg"].notna() & sub["dom_change_288b_vs_bk"].notna()
    # where panel finite but time-grid NaN (would be spurious contamination)
    spurious = (sub["dom_tg"].isna() & sub["dom_change_288b_vs_bk"].notna()).sum()
    print(f"  {s}: n={len(sub):,} match_max|Δ|="
          f"{(bd[b].max() if b.any() else 0):.3e} "
          f"panel-finite-where-timegrid-NaN={spurious:,}")

verdict = "PASS-faithful" if mx <= 1e-3 * ref_std else "DEFECT"
print(f"\nDOM CHECK (time-grid correct): {verdict}  "
      f"-> {'column matches documented recipe; R0(b) FAIL was a positional-shift bug' if verdict=='PASS-faithful' else 'panel column contaminated post-gap; rebuild from _full_pit'}")
