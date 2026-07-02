"""Does the SIDE-regime mean-reversion signal (model pred) have alpha edge at OTHER horizons in 2024?
Strategy holds ~4h. Build forward cumulative residual-alpha at k*4h horizons from the panel's per-bar
alpha_vs_btc_realized, merge the OOS preds, and measure per-cycle:
  * IC   = Spearman(pred, fwd_alpha_k)  averaged over cycles
  * L-S  = mean fwd_alpha_k of top-3 pred  minus  bottom-3 pred  (the strategy's actual bet), in bps
Split by regime-year (side 2023/2024/2025). If edge is absent at 4h but present at longer h, holding
longer would help; if it's absent/negative at ALL horizons in 2024, the edge was simply gone.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
HORIZONS = [1, 2, 3, 6, 12, 18]   # in 4h bars => 4h,8h,12h,24h,48h,72h

# 1) regime per open_time (from a replay's regime.csv)
reg = pd.read_csv(REPO/"live/state/longtail/regime_only/state/regime.csv")
reg["open_time"] = pd.to_datetime(reg["open_time"], utc=True)
reg = reg[["open_time", "regime", "btc_ret_30d"]].drop_duplicates("open_time")

# 2) panel per-bar residual alpha -> forward cumulative alpha at each horizon
pan = pd.read_parquet(REPO/"outputs/vBTC_features/panel_expanded_v0.parquet",
                      columns=["symbol", "open_time", "alpha_vs_btc_realized"])
pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
pan = pan.sort_values(["symbol", "open_time"])
a = pan.groupby("symbol")["alpha_vs_btc_realized"]
for k in HORIZONS:
    # forward sum of the NEXT k bars' realized residual alpha (hold from open_t to open_{t+k})
    pan[f"fa{k}"] = a.transform(lambda s: s.shift(-1).rolling(k).sum().shift(-(k-1)))

# 3) OOS preds
pr = pd.read_parquet(REPO/"live/state/convexity/hl_lean175_oos/v0full_hl60.parquet",
                     columns=["symbol", "open_time", "pred"])
pr["open_time"] = pd.to_datetime(pr["open_time"], utc=True)
df = pr.merge(pan[["symbol", "open_time"] + [f"fa{k}" for k in HORIZONS]], on=["symbol", "open_time"], how="left")
df = df.merge(reg, on="open_time", how="left")

def analyze(mask, label):
    d = df[mask]
    cyc = d.groupby("open_time")
    print(f"\n=== {label}  (cycles={cyc.ngroups}, sym-rows={len(d)}) ===")
    print(f"  {'horizon':>8}  {'meanIC':>8}  {'IC>0%':>6}  {'gross_LS':>8}  {'g_sh':>7}   {'net_LS':>8}  {'net_sh':>7}  (net = gross - 18bps RT cost)")
    for k in HORIZONS:
        ics, ls = [], []
        for _, g in cyc:
            g2 = g.dropna(subset=["pred", f"fa{k}"])
            if len(g2) < 6: continue
            ics.append(spearmanr(g2["pred"], g2[f"fa{k}"]).correlation)
            top = g2.nlargest(3, "pred")[f"fa{k}"].mean(); bot = g2.nsmallest(3, "pred")[f"fa{k}"].mean()
            ls.append((top - bot) * 1e4)   # long high-pred, short low-pred; bps
        ics = np.array([x for x in ics if np.isfinite(x)]); ls = np.array([x for x in ls if np.isfinite(x)])
        if not len(ics): continue
        # annualized Sharpe of the per-cycle L-S series, scaled by cycles/yr (cycles are 4h => 2190/yr) but held k*4h
        # -> non-overlapping-equiv sharpe ~ mean/std * sqrt(2190/k)
        ann = ls.mean()/ls.std()*np.sqrt(2190.0/k) if ls.std() > 0 else np.nan
        # NET of cost: one round-trip per hold. RT cost ~= 2 legs * 9 bps = 18 bps charged once per k-bar hold.
        RT = 18.0
        net = ls - RT
        net_ann = net.mean()/net.std()*np.sqrt(2190.0/k) if net.std() > 0 else np.nan
        print(f"  {str(k*4)+'h':>8}  {ics.mean():+8.4f}  {100*(ics>0).mean():5.0f}%  {ls.mean():+8.1f}  {ann:+7.2f}   {net.mean():+8.1f}  {net_ann:+7.2f}")

for yr in [2023, 2024, 2025]:
    analyze((df["regime"] == "side") & (df["open_time"].dt.year == yr), f"SIDE {yr}")
analyze((df["regime"] == "side"), "SIDE all-OOS 2023-2025")
analyze((df["regime"] == "bull") & (df["open_time"].dt.year == 2024), "BULL 2024 (ref)")
