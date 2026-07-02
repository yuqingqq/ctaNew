"""Alpha-residual TERM STRUCTURE: the model pred is trained on 4h-fwd residual, but we HOLD 24h.
Where does the predictive residual-alpha actually live across forward horizons?
  * MARGINAL IC[j]   = xs corr( pred_t , residual_alpha of forward bar j )   (single 4h bar, j=1..18)
                       -> shows the term structure: which future bars the signal predicts.
  * CUMULATIVE IC[k] = xs corr( pred_t , sum of residual alpha over next k bars )  (what a k-bar hold sees)
  * CUM L-S[k]       = top3-bot3 by pred, realized cum residual alpha over k bars (bps) — capture per hold length.
Compares in-sample 2025-10+ vs OOS 2024. If marginal IC is +for bars 1..6 then ~0/neg after, 24h(=6 bars) is right;
if it stays + past 6, we under-hold; if it decays by bar 2-3, we over-hold.
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")
R = "/home/yuqing/ctaNew"; JMAX = 18  # up to 72h

pan = pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",
                      columns=["symbol", "open_time", "alpha_vs_btc_realized"])
pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True); pan = pan.sort_values(["symbol", "open_time"])
g = pan.groupby("symbol")["alpha_vs_btc_realized"]
# marginal forward bar j (j=1 = bar [t,t+1] = row t): shift(-(j-1))
for j in range(1, JMAX + 1):
    pan[f"m{j}"] = g.shift(-(j - 1))
# cumulative over next k bars
for k in [1, 3, 6, 12, 18]:
    pan[f"c{k}"] = g.transform(lambda s: s.shift(-1).rolling(k).sum().shift(-(k - 1))) + pan["alpha_vs_btc_realized"] * 0  # placeholder
# simpler cumulative: sum of m1..mk
for k in [1, 3, 6, 9, 12, 18]:
    pan[f"cum{k}"] = pan[[f"m{j}" for j in range(1, k + 1)]].sum(axis=1, min_count=k)

def run(pred_path, label, y0=None, y1=None):
    pr = pd.read_parquet(pred_path, columns=["symbol", "open_time", "pred"]); pr["open_time"] = pd.to_datetime(pr["open_time"], utc=True)
    if y0: pr = pr[(pr.open_time >= pd.Timestamp(y0, tz="UTC")) & (pr.open_time < pd.Timestamp(y1, tz="UTC"))]
    df = pr.merge(pan, on=["symbol", "open_time"], how="left")
    print(f"\n=== {label} (cycles={df['open_time'].nunique()}) ===")
    # marginal IC term structure
    print("  MARGINAL IC by forward bar (each = one 4h bar ahead):")
    mline = "   "
    for j in range(1, JMAX + 1):
        ics = []
        for _, gc in df.groupby("open_time"):
            gg = gc.dropna(subset=["pred", f"m{j}"])
            if len(gg) >= 6: ics.append(spearmanr(gg["pred"], gg[f"m{j}"]).correlation)
        ics = np.array([x for x in ics if np.isfinite(x)])
        if j in [1, 2, 3, 4, 6, 9, 12, 18]:
            mline += f" b{j}({j*4}h):{ics.mean():+.4f}"
    print(mline)
    # cumulative L-S by hold length
    print("  CUMULATIVE L-S (top3-bot3 by pred, realized cum residual bps) + IC:")
    for k in [1, 3, 6, 9, 12, 18]:
        ls, ics = [], []
        for _, gc in df.groupby("open_time"):
            gg = gc.dropna(subset=["pred", f"cum{k}"])
            if len(gg) < 6: continue
            ls.append((gg.nlargest(3, "pred")[f"cum{k}"].mean() - gg.nsmallest(3, "pred")[f"cum{k}"].mean()) * 1e4)
            ics.append(spearmanr(gg["pred"], gg[f"cum{k}"]).correlation)
        ls = np.array([x for x in ls if np.isfinite(x)]); ics = np.array([x for x in ics if np.isfinite(x)])
        tag = "  <-- HOLD" if k == 6 else ""
        print(f"     hold {k*4:3d}h ({k:2d}b): cum L-S {ls.mean():+7.1f} bps | IC {ics.mean():+.4f} | per-bar {ls.mean()/k:+6.1f}{tag}")

run(f"{R}/live/state/convexity/hl_lean175/v0full_hl60.parquet", "IN-SAMPLE 2025-10+")
run(f"{R}/live/state/convexity/hl_lean175_oos/v0full_hl60.parquet", "OOS 2024", "2024-01-01", "2025-01-01")
run(f"{R}/live/state/convexity/hl_lean175_oos/v0full_hl60.parquet", "OOS 2023", "2023-01-01", "2024-01-01")
