"""Why was the SIDE mean-rev edge smaller in 2024? Decompose into 3 hypotheses at the 24h horizon:
  (1) SCALE     — less opportunity: cross-sectional dispersion of fwd alpha; ORACLE L-S (top3-bot3 by REALIZED fa)
  (2) SELECTION — wrong symbols: model L-S / oracle L-S (capture %), and IC
  (3) TAILS     — more adverse tails: skew/kurtosis/CVaR5/worst of the per-cycle model L-S, split long vs short leg
Compares in-sample 2025-10+ vs OOS years 2023/2024/2025-H1. Long high-pred, short low-pred; fwd alpha over 6 bars=24h.
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr, skew, kurtosis
import warnings; warnings.filterwarnings("ignore")
R = "/home/yuqing/ctaNew"; K = 6  # 24h

pan = pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",
                      columns=["symbol", "open_time", "alpha_vs_btc_realized"])
pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True); pan = pan.sort_values(["symbol", "open_time"])
pan["fa"] = pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s: s.shift(-1).rolling(K).sum().shift(-(K-1)))

def load(pred, reg):
    r = pd.read_csv(reg); r["open_time"] = pd.to_datetime(r["open_time"], utc=True)
    r = r[["open_time", "regime"]].drop_duplicates("open_time")
    p = pd.read_parquet(pred, columns=["symbol", "open_time", "pred"]); p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    return p.merge(pan[["symbol", "open_time", "fa"]], on=["symbol", "open_time"]).merge(r, on="open_time")

IS = load(f"{R}/live/state/convexity/hl_lean175/v0full_hl60.parquet", f"{R}/live/state/longtail/is_regime/state/regime.csv")
OOS = load(f"{R}/live/state/convexity/hl_lean175_oos/v0full_hl60.parquet", f"{R}/live/state/longtail/oos_regime/state/regime.csv")

def decomp(df, label):
    d = df[df.regime == "side"]
    disp, orc, mod, ml_long, ml_short, ics = [], [], [], [], [], []
    for _, g in d.groupby("open_time"):
        g2 = g.dropna(subset=["pred", "fa"])
        if len(g2) < 6: continue
        disp.append(g2["fa"].std() * 1e4)
        orc.append((g2.nlargest(3, "fa")["fa"].mean() - g2.nsmallest(3, "fa")["fa"].mean()) * 1e4)
        xs_mean = g2["fa"].mean()
        longp = g2.nlargest(3, "pred")["fa"].mean(); shortp = g2.nsmallest(3, "pred")["fa"].mean()
        mod.append((longp - shortp) * 1e4)
        ml_long.append((longp - xs_mean) * 1e4)          # long-leg alpha vs xs-mean
        ml_short.append((xs_mean - shortp) * 1e4)        # short-leg alpha (short => profit when name underperforms)
        ics.append(spearmanr(g2["pred"], g2["fa"]).correlation)
    disp, orc, mod = map(lambda x: np.array([v for v in x if np.isfinite(v)]), (disp, orc, mod))
    ml_long = np.array([v for v in ml_long if np.isfinite(v)]); ml_short = np.array([v for v in ml_short if np.isfinite(v)])
    ics = np.array([v for v in ics if np.isfinite(v)])
    cvar5 = np.mean(np.sort(mod)[:max(1, len(mod)//20)])
    print(f"\n=== {label}  (side cycles={len(mod)}) — 24h ===")
    print(f"  SCALE      xs-dispersion {disp.mean():6.0f} bps | ORACLE L-S {orc.mean():6.0f} bps")
    print(f"  SELECTION  model L-S {mod.mean():+6.1f} | capture {100*mod.mean()/orc.mean():4.0f}% of oracle | IC {ics.mean():+.4f}")
    print(f"  TAILS      L-S std {mod.std():5.0f}  skew {skew(mod):+.2f}  kurt {kurtosis(mod):+.1f}  CVaR5 {cvar5:+6.0f}  worst {mod.min():+6.0f}")
    print(f"  LEGS       long-leg {ml_long.mean():+5.1f} (skew {skew(ml_long):+.2f}) | short-leg {ml_short.mean():+5.1f} (skew {skew(ml_short):+.2f})")

decomp(IS, "IN-SAMPLE 2025-10+")
for yr in [2023, 2024, 2025]:
    decomp(OOS[OOS.open_time.dt.year == yr], f"OOS {yr}")
