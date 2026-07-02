"""Relative short-term DECOUPLING significance: per name, how many sigmas its short-term corr_to_btc_1d has
dropped below its OWN trailing norm (PIT). decouple_z = (corr_now - trailing_mean) / trailing_std, strictly past.
Negative = decoupled below its norm (moving idiosyncratically right now).
Test: does a more-significant decoupling predict STRONGER residual-alpha reversion (the model's edge)?
Bucket by decouple_z -> IC(pred->fwd24 alpha) + L-S. Compare in-sample vs 2024/2023. If relative decoupling
predicts the edge in 2024 (where ABSOLUTE low-corr did not), it's a regime-robuster idiosyncratic-setup signal.
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")
R = "/home/yuqing/ctaNew"; K = 6; W = 540  # trailing ~90d for the name's own corr norm

pan = pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",
                      columns=["symbol", "open_time", "corr_to_btc_1d", "alpha_vs_btc_realized"])
pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True); pan = pan.sort_values(["symbol", "open_time"])
g = pan.groupby("symbol")
pan["fa24"] = g["alpha_vs_btc_realized"].transform(lambda s: s.shift(-1).rolling(K).sum().shift(-(K-1)))
# PIT trailing corr norm (strictly past): shift(1) before rolling
cmean = g["corr_to_btc_1d"].transform(lambda s: s.shift(1).rolling(W, min_periods=60).mean())
cstd = g["corr_to_btc_1d"].transform(lambda s: s.shift(1).rolling(W, min_periods=60).std())
pan["decouple_z"] = (pan["corr_to_btc_1d"] - cmean) / cstd.replace(0, np.nan)   # negative = decoupled below own norm

def run(pred, label, y0=None, y1=None):
    p = pd.read_parquet(pred, columns=["symbol", "open_time", "pred"]); p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    if y0: p = p[(p.open_time >= pd.Timestamp(y0, tz="UTC")) & (p.open_time < pd.Timestamp(y1, tz="UTC"))]
    d = p.merge(pan[["symbol", "open_time", "corr_to_btc_1d", "decouple_z", "fa24"]], on=["symbol", "open_time"])
    d = d.dropna(subset=["pred", "fa24", "decouple_z"])
    d["zb"] = pd.qcut(d["decouple_z"], 5, labels=["Q1_most_decoupled", "Q2", "Q3", "Q4", "Q5_coupled_up"], duplicates="drop")
    print(f"\n=== {label} (rows {len(d)}) — reversion edge by RELATIVE decoupling z ===")
    print(f"  {'z-bucket':>18} {'z_med':>6} {'abs_corr':>8} {'IC':>8} {'|IC|/pooled':>10}")
    for q, gq in d.groupby("zb"):
        ics = []
        for _, gc in gq.groupby("open_time"):
            if len(gc) >= 4: ics.append(spearmanr(gc["pred"], gc["fa24"]).correlation)
        ics = np.array([x for x in ics if np.isfinite(x)])
        print(f"  {str(q):>18} {gq['decouple_z'].median():+6.2f} {gq['corr_to_btc_1d'].median():8.3f} {np.nanmean(ics):+8.4f}")

run(f"{R}/live/state/convexity/hl_lean175/v0full_hl60.parquet", "IN-SAMPLE 2025-10+")
run(f"{R}/live/state/convexity/hl_lean175_oos/v0full_hl60.parquet", "OOS 2024", "2024-01-01", "2025-01-01")
run(f"{R}/live/state/convexity/hl_lean175_oos/v0full_hl60.parquet", "OOS 2023", "2023-01-01", "2024-01-01")
