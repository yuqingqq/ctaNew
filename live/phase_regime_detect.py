"""Can we DETECT the trend-vs-revert regime PIT and does it predict the forward reversion edge?
Two candidate PIT detectors (known at decision time t):
  A) xs_corr   = per-cycle mean corr_to_btc_1d (alt-BTC co-movement; high=beta/trend regime)  [already trailing]
  B) tr_ac1    = market-level trailing lag-1 autocorr of residual alpha over trailing 30d (PIT, strictly past)
Test: bucket SIDE cycles (all OOS+IS) by the detector; measure forward 24h long-leg alpha, short-leg, L-S.
If high-corr / positive-ac1 cycles have a DEAD/neg long leg and low-corr/neg-ac1 have a live one, the detector
is usable to gate the long leg. Reports the spread and an IC of detector->forward-edge.
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")
R = "/home/yuqing/ctaNew"; K = 6; WIN = 180  # trailing 30d for ac1

reg = pd.read_csv(f"{R}/live/state/longtail/full_regime/state/regime.csv")
reg["open_time"] = pd.to_datetime(reg["open_time"], utc=True)
reg = reg[["open_time", "regime"]].drop_duplicates("open_time")
pan = pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",
                      columns=["symbol", "open_time", "alpha_vs_btc_realized", "return_1d", "corr_to_btc_1d"])
pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True); pan = pan.sort_values(["symbol", "open_time"])
g = pan.groupby("symbol")
pan["fa24"] = g["alpha_vs_btc_realized"].transform(lambda s: s.shift(-1).rolling(K).sum().shift(-(K-1)))
# PIT trailing lag-1 autocorr per symbol: corr over trailing WIN bars of (alpha_{t-1}, alpha_t), strictly past
def tr_ac1(s):
    prev = s.shift(1)
    # rolling corr of s vs prev over WIN, shifted by 1 so it uses only past
    return s.rolling(WIN).corr(prev).shift(1)
pan["tr_ac1"] = g["alpha_vs_btc_realized"].transform(tr_ac1)
pan = pan.merge(reg, on="open_time", how="left")
d = pan[pan.regime == "side"].copy()

# per-cycle: detector values (PIT) + forward legs
rows = []
for t, gc in d.groupby("open_time"):
    g2 = gc.dropna(subset=["fa24"])
    if len(g2) < 6: continue
    xs_mean = g2["fa24"].mean()
    # long leg = top-3 by -return_1d proxy? No: use the actual model direction proxy = reversal signal -return_1d
    # (the long leg buys recent losers). long-leg alpha vs xs-mean:
    lo = g2.nlargest(3, [c for c in ["return_1d"] if c in g2][0])  # placeholder to keep shape
    rows.append(dict(open_time=t,
                     xs_corr=gc["corr_to_btc_1d"].mean(),
                     tr_ac1=gc["tr_ac1"].mean(),
                     # long leg = buy the 3 biggest recent losers (-return_1d) -> their fwd alpha minus xs-mean
                     long_leg=(g2.nsmallest(3, "return_1d")["fa24"].mean() - xs_mean) * 1e4,
                     short_leg=(xs_mean - g2.nlargest(3, "return_1d")["fa24"].mean()) * 1e4,
                     ls=(g2.nsmallest(3, "return_1d")["fa24"].mean() - g2.nlargest(3, "return_1d")["fa24"].mean()) * 1e4))
c = pd.DataFrame(rows).dropna(subset=["xs_corr"])
print(f"side cycles with detector: {len(c)} (tr_ac1 available: {c['tr_ac1'].notna().sum()})\n")

def buckets(det, name):
    cc = c.dropna(subset=[det]).copy()
    cc["q"] = pd.qcut(cc[det], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
    print(f"=== detector {name} -> forward 24h reversion legs (bps), by quartile ===")
    print(f"  {'quartile':>9} {'det_mean':>9} {'long_leg':>9} {'short_leg':>9} {'L-S':>8} {'n':>5}")
    for q, gq in cc.groupby("q"):
        print(f"  {str(q):>9} {gq[det].mean():+9.3f} {gq['long_leg'].mean():+9.1f} {gq['short_leg'].mean():+9.1f} {gq['ls'].mean():+8.1f} {len(gq):5d}")
    ic_long = spearmanr(cc[det], cc["long_leg"]).correlation
    ic_ls = spearmanr(cc[det], cc["ls"]).correlation
    print(f"  IC(detector -> long_leg) {ic_long:+.4f} | IC(detector -> L-S) {ic_ls:+.4f}  (want NEG: high detector=worse)\n")

buckets("xs_corr", "xs_corr (alt-BTC co-movement)")
buckets("tr_ac1", "tr_ac1 (trailing residual autocorr)")
