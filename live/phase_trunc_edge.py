"""RAW, GATE-FREE predictor edge (long/short) — no REGIME_GATE, no BULL_DEEP_THR, no SHORT_MIN_RET3D,
no conc_cap, no sizing, no sleeves. Just: rank by pred each cycle -> top-K long / bottom-K short ->
RAW forward residual alpha per leg. Compare FULL universe vs TRUNCATED (exclude top-80 high-vol, monthly).
Answers: what is the pure long/short alpha edge, and how does truncation change each leg?
Sign: long PnL ~ +long_edge; short PnL ~ -short_edge; L/S captured = long_edge - short_edge (bps).
Usage: python3 live/phase_trunc_edge.py
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")
R = "/home/yuqing/ctaNew"; K = 3; H = 6   # top/bottom-3, 24h fwd

pan = pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet", columns=["symbol", "open_time", "alpha_vs_btc_realized"])
pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True); pan = pan.sort_values(["symbol", "open_time"])
pan["fa"] = pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s: s.shift(-1).rolling(H).sum().shift(-(H-1)))

def edges(pred_path, allow_path, label, y0=None, y1=None):
    p = pd.read_parquet(pred_path, columns=["symbol", "open_time", "pred"]); p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    if y0: p = p[(p.open_time >= pd.Timestamp(y0, tz="UTC")) & (p.open_time < pd.Timestamp(y1, tz="UTC"))]
    d = p.merge(pan[["symbol", "open_time", "fa"]], on=["symbol", "open_time"]).dropna(subset=["pred", "fa"])
    allow = None
    if allow_path:
        a = pd.read_parquet(allow_path); a["open_time"] = pd.to_datetime(a["open_time"], utc=True)
        allow = a.groupby("open_time")["symbol"].apply(set).to_dict()
    lo, sh, ls, ics, ncyc = [], [], [], [], 0
    for ot, g in d.groupby("open_time"):
        if allow is not None:
            al = allow.get(ot)
            if al is not None: g = g[g["symbol"].isin(al)]
        if len(g) < 2 * K: continue
        ncyc += 1
        longs = g.nlargest(K, "pred")["fa"].mean(); shorts = g.nsmallest(K, "pred")["fa"].mean()
        lo.append(longs * 1e4); sh.append(shorts * 1e4); ls.append((longs - shorts) * 1e4)
        ics.append(spearmanr(g["pred"], g["fa"]).correlation)
    lo, sh, ls = map(lambda x: np.array([v for v in x if np.isfinite(v)]), (lo, sh, ls))
    ic = np.nanmean([x for x in ics if np.isfinite(x)])
    print(f"  {label:34s} n={ncyc:5d} | LONG_edge {np.mean(lo):+6.1f}  SHORT_edge {np.mean(sh):+6.1f} "
          f"(short PnL {-np.mean(sh):+6.1f}) | L-S {np.mean(ls):+6.1f} bps | IC {ic:+.4f}")

for lbl, y0, y1, ap_full, ap_tr, pred in [
    ("IN-SAMPLE 2025-10+", None, None, None, f"{R}/live/state/longtail/volexcl_allow.parquet", f"{R}/live/state/convexity/hl_lean175/v0full_hl60.parquet"),
    ("OOS 2024",           "2024-01-01", "2025-01-01", None, f"{R}/live/state/longtail/volexcl_allow_oos.parquet", f"{R}/live/state/convexity/hl_lean175_oos/v0full_hl60.parquet"),
    ("OOS 2023",           "2023-01-01", "2024-01-01", None, f"{R}/live/state/longtail/volexcl_allow_oos.parquet", f"{R}/live/state/convexity/hl_lean175_oos/v0full_hl60.parquet"),
]:
    print(f"\n=== {lbl} (gate-free top{K}/bot{K}, 24h raw residual alpha) ===")
    edges(pred, None, "FULL universe", y0, y1)
    edges(pred, ap_tr, "TRUNCATED (excl top-80 high-vol)", y0, y1)
