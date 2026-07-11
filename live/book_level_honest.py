"""Honest performance measured at BOOK LEVEL (path-INDEPENDENT), per CLAUDE.md pitfall #4: full-stack replay
through the DD-stop + regime overlays amplifies tiny prediction differences ~10-20x, so it is unreliable for
comparing near-identical books (the honest _honest books vs old _cleanfix books have 0.996-corr preds yet the
full-stack Sharpe swings +2.09<->+2.45). The robust honest number is per-cycle rank-IC + the 1L/2S selection
spread with NO path-coupled overlays. Books = the audit-honest gap-clean-panel retrain (_honest suffix).
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
S = "live/state/convexity"

def metrics(base_p, long_p, label):
    b = pd.read_parquet(base_p); l = pd.read_parquet(long_p)
    b["open_time"] = pd.to_datetime(b["open_time"], utc=True); l["open_time"] = pd.to_datetime(l["open_time"], utc=True)
    lg = l.groupby("open_time"); rows = []; ics = []
    for t, g in b.groupby("open_time"):
        if len(g) < 5: continue
        ics.append(spearmanr(g["pred"], g["alpha_A"]).correlation)
        try: gl = lg.get_group(t)
        except KeyError: continue
        L = gl.nlargest(1, "pred"); Sh = g.nsmallest(2, "pred")
        if len(L) < 1 or len(Sh) < 2: continue
        la = float(L.iloc[0]["alpha_A"] * 1e4); sa = float(Sh["alpha_A"].mean() * 1e4)
        rows.append((t, 0.5 * la - 0.5 * sa))   # 1L/2S selection spread (residual bps), path-independent
    d = pd.DataFrame(rows, columns=["t", "net"]); d["t"] = pd.to_datetime(d["t"], utc=True)
    dd = d.groupby(d["t"].dt.date)["net"].sum()
    sh = dd.mean() / dd.std() * np.sqrt(365) if dd.std() > 0 else np.nan
    print(f"{label}: rank-IC {np.nanmean(ics):+.4f} | 1L2S selection-spread DAILY Sharpe {sh:+.2f} (path-independent) | n {len(d)}")

if __name__ == "__main__":
    metrics(f"{S}/hl_tgt_res_base_honest/v0full_hl60.parquet", f"{S}/hl_tgt_res_long_honest/v0full_hl60.parquet", "RECENT honest")
    metrics(f"{S}/hl_v4base_oos_honest/v0full_hl60.parquet",  f"{S}/hl_v4long_oos_honest/v0full_hl60.parquet",  "OOS honest   ")
