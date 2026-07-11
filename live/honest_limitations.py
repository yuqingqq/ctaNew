"""Honest limitation re-measurement on the AUDITED-honest artifacts (gap-clean panel, honest gate, grid-safe
regime, _honest walk-forward books). Everything at BOOK LEVEL (path-INDEPENDENT: 1L/2S selection spread + rank-IC,
NO DD-stop/regime overlays) per pitfall #4 — the old doc's per-regime FULL-STACK Sharpes are path-coupled/unreliable.
Per era x regime + the long/short leg split. Feeds the rewritten V4_LIMITATIONS_DIAGNOSIS.md.
"""
import pandas as pd, numpy as np, sys
sys.path.insert(0, "live")
from attribution_v4_regime import btc_reg   # grid-safe (audit #5 fixed)
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")
S = "live/state/convexity"

def load(bd, ld):
    b = pd.read_parquet(f"{S}/{bd}/v0full_hl60.parquet"); l = pd.read_parquet(f"{S}/{ld}/v0full_hl60.parquet")
    for d in (b, l): d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    return b, l

def dsh(x, col):
    dd = x.groupby(x["t"].dt.date)[col].sum()
    return dd.mean() / dd.std() * np.sqrt(365) if dd.std() > 0 else np.nan

def measure(base, long, reg, era):
    lg = long.groupby("open_time"); rows = []; Lp = []; Sp = []
    for t, g in base.groupby("open_time"):
        rg = reg.get(t)
        if rg is None or len(g) < 5: continue
        ic = spearmanr(g["pred"], g["alpha_A"]).correlation
        try: gl = lg.get_group(t)
        except KeyError: continue
        L = gl.nlargest(1, "pred"); Sh = g.nsmallest(2, "pred")
        if len(L) < 1 or len(Sh) < 2: continue
        la = float(L.iloc[0]["alpha_A"] * 1e4); sl = [float(-r["alpha_A"] * 1e4) for _, r in Sh.iterrows()]
        if np.isfinite(la): Lp.append(la)
        Sp += [x for x in sl if np.isfinite(x)]
        # sl = short PnL (=-alpha). book net = long PnL + short PnL (ADD; earlier version subtracted = bug)
        rows.append((t, rg, 0.5 * la + 0.5 * np.nanmean(sl), 0.5 * la, 0.5 * np.nanmean(sl), ic))
    d = pd.DataFrame(rows, columns=["t", "rg", "net", "long_c", "short_c", "ic"])
    print(f"\n===== {era}: per-regime BOOK-LEVEL 1L/2S selection spread (path-independent) =====")
    for rg in ["side", "bear", "bull", "deepbull"]:
        x = d[d.rg == rg]
        if len(x) < 20: print(f"  {rg:9s}: n={len(x):5d} (thin)"); continue
        print(f"  {rg:9s}: n={len(x):5d} | sel-spread Daily Sharpe {dsh(x,'net'):+.2f} | rank-IC {x['ic'].mean():+.3f} | net {x['net'].sum():+7.0f} bps")
    print(f"  {'ALL':9s}: n={len(d):5d} | sel-spread Daily Sharpe {dsh(d,'net'):+.2f} | rank-IC {d['ic'].mean():+.3f}")
    # long vs short legs (path-independent)
    Lp = np.array(Lp); Sp = np.array(Sp)
    print(f"  LONG  leg: hit {(Lp>0).mean()*100:4.1f}% median {np.median(Lp):+6.1f} mean {Lp.mean():+6.1f} | daily Sharpe(0.5*long) {dsh(d,'long_c'):+.2f}")
    print(f"  SHORT leg: hit {(Sp>0).mean()*100:4.1f}% median {np.median(Sp):+6.1f} mean {Sp.mean():+6.1f} | daily Sharpe(0.5*short) {dsh(d,'short_c'):+.2f}")
    return d

if __name__ == "__main__":
    reg = btc_reg()
    measure(*load("hl_tgt_res_base_honest", "hl_tgt_res_long_honest"), reg, "RECENT 2025-10+")
    measure(*load("hl_v4base_oos_honest", "hl_v4long_oos_honest"), reg, "OOS 2023-25")
    print("\nHONESTLIMITSDONE")
