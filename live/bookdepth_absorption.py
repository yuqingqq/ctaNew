"""LASTING-PATTERN candidate: ABSORPTION ASYMMETRY. Not the book's STATE (imbalance/liq — decays or redundant) but
its RESPONSE to price — a slow behavioral trait. Per name, over a trailing window, how does depth co-move with return?
  bid thickens when price FALLS  -> structural demand / dip-absorption  -> candidate for reversion/outperformance
  bid thins   when price FALLS  -> fragility                           -> continuation/underperformance
Reconstruct per-side depth from the cached near book (no re-fetch): total=exp(l2_liq1); bidN=total*(1+imb1)/2;
askN=total*(1-imb1)/2. Per 4h obs bar: r=past 4h return; dlb=Δlog(bidN); dla=Δlog(askN); dlt=Δlog(total). Trailing
W=30-bar beta of each depth-change on return (cov/var). Features (PIT, shifted +4h to decision bar):
  absorp_bid = -beta_bid              (high = bid thickens as price falls = support)
  absorp_net = beta_ask - beta_bid    (bid-supportive AND ask-thinning = bullish structural demand)
  resil_tot  = -beta_tot              (total book thickens on down = resilient, thins = fragile)
Gate: cross-sectional rank-IC vs fwd alpha (+ raw fwd return), BOTH eras, day-clustered CI. Escalate to the real
per-symbol RidgeCV ablation ONLY if a construction is same-sign + CI-off-zero in BOTH eras (else it's dead here).
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.bookdepth_persist import close_4h, day_boot   # reuse PIT 4h-close + day-clustered bootstrap
CACHE = Path("/home/yuqing/ctaNew/data/ml/cache")
rng = np.random.default_rng(23); W = 30; MINP = 15

def _seg_betas(bidN, askN, tot, r):
    dlb = np.log(bidN).diff(); dla = np.log(askN).diff(); dlt = np.log(tot).diff()
    vr = r.rolling(W, min_periods=MINP).var()
    bbid = dlb.rolling(W, min_periods=MINP).cov(r) / vr
    bask = dla.rolling(W, min_periods=MINP).cov(r) / vr
    btot = dlt.rolling(W, min_periods=MINP).cov(r) / vr
    return pd.DataFrame({"absorp_bid": -bbid, "absorp_net": bask - bbid, "resil_tot": -btot}, index=r.index)

def build_sym(f):
    sym = Path(f).stem[3:]
    try:
        d = pd.read_parquet(f)
    except Exception:
        return None
    if not {"l2_imb1", "l2_liq1"}.issubset(d.columns): return None
    d.index = pd.to_datetime(d.index, utc=True); d = d.sort_index()
    tot = np.exp(d["l2_liq1"]); imb = d["l2_imb1"].clip(-0.999, 0.999)
    bidN = tot * (1 + imb) / 2; askN = tot * (1 - imb) / 2
    lo, hi = str(d.index.min().date()), str((d.index.max() + pd.Timedelta("1D")).date())
    c = close_4h(sym, lo, hi)
    if c is None: return None
    grid = d.index                                        # obs-bar grid
    r = c.reindex(grid.union(c.index)).sort_index().ffill().reindex(grid).pct_change()   # strictly-past 4h return
    seg = (grid.to_series().diff() > pd.Timedelta("8h")).cumsum().values                  # gap-aware
    parts = []
    for g in pd.unique(seg):
        m = seg == g
        parts.append(_seg_betas(bidN[m], askN[m], tot[m], r[m]))
    out = pd.concat(parts)
    out.index = grid + pd.Timedelta("4h")                 # PIT decision bar
    out["symbol"] = sym; out["open_time"] = out.index
    return out.reset_index(drop=True)

def xic(df, feat, tgt):
    return df.groupby("open_time").apply(lambda g: g[feat].corr(g[tgt], method="spearman")
                                         if g[[feat, tgt]].dropna().shape[0] >= 8 else np.nan).dropna()

def main():
    rows = []
    files = [f for f in glob.glob(str(CACHE / "l2_*.parquet")) if "BTCUSDT" not in f]
    for i, f in enumerate(files):
        o = build_sym(f)
        if o is not None: rows.append(o)
        if (i + 1) % 40 == 0: print(f"  built {i+1}/{len(files)}", flush=True)
    L = pd.concat(rows, ignore_index=True)
    pan = pd.read_parquet("/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet",
                          columns=["symbol", "open_time", "return_pct", "alpha_vs_btc_realized"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    m = pan.merge(L, on=["symbol", "open_time"], how="inner")
    cut = pd.Timestamp("2025-10-01", tz="UTC")
    eras = {"RECENT": m[m.open_time >= cut], "OOS": m[m.open_time < cut]}
    print(f"\nmerged {len(m)} rows | {m.symbol.nunique()} syms | RECENT {len(eras['RECENT'])} OOS {len(eras['OOS'])}")
    print("ABSORPTION asymmetry (book RESPONSE to price) -> fwd; a LASTING behavioral trait?\n")
    for tgt, lab in [("alpha_vs_btc_realized", "ALPHA (strategy target)"), ("return_pct", "raw fwd return")]:
        print(f"### target = {lab} : cross-sectional rank-IC, both eras ###")
        print(f"{'feature':11s} | {'RECENT IC [CI]':26s} | {'OOS IC [CI]':26s} | both-era?")
        for feat in ["absorp_bid", "absorp_net", "resil_tot"]:
            cells = {}
            for era, sub in eras.items():
                ic = xic(sub, feat, tgt); lo, up = day_boot(ic); cells[era] = (ic.mean(), lo, up)
            (ra, rl, ru), (oa, ol, ou) = cells["RECENT"], cells["OOS"]
            both = "YES" if (np.sign(ra) == np.sign(oa) and (rl > 0 or ru < 0) and (ol > 0 or ou < 0)) else "no"
            print(f"{feat:11s} | {ra:+.4f} [{rl:+.4f},{ru:+.4f}] | {oa:+.4f} [{ol:+.4f},{ou:+.4f}] | {both}")
        print()
    print("read: any construction same-sign + CI-off-zero BOTH eras -> escalate to real ablation; else dead. ABSORPDONE")

if __name__ == "__main__":
    main()
