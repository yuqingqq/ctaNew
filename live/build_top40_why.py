"""Why is the edge there in the DEPLOYABLE top-40 (big names) universe — not the long tail? Re-run the mechanism
tests restricted to top-40 by ADV, both eras:
  A. LOADINGS within top-40: corr(pred, factor) for reversal (return_1d/ret_3d) vs vol (rvol/atr/idio_vol).
  B. LEG COMPOSITION within top-40: is short leg still higher-vol / recent-winner?
  C. OVERREACTION: rank-IC for big recent movers vs calm, within top-40.
  D. ATTRIBUTION: strat L/S ~ pure-reversal + pure-low-vol, within top-40 (which earns).
Run: python3 -u -m live.build_top40_why
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL

PYR = 6 * 365.0
NTOP = 40
REV = ["return_1d", "ret_3d"]
VOL = ["rvol_7d", "atr_pct", "idio_vol_to_btc_1d"]


def adv_rank():
    out = {}
    for f in glob.glob("data/ml/cache/flow_*.parquet"):
        sym = f.split("/")[-1].replace("flow_", "").replace(".parquet", "")
        try:
            d = pd.read_parquet(f, columns=["total_volume", "vwap"])
            out[sym] = float((d["total_volume"] * d["vwap"]).mean())
        except Exception:
            pass
    return pd.Series(out).sort_values(ascending=False).index.tolist()


def ls_ret(d, score):
    dd = d.dropna(subset=[score, "return_pct"]).copy()
    dd["_r"] = dd.groupby("open_time")[score].rank(pct=True)

    def f(g):
        lo = g["_r"] >= 0.8; sh_ = g["_r"] <= 0.2
        return (g.loc[lo, "return_pct"].mean() - g.loc[sh_, "return_pct"].mean()
                ) if (lo.sum() >= 2 and sh_.sum() >= 2) else np.nan
    return dd.groupby("open_time").apply(f, include_groups=False).dropna()


def sh(x):
    return x.mean() / x.std() * np.sqrt(PYR) if len(x) and x.std() > 0 else np.nan


def main():
    top = set(adv_rank()[:NTOP])
    PAN = build_panel()
    cols = list(set(["return_pct"] + REV + VOL))
    miss = [c for c in cols if c not in PAN.columns]
    ex = pd.read_parquet(FULL, columns=["symbol", "open_time"] + miss)
    ex["open_time"] = pd.to_datetime(ex["open_time"], utc=True)
    PAN = PAN.merge(ex, on=["symbol", "open_time"], how="left")
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(PAN[["symbol", "open_time", "alpha_vs_btc_realized"] + cols],
                       on=["symbol", "open_time"], how="left")
        d = d[d["symbol"].isin(top)].dropna(subset=["pred", "return_pct", "alpha_vs_btc_realized"])
        print(f"===== TOP-{NTOP}  {era}  ({d.symbol.nunique()} names) =====", flush=True)
        # A. loadings
        print("  A. loadings corr(pred,factor):", flush=True)
        for grp, cs in (("reversal", REV), ("vol", VOL)):
            for c in cs:
                dd = d.dropna(subset=[c])
                ic = dd.groupby("open_time").apply(
                    lambda g: spearmanr(g["pred"], g[c]).correlation if len(g) >= 8 else np.nan,
                    include_groups=False).dropna().mean()
                print(f"      [{grp:<8}] {c:<20} {ic:+.3f}", flush=True)
        # B. leg composition
        d["rk"] = d.groupby("open_time")["pred"].rank(pct=True)
        lg = d["rk"] >= 0.8; shrt = d["rk"] <= 0.2
        print("  B. leg pctile (short vs long):", flush=True)
        for c in ["rvol_7d", "atr_pct", "return_1d"]:
            d["vp"] = d.groupby("open_time")[c].rank(pct=True)
            print(f"      {c:<20} LONG {d.loc[lg,'vp'].mean():.2f} | SHORT {d.loc[shrt,'vp'].mean():.2f}", flush=True)
        # C. overreaction
        d["bigrank"] = d.groupby("open_time")["return_1d"].transform(lambda s: s.abs().rank(pct=True))
        def gic(sub):
            return sub.groupby("open_time").apply(
                lambda g: spearmanr(g["pred"], g["alpha_vs_btc_realized"]).correlation if len(g) >= 6 else np.nan,
                include_groups=False).dropna().mean()
        bg = gic(d[d.bigrank > 0.5]); cm = gic(d[d.bigrank <= 0.5])
        print(f"  C. overreaction: rank-IC BIG-mover {bg:+.4f} vs CALM {cm:+.4f} "
              f"({'big stronger' if bg > cm else 'calm stronger'})", flush=True)
        # D. attribution
        d["rev_s"] = -d["return_1d"]; d["vol_s"] = -d["rvol_7d"]
        strat = ls_ret(d, "pred"); rev = ls_ret(d, "rev_s"); vol = ls_ret(d, "vol_s")
        j = pd.concat([strat.rename("s"), rev.rename("r"), vol.rename("v")], axis=1).dropna()
        X = np.column_stack([np.ones(len(j)), j["r"], j["v"]])
        beta, *_ = np.linalg.lstsq(X, j["s"].to_numpy(), rcond=None)
        r2 = 1 - np.var(j["s"].to_numpy() - X @ beta) / np.var(j["s"].to_numpy())
        print(f"  D. attribution: pure-reversal Sh {sh(rev):+.2f} (β {beta[1]:+.2f}) | pure-low-vol Sh {sh(vol):+.2f} "
              f"(β {beta[2]:+.2f}) | strat Sh {sh(j['s']):+.2f} | R² {r2:.2f}", flush=True)
        print("", flush=True)
    print("TOP40WHYDONE", flush=True)


if __name__ == "__main__":
    main()
