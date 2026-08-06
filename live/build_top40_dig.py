"""Dig deeper: WHY does the per-symbol model earn in top-40 when the naive cross-sectional factor loses?
WITHIN-vs-BETWEEN decomposition of each feature (PIT rolling-60d symbol norm):
  between_i,t = symbol's own trailing mean (structural: 'SOL is always high-vol')
  within_i,t  = feature - own mean (deviation: 'SOL is hotter than ITS norm right now')
If the edge lives in WITHIN (IC strong) not BETWEEN (~0), the mechanism is per-symbol (time-series) mean-reversion /
self-normalization — which the per-symbol Ridge captures and a naive cross-sectional rank (dominated by between)
misses. Also corr(pred, within) vs corr(pred, between). Top-40, both eras. Run: python3 -u -m live.build_top40_dig
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL

NTOP = 40
CUT = pd.Timestamp("2025-10-01", tz="UTC")
FEATS = ["rvol_7d", "return_1d", "atr_pct"]


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


def ic(d, col, tgt="alpha_vs_btc_realized"):
    return d.dropna(subset=[col, tgt]).groupby("open_time").apply(
        lambda g: spearmanr(g[col], g[tgt]).correlation if len(g) >= 8 else np.nan,
        include_groups=False).dropna()


def main():
    top = set(adv_rank()[:NTOP])
    PAN = build_panel()
    miss = [c for c in FEATS if c not in PAN.columns]
    if miss:
        ex = pd.read_parquet(FULL, columns=["symbol", "open_time"] + miss)
        ex["open_time"] = pd.to_datetime(ex["open_time"], utc=True)
        PAN = PAN.merge(ex, on=["symbol", "open_time"], how="left")
    PAN = PAN.sort_values(["symbol", "open_time"])
    for f in FEATS:
        m = PAN.groupby("symbol")[f].transform(lambda s: s.rolling(360, min_periods=60).mean().shift(1))
        PAN[f + "_btw"] = m
        PAN[f + "_wth"] = PAN[f] - m
    P40 = PAN[PAN["symbol"].isin(top)].copy()
    print(f"TOP-{NTOP} within/between IC vs fwd alpha (edge in WITHIN => per-symbol mean-reversion):\n", flush=True)
    for era, lo, hi in (("RECENT", CUT, None), ("OOS", None, CUT)):
        d = P40 if era == "RECENT" and False else P40
        d = P40[P40["open_time"] >= CUT] if era == "RECENT" else P40[P40["open_time"] < CUT]
        print(f"===== {era} =====", flush=True)
        for f in FEATS:
            print(f"  {f:<12} raw {ic(d, f).mean():+.4f} | WITHIN {ic(d, f+'_wth').mean():+.4f} | "
                  f"BETWEEN {ic(d, f+'_btw').mean():+.4f}", flush=True)
        print("", flush=True)
    # correlation of the model's pred with within vs between (does the model use the deviation?)
    print("corr(pred, within) vs corr(pred, between) — does the per-symbol model USE the deviation?:", flush=True)
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(PAN[["symbol", "open_time"] + [f + s for f in FEATS for s in ("_wth", "_btw")]],
                       on=["symbol", "open_time"], how="left")
        d = d[d["symbol"].isin(top)].dropna(subset=["pred"])
        def cc(col):
            return d.dropna(subset=[col]).groupby("open_time").apply(
                lambda g: spearmanr(g["pred"], g[col]).correlation if len(g) >= 8 else np.nan,
                include_groups=False).dropna().mean()
        print(f"  {era}: " + " | ".join(
            f"{f}: wth {cc(f+'_wth'):+.2f} btw {cc(f+'_btw'):+.2f}" for f in FEATS), flush=True)
    print("\nTOP40DIGDONE", flush=True)


if __name__ == "__main__":
    main()
