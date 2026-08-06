"""WHY is the thin edge there, and where are its relatives? Test the overreaction/lottery-premium hypothesis.

Part 1 SIBLINGS: canonical lottery factors computable free — MAX (Bali: trailing max 4h return) and idiosyncratic
SKEW (Boyer) — screened for RAW and ORTHOGONALIZED IC vs the EXISTING edge factors {return_1d, ret_3d, rvol_7d,
atr_pct, idio_vol}. If orth IC ~0 => SIBLING (same one factor, redundant) — confirms "why" = one behavioral factor.
Part 2 MECHANISM: is the edge (per-symbol Ridge pred) stronger where overreaction is likeliest — ILLIQUID names
(ADV split) and BIG recent movers (|return_1d| split)? If yes => overreaction/liquidity-provision premium living in
thin, high-attention names (also explains the cost wall: alpha and cost are the same names).
Run: python3 -u -m live.build_edge_why
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL
from live.orthogonal_harness import screen, _fmt

CUT = pd.Timestamp("2025-10-01", tz="UTC")
EDGE_CTRL = ["return_1d", "ret_3d", "rvol_7d", "atr_pct", "idio_vol_to_btc_1d"]
SIB = ["max_7d", "max_14d", "ret_skew_14d", "ret_skew_30d"]


def adv_series():
    out = {}
    for f in glob.glob("data/ml/cache/flow_*.parquet"):
        sym = f.split("/")[-1].replace("flow_", "").replace(".parquet", "")
        try:
            d = pd.read_parquet(f, columns=["total_volume", "vwap"])
            out[sym] = float((d["total_volume"] * d["vwap"]).mean())
        except Exception:
            pass
    return pd.Series(out)


def grp_ic(sub, tgt):
    return sub.groupby("open_time").apply(
        lambda g: spearmanr(g["pred"], g[tgt]).correlation if len(g) >= 8 else np.nan,
        include_groups=False).dropna()


def main():
    PAN = build_panel()
    need = list(set(["return_pct", "return_1d"] + EDGE_CTRL))
    miss = [c for c in need if c not in PAN.columns]
    ex = pd.read_parquet(FULL, columns=["symbol", "open_time"] + miss)
    ex["open_time"] = pd.to_datetime(ex["open_time"], utc=True)
    PAN = PAN.merge(ex, on=["symbol", "open_time"], how="left").sort_values(["symbol", "open_time"])
    r = PAN.groupby("symbol")["return_pct"]
    PAN["max_7d"] = r.transform(lambda s: s.shift(1).rolling(42).max())
    PAN["max_14d"] = r.transform(lambda s: s.shift(1).rolling(84).max())
    PAN["ret_skew_14d"] = r.transform(lambda s: s.shift(1).rolling(84).skew())
    PAN["ret_skew_30d"] = r.transform(lambda s: s.shift(1).rolling(180).skew())

    print("===== PART 1: SIBLINGS (lottery factors) — orthogonalized IC vs the existing edge factors =====",
          flush=True)
    print("  (raw IC then ORTH IC; ORTH ~0 => redundant SIBLING = same one factor)", flush=True)
    res = screen(PAN, SIB, controls=EDGE_CTRL)
    print(f"  {'sibling':<16}{'RAW OOS':<28}{'ORTH OOS (vs edge)':<28}{'ORTH RECENT':<28}", flush=True)
    for c in SIB:
        print(f"  {c:<16}{_fmt(res[c]['raw']['OOS']):<28}{_fmt(res[c]['orth']['OOS']):<28}"
              f"{_fmt(res[c]['orth']['RECENT']):<28}", flush=True)

    print("\n===== PART 2: MECHANISM — edge rank-IC in overreaction-prone subsets =====", flush=True)
    ADV = adv_series()
    med = ADV.median()
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(PAN[["symbol", "open_time", "alpha_vs_btc_realized", "return_1d"]],
                       on=["symbol", "open_time"], how="inner").dropna(subset=["pred", "alpha_vs_btc_realized"])
        d["liq"] = d["symbol"].map(ADV) > med
        d["bigrank"] = d.groupby("open_time")["return_1d"].transform(lambda s: s.abs().rank(pct=True))
        il = grp_ic(d[~d["liq"]], "alpha_vs_btc_realized").mean()
        lq = grp_ic(d[d["liq"]], "alpha_vs_btc_realized").mean()
        bg = grp_ic(d[d["bigrank"] > 0.5], "alpha_vs_btc_realized").mean()
        cm = grp_ic(d[d["bigrank"] <= 0.5], "alpha_vs_btc_realized").mean()
        print(f"  {era}: edge rank-IC  ILLIQUID {il:+.4f} vs LIQUID {lq:+.4f}  ({'illiq stronger' if il>lq else 'liq stronger'})"
              f"  |  BIG-mover {bg:+.4f} vs CALM {cm:+.4f}  ({'big stronger' if bg>cm else 'calm stronger'})",
              flush=True)
    print("\nEDGEWHYDONE", flush=True)


if __name__ == "__main__":
    main()
