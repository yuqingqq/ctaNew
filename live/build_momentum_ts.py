"""Find a DIFFERENT-ROOT edge: the return TERM STRUCTURE (short-horizon reversal vs longer-horizon momentum).
Our edge = short-horizon reversal (return_1d/ret_3d, negative IC). Equities show reversal flips to MOMENTUM at
longer horizons (underreaction to persistent info) — a different behavioral root that can coexist. Compute trailing
cumulative returns at 7/14/30/60d, screen each: RAW IC (shows the reversal->momentum flip) + ORTHOGONALIZED IC vs
the existing edge {return_1d, ret_3d, rvol_7d, atr_pct, idio_vol} (does a longer-horizon momentum add ORTHOGONAL,
both-era signal beyond our reversal+vol edge?). Run: python3 -u -m live.build_momentum_ts
"""
from __future__ import annotations

import pandas as pd

from live.v0_feature_ablation import build_panel
from live.build_alpha_beta_decomp import FULL
from live.orthogonal_harness import screen, _fmt

EDGE_CTRL = ["return_1d", "ret_3d", "rvol_7d", "atr_pct", "idio_vol_to_btc_1d"]
HORIZ = {"ret_7d": 42, "ret_14d": 84, "ret_30d": 180, "ret_60d": 360}


def main():
    PAN = build_panel()
    miss = [c for c in set(["return_pct"] + EDGE_CTRL) if c not in PAN.columns]
    ex = pd.read_parquet(FULL, columns=["symbol", "open_time"] + miss)
    ex["open_time"] = pd.to_datetime(ex["open_time"], utc=True)
    PAN = PAN.merge(ex, on=["symbol", "open_time"], how="left").sort_values(["symbol", "open_time"])
    r = PAN.groupby("symbol")["return_pct"]
    for name, w in HORIZ.items():
        PAN[name] = r.transform(lambda s: s.shift(1).rolling(w).sum())   # trailing cumulative return, PIT
    res = screen(PAN, list(HORIZ), controls=EDGE_CTRL)
    print("return TERM STRUCTURE — RAW IC (reversal<0 / momentum>0) then ORTH IC vs existing edge:", flush=True)
    print(f"  {'horizon':<10}{'RAW OOS':<28}{'RAW RECENT':<28}", flush=True)
    for c in HORIZ:
        print(f"  {c:<10}{_fmt(res[c]['raw']['OOS']):<28}{_fmt(res[c]['raw']['RECENT']):<28}", flush=True)
    print(f"\n  {'horizon':<10}{'ORTH OOS (vs edge)':<28}{'ORTH RECENT':<28}   (* CI excl 0; +orth both = new momentum edge)",
          flush=True)
    for c in HORIZ:
        print(f"  {c:<10}{_fmt(res[c]['orth']['OOS']):<28}{_fmt(res[c]['orth']['RECENT']):<28}", flush=True)
    print("\nMOMTSDONE", flush=True)


if __name__ == "__main__":
    main()
