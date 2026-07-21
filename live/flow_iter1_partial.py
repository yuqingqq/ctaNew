"""iter1 — the clean 'information beyond price' test on complete data.

For each flow feature and forward horizon, cross-sectional rank-IC:
  RAW      = does the feature predict at all?
  PARTIAL  = does it predict AFTER residualizing on the trailing-return price set
             [tr_5m,tr_15m,tr_30m,tr_1h] per bar_time?  (== info beyond price)
Both eras (OOS<2025-10-01, RECENT), day-clustered CIs. A feature ADDS beyond price
only if its PARTIAL IC is same-sign CI-off-zero in BOTH eras.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from live.flow_harness import (
    load_panel, era_masks, ci, xsic, partial_xsic, fmt,
    FLOW, TRAIL, HORIZONS,
)

PRICE_SET = list(TRAIL)


def be(D, feat, tgt, masks, controls=None):
    o = {}
    for era in ("OOS", "REC"):
        ic = (partial_xsic(D, feat, controls, tgt, row_mask=masks[era]) if controls
              else xsic(D, feat, tgt, row_mask=masks[era]))
        o[era] = ci(ic)
    (oa, ol, ou), (ra, rl, ru) = o["OOS"], o["REC"]
    o["both"] = bool(np.sign(oa) == np.sign(ra) and (ol > 0 or ou < 0) and (rl > 0 or ru < 0))
    return o


def main():
    cols = ["bar_time", *FLOW, *TRAIL, *[f"fwd_{k}" for k in HORIZONS]]
    D = load_panel(cols)
    masks = era_masks(D)
    print(f"panel {len(D):,} rows | OOS {int(masks['OOS'].sum()):,} | REC {int(masks['REC'].sum()):,}")
    print(f"price control set = {PRICE_SET}\n")
    print("collinearity guard — max |corr(feat, price control)| (>0.95 => partial artifact-prone):")
    for feat in FLOW:
        cc = max(abs(float(D[[feat, c]].corr().iloc[0, 1])) for c in PRICE_SET)
        print(f"  {feat:>26}: {cc:.3f}")
    print()
    print("RAW vs PARTIAL(|price) cross-sectional rank-IC, OOS[CI] | REC[CI] | both-era?\n")

    survivors = []
    for feat in FLOW:
        print(f"### {feat}")
        for k in HORIZONS:
            tgt = f"fwd_{k}"
            raw = be(D, feat, tgt, masks)
            par = be(D, feat, tgt, masks, controls=PRICE_SET)
            print(f"  {k:>4} RAW : {fmt(raw)}")
            print(f"       PART: {fmt(par)}   <- beyond price")
            if par["both"]:
                (oa, _, _), (ra, _, _) = par["OOS"], par["REC"]
                survivors.append((feat, k, oa, ra))
        print(flush=True)

    print("=" * 70)
    if survivors:
        print("PARTIAL both-era CI-off-zero survivors (feature adds beyond price):")
        for feat, k, oa, ra in survivors:
            print(f"  {feat:>26} @{k:<4} OOS {oa:+.4f} / REC {ra:+.4f}")
    else:
        print("NO flow feature has a both-era CI-off-zero PARTIAL IC at any horizon.")
        print("=> unconditional 'information beyond price' = NULL on complete data (redundant with price).")
    print("\nITER1DONE", flush=True)


if __name__ == "__main__":
    main()
