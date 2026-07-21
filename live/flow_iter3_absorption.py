"""iter3 — the charter's CENTRAL hypothesis: conditional / absorption alpha (era-locked).

Mechanism: flow carries usable info only in the regime where it DECOUPLES from price —
absorption = large aggressive flow that did NOT move price. Outside that regime it is the
vol-proxy noise iter2 showed. Test whether flow's forward IC (partial vs price+vol)
concentrates + becomes both-era robust in the ABSORBED regime.

Absorption score (PIT, cross-sectional, scale-free): at each bar_time,
    absorption = pct_rank(|signed_pressure|) - pct_rank(|price move|)
high (>0) = lots of flow, little price move = ABSORBED; low (<0) = flow moved price ("spent").
ERA-LOCK (anti-favorable-corner): the median split point is computed on OOS ONLY and applied
UNCHANGED to RECENT (and, as a robustness flip, computed on REC and applied to OOS).
Then partial-IC(signed_pressure -> fwd | returns+vol) in absorbed vs spent, both eras.
Confirmed only if flow's IC is materially larger AND both-era CI-off-zero in ABSORBED.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from live.flow_harness import era_masks, ci, partial_xsic, fmt, CUT, HORIZONS
from live.flow_iter2_vol import _load_sym, RET, VOL

FLOWFEAT = "signed_pressure_5min"
MOVE = "tr_5m"  # ~= return_5min (the 5-min price move), for the decoupling denominator


def be_masked(D, feat, tgt, controls, row_mask, em):
    o = {}
    for era in ("OOS", "REC"):
        o[era] = ci(partial_xsic(D, feat, controls, tgt, row_mask=row_mask & em[era]))
    (oa, ol, ou), (ra, rl, ru) = o["OOS"], o["REC"]
    o["both"] = bool(np.sign(oa) == np.sign(ra) and (ol > 0 or ou < 0) and (rl > 0 or ru < 0))
    return o


def main():
    import glob
    from live.flow_harness import SLIM
    syms = sorted(p.split("/")[-1][:-8] for p in glob.glob(f"{SLIM}/*.parquet"))
    D = pd.concat([x for x in (_load_sym(s) for s in syms) if x is not None], ignore_index=True)
    for c in D.columns:
        if D[c].dtype == np.float64:
            D[c] = D[c].astype(np.float32)
    D["bar_time"] = pd.to_datetime(D["bar_time"], utc=True)

    # absorption score: within-bar_time percentile of |flow| minus percentile of |move|
    D["_af"] = D[FLOWFEAT].abs()
    D["_am"] = D[MOVE].abs()
    grp = D.groupby("bar_time")
    D["absorption"] = (grp["_af"].rank(pct=True) - grp["_am"].rank(pct=True)).astype(np.float32)
    D.drop(columns=["_af", "_am"], inplace=True)

    em = era_masks(D)
    print(f"panel {len(D):,} | OOS {int(em['OOS'].sum()):,} | REC {int(em['REC'].sum()):,}")
    print(f"absorption: mean {D['absorption'].mean():.3f} sd {D['absorption'].std():.3f}\n")

    for lock_era, apply_era in [("OOS", "REC"), ("REC", "OOS")]:
        cut = float(np.nanmedian(D.loc[em[lock_era], "absorption"].to_numpy()))
        absorbed = (D["absorption"].to_numpy() > cut)
        spent = ~absorbed
        print(f"===== era-lock split on {lock_era} (median absorption={cut:+.3f}), applied to {apply_era} =====")
        for k in ["5m", "30m", "1h"]:
            tgt = f"fwd_{k}"
            full = be_masked(D, FLOWFEAT, tgt, RET + VOL, np.ones(len(D), bool), em)
            ab = be_masked(D, FLOWFEAT, tgt, RET + VOL, absorbed, em)
            sp = be_masked(D, FLOWFEAT, tgt, RET + VOL, spent, em)
            print(f"  {k:>4} FULL    : {fmt(full)}")
            print(f"       ABSORBED: {fmt(ab)}   {'BOTH' if ab['both'] else ''}")
            print(f"       SPENT   : {fmt(sp)}")
        print(flush=True)

    print("read: conditional edge only if ABSORBED IC is materially larger AND both-era CI-off-zero,")
    print("robust to which era the split was locked on. Else absorption adds no usable conditioning.")
    print("ITER3DONE", flush=True)


if __name__ == "__main__":
    main()
