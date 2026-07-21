"""iter4 — interrogate the era-disagreement + stress-test the recovery.

Every test so far: OOS partial-IC ~0/neg, REC positive. Two questions:
  (1) TEMPORAL: is the REC positivity a smooth regime drift or a discrete jump? (quarterly partial-IC)
  (2) RECOVERY INTEGRITY: REC leans 6x harder on gap-recovery (9.6% vs 1.6% of valid rows). Does the
      REC-positive signal SURVIVE excluding recovered-gap windows? If it vanishes, the recovery
      (correct for coverage) manufactured the recent signal — a critical data-quality finding.
Features: signed_pressure_5min (flow) + imb_change_5min (imbalance). partial-IC vs returns+vol.
"""
from __future__ import annotations
import glob
import numpy as np, pandas as pd
from live.flow_harness import SLIM, SRC, era_masks, ci, partial_xsic, fmt, CUT, HORIZONS
from live.flow_iter2_vol import RET, VOL

FEATS = ["signed_pressure_5min", "imb_change_5min"]


def _load4(sym):
    fp = f"{SLIM}/{sym}.parquet"
    if not glob.glob(fp):
        return None
    d = pd.read_parquet(fp, columns=["bar_time", "price", *FEATS, *RET,
                                     *[f"fwd_{k}" for k in HORIZONS]])
    d["bar_time"] = pd.to_datetime(d["bar_time"], utc=True)
    d = d.drop_duplicates("bar_time").sort_values("bar_time").set_index("bar_time")
    full = pd.date_range(d.index.min(), d.index.max(), freq="5min")
    p = d["price"].reindex(full)
    ret = p.pct_change(fill_method=None)
    aug = pd.DataFrame({"rv_30m": ret.rolling(6, min_periods=4).std(),
                        "rv_1h": ret.rolling(12, min_periods=8).std(),
                        "rv_4h": ret.rolling(48, min_periods=32).std()}, index=full)
    d = d.join(aug, how="left")
    # recovery flags from source
    sf = sorted(glob.glob(f"{SRC}/{sym}/*.parquet"))
    src = pd.read_parquet(sf, columns=["bar_time", "any_raw_gap_5min",
                                       "recovered_internal_gap_5min", "quality_valid_5min"])
    src["bar_time"] = pd.to_datetime(src["bar_time"], utc=True)
    src = src[src["quality_valid_5min"].fillna(False)].drop_duplicates("bar_time").set_index("bar_time")
    d = d.join(src[["any_raw_gap_5min", "recovered_internal_gap_5min"]], how="left")
    out = d.reset_index(names="bar_time")
    keep = ["bar_time", *FEATS, *RET, *[f"fwd_{k}" for k in HORIZONS], "rv_30m", "rv_1h", "rv_4h",
            "any_raw_gap_5min", "recovered_internal_gap_5min"]
    out = out[keep]
    for c in out.columns:
        if out[c].dtype == np.float64:
            out[c] = out[c].astype(np.float32)
    return out


def be(D, feat, tgt, mask, em):
    o = {}
    for era in ("OOS", "REC"):
        o[era] = ci(partial_xsic(D, feat, RET + VOL, tgt, row_mask=mask & em[era]))
    (oa, ol, ou), (ra, rl, ru) = o["OOS"], o["REC"]
    o["both"] = bool(np.sign(oa) == np.sign(ra) and (ol > 0 or ou < 0) and (rl > 0 or ru < 0))
    return o


def main():
    syms = sorted(p.split("/")[-1][:-8] for p in glob.glob(f"{SLIM}/*.parquet"))
    D = pd.concat([x for x in (_load4(s) for s in syms) if x is not None], ignore_index=True)
    D["bar_time"] = pd.to_datetime(D["bar_time"], utc=True)
    em = era_masks(D)
    rec_gap = D["recovered_internal_gap_5min"].fillna(False).to_numpy()
    raw_gap = D["any_raw_gap_5min"].fillna(False).to_numpy()
    print(f"panel {len(D):,} | recovered-gap rows: OOS {rec_gap[em['OOS']].mean():.3f} "
          f"REC {rec_gap[em['REC']].mean():.3f} | any-raw-gap: OOS {raw_gap[em['OOS']].mean():.3f} "
          f"REC {raw_gap[em['REC']].mean():.3f}\n")

    print("=== (2) RECOVERY INTEGRITY: partial-IC vs returns+vol, ALL vs EXCLUDING recovered/gap windows ===")
    allrows = np.ones(len(D), bool)
    for feat in FEATS:
        print(f"### {feat}")
        for k in ["5m", "1h"]:
            tgt = f"fwd_{k}"
            print(f"  {k:>3} ALL          : {fmt(be(D, feat, tgt, allrows, em))}")
            print(f"      excl-recovered: {fmt(be(D, feat, tgt, ~rec_gap, em))}")
            print(f"      excl-any-gap  : {fmt(be(D, feat, tgt, ~raw_gap, em))}", flush=True)
        print()

    print("=== (1) TEMPORAL: quarterly partial-IC (signed_pressure @5m vs returns+vol) ===")
    q = D["bar_time"].dt.to_period("Q").astype(str).to_numpy()
    for qq in sorted(set(q)):
        m = (q == qq)
        if m.sum() < 5000:
            continue
        icq = partial_xsic(D, "signed_pressure_5min", RET + VOL, "fwd_5m", row_mask=m)
        a, lo, hi = ci(icq)
        tag = "REC" if pd.Period(qq) >= pd.Period("2025Q4") else "oos"
        print(f"  {qq} [{tag}] IC {a:+.4f} [{lo:+.4f},{hi:+.4f}]  n_ts={len(icq)}", flush=True)
    print("\nITER4DONE", flush=True)


if __name__ == "__main__":
    main()
