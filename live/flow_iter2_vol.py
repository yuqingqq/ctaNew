"""iter2 — is the flow's 'beyond returns' signal actually just VOLATILITY/LIQUIDITY?

iter1 showed several flow features keep a both-era partial-IC after controlling for
trailing RETURNS. But the prior project verdict was that OB info is redundant with VOL
features (idio_vol/rvol/atr), and the survivors (sell_to_bid, buy_to_ask, depth residuals)
are flow-normalized-by-depth = vol/liquidity-laden. So the decisive control adds realized
vol (+ liquidity level). For each feature/horizon print partial-IC under:
  RET      = trailing returns only            (iter1's control)
  RET+VOL  = returns + realized vol (30m,1h,4h)
  +LIQ     = returns + vol + log depth level   (liquidity)
If the both-era survivor COLLAPSES going RET -> RET+VOL/LIQ, it was vol/liquidity redundancy
(confirms prior). If it SURVIVES even RET+VOL+LIQ, it is genuinely new -> escalate.
"""
from __future__ import annotations
import glob
import numpy as np, pandas as pd
from live.flow_harness import (
    SLIM, SRC, era_masks, ci, partial_xsic, fmt, FLOW, TRAIL, HORIZONS,
)

RET = list(TRAIL)
VOL = ["rv_30m", "rv_1h", "rv_4h"]
LIQ = ["liq_log"]


def _load_sym(sym: str) -> pd.DataFrame | None:
    fp = f"{SLIM}/{sym}.parquet"
    files = sorted(glob.glob(fp))
    if not files:
        return None
    d = pd.read_parquet(files[0], columns=["bar_time", "price", *FLOW, *TRAIL,
                                           *[f"fwd_{k}" for k in HORIZONS]])
    d["bar_time"] = pd.to_datetime(d["bar_time"], utc=True)
    d = d.drop_duplicates("bar_time").sort_values("bar_time").set_index("bar_time")
    full = pd.date_range(d.index.min(), d.index.max(), freq="5min")
    p = d["price"].reindex(full)
    ret = p.pct_change(fill_method=None)
    rv = {"rv_30m": ret.rolling(6, min_periods=4).std(),
          "rv_1h": ret.rolling(12, min_periods=8).std(),
          "rv_4h": ret.rolling(48, min_periods=32).std()}
    aug = pd.DataFrame(rv, index=full)
    out = d.join(aug, how="left").reset_index(names="bar_time")
    keep = ["bar_time", *FLOW, *TRAIL, *[f"fwd_{k}" for k in HORIZONS], *VOL]
    out = out[keep]
    for c in out.columns:
        if out[c].dtype == np.float64:
            out[c] = out[c].astype(np.float32)
    return out


def be(D, feat, tgt, masks, controls):
    o = {}
    for era in ("OOS", "REC"):
        o[era] = ci(partial_xsic(D, feat, controls, tgt, row_mask=masks[era]))
    (oa, ol, ou), (ra, rl, ru) = o["OOS"], o["REC"]
    o["both"] = bool(np.sign(oa) == np.sign(ra) and (ol > 0 or ou < 0) and (rl > 0 or ru < 0))
    return o


def main():
    syms = sorted(p.split("/")[-1][:-8] for p in glob.glob(f"{SLIM}/*.parquet"))
    D = pd.concat([x for x in (_load_sym(s) for s in syms) if x is not None], ignore_index=True)
    for c in D.columns:
        if D[c].dtype == np.float64:
            D[c] = D[c].astype(np.float32)
    D["bar_time"] = pd.to_datetime(D["bar_time"], utc=True)
    masks = era_masks(D)
    print(f"panel {len(D):,} rows | OOS {int(masks['OOS'].sum()):,} | REC {int(masks['REC'].sum()):,}")
    print(f"vol coverage {D[VOL].notna().all(axis=1).mean():.3f}\n")
    print("partial-IC under widening controls. OOS[CI] | REC[CI] | both?  (feat @ horizon)\n")

    survive_vol = []
    for feat in FLOW:
        print(f"### {feat}")
        for k in HORIZONS:
            tgt = f"fwd_{k}"
            r = be(D, feat, tgt, masks, RET)
            rv = be(D, feat, tgt, masks, RET + VOL)
            flag = "SURVIVES-VOL" if rv["both"] else ("KILLED-BY-VOL" if r["both"] else "")
            print(f"  {k:>4} RET    : {fmt(r)}")
            print(f"       +VOL   : {fmt(rv)}   {flag}")
            if rv["both"]:
                survive_vol.append((feat, k, rv["OOS"][0], rv["REC"][0]))
        print(flush=True)

    print("=" * 70)
    if survive_vol:
        print("Both-era survivors of RET+VOL control (beyond price+vol):")
        for feat, k, oa, ra in survive_vol:
            print(f"  {feat:>26} @{k:<4} OOS {oa:+.4f} / REC {ra:+.4f}")
    else:
        print("NO feature survives the RET+VOL control both-era.")
        print("=> iter1's 'beyond returns' signal WAS vol redundancy (confirms prior verdict).")
    print("\nITER2DONE", flush=True)


if __name__ == "__main__":
    main()
