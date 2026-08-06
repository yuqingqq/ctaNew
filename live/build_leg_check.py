"""Which leg holds the high-vol lottery names? Measure avg vol/reversal of the LONG vs SHORT leg of the deployed
book (per-symbol Ridge -> quintile L/S), both eras. Settles: short leg = high-vol (low-vol anomaly) or not.
Run: python3 -u -m live.build_leg_check
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL

COLS = ["rvol_7d", "atr_pct", "return_1d"]


def main():
    PAN = build_panel()
    ex = pd.read_parquet(FULL, columns=["symbol", "open_time"] + COLS)
    ex["open_time"] = pd.to_datetime(ex["open_time"], utc=True)
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(ex, on=["symbol", "open_time"], how="inner").dropna(subset=["pred"] + COLS)
        d["rk"] = d.groupby("open_time")["pred"].rank(pct=True)
        lg = d[d["rk"] >= 0.8]        # LONG = high predicted alpha
        sh = d[d["rk"] <= 0.2]        # SHORT = low predicted alpha
        print(f"===== {era} =====  (LONG = top-quintile pred, SHORT = bottom-quintile)", flush=True)
        for c in COLS:
            lo, so = lg[c].median(), sh[c].median()
            hi_leg = "SHORT" if so > lo else "LONG"
            print(f"  {c:<12} LONG median {lo:+.4f} | SHORT median {so:+.4f}  -> higher in {hi_leg}", flush=True)
        # cross-sectional-rank version (robust to scale): avg percentile of each leg's vol
        for c in ("rvol_7d", "atr_pct"):
            d["vr"] = d.groupby("open_time")[c].rank(pct=True)
            print(f"  {c} cross-sec pctile: LONG {d.loc[lg.index,'vr'].mean():.2f} | "
                  f"SHORT {d.loc[sh.index,'vr'].mean():.2f}  (0.5=median; >0.5 = higher-vol leg)", flush=True)
        print("", flush=True)
    print("LEGCHECKDONE", flush=True)


if __name__ == "__main__":
    main()
