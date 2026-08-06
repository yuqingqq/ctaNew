"""ORTHOGONAL-DATA loop iter 2 (gate G2): do the G1 survivors add INCREMENTAL rank-IC to V0 through the REAL
per-symbol RidgeCV pipeline (gen from v0_feature_ablation)? Paired day-clustered CI, both eras. Also test the
two together. Survivors from iter-1 screen: oi_chg_1d (+), tt_pos_chg_1d (-).

Neutral NaN imputation (per-bar median then 0) so warmup rows don't silently drop symbols in gen().
Run: python3 -u -m live.orth_pipeline
"""
from __future__ import annotations

import numpy as np

from live.v0_feature_ablation import gen, perbar_ic, paired_ci, V0, RECENT_CUTS, OOS_CUTS
from live.orthogonal_harness import build_panel_with_metrics

SURV = ["oi_chg_1d", "tt_pos_chg_1d"]


def main():
    PAN = build_panel_with_metrics()
    for c in SURV:
        PAN[c] = PAN.groupby("open_time")[c].transform(lambda s: s.fillna(s.median()))
        PAN[c] = PAN[c].fillna(0.0)
    PAN = PAN.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    print(f"panel {len(PAN):,} rows | {PAN.symbol.nunique()} syms | V0={len(V0)} + survivors {SURV}\n", flush=True)
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        print(f"================= {era} =================", flush=True)
        ib = perbar_ic(gen(PAN, list(V0), cuts))
        print(f"  BASELINE V0 rank-IC {ib.mean():+.4f}  [gate ~+0.030 rec / +0.021 oos]", flush=True)
        for c in SURV:
            d, lo, up = paired_ci(ib, perbar_ic(gen(PAN, list(V0) + [c], cuts)))
            tag = "ADDS (CI>0)" if (np.isfinite(lo) and lo > 0) else (
                  "hurts (CI<0)" if (np.isfinite(up) and up < 0) else "within noise")
            print(f"    +{c:16s} Δ {d:+.4f} [{lo:+.4f},{up:+.4f}]  {tag}", flush=True)
        d, lo, up = paired_ci(ib, perbar_ic(gen(PAN, list(V0) + SURV, cuts)))
        tag = "ADDS (CI>0)" if (np.isfinite(lo) and lo > 0) else (
              "hurts (CI<0)" if (np.isfinite(up) and up < 0) else "within noise")
        print(f"    +{'BOTH':16s} Δ {d:+.4f} [{lo:+.4f},{up:+.4f}]  {tag}", flush=True)
        print("", flush=True)
    print("ORTHPIPEDONE", flush=True)


if __name__ == "__main__":
    main()
