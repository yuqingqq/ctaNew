"""X127 — focused robustness of the HYSTERESIS (fuller re-entry heal) efficiency variant.

iter-013 finding: raising the re-entry heal from 0.50 -> 0.75/0.90 is the ONLY variant that PARETO-WINS
on HL70 (identical maxDD, lower cost, higher Sharpe/Calmar) by cutting a true whipsaw round-trip,
without giving back DD-cap. On EXT/S44 it is neutral (S44 identical) or slightly costlier (EXT).
This script verifies the HL70 Pareto win is robust across k and cost (not a single-cell artifact) and
quantifies exactly which round-trip(s) hysteresis removes.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util as _ilu

REPO = Path("/home/yuqing/ctaNew")
SCRIPTS = REPO/"research/convexity_portable_2026-05-20/scripts"

_s = _ilu.spec_from_file_location("x126", SCRIPTS/"X126_volnorm_efficiency.py")
x126 = _ilu.module_from_spec(_s); _s.loader.exec_module(x126)
build_universe = x126.build_universe
metrics = x126.metrics; gross_unit = x126.gross_unit
run_eff = x126.run_eff_heldbook
HL70_PREDS, EXT_PREDS, S44_PREDS = x126.HL70_PREDS, x126.EXT_PREDS, x126.S44_PREDS


def main():
    t0 = time.time()
    print("X127 — hysteresis (fuller-heal) Pareto robustness across k x cost", flush=True)
    panels = {}
    for name, pp in (("HL70", HL70_PREDS), ("EXT", EXT_PREDS), ("S44", S44_PREDS)):
        U = build_universe(pp, name); panels[name] = U

    for name in ("HL70", "EXT", "S44"):
        U = panels[name]
        print(f"\n{'='*100}\n{name}: binary(heal=0.50) vs hyst(heal=0.90) at k x cost\n{'='*100}", flush=True)
        print(f"{'k':>4}{'cost':>6} | {'bin_ddRed':>10}{'bin_cost':>9}{'bin_RT':>7} | "
              f"{'hyst_ddRed':>11}{'hyst_cost':>10}{'hyst_RT':>8} | {'dCost':>7}{'dSharpe':>8}  verdict", flush=True)
        for cb in (1.0, 3.0, 4.5):
            base = gross_unit(U["cyc"]["base"], U["rs"], cb*1e-4)*1e4; bm = metrics(base)
            for k in (1.5, 2.0, 2.5):
                pb, gb, sb, rb, _ = run_eff(U["cyc"]["base"], U["rs"], cb*1e-4, k, variant="binary")
                ph, gh, sh, rh, _ = run_eff(U["cyc"]["base"], U["rs"], cb*1e-4, k, variant="hyst", heal=0.90)
                mb = metrics(pb*1e4); mh = metrics(ph*1e4)
                ddb = (1-mb["maxDD"]/bm["maxDD"])*100; ddh = (1-mh["maxDD"]/bm["maxDD"])*100
                cb_ = (1-mb["tot"]/bm["tot"])*100; ch_ = (1-mh["tot"]/bm["tot"])*100
                pareto = (ddh >= ddb - 0.5) and (ch_ < cb_ - 0.2)
                neutral = abs(ddh-ddb) < 0.5 and abs(ch_-cb_) < 0.2
                v = "PARETO-WIN" if pareto else ("neutral" if neutral else ("worse-cost" if ch_>cb_+0.2 else ""))
                print(f"{k:>4.1f}{cb:>6.1f} | {ddb:>10.1f}{cb_:>9.1f}{rb:>7} | "
                      f"{ddh:>11.1f}{ch_:>10.1f}{rh:>8} | {ch_-cb_:>+7.1f}{mh['Sharpe']-mb['Sharpe']:>+8.2f}  {v}",
                      flush=True)
    print(f"\nDone [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
