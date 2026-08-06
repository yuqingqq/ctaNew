"""ORTHOGONAL-DATA loop iter 5: broaden the candidate families and re-screen (G1). Rebuilds the metrics-feature
cache to include the expanded set, adds a post-merge interaction (oi_price_div = oi_chg_1d * sign(return_1d)),
and runs the orthogonalized-IC screen (residual vs {vol, reversal}) both eras. Any both-era survivor (CI-excludes-0
same sign) is a candidate for the G2/G3 pipeline. Run: python3 -u -m live.orth_iter5_screen
"""
from __future__ import annotations

import numpy as np

from live.orthogonal_harness import build_panel_with_metrics, screen, _fmt

EXPANDED = ["tt_pos_chg_3d", "gl_acc_chg_1d", "smart_dumb_chg_1d", "taker_chg_3d", "oi_z", "oi_price_div"]


def main():
    d = build_panel_with_metrics(rebuild=True)
    d["oi_price_div"] = d["oi_chg_1d"] * np.sign(d["return_1d"])
    cov = d.dropna(subset=[c for c in EXPANDED if c in d.columns])["symbol"].nunique()
    print(f"panel+metrics(expanded): {len(d):,} rows | {d.symbol.nunique()} syms | {cov} syms full\n", flush=True)
    r = screen(d, EXPANDED)
    print(f"{'candidate':<18}{'ORTH OOS (vs vol+rev)':<30}{'ORTH RECENT':<30}   (* = CI excludes 0)", flush=True)
    for c in EXPANDED:
        print(f"  {c:<16}{_fmt(r[c]['orth']['OOS']):<30}{_fmt(r[c]['orth']['RECENT']):<30}", flush=True)
    print("\n(raw IC, for context):", flush=True)
    for c in EXPANDED:
        print(f"  {c:<16}{_fmt(r[c]['raw']['OOS']):<30}{_fmt(r[c]['raw']['RECENT']):<30}", flush=True)
    surv = [c for c in EXPANDED
            if (r[c]['orth']['OOS'][1] > 0 or r[c]['orth']['OOS'][2] < 0)
            and (r[c]['orth']['RECENT'][1] > 0 or r[c]['orth']['RECENT'][2] < 0)
            and np.sign(r[c]['orth']['OOS'][0]) == np.sign(r[c]['orth']['RECENT'][0])]
    print(f"\nG1 both-era survivors (orth CI excludes 0, same sign): {surv}", flush=True)
    print("ORTH5DONE", flush=True)


if __name__ == "__main__":
    main()
