"""REVIEW-2 scratch: probe whether ANY reasonable OB subset flips the 'never significantly positive' verdict.
Reuses the EXACT harness (build/per_bar/sharpe/d_ci from l2_influence_quant; gen from bookdepth_real_ablation).
Tests: imb-only, single features, liq+shape, instability. Reports Δrank-IC & Δspread [day-clustered CI] both eras.
Also prints raw standalone IC of each OB feature (cherry-pick diagnostic) + coverage (bars/days) for thinness check.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
os.environ["V4_PANEL"] = str(REPO / "outputs/vBTC_features/panel_expanded_v0_clean.parquet")
from live.bookdepth_real_ablation import gen, V0_LEAN, RECENT_CUTS, OOS_CUTS
from live.l2_influence_quant import build, per_bar, sharpe, d_ci
from scipy.stats import spearmanr

OB6 = ["imb_ewma", "l2_imb1", "l2_liq1", "l2_slope", "l2_asym1", "l2_imbstd"]
SUBSETS = {
    "imb_ewma_only":   ["imb_ewma"],
    "l2_imb1_only":    ["l2_imb1"],
    "imb_pair":        ["imb_ewma", "l2_imb1"],
    "liq+shape":       ["l2_liq1", "l2_slope", "l2_asym1"],
    "imbstd_only":     ["l2_imbstd"],
    "asym_only":       ["l2_asym1"],
    "OB6_full":        OB6,
}

def raw_ic(PAN, feat, lo, hi):
    """standalone per-bar spearman IC of a single OB feature vs alpha_A on covered bars in [lo,hi)."""
    p = PAN[(PAN["_covered"]) & (PAN["open_time"] >= lo) & (PAN["open_time"] < hi)]
    ics = []
    for t, gg in p.groupby("open_time"):
        if len(gg) < 5: continue
        c = spearmanr(gg[feat], gg["alpha_vs_btc_realized"]).correlation
        if np.isfinite(c): ics.append(c)
    return np.mean(ics) if ics else np.nan

def main():
    PAN = build()
    print(f"panel {len(PAN)} rows | covered {int(PAN['_covered'].sum())}\n")
    eras = {"RECENT": (RECENT_CUTS, RECENT_CUTS[0], RECENT_CUTS[-1]),
            "OOS":    (OOS_CUTS,    OOS_CUTS[0],    OOS_CUTS[-1])}

    # ---- standalone raw IC (which single feature 'looks best') ----
    print("=== standalone raw per-bar IC of each OB feature vs alpha_A (covered bars) ===")
    for feat in OB6:
        r = raw_ic(PAN, feat, eras["RECENT"][1], eras["RECENT"][2])
        o = raw_ic(PAN, feat, eras["OOS"][1], eras["OOS"][2])
        print(f"  {feat:12s}  RECENT {r:+.4f}   OOS {o:+.4f}")
    print()

    # ---- baseline once per era, then each subset ----
    for era, (cuts, lo, hi) in eras.items():
        pb = gen(PAN, V0_LEAN, cuts); icb, spb = per_bar(pb)
        # coverage / thinness diagnostics
        ndays = pd.to_datetime(icb.index, utc=True).floor("1D").nunique()
        print(f"### {era}: base rank-IC {icb.mean():+.4f}  base sel-Sharpe {sharpe(spb):+.2f} "
              f"| {len(icb)} eval-bars over {ndays} days ###")
        for name, sub in SUBSETS.items():
            pa = gen(PAN, V0_LEAN + sub, cuts); ica, spa = per_bar(pa)
            dic, (il, ih) = d_ci(icb, ica)
            dsp, (sl, sh_) = d_ci(spb, spa)
            icflag = "HURTS" if ih < 0 else ("HELPS" if il > 0 else "noise")
            spflag = "HELPS" if sl > 0 else ("HURTS" if sh_ < 0 else "noise")
            print(f"  +{name:14s} ΔIC {dic:+.4f} [{il:+.4f},{ih:+.4f}] {icflag:5s} | "
                  f"+OB Sharpe {sharpe(spa):+.2f}  Δspread {dsp*1e4:+.2f}bps [{sl*1e4:+.2f},{sh_*1e4:+.2f}] {spflag}")
        print()
    print("REVIEW2DONE")

if __name__ == "__main__":
    main()
