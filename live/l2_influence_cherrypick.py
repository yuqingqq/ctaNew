"""Decisive cherry-pick check (the review's key job, run directly on the validated harness): does ANY order-book
feature subset — each single feature + combos — produce a Δrank-IC OR Δselection-spread that is POSITIVE with CI-off-
zero in BOTH eras vs the V0_LEAN baseline? If none, "OB influence = negative-to-neutral, never significantly positive"
is robust (not a hidden-subset artifact). Reuses the validated real pipeline (baseline reproduces +0.030).
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.l2_influence_quant import build, per_bar, d_ci, sharpe
from live.bookdepth_real_ablation import gen, V0_LEAN, RECENT_CUTS, OOS_CUTS

SUBSETS = {
    "imb1 (imbalance)": ["l2_imb1"], "imb_ewma (sustained)": ["imb_ewma"], "liq1 (liquidity)": ["l2_liq1"],
    "slope (shape)": ["l2_slope"], "asym1 (asymmetry)": ["l2_asym1"], "imbstd (instability)": ["l2_imbstd"],
    "liq1+slope": ["l2_liq1", "l2_slope"], "ALL 6": ["imb_ewma", "l2_imb1", "l2_liq1", "l2_slope", "l2_asym1", "l2_imbstd"],
}

def main():
    PAN = build()
    res = {}  # (subset, era) -> (dic, dic_ci, dsp_bps, dsp_ci)
    base_ic = {}
    for era, cuts in [("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)]:
        pb = gen(PAN, V0_LEAN, cuts); icb, spb = per_bar(pb); base_ic[era] = icb.mean()
        print(f"[{era}] baseline rank-IC {icb.mean():+.4f} (gate: ~+0.030 rec / +0.017 oos)", flush=True)
        for name, ss in SUBSETS.items():
            pa = gen(PAN, V0_LEAN + ss, cuts); ica, spa = per_bar(pa)
            dic, (il, ih) = d_ci(icb, ica); dsp, (sl, sh) = d_ci(spb, spa)
            res[(name, era)] = (dic, il, ih, dsp * 1e4, sl * 1e4, sh * 1e4)
            print(f"    {name:22s} Δrank-IC {dic:+.4f} [{il:+.4f},{ih:+.4f}] | Δspread {dsp*1e4:+.2f}bps [{sl*1e4:+.2f},{sh*1e4:+.2f}]", flush=True)
    print("\n=== BOTH-ERA POSITIVE? (the cherry-pick verdict) ===")
    any_pos = False
    for name in SUBSETS:
        r_ic = res[(name, "RECENT")]; o_ic = res[(name, "OOS")]
        ic_pos = r_ic[1] > 0 and o_ic[1] > 0                       # both eras rank-IC CI lower bound > 0
        sp_pos = r_ic[4] > 0 and o_ic[4] > 0                       # both eras spread CI lower bound > 0
        verdict = "POSITIVE both-era!" if (ic_pos or sp_pos) else "no"
        if ic_pos or sp_pos: any_pos = True
        print(f"  {name:22s} rank-IC both-CI>0: {ic_pos} | spread both-CI>0: {sp_pos} -> {verdict}")
    print(f"\nVERDICT: {'FOUND a both-era-positive OB subset -> revise the claim' if any_pos else 'NO OB subset is both-era CI-positive -> negative-to-neutral claim HOLDS'}")
    print("CHERRYDONE")

if __name__ == "__main__":
    main()
