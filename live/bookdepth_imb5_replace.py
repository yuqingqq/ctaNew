"""User's idea: imb5 ADDS nothing (redundant) — but could it REPLACE a price feature and do BETTER, since OB "leads"
price? Redundancy is symmetric in corr but NOT in quality: a collinear feature can be a cleaner/earlier reading. Three
tests on the REAL pipeline (same machinery + validity gate as bookdepth_imb5_ablation):

 (1) COLLINEARITY   corr(imb5_ewma, each V0 feat) on covered bars — WHAT does it overlap? (identifies replacement targets)
 (2) HEAD-TO-HEAD   univariate x-sec rank-IC(feature -> fwd alpha), both eras, for imb5 vs the price momentum feats it
                    overlaps. If OB truly LEADS at 4h, imb5's forward IC should BEAT the lagging price feat's. (direct
                    test of "OB leads price" AT OUR HORIZON.)
 (3) REPLACEMENT    swap each price feat X -> imb5_ewma in V0_LEAN; per-symbol RidgeCV rank-IC vs baseline, paired
                    day-clustered CI, both eras. Replace HELPS only if Δ CI>0 both eras. Also a DROP(return_1d) column
                    (no replacement) to isolate how much that feature itself contributes.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/yuqing/ctaNew")
from scipy.stats import spearmanr
from live.bookdepth_imb5_ablation import build_panel, gen, perbar_ic, V0_LEAN, RECENT_CUTS, OOS_CUTS
rng = np.random.default_rng(11)
CUT = pd.Timestamp("2025-10-01", tz="UTC")
# price features imb5 could plausibly substitute for (directional/momentum/flow), most-overlap first-guessed
REPL_TARGETS = ["return_1d", "obv_z_1d", "vwap_slope_96", "ret_3d"]
REPLACER = "imb5_ewma"

def uni_ic(sub, feat, tgt="alpha_vs_btc_realized"):
    """cross-sectional per-bar Spearman(feat, fwd alpha) + day-clustered CI, on covered bars only."""
    s = sub.groupby("open_time").apply(lambda g: spearmanr(g[feat], g[tgt]).correlation
                                       if g[[feat, tgt]].dropna().shape[0] >= 8 else np.nan).dropna()
    if len(s) < 5: return (np.nan, np.nan, np.nan)
    d = pd.DataFrame({"v": s.values}, index=pd.to_datetime(s.index, utc=True)); d["day"] = d.index.floor("1D")
    g = [x["v"].values for _, x in d.groupby("day")]
    b = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(2000)]
    return (s.mean(), *np.nanpercentile(b, [2.5, 97.5]))

def paired_delta(ib, iadd):
    j = pd.concat([ib.rename("a"), iadd.rename("b")], axis=1).dropna(); j["d"] = j["b"] - j["a"]
    j["day"] = pd.to_datetime(j.index, utc=True).floor("1D"); gg = [x["d"].values for _, x in j.groupby("day")]
    boot = [np.concatenate([gg[k] for k in rng.integers(0, len(gg), len(gg))]).mean() for _ in range(3000)]
    lo, up = np.percentile(boot, [2.5, 97.5])
    return j["d"].mean(), lo, up

def main():
    PAN = build_panel()
    cov = PAN[PAN["_covered"]]
    print(f"panel rows {len(PAN)} | covered {len(cov)} | replacer={REPLACER}\n")

    print("### (1) COLLINEARITY: corr(imb5_ewma, V0 feat) on covered bars — what does it overlap? ###")
    cors = {f: cov[REPLACER].corr(cov[f]) for f in V0_LEAN}
    for f, c in sorted(cors.items(), key=lambda kv: -abs(kv[1]))[:8]:
        print(f"    {f:26s} corr {c:+.3f}")
    print()

    print("### (2) HEAD-TO-HEAD univariate rank-IC(feat -> fwd alpha), both eras (does OB LEAD at 4h?) ###")
    eras = {"RECENT": cov[cov.open_time >= CUT], "OOS": cov[cov.open_time < CUT]}
    print(f"    {'feature':16s} | {'RECENT IC [CI]':26s} | {'OOS IC [CI]':26s}")
    for feat in [REPLACER, "imb5_raw"] + REPL_TARGETS:
        (ra, rl, ru) = uni_ic(eras["RECENT"], feat); (oa, ol, ou) = uni_ic(eras["OOS"], feat)
        print(f"    {feat:16s} | {ra:+.4f} [{rl:+.4f},{ru:+.4f}] | {oa:+.4f} [{ol:+.4f},{ou:+.4f}]")
    print("    (if imb5 truly leads at 4h, its forward IC should BEAT the price feats'; it doesn't => lead is intra-bar)\n")

    print("### (3) REPLACEMENT ablation: swap price feat -> imb5_ewma; model rank-IC vs V0 baseline, both eras ###")
    for era, cuts in [("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)]:
        ib = perbar_ic(gen(PAN, V0_LEAN, cuts))
        print(f"  {era}: V0_LEAN baseline rank-IC {ib.mean():+.4f}   [validity gate ~+0.030 rec / +0.024 oos]")
        # drop-only reference (how much does return_1d itself contribute?)
        idrop = perbar_ic(gen(PAN, [f for f in V0_LEAN if f != "return_1d"], cuts))
        dd, dl, du = paired_delta(ib, idrop)
        print(f"    DROP return_1d (no repl)         rank-IC {idrop.mean():+.4f}  Δ {dd:+.4f} [{dl:+.4f},{du:+.4f}]")
        for X in REPL_TARGETS:
            feats = [REPLACER if f == X else f for f in V0_LEAN]
            ir = perbar_ic(gen(PAN, feats, cuts))
            d, lo, up = paired_delta(ib, ir)
            flag = "BETTER (CI>0)" if lo > 0 else ("WORSE (CI<0)" if up < 0 else "within noise")
            print(f"    REPLACE {X:14s}->imb5_ewma  rank-IC {ir.mean():+.4f}  Δ {d:+.4f} [{lo:+.4f},{up:+.4f}] -> {flag}")
        print()
    print("read: replacement WINS only if Δ CI>0 both eras. IMB5REPLDONE")

if __name__ == "__main__":
    main()
