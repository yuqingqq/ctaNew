"""X91 — Does the V3.1 sleeve's rolling-IC universe selection (TOP_N=15) HURT?

Held-book (all 44 syms) net = +1.89 vs V3.1 sleeve (TOP_N=15 rolling-IC) +1.19 — a
+0.7 gap. Prime suspect: the rolling-IC universe selection (select top-15 by past IC,
then K=3 from those). Prior vBTC finding: IC-selector noise-dominated, ALL-eligible
beats top-15. Test: sweep sleeve TOP_N ∈ {15, 25, 35, all} on V0 3yr preds.
If larger/all universe beats TOP_N=15 → the selector hurts (drop it).
"""
from __future__ import annotations
import sys, importlib.util, time
from pathlib import Path

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scripts"))
RCACHE = REPO/"research/convexity_portable_2026-05-20/results/_cache"
spec = importlib.util.spec_from_file_location("x6", REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
import phase_ah_sleeve as P


def main():
    t0=time.time()
    print("=== X91 sleeve universe-size sweep (does rolling-IC TOP_N hurt?) ===\n", flush=True)
    pred = RCACHE/"x70_v0_3yr_preds.parquet"
    print(f"V0 3yr preds. Default TOP_N={P.TOP_N}\n")
    print(f"  {'TOP_N':<8}{'Sharpe':>9}{'folds+':>9}{'conc':>8}{'totPnL':>10}")
    for tn in [15, 25, 35, None]:
        P.TOP_N = tn
        m = x6.run_sleeve_on_preds(pred, f"x91_tn{tn}")
        sh=m.get('sharpe','?'); fp=m.get('folds_pos','?'); conc=m.get('concentration','?'); pnl=m.get('totPnL','?')
        sh_s=f"{sh:+.2f}" if isinstance(sh,(int,float)) else str(sh)
        print(f"  {str(tn):<8}{sh_s:>9}{str(fp):>9}{str(conc):>8}{str(pnl):>10}", flush=True)
    print(f"\nVERDICT: if TOP_N=None(all) or larger > 15 → rolling-IC selection HURTS, drop it.")
    print(f"(held-book all-syms was +1.89 vs sleeve TOP_N=15 +1.19)")
    print(f"Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
