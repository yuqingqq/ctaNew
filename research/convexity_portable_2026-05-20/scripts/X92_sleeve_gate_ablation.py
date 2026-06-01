"""X92 — Ablate the sleeve's conv_gate (and combine with TOP_N) to close the
held-book(+0.83) vs sleeve(+0.12) gap on portable V0.

Hypothesis: the conv_gate (skip cycles where dispersion < 30th pctile, GATE_PCTILE=0.30)
GAPS the overlapping 24h book — the SAME gap-artifact that broke H3f. It was tuned for
the vBTC strategy but may hurt the portable V0 over 3yr. Disabling it (GATE_PCTILE→0,
never skip) should recover performance toward the held-book.

Sweep GATE_PCTILE ∈ {0.30 (default), 0.15, 0.0 (off)} × TOP_N ∈ {15, 25} on V0 3yr.
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
    print("=== X92 sleeve conv_gate ablation (V0 3yr) ===\n", flush=True)
    pred=RCACHE/"x70_v0_3yr_preds.parquet"
    print(f"defaults: GATE_PCTILE={P.GATE_PCTILE}, TOP_N={P.TOP_N}")
    print(f"ref: held-book V0 +0.83, sleeve default +0.12\n")
    print(f"  {'GATE_PCTILE':<12}{'TOP_N':<7}{'Sharpe':>9}{'folds+':>8}{'conc':>7}{'totPnL':>9}")
    for gp in [0.30, 0.15, 0.0]:
        for tn in [15, 25]:
            P.GATE_PCTILE=gp; P.TOP_N=tn
            m=x6.run_sleeve_on_preds(pred, f"x92_g{gp}_t{tn}")
            sh=m.get('sharpe','?'); sh_s=f"{sh:+.2f}" if isinstance(sh,(int,float)) else str(sh)
            print(f"  {gp:<12.2f}{tn:<7}{sh_s:>9}{str(m.get('folds_pos','?')):>8}{str(m.get('concentration','?')):>7}{str(m.get('totPnL','?')):>9}", flush=True)
    print(f"\nVERDICT: if GATE_PCTILE=0 (gate off) >> 0.30 → conv_gate gaps the book & hurts portable V0.")
    print(f"Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
