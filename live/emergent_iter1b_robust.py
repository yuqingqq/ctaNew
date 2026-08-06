"""iter1b: is the OOS->REC manifold DRIFT real, or a composition artifact?

iter1 found the cross-symbol structure de-universalizes and gains dimension in RECENT
(+0.96->+0.80 similarity, effdim 2.89->3.36). But RECENT has 177 syms vs OOS 166 (newly
listed thin names) and 6x more gap-recovery. Control for composition: re-run the structure
pass on (a) the FIXED both-era set, and (b) MATURE names (long OOS history). If the drift
persists on the mature fixed set, it is a genuine regime change in the joint structure; if it
vanishes, "drift" was just newly-listed thin symbols entering the recent cross-section.

Run:  python3 -m live.emergent_iter1b_robust
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.emergent_iter1_manifold import MIN_ROWS, per_symbol_structure

PERSYM = "/home/yuqing/ctaNew/live/emergent_iter1_persym.parquet"


def main() -> None:
    R = pd.read_parquet(PERSYM)
    both = R[(R["n_OOS"] >= MIN_ROWS) & (R["n_REC"] >= MIN_ROWS)]
    fixed = set(both["sym"])
    mature = set(both.loc[both["n_OOS"] >= 100_000, "sym"])  # long OOS history
    print(f"full: {len(R)} syms | both-era fixed: {len(fixed)} | mature(nOOS>=100k): {len(mature)}",
          flush=True)
    print("\nComparison target (iter1 full-177 REC): universality +0.80, pooled effdim 3.36\n",
          flush=True)
    per_symbol_structure(syms=fixed, tag="FIXED both-era (drop REC-only newborns)")
    per_symbol_structure(syms=mature, tag="MATURE (nOOS>=100k)")


if __name__ == "__main__":
    main()
