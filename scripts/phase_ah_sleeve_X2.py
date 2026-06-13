"""V3.1 sleeve on X2 predictions (LGBM with sym_id on 110-panel, unclipped target_A).

Tests whether the V3.1 sleeve machinery applied to portable-LGBM predictions
on 110-panel matches V3.1's HL-50 Sharpe of +3.00.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import phase_ah_sleeve as P
P.APD_PATH = REPO / "research/convexity_portable_2026-05-20/results/_cache/all_predictions_X2_lgbm.parquet"
P.OUT = REPO / "outputs/vBTC_sleeve_X2"
P.OUT.mkdir(parents=True, exist_ok=True)
P.N_PLACEBO_SEEDS = 100

if __name__ == "__main__":
    P.main()
