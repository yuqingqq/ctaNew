"""Phase AH V3.1 sleeve replay on R2a predictions (WINNER_21 + rvol_7d + ret_3d + btc_rvol_7d).

Wrapper that imports phase_ah_sleeve's main but overrides APD_PATH and OUT
dir. Production baseline (same script) is Sharpe +2.23 on outputs/vBTC_audit_panel/all_predictions.parquet.
We rerun identical sleeve machinery on R2a's all_predictions_R2a.parquet
and report a per-fold Sharpe table to check whether any lift is concentrated
in a single fold (the Phase Q W23 failure mode).
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import phase_ah_sleeve as P
P.APD_PATH = REPO / "research/portable_alpha_2026-05-19/results/_cache/all_predictions_R2a.parquet"
P.OUT = REPO / "outputs/vBTC_sleeve_R2a"
P.OUT.mkdir(parents=True, exist_ok=True)
P.N_PLACEBO_SEEDS = 100

if __name__ == "__main__":
    P.main()
