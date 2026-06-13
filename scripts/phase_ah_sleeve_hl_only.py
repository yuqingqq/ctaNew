"""V3.1 sleeve replay on HL-tradeable 50-symbol subset.

Uses the V3.1 51-panel model predictions, but restricts the universe to the
50 HL-tradeable symbols (drops BTC, which isn't on HL). This isolates
'V3.1 model on the executable subset' from 'V3.1 trained on a different
universe' confounds.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import phase_ah_sleeve as P
P.APD_PATH = REPO / "research/convexity_portable_2026-05-20/results/_cache/apd_hl_only.parquet"
P.OUT = REPO / "outputs/vBTC_sleeve_hl_only"
P.OUT.mkdir(parents=True, exist_ok=True)
P.N_PLACEBO_SEEDS = 100

if __name__ == "__main__":
    P.main()
