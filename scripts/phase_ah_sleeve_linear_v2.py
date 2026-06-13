"""V3.1 sleeve on linear V2 predictions (step34, 50-sym HL-tradeable subset).

Apples-to-apples comparison vs LGBM (V3.1) on HL-50 which gave Sharpe +3.00.
Linear V2 from linear_model arc Step 34 (NaN-fixed Ridge, R3_BTC + V2 features,
heavy-tail rank-transform fold-0, BTC-frame target_A).
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import phase_ah_sleeve as P
P.APD_PATH = REPO / "linear_model/results/step34_v1_fixed/v2_fixed_predictions.parquet"
P.OUT = REPO / "outputs/vBTC_sleeve_linear_v2"
P.OUT.mkdir(parents=True, exist_ok=True)
P.N_PLACEBO_SEEDS = 100

if __name__ == "__main__":
    P.main()
