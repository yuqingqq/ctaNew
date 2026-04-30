"""alpha_v3 restricted to BTC + ETH (drop SOL).

Diagnosis from alpha_v3_thr.py:
  - Per-symbol threshold q=0.95:
        BTC: alpha=+7.33, IC=+0.080, trigger=3.0%, net=-9.62 (cost dominated)
        ETH: alpha=+9.34, IC=+0.064, trigger=16.5%, net=-0.91 (close to break-even)
        SOL: alpha=-4.83, IC=+0.035, trigger=45.7%, net=-22.37 (regime shift; over-triggers)

  SOL's OOS prediction distribution is wider than its cal distribution, so its
  trigger rate breaks calibration. SOL alpha is also weakest (lowest IC).

This probe: train pooled on BTC + ETH only, evaluate on BTC + ETH only.
Tests whether a 2-symbol strategy on the symbols where alpha works is viable.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from ml.research.alpha_v3_thr import _run, _summary

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    btceth = ["BTCUSDT", "ETHUSDT"]

    print("=" * 70)
    print("v3 RESTRICTED TO BTC + ETH (drop SOL)")
    print("=" * 70)

    configs = [
        ("v3 BTCETH per-sym q=0.95", "v3", 0.95, True),
        ("v3 BTCETH per-sym q=0.97", "v3", 0.97, True),
        ("v3 BTCETH per-sym q=0.99", "v3", 0.99, True),
    ]

    print("\n--- WALK-FORWARD ---")
    wf_summaries = []
    for label, ver, q, ps in configs:
        df, _ = _run(btceth, version=ver, mode="walkforward", threshold_q=q,
                       per_symbol_threshold=ps)
        _summary(df, f"WF, {label}")
        if not df.empty:
            wf_summaries.append({
                "config": label, "phase": "WF",
                "n": int(df["n"].sum()),
                "trig_pct": df["trigger_rate"].mean() * 100,
                "alpha": df["alpha_pnl_bps"].mean(),
                "net": df["net_bps"].mean(),
                "ic_avg": df["ic_pred_alpha"].mean(),
                "folds_pos": int((df["net_bps"] > 0).sum()),
                "folds": len(df),
            })

    print("\n--- OOS HOLDOUT ---")
    oos_summaries = []
    for label, ver, q, ps in configs:
        df, _ = _run(btceth, version=ver, mode="oos_holdout",
                       threshold_q=q, per_symbol_threshold=ps)
        _summary(df, f"OOS, {label}")
        if not df.empty:
            oos_summaries.append({
                "config": label, "phase": "OOS",
                "n": int(df["n"].sum()),
                "trig_pct": df["trigger_rate"].mean() * 100,
                "alpha": df["alpha_pnl_bps"].mean(),
                "net": df["net_bps"].mean(),
                "ic_avg": df["ic_pred_alpha"].mean(),
                "folds_pos": int((df["net_bps"] > 0).sum()),
                "folds": len(df),
            })

    print("\n" + "=" * 70)
    print("SUMMARY (BTC + ETH only)")
    print("=" * 70)
    summary = pd.DataFrame(wf_summaries + oos_summaries)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
