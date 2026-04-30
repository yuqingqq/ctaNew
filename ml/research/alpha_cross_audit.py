"""Cross-asset feature audit for alpha prediction.

Question: Does adding the reference symbol's features (or relative features
like excess return / spread / beta / correlation) improve IC vs alpha?

For each (target symbol, candidate cross feature):
  - Compute IC(cross_feature_at_t, alpha_target_at_t)  in-sample only

Cross features tested:
  1. Relative features computed by features_ml.cross_asset:
       excess_ret_3, excess_ret_12, excess_ret_48
       beta_ref_1d, corr_ref_1d
       spread_log_vs_ref, spread_zscore_1d_vs_ref, spread_zscore_7d_vs_ref
  2. Reference symbol's own features (ref_atr_pct, ref_return_1d, ref_ema_slope_20_1h, etc.)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from features_ml.cross_asset import add_cross_asset_features
from ml.research.alpha_feature_audit import _build_alpha_target, _ic, BETA_WINDOWS, REF_OF
from ml.research.trend_pooled_v2 import _build_symbol_features

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HORIZON = 48
HOLDOUT_DAYS = 90

# Reference features to copy into target frame as covariates.
REF_FEATS_TO_BORROW = [
    "atr_pct", "return_1d", "ema_slope_20_1h", "realized_vol_1h",
    "atr_zscore_1d", "vpin", "tfi_smooth", "signed_volume",
]


def _build_cross_features(symbol: str, holdout_start: pd.Timestamp):
    feats, _ = _build_symbol_features(symbol)
    ref_feats, _ = _build_symbol_features(REF_OF[symbol])
    ref_label = REF_OF[symbol][:3].lower()
    enriched = add_cross_asset_features(feats, ref_feats, ref_label=ref_label)
    # Borrow ref features
    ref_aligned = ref_feats[REF_FEATS_TO_BORROW].reindex(enriched.index)
    ref_aligned = ref_aligned.add_prefix(f"{ref_label}_")
    enriched = enriched.join(ref_aligned)
    cross_cols = [c for c in enriched.columns if c not in feats.columns]
    is_enriched = enriched[enriched.index < holdout_start]
    is_ref = ref_feats[ref_feats.index < holdout_start]
    is_my_close = is_enriched["close"]

    # Build alpha target at β=1d (default)
    comps = _build_alpha_target(is_my_close, is_ref["close"], HORIZON, BETA_WINDOWS["1d"])
    alpha_target = comps["alpha_target"]
    raw_target = comps["raw_target"]

    rows = []
    for col in cross_cols:
        x = is_enriched[col]
        ic_alpha_p, ic_alpha_s = _ic(x, alpha_target)
        ic_raw_p, ic_raw_s = _ic(x, raw_target)
        rows.append({
            "symbol": symbol, "feature": col,
            "ic_alpha_pearson": ic_alpha_p, "ic_alpha_spearman": ic_alpha_s,
            "ic_raw_pearson": ic_raw_p, "ic_raw_spearman": ic_raw_s,
            "lift_spearman": ic_alpha_s - ic_raw_s,
        })
    return pd.DataFrame(rows)


def main():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    feats0, _ = _build_symbol_features(symbols[0])
    data_end = feats0.index.max()
    holdout_start = data_end - pd.Timedelta(days=HOLDOUT_DAYS)

    print("=" * 70)
    print("CROSS-ASSET FEATURE AUDIT (in-sample, β=1d alpha)")
    print("=" * 70)

    audits = [_build_cross_features(s, holdout_start) for s in symbols]
    df = pd.concat(audits, ignore_index=True)

    print("\n--- Per-symbol cross-feature IC (Spearman) ---")
    for s in symbols:
        sub = df[df["symbol"] == s].set_index("feature")[
            ["ic_alpha_spearman", "ic_raw_spearman", "lift_spearman"]
        ].sort_values("ic_alpha_spearman", key=abs, ascending=False)
        print(f"\n{s}:")
        print(sub.round(4).to_string())

    print("\n--- Avg cross-feature IC (across symbols) ---")
    g = df.groupby("feature").agg(
        ic_alpha_mean=("ic_alpha_spearman", "mean"),
        ic_alpha_abs_mean=("ic_alpha_spearman", lambda x: x.abs().mean()),
        lift_mean=("lift_spearman", "mean"),
        lift_abs_mean=("lift_spearman", lambda x: x.abs().mean()),
    ).round(4).sort_values("ic_alpha_abs_mean", ascending=False)
    print(g.to_string())

    out_dir = Path("outputs")
    df.to_csv(out_dir / "alpha_cross_feature_audit.csv", index=False)
    print(f"\nSaved to {out_dir}/alpha_cross_feature_audit.csv")


if __name__ == "__main__":
    main()
