"""Audit the new alpha-tailored features (features_ml/alpha_features.py).

For every new feature, compute IC vs alpha target (β=1d) per symbol on
in-sample data only. Compare against the existing best base + cross features.

Outputs:
  - Per-symbol ranked IC table
  - Cross-symbol average |IC| (consistency)
  - Recommended feature set for v3 model
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from features_ml.alpha_features import add_alpha_features, alpha_feature_columns
from ml.research.alpha_feature_audit import _build_alpha_target, _ic, BETA_WINDOWS, REF_OF
from ml.research.trend_pooled_v2 import _build_symbol_features

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HORIZON = 48
HOLDOUT_DAYS = 90


def _audit_symbol(symbol: str, holdout_start: pd.Timestamp) -> pd.DataFrame:
    feats, _ = _build_symbol_features(symbol)
    ref_feats, _ = _build_symbol_features(REF_OF[symbol])
    ref_label = REF_OF[symbol][:3].lower()
    enriched = add_alpha_features(feats, ref_feats, ref_label=ref_label)
    new_cols = alpha_feature_columns(ref_label)

    is_enriched = enriched[enriched.index < holdout_start]
    is_ref_close = ref_feats["close"][ref_feats.index < holdout_start]
    comps = _build_alpha_target(is_enriched["close"], is_ref_close,
                                  HORIZON, BETA_WINDOWS["1d"])
    alpha_target = comps["alpha_target"]
    raw_target = comps["raw_target"]

    rows = []
    for col in new_cols:
        if col not in is_enriched.columns:
            continue
        x = is_enriched[col]
        ic_alpha_p, ic_alpha_s = _ic(x, alpha_target)
        ic_raw_p, ic_raw_s = _ic(x, raw_target)
        rows.append({
            "symbol": symbol,
            "feature": col.replace(f"_vs_{ref_label}", "_vs_ref")
                          .replace(f"{ref_label}_", "ref_"),
            "raw_col": col,
            "ic_alpha_pearson": ic_alpha_p, "ic_alpha_spearman": ic_alpha_s,
            "ic_raw_spearman": ic_raw_s,
            "lift_spearman": ic_alpha_s - ic_raw_s,
        })
    return pd.DataFrame(rows)


def main():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    feats0, _ = _build_symbol_features(symbols[0])
    holdout_start = feats0.index.max() - pd.Timedelta(days=HOLDOUT_DAYS)

    print("=" * 70)
    print("ALPHA-TAILORED FEATURE AUDIT (in-sample, β=1d)")
    print("=" * 70)

    audits = [_audit_symbol(s, holdout_start) for s in symbols]
    df = pd.concat(audits, ignore_index=True)

    print("\n--- Per-symbol IC vs alpha (Spearman) ---")
    for s in symbols:
        sub = df[df["symbol"] == s].set_index("feature")[
            ["ic_alpha_spearman", "ic_raw_spearman", "lift_spearman"]
        ].sort_values("ic_alpha_spearman", key=abs, ascending=False)
        print(f"\n{s}:")
        print(sub.round(4).to_string())

    print("\n--- Cross-symbol consistency (|IC| ranked) ---")
    g = df.groupby("feature").agg(
        ic_alpha_mean=("ic_alpha_spearman", "mean"),
        ic_alpha_abs_mean=("ic_alpha_spearman", lambda x: x.abs().mean()),
        ic_alpha_min_abs=("ic_alpha_spearman", lambda x: x.abs().min()),
        lift_abs_mean=("lift_spearman", lambda x: x.abs().mean()),
        sign_consistent=("ic_alpha_spearman",
                          lambda x: int((np.sign(x).value_counts().iloc[0]
                                         if len(x) > 0 else 0) == len(x))),
    ).round(4).sort_values("ic_alpha_abs_mean", ascending=False)
    print(g.to_string())

    out_dir = Path("outputs")
    df.to_csv(out_dir / "alpha_v3_feature_audit.csv", index=False)
    print(f"\nSaved to {out_dir}/alpha_v3_feature_audit.csv")


if __name__ == "__main__":
    main()
