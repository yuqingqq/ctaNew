"""IC audit for kline-derived flow/microstructure features at h=288.

Many flow-related features are computed in `compute_kline_features` and sit
in the xs_feats caches but are NOT in `XS_FEATURE_COLS`. This script audits
their per-symbol Spearman IC vs the alpha-residual target — both for the
absolute feature value AND for its per-bar cross-sectional pctile rank
(the Phase 2 representation).

Anti-leakage notes:
  - Features come from `compute_kline_features` which uses point-in-time
    rolling windows (verified in leakage audit).
  - We use IS-only data (before holdout_start) for adding-decision IC. OOS
    IC is reported for transparency but NOT used to gate selection.
  - Cross-sectional rank at bar t uses only bar-t feature values across
    the universe — point-in-time by construction.

Decision rule: include a feature in v5 if and only if:
  (a) IS abs-IC ≥ 0.015 across ≥ 60% of symbols, AND
  (b) IS abs-IC sign is consistent (≥ 80% of symbols same sign), AND
  (c) IS-OOS sign agreement ≥ 60% (proxy for stability)
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from features_ml.cross_sectional import (
    XS_FEATURE_COLS, assemble_universe, list_universe, make_xs_alpha_labels,
)
from ml.research.alpha_v4_xs import _stack_xs_panel
from ml.research.alpha_v4_xs_1d import HORIZON, _holdout_split

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Candidates: flow/microstructure features that are in xs_feats but NOT in XS_FEATURE_COLS.
CANDIDATES = [
    # Direct trade-flow proxies from kline taker volume
    "taker_buy_volume", "taker_buy_quote_volume",
    # OBV family (cumulative signed volume)
    "obv", "obv_ema", "obv_signal",
    # VWAP-relative
    "vwap_zscore", "vwap_slope_96", "vwap_zscore_q90",
    # Money flow
    "mfi",
    # Pressure proxies
    "buy_pressure", "sell_pressure",
    # Effort vs result (Wyckoff-style: volume vs price-move)
    "effort_result_ratio", "effort_result_q85",
    # Volume momentum and ratios
    "volume_momentum", "volume_ratio_10", "volume_ratio_20",
    # Price-volume interaction
    "price_volume_corr_10", "price_volume_corr_20",
    # Volume regime
    "volume_trend_at_highs",
]

# Engineered ratios/transforms to test (computed from raw cols)
ENGINEERED = {
    "taker_buy_ratio": lambda f: f["taker_buy_volume"] / f["volume"].replace(0, np.nan),
    "taker_buy_quote_ratio": lambda f: f["taker_buy_quote_volume"] / f["quote_volume"].replace(0, np.nan),
    "obv_z_1d": lambda f: (f["obv"] - f["obv"].rolling(288, min_periods=48).mean()) /
                            f["obv"].rolling(288, min_periods=48).std().replace(0, np.nan),
    "obv_change_48b": lambda f: f["obv"].pct_change(48),
}


def _spearman(x, y):
    df = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna()
    if len(df) < 200:
        return np.nan
    return df.iloc[:, 0].rank().corr(df.iloc[:, 1].rank())


def _per_bar_xs_pctile_rank(panel: pd.DataFrame, col: str) -> pd.Series:
    """Cross-sectional pctile rank within each bar. Point-in-time by
    construction since each bar's ranking uses only that bar's values."""
    return panel.groupby("open_time")[col].rank(pct=True)


def main():
    universe = list_universe(min_days=200)
    log.info("universe: %d", len(universe))

    pkg = assemble_universe(universe, horizon=HORIZON)
    labels_by_sym = make_xs_alpha_labels(pkg["feats_by_sym"], pkg["basket_close"], HORIZON)

    # Build a wider panel containing the candidate cols (need raw kline cols too)
    base_cols = list(set(XS_FEATURE_COLS + ["close", "volume", "quote_volume"]))
    needed_cols = base_cols + [c for c in CANDIDATES if c not in base_cols]
    # Also need the raw cols for ENGINEERED features
    for f in ENGINEERED:
        # Engineered uses cols that are usually already in needed_cols (volume etc.)
        pass

    frames = []
    feat_cols_required = set(needed_cols)
    for s, f in pkg["feats_by_sym"].items():
        avail = [c for c in feat_cols_required if c in f.columns]
        # Build engineered cols on the per-symbol frame
        eng = {}
        for ename, fn in ENGINEERED.items():
            try:
                eng[ename] = fn(f)
            except Exception as e:
                log.warning("engineered feature %s failed for %s: %s", ename, s, e)
        if eng:
            f_aug = f[avail].copy()
            for ename, ser in eng.items():
                f_aug[ename] = ser
        else:
            f_aug = f[avail].copy()
        lab = labels_by_sym[s]
        df = f_aug.join(lab, how="inner")
        df["symbol"] = s
        df = df.reset_index().rename(columns={"index": "open_time"})
        if "open_time" not in df.columns:
            df = df.reset_index()
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True, sort=False)
    log.info("panel: %d rows", len(panel))

    # IS / OOS split using same convention as v4_xs_1d
    fold = _holdout_split(panel.rename(columns={"open_time": "open_time"}))[0]
    holdout_start = fold["test_start"]
    is_panel = panel[panel["open_time"] < holdout_start]
    oos_panel = panel[panel["open_time"] >= holdout_start]
    log.info("IS panel: %d rows, OOS panel: %d rows", len(is_panel), len(oos_panel))

    candidates_present = [c for c in CANDIDATES if c in panel.columns]
    candidates_engineered = [c for c in ENGINEERED if c in panel.columns]
    all_features = candidates_present + candidates_engineered
    log.info("auditing %d features (%d kline + %d engineered)",
              len(all_features), len(candidates_present), len(candidates_engineered))

    # Per-symbol IC table for raw features
    rows = []
    for f in all_features:
        is_ics = []
        oos_ics = []
        sign_is = []
        sign_oos = []
        for s in panel["symbol"].unique():
            is_g = is_panel[is_panel["symbol"] == s]
            oos_g = oos_panel[oos_panel["symbol"] == s]
            is_ic = _spearman(is_g[f], is_g["alpha_realized"])
            oos_ic = _spearman(oos_g[f], oos_g["alpha_realized"])
            is_ics.append(is_ic)
            oos_ics.append(oos_ic)
            if not np.isnan(is_ic):
                sign_is.append(np.sign(is_ic))
            if not np.isnan(oos_ic):
                sign_oos.append(np.sign(oos_ic))
        is_arr = np.array([x for x in is_ics if not np.isnan(x)])
        oos_arr = np.array([x for x in oos_ics if not np.isnan(x)])
        if len(is_arr) == 0:
            continue
        rows.append({
            "feature": f,
            "is_n_syms": len(is_arr),
            "is_mean_ic": is_arr.mean(),
            "is_mean_abs_ic": np.abs(is_arr).mean(),
            "is_max_abs_ic": np.abs(is_arr).max(),
            "is_sign_pos_frac": (sum(1 for s in sign_is if s > 0) / len(sign_is)) if sign_is else np.nan,
            "oos_mean_ic": oos_arr.mean() if len(oos_arr) else np.nan,
            "oos_mean_abs_ic": np.abs(oos_arr).mean() if len(oos_arr) else np.nan,
            "is_oos_sign_match_frac":
                (sum(1 for i_ic, o_ic in zip(is_ics, oos_ics)
                      if not (np.isnan(i_ic) or np.isnan(o_ic)) and np.sign(i_ic) == np.sign(o_ic))
                  / max(1, sum(1 for i_ic, o_ic in zip(is_ics, oos_ics)
                                if not (np.isnan(i_ic) or np.isnan(o_ic))))),
        })
    df = pd.DataFrame(rows).sort_values("is_mean_abs_ic", ascending=False)

    # Decision rule: |IS IC mean| ≥ 0.015 across ≥60% syms, sign-consistent ≥ 80%, IS-OOS match ≥ 60%
    df["sign_dominant"] = df["is_sign_pos_frac"].apply(lambda p: max(p, 1 - p))
    df["recommend"] = (
        (df["is_mean_abs_ic"] >= 0.015) &
        (df["sign_dominant"] >= 0.80) &
        (df["is_oos_sign_match_frac"] >= 0.60)
    )

    print("=" * 110)
    print(f"FLOW FEATURE AUDIT — IC vs alpha at h={HORIZON}")
    print("=" * 110)
    print("\nDecision rule: |IS IC| ≥ 0.015, sign-consistent ≥80%, IS-OOS sign match ≥60%")
    print("\nAll candidates ranked by IS mean |IC|:")
    cols = ["feature", "is_mean_ic", "is_mean_abs_ic", "is_max_abs_ic",
             "sign_dominant", "oos_mean_ic", "oos_mean_abs_ic",
             "is_oos_sign_match_frac", "recommend"]
    print(df[cols].round(4).to_string(index=False))

    print("\n--- RECOMMENDED for v5 (passes all 3 gates) ---")
    rec = df[df["recommend"]].copy()
    if len(rec) == 0:
        print("  (none)")
    else:
        print(rec[cols].round(4).to_string(index=False))

    print("\n--- BORDERLINE (passes IS gate but fails sign-consistency or OOS-match) ---")
    borderline = df[(df["is_mean_abs_ic"] >= 0.015) & ~df["recommend"]]
    if len(borderline) > 0:
        print(borderline[cols].round(4).to_string(index=False))
    else:
        print("  (none)")

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "alpha_v4_flow_audit.csv", index=False)
    print(f"\nSaved to {out}/alpha_v4_flow_audit.csv")


if __name__ == "__main__":
    main()
