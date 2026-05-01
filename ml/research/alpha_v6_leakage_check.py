"""Empirical leakage check for v6, memory-lean version.

Approach: load only what we need (no full assemble_universe with funding
side-effects). Reuse xs_feats caches directly.

Tests:
1. Forward-peek shift: for each feature, compute IC at shift=+1, 0, -1.
   A clean PIT feature should have similar magnitudes; a leaky feature
   should show larger |IC| at shift=-1 (using "future" feature value).
2. Sanity positive control: a feature = alpha[t+1] should yield IC ≈ 1.
3. xs_rank manual recomputation check: per-bar pctile rank should be
   reproducible from bar-t cross-section alone.
"""
from __future__ import annotations

import gc
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from features_ml.cross_sectional import (
    XS_BASE_FEATURES, XS_CROSS_FEATURES, XS_FLOW_FEATURES, XS_RANK_FEATURES,
    XS_FEATURE_COLS_V6_CLEAN, XS_RANK_SOURCES,
    add_basket_features, add_engineered_flow_features, add_xs_rank_features,
    build_basket, build_kline_features, list_universe, make_xs_alpha_labels,
)
from ml.research.alpha_v4_xs_1d import HORIZON, _holdout_split

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# FEATURE_SET=v6 (default) or v6_clean — chooses which feature list to audit.
FEATURE_SET = os.environ.get("FEATURE_SET", "v6").lower()
if FEATURE_SET == "v6_clean":
    V6_FEATURES = [c for c in XS_FEATURE_COLS_V6_CLEAN if c != "sym_id"]
else:
    V6_FEATURES = (XS_BASE_FEATURES + XS_CROSS_FEATURES + XS_FLOW_FEATURES + XS_RANK_FEATURES)
log.info("Auditing %d features (FEATURE_SET=%s)", len(V6_FEATURES), FEATURE_SET)


def _spearman(x, y):
    df = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna()
    if len(df) < 200:
        return np.nan
    return df.iloc[:, 0].rank().corr(df.iloc[:, 1].rank())


def _build_v6_panel_lean():
    """Bypass assemble_universe (which now pulls funding); build minimally."""
    universe = list_universe(min_days=200)
    log.info("universe: %d", len(universe))
    feats_by_sym = {}
    for s in universe:
        f = build_kline_features(s)
        if not f.empty:
            feats_by_sym[s] = f
    closes = pd.DataFrame({s: f["close"] for s, f in feats_by_sym.items()}).sort_index()
    basket_ret, basket_close = build_basket(closes)
    sym_to_id = {s: i for i, s in enumerate(sorted(feats_by_sym.keys()))}

    enriched = {}
    for s, f in feats_by_sym.items():
        f = f.reindex(closes.index)
        f = add_basket_features(f, basket_close, basket_ret)
        f = add_engineered_flow_features(f)
        f["sym_id"] = sym_to_id[s]
        enriched[s] = f

    labels = make_xs_alpha_labels(enriched, basket_close, HORIZON)
    rank_cols = XS_RANK_FEATURES
    src_cols = list({s for s, _ in XS_RANK_SOURCES.items()})
    needed = list(set(V6_FEATURES + ["sym_id", "autocorr_pctile_7d"] + src_cols) - set(rank_cols))

    frames = []
    for s, f in enriched.items():
        avail = [c for c in needed if c in f.columns]
        df = f[avail].join(labels[s], how="inner")
        df["symbol"] = s
        df = df.reset_index().rename(columns={"index": "open_time"})
        frames.append(df)
        del f
    panel = pd.concat(frames, ignore_index=True, sort=False)
    del frames, enriched, feats_by_sym
    gc.collect()
    panel = add_xs_rank_features(panel, sources=XS_RANK_SOURCES)
    panel = panel.dropna(subset=rank_cols + ["autocorr_pctile_7d"])
    return panel


def main():
    panel = _build_v6_panel_lean()
    log.info("panel: %d rows", len(panel))

    fold = _holdout_split(panel)[0]
    is_panel = panel[panel["open_time"] < fold["test_start"]].copy()
    del panel
    gc.collect()
    log.info("IS panel: %d rows", len(is_panel))

    print("=" * 100)
    print("TEST 1: Forward-peek shift (per-symbol IC, IS only)")
    print("=" * 100)
    print(f"\n  Mean per-symbol IC of feature(shifted by N) vs alpha_realized.")
    print(f"  shift=+1 = lagged feature (more conservative).")
    print(f"  shift=-1 = use future feature value (would inflate IC if feature was already too current).")
    print(f"  For PIT features: |IC(-1)| ≈ |IC(0)|. For leaky features: |IC(-1)| >> |IC(0)|.\n")
    print(f"  {'feature':<32} {'IC(+1)':>9} {'IC(0)':>9} {'IC(-1)':>9} {'Δfwd':>9} {'verdict':<25}")
    suspicious = []
    for f in V6_FEATURES:
        if f not in is_panel.columns:
            print(f"  {f:<32} (column missing)")
            continue
        ics_per_shift = {}
        for shift in (+1, 0, -1):
            ics = []
            for s, g in is_panel.groupby("symbol"):
                if len(g) < 500:
                    continue
                x = g[f].shift(shift)
                ic = _spearman(x, g["alpha_realized"])
                if not np.isnan(ic):
                    ics.append(ic)
            ics_per_shift[shift] = np.mean(ics) if ics else np.nan
        ic_lag = ics_per_shift[+1]
        ic_now = ics_per_shift[0]
        ic_lead = ics_per_shift[-1]
        delta = abs(ic_lead) - abs(ic_now)
        if abs(ic_now) > 0.005 and abs(ic_lead) > abs(ic_now) * 1.3 and delta > 0.005:
            verdict = "SUSPICIOUS"; suspicious.append(f)
        elif delta > 0.01:
            verdict = "SUSPICIOUS"; suspicious.append(f)
        else:
            verdict = "ok"
        print(f"  {f:<32} {ic_lag:>+9.4f} {ic_now:>+9.4f} {ic_lead:>+9.4f} {delta:>+9.4f} {verdict:<25}")

    if suspicious:
        print(f"\n  ⚠️ Flagged: {suspicious}")
    else:
        print(f"\n  ✓ No features flagged. All forward-shift Δ|IC| ≤ 0.01.")

    print("\n" + "=" * 100)
    print("TEST 2: Sanity — deliberately-leaky feature (alpha[t+1]) should be flagged")
    print("=" * 100)
    is_panel["leaky_test"] = is_panel.groupby("symbol")["alpha_realized"].shift(-1)
    ics = []
    for s, g in is_panel.groupby("symbol"):
        if len(g) < 500:
            continue
        ic = _spearman(g["leaky_test"], g["alpha_realized"])
        if not np.isnan(ic):
            ics.append(ic)
    ic_leaky = np.mean(ics) if ics else np.nan
    print(f"  IC(alpha[t+1] vs alpha[t]) = {ic_leaky:+.4f}")
    if abs(ic_leaky) > 0.5:
        print(f"  ✓ Methodology validates: known-leaky feature has very high IC")
    else:
        print(f"  ⚠️ Known-leaky feature's IC is unexpectedly low (alpha may have low autocorrelation)")

    print("\n" + "=" * 100)
    print("TEST 3: xs_rank manual recomputation check")
    print("=" * 100)
    sample_t = is_panel["open_time"].iloc[len(is_panel) // 2]
    g = is_panel[is_panel["open_time"] == sample_t][["symbol", "return_1d", "return_1d_xs_rank"]].dropna()
    if len(g) >= 5:
        manual = g["return_1d"].rank(pct=True).to_numpy()
        stored = g["return_1d_xs_rank"].to_numpy()
        max_diff = np.abs(manual - stored).max()
        print(f"  Sample bar {sample_t}, n_syms={len(g)}: max|manual_rank - stored_rank| = {max_diff:.6e}")
        if max_diff < 1e-9:
            print(f"  ✓ xs_rank reproducible from bar-t cross-section alone — PIT confirmed")
        else:
            print(f"  ⚠️ Mismatch — investigate xs_rank computation")
    else:
        print(f"  Sample bar had {len(g)} symbols; need ≥ 5")

    print("\n" + "=" * 100)
    print("TEST 4: Embargo + exit_time purging (OOS holdout fold)")
    print("=" * 100)
    print(f"  cal_end:    {fold['cal_end']}")
    print(f"  test_start: {fold['test_start']}")
    gap = fold["test_start"] - fold["cal_end"]
    print(f"  gap: {gap}, embargo configured: {fold['embargo']}")
    if gap >= fold["embargo"]:
        print(f"  ✓ cal-to-test gap ≥ embargo")
    else:
        print(f"  ⚠️ gap < embargo")


if __name__ == "__main__":
    main()
