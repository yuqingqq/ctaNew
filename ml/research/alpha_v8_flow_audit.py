"""IC audit for true microstructure features (Phase 3 outputs).

After scripts/build_aggtrade_features.py has produced flow_<SYMBOL>.parquet
caches, this script audits each feature's IC vs alpha at h=288.

Anti-leakage:
  - Per-bar features (buy_volume, tfi, vwap, kyle_lambda) are computed only
    from trades within bar [t, t+5min) — no future info.
  - Rolling features (vpin, tfi_smooth, signed_volume_z) use left-aligned
    windows that include bar-t value, same convention as kline features.
  - Audit uses IS-only data (before holdout_start) for the gate decision.
  - Cross-sectional rank versions (computed below) use only bar-t universe.

Decision rule (same as flow / funding audits):
  recommend if (a) IS abs-IC ≥ 0.015, (b) sign-consistent ≥80% of symbols,
  (c) IS-OOS sign match ≥60%.

Universe: top-10 most-liquid USDM perps. Other 15 v6 symbols don't have
aggTrades pulled — for v8 evaluation, the universe shrinks to these 10.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from features_ml.cross_sectional import (
    XS_FEATURE_COLS_V6, XS_RANK_SOURCES, add_xs_rank_features,
    assemble_universe, list_universe, make_xs_alpha_labels,
)
from ml.research.alpha_v4_xs import _stack_xs_panel
from ml.research.alpha_v4_xs_1d import HORIZON, _holdout_split

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE_DIR = Path("data/ml/cache")

# Trade-flow features to audit. All come from features_ml/trade_flow.py.
FLOW_CANDIDATES = [
    "buy_volume", "sell_volume", "signed_volume", "tfi",
    "buy_count", "sell_count", "aggressor_count_ratio",
    "avg_trade_size", "max_trade_size", "large_trade_volume",
    "large_trade_count", "vwap_dev_bps", "kyle_lambda",
    "vpin", "tfi_smooth", "signed_volume_z",
]

# Engineered transforms to also test (computed from raw features).
ENGINEERED = {
    "log_signed_vol_ratio": lambda f: np.log((f["buy_volume"] + 1) / (f["sell_volume"] + 1)),
    "tfi_z_1d": lambda f: ((f["tfi"] - f["tfi"].rolling(288, min_periods=48).mean())
                            / f["tfi"].rolling(288, min_periods=48).std().replace(0, np.nan)),
    "kyle_lambda_z_1d": lambda f: ((f["kyle_lambda"] - f["kyle_lambda"].rolling(288, min_periods=48).mean())
                                     / f["kyle_lambda"].rolling(288, min_periods=48).std().replace(0, np.nan)),
}


def _spearman(x, y):
    df = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna()
    if len(df) < 200:
        return np.nan
    return df.iloc[:, 0].rank().corr(df.iloc[:, 1].rank())


def _attach_flow_features(feats: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Load flow features for symbol, attach to feats by aligning on index."""
    cache = CACHE_DIR / f"flow_{symbol}.parquet"
    if not cache.exists():
        log.warning("[%s] no flow cache; skipping", symbol)
        return feats
    flow = pd.read_parquet(cache)
    if flow.index.tz is None:
        flow.index = flow.index.tz_localize("UTC")
    out = feats.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    avail = [c for c in FLOW_CANDIDATES if c in flow.columns]
    out = out.join(flow[avail], how="left")
    # Engineered
    for name, fn in ENGINEERED.items():
        try:
            out[name] = fn(out)
        except Exception as e:
            log.warning("[%s] engineered %s failed: %s", symbol, name, e)
    return out


def main():
    universe = list_universe(min_days=200)
    # Restrict to symbols with flow caches
    syms_with_flow = [s for s in universe if (CACHE_DIR / f"flow_{s}.parquet").exists()]
    log.info("universe: %d full, %d with flow features", len(universe), len(syms_with_flow))
    if len(syms_with_flow) < 5:
        log.error("Need ≥5 symbols with flow caches. Run pull_aggtrades + build_aggtrade_features first.")
        return

    # Build full v6 panel for context (kline-only features)
    pkg = assemble_universe(universe, horizon=HORIZON)
    labels_by_sym = make_xs_alpha_labels(pkg["feats_by_sym"], pkg["basket_close"], HORIZON)

    # Attach flow features only to symbols that have them
    augmented = {}
    for s, f in pkg["feats_by_sym"].items():
        if s in syms_with_flow:
            augmented[s] = _attach_flow_features(f, s)
        else:
            augmented[s] = f

    # Stack — only for the syms with flow (so audit is fair)
    cols = list({c for c in XS_FEATURE_COLS_V6 if c != "sym_id" and not c.endswith("_xs_rank")} | set(FLOW_CANDIDATES) | set(ENGINEERED))
    frames = []
    for s in syms_with_flow:
        f = augmented[s]
        lab = labels_by_sym[s]
        avail = [c for c in cols if c in f.columns]
        df = f[avail].join(lab, how="inner")
        df["symbol"] = s
        df = df.reset_index().rename(columns={"index": "open_time"})
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True, sort=False)
    log.info("panel (flow-only universe): %d rows, %d symbols", len(panel), len(syms_with_flow))

    fold = _holdout_split(panel)[0]
    is_panel = panel[panel["open_time"] < fold["test_start"]]
    oos_panel = panel[panel["open_time"] >= fold["test_start"]]
    log.info("IS: %d, OOS: %d", len(is_panel), len(oos_panel))

    flow_features = list(FLOW_CANDIDATES) + list(ENGINEERED.keys())
    rows = []
    for f in flow_features:
        if f not in panel.columns:
            continue
        is_ics, oos_ics, signs_is, signs_oos = [], [], [], []
        for s in syms_with_flow:
            ig = is_panel[is_panel["symbol"] == s]
            og = oos_panel[oos_panel["symbol"] == s]
            ic_is = _spearman(ig[f], ig["alpha_realized"])
            ic_oos = _spearman(og[f], og["alpha_realized"])
            is_ics.append(ic_is)
            oos_ics.append(ic_oos)
            if not np.isnan(ic_is): signs_is.append(np.sign(ic_is))
            if not np.isnan(ic_oos): signs_oos.append(np.sign(ic_oos))
        is_arr = np.array([x for x in is_ics if not np.isnan(x)])
        oos_arr = np.array([x for x in oos_ics if not np.isnan(x)])
        if len(is_arr) == 0:
            continue
        rows.append({
            "feature": f,
            "is_n": len(is_arr),
            "is_mean_ic": is_arr.mean(),
            "is_mean_abs_ic": np.abs(is_arr).mean(),
            "is_max_abs_ic": np.abs(is_arr).max(),
            "sign_pos_frac": (sum(1 for s in signs_is if s > 0) / len(signs_is)) if signs_is else np.nan,
            "oos_mean_ic": oos_arr.mean() if len(oos_arr) else np.nan,
            "oos_mean_abs_ic": np.abs(oos_arr).mean() if len(oos_arr) else np.nan,
            "is_oos_sign_match_frac":
                (sum(1 for i_ic, o_ic in zip(is_ics, oos_ics)
                      if not (np.isnan(i_ic) or np.isnan(o_ic)) and np.sign(i_ic) == np.sign(o_ic))
                  / max(1, sum(1 for i_ic, o_ic in zip(is_ics, oos_ics)
                                if not (np.isnan(i_ic) or np.isnan(o_ic))))),
        })
    df = pd.DataFrame(rows)
    df["sign_dominant"] = df["sign_pos_frac"].apply(lambda p: max(p, 1 - p) if not pd.isna(p) else np.nan)
    df["recommend"] = (
        (df["is_mean_abs_ic"] >= 0.015) &
        (df["sign_dominant"] >= 0.80) &
        (df["is_oos_sign_match_frac"] >= 0.60)
    )
    df = df.sort_values("is_mean_abs_ic", ascending=False)

    print("=" * 110)
    print(f"AGGTRADES MICROSTRUCTURE FEATURE AUDIT — IC vs alpha at h={HORIZON}")
    print(f"  universe: {len(syms_with_flow)} symbols with flow caches: {syms_with_flow}")
    print("=" * 110)
    print(df.round(4).to_string(index=False))

    print("\n--- RECOMMENDED for v8 (passes all 3 gates) ---")
    rec = df[df["recommend"]].copy()
    if len(rec) == 0:
        print("  (none)")
    else:
        print(rec.round(4).to_string(index=False))

    print("\n--- BORDERLINE ---")
    border = df[(df["is_mean_abs_ic"] >= 0.015) & ~df["recommend"]]
    if len(border) > 0:
        print(border.round(4).to_string(index=False))
    else:
        print("  (none)")

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "alpha_v8_flow_audit.csv", index=False)
    print(f"\nSaved to {out}/alpha_v8_flow_audit.csv")


if __name__ == "__main__":
    main()
