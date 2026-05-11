"""Intraday-feature IC probe for v7 xyz.

Compute simple intraday-derived features from cached Polygon 5m bars and
test their pooled cross-sectional IC against fwd_resid_1d (the same target
v7 trades on) and fwd_resid_5d (the v7 training target).

Features tested (each computed at session close, predicts NEXT day's
residual):
  - opening_gap        = day0_open / prev_close - 1
  - first_30min_ret    = bar_10am / day_open - 1
  - last_30min_ret     = day_close / bar_3:30pm - 1
  - first_vs_last      = first_30min_ret - last_30min_ret  (intraday momentum)
  - intraday_range     = (high - low) / open
  - vwap_dev           = day_close / day_vwap - 1
  - close_pos_in_range = (close - low) / (high - low)   ∈ [0,1]
  - day0_intra         = day_close / day_open - 1
  - signed_overnight   = sign(opening_gap) * abs(opening_gap)  (directional)

Cross-sectional residualization: subtract per-day cross-sectional median
of the feature across the 15 names, so IC measures relative-rank
information after removing market-wide moves.

Compare to v7 ensemble in-sample IC of ~+0.16 on fwd_resid_1d (memory).
Individual feature IC > +0.02 (pooled) is a non-trivial standalone signal
worth incorporating; > +0.04 is competitive with the strongest A features.

Usage:
    python -m ml.research.alpha_v9_xyz_intraday_ic
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"
PRED_CACHE = CACHE / "v7_tier_a_walkfwd_preds.parquet"
XYZ_NAMES = ["AAPL", "AMD", "AMZN", "COST", "GOOGL", "INTC", "LLY", "META",
             "MSFT", "MU", "NFLX", "NVDA", "ORCL", "PLTR", "TSLA"]


def session_features(poly: pd.DataFrame) -> pd.DataFrame:
    """Per-session intraday features at close. ts is the session date
    (UTC midnight aligned with the daily preds parquet)."""
    df = poly.copy()
    df["ts_et"] = df["ts"].dt.tz_convert("America/New_York")
    df["date"] = df["ts_et"].dt.normalize().dt.tz_localize(None)
    df["min_of_day"] = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
    reg = df[(df["min_of_day"] >= 9 * 60 + 30) & (df["min_of_day"] < 16 * 60)]

    # Per-session base
    base = reg.groupby("date").agg(
        sess_open=("open", "first"),
        sess_close=("close", "last"),
        sess_high=("high", "max"),
        sess_low=("low", "min"),
        sess_vol=("volume", "sum"),
        n_bars=("close", "count"),
    ).reset_index()
    base = base[base["n_bars"] >= 30]

    # VWAP for the session
    vwp = reg.copy()
    vwp["pv"] = vwp["close"] * vwp["volume"]
    vwap_per_day = vwp.groupby("date").agg(
        pv_sum=("pv", "sum"), v_sum=("volume", "sum")
    ).reset_index()
    vwap_per_day["vwap"] = vwap_per_day["pv_sum"] / vwap_per_day["v_sum"]
    base = base.merge(vwap_per_day[["date", "vwap"]], on="date", how="left")

    # First-30min and last-30min closing prices
    first_30 = reg[reg["min_of_day"] < 9 * 60 + 30 + 30]  # 9:30-10:00
    f30 = first_30.groupby("date").agg(first30_close=("close", "last")).reset_index()
    last_30 = reg[reg["min_of_day"] >= 15 * 60 + 30]  # 15:30-16:00
    l30 = last_30.groupby("date").agg(last30_open=("open", "first")).reset_index()
    base = base.merge(f30, on="date", how="left").merge(l30, on="date", how="left")

    # Compute feature columns
    base = base.sort_values("date").reset_index(drop=True)
    base["prev_close"] = base["sess_close"].shift(1)
    base["opening_gap"] = base["sess_open"] / base["prev_close"] - 1
    base["first_30_ret"] = base["first30_close"] / base["sess_open"] - 1
    base["last_30_ret"] = base["sess_close"] / base["last30_open"] - 1
    base["first_vs_last"] = base["first_30_ret"] - base["last_30_ret"]
    base["intraday_range"] = (base["sess_high"] - base["sess_low"]) / base["sess_open"]
    base["vwap_dev"] = base["sess_close"] / base["vwap"] - 1
    base["close_pos_in_range"] = (
        (base["sess_close"] - base["sess_low"]) /
        (base["sess_high"] - base["sess_low"]).replace(0, np.nan))
    base["day0_intra"] = base["sess_close"] / base["sess_open"] - 1
    return base[["date", "opening_gap", "first_30_ret", "last_30_ret",
                  "first_vs_last", "intraday_range", "vwap_dev",
                  "close_pos_in_range", "day0_intra"]]


def main() -> None:
    log.info("loading cached preds (target source) ...")
    preds = pd.read_parquet(PRED_CACHE)
    preds["date"] = pd.to_datetime(preds["ts"]).dt.tz_convert(None).dt.normalize()
    # We only need rows for the xyz names
    preds = preds[preds["symbol"].isin(XYZ_NAMES)].copy()
    log.info("  preds rows for xyz: %d  date range %s..%s",
             len(preds), preds["date"].min().date(), preds["date"].max().date())

    feature_cols = ["opening_gap", "first_30_ret", "last_30_ret",
                     "first_vs_last", "intraday_range", "vwap_dev",
                     "close_pos_in_range", "day0_intra"]

    all_rows = []
    for sym in XYZ_NAMES:
        poly_path = CACHE / f"poly_{sym}_5m.parquet"
        if not poly_path.exists():
            log.info("  %s: no polygon cache; skip", sym); continue
        poly = pd.read_parquet(poly_path)
        feats = session_features(poly)
        feats["symbol"] = sym
        # Merge with cached daily preds (which has fwd_resid_1d, fwd_resid_5d, etc.)
        sub = preds[preds["symbol"] == sym][["date", "fwd_resid_1d", "fwd_resid_5d",
                                                "pred"]].copy()
        merged = feats.merge(sub, on="date", how="inner")
        log.info("  %-6s feat_rows=%d  joined_rows=%d", sym, len(feats), len(merged))
        all_rows.append(merged)
    df = pd.concat(all_rows, ignore_index=True)
    log.info("\n  pooled rows: %d", len(df))

    # Cross-sectional residualization: subtract per-date median of each feature
    df = df.dropna(subset=feature_cols + ["fwd_resid_1d"])
    log.info("  pooled rows after dropna: %d", len(df))
    for c in feature_cols:
        df[c + "_xs"] = df[c] - df.groupby("date")[c].transform("median")

    # Pooled IC: Pearson and Spearman against fwd_resid_1d and fwd_resid_5d
    log.info("\n=== Pooled cross-sectional IC vs fwd_resid_1d (XYZ-15, 2024-2026) ===")
    log.info("  %-22s %10s %10s %10s",
             "feature", "Pearson", "Spearman", "n")
    for c in feature_cols:
        x = df[c + "_xs"].to_numpy()
        y1 = df["fwd_resid_1d"].to_numpy()
        msk = np.isfinite(x) & np.isfinite(y1)
        if msk.sum() < 100: continue
        p_ic = np.corrcoef(x[msk], y1[msk])[0, 1]
        # Rank IC
        rx = pd.Series(x[msk]).rank().to_numpy()
        ry = pd.Series(y1[msk]).rank().to_numpy()
        s_ic = np.corrcoef(rx, ry)[0, 1]
        log.info("  %-22s %+10.4f %+10.4f %10d", c, p_ic, s_ic, int(msk.sum()))

    log.info("\n=== Pooled cross-sectional IC vs fwd_resid_5d (training target) ===")
    log.info("  %-22s %10s %10s %10s",
             "feature", "Pearson", "Spearman", "n")
    for c in feature_cols:
        x = df[c + "_xs"].to_numpy()
        y5 = df["fwd_resid_5d"].to_numpy()
        msk = np.isfinite(x) & np.isfinite(y5)
        if msk.sum() < 100: continue
        p_ic = np.corrcoef(x[msk], y5[msk])[0, 1]
        rx = pd.Series(x[msk]).rank().to_numpy()
        ry = pd.Series(y5[msk]).rank().to_numpy()
        s_ic = np.corrcoef(rx, ry)[0, 1]
        log.info("  %-22s %+10.4f %+10.4f %10d", c, p_ic, s_ic, int(msk.sum()))

    # v7 model IC for reference
    log.info("\n=== Reference: v7 ensemble pred IC on same panel ===")
    pred_xs = df["pred"] - df.groupby("date")["pred"].transform("median")
    msk = np.isfinite(pred_xs) & np.isfinite(df["fwd_resid_1d"])
    p1 = np.corrcoef(pred_xs[msk], df["fwd_resid_1d"][msk])[0, 1]
    msk5 = np.isfinite(pred_xs) & np.isfinite(df["fwd_resid_5d"])
    p5 = np.corrcoef(pred_xs[msk5], df["fwd_resid_5d"][msk5])[0, 1]
    log.info("  v7 pred (cross-sectional)   Pearson IC vs 1d: %+.4f   vs 5d: %+.4f",
             p1, p5)

    # Per-symbol IC for the strongest features (sanity: not just one name driving)
    log.info("\n=== Per-symbol IC (Pearson) for top features vs fwd_resid_1d ===")
    log.info("  %-6s %12s %12s %12s %12s",
             "sym", "vwap_dev", "first_vs_last", "day0_intra", "open_gap")
    for sym in XYZ_NAMES:
        g = df[df["symbol"] == sym]
        if len(g) < 50: continue
        ics = {}
        for c in ["vwap_dev", "first_vs_last", "day0_intra", "opening_gap"]:
            x = g[c + "_xs"].to_numpy(); y = g["fwd_resid_1d"].to_numpy()
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 30: ics[c] = np.nan; continue
            ics[c] = np.corrcoef(x[m], y[m])[0, 1]
        log.info("  %-6s %+12.4f %+12.4f %+12.4f %+12.4f",
                 sym, ics["vwap_dev"], ics["first_vs_last"],
                 ics["day0_intra"], ics["opening_gap"])


if __name__ == "__main__":
    main()
