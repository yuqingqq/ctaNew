"""Does Binance bookDepth add to the residual model at a holding-aligned horizon?

This is deliberately the same model/evaluation frame as ``l2_influence_quant``:
per-symbol RidgeCV, V0_LEAN preprocessing, 60-day exponential weighting, an
exit-time label purge, one-day embargo, and per-bar cross-sectional rank-IC.

The earlier L2 ablation used the incumbent next-4h BTC-residual label.  This
test additionally builds gap-safe sums of the next 3 and 6 non-overlapping
4h residual labels (12h and 24h).  Thus it tests whether coarse book state is
useful at the horizon of an overlapping one-day sleeve, without accidentally
turning a data gap into a long forward label.

This is a signal-quality test, not an execution backtest.  It does not claim
that a positive IC would clear one-day trading costs.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
os.environ["V4_PANEL"] = str(REPO / "outputs/vBTC_features/panel_expanded_v0_clean.parquet")

from live.bookdepth_real_ablation import HL, RECENT_CUTS, OOS_CUTS, V0_LEAN, x6
from live.l2_influence_quant import OB, build, d_ci

EMBARGO = pd.Timedelta(days=1)
BAR = pd.Timedelta(hours=4)
HORIZONS = {"4h": 1, "12h": 3, "24h": 6}


def make_horizon_label(panel: pd.DataFrame, bars: int) -> pd.DataFrame:
    """Add a gap-safe cumulative residual label and its exact exit time.

    ``alpha_vs_btc_realized`` is the audited next-4h residual.  Reindexing to
    a complete 4h grid makes a cumulative label NaN whenever any component
    bar is absent, rather than silently spanning an outage by row count.
    """
    out = panel.copy()
    pieces = []
    for symbol, group in out.groupby("symbol", sort=False):
        g = group.sort_values("open_time")
        index = pd.date_range(g["open_time"].min(), g["open_time"].max(), freq="4h", tz="UTC")
        alpha = g.set_index("open_time")["alpha_vs_btc_realized"].reindex(index)
        label = alpha.rolling(bars, min_periods=bars).sum().shift(-(bars - 1))
        pieces.append(pd.DataFrame({"symbol": symbol, "open_time": index, "h_alpha": label.values}))
    labels = pd.concat(pieces, ignore_index=True)
    out = out.merge(labels, on=["symbol", "open_time"], how="left", validate="one_to_one")
    out["h_exit_time"] = out["open_time"] + bars * BAR

    cross = out.groupby("open_time")["h_alpha"]
    sd = cross.transform("std").replace(0, np.nan)
    out["h_target_z"] = ((out["h_alpha"] - cross.transform("mean")) / sd).clip(-10, 10)
    return out


def predict(panel: pd.DataFrame, features: list[str], cuts: list[pd.Timestamp]) -> pd.DataFrame:
    """Walk-forward, per-symbol predictions; fail loudly on any model error."""
    rows, failures = [], []
    for fold, (start, end) in enumerate(zip(cuts[:-1], cuts[1:])):
        train_cut = start - EMBARGO
        train = panel[(panel["h_exit_time"] < train_cut) & panel["h_target_z"].notna()]
        test = panel[(panel["open_time"] >= start) & (panel["open_time"] < end)]
        train_end = train["open_time"].max()
        for symbol, group in train.groupby("symbol", sort=False):
            if len(group) < 300:
                continue
            test_group = test[test["symbol"] == symbol]
            if test_group.empty:
                continue
            try:
                sstats, hstats = x6.fit_preproc(group, features)
                x_train = x6.apply_preproc(group, features, sstats, hstats)
                age_days = (train_end - group["open_time"]).dt.total_seconds().to_numpy() / 86400.0
                weights = np.exp(-age_days / HL)
                model = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(
                    x_train, group["h_target_z"].to_numpy(), sample_weight=weights
                )
                rows.append(pd.DataFrame({
                    "open_time": test_group["open_time"].values,
                    "h_alpha": test_group["h_alpha"].values,
                    "pred": model.predict(x6.apply_preproc(test_group, features, sstats, hstats)),
                    "cov": test_group["_covered"].values,
                    "fold": fold,
                }))
            except Exception as exc:  # a silent dropped symbol invalidates an ablation
                failures.append(f"fold={fold} symbol={symbol}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError("walk-forward model failures:\n" + "\n".join(failures[:20]))
    if not rows:
        raise RuntimeError("walk-forward prediction produced no rows")
    return pd.concat(rows, ignore_index=True)


def per_bar_rank_ic(predictions: pd.DataFrame) -> pd.Series:
    covered = predictions[predictions["cov"] & predictions["h_alpha"].notna()]
    return covered.groupby("open_time").apply(
        lambda g: spearmanr(g["pred"], g["h_alpha"]).correlation if len(g) >= 5 else np.nan
    ).dropna()


def run_horizon(panel: pd.DataFrame, label: str, bars: int) -> None:
    hpanel = make_horizon_label(panel, bars)
    print(f"\n{'=' * 86}\n{label} target ({bars} x 4h residual bars)\n{'=' * 86}")
    print(f"labelled rows {int(hpanel['h_target_z'].notna().sum())} | L2-covered rows "
          f"{int((hpanel['_covered'] & hpanel['h_target_z'].notna()).sum())}")
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        baseline = predict(hpanel, V0_LEAN, cuts)
        with_ob = predict(hpanel, V0_LEAN + OB, cuts)
        base_ic, ob_ic = per_bar_rank_ic(baseline), per_bar_rank_ic(with_ob)
        delta, (low, high) = d_ci(base_ic, ob_ic)
        verdict = "HURTS (CI<0)" if high < 0 else ("HELPS (CI>0)" if low > 0 else "within noise")
        print(f"  {era:6s} baseline {base_ic.mean():+.4f} | +OB {ob_ic.mean():+.4f} | "
              f"delta {delta:+.4f} [{low:+.4f},{high:+.4f}] -> {verdict} "
              f"({len(base_ic)} covered bars)")


def main() -> None:
    panel = build()
    print(f"panel rows {len(panel)} | L2-covered {int(panel['_covered'].sum())} | "
          f"V0_LEAN={len(V0_LEAN)} | OB={len(OB)}")
    for label, bars in HORIZONS.items():
        run_horizon(panel, label, bars)
    print("L2HORIZONABLDONE")


if __name__ == "__main__":
    main()
