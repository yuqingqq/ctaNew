"""Phase 2: build 51-name clean panel for BTC-residual strategy.

Universe: 36 cleaned local + 15 new high-volume additions.
  Drop:  MKR (Binance-delisted), 1000PEPE, 1000SHIB (not on HL),
         CHIP (21 days only), MEGA (97 days only)
  Add:   ZEC, ONDO, STRK, HYPE, TAO, ENA, JTO, ASTER, PUMP, VVV, BIO,
         JUP, PENGU, VIRTUAL, PENDLE   (all ≥230 days history)

Builds wide panel with:
  - kline features (per-symbol)
  - BTC reference (close, returns, forward returns)
  - β to BTC (rolling 1d, point-in-time)
  - alpha-vs-BTC target (β-adjusted, z-scored)
  - basket-residual columns kept for comparison

Output: cached panel parquet for downstream Phase 3 + training.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import (
    list_universe, build_kline_features, build_basket,
    add_basket_features, add_engineered_flow_features,
    add_xs_rank_features, XS_RANK_SOURCES,
    XS_FEATURE_COLS_V6_CLEAN, make_xs_alpha_labels,
)
from ml.research.alpha_v8_h48_audit import aggregate_4h_flow, AGGTRADE_4H_TO_ADD, CACHE_DIR

OUT_DIR = REPO / "outputs/vBTC_panel"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_CACHE = OUT_DIR / "panel_51.parquet"

DROP_SYMBOLS = {"MKRUSDT", "1000PEPEUSDT", "1000SHIBUSDT", "CHIPUSDT", "MEGAUSDT"}

NEW_SYMBOLS = {"ZECUSDT", "ONDOUSDT", "STRKUSDT", "HYPEUSDT", "TAOUSDT",
               "ENAUSDT", "JTOUSDT", "ASTERUSDT", "PUMPUSDT", "VVVUSDT",
               "BIOUSDT", "JUPUSDT", "PENGUUSDT", "VIRTUALUSDT", "PENDLEUSDT"}

HORIZON = 48
BETA_WINDOW = 288
BTC_SYMBOL = "BTCUSDT"


def build_panel_51(force_rebuild: bool = False) -> pd.DataFrame:
    """Build cleaned 51-name panel with both basket-residual and BTC-residual targets."""
    if PANEL_CACHE.exists() and not force_rebuild:
        print(f"  Loading cached panel from {PANEL_CACHE}")
        return pd.read_parquet(PANEL_CACHE)

    universe_full = sorted(list_universe(min_days=200))
    universe = sorted([s for s in universe_full if s not in DROP_SYMBOLS])
    print(f"  Universe: {len(universe)} symbols (dropped: {sorted(DROP_SYMBOLS & set(universe_full))})")
    expected_new = sorted(NEW_SYMBOLS & set(universe))
    print(f"  Includes new: {expected_new}")

    # Per-symbol kline features
    print("  Building kline features...")
    feats = {s: build_kline_features(s) for s in universe}
    feats = {s: f for s, f in feats.items() if not f.empty}
    closes = pd.DataFrame({s: feats[s]["close"] for s in feats}).sort_index()

    # Basket (uses all 51 names — re-derived for this universe)
    print("  Building basket...")
    basket_ret, basket_close = build_basket(closes)
    sym_to_id = {s: i for i, s in enumerate(sorted(feats.keys()))}

    # Enrich each symbol with basket features (β_short_vs_bk etc.) + flow
    print("  Enriching per-symbol features...")
    enriched = {}
    for s in feats:
        f = feats[s].reindex(closes.index)
        f = add_basket_features(f, basket_close, basket_ret)
        f = add_engineered_flow_features(f)
        f["sym_id"] = sym_to_id[s]
        if f.index.tz is None:
            f.index = f.index.tz_localize("UTC")
        cache = CACHE_DIR / f"flow_{s}.parquet"
        if cache.exists():
            flow = pd.read_parquet(cache)
            if flow.index.tz is None:
                flow.index = flow.index.tz_localize("UTC")
            f = f.join(aggregate_4h_flow(flow), how="left")
        enriched[s] = f

    # Basket-residual labels
    print("  Computing basket-residual labels...")
    labels = make_xs_alpha_labels(enriched, basket_close, HORIZON)

    # Stack into long panel
    print("  Stacking...")
    rank_cols = [c for c in XS_FEATURE_COLS_V6_CLEAN if c.endswith("_xs_rank")]
    src_cols = list({s for s, d in XS_RANK_SOURCES.items() if d in rank_cols})
    needed = list(set(list(XS_FEATURE_COLS_V6_CLEAN)
                       + ["sym_id", "autocorr_pctile_7d", "beta_short_vs_bk"]
                       + src_cols + AGGTRADE_4H_TO_ADD) - set(rank_cols))
    frames = []
    for s, f in enriched.items():
        avail = [c for c in needed if c in f.columns]
        df = f[avail].join(labels[s], how="inner")
        df["symbol"] = s
        df = df.reset_index().rename(columns={"index": "open_time"})
        for c in df.select_dtypes("float64").columns:
            df[c] = df[c].astype("float32")
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True, sort=False)
    panel = add_xs_rank_features(panel, sources=XS_RANK_SOURCES)
    for c in rank_cols:
        if c in panel.columns:
            panel[c] = panel[c].astype("float32")
    panel = panel.dropna(subset=list(XS_FEATURE_COLS_V6_CLEAN)
                          + ["autocorr_pctile_7d", "demeaned_target", "return_pct"])
    print(f"  Basket-residual panel: {len(panel):,} rows × {panel['symbol'].nunique()} syms")

    # Add BTC reference + β-adjusted BTC residual target
    print("  Adding β-to-BTC + BTC-residual target...")
    btc_close = enriched[BTC_SYMBOL]["close"].copy()
    if btc_close.index.tz is None:
        btc_close.index = btc_close.index.tz_localize("UTC")
    btc_ret_full = btc_close.pct_change()
    btc_fwd_full = btc_close.pct_change(HORIZON).shift(-HORIZON)

    rows = []
    for s in sorted(enriched.keys()):
        f = enriched[s]
        if f.empty: continue
        my_close = f["close"].copy()
        if my_close.index.tz is None:
            my_close.index = my_close.index.tz_localize("UTC")
        my_ret = my_close.pct_change()
        joined = pd.DataFrame({"my_ret": my_ret,
                                 "btc_ret": btc_ret_full.reindex(my_close.index, method="ffill"),
                                 "btc_fwd": btc_fwd_full.reindex(my_close.index, method="ffill")}).dropna(subset=["my_ret","btc_ret"])
        if len(joined) < BETA_WINDOW + 50: continue
        cov = (joined["my_ret"] * joined["btc_ret"]).rolling(BETA_WINDOW).mean() - \
              joined["my_ret"].rolling(BETA_WINDOW).mean() * joined["btc_ret"].rolling(BETA_WINDOW).mean()
        var = joined["btc_ret"].rolling(BETA_WINDOW).var().replace(0, np.nan)
        beta_pit = (cov / var).clip(-5, 5).shift(1)
        sub = pd.DataFrame({
            "open_time": joined.index, "symbol": s,
            "beta_to_btc": beta_pit.values,
            "btc_fwd": joined["btc_fwd"].values,
        })
        rows.append(sub)
    btc_meta = pd.concat(rows, ignore_index=True)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    btc_meta["open_time"] = pd.to_datetime(btc_meta["open_time"], utc=True)
    panel = panel.merge(btc_meta, on=["symbol", "open_time"], how="left")
    panel["alpha_vs_btc_realized"] = panel["return_pct"] - panel["beta_to_btc"] * panel["btc_fwd"]

    # Z-score BTC target (per-symbol expanding mean shifted by horizon, 7d rolling std)
    panel = panel.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    g = panel.groupby("symbol")["alpha_vs_btc_realized"]
    rmean = g.transform(lambda s: s.expanding(min_periods=288).mean().shift(HORIZON))
    rstd = g.transform(lambda s: s.rolling(288 * 7, min_periods=288).std().shift(HORIZON))
    panel["btc_target"] = (panel["alpha_vs_btc_realized"] - rmean) / rstd.replace(0, np.nan)

    # Save
    panel.to_parquet(PANEL_CACHE, compression="zstd")
    print(f"  Saved → {PANEL_CACHE}  ({len(panel):,} rows × {panel.shape[1]} cols)")
    return panel


def main():
    panel = build_panel_51(force_rebuild=True)

    # Verification
    print("\n" + "=" * 90)
    print("PANEL VERIFICATION")
    print("=" * 90)
    print(f"  Total rows:         {len(panel):,}")
    print(f"  Unique symbols:     {panel['symbol'].nunique()}")
    print(f"  Time span:          {panel['open_time'].min()} → {panel['open_time'].max()}")
    print(f"  Cycles per symbol:")
    for s, n in panel.groupby("symbol").size().sort_values().items():
        flag = "  (NEW)" if s in NEW_SYMBOLS else ""
        print(f"    {s:<14}  {n:>7,} rows{flag}")
    print(f"\n  btc_target non-NaN: {panel['btc_target'].notna().sum():,}")
    print(f"  beta_to_btc non-NaN: {panel['beta_to_btc'].notna().sum():,}")
    print(f"  demeaned_target non-NaN: {panel['demeaned_target'].notna().sum():,}")


if __name__ == "__main__":
    main()
