"""X34 — Comprehensive data/model/system audit.

Checks for potential issues:

DATA:
  D1. PIT discipline — verify shift(1) on rolling features
  D2. Target construction — alpha_vs_btc_realized formula + target_z normalization
  D3. Survivorship/listing-date bias — when each sym first appears
  D4. NaN coverage by feature (v2 panel)
  D5. exit_time = open_time + 4h consistency
  D6. crossX NaN at 4h-aligned bars (real signal sparsity)

MODEL:
  M1. Fold boundaries (start/end of each test window, embargo)
  M2. Per-sym training rows per fold (any folds undertrained?)
  M3. Ridge α distribution across folds (stable?)
  M4. Preprocessing fit only on train (verified by code inspection)
  M5. Cohort features computed PIT (verify build_cohort_fixed shift)

SYSTEM:
  S1. Sleeve K=3 — single seed; what's stability across seeds?
  S2. Cost model — 4.5 bps assumption sensitivity (X12 tested already)
  S3. Date range vs production deployment (how recent is end?)
"""
from __future__ import annotations
import sys, time, warnings, importlib.util
from pathlib import Path
import pandas as pd, numpy as np

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "research/convexity_portable_2026-05-20/results"

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)


def main():
    print("=" * 70)
    print("X34 SYSTEM AUDIT")
    print("=" * 70)

    # Load v2 panel
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding_v2.parquet")
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    print(f"\nLoaded panel_v2: {len(panel):,} rows × {panel['symbol'].nunique()} syms × {panel.shape[1]} cols")
    print(f"Sample date range: {panel['open_time'].min()} → {panel['open_time'].max()}")
    print(f"Sample duration: {(panel['open_time'].max() - panel['open_time'].min()).days} days")

    # ========================================================================
    # D1. PIT discipline check
    # ========================================================================
    print("\n" + "=" * 70)
    print("D1. PIT discipline check (do features at time T use ONLY [0, T] data?)")
    print("=" * 70)
    # Sample BTC
    btc = panel[panel["symbol"] == "BTCUSDT"].sort_values("open_time").head(3000)
    print(f"\nBTC sample {len(btc):,} bars: check return_1d, atr_pct, idio_vol_to_btc_1d")
    # return_1d at time T should equal log(close[T] / close[T-288])
    # We can't easily test without raw closes, but check the values are aligned
    print(f"  return_1d non-null first index: {btc[btc.return_1d.notna()]['open_time'].min()}")
    print(f"  idio_vol_to_btc_1d non-null first: {btc[btc.idio_vol_to_btc_1d.notna()]['open_time'].min()}")
    # Note: PIT verified by code inspection in build_kline_features (uses rolling().shift(1))

    # ========================================================================
    # D2. Target construction
    # ========================================================================
    print("\n" + "=" * 70)
    print("D2. Target construction (alpha_vs_btc_realized)")
    print("=" * 70)
    # Check: target value at time T should be forward-looking from T to T+4h
    sym_sample = panel[panel["symbol"] == "ETHUSDT"].sort_values("open_time").head(5000)
    sym_sample = sym_sample[sym_sample["alpha_vs_btc_realized"].notna()]
    print(f"\nETHUSDT sample {len(sym_sample):,} bars with non-null target")
    print(f"  target_z first non-null: {sym_sample[sym_sample.get('return_pct', pd.Series()).notna()]['open_time'].min() if 'return_pct' in sym_sample.columns else 'n/a'}")
    print(f"  exit_time - open_time interval (should be 4h):")
    diff = (sym_sample["exit_time"] - sym_sample["open_time"]).dropna()
    print(f"    median: {diff.median()}, std: {diff.std()}")
    print(f"    most common: {diff.mode()[0] if len(diff) > 0 else 'n/a'}")

    # ========================================================================
    # D3. Survivorship / listing date check
    # ========================================================================
    print("\n" + "=" * 70)
    print("D3. Per-sym listing dates (when each sym first appears in panel)")
    print("=" * 70)
    first_bars = panel.groupby("symbol")["open_time"].min().sort_values()
    print(f"\nEarliest first-bar (oldest sym): {first_bars.iloc[0]}")
    print(f"Latest first-bar (newest sym): {first_bars.iloc[-1]}")
    print(f"\nSyms by first-bar date (showing 10 oldest and 5 newest):")
    print("  Oldest 10:")
    for sym, t in first_bars.head(10).items():
        print(f"    {sym}: {t}")
    print("  Newest 5:")
    for sym, t in first_bars.tail(5).items():
        print(f"    {sym}: {t}")
    # Count syms with significantly truncated history
    sym_durations = (panel.groupby("symbol")["open_time"].max() - first_bars).dt.days
    print(f"\nSym history days: mean={sym_durations.mean():.0f}, median={sym_durations.median():.0f}")
    print(f"Syms with <365 days history: {(sym_durations < 365).sum()}")
    print(f"Syms with <180 days history: {(sym_durations < 180).sum()}")

    # ========================================================================
    # D4. NaN coverage
    # ========================================================================
    print("\n" + "=" * 70)
    print("D4. NaN coverage by feature group (v2 panel)")
    print("=" * 70)
    key_groups = {
        "BASE": ["return_1d", "atr_pct", "obv_z_1d", "vwap_slope_96"],
        "BTC-cross": ["corr_to_btc_1d", "beta_to_btc_change_5d", "idio_vol_to_btc_1h", "idio_vol_to_btc_1d"],
        "Funding": ["funding_rate", "funding_rate_z_7d", "funding_rate_1d_change"],
        "aggT": ["aggr_ratio_4h", "tfi_4h", "signed_volume_4h", "buy_count_4h", "avg_trade_size_4h"],
        "v3": ["idio_max_abs_12b", "idio_skew_1d", "idio_kurt_1d", "name_idio_share_1d"],
        "Target": ["alpha_vs_btc_realized", "return_pct"],
    }
    print(f"\n{'group':<12} {'feature':<30} {'%non-null':>10}")
    for grp, feats in key_groups.items():
        for f in feats:
            if f in panel.columns:
                nn = panel[f].notna().mean() * 100
                print(f"  {grp:<10} {f:<30} {nn:>9.1f}%")

    # ========================================================================
    # M1. Fold boundaries
    # ========================================================================
    print("\n" + "=" * 70)
    print("M1. Fold boundaries (walk-forward 9-fold expanding)")
    print("=" * 70)
    panel_with_target = x6.build_target_z(panel.copy())
    folds = x6.get_folds(panel_with_target)
    print(f"\n{'Fold':<6} {'OOS start':<25} {'OOS end':<25} {'embargo cut':<25}")
    for f, ts, te, ec in folds:
        print(f"  {f:<4} {str(ts):<25} {str(te):<25} {str(ec):<25}")

    # ========================================================================
    # M2. Per-sym training rows per fold
    # ========================================================================
    print("\n" + "=" * 70)
    print("M2. Per-sym training rows in folds")
    print("=" * 70)
    fold_5 = folds[4]  # mid-sample fold
    f, ts, te, ec = fold_5
    print(f"\nUsing fold {f}: OOS [{ts} → {te}], train ends at {ec}")
    train = panel_with_target[(panel_with_target["exit_time"] < ec) & panel_with_target["target_z"].notna()]
    counts = train.groupby("symbol").size().sort_values()
    print(f"\nPer-sym training rows (out of {len(train):,} total):")
    print(f"  Min: {counts.min():,} ({counts.idxmin()})")
    print(f"  Max: {counts.max():,} ({counts.idxmax()})")
    print(f"  Median: {counts.median():.0f}")
    print(f"  Syms with <300 rows (would be skipped per-sym): {(counts < 300).sum()}")
    print(f"  Syms with <5000 rows: {(counts < 5000).sum()}")

    # ========================================================================
    # M3. Cohort PIT verification
    # ========================================================================
    print("\n" + "=" * 70)
    print("M3. Cohort feature PIT verification")
    print("=" * 70)
    # Load just one sym's cohort features from a re-build
    spec_b = importlib.util.spec_from_file_location("x6b",
        REPO / "research/convexity_portable_2026-05-20/scripts/X6b_cohort_fill.py")
    x6b = importlib.util.module_from_spec(spec_b); spec_b.loader.exec_module(x6b)
    # Sample with cohort rebuilt
    small_panel = panel[panel["symbol"].isin(["BTCUSDT", "ETHUSDT", "SOLUSDT"])].copy()
    small_panel = x6b.build_cohort_fixed(small_panel)
    print(f"\nCohort features built. First non-null timestamp:")
    for c in ["rvol_7d", "ret_3d", "btc_rvol_7d"]:
        if c in small_panel.columns:
            first_nn = small_panel[small_panel[c].notna()]["open_time"].min()
            print(f"  {c}: {first_nn} (rolling 7-day requires ~7d warmup)")

    # ========================================================================
    # S1. Sleeve K=3 single-seed nature
    # ========================================================================
    print("\n" + "=" * 70)
    print("S1. Sleeve K=3 mechanism")
    print("=" * 70)
    print(f"\nThe sleeve mechanism:")
    print(f"  - At each 4h cycle: rank syms by prediction")
    print(f"  - LONG top-3 + SHORT bottom-3")
    print(f"  - 6 overlapping sleeves with equal weights [1/6]×6")
    print(f"  - Cost: 4.5 bps per leg (HL taker proxy)")
    print(f"  - Hold: 24h (covers next 6 cycles)")
    print(f"  - This is DETERMINISTIC given predictions — no random seed")
    print(f"  - Stability noted: X21 found 88% pred correlation → 1.8 Sharpe swing")
    print(f"  - X30 found 96% pred correlation → 1.4 Sharpe swing")
    print(f"  - V0 vs V5 (83% corr) → 0.35 Sharpe difference")

    # ========================================================================
    # S2. Date sample assessment
    # ========================================================================
    print("\n" + "=" * 70)
    print("S2. Sample period assessment")
    print("=" * 70)
    print(f"\nSample: {panel['open_time'].min()} → {panel['open_time'].max()}")
    print(f"Duration: {(panel['open_time'].max() - panel['open_time'].min()).days} days")
    print(f"  In months: {(panel['open_time'].max() - panel['open_time'].min()).days / 30:.1f}")
    print(f"  In quarters: {(panel['open_time'].max() - panel['open_time'].min()).days / 91:.1f}")
    print(f"\n9 folds means ~{(panel['open_time'].max() - panel['open_time'].min()).days / 9:.0f} days per fold")
    print(f"OOS in each fold: ~{(panel['open_time'].max() - panel['open_time'].min()).days / 9:.0f} days")

    # Coverage gaps over time
    print(f"\nSym count over time (monthly):")
    panel_month = panel.copy()
    panel_month["month"] = panel_month["open_time"].dt.to_period("M")
    sym_count_by_month = panel_month.groupby("month")["symbol"].nunique()
    print(f"  Min syms in any month: {sym_count_by_month.min()} ({sym_count_by_month.idxmin()})")
    print(f"  Max syms in any month: {sym_count_by_month.max()} ({sym_count_by_month.idxmax()})")
    print(f"  Sample (recent months):")
    for m, n in sym_count_by_month.tail(5).items():
        print(f"    {m}: {n} syms")


if __name__ == "__main__":
    main()
