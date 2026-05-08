"""v7 multi-alpha cross-sectional residual model.

Combines 4 alpha categories into one LGBM. Each category targets a
different aspect of equity-residual return; they should be relatively
independent so combination has additive value.

Setup:
  Universe: S&P 100 daily, 2013-2026 (~100 names with full history).
            Covers most xyz US equities (mega-caps + Eli Lilly etc.) plus
            mid-caps for additional cross-sectional dispersion.
  Anchor: leave-one-out 100-name basket.
  Cadence: daily, 1d hold, daily rebalance.

Feature groups:
  A — price-pattern (10 features): daily-cadence ports of v6_clean. Windows
      now in trading-day units (1d, 5d, 22d, 60d).
  B — PEAD (4 features): days_since_earnings, surprise_pct, event_day_residual,
      decay_weighted_signal.
  C — cross-asset (8 features): named conditional inputs (returns of
      SPY/TLT/UUP/VIX/SOXX/XLK/GLD, plus rolling correlations).
  D — calendar (4 features): day_of_week, day_of_month, month_end, year_end.

Training: pooled LGBM with sym_id, walk-forward expanding window, 6 folds.
Cost: turnover-weighted, 5 bps/trade-side default.

Per-group ablation: train models with each subset of groups (A only, B only,
A+B, etc.) to see which combinations actually add Sharpe.
"""
from __future__ import annotations

import logging
import warnings
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data_collectors.sp100_loader import load_universe

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"

# === config ===
BETA_WINDOW = 60                   # 60-day rolling beta
FWD_HORIZON = 1                    # 1-day forward residual return
HOLD_DAYS = 1
TOP_K = 10                         # 10 long, 10 short out of valid universe (~100)
COST_PER_TRADE_BPS = 5
SEEDS = (42, 7, 123, 99, 314)
PEAD_MAX_DAYS = 60

ANCHOR_TICKERS = ["SPY", "TLT", "UUP", "VIX", "TNX", "SOXX", "XLK", "GLD"]

LGB_PARAMS = dict(
    objective="regression", metric="rmse",
    num_leaves=31, max_depth=6, learning_rate=0.03,
    n_estimators=300, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=5, min_child_samples=200, verbose=-1,
)


# ---- data --------------------------------------------------------------

def load_anchors() -> pd.DataFrame:
    """Wide-format anchor returns: ts × {SPY_ret, TLT_ret, ...}."""
    out = None
    for sym in ANCHOR_TICKERS:
        cache = CACHE / f"yf_{sym}_1d_anchor.parquet"
        if not cache.exists():
            log.warning("  anchor missing: %s", sym)
            continue
        df = pd.read_parquet(cache)
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.normalize().astype("datetime64[ns, UTC]")
        df = df.set_index("ts")
        df[f"{sym}_ret"] = np.log(df["close"] / df["close"].shift(1))
        df[f"{sym}_close"] = df["close"]
        df = df[[f"{sym}_ret", f"{sym}_close"]]
        out = df if out is None else out.join(df, how="outer")
    return out.reset_index()


def add_returns_and_basket(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    panel["ret"] = (panel.groupby("symbol")["close"]
                    .transform(lambda s: np.log(s / s.shift(1))))
    grp_ts = panel.groupby("ts")["ret"]
    total = grp_ts.transform("sum")
    n = grp_ts.transform("count")
    panel["bk_ret"] = (total - panel["ret"].fillna(0)) / (n - 1).replace(0, np.nan)
    return panel


def add_residual(panel: pd.DataFrame) -> pd.DataFrame:
    def _beta(g):
        cov = (g["ret"] * g["bk_ret"]).rolling(BETA_WINDOW).mean() - \
              g["ret"].rolling(BETA_WINDOW).mean() * g["bk_ret"].rolling(BETA_WINDOW).mean()
        var = g["bk_ret"].rolling(BETA_WINDOW).var().replace(0, np.nan)
        return (cov / var).clip(-5, 5).shift(1)
    panel["beta"] = panel.groupby("symbol", group_keys=False).apply(_beta).values
    panel["resid"] = panel["ret"] - panel["beta"] * panel["bk_ret"]
    panel["fwd_resid_1d"] = panel.groupby("symbol", group_keys=False)["resid"].shift(-1)
    return panel


# ---- feature group A: price-pattern -----------------------------------

def add_features_A(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    g = panel.groupby("symbol", group_keys=False)
    panel["A_ret_1d"] = g["close"].apply(lambda s: s.pct_change(1).shift(1))
    panel["A_ret_5d"] = g["close"].apply(lambda s: s.pct_change(5).shift(1))
    panel["A_ret_22d"] = g["close"].apply(lambda s: s.pct_change(22).shift(1))
    panel["A_ret_60d"] = g["close"].apply(lambda s: s.pct_change(60).shift(1))
    panel["A_vol_22d"] = g["ret"].apply(lambda s: s.rolling(22).std().shift(1))
    panel["A_vol_60d"] = g["ret"].apply(lambda s: s.rolling(60).std().shift(1))
    panel["A_idio_ret_5d"] = g["resid"].apply(lambda s: s.rolling(5).sum().shift(1))
    panel["A_idio_ret_22d"] = g["resid"].apply(lambda s: s.rolling(22).sum().shift(1))
    panel["A_idio_vol_22d"] = g["resid"].apply(lambda s: s.rolling(22).std().shift(1))
    # OBV proxy on daily (using close direction × volume)
    def _obv(g_):
        sign = np.sign(g_["close"].diff().fillna(0))
        return (sign * g_["volume"]).cumsum()
    panel["A_obv"] = g.apply(_obv).reset_index(level=0, drop=True)
    panel["A_obv_z_22d"] = ((panel["A_obv"] - g["A_obv"].apply(
        lambda s: s.rolling(22).mean()))
        / g["A_obv"].apply(lambda s: s.rolling(22).std()).replace(0, np.nan)
    ).clip(-5, 5).shift(1)
    return panel, ["A_ret_1d", "A_ret_5d", "A_ret_22d", "A_ret_60d",
            "A_vol_22d", "A_vol_60d", "A_idio_ret_5d", "A_idio_ret_22d",
            "A_idio_vol_22d", "A_obv_z_22d"]


# ---- feature group B: PEAD ---------------------------------------------

def add_features_B(panel: pd.DataFrame, earnings: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    earn = earnings.copy()
    earn["ts"] = pd.to_datetime(earn["ts"], utc=True).dt.normalize().astype("datetime64[ns, UTC]")
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True).astype("datetime64[ns, UTC]")
    earn = earn.sort_values(["symbol", "ts"]).reset_index(drop=True)
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)

    out_chunks = []
    for sym, g in panel.groupby("symbol"):
        e = earn[earn["symbol"] == sym][["ts", "surprise_pct"]].dropna(subset=["surprise_pct"])
        if e.empty:
            g = g.copy()
            g["B_days_since_earn"] = np.nan
            g["B_surprise_pct"] = np.nan
            g["B_event_day_resid"] = np.nan
            g["B_decay_signal"] = np.nan
            out_chunks.append(g)
            continue
        e_sorted = e.sort_values("ts").rename(columns={"ts": "earnings_ts"})
        merged = pd.merge_asof(
            g.sort_values("ts"), e_sorted,
            left_on="ts", right_on="earnings_ts",
            allow_exact_matches=False, direction="backward",
        )
        merged["B_days_since_earn"] = (merged["ts"] - merged["earnings_ts"]).dt.days
        valid = merged["B_days_since_earn"].between(1, PEAD_MAX_DAYS, inclusive="both")
        merged["B_surprise_pct"] = merged["surprise_pct"].where(valid, np.nan)
        # event_day_resid: residual return on the earnings day itself
        # (find resid at earnings_ts, broadcast forward to all valid days)
        merged["B_event_day_resid"] = np.nan
        # need to look up resid at the earnings_ts row for this symbol
        # easier: do a merge_asof on a separate dataframe of (sym, earnings_ts -> resid_on_that_day)
        out_chunks.append(merged)
    panel_b = pd.concat(out_chunks, ignore_index=True)

    # populate B_event_day_resid by joining (sym, earnings_ts) -> resid on that earnings day
    earn_resid = panel.merge(earn[["symbol", "ts"]].rename(columns={"ts": "earnings_ts"}),
                              left_on=["symbol", "ts"], right_on=["symbol", "earnings_ts"],
                              how="inner")[["symbol", "earnings_ts", "resid"]]
    earn_resid = earn_resid.rename(columns={"resid": "B_event_day_resid_join"})
    # Dedupe: multiple earnings entries on same date (rare but possible) → keep first
    earn_resid = earn_resid.drop_duplicates(subset=["symbol", "earnings_ts"], keep="first")
    panel_b = panel_b.drop(columns=["B_event_day_resid"]).merge(
        earn_resid, on=["symbol", "earnings_ts"], how="left")
    panel_b = panel_b.drop_duplicates(subset=["symbol", "ts"], keep="first")
    panel_b = panel_b.rename(columns={"B_event_day_resid_join": "B_event_day_resid"})
    valid_b = panel_b["B_days_since_earn"].between(1, PEAD_MAX_DAYS, inclusive="both")
    panel_b["B_event_day_resid"] = panel_b["B_event_day_resid"].where(valid_b, np.nan)
    panel_b["B_decay_signal"] = (panel_b["B_event_day_resid"]
        * (1 - panel_b["B_days_since_earn"].clip(upper=PEAD_MAX_DAYS) / PEAD_MAX_DAYS))

    # Copy back to original panel order
    panel_b = panel_b.sort_values(["symbol", "ts"]).reset_index(drop=True)
    for c in ("B_days_since_earn", "B_surprise_pct", "B_event_day_resid", "B_decay_signal"):
        panel[c] = panel_b[c].values

    return panel, ["B_days_since_earn", "B_surprise_pct", "B_event_day_resid", "B_decay_signal"]


# ---- feature group C: cross-asset --------------------------------------

def add_features_C(panel: pd.DataFrame, anchors: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Cross-asset features: same-day returns + rolling correlations of
    each name's return with each anchor over BETA_WINDOW."""
    anchors = anchors.copy()
    anchors["ts"] = pd.to_datetime(anchors["ts"], utc=True).dt.normalize().astype("datetime64[ns, UTC]")
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True).astype("datetime64[ns, UTC]")
    panel = panel.merge(anchors, on="ts", how="left")
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    g = panel.groupby("symbol", group_keys=False)

    feats = []
    for a in ("SPY", "TLT", "UUP", "VIX", "SOXX", "GLD"):
        col_ret = f"{a}_ret"
        if col_ret not in panel.columns:
            continue
        # 5-day cumulative anchor return (regime indicator)
        panel[f"C_{a}_ret_5d"] = panel[col_ret].rolling(5).sum().shift(1)
        # rolling correlation of name's return with anchor (regime-conditioning)
        panel[f"C_{a}_corr_60d"] = (
            g.apply(lambda gg: gg["ret"].rolling(BETA_WINDOW).corr(gg[col_ret]))
            .reset_index(level=0, drop=True).shift(1)
        ).clip(-1, 1)
        feats.extend([f"C_{a}_ret_5d", f"C_{a}_corr_60d"])
    return panel, feats


# ---- feature group D: calendar ----------------------------------------

def add_features_D(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    et = panel["ts"].dt.tz_convert("America/New_York")
    panel["D_dow"] = et.dt.dayofweek.astype(float)
    panel["D_dom"] = et.dt.day.astype(float)
    panel["D_is_month_end"] = (et.dt.day >= 25).astype(float)
    panel["D_is_year_end"] = ((et.dt.month == 12) & (et.dt.day >= 15)).astype(float)
    return panel, ["D_dow", "D_dom", "D_is_month_end", "D_is_year_end"]


# ---- training + portfolio ---------------------------------------------

def make_folds(panel: pd.DataFrame, train_min_days: int = 365 * 3,
               test_days: int = 365, embargo_days: int = 5) -> list[tuple]:
    panel = panel.sort_values("ts")
    t0 = panel["ts"].min().normalize()
    t_max = panel["ts"].max()
    folds = []
    days = train_min_days
    while True:
        train_end = t0 + timedelta(days=days)
        test_start = train_end + timedelta(days=embargo_days)
        test_end = test_start + timedelta(days=test_days)
        if test_start >= t_max:
            break
        if test_end > t_max:
            test_end = t_max
        folds.append((train_end, test_start, test_end))
        days += test_days
    return folds


def fit_predict(train: pd.DataFrame, test: pd.DataFrame,
                features: list[str], label: str = "fwd_resid_1d") -> pd.DataFrame:
    train_ = train.dropna(subset=features + [label])
    if len(train_) < 1000:
        return pd.DataFrame()
    preds = []
    sub = test.dropna(subset=features).copy()
    for seed in SEEDS:
        m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
        m.fit(train_[features], train_[label])
        preds.append(m.predict(sub[features]))
    sub["pred"] = np.mean(preds, axis=0)
    return sub


def construct_portfolio(test_pred: pd.DataFrame, signal: str,
                         pnl_label: str, top_k: int = TOP_K,
                         cost_bps: float = COST_PER_TRADE_BPS) -> pd.DataFrame:
    sub = test_pred.dropna(subset=[signal, pnl_label]).copy()
    rows = []
    prev_long: set = set()
    prev_short: set = set()
    for ts, bar in sub.groupby("ts"):
        if len(bar) < 2 * top_k:
            continue
        bar = bar.sort_values(signal)
        long_leg = set(bar.tail(top_k)["symbol"])
        short_leg = set(bar.head(top_k)["symbol"])
        long_changes = len(long_leg.symmetric_difference(prev_long))
        short_changes = len(short_leg.symmetric_difference(prev_short))
        turnover = (long_changes + short_changes) / (2 * top_k)
        cost = turnover * cost_bps / 1e4
        long_alpha = bar[bar["symbol"].isin(long_leg)][pnl_label].mean()
        short_alpha = bar[bar["symbol"].isin(short_leg)][pnl_label].mean()
        spread = long_alpha - short_alpha
        rows.append({
            "ts": ts, "spread_alpha": spread,
            "long_alpha": long_alpha, "short_alpha": short_alpha,
            "turnover": turnover, "cost": cost,
            "net_alpha": spread - cost, "n_universe": len(bar),
        })
        prev_long, prev_short = long_leg, short_leg
    return pd.DataFrame(rows)


def metrics(pnl: pd.DataFrame) -> dict:
    if pnl.empty:
        return {"n": 0}
    n = len(pnl)
    g_sh = (pnl["spread_alpha"].mean() / pnl["spread_alpha"].std()
            * np.sqrt(252)) if pnl["spread_alpha"].std() > 0 else 0
    n_sh = (pnl["net_alpha"].mean() / pnl["net_alpha"].std()
            * np.sqrt(252)) if pnl["net_alpha"].std() > 0 else 0
    return {
        "n": n,
        "gross_bps": pnl["spread_alpha"].mean() * 1e4,
        "net_bps": pnl["net_alpha"].mean() * 1e4,
        "annual_cost_bps": pnl["cost"].mean() * 1e4 * 252,
        "turnover_pct": pnl["turnover"].mean() * 100,
        "univ_size": pnl["n_universe"].mean(),
        "gross_sharpe": g_sh, "net_sharpe": n_sh,
        "hit_rate": float((pnl["spread_alpha"] > 0).mean()),
    }


def bootstrap_ci(pnl: pd.DataFrame, block_days: int = 60,
                 n_boot: int = 2000) -> tuple[float, float]:
    if pnl.empty or len(pnl) < block_days * 2:
        return np.nan, np.nan
    arr = pnl["net_alpha"].values
    n_blocks = max(1, len(arr) // block_days)
    rng = np.random.default_rng(42)
    sh = []
    for _ in range(n_boot):
        starts = rng.integers(0, len(arr) - block_days + 1, size=n_blocks)
        sample = np.concatenate([arr[s:s + block_days] for s in starts])
        if sample.std() > 0:
            sh.append(sample.mean() / sample.std() * np.sqrt(252))
    if not sh:
        return np.nan, np.nan
    return float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))


# ---- main --------------------------------------------------------------

def main() -> None:
    log.info("loading S&P 100 universe + earnings + cross-asset anchors...")
    panel, earnings, surv = load_universe()
    if panel.empty:
        log.error("no universe data")
        return
    anchors = load_anchors()

    panel = add_returns_and_basket(panel)
    panel = add_residual(panel)

    log.info("residualization sanity: median beta=%.2f IQR=[%.2f,%.2f]",
             panel["beta"].median(),
             panel["beta"].quantile(0.25), panel["beta"].quantile(0.75))

    panel, feats_A = add_features_A(panel)
    panel, feats_B = add_features_B(panel, earnings)
    panel, feats_C = add_features_C(panel, anchors)
    panel, feats_D = add_features_D(panel)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes

    log.info("feature groups: A(price)=%d  B(PEAD)=%d  C(cross-asset)=%d  D(calendar)=%d",
             len(feats_A), len(feats_B), len(feats_C), len(feats_D))

    feature_groups = {
        "A_price": feats_A,
        "B_pead": feats_B,
        "C_cross_asset": feats_C,
        "D_calendar": feats_D,
    }
    all_features = sum(feature_groups.values(), []) + ["sym_id"]

    folds = make_folds(panel, train_min_days=365 * 3, test_days=365)
    log.info("\nfolds:")
    for i, f in enumerate(folds):
        log.info("  fold %d: train<=%s  test=[%s, %s]",
                 i + 1, f[0].strftime("%Y-%m-%d"),
                 f[1].strftime("%Y-%m-%d"), f[2].strftime("%Y-%m-%d"))

    # === ablation: try each feature group + combinations ===
    label = "fwd_resid_1d"
    pnl_label = "fwd_resid_1d"
    panel[pnl_label] = panel["fwd_resid_1d"]
    ablation_configs = {
        "A only": feats_A + ["sym_id"],
        "B only": feats_B + ["sym_id"],
        "C only": feats_C + ["sym_id"],
        "D only": feats_D + ["sym_id"],
        "A+B": feats_A + feats_B + ["sym_id"],
        "A+B+C": feats_A + feats_B + feats_C + ["sym_id"],
        "ALL (A+B+C+D)": all_features,
    }

    results = {}
    for cfg_name, feats in ablation_configs.items():
        log.info("\n>>> CONFIG: %s  (n_features=%d)", cfg_name, len(feats))
        all_pnls = []
        for fold in folds:
            train_end, test_start, test_end = fold
            train = panel[panel["ts"] <= train_end].copy()
            test = panel[(panel["ts"] >= test_start) & (panel["ts"] <= test_end)].copy()
            test_pred = fit_predict(train, test, feats, label)
            if test_pred.empty:
                continue
            lp = construct_portfolio(test_pred, "pred", pnl_label)
            if not lp.empty:
                lp["fold"] = fold[1].year
                all_pnls.append(lp)
        if not all_pnls:
            continue
        st = pd.concat(all_pnls, ignore_index=True)
        m = metrics(st)
        lo, hi = bootstrap_ci(st)
        results[cfg_name] = (m, lo, hi, st)
        log.info("  STITCHED: n=%d gross=%+.2fbps/d net=%+.2fbps/d  net_Sh=%+.2f  [%+.2f, %+.2f]  hit=%.0f%%",
                 m["n"], m["gross_bps"], m["net_bps"], m["net_sharpe"],
                 lo, hi, 100 * m["hit_rate"])

    log.info("\n=== ABLATION SUMMARY (cost=%d bps/trade-side, top_k=%d) ===",
             COST_PER_TRADE_BPS, TOP_K)
    log.info("  %-18s %5s %10s %10s %10s %18s",
             "config", "n", "gross/d", "net/d", "net_Sh", "95% CI")
    for cfg, (m, lo, hi, _) in results.items():
        log.info("  %-18s %5d %+8.2fbps %+8.2fbps %+10.2f  [%+.2f, %+.2f]",
                 cfg, m["n"], m["gross_bps"], m["net_bps"],
                 m["net_sharpe"], lo, hi)


if __name__ == "__main__":
    main()
