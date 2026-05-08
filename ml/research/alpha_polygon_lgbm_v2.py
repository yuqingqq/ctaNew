"""Polygon 5m × 2y LGBM probe — v2 with the 4 high-impact fixes applied.

Changes vs alpha_polygon_lgbm.py:
  Fix 1 — Calendar-correct window sizes for equity RTH 5m bars:
            BARS_1D = 78 (not 288: equity RTH has 78 bars/day, not 288)
            BARS_5D = 390 (1 trading week, was BARS_7D = 2016 = 7 calendar days)
            BETA_WINDOW = 390 (1 trading week — stable beta at 5m)
          Many features that nominally measured "1d momentum" or "7d z-score"
          on crypto were measuring 3.7d / 26d at equity RTH cadence. Fixed.

  Fix 2 — Drop label rows whose forward window crosses overnight.
          Forward sum of 48 5m residuals from bar t includes the close-to-open
          gap return when t falls in the last 4h of an RTH day. Those gaps are
          news/event-driven noise. Mask: keep only bars at position 0..29
          within day (where 0..77 = full RTH day, H=48). Loses ~62% of bars.

  Fix 3 — De-mean label per ts before training.
          fwd_resid_h has time-varying cross-sectional mean (drift, basket
          self-bias residue). De-meaning forces LGBM to learn rank/cross-
          sectional structure rather than aggregate level.

  Fix 4 — Leave-one-out basket.
          Each symbol residualizes against bk_ret_loo_i = (sum(ret) - ret_i)
          / (N - 1). Removes the 1/N self-bias in beta estimation.

Setup otherwise matches alpha_polygon_lgbm.py: 12 names, 6 expanding-window
folds, 5-seed LGBM ensemble, top-K=3 long/short rebalance every 48 bars
when forward window is clean intraday.
"""
from __future__ import annotations

import logging
import warnings
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data_collectors.polygon_loader import fetch_aggs

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

UNIVERSE = [
    "NVDA", "TSLA", "AMD", "AMZN", "GOOGL", "META",
    "AAPL", "MSFT", "ORCL", "INTC", "MU", "NFLX",
]

# === FIX 1: calendar-correct bar windows for equity RTH 5m ===
RTH_BARS_PER_DAY = 78
BARS_1H = 12
BARS_4H = 48
BARS_1D = 78          # was 288 (crypto)
BARS_5D = 390         # was 2016 = 7 calendar days at crypto; here 5 trading days
BETA_WINDOW = 390     # 1 trading week — stable beta for residualization

H = BARS_4H           # forward horizon = 4h
HOLD = BARS_4H        # rebalance cadence
TOP_K = 3
COST_BPS_DEFAULT = 24
SEEDS = (42, 7, 123, 99, 314)

LGB_PARAMS = dict(
    objective="regression", metric="rmse",
    num_leaves=31, max_depth=6, learning_rate=0.03,
    n_estimators=200, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=5, min_child_samples=200, verbose=-1,
)


# ---- data --------------------------------------------------------------

def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    et = df["ts"].dt.tz_convert("America/New_York")
    in_rth = (
        (et.dt.dayofweek < 5)
        & ((et.dt.hour > 9) | ((et.dt.hour == 9) & (et.dt.minute >= 30)))
        & (et.dt.hour < 16)
    )
    return df[in_rth].copy()


def load_panel() -> pd.DataFrame:
    frames = []
    for sym in UNIVERSE:
        df = fetch_aggs(sym, "5m")
        if df.empty:
            continue
        df = filter_rth(df)
        df["symbol"] = sym
        keep = ["ts", "symbol", "open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]]
        log.info("  %-6s n=%6d  %s -> %s", sym, len(df),
                 df["ts"].iloc[0].strftime("%Y-%m-%d"),
                 df["ts"].iloc[-1].strftime("%Y-%m-%d"))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---- returns + LOO basket ----------------------------------------------

def add_returns(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    panel["ret"] = (panel.groupby("symbol")["close"]
                    .transform(lambda s: np.log(s / s.shift(1))))
    return panel


def add_loo_basket(panel: pd.DataFrame) -> pd.DataFrame:
    """=== FIX 4 ===  Leave-one-out basket: each symbol's residualization
    uses bk_ret_loo_i = (sum_j ret_j - ret_i) / (N - 1) so it does NOT
    contain itself."""
    grp_ts = panel.groupby("ts")["ret"]
    total = grp_ts.transform("sum")        # NaNs treated as 0
    n = grp_ts.transform("count")           # excludes NaN
    panel["bk_ret"] = (total - panel["ret"].fillna(0)) / (n - 1).replace(0, np.nan)
    # Per-symbol cumulative basket index
    panel["bk_close"] = (panel.groupby("symbol", group_keys=False)["bk_ret"]
                          .apply(lambda s: (1.0 + s.fillna(0.0)).cumprod()))
    return panel


# ---- residualization ---------------------------------------------------

def add_residualization(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)

    def _beta(g):
        cov = (g["ret"] * g["bk_ret"]).rolling(BETA_WINDOW).mean() - \
              g["ret"].rolling(BETA_WINDOW).mean() * g["bk_ret"].rolling(BETA_WINDOW).mean()
        var = g["bk_ret"].rolling(BETA_WINDOW).var().replace(0, np.nan)
        return (cov / var).clip(-5, 5).shift(1)

    panel["beta"] = (panel.groupby("symbol", group_keys=False)
                     .apply(_beta).values)
    panel["resid"] = panel["ret"] - panel["beta"] * panel["bk_ret"]
    return panel


# ---- features (calendar-correct windows; per-symbol shifts internal) ---

def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    g = panel.groupby("symbol", group_keys=False)

    # ---- BASE ---------------------------------------------------------
    panel["return_1d"] = g["close"].apply(
        lambda s: s.pct_change(BARS_1D).shift(1))   # shift INSIDE apply (no cross-symbol leak)

    def _ema_slope(s):
        e = s.ewm(span=20, adjust=False).mean()
        return ((e - e.shift(BARS_1H)) / s.replace(0, np.nan)).shift(1)
    panel["ema_slope_20_1h"] = g["close"].apply(_ema_slope)

    def _atr(g_):
        h, l, c = g_["high"], g_["low"], g_["close"]
        tr = pd.concat([(h - l), (h - c.shift(1)).abs(),
                        (l - c.shift(1)).abs()], axis=1).max(axis=1)
        return (tr.rolling(14).mean() / c.replace(0, np.nan)).shift(1)
    panel["atr_pct"] = g.apply(_atr).reset_index(level=0, drop=True)

    panel["volume_ma_50"] = g["volume"].apply(
        lambda s: s.rolling(50).mean().shift(1))

    def _bsh(s):
        return s.rolling(BARS_1D).apply(
            lambda w: len(w) - 1 - int(np.argmax(w.values)), raw=False).shift(1)
    panel["bars_since_high"] = g["close"].apply(_bsh)

    h_of_d = panel["ts"].dt.hour + panel["ts"].dt.minute / 60.0
    panel["hour_cos"] = np.cos(2 * np.pi * h_of_d / 24.0)
    panel["hour_sin"] = np.sin(2 * np.pi * h_of_d / 24.0)

    # ---- CROSS / BASKET-RELATIVE  (per-symbol now, since LOO basket) ---
    panel["dom_level_vs_bk"] = np.log(panel["close"] / panel["bk_close"])

    for lag in (BARS_1H, BARS_4H, BARS_1D):
        panel[f"dom_change_{lag}b_vs_bk"] = (
            panel["dom_level_vs_bk"]
            - g["dom_level_vs_bk"].shift(lag)
        )

    for w, name in ((BARS_1D, "1d"), (BARS_5D, "5d")):
        rmean = g["dom_level_vs_bk"].apply(
            lambda s: s.rolling(w, min_periods=max(20, w // 4)).mean())
        rstd = g["dom_level_vs_bk"].apply(
            lambda s: s.rolling(w, min_periods=max(20, w // 4)).std()).replace(0, np.nan)
        panel[f"dom_z_{name}_vs_bk"] = ((panel["dom_level_vs_bk"] - rmean) / rstd).clip(-5, 5)

    # bk_ret_* and bk_realized_vol are now per-symbol (LOO basket)
    panel["bk_ret_1h"] = g["bk_close"].apply(lambda s: s.pct_change(BARS_1H))
    panel["bk_ret_4h"] = g["bk_close"].apply(lambda s: s.pct_change(BARS_4H))
    panel["bk_realized_vol_1h"] = g["bk_ret"].apply(
        lambda s: s.rolling(BARS_1H).std())

    panel["corr_1d_vs_bk"] = (g.apply(
        lambda gg: gg["ret"].rolling(BETA_WINDOW).corr(gg["bk_ret"]))
        .reset_index(level=0, drop=True)).clip(-1, 1)
    panel["corr_change_3d_vs_bk"] = (
        panel["corr_1d_vs_bk"]
        - g["corr_1d_vs_bk"].shift(3 * BARS_1D)
    )

    beta_pit = panel["beta"]
    panel["idio_ret_1h_vs_bk"] = (
        g["close"].apply(lambda s: s.pct_change(BARS_1H))
        - beta_pit * g["bk_close"].apply(lambda s: s.pct_change(BARS_1H))
    )
    panel["idio_ret_4h_vs_bk"] = (
        g["close"].apply(lambda s: s.pct_change(BARS_4H))
        - beta_pit * g["bk_close"].apply(lambda s: s.pct_change(BARS_4H))
    )
    panel["idio_vol_1h_vs_bk"] = g["resid"].apply(
        lambda s: s.rolling(BARS_1H).std())
    panel["idio_vol_1d_vs_bk"] = g["resid"].apply(
        lambda s: s.rolling(BARS_1D).std())

    # ---- FLOW ---------------------------------------------------------
    def _obv(g_):
        sign = np.sign(g_["close"].diff().fillna(0))
        return (sign * g_["volume"]).cumsum()
    panel["obv"] = g.apply(_obv).reset_index(level=0, drop=True)
    panel["obv_z_1d"] = (((panel["obv"] - g["obv"].apply(
        lambda s: s.rolling(BARS_1D).mean()))
        / g["obv"].apply(lambda s: s.rolling(BARS_1D).std()).replace(0, np.nan)
    ).clip(-5, 5)).shift(1)
    panel["obv_signal"] = (
        panel["obv"] - g["obv"].apply(lambda s: s.rolling(50).mean())
    ).shift(1)

    def _vwap(g_):
        date = g_["ts"].dt.date
        tp = (g_["high"] + g_["low"] + g_["close"]) / 3.0
        cum_tpv = (tp * g_["volume"]).groupby(date).cumsum()
        cum_v = g_["volume"].groupby(date).cumsum()
        return cum_tpv / cum_v.replace(0, np.nan)
    panel["vwap"] = g.apply(_vwap).reset_index(level=0, drop=True)
    panel["vwap_zscore"] = ((panel["close"] - panel["vwap"]) / panel["vwap"]
                             .replace(0, np.nan)).clip(-0.1, 0.1).shift(1)
    panel["vwap_slope_96"] = ((panel["vwap"] - g["vwap"].shift(96))
                                / panel["vwap"].replace(0, np.nan)).shift(1)

    def _mfi(g_):
        tp = (g_["high"] + g_["low"] + g_["close"]) / 3.0
        mf = tp * g_["volume"]
        pos = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
        return (100 - 100 / (1 + pos / neg.replace(0, np.nan))).shift(1)
    panel["mfi"] = g.apply(_mfi).reset_index(level=0, drop=True)

    for w in (10, 20):
        panel[f"price_volume_corr_{w}"] = g.apply(
            lambda gg: gg["close"].rolling(w).corr(gg["volume"]).shift(1)
        ).reset_index(level=0, drop=True)

    # ---- XS RANK ------------------------------------------------------
    for src in ("return_1d", "atr_pct", "ema_slope_20_1h",
                "idio_vol_1d_vs_bk", "obv_z_1d", "vwap_zscore",
                "bars_since_high"):
        if src in panel.columns:
            panel[f"{src}_xs_rank"] = panel.groupby("ts")[src].rank(pct=True)

    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    return panel


# ---- label (with overnight mask + per-ts demean) ------------------------

def add_label(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    """=== FIX 2 + FIX 3 ===
    fwd_resid_<h>      : raw H-bar forward residual sum (used for portfolio P&L)
    fwd_resid_<h>_clean: same, but NaN where forward window crosses overnight.
                         This is what's used for IC / model evaluation.
    fwd_resid_<h>_demean: clean version, minus per-ts cross-sectional mean.
                          Used for LGBM training so model learns rank-not-level.
    """
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    g = panel.groupby("symbol", group_keys=False)

    # raw forward sum (for P&L realization in portfolio)
    panel[f"fwd_resid_{h}"] = (
        g["resid"].apply(lambda s: s.rolling(h).sum().shift(-h)).values
    )

    # FIX 2: position within RTH day; mask if forward window crosses overnight
    et_date = panel["ts"].dt.tz_convert("America/New_York").dt.date
    panel["pos_in_day"] = (panel.groupby([panel["symbol"], et_date])
                                .cumcount())
    keep = panel["pos_in_day"] + h <= (RTH_BARS_PER_DAY - 1)
    panel[f"fwd_resid_{h}_clean"] = panel[f"fwd_resid_{h}"].where(keep, np.nan)

    # FIX 3: per-ts demean of clean label (for training)
    ts_mean = panel.groupby("ts")[f"fwd_resid_{h}_clean"].transform("mean")
    panel[f"fwd_resid_{h}_demean"] = panel[f"fwd_resid_{h}_clean"] - ts_mean
    return panel


# ---- folds + training --------------------------------------------------

def make_folds(panel: pd.DataFrame, train_min_days: int = 180,
               test_days: int = 90, embargo_bars: int = 60) -> list[tuple]:
    panel = panel.sort_values("ts")
    t0 = panel["ts"].min().normalize()
    t_max = panel["ts"].max()
    folds = []
    days = train_min_days
    while True:
        train_end = t0 + timedelta(days=days)
        test_start = train_end + timedelta(minutes=embargo_bars * 5)
        test_end = test_start + timedelta(days=test_days)
        if test_start >= t_max:
            break
        if test_end > t_max:
            test_end = t_max
        folds.append((train_end, test_start, test_end))
        days += test_days
    return folds


def fit_predict(train: pd.DataFrame, test: pd.DataFrame,
                features: list[str], train_label: str) -> pd.DataFrame:
    """Train on `train_label` (e.g. fwd_resid_48_demean). Predict on test."""
    train_ = train.dropna(subset=features + [train_label])
    if len(train_) < 1000:
        return pd.DataFrame()
    log.info("    train n=%d, test n=%d, features=%d",
             len(train_), len(test), len(features))
    preds = []
    sub = test.dropna(subset=features).copy()
    for seed in SEEDS:
        m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
        m.fit(train_[features], train_[train_label])
        preds.append(m.predict(sub[features]))
    sub["pred"] = np.mean(preds, axis=0)
    return sub


# ---- portfolio + metrics -----------------------------------------------

def portfolio(test_pred: pd.DataFrame, signal: str,
              pnl_label: str, top_k: int = TOP_K) -> pd.DataFrame:
    """Use `signal` for ranking, `pnl_label` (raw forward residual) for P&L.

    Only rebalance at bars where the clean-label condition holds (forward
    window doesn't cross overnight) — i.e., `fwd_resid_<h>_clean` is not NaN.
    That makes the strategy actually tradeable: enter at start-of-day bars,
    hold 4h, exit before close.
    """
    sub = test_pred.dropna(subset=[signal, pnl_label]).copy()
    # only consider bars where the position-in-day allowed clean labeling
    if "pos_in_day" in sub.columns:
        sub = sub[sub["pos_in_day"] + H <= (RTH_BARS_PER_DAY - 1)]

    unique_ts = sorted(sub["ts"].unique())
    if not unique_ts:
        return pd.DataFrame()
    rebal_ts = unique_ts[::HOLD]
    rows = []
    for ts in rebal_ts:
        bar = sub[sub["ts"] == ts]
        if len(bar) < 2 * top_k:
            continue
        bar = bar.sort_values(signal)
        long_leg = bar.tail(top_k)
        short_leg = bar.head(top_k)
        rows.append({
            "ts": ts,
            "spread_alpha": long_leg[pnl_label].mean()
                            - short_leg[pnl_label].mean(),
            "long_alpha": long_leg[pnl_label].mean(),
            "short_alpha": short_leg[pnl_label].mean(),
        })
    return pd.DataFrame(rows)


def metrics(pnl: pd.DataFrame, cost_bps: float) -> dict:
    if pnl.empty:
        return {"n": 0}
    pnl = pnl.copy()
    pnl["net"] = pnl["spread_alpha"] - cost_bps / 1e4
    rebals_per_year = (252 * RTH_BARS_PER_DAY) / HOLD
    g_sh = (pnl["spread_alpha"].mean() / pnl["spread_alpha"].std()
            * np.sqrt(rebals_per_year)) if pnl["spread_alpha"].std() > 0 else 0
    n_sh = (pnl["net"].mean() / pnl["net"].std()
            * np.sqrt(rebals_per_year)) if pnl["net"].std() > 0 else 0
    return {
        "n": len(pnl),
        "gross_bps": pnl["spread_alpha"].mean() * 1e4,
        "net_bps": pnl["net"].mean() * 1e4,
        "gross_sharpe": g_sh,
        "net_sharpe": n_sh,
        "hit_rate": float((pnl["spread_alpha"] > 0).mean()),
    }


def bootstrap_ci(pnl: pd.DataFrame, cost_bps: float,
                 block_days: int = 30, n_boot: int = 2000) -> tuple[float, float]:
    if pnl.empty:
        return np.nan, np.nan
    pnl = pnl.copy()
    pnl["net"] = pnl["spread_alpha"] - cost_bps / 1e4
    pnl["date"] = pnl["ts"].dt.date
    daily = pnl.groupby("date")["net"].sum()
    arr = daily.values
    if len(arr) < block_days * 2:
        return np.nan, np.nan
    n_blocks = max(1, len(arr) // block_days)
    rng = np.random.default_rng(42)
    sh = []
    for _ in range(n_boot):
        starts = rng.integers(0, len(arr) - block_days + 1, size=n_blocks)
        sample = np.concatenate([arr[s:s + block_days] for s in starts])
        if sample.std() > 0:
            sh.append(sample.mean() / sample.std() * np.sqrt(252))
    return (float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))) if sh else (np.nan, np.nan)


# ---- feature list ------------------------------------------------------

FEATURES = [
    "return_1d", "ema_slope_20_1h", "atr_pct", "volume_ma_50",
    "bars_since_high", "hour_cos", "hour_sin",
    "dom_level_vs_bk", "dom_change_12b_vs_bk", "dom_change_48b_vs_bk",
    "dom_change_78b_vs_bk", "dom_z_1d_vs_bk", "dom_z_5d_vs_bk",
    "bk_ret_1h", "bk_ret_4h", "bk_realized_vol_1h",
    "beta", "corr_1d_vs_bk", "corr_change_3d_vs_bk",
    "idio_ret_1h_vs_bk", "idio_ret_4h_vs_bk",
    "idio_vol_1h_vs_bk", "idio_vol_1d_vs_bk",
    "obv_z_1d", "obv_signal", "vwap_zscore", "mfi",
    "price_volume_corr_10", "price_volume_corr_20",
    "return_1d_xs_rank", "atr_pct_xs_rank", "ema_slope_20_1h_xs_rank",
    "idio_vol_1d_vs_bk_xs_rank", "obv_z_1d_xs_rank",
    "vwap_zscore_xs_rank", "bars_since_high_xs_rank",
    "sym_id",
]


# ---- main --------------------------------------------------------------

def main() -> None:
    log.info("loading polygon 5m panel for %d names...", len(UNIVERSE))
    panel = load_panel()
    if panel.empty:
        log.error("no data — fetch first via 'python3 -m data_collectors.polygon_loader'")
        return

    log.info("computing v6_clean features (calendar-correct + LOO basket)...")
    panel = add_returns(panel)
    panel = add_loo_basket(panel)
    panel = add_residualization(panel)
    panel = add_features(panel)
    panel = add_label(panel, H)

    sub = panel[panel["beta"].notna()]
    var_ratio = (sub.groupby("symbol")["resid"].var()
                 / sub.groupby("symbol")["ret"].var()).median()
    log.info("LOO residualization sanity: median beta=%.2f IQR=[%.2f,%.2f]  "
             "var(resid)/var(ret)=%.2f",
             sub["beta"].median(), sub["beta"].quantile(0.25),
             sub["beta"].quantile(0.75), var_ratio)

    n_total = len(panel)
    n_clean = panel[f"fwd_resid_{H}_clean"].notna().sum()
    log.info("overnight-mask: kept %d/%d label rows (%.0f%%)",
             n_clean, n_total, 100 * n_clean / n_total)

    feats = [f for f in FEATURES if f in panel.columns]
    folds = make_folds(panel, train_min_days=180, test_days=90)
    log.info("\nfolds:")
    for i, (te, ts_, te2) in enumerate(folds):
        log.info("  fold %d: train<=%s  test=[%s, %s]",
                 i + 1, te.strftime("%Y-%m-%d"),
                 ts_.strftime("%Y-%m-%d"), te2.strftime("%Y-%m-%d"))

    pnl_label = f"fwd_resid_{H}"
    train_label = f"fwd_resid_{H}_demean"

    lgbm_pnls = []
    simple_pnls = []
    for i, (train_end, test_start, test_end) in enumerate(folds):
        train = panel[panel["ts"] <= train_end].copy()
        test = panel[(panel["ts"] >= test_start) & (panel["ts"] <= test_end)].copy()
        log.info("\n>>> Fold %d (test %s -> %s)", i + 1,
                 test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d"))

        log.info("  [LGBM]")
        test_pred = fit_predict(train, test, feats, train_label)
        if not test_pred.empty:
            # carry pos_in_day for portfolio mask
            test_pred = test_pred.merge(
                test[["ts", "symbol", "pos_in_day"]], on=["ts", "symbol"], how="left")
            lp = portfolio(test_pred, "pred", pnl_label)
            if not lp.empty:
                m = metrics(lp, cost_bps=COST_BPS_DEFAULT)
                log.info("    n=%d gross=%+.1fbps net=%+.1fbps net_Sh=%+.2f hit=%.0f%%",
                         m["n"], m["gross_bps"], m["net_bps"],
                         m["net_sharpe"], 100 * m["hit_rate"])
                lp["fold"] = i + 1
                lgbm_pnls.append(lp)

        log.info("  [SIMPLE return_1d]")
        sp = portfolio(test, "return_1d", pnl_label)
        if not sp.empty:
            m = metrics(sp, cost_bps=COST_BPS_DEFAULT)
            log.info("    n=%d gross=%+.1fbps net=%+.1fbps net_Sh=%+.2f hit=%.0f%%",
                     m["n"], m["gross_bps"], m["net_bps"],
                     m["net_sharpe"], 100 * m["hit_rate"])
            sp["fold"] = i + 1
            simple_pnls.append(sp)

    log.info("\n=== STITCHED OOS METRICS (cost=%d bps) ===", COST_BPS_DEFAULT)
    log.info("  %-22s %6s %12s %12s %12s %18s",
             "strategy", "n", "gross/4h", "net/4h", "net_Sh", "95% CI")
    for label_, all_pnls in [("LGBM v2", lgbm_pnls), ("simple return_1d", simple_pnls)]:
        if not all_pnls:
            continue
        st = pd.concat(all_pnls, ignore_index=True)
        m = metrics(st, cost_bps=COST_BPS_DEFAULT)
        lo, hi = bootstrap_ci(st, cost_bps=COST_BPS_DEFAULT)
        log.info("  %-22s %6d %+10.1fbps %+10.1fbps %+12.2f  [%+.2f, %+.2f]",
                 label_, m["n"], m["gross_bps"], m["net_bps"],
                 m["net_sharpe"], lo, hi)

    if lgbm_pnls:
        st = pd.concat(lgbm_pnls, ignore_index=True)
        log.info("\n=== LGBM v2 cost sensitivity ===")
        log.info("  %-12s %-15s %-12s %-18s",
                 "cost (bps)", "net /4h", "net_Sharpe", "95% CI")
        for c in (0, 6, 12, 18, 24):
            m = metrics(st, cost_bps=c)
            lo, hi = bootstrap_ci(st, cost_bps=c)
            log.info("  %-12d %+13.1f %+12.2f  [%+.2f, %+.2f]",
                     c, m["net_bps"], m["net_sharpe"], lo, hi)


if __name__ == "__main__":
    main()
