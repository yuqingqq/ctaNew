"""LGBM walk-forward on 3 years of yfinance 1h cash equity data.

Addresses the legitimate complaint about the earlier 60d × 5m LGBM probe:
30 days of training per fold is way too thin for an ML model to extract
stable patterns. This run uses ~12 months of training per fold across a
3-year window — comparable train-data size to v6_clean's crypto baseline.

Setup:
  Universe: 12 mature US tech names (NVDA, TSLA, AMD, AMZN, GOOGL, META,
            AAPL, MSFT, ORCL, INTC, MU, NFLX). All have 3y of 1h history.
  Data: yfinance 1h, period=730d (yields ~3y in practice), RTH only.
  Anchor: equal-weight in-universe basket (~12 names).
  Features: v6_clean port with windows scaled to 1h cadence:
            1h = 1 bar, 4h = 4 bars, 1d = 7 bars (RTH), 7d = 49 bars.
            ~25 features (base + cross + flow + xs_rank), no funding.
  Label: fwd_resid_4h = 4-bar forward sum of residuals (same calendar
            time as 5m × h=48).
  Hold: 4 bars. Rebalance every 4 bars from start of each test fold.

Walk-forward (expanding window, 4 folds):
  Fold 1: train days 0-365,  test days 365-545
  Fold 2: train days 0-545,  test days 545-725
  Fold 3: train days 0-725,  test days 725-905
  Fold 4: train days 0-905,  test days 905-end
  Embargo: 4 bars + 12 buffer = 16 bars between train/test.

Compare:
  Strategy A: LGBM ensemble (5 seeds), 25 features.
  Strategy B: Simple rule (long top-K by trailing 1d return). Baseline.

Cost: 24 bps RT / rebalance. Also report 0/12/36 bps.
"""
from __future__ import annotations

import logging
import warnings
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"

UNIVERSE = [
    "NVDA", "TSLA", "AMD", "AMZN", "GOOGL", "META",
    "AAPL", "MSFT", "ORCL", "INTC", "MU", "NFLX",
]

# 1h cadence, RTH only → ~7 bars/day
INTERVAL = "1h"
PERIOD = "730d"
BARS_1H = 1
BARS_4H = 4
BARS_1D = 7
BARS_7D = 49
BETA_WINDOW = 49           # ~7 RTH days
HOLD_BARS = 4              # 4h hold
H = 4                       # forward horizon = 4 bars = 4h
TOP_K = 3
COST_BPS = 24
SEEDS = (42, 7, 123, 99, 314)
BARS_PER_RTH_YEAR = 252 * 7

LGB_PARAMS = dict(
    objective="regression", metric="rmse",
    num_leaves=31, max_depth=6, learning_rate=0.03,
    n_estimators=300, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=5, min_child_samples=200, verbose=-1,
)


# ---- data --------------------------------------------------------------

def fetch_yf(symbol: str) -> pd.DataFrame:
    cache = CACHE / f"yf_{symbol}_1h_long.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = yf.Ticker(symbol).history(period=PERIOD, interval=INTERVAL,
                                   auto_adjust=True, prepost=False)
    if df.empty:
        return df
    df.index = df.index.tz_convert("UTC")
    df = df.reset_index().rename(columns={
        "Datetime": "ts", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["symbol"] = symbol
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    df.to_parquet(cache)
    return df


def load_panel() -> pd.DataFrame:
    frames = []
    for s in UNIVERSE:
        df = fetch_yf(s)
        if df.empty:
            continue
        log.info("  %-6s n=%5d  %s -> %s", s, len(df),
                 df["ts"].iloc[0].strftime("%Y-%m-%d"),
                 df["ts"].iloc[-1].strftime("%Y-%m-%d"))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---- features (v6_clean port, 1h cadence) ------------------------------

def add_returns(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    panel["ret"] = (panel.groupby("symbol")["close"]
                    .transform(lambda s: np.log(s / s.shift(1))))
    return panel


def build_basket(panel: pd.DataFrame) -> pd.DataFrame:
    bk_ret = panel.groupby("ts")["ret"].mean().rename("bk_ret")
    bk_close = (1.0 + bk_ret.fillna(0.0)).cumprod().rename("bk_close")
    return pd.concat([bk_ret, bk_close], axis=1).reset_index()


def add_features_and_resid(panel: pd.DataFrame, bk: pd.DataFrame) -> pd.DataFrame:
    panel = panel.merge(bk, on="ts", how="left").sort_values(["symbol", "ts"]).reset_index(drop=True)
    g = panel.groupby("symbol", group_keys=False)

    # rolling beta (1h cadence: 49 bars = ~7 RTH days)
    def _beta(gg):
        cov = (gg["ret"] * gg["bk_ret"]).rolling(BETA_WINDOW).mean() - \
              gg["ret"].rolling(BETA_WINDOW).mean() * gg["bk_ret"].rolling(BETA_WINDOW).mean()
        var = gg["bk_ret"].rolling(BETA_WINDOW).var().replace(0, np.nan)
        return (cov / var).clip(-5, 5).shift(1)
    panel["beta_short_vs_bk"] = g.apply(_beta).reset_index(level=0, drop=True)
    panel["resid"] = panel["ret"] - panel["beta_short_vs_bk"] * panel["bk_ret"]

    # base
    panel["return_1d"] = g["close"].apply(lambda s: s.pct_change(BARS_1D)).shift(1)
    def _ema_slope(s):
        e = s.ewm(span=20, adjust=False).mean()
        return (e - e.shift(BARS_1H)) / s.replace(0, np.nan)
    panel["ema_slope_20_1h"] = g["close"].apply(_ema_slope).shift(1)
    def _atr(gg):
        h, l, c = gg["high"], gg["low"], gg["close"]
        tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        return (tr.rolling(14).mean() / c.replace(0, np.nan))
    panel["atr_pct"] = g.apply(_atr).reset_index(level=0, drop=True).shift(1)
    panel["volume_ma_50"] = g["volume"].apply(lambda s: s.rolling(50).mean()).shift(1)
    def _bsh(s):
        return s.rolling(BARS_1D).apply(lambda w: len(w) - 1 - int(np.argmax(w.values)), raw=False)
    panel["bars_since_high"] = g["close"].apply(_bsh).shift(1)
    h_of_d = panel["ts"].dt.hour + panel["ts"].dt.minute / 60.0
    panel["hour_cos"] = np.cos(2 * np.pi * h_of_d / 24.0)
    panel["hour_sin"] = np.sin(2 * np.pi * h_of_d / 24.0)

    # cross / basket-relative
    panel["dom_level_vs_bk"] = np.log(panel["close"] / panel["bk_close"])
    for lag in (BARS_1H, BARS_4H, BARS_1D):
        panel[f"dom_change_{lag}b_vs_bk"] = (panel["dom_level_vs_bk"]
                                              - g["dom_level_vs_bk"].shift(lag))
    for w, name in ((BARS_1D, "1d"), (BARS_7D, "7d")):
        rmean = g["dom_level_vs_bk"].apply(
            lambda s: s.rolling(w, min_periods=max(7, w // 4)).mean())
        rstd = g["dom_level_vs_bk"].apply(
            lambda s: s.rolling(w, min_periods=max(7, w // 4)).std()).replace(0, np.nan)
        panel[f"dom_z_{name}_vs_bk"] = ((panel["dom_level_vs_bk"] - rmean) / rstd).clip(-5, 5)

    bk_close_s = panel["bk_close"]
    for h in (BARS_1H, BARS_4H):
        panel[f"bk_ret_{h}b"] = bk_close_s.pct_change(h)
    bk_ema_long = bk_close_s.ewm(span=BARS_4H, adjust=False).mean()
    panel["bk_ema_slope_4h"] = ((bk_ema_long - bk_ema_long.shift(BARS_1H))
                                 / bk_close_s.replace(0, np.nan))
    # bk_realized_vol_1h: degenerate at 1h cadence (window=1 → NaN). Use 4h instead.
    panel["bk_realized_vol_4h"] = panel["bk_ret"].rolling(BARS_4H).std()

    def _rolling_corr(gg):
        cov = (gg["ret"] * gg["bk_ret"]).rolling(BETA_WINDOW).mean() - \
              gg["ret"].rolling(BETA_WINDOW).mean() * gg["bk_ret"].rolling(BETA_WINDOW).mean()
        std_r = gg["ret"].rolling(BETA_WINDOW).std()
        std_b = gg["bk_ret"].rolling(BETA_WINDOW).std()
        return (cov / (std_r * std_b).replace(0, np.nan)).clip(-1, 1)
    panel["corr_1d_vs_bk"] = g.apply(_rolling_corr).reset_index(level=0, drop=True)
    panel["corr_change_3d_vs_bk"] = (panel["corr_1d_vs_bk"]
                                       - g["corr_1d_vs_bk"].shift(3 * BARS_1D))

    beta_pit = panel["beta_short_vs_bk"]
    idio_1bar = panel["ret"] - beta_pit * panel["bk_ret"]
    panel["idio_1bar"] = idio_1bar
    for h in (BARS_1H, BARS_4H):
        my_h = g["close"].apply(lambda s: s.pct_change(h))
        bk_h = bk_close_s.pct_change(h)
        panel[f"idio_ret_{h}b_vs_bk"] = my_h - beta_pit * bk_h
    # idio_vol_1h_vs_bk: degenerate at 1h cadence. Use 4h instead.
    panel["idio_vol_4h_vs_bk"] = idio_1bar.rolling(BARS_4H).std()
    panel["idio_vol_1d_vs_bk"] = idio_1bar.rolling(BARS_1D).std()

    # flow
    def _obv(gg):
        sign = np.sign(gg["close"].diff().fillna(0))
        return (sign * gg["volume"]).cumsum()
    panel["obv"] = g.apply(_obv).reset_index(level=0, drop=True)
    panel["obv_z_1d"] = ((panel["obv"] - g["obv"].apply(
        lambda s: s.rolling(BARS_1D).mean()))
        / g["obv"].apply(lambda s: s.rolling(BARS_1D).std()).replace(0, np.nan)
        ).clip(-5, 5).shift(1)
    panel["obv_signal"] = ((panel["obv"] - g["obv"].apply(
        lambda s: s.rolling(50).mean())).shift(1))

    def _vwap(gg):
        date = gg["ts"].dt.date
        tp = (gg["high"] + gg["low"] + gg["close"]) / 3.0
        cum_tpv = (tp * gg["volume"]).groupby(date).cumsum()
        cum_v = gg["volume"].groupby(date).cumsum()
        return cum_tpv / cum_v.replace(0, np.nan)
    panel["vwap"] = g.apply(_vwap).reset_index(level=0, drop=True)
    panel["vwap_zscore"] = ((panel["close"] - panel["vwap"]) / panel["vwap"]
                             .replace(0, np.nan)).clip(-0.1, 0.1).shift(1)

    def _mfi(gg):
        tp = (gg["high"] + gg["low"] + gg["close"]) / 3.0
        mf = tp * gg["volume"]
        pos = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
        return 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    panel["mfi"] = g.apply(_mfi).reset_index(level=0, drop=True).shift(1)

    for w in (10, 20):
        panel[f"price_volume_corr_{w}"] = (
            g.apply(lambda gg: gg["close"].rolling(w).corr(gg["volume"]))
             .reset_index(level=0, drop=True).shift(1)
        )

    # xs rank
    for src in ("return_1d", "atr_pct", "ema_slope_20_1h",
                "idio_vol_1d_vs_bk", "obv_z_1d", "vwap_zscore",
                "bars_since_high"):
        panel[f"{src}_xs_rank"] = panel.groupby("ts")[src].rank(pct=True)

    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    return panel


def add_label(panel: pd.DataFrame) -> pd.DataFrame:
    g = panel.groupby("symbol", group_keys=False)
    panel["fwd_resid_4h"] = g["idio_1bar"].apply(
        lambda s: s.rolling(H).sum().shift(-H)).values
    return panel


FEATURES = [
    "return_1d", "ema_slope_20_1h", "atr_pct", "volume_ma_50",
    "bars_since_high", "hour_cos", "hour_sin",
    "dom_level_vs_bk", "dom_change_1b_vs_bk", "dom_change_4b_vs_bk",
    "dom_change_7b_vs_bk", "dom_z_1d_vs_bk", "dom_z_7d_vs_bk",
    "bk_ret_1b", "bk_ret_4b", "bk_realized_vol_4h",
    "beta_short_vs_bk", "corr_1d_vs_bk", "corr_change_3d_vs_bk",
    "idio_ret_1b_vs_bk", "idio_ret_4b_vs_bk",
    "idio_vol_4h_vs_bk", "idio_vol_1d_vs_bk",
    "obv_z_1d", "obv_signal", "vwap_zscore", "mfi",
    "price_volume_corr_10", "price_volume_corr_20",
    "return_1d_xs_rank", "atr_pct_xs_rank", "ema_slope_20_1h_xs_rank",
    "idio_vol_1d_vs_bk_xs_rank", "obv_z_1d_xs_rank",
    "vwap_zscore_xs_rank", "bars_since_high_xs_rank",
    "sym_id",
]


# ---- walk-forward ------------------------------------------------------

def make_folds(panel: pd.DataFrame, train_min_days: int = 365,
               test_days: int = 180, embargo_bars: int = 16) -> list[tuple]:
    panel = panel.sort_values("ts")
    t0 = panel["ts"].min().normalize()
    t_max = panel["ts"].max()
    folds = []
    days = train_min_days
    while True:
        train_end = t0 + timedelta(days=days)
        test_start = train_end + timedelta(hours=embargo_bars)
        test_end = test_start + timedelta(days=test_days)
        if test_start >= t_max:
            break
        if test_end > t_max:
            test_end = t_max
        folds.append((train_end, test_start, test_end))
        days += test_days
    return folds


def fit_predict(train: pd.DataFrame, test: pd.DataFrame,
                features: list[str], label: str) -> pd.DataFrame:
    train_ = train.dropna(subset=features + [label])
    if len(train_) < 1000:
        log.warning("  train size %d too small, skipping", len(train_))
        return pd.DataFrame()
    log.info("  train n=%d, test n=%d, features=%d",
             len(train_), len(test), len(features))
    preds = []
    for seed in SEEDS:
        m = lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
        m.fit(train_[features], train_[label])
        sub = test.dropna(subset=features).copy()
        sub["pred"] = m.predict(sub[features])
        preds.append(sub["pred"].values)
    sub = test.dropna(subset=features).copy()
    sub["pred"] = np.mean(preds, axis=0)
    return sub


# ---- portfolio ---------------------------------------------------------

def portfolio(test_pred: pd.DataFrame, signal: str, label: str,
              top_k: int = TOP_K) -> pd.DataFrame:
    sub = test_pred.dropna(subset=[signal, label]).copy()
    unique_ts = sorted(sub["ts"].unique())
    rebal_ts = unique_ts[::HOLD_BARS]
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
            "spread_alpha": long_leg[label].mean() - short_leg[label].mean(),
            "long_alpha": long_leg[label].mean(),
            "short_alpha": short_leg[label].mean(),
        })
    return pd.DataFrame(rows)


def metrics(pnl: pd.DataFrame, cost_bps: float = COST_BPS) -> dict:
    if pnl.empty:
        return {"n": 0}
    pnl = pnl.copy()
    pnl["net"] = pnl["spread_alpha"] - cost_bps / 1e4
    rebals_per_year = BARS_PER_RTH_YEAR / HOLD_BARS
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


def bootstrap_ci(pnl: pd.DataFrame, cost_bps: float = COST_BPS,
                 block_days: int = 30, n_boot: int = 2000) -> tuple[float, float]:
    if pnl.empty or len(pnl) < block_days * 2:
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
    if not sh:
        return np.nan, np.nan
    return float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))


# ---- main --------------------------------------------------------------

def main() -> None:
    log.info("loading 12-name 1h panel ~3y...")
    panel = load_panel()
    panel = add_returns(panel)
    bk = build_basket(panel)
    log.info("computing v6_clean features (1h-scaled)...")
    panel = add_features_and_resid(panel, bk)
    panel = add_label(panel)

    folds = make_folds(panel, train_min_days=365, test_days=180)
    log.info("\nfolds:")
    for i, (te, ts_, te2) in enumerate(folds):
        log.info("  fold %d: train<=%s  test=[%s, %s]",
                 i + 1, te.strftime("%Y-%m-%d"),
                 ts_.strftime("%Y-%m-%d"),
                 te2.strftime("%Y-%m-%d"))

    label = "fwd_resid_4h"
    feats = [f for f in FEATURES if f in panel.columns]

    lgbm_pnls = []
    simple_pnls = []
    for i, (train_end, test_start, test_end) in enumerate(folds):
        train = panel[panel["ts"] <= train_end].copy()
        test = panel[(panel["ts"] >= test_start) & (panel["ts"] <= test_end)].copy()

        log.info("\n>>> Fold %d (test %s -> %s)", i + 1,
                 test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d"))

        # LGBM
        log.info("  [LGBM]")
        test_pred = fit_predict(train, test, feats, label)
        if not test_pred.empty:
            lp = portfolio(test_pred, "pred", label)
            if not lp.empty:
                m = metrics(lp)
                log.info("    n=%d gross=%+.1f net=%+.1f net_sharpe=%+.2f hit=%.0f%%",
                         m["n"], m["gross_bps"], m["net_bps"],
                         m["net_sharpe"], 100 * m["hit_rate"])
                lp["fold"] = i + 1
                lgbm_pnls.append(lp)

        # Simple baseline (long top-K return_1d)
        log.info("  [SIMPLE return_1d]")
        sp = portfolio(test, "return_1d", label)
        if not sp.empty:
            m = metrics(sp)
            log.info("    n=%d gross=%+.1f net=%+.1f net_sharpe=%+.2f hit=%.0f%%",
                     m["n"], m["gross_bps"], m["net_bps"],
                     m["net_sharpe"], 100 * m["hit_rate"])
            sp["fold"] = i + 1
            simple_pnls.append(sp)

    log.info("\n=== STITCHED OOS METRICS (cost=%d bps) ===", COST_BPS)
    log.info("  %-22s %6s %12s %12s %12s %18s",
             "strategy", "n", "gross/4h", "net/4h", "net_Sh", "95% CI")
    for label_, all_pnls in [("LGBM", lgbm_pnls), ("simple_return_1d", simple_pnls)]:
        if not all_pnls:
            continue
        st = pd.concat(all_pnls, ignore_index=True)
        m = metrics(st)
        lo, hi = bootstrap_ci(st)
        log.info("  %-22s %6d %+10.1fbps %+10.1fbps %+12.2f  [%+.2f, %+.2f]",
                 label_, m["n"], m["gross_bps"], m["net_bps"],
                 m["net_sharpe"], lo, hi)

    # Cost sensitivity for LGBM
    if lgbm_pnls:
        st = pd.concat(lgbm_pnls, ignore_index=True)
        log.info("\n=== LGBM cost sensitivity ===")
        log.info("  %-12s %-15s %-12s %-18s",
                 "cost (bps)", "net /4h (bps)", "net Sharpe", "95% CI")
        for c in (0, 6, 12, 18, 24, 36):
            m = metrics(st, cost_bps=c)
            lo, hi = bootstrap_ci(st, cost_bps=c)
            log.info("  %-12d %+13.1f %+12.2f  [%+.2f, %+.2f]",
                     c, m["net_bps"], m["net_sharpe"], lo, hi)


if __name__ == "__main__":
    main()
