"""Option B probe: v6_clean cross-sectional features on yfinance 5m cash equity.

Tests whether the v6_clean feature set (developed and selected on 5m crypto
perps) shows IC against forward residual returns on REAL US-equity 5m bars
during regular trading hours. This isolates the question "do these features
have any grip on equity-residual alpha" from the xyz-perp-tracking-quality
question.

If positive → port to xyz with confidence (we have already shown daily-corr
is 0.99 between xyz and cash) and add basis as an additional feature.
If negative → the alpha-residual idea is dead at this horizon regardless of
where execution happens.

Setup:
  Universe: 18 liquid US tech names (same as alpha_xyz_probe.py).
  Data: yfinance 5m, period=60d, RTH only, auto_adjust=True (split/div-aware).
  Anchor: equal-weight in-universe basket. The cleanest cross-sectional
          residualizer (validated in alpha_xyz_probe vs SP500 / XYZ100).
  Horizon: h=48 5m bars = 4h forward, native v6_clean cadence.
           Also h=12 (1h forward) as robustness check; cross-session
           contamination at h=48 (last 4h of the day spans overnight).

Features (~25, ported from features_ml/cross_sectional.py with same BAR
counts as crypto v6_clean — different calendar windows):
  base (5):       return_1d (288b), ema_slope_20_1h (20b), atr_pct (14b),
                  volume_ma_50, hour_cos / hour_sin (encoded on RTH minute)
  cross/bk (8):   dom_level, dom_z_1d (288b), dom_z_7d (2016b),
                  dom_change_12b/48b/288b, bk_ret_48b, bk_ema_slope_4h,
                  idio_ret_12b/48b, idio_vol_1h/1d, corr_change_3d
  flow (7):       obv_z_1d, vwap_slope_96, vwap_zscore, mfi_14,
                  obv_signal, price_volume_corr_10, price_volume_corr_20
  xs rank (7):    pctile rank across universe at each ts of:
                  return_1d, atr_pct, ema_slope_20_1h, idio_vol_1d,
                  obv_z_1d, vwap_zscore, bars_since_high

NOT included (vs v6_clean): funding-rate features (no funding on cash equity).

All features computed once per (symbol, ts), then .shift(1) before being
ranked / used → point-in-time. fwd_resid_h is .shift(-h) of rolling sum
of residuals.

IC: per-timestamp Spearman between cross-sectional rank of feature and
fwd_resid_h. Aggregated by mean. 95% CI from 7-day block bootstrap on
daily-mean IC.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    "NVDA", "TSLA", "AMD", "AMZN", "GOOGL", "META", "COIN", "MSTR",
    "PLTR", "INTC", "MU", "SNDK", "HOOD", "AAPL", "MSFT", "ORCL",
    "NFLX", "CRCL",
]

INTERVAL = "5m"
DAYS_BACK = 60
HORIZONS = (12, 48)         # 1h, 4h forward (in 5m bars)
EMBARGO = 12                # 1h embargo for label purging
BOOT_N = 1000
BLOCK_DAYS = 5

# v6_clean window constants (in BARS, scale-invariant). Kept identical to
# the crypto pipeline: same statistical-lag depth, just on equity 5m data.
BARS_1H = 12
BARS_4H = 48
BARS_1D = 288
BARS_7D = 2016
BETA_WINDOW = 288


# ---- data fetch --------------------------------------------------------

def fetch_yf(symbol: str) -> pd.DataFrame:
    cache = CACHE / f"yf_{symbol}_{INTERVAL}_{DAYS_BACK}d_full.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    import yfinance as yf
    df = yf.Ticker(symbol).history(
        period=f"{DAYS_BACK}d", interval=INTERVAL,
        auto_adjust=True, prepost=False,
    )
    if df.empty:
        return df
    df.index = df.index.tz_convert("UTC")
    df = df.reset_index().rename(columns={
        "Datetime": "ts", "Date": "ts",
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["symbol"] = symbol
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    df.to_parquet(cache)
    return df


def load_panel() -> pd.DataFrame:
    frames = []
    for sym in UNIVERSE:
        df = fetch_yf(sym)
        if df.empty:
            log.warning("  %s: empty", sym)
            continue
        log.info("  %-6s n=%5d  %s -> %s",
                 sym, len(df),
                 df["ts"].iloc[0].strftime("%Y-%m-%d %H:%M"),
                 df["ts"].iloc[-1].strftime("%Y-%m-%d %H:%M"))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---- features ----------------------------------------------------------

def add_returns(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    panel["ret"] = (panel.groupby("symbol")["close"]
                    .transform(lambda s: np.log(s / s.shift(1))))
    return panel


def build_basket(panel: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight in-universe basket return + cumulative basket index."""
    bk_ret = panel.groupby("ts")["ret"].mean().rename("bk_ret")
    bk_close = (1.0 + bk_ret.fillna(0.0)).cumprod().rename("bk_close")
    bk = pd.concat([bk_ret, bk_close], axis=1).reset_index()
    return bk


def add_base_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    g = panel.groupby("symbol", group_keys=False)

    # return_1d: 288-bar return
    panel["return_1d"] = g["close"].apply(lambda s: s.pct_change(BARS_1D)).shift(1)
    # ema_slope_20_1h: slope of 20-bar EMA
    def ema_slope(s):
        e = s.ewm(span=20, adjust=False).mean()
        return ((e - e.shift(BARS_1H)) / s.replace(0, np.nan))
    panel["ema_slope_20_1h"] = g["close"].apply(ema_slope).shift(1)
    # atr_pct: 14-bar ATR / close
    def atr(g_):
        h, l, c = g_["high"], g_["low"], g_["close"]
        tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        return (tr.rolling(14).mean() / c.replace(0, np.nan))
    panel["atr_pct"] = g.apply(atr).reset_index(level=0, drop=True).shift(1)
    # volume_ma_50: 50-bar volume mean
    panel["volume_ma_50"] = g["volume"].apply(lambda s: s.rolling(50).mean()).shift(1)
    # bars_since_high: 288-bar trailing argmax distance
    def bsh(s):
        return s.rolling(BARS_1D).apply(
            lambda w: len(w) - 1 - int(np.argmax(w.values)), raw=False)
    panel["bars_since_high"] = g["close"].apply(bsh).shift(1)
    # hour cyclical encoding (using UTC hour)
    h = panel["ts"].dt.hour + panel["ts"].dt.minute / 60.0
    panel["hour_cos"] = np.cos(2 * np.pi * h / 24.0)
    panel["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
    return panel


def add_cross_features(panel: pd.DataFrame, bk: pd.DataFrame) -> pd.DataFrame:
    panel = panel.merge(bk, on="ts", how="left")
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    g = panel.groupby("symbol", group_keys=False)

    # 1. Dominance: log(close / bk_close)
    panel["dom_level_vs_bk"] = np.log(panel["close"] / panel["bk_close"])
    for h in (12, 48, 288):
        panel[f"dom_change_{h}b_vs_bk"] = (panel["dom_level_vs_bk"]
                                            - g["dom_level_vs_bk"].shift(h))
    for w, name in ((BARS_1D, "1d"), (BARS_7D, "7d")):
        rmean = g["dom_level_vs_bk"].apply(
            lambda s: s.rolling(w, min_periods=max(48, w // 4)).mean())
        rstd = g["dom_level_vs_bk"].apply(
            lambda s: s.rolling(w, min_periods=max(48, w // 4)).std()).replace(0, np.nan)
        panel[f"dom_z_{name}_vs_bk"] = ((panel["dom_level_vs_bk"] - rmean) / rstd).clip(-5, 5)

    # 2. Basket recent state
    bk_close_s = panel["bk_close"]
    for h in (BARS_1H, BARS_4H):
        panel[f"bk_ret_{h}b"] = bk_close_s.pct_change(h)
    bk_ema_long = bk_close_s.ewm(span=BARS_4H, adjust=False).mean()
    panel["bk_ema_slope_4h"] = ((bk_ema_long - bk_ema_long.shift(12))
                                 / bk_close_s.replace(0, np.nan))
    panel["bk_realized_vol_1h"] = panel["bk_ret"].rolling(BARS_1H).std()

    # 3. Beta and correlation
    def _rolling_beta(g_):
        ret = g_["ret"]
        bret = g_["bk_ret"]
        cov = (ret * bret).rolling(BETA_WINDOW).mean() - \
              ret.rolling(BETA_WINDOW).mean() * bret.rolling(BETA_WINDOW).mean()
        var = bret.rolling(BETA_WINDOW).var().replace(0, np.nan)
        beta = (cov / var).clip(-5, 5)
        return beta.shift(1)
    panel["beta_short_vs_bk"] = g.apply(_rolling_beta).reset_index(level=0, drop=True)

    def _rolling_corr(g_):
        ret, bret = g_["ret"], g_["bk_ret"]
        cov = (ret * bret).rolling(BETA_WINDOW).mean() - \
              ret.rolling(BETA_WINDOW).mean() * bret.rolling(BETA_WINDOW).mean()
        std_r = ret.rolling(BETA_WINDOW).std()
        std_b = bret.rolling(BETA_WINDOW).std()
        return (cov / (std_r * std_b).replace(0, np.nan)).clip(-1, 1)
    panel["corr_1d_vs_bk"] = g.apply(_rolling_corr).reset_index(level=0, drop=True)
    panel["corr_change_3d_vs_bk"] = (panel["corr_1d_vs_bk"]
                                      - g["corr_1d_vs_bk"].shift(3 * BARS_1D))

    # 4. Idiosyncratic returns / vol
    beta_pit = panel["beta_short_vs_bk"]
    idio_1bar = panel["ret"] - beta_pit * panel["bk_ret"]
    panel["idio_1bar"] = idio_1bar  # used to build forward residual label
    for h in (BARS_1H, BARS_4H):
        my_h = g["close"].apply(lambda s: s.pct_change(h))
        bk_h = bk_close_s.pct_change(h)
        panel[f"idio_ret_{h}b_vs_bk"] = my_h - beta_pit * bk_h
    panel["idio_vol_1h_vs_bk"] = idio_1bar.rolling(BARS_1H).std()
    panel["idio_vol_1d_vs_bk"] = idio_1bar.rolling(BARS_1D).std()
    return panel


def add_flow_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["symbol", "ts"]).reset_index(drop=True)
    g = panel.groupby("symbol", group_keys=False)

    # OBV: cumulative volume signed by close direction
    def _obv(g_):
        sign = np.sign(g_["close"].diff().fillna(0))
        return (sign * g_["volume"]).cumsum()
    panel["obv"] = g.apply(_obv).reset_index(level=0, drop=True)
    panel["obv_z_1d"] = ((panel["obv"] - g["obv"].apply(
        lambda s: s.rolling(BARS_1D).mean()))
        / g["obv"].apply(lambda s: s.rolling(BARS_1D).std()).replace(0, np.nan)
        ).clip(-5, 5).shift(1)
    panel["obv_signal"] = ((panel["obv"] - g["obv"].apply(
        lambda s: s.rolling(50).mean())).shift(1))

    # VWAP cumulative within-day (typical price * vol cumsum / vol cumsum)
    def _vwap(g_):
        # within-day VWAP: reset cumsum every UTC day
        date = g_["ts"].dt.date
        tp = (g_["high"] + g_["low"] + g_["close"]) / 3.0
        tpv = tp * g_["volume"]
        cum_tpv = tpv.groupby(date).cumsum()
        cum_v = g_["volume"].groupby(date).cumsum()
        vwap = cum_tpv / cum_v.replace(0, np.nan)
        return vwap
    panel["vwap"] = g.apply(_vwap).reset_index(level=0, drop=True)
    panel["vwap_zscore"] = ((panel["close"] - panel["vwap"]) / panel["vwap"]
                            .replace(0, np.nan)).clip(-0.1, 0.1).shift(1)
    panel["vwap_slope_96"] = ((panel["vwap"] - g["vwap"].shift(96))
                               / panel["vwap"].replace(0, np.nan)).shift(1)

    # MFI 14
    def _mfi(g_):
        tp = (g_["high"] + g_["low"] + g_["close"]) / 3.0
        mf = tp * g_["volume"]
        pos = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
        return 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    panel["mfi"] = g.apply(_mfi).reset_index(level=0, drop=True).shift(1)

    # price-volume correlations
    def _pv_corr(s_close, s_vol, w):
        return s_close.rolling(w).corr(s_vol)
    for w in (10, 20):
        panel[f"price_volume_corr_{w}"] = (
            g.apply(lambda gg: _pv_corr(gg["close"], gg["volume"], w))
             .reset_index(level=0, drop=True).shift(1)
        )
    return panel


def add_xs_rank_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional pctile rank at each ts."""
    sources = {
        "return_1d_xs_rank": "return_1d",
        "atr_pct_xs_rank": "atr_pct",
        "ema_slope_20_1h_xs_rank": "ema_slope_20_1h",
        "idio_vol_1d_vs_bk_xs_rank": "idio_vol_1d_vs_bk",
        "obv_z_1d_xs_rank": "obv_z_1d",
        "vwap_zscore_xs_rank": "vwap_zscore",
        "bars_since_high_xs_rank": "bars_since_high",
    }
    for new, src in sources.items():
        panel[new] = panel.groupby("ts")[src].rank(pct=True)
    return panel


def add_label(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    """Forward residual return summed over h bars, .shift(-h) so at bar t
    we have sum(idio_1bar[t+1..t+h]). Embargoed h+EMBARGO from training."""
    g = panel.groupby("symbol", group_keys=False)
    panel[f"fwd_resid_{h}"] = (
        g["idio_1bar"].apply(
            lambda s: s.rolling(h).sum().shift(-h)
        ).values
    )
    return panel


# ---- IC + bootstrap ----------------------------------------------------

def per_ts_ic(panel: pd.DataFrame, feat: str, label: str) -> pd.Series:
    sub = panel.dropna(subset=[feat, label])
    def _ic(g):
        if len(g) < 4: return np.nan
        r, _ = spearmanr(g[feat], g[label])
        return r
    return sub.groupby("ts").apply(_ic).dropna()


def block_bootstrap_ci(daily_ic: pd.Series,
                       n: int = BOOT_N, block_days: int = BLOCK_DAYS) -> tuple:
    daily = daily_ic.resample("1D").mean().dropna()
    arr = daily.values
    if len(arr) < block_days * 2:
        return float(arr.mean()) if len(arr) else np.nan, np.nan, np.nan
    n_blocks = max(1, len(arr) // block_days)
    rng = np.random.default_rng(42)
    means = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, len(arr) - block_days + 1, size=n_blocks)
        sample = np.concatenate([arr[s:s + block_days] for s in starts])
        means[i] = sample.mean()
    return (float(arr.mean()),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


# ---- main --------------------------------------------------------------

FEATURE_GROUPS = {
    "BASE": ["return_1d", "ema_slope_20_1h", "atr_pct",
             "volume_ma_50", "bars_since_high", "hour_cos", "hour_sin"],
    "CROSS": ["dom_level_vs_bk", "dom_z_1d_vs_bk", "dom_z_7d_vs_bk",
              "dom_change_12b_vs_bk", "dom_change_48b_vs_bk",
              "dom_change_288b_vs_bk", "bk_ret_12b", "bk_ret_48b",
              "bk_ema_slope_4h", "bk_realized_vol_1h",
              "beta_short_vs_bk", "corr_1d_vs_bk", "corr_change_3d_vs_bk",
              "idio_ret_12b_vs_bk", "idio_ret_48b_vs_bk",
              "idio_vol_1h_vs_bk", "idio_vol_1d_vs_bk"],
    "FLOW": ["obv_z_1d", "obv_signal", "vwap_zscore", "vwap_slope_96",
             "mfi", "price_volume_corr_10", "price_volume_corr_20"],
    "XS_RANK": ["return_1d_xs_rank", "atr_pct_xs_rank",
                "ema_slope_20_1h_xs_rank", "idio_vol_1d_vs_bk_xs_rank",
                "obv_z_1d_xs_rank", "vwap_zscore_xs_rank",
                "bars_since_high_xs_rank"],
}


def report_ic(panel: pd.DataFrame, h: int) -> None:
    label = f"fwd_resid_{h}"
    n_ts = panel[label].notna().groupby(panel["ts"]).any().sum()
    n_sym = panel["symbol"].nunique()
    log.info("\n--- IC at h=%d (%dh forward), n_symbols=%d, n_ts_with_label=%d ---",
             h, h * 5 // 60, n_sym, n_ts)
    log.info("  %-30s %8s %8s %8s %6s",
             "feature", "IC", "lo95", "hi95", "n_ts")
    for group, feats in FEATURE_GROUPS.items():
        for f in feats:
            if f not in panel.columns:
                continue
            ic = per_ts_ic(panel, f, label)
            if len(ic) == 0:
                continue
            mu, lo, hi = block_bootstrap_ci(ic)
            sig = "***" if (not np.isnan(lo) and (lo > 0 or hi < 0)) else "   "
            log.info("  %-30s %+8.4f %+8.4f %+8.4f %6d %s",
                     f"[{group[0]}] {f}", mu, lo, hi, len(ic), sig)


def main() -> None:
    log.info("loading yfinance 5m panel: %d names, %dd, RTH only",
             len(UNIVERSE), DAYS_BACK)
    panel = load_panel()
    if panel.empty:
        log.error("no data")
        return
    panel = add_returns(panel)
    bk = build_basket(panel)
    log.info("computing features...")
    panel = add_base_features(panel)
    panel = add_cross_features(panel, bk)
    panel = add_flow_features(panel)
    panel = add_xs_rank_features(panel)

    # sanity
    sub = panel[panel["beta_short_vs_bk"].notna()]
    if not sub.empty:
        var_ratio = (sub.groupby("symbol")["idio_1bar"].var()
                     / sub.groupby("symbol")["ret"].var()).median()
        log.info("residualization sanity: median beta=%.2f IQR=[%.2f,%.2f]  "
                 "var(idio)/var(ret)=%.2f",
                 sub["beta_short_vs_bk"].median(),
                 sub["beta_short_vs_bk"].quantile(0.25),
                 sub["beta_short_vs_bk"].quantile(0.75),
                 var_ratio)

    panel = panel.set_index("ts").reset_index()
    for h in HORIZONS:
        panel = add_label(panel, h)
        report_ic(panel, h)


if __name__ == "__main__":
    main()
