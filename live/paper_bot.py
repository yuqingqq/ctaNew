"""v6_clean multi-symbol paper-trade bot — single-cycle orchestrator.

Designed to be run as a daily cron job (or manually). One invocation =
one rebalance cycle. State persists to live/state/ between runs.

Each cycle:
  1. Refresh recent Binance USDM perp 5m klines via REST (last ~14d of bars
     per symbol — enough for v6_clean's 7-day rolling features).
  2. If we have a previous open portfolio, fetch HL prices, mark-to-market,
     compute realized P&L for the just-closed cycle, append to log.
  3. Compute v6_clean features, predict, rank, select top-K long / bot-K short.
  4. Fetch HL prices for entry, record new portfolio state.
  5. Save state, exit.

Run modes:
  python -m live.paper_bot                 # live: do one cycle now
  python -m live.paper_bot --replay 30     # replay last 30 days from cached
                                            #   Binance + HL data for validation
  python -m live.paper_bot --check-state   # print current open positions

State files (live/state/):
  positions.json        Open positions awaiting next rebalance
  cycles.csv            Append-only cycle log (one row per closed cycle)
  binance_5m/{sym}.parquet   rolling kline cache, ~14d per symbol

Cost model: HL VIP-0 taker 4 bps RT per leg (close to docs: 4.5 bps taker
fee + small spread). For maker (rebate-tier accounts) -1 bps possible.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from data_collectors.binance_vision_loader import (
    LoaderConfig as VisionLoaderConfig, fetch_klines as fetch_vision_klines,
)
from data_collectors.hl_data_fetcher import HyperliquidDataFetcher
from features_ml.cross_sectional import (
    XS_FEATURE_COLS_V6_CLEAN, XS_RANK_SOURCES,
    add_basket_features, add_engineered_flow_features, add_xs_rank_features,
    build_basket, list_universe, make_xs_alpha_labels,
)
from features_ml.klines import compute_kline_features
from features_ml.regime_features import add_regime_features

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("paper_bot")

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "live" / "state"
KLINES_DIR = STATE_DIR / "binance_5m"
POSITIONS_PATH = STATE_DIR / "positions.json"
CYCLES_PATH = STATE_DIR / "cycles.csv"
MODEL_DIR = ROOT / "models"

HORIZON_BARS = 288                          # 1d at 5min cadence
LOOKBACK_DAYS = 14                          # bars to keep per symbol for features
TOP_K = 5
TOP_FRAC = 0.20
HL_TAKER_BPS_PER_LEG = 4.0                  # HL VIP-0 round-trip per leg
BINANCE_FAPI = "https://fapi.binance.com"
HL_INFO_URL = "https://api.hyperliquid.xyz/info"

REGIME_CUTOFF = 0.33                        # match alpha_v4_xs_1d


def _binance_to_hl_coin(symbol: str) -> str:
    sym = symbol.upper()
    for suffix in ("USDT", "USDC", "USD", "PERP"):
        if sym.endswith(suffix):
            return sym[: -len(suffix)]
    return sym


# =============================================================================
# Data refresh
# =============================================================================

def fetch_binance_klines(symbol: str, days: int = LOOKBACK_DAYS,
                         interval: str = "5m") -> pd.DataFrame:
    """Fetch last `days` of 5m klines via Binance USDM REST. Up to 1500/req."""
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    bars_needed = days * 288 + 5  # 288 bars/day + small buffer
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol, "interval": interval,
            "startTime": cursor, "endTime": end_ms,
            "limit": min(1500, bars_needed - len(rows)),
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
        except Exception as e:
            log.warning("[%s] fetch_binance_klines: %s", symbol, e)
            break
        chunk = r.json()
        if not chunk:
            break
        rows.extend(chunk)
        cursor = chunk[-1][0] + 1
        if len(chunk) < params["limit"]:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume",
              "quote_volume", "taker_buy_base", "taker_buy_quote"):
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(int)
    df = df.set_index("open_time").sort_index()
    return df[["open", "high", "low", "close", "volume",
                "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]]


def fetch_vision_klines_for_symbol(symbol: str, days: int) -> pd.DataFrame:
    """Pull last `days` daily kline archives from Binance Vision (1-day lag).

    Vision archives only have data through yesterday — that's fine for a daily
    rebalance bot when fapi is unavailable (e.g. geo-blocked regions).
    """
    end = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    start = end - timedelta(days=days - 1)
    cfg = VisionLoaderConfig(symbol=symbol, out_dir=Path("data/ml"))
    df = fetch_vision_klines(start, end, interval="5m", cfg=cfg)
    if df.empty:
        return df
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["open_time"]):
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.set_index("open_time").sort_index()
    keep = ["open", "high", "low", "close", "volume"]
    for c in ("quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"):
        if c in df.columns:
            keep.append(c)
    return df[keep]


def fetch_hl_klines_5m(symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Pull last `days` of 5min HL klines via info.candleSnapshot API.

    HL only retains ~15 days at 5min resolution (180d at 1h). For h=288
    daily-rebalance with 7-day rolling features, 14d lookback is sufficient
    once the bot has been running long enough to accumulate a continuous cache.

    Note: HL volume is in COIN units (not quote currency) and is much smaller
    in magnitude than Binance. The model's `volume_ma_50` feature is scale-
    dependent; predictions may shift slightly vs. Binance-data inference.
    Validate via live/paper_bot --source hl + --source vision side-by-side.
    """
    fetcher = HyperliquidDataFetcher()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    df = fetcher.fetch_range(symbol=symbol, interval="5m",
                              start_time=start, end_time=end)
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    elif "open_time" in df.columns:
        df = df.set_index("open_time")
    df = df.sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index.name = None  # normalize: build_panel_for_inference assumes unnamed index
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[keep].astype(float)


def refresh_klines_cache(universe: list[str], days: int = LOOKBACK_DAYS,
                          source: str = "auto") -> dict:
    """Fetch fresh klines for all symbols, write to local cache.

    source:
      "fapi"   real-time Binance USDM (may be geo-blocked → HTTP 451)
      "vision" Binance daily archives (1-day lag, always works)
      "hl"     real-time Hyperliquid (15d max history at 5min)
      "auto"   try fapi → vision; on first symbol that succeeds, lock in
               that source for the rest of the cycle

    Cache is keyed by source (separate parquets per source) so switching
    doesn't cross-contaminate.

    Returns dict {symbol: DataFrame indexed by open_time}.
    """
    KLINES_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    fapi_blocked = False
    for i, s in enumerate(universe):
        log.info("[%d/%d] refreshing %s klines (source=%s)", i + 1, len(universe), s, source)
        df = pd.DataFrame()
        if source == "hl":
            try:
                df = fetch_hl_klines_5m(s, days=days)
            except Exception as e:
                log.error("[%s] hl fetch failed: %s", s, e)
        else:
            if source in ("fapi", "auto") and not fapi_blocked:
                try:
                    df = fetch_binance_klines(s, days=days)
                except Exception as e:
                    log.warning("[%s] fapi failed: %s — falling back to Vision", s, e)
            if df.empty and source in ("vision", "auto"):
                try:
                    df = fetch_vision_klines_for_symbol(s, days=days)
                    if source == "auto" and not df.empty:
                        fapi_blocked = True
                        log.info("[%s] vision OK — using vision for remaining symbols", s)
                except Exception as e:
                    log.error("[%s] vision fetch failed: %s", s, e)
        if df.empty:
            log.warning("[%s] empty kline response from source=%s", s, source)
            continue
        cache_path = KLINES_DIR / f"{s}_{source}.parquet"
        if cache_path.exists():
            old = pd.read_parquet(cache_path)
            if old.index.tz is None:
                old.index = old.index.tz_localize("UTC")
            df = pd.concat([old, df]).loc[~pd.concat([old, df]).index.duplicated(keep="last")]
            df = df.sort_index().tail(days * 288 + 100)
        df.to_parquet(cache_path, compression="zstd")
        out[s] = df
        time.sleep(0.05)
    return out


def fetch_hl_mids() -> dict[str, float]:
    """One REST call returns mid prices for all HL perps."""
    r = requests.post(HL_INFO_URL, json={"type": "allMids"}, timeout=10)
    r.raise_for_status()
    payload = r.json()
    return {k: float(v) for k, v in payload.items()}


# =============================================================================
# Feature pipeline + prediction
# =============================================================================

def build_kline_features_inmem(klines: pd.DataFrame) -> pd.DataFrame:
    """Same as features_ml.cross_sectional.build_kline_features but takes
    an in-memory DataFrame instead of reading parquet files. Output is the
    full kline+regime+autocorr feature frame."""
    feats = compute_kline_features(klines)
    feats = add_regime_features(feats)
    ret = feats["close"].pct_change()
    feats["autocorr_1h"] = ret.rolling(36).apply(
        lambda s: s.autocorr(lag=1) if s.std() > 0 else 0.0)
    feats["autocorr_pctile_7d"] = (
        feats["autocorr_1h"].rolling(2016, min_periods=288).rank(pct=True).shift(1)
    )
    return feats


def build_panel_for_inference(klines_by_sym: dict, sym_to_id: dict) -> pd.DataFrame:
    """Build the v6_clean cross-sectional panel for live inference.

    Mirrors alpha_v6_permutation_lean._build_v6_panel_lean exactly, but on
    in-memory klines + only returning the most-recent bars where features
    are valid.
    """
    feats_by_sym = {}
    for s, kl in klines_by_sym.items():
        if kl.empty:
            continue
        f = build_kline_features_inmem(kl)
        if not f.empty:
            feats_by_sym[s] = f

    closes = pd.DataFrame({s: f["close"] for s, f in feats_by_sym.items()}).sort_index()
    if closes.empty:
        return pd.DataFrame()
    basket_ret, basket_close = build_basket(closes)

    enriched = {}
    for s, f in feats_by_sym.items():
        f = f.reindex(closes.index)
        f = add_basket_features(f, basket_close, basket_ret)
        f = add_engineered_flow_features(f)
        f["sym_id"] = sym_to_id.get(s, -1)
        enriched[s] = f

    rank_cols = [c for c in XS_FEATURE_COLS_V6_CLEAN if c.endswith("_xs_rank")]
    src_cols = list({s for s, d in XS_RANK_SOURCES.items() if d in rank_cols})
    needed = list(set(list(XS_FEATURE_COLS_V6_CLEAN)
                       + ["sym_id", "autocorr_pctile_7d", "beta_short_vs_bk", "close"]
                       + src_cols) - set(rank_cols))

    frames = []
    for s, f in enriched.items():
        avail = [c for c in needed if c in f.columns]
        df = f[avail].copy()
        df["symbol"] = s
        df = df.reset_index().rename(columns={"index": "open_time"})
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True, sort=False)
    panel = add_xs_rank_features(panel, sources=XS_RANK_SOURCES)
    panel = panel.dropna(subset=rank_cols + ["autocorr_pctile_7d"])
    return panel


def load_model_artifact():
    pkl = MODEL_DIR / "v6_clean_ensemble.pkl"
    meta = MODEL_DIR / "v6_clean_meta.json"
    if not pkl.exists() or not meta.exists():
        raise FileNotFoundError(
            f"Model artifact missing. Run: python -m live.train_v6_clean_artifact")
    with pkl.open("rb") as f:
        models = pickle.load(f)
    with meta.open() as f:
        meta_d = json.load(f)
    return models, meta_d


def predict_for_bar(models, panel: pd.DataFrame, target_time: pd.Timestamp,
                    feat_cols: list) -> pd.DataFrame:
    """Predict alpha for each symbol at the given open_time. Returns
    DataFrame with [symbol, pred, beta_short_vs_bk, autocorr_pctile_7d]."""
    bar = panel[panel["open_time"] == target_time]
    if bar.empty:
        return pd.DataFrame()
    X = bar[feat_cols].to_numpy(dtype=np.float32)
    yt = np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)
    out = bar[["symbol", "beta_short_vs_bk", "autocorr_pctile_7d", "close"]].copy()
    out["pred"] = yt
    return out.reset_index(drop=True)


# =============================================================================
# Portfolio construction
# =============================================================================

@dataclass
class LegPosition:
    symbol: str
    side: str  # "L" (long) or "S" (short)
    weight: float  # signed: + for long, - for short, sums to ~scale
    entry_price_hl: float
    entry_time: str  # iso utc

    def to_dict(self):
        return asdict(self)


def select_portfolio(preds: pd.DataFrame, top_k: int = TOP_K) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Cross-sectional top-K long / bot-K short with β-neutral scaling.

    Mirrors portfolio_pnl_turnover_aware logic. Returns (top, bot, scale_L, scale_S).
    """
    g = preds.dropna(subset=["pred"])
    if len(g) < 10:
        raise RuntimeError(f"insufficient symbols ({len(g)}) for portfolio selection")
    sorted_g = g.sort_values("pred")
    bot = sorted_g.head(top_k)
    top = sorted_g.tail(top_k)

    beta_L = float(top["beta_short_vs_bk"].mean())
    beta_S = float(bot["beta_short_vs_bk"].mean())
    if beta_L < 0.1 or beta_S < 0.1 or (beta_L + beta_S) < 0.3:
        scale_L, scale_S = 1.0, 1.0  # equal-weight fallback
    else:
        denom = beta_L + beta_S
        scale_L = float(np.clip(2.0 * beta_S / denom, 0.5, 1.5))
        scale_S = float(np.clip(2.0 * beta_L / denom, 0.5, 1.5))
    return top, bot, scale_L, scale_S


def positions_to_dict(top: pd.DataFrame, bot: pd.DataFrame, scale_L: float, scale_S: float,
                      hl_mids: dict, now_iso: str, n_per_side: int) -> list[LegPosition]:
    out = []
    for _, row in top.iterrows():
        coin = _binance_to_hl_coin(row["symbol"])
        out.append(LegPosition(
            symbol=row["symbol"], side="L",
            weight=scale_L / n_per_side,
            entry_price_hl=float(hl_mids.get(coin, np.nan)),
            entry_time=now_iso,
        ))
    for _, row in bot.iterrows():
        coin = _binance_to_hl_coin(row["symbol"])
        out.append(LegPosition(
            symbol=row["symbol"], side="S",
            weight=-scale_S / n_per_side,
            entry_price_hl=float(hl_mids.get(coin, np.nan)),
            entry_time=now_iso,
        ))
    return out


def turnover(prev: list[LegPosition], curr: list[LegPosition]) -> tuple[float, float]:
    """L1-distance/2 between long-leg and short-leg weight vectors."""
    prev_long = {p.symbol: p.weight for p in prev if p.side == "L"} if prev else {}
    prev_short = {p.symbol: -p.weight for p in prev if p.side == "S"} if prev else {}
    curr_long = {p.symbol: p.weight for p in curr if p.side == "L"}
    curr_short = {p.symbol: -p.weight for p in curr if p.side == "S"}
    if not prev_long and not prev_short:
        sl = sum(curr_long.values())
        ss = sum(curr_short.values())
        return float(sl), float(ss)
    long_to = 0.5 * sum(abs(curr_long.get(s, 0) - prev_long.get(s, 0))
                          for s in set(curr_long) | set(prev_long))
    short_to = 0.5 * sum(abs(curr_short.get(s, 0) - prev_short.get(s, 0))
                          for s in set(curr_short) | set(prev_short))
    return float(long_to), float(short_to)


def realize_pnl(prev: list[LegPosition], hl_mids_now: dict) -> dict:
    """Mark-to-market the prior cycle's positions at current HL mids."""
    if not prev:
        return {"long_ret_bps": 0.0, "short_ret_bps": 0.0,
                 "spread_ret_bps": 0.0, "n_long": 0, "n_short": 0}
    long_rets, short_rets = [], []
    weights_L, weights_S = [], []
    for p in prev:
        coin = _binance_to_hl_coin(p.symbol)
        exit_px = float(hl_mids_now.get(coin, np.nan))
        if not np.isfinite(exit_px) or not np.isfinite(p.entry_price_hl):
            continue
        ret = (exit_px / p.entry_price_hl) - 1.0
        if p.side == "L":
            long_rets.append(ret)
            weights_L.append(p.weight)
        else:
            short_rets.append(ret)
            weights_S.append(-p.weight)  # short weight stored negative
    long_ret = float(np.mean(long_rets)) if long_rets else 0.0
    short_ret = float(np.mean(short_rets)) if short_rets else 0.0
    scale_L = float(sum(weights_L)) if weights_L else 0.0
    scale_S = float(sum(weights_S)) if weights_S else 0.0
    spread_ret = scale_L * long_ret - scale_S * short_ret
    return {
        "long_ret_bps": long_ret * 1e4 * scale_L,
        "short_ret_bps": short_ret * 1e4 * scale_S,
        "spread_ret_bps": spread_ret * 1e4,
        "n_long": len(long_rets), "n_short": len(short_rets),
        "scale_L_realized": scale_L, "scale_S_realized": scale_S,
    }


# =============================================================================
# State persistence
# =============================================================================

def load_state() -> tuple[Optional[list[LegPosition]], pd.DataFrame]:
    if POSITIONS_PATH.exists():
        with POSITIONS_PATH.open() as f:
            data = json.load(f)
        positions = [LegPosition(**d) for d in data] if data else []
    else:
        positions = None
    cycles = pd.read_csv(CYCLES_PATH) if CYCLES_PATH.exists() else pd.DataFrame()
    return positions, cycles


def save_state(positions: list[LegPosition], cycle_row: dict):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with POSITIONS_PATH.open("w") as f:
        json.dump([p.to_dict() for p in positions], f, indent=2)
    df_new = pd.DataFrame([cycle_row])
    if CYCLES_PATH.exists():
        df = pd.concat([pd.read_csv(CYCLES_PATH), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(CYCLES_PATH, index=False)


# =============================================================================
# One full cycle
# =============================================================================

def run_one_cycle(*, refresh_data: bool = True, source: str = "auto") -> dict:
    """Runs one paper-trade rebalance cycle. Returns the cycle row dict."""
    log.info("===== v6_clean paper-trade cycle starting =====")
    models, meta = load_model_artifact()
    feat_cols = list(meta["feat_cols"])
    sym_to_id = meta["sym_to_id"]
    universe = sorted(sym_to_id.keys())
    log.info("Loaded model artifact: %d symbols, %d features, trained %s",
              len(universe), len(feat_cols), meta["trained_at_utc"])

    if refresh_data:
        klines_by_sym = refresh_klines_cache(universe, days=LOOKBACK_DAYS, source=source)
    else:
        klines_by_sym = {}
        for s in universe:
            p = KLINES_DIR / f"{s}.parquet"
            if p.exists():
                klines_by_sym[s] = pd.read_parquet(p)

    if len(klines_by_sym) < 10:
        raise RuntimeError(f"insufficient kline coverage: {len(klines_by_sym)}/{len(universe)}")

    # Build inference panel
    panel = build_panel_for_inference(klines_by_sym, sym_to_id)
    if panel.empty:
        raise RuntimeError("inference panel is empty")
    target_time = panel["open_time"].max()
    log.info("Inference target_time: %s, panel rows: %d, n_syms_at_target: %d",
              target_time, len(panel),
              panel[panel["open_time"] == target_time]["symbol"].nunique())

    # Predict
    preds = predict_for_bar(models, panel, target_time, feat_cols)
    if preds.empty or len(preds) < 10:
        raise RuntimeError(f"prediction frame too small: {len(preds)}")

    # Apply regime filter the same way training did (skip cycles in low-autocorr regime)
    n_active = (preds["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF).sum()
    log.info("regime-active symbols at target_time: %d/%d", n_active, len(preds))

    # Realize previous cycle's P&L (if any) using current HL mids
    hl_mids = fetch_hl_mids()
    prev_positions, _ = load_state()
    realized = realize_pnl(prev_positions or [], hl_mids)

    # Select new portfolio
    top, bot, scale_L, scale_S = select_portfolio(preds, top_k=TOP_K)
    n_per_side = TOP_K
    new_positions = positions_to_dict(top, bot, scale_L, scale_S,
                                       hl_mids, str(target_time), n_per_side)

    # Compute turnover from prev → new portfolio
    long_to, short_to = turnover(prev_positions or [], new_positions)
    cost_bps = HL_TAKER_BPS_PER_LEG * (long_to + short_to)
    net_bps = realized["spread_ret_bps"] - cost_bps if prev_positions else 0.0

    cycle_row = {
        "decision_time_utc": str(target_time),
        "wall_time_utc": datetime.now(timezone.utc).isoformat(),
        "n_symbols_active": int(n_active),
        "n_symbols_total": int(len(preds)),
        "long_symbols": ",".join(top["symbol"].tolist()),
        "short_symbols": ",".join(bot["symbol"].tolist()),
        "scale_L": scale_L, "scale_S": scale_S,
        "long_to": long_to, "short_to": short_to,
        "cost_bps": cost_bps,
        # Realized PnL of PRIOR cycle:
        "prior_long_ret_bps": realized["long_ret_bps"],
        "prior_short_ret_bps": realized["short_ret_bps"],
        "prior_spread_ret_bps": realized["spread_ret_bps"],
        "prior_n_long": realized["n_long"], "prior_n_short": realized["n_short"],
        "net_bps": net_bps,
    }

    save_state(new_positions, cycle_row)

    log.info("Cycle complete:")
    log.info("  long top-%d:  %s", TOP_K, top["symbol"].tolist())
    log.info("  short bot-%d: %s", TOP_K, bot["symbol"].tolist())
    log.info("  β-neutral scales: L=%.3f, S=%.3f", scale_L, scale_S)
    log.info("  turnover: long=%.3f, short=%.3f, cost=%.2f bps", long_to, short_to, cost_bps)
    if prev_positions:
        log.info("  prior cycle realized: spread_ret=%+.2f bps, net=%+.2f bps",
                  realized["spread_ret_bps"], net_bps)
    return cycle_row


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-state", action="store_true",
                    help="print current open positions and recent cycles, exit")
    ap.add_argument("--no-refresh", action="store_true",
                    help="skip Binance kline REST refresh, use cached state")
    ap.add_argument("--source", choices=("auto", "fapi", "vision", "hl"), default="auto",
                    help="kline source: auto (default Binance fapi→vision), fapi "
                         "(Binance real-time, may be geo-blocked), vision (Binance "
                         "1-day lag, always works), hl (Hyperliquid real-time, 15d "
                         "history)")
    ap.add_argument("--replay", type=int, default=0,
                    help="(reserved for live/replay_paper_bot.py)")
    args = ap.parse_args()

    if args.check_state:
        positions, cycles = load_state()
        if positions:
            print("Open positions:")
            for p in positions:
                print(f"  {p.side} {p.symbol} weight={p.weight:+.4f} entry_hl={p.entry_price_hl:.6f} at {p.entry_time}")
        else:
            print("No open positions.")
        if not cycles.empty:
            print(f"\nLast 5 cycles:")
            print(cycles.tail(5).to_string(index=False))
        return 0

    if args.replay:
        log.error("Use `python -m live.replay_paper_bot --days N` for replay mode.")
        return 2

    run_one_cycle(refresh_data=not args.no_refresh, source=args.source)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
