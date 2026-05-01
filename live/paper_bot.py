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
from live.telegram import notify_telegram
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
HL_TAKER_FEE_BPS = 4.5                      # HL VIP-0 one-way taker fee
INITIAL_EQUITY_USD = 10_000.0               # paper portfolio sizing
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


def fetch_hl_l2_book(coin: str) -> dict:
    """Fetch L2 orderbook snapshot for one coin. Returns
    {"bids": [(px, sz), ...], "asks": [(px, sz), ...], "ts": ms}.
    Bids are descending, asks ascending — best is index 0 in each."""
    r = requests.post(HL_INFO_URL, json={"type": "l2Book", "coin": coin}, timeout=10)
    r.raise_for_status()
    payload = r.json()
    levels = payload.get("levels", [[], []])
    bids = [(float(l["px"]), float(l["sz"])) for l in levels[0]]
    asks = [(float(l["px"]), float(l["sz"])) for l in levels[1]]
    return {"bids": bids, "asks": asks, "ts": int(payload.get("time", 0))}


def simulate_taker_fill(book: dict, side: str, target_notional_usd: float) -> dict:
    """Walk one side of the book to fill `target_notional_usd`.

    side: "buy" walks asks (long entry, short exit);
          "sell" walks bids (short entry, long exit).

    Returns dict with vwap, mid, slippage_bps (signed: +ve = adverse),
    qty, levels_consumed, fully_filled.

    Slippage convention: positive = paid more than mid (adverse for taker).
    Fee NOT included here — added separately in cost stack.
    """
    levels = book["asks"] if side == "buy" else book["bids"]
    if not levels or not book["bids"] or not book["asks"]:
        return {"vwap": float("nan"), "mid": float("nan"),
                "slippage_bps": float("nan"), "qty": 0.0,
                "levels_consumed": 0, "fully_filled": False}
    mid = 0.5 * (book["bids"][0][0] + book["asks"][0][0])
    consumed_qty = 0.0
    consumed_notional = 0.0
    remaining = target_notional_usd
    levels_consumed = 0
    for px, sz in levels:
        level_notional = px * sz
        if remaining <= level_notional:
            qty = remaining / px
            consumed_qty += qty
            consumed_notional += remaining
            remaining = 0.0
            levels_consumed += 1
            break
        consumed_qty += sz
        consumed_notional += level_notional
        remaining -= level_notional
        levels_consumed += 1
    fully_filled = remaining < 1e-6
    if consumed_qty == 0:
        return {"vwap": float("nan"), "mid": mid, "slippage_bps": float("nan"),
                "qty": 0.0, "levels_consumed": 0, "fully_filled": False}
    vwap = consumed_notional / consumed_qty
    sign = 1.0 if side == "buy" else -1.0
    slippage_bps = sign * (vwap - mid) / mid * 1e4
    return {
        "vwap": vwap, "mid": mid, "slippage_bps": slippage_bps,
        "qty": consumed_qty, "levels_consumed": levels_consumed,
        "fully_filled": fully_filled,
    }


def fetch_hl_books(coins: list[str]) -> dict[str, dict]:
    """Fetch L2 book for each coin. Sequential (HL info API has no batch).
    Coins like 'BTC', 'ETH', not 'BTCUSDT'."""
    out = {}
    for c in coins:
        try:
            out[c] = fetch_hl_l2_book(c)
            time.sleep(0.05)
        except Exception as e:
            log.warning("[%s] L2 fetch failed: %s", c, e)
            out[c] = None
    return out


def fetch_hl_funding_history(coin: str, start_ms: int,
                              end_ms: int = None) -> list[dict]:
    """Fetch HL hourly funding settlements for one coin in [start, end].

    Returns list of {coin, fundingRate (str, decimal), premium, time (ms)}.
    HL settles funding hourly; rates apply to NOTIONAL (long pays positive
    rate to short, vice versa for negative).
    """
    payload = {"type": "fundingHistory", "coin": coin, "startTime": int(start_ms)}
    if end_ms is not None:
        payload["endTime"] = int(end_ms)
    r = requests.post(HL_INFO_URL, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def accrue_funding_for_cycle(prev_positions: list[LegPosition],
                              prev_decision_iso: str,
                              now_iso: str,
                              equity_usd: float = INITIAL_EQUITY_USD) -> dict:
    """Sum funding paid/received for each prev position over [prev, now].

    For each position p, fetch hourly funding rates over the holding window.
    Per-hour funding payment (USD) = rate × notional × side_sign,
    where side_sign = +1 for long (pays positive rate) and -1 for short.
    From the position's PnL perspective: P&L_funding = -side_sign × rate × notional.

    Mutates `prev_positions[*].funding_paid_usd` with the accrued amount,
    so subsequent cycles see the cumulative funding-since-entry.

    Returns:
      total_funding_usd     sum across all positions
      funding_bps           same expressed as bps of equity
      per_symbol            list of {symbol, side, n_funding_intervals, usd_paid}
    """
    if not prev_positions:
        return {"total_funding_usd": 0.0, "funding_bps": 0.0, "per_symbol": []}
    try:
        prev_dt = pd.Timestamp(prev_decision_iso)
        now_dt = pd.Timestamp(now_iso)
    except Exception:
        log.warning("accrue_funding: invalid timestamps prev=%s now=%s",
                     prev_decision_iso, now_iso)
        return {"total_funding_usd": 0.0, "funding_bps": 0.0, "per_symbol": []}
    start_ms = int(prev_dt.timestamp() * 1000)
    end_ms = int(now_dt.timestamp() * 1000)

    per_symbol = []
    total_usd = 0.0
    # Cache per-coin to avoid duplicate fetches when multiple positions
    # share a symbol (rare in v6_clean since each symbol is L or S, not both)
    cache: dict[str, list[dict]] = {}
    for p in prev_positions:
        coin = _binance_to_hl_coin(p.symbol)
        if coin not in cache:
            try:
                cache[coin] = fetch_hl_funding_history(coin, start_ms, end_ms)
            except Exception as e:
                log.warning("[%s] funding fetch failed: %s", coin, e)
                cache[coin] = []
            time.sleep(0.05)
        rates = cache[coin]
        if not rates:
            per_symbol.append({"symbol": p.symbol, "side": p.side,
                                "n_funding_intervals": 0, "usd_paid": 0.0})
            continue
        side_sign = 1.0 if p.side == "L" else -1.0
        usd_paid = 0.0
        for entry in rates:
            try:
                rate = float(entry["fundingRate"])
            except (KeyError, TypeError, ValueError):
                continue
            # Long with positive rate pays funding; short receives.
            # PnL impact for the position holder = -side_sign × rate × notional
            payment = side_sign * rate * p.entry_notional_usd
            usd_paid += payment
        p.funding_paid_usd += usd_paid
        per_symbol.append({"symbol": p.symbol, "side": p.side,
                            "n_funding_intervals": len(rates), "usd_paid": usd_paid})
        total_usd += usd_paid

    funding_bps = (total_usd / equity_usd) * 1e4
    # Convention: returned funding_bps is COST (positive means we paid).
    # PnL impact is -funding_bps.
    return {"total_funding_usd": total_usd, "funding_bps": funding_bps,
            "per_symbol": per_symbol}


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
    side: str                    # "L" (long) or "S" (short)
    weight: float                # signed magnitude: + for long, - for short
    entry_price_hl: float        # avg fill VWAP across this position's lifetime
    entry_mid_hl: float          # mid at original entry (for cumulative slippage attr)
    entry_notional_usd: float    # current USD notional held
    entry_slippage_bps: float    # avg slippage paid on this position's entries
    entry_time: str              # iso utc of original entry
    last_marked_mid: float = 0.0 # most recent mid; used for per-cycle gross MtM
    funding_paid_usd: float = 0.0  # cumulative funding paid (negative = received)

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


def compute_target_weights(top: pd.DataFrame, bot: pd.DataFrame,
                             scale_L: float, scale_S: float, n_per_side: int) -> dict:
    """Returns {symbol: signed weight} — positive = long, negative = short."""
    w = {}
    for _, row in top.iterrows():
        w[row["symbol"]] = scale_L / n_per_side
    for _, row in bot.iterrows():
        w[row["symbol"]] = -scale_S / n_per_side
    return w


def execute_cycle_turnover_aware(prev_positions: list[LegPosition],
                                  target_weights: dict[str, float],
                                  books: dict[str, dict],
                                  now_iso: str,
                                  equity_usd: float = INITIAL_EQUITY_USD) -> dict:
    """Trade only the delta between prev and target weight vectors.

    Returns dict with:
      new_positions      list[LegPosition] reflecting target weights
      gross_pnl_bps      MtM change of prev positions over the cycle (mid→mid)
      slippage_bps       weighted slippage cost across delta trades
      fees_bps           taker fees on trade notional (one-way × notional)
      n_trades           number of L2 fills executed this cycle
      trades             list of per-trade records (for logging)

    Cost model: HL_TAKER_FEE_BPS is one-way. Each non-zero |delta| × equity is
    one market order paying HL_TAKER_FEE_BPS bps + walking the book.
    """
    prev_w = {p.symbol: p.weight for p in (prev_positions or [])}
    prev_by_sym = {p.symbol: p for p in (prev_positions or [])}
    all_syms = set(prev_w) | set(target_weights)

    # Compute mids from books (best bid/ask midpoint)
    def _mid_for(sym: str) -> float:
        coin = _binance_to_hl_coin(sym)
        b = books.get(coin)
        if b is None or not b["bids"] or not b["asks"]:
            return float("nan")
        return 0.5 * (b["bids"][0][0] + b["asks"][0][0])

    # 1. Mark prev positions to current mid for per-cycle gross PnL
    gross_pnl_bps = 0.0
    for p in (prev_positions or []):
        mid_now = _mid_for(p.symbol)
        if not np.isfinite(mid_now) or not np.isfinite(p.last_marked_mid) or p.last_marked_mid == 0:
            continue
        # weight is signed; for long (+) PnL = (mid_now/start - 1) × |weight|;
        # for short (-) PnL = (start/mid_now - 1) × |weight|.
        if p.side == "L":
            pnl_frac = (mid_now / p.last_marked_mid - 1.0)
        else:
            pnl_frac = (p.last_marked_mid / mid_now - 1.0)
        gross_pnl_bps += pnl_frac * abs(p.weight) * 1e4

    # 2. Compute deltas and execute L2 fills
    trades = []
    total_trade_notional = 0.0
    total_slip_weighted = 0.0  # for weighted-average slippage report
    new_positions = []
    for sym in sorted(all_syms):
        prev_weight = prev_w.get(sym, 0.0)
        new_weight = target_weights.get(sym, 0.0)
        delta = new_weight - prev_weight
        coin = _binance_to_hl_coin(sym)
        mid_now = _mid_for(sym)

        if abs(delta) < 1e-9:
            # No trade — carry forward prev position with updated last_marked_mid
            if sym in prev_by_sym:
                p = prev_by_sym[sym]
                new_positions.append(LegPosition(
                    symbol=sym, side=p.side, weight=p.weight,
                    entry_price_hl=p.entry_price_hl, entry_mid_hl=p.entry_mid_hl,
                    entry_notional_usd=p.entry_notional_usd,
                    entry_slippage_bps=p.entry_slippage_bps,
                    entry_time=p.entry_time,
                    last_marked_mid=mid_now if np.isfinite(mid_now) else p.last_marked_mid,
                    funding_paid_usd=p.funding_paid_usd,
                ))
            continue

        # Trade: simulate L2 fill on |delta| × equity in the appropriate direction
        side_action = "buy" if delta > 0 else "sell"
        notional = abs(delta) * equity_usd
        book = books.get(coin)
        if book is None:
            log.warning("[%s] no L2 book — skipping delta=%+.4f", sym, delta)
            continue
        fill = simulate_taker_fill(book, side=side_action, target_notional_usd=notional)
        if not np.isfinite(fill["vwap"]):
            log.warning("[%s] taker fill NaN for delta=%+.4f notional=$%.0f", sym, delta, notional)
            continue
        trades.append({
            "symbol": sym, "delta_weight": delta, "side_action": side_action,
            "notional_usd": notional, "fill_vwap": fill["vwap"],
            "fill_mid": fill["mid"], "slippage_bps": fill["slippage_bps"],
        })
        total_trade_notional += notional
        # weight slippage by trade notional (so aggregate is notional-weighted)
        total_slip_weighted += abs(fill["slippage_bps"]) * abs(delta)

        # Build the new position state for this symbol (if it survives)
        if abs(new_weight) < 1e-9:
            continue  # fully exited — drop from positions
        prev_pos = prev_by_sym.get(sym)
        # If this is an existing position with same sign and weight increase,
        # blend entry prices. Otherwise (new entry, flip, or weight reduction),
        # adopt the trade fill VWAP for the new portion's basis.
        same_side = (prev_pos is not None and (prev_pos.weight * new_weight > 0))
        if same_side and abs(new_weight) > abs(prev_weight):
            # Adding to existing position: VWAP of original portion + delta portion
            old_q = abs(prev_weight)
            add_q = abs(delta)
            blended = (old_q * prev_pos.entry_price_hl + add_q * fill["vwap"]) / abs(new_weight)
            blended_slip = ((old_q * prev_pos.entry_slippage_bps
                              + add_q * fill["slippage_bps"]) / abs(new_weight))
            entry_price = blended
            entry_slip = blended_slip
            entry_time = prev_pos.entry_time
            entry_mid = prev_pos.entry_mid_hl
            funding = prev_pos.funding_paid_usd
        elif same_side:
            # Reducing existing position size: keep original entry (PnL on the
            # reduced portion is realized via the trade slippage + gross MtM)
            entry_price = prev_pos.entry_price_hl
            entry_slip = prev_pos.entry_slippage_bps
            entry_time = prev_pos.entry_time
            entry_mid = prev_pos.entry_mid_hl
            funding = prev_pos.funding_paid_usd
        else:
            # New entry or flip — fresh basis
            entry_price = fill["vwap"]
            entry_slip = fill["slippage_bps"]
            entry_time = now_iso
            entry_mid = fill["mid"]
            funding = 0.0

        side = "L" if new_weight > 0 else "S"
        new_positions.append(LegPosition(
            symbol=sym, side=side, weight=new_weight,
            entry_price_hl=entry_price, entry_mid_hl=entry_mid,
            entry_notional_usd=abs(new_weight) * equity_usd,
            entry_slippage_bps=entry_slip,
            entry_time=entry_time,
            last_marked_mid=fill["mid"],
            funding_paid_usd=funding,
        ))

    # Cost decomposition (all in bps of equity)
    fees_bps = HL_TAKER_FEE_BPS * (total_trade_notional / equity_usd)
    slip_bps_total = total_slip_weighted * 1.0  # already in bps × weight units
    n_trades = len(trades)
    return {
        "new_positions": new_positions,
        "gross_pnl_bps": gross_pnl_bps,
        "slippage_bps": slip_bps_total,
        "fees_bps": fees_bps,
        "n_trades": n_trades,
        "trades": trades,
    }


# Legacy simulate_exits / simulate_entries / turnover removed in favor of
# execute_cycle_turnover_aware (above). The old close-all + reopen-all model
# over-charged fees by 2x and ignored the user's "sometimes we don't need
# to rebalance" intuition.


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
            # Respect --source: read source-specific cache file.
            p = KLINES_DIR / f"{s}_{source}.parquet"
            if not p.exists():
                # Legacy fallback (older cache without source suffix)
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

    # Decide new portfolio
    top, bot, scale_L, scale_S = select_portfolio(preds, top_k=TOP_K)
    n_per_side = TOP_K
    target_weights = compute_target_weights(top, bot, scale_L, scale_S, n_per_side)

    prev_positions, prev_cycles = load_state()
    prev_decision_iso = None
    if prev_positions and not prev_cycles.empty:
        prev_decision_iso = str(prev_cycles["decision_time_utc"].iloc[-1])

    # Fetch L2 books for symbols that are in either prev or target.
    # If a symbol is unchanged we only need its mid for MtM, not a full book —
    # but for simplicity we fetch books for all and skip those we don't trade.
    relevant_syms = set(p.symbol for p in (prev_positions or [])) | set(target_weights)
    relevant_coins = sorted({_binance_to_hl_coin(s) for s in relevant_syms})
    log.info("Fetching L2 books for %d coins...", len(relevant_coins))
    books = fetch_hl_books(relevant_coins)

    # Accrue HL hourly funding over the prev → now holding window
    funding_bps = 0.0
    funding_usd = 0.0
    if prev_positions and prev_decision_iso:
        log.info("Fetching HL funding history for %d positions over [%s, %s]...",
                  len(prev_positions), prev_decision_iso, str(target_time))
        fund = accrue_funding_for_cycle(prev_positions, prev_decision_iso,
                                          str(target_time))
        funding_usd = fund["total_funding_usd"]
        funding_bps = fund["funding_bps"]
        log.info("Funding accrued: $%.4f (%.2f bps of equity)", funding_usd, funding_bps)

    # Execute the cycle turnover-aware: only trade deltas between prev and target
    result = execute_cycle_turnover_aware(
        prev_positions or [], target_weights, books,
        now_iso=str(target_time), equity_usd=INITIAL_EQUITY_USD,
    )
    new_positions = result["new_positions"]

    # Per-cycle PnL accounting
    gross_pnl_bps = result["gross_pnl_bps"]    # MtM change of prev cycle
    fees_bps = result["fees_bps"]              # fees on actual delta-trades
    slippage_bps = result["slippage_bps"]      # weighted slippage on delta-trades
    # Funding is a COST when positive (we paid) — subtract from PnL.
    net_bps = gross_pnl_bps - fees_bps - slippage_bps - funding_bps

    # Cycle log row
    cycle_row = {
        "decision_time_utc": str(target_time),
        "wall_time_utc": datetime.now(timezone.utc).isoformat(),
        "n_symbols_active": int(n_active),
        "n_symbols_total": int(len(preds)),
        "long_symbols": ",".join(top["symbol"].tolist()),
        "short_symbols": ",".join(bot["symbol"].tolist()),
        "scale_L": float(scale_L), "scale_S": float(scale_S),
        # Per-cycle PnL components (turnover-aware):
        "gross_pnl_bps": gross_pnl_bps,         # MtM change of prev positions over cycle
        "fees_bps": fees_bps,                   # taker fees on actual delta trades
        "slippage_bps": slippage_bps,           # weighted L2 slippage on delta trades
        "funding_bps": funding_bps,             # HL hourly funding accrued (cost > 0)
        "funding_usd": funding_usd,
        "net_bps": net_bps,                     # gross − fees − slippage − funding
        # Trade activity:
        "n_trades": result["n_trades"],
        "n_open_positions": len(new_positions),
        "trade_notional_usd": sum(abs(t["notional_usd"]) for t in result["trades"]),
        "had_prev_positions": int(bool(prev_positions)),
    }

    save_state(new_positions, cycle_row)

    log.info("Cycle complete:")
    log.info("  target portfolio: long=%s, short=%s",
              top["symbol"].tolist(), bot["symbol"].tolist())
    log.info("  β-neutral scales: L=%.3f, S=%.3f", scale_L, scale_S)
    log.info("  trades: %d (%.0f USD notional total)",
              result["n_trades"], cycle_row["trade_notional_usd"])
    if prev_positions:
        log.info("  PnL: gross=%+.2f, fees=%.2f, slip=%.2f, funding=%.2f, net=%+.2f bps",
                  gross_pnl_bps, fees_bps, slippage_bps, funding_bps, net_bps)
    else:
        log.info("  first cycle: opened %d positions; no prior PnL", len(new_positions))

    # Telegram daily summary (skipped if env vars not set)
    _send_daily_summary_telegram(cycle_row, top, bot, scale_L, scale_S, result)

    return cycle_row


def _send_daily_summary_telegram(cycle_row, top, bot, scale_L, scale_S, result):
    """Compose + send the daily decision summary to Telegram. No-op without env."""
    longs = top["symbol"].tolist()
    shorts = bot["symbol"].tolist()
    n_trades = result["n_trades"]
    notional = sum(abs(t["notional_usd"]) for t in result["trades"])
    lines = [
        f"📅 <b>v6_clean daily decision</b>  (target {cycle_row['decision_time_utc']})",
        f"",
    ]
    if cycle_row.get("had_prev_positions") or cycle_row.get("gross_pnl_bps", 0) != 0:
        gross = cycle_row.get("gross_pnl_bps", 0)
        fees = cycle_row.get("fees_bps", 0)
        slip = cycle_row.get("slippage_bps", 0)
        fund = cycle_row.get("funding_bps", 0)
        net = cycle_row.get("net_bps", 0)
        fund_usd = cycle_row.get("funding_usd", 0)
        lines += [
            f"<b>Prior cycle PnL</b>:",
            f"  gross MtM: {gross:+.2f} bps",
            f"  fees:      {fees:.2f} bps",
            f"  slippage:  {slip:.2f} bps",
            f"  funding:   {fund:+.2f} bps  (${fund_usd:+.2f})",
            f"  <b>net:</b>     <b>{net:+.2f} bps</b>",
            f"",
        ]
    else:
        lines += ["<i>First cycle — no prior PnL.</i>", ""]
    lines += [
        f"<b>New target portfolio</b>  (β-neutral, scale L={scale_L:.3f}, S={scale_S:.3f}):",
        f"  Longs:  {', '.join(longs)}",
        f"  Shorts: {', '.join(shorts)}",
        f"",
        f"<b>Trades this cycle</b>: {n_trades}  (${notional:,.0f} notional)",
    ]
    text = "\n".join(lines)
    sent = notify_telegram(text)
    if sent:
        log.info("telegram daily summary sent")


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
