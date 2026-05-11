"""v7 xyz alpha-residual paper-trade bot — daily cycle, shadow mode.

Each invocation = one decision cycle. Designed to run as a daily cron after
US RTH close (~21:30 UTC). State persists in live/state/xyz/.

Cycle steps:
  1. Load v7 model artifact (15 models) and previous state.
  2. Refresh yfinance daily data for full S&P 100 (incremental, last ~10 days).
  3. Build full panel (returns, basket, residual, A+B+F_sector features) and
     validate feature order matches the artifact's frozen feat_cols.
  4. Predict for the latest bar using all 15 LGBMs (3 horizons × 5 seeds),
     averaged. NaN features are routed by LGBM's learned default branches.
  5. Filter to active universe preset, rank, apply hysteresis from prior state.
  6. Apply dispersion gate (PIT 60-pctile of trailing 252d).
  7. Fetch xyz L2 books and simulate taker fills:
       - For ROTATED-OUT names: simulate exit fill against current L2.
       - For NEWLY-ENTERED names: simulate entry fill, record vwap as next
         cycle's mark reference (captures entry slippage at first close).
       - For HELD names: mid-to-mid mark only (no transaction, no slippage).
     Apply turnover-aware fee on rotation events only (default 0.8 bps/side).
  8. Save new state.

Run modes:
    python -m live.xyz_paper_bot                   # one cycle now
    python -m live.xyz_paper_bot --universe tier_ab  # default 11 names K=4/M=1
    python -m live.xyz_paper_bot --universe tier_a   # 8 names K=3/M=1
    python -m live.xyz_paper_bot --universe full15   # 15 names K=5/M=2
    python -m live.xyz_paper_bot --notional-usd N    # per-leg shadow notional
    python -m live.xyz_paper_bot --taker-fee-bps F   # override default 0.8
    python -m live.xyz_paper_bot --no-refresh        # skip yfinance refresh
    python -m live.xyz_paper_bot --check-state       # dump current state

Cost model: per-side taker fee + walk-the-book slippage simulation.
Default config: taker fee 0.8 bps, $10k/leg notional. Backtest at 3.5 bps/side
total cost (slippage 2.7 + fee 0.8) gives active Sharpe +3.11 (CI [+1.79, +4.49]).
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("xyz_paper_bot")

STATE_DIR = ROOT / "live" / "state" / "xyz"
STATE_DIR.mkdir(parents=True, exist_ok=True)
POSITIONS_PATH = STATE_DIR / "positions.json"
CYCLES_PATH = STATE_DIR / "cycles.csv"
PREDICTIONS_PATH = STATE_DIR / "predictions.csv"

MODEL_DIR = ROOT / "models"
ARTIFACT_PATH = MODEL_DIR / "v7_xyz_ensemble.pkl"
META_PATH = MODEL_DIR / "v7_xyz_meta.json"

CACHE = ROOT / "data" / "ml" / "cache"

HL_INFO_URL = "https://api.hyperliquid.xyz/info"
COST_BPS_PER_SIDE = 1.5  # default for shadow accounting (used if L2 fills fail)
DEFAULT_NOTIONAL_USD = 10_000.0  # per-leg notional for fill simulation
DEFAULT_TAKER_FEE_BPS = 0.8  # current xyz growth-mode taker (per user, 2026-05-08)

try:
    from live.telegram import notify_telegram
except Exception:
    def notify_telegram(text, **kw):  # type: ignore
        return False


# ---- model artifact loading -------------------------------------------------

def load_artifact():
    with ARTIFACT_PATH.open("rb") as fh:
        models = pickle.load(fh)
    with META_PATH.open() as fh:
        meta = json.load(fh)
    log.info("loaded artifact: %d models, %d features",
             len(models), len(meta["feat_cols"]))
    log.info("  trained_at=%s", meta.get("trained_at_utc", "?"))
    log.info("  trained_universe_size=%d  presets_available=%s",
             len(meta.get("training_universe", [])),
             list(meta.get("execution_universe_presets", {}).keys())
              or "(legacy meta)")
    return models, meta


# ---- state persistence ------------------------------------------------------

def load_state() -> Optional[dict]:
    if not POSITIONS_PATH.exists():
        return None
    with POSITIONS_PATH.open() as fh:
        return json.load(fh)


def _json_default(o):
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(f"unserializable: {type(o)}")


def save_state(state: dict):
    """Atomic save via tmp + rename — durable against partial writes."""
    tmp = POSITIONS_PATH.with_suffix(POSITIONS_PATH.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(state, fh, indent=2, default=_json_default)
        fh.flush()
        try:
            import os as _os
            _os.fsync(fh.fileno())
        except (OSError, AttributeError):
            pass
    tmp.replace(POSITIONS_PATH)  # POSIX atomic rename


def append_cycle_row(row: dict):
    df = pd.DataFrame([row])
    if CYCLES_PATH.exists():
        df.to_csv(CYCLES_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(CYCLES_PATH, index=False)


def _flush_pending_cycle_row(state: dict) -> dict:
    """If state has a pending cycle row from a prior run, flush it to
    cycles.csv and return state with the field removed.

    Exactly-once semantics under crash: dedup by decision_ts before append.
    If a previous run died after append but before strip+save, this run
    sees the pending row, observes it's already in cycles.csv, and skips
    the append while still stripping the field.
    """
    pending = state.get("pending_cycle_row")
    if pending is None:
        return state
    decision_ts = str(pending.get("decision_ts"))
    already_present = False
    if CYCLES_PATH.exists():
        try:
            existing = pd.read_csv(CYCLES_PATH, usecols=["decision_ts"])
            already_present = (existing["decision_ts"].astype(str) == decision_ts).any()
        except Exception as e:
            log.warning("could not read cycles.csv for dedup check: %s", e)
    if already_present:
        log.info("pending cycle row for %s already in cycles.csv — skip duplicate append",
                 decision_ts)
    else:
        log.info("flushing pending cycle row from prior cycle (decision_ts=%s)",
                 decision_ts)
        append_cycle_row(pending)
    state = {k: v for k, v in state.items() if k != "pending_cycle_row"}
    save_state(state)
    return state


def _append_prediction_rows(ts_iso: str, rows: list[dict]):
    if PREDICTIONS_PATH.exists():
        try:
            existing_ts = pd.read_csv(PREDICTIONS_PATH, usecols=["ts"])
            if (existing_ts["ts"].astype(str) == ts_iso).any():
                log.info("predictions for %s already logged — skip duplicate append", ts_iso)
                return
        except Exception as e:
            log.warning("could not read predictions.csv for dedup check: %s", e)
    out = pd.DataFrame(rows)
    out["ts"] = ts_iso
    out = out[["symbol", "pred", "ts"]]
    if PREDICTIONS_PATH.exists():
        out.to_csv(PREDICTIONS_PATH, mode="a", header=False, index=False)
    else:
        out.to_csv(PREDICTIONS_PATH, index=False)


def _prediction_payload(ts: pd.Timestamp, preds: pd.DataFrame) -> dict:
    rows = []
    for _, r in preds[["symbol", "pred"]].iterrows():
        rows.append({"symbol": str(r["symbol"]), "pred": float(r["pred"])})
    return {"ts": pd.Timestamp(ts).isoformat(), "rows": rows}


def _flush_pending_prediction_rows(state: dict) -> dict:
    pending = state.get("pending_prediction_rows")
    if pending is None:
        return state
    ts_iso = str(pending.get("ts"))
    rows = pending.get("rows") or []
    if rows:
        _append_prediction_rows(ts_iso, rows)
    state = {k: v for k, v in state.items() if k != "pending_prediction_rows"}
    save_state(state)
    return state


def append_predictions(ts: pd.Timestamp, preds: pd.DataFrame):
    payload = _prediction_payload(ts, preds)
    _append_prediction_rows(payload["ts"], payload["rows"])


def _require_full_fill(fill: dict, symbol: str, side: str,
                       notional_usd: float) -> dict:
    """Return fill if complete; otherwise abort before state/accounting changes."""
    if not np.isfinite(fill.get("vwap", float("nan"))):
        raise RuntimeError(f"{symbol} {side} fill has no finite vwap")
    if not fill.get("fully_filled", False):
        filled = float(fill.get("filled_notional", 0.0) or 0.0)
        raise RuntimeError(
            f"{symbol} {side} L2 depth insufficient: filled ${filled:,.2f} "
            f"of ${notional_usd:,.2f}"
        )
    return fill


# ---- yfinance incremental refresh ------------------------------------------

def refresh_yf_cache(symbols: list[str], days_back: int = 10) -> int:
    """For each symbol, refetch the last `days_back` days from yfinance and
    merge into the cached parquet. Returns number of symbols updated.
    Cache key is the same as data_collectors.sp100_loader.fetch_daily.
    """
    import yfinance as yf
    n_updated = 0
    end = datetime.now(timezone.utc)
    start = end.replace(hour=0, minute=0, second=0, microsecond=0)
    start = start - pd.Timedelta(days=days_back)
    for i, sym in enumerate(symbols, 1):
        cache_path = CACHE / f"yf_{sym}_1d_sp100.parquet"
        try:
            df_new = yf.Ticker(sym).history(
                start=start.strftime("%Y-%m-%d"),
                end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d", auto_adjust=True,
            )
        except Exception as e:
            log.warning("  %s refresh failed: %s", sym, e)
            continue
        if df_new.empty:
            continue
        df_new.index = df_new.index.tz_convert("UTC")
        df_new = df_new.reset_index().rename(columns={
            "Date": "ts", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        df_new = df_new[["ts", "open", "high", "low", "close", "volume"]]
        df_new["ts"] = pd.to_datetime(df_new["ts"], utc=True).dt.normalize()
        df_new["symbol"] = sym
        df_new = df_new.dropna(subset=["close"])

        if cache_path.exists():
            df_old = pd.read_parquet(cache_path)
            combined = pd.concat([df_old, df_new], ignore_index=True)
            combined = (combined.drop_duplicates(subset=["ts"], keep="last")
                                   .sort_values("ts").reset_index(drop=True))
        else:
            combined = df_new
        combined.to_parquet(cache_path)
        n_updated += 1
        time.sleep(0.05)  # gentle pacing
    return n_updated


# ---- panel build ------------------------------------------------------------

def build_panel():
    """Build full S&P 100 panel with v7 features computed through latest day.
    Returns (panel, regime, anchors, feats, sym_to_id).
    """
    from data_collectors.sp100_loader import load_universe
    from ml.research.alpha_v7_freq_sweep import add_residual_and_label
    from ml.research.alpha_v7_multi import (
        add_features_A, add_returns_and_basket, load_anchors,
    )
    from ml.research.alpha_v7_pead_fixed import add_features_B_fixed
    from ml.research.alpha_v7_push import add_sector_features
    from ml.research.alpha_v7_regime import compute_regime_indicators

    panel, earnings, surv = load_universe()
    if panel.empty:
        raise RuntimeError("empty panel from load_universe")
    anchors = load_anchors()
    panel = add_returns_and_basket(panel)
    panel = add_residual_and_label(panel, 1)  # need fwd_resid_1d for IC sanity
    panel, feats_A = add_features_A(panel)
    panel, feats_B = add_features_B_fixed(panel, earnings)
    panel, feats_F = add_sector_features(panel)
    feats = feats_A + feats_B + feats_F + ["sym_id"]
    regime = compute_regime_indicators(panel, anchors)
    return panel, regime, anchors, feats


def map_sym_id(panel: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Use the artifact's frozen sym_to_id map (must match training)."""
    sym_to_id = meta["sym_to_id"]
    panel = panel.copy()
    panel["sym_id"] = panel["symbol"].map(sym_to_id).astype("Int64")
    return panel


def predict_for_latest_bar(panel: pd.DataFrame, models: dict,
                             feats: list[str],
                             require_features: list[str] | None = None) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Run ensemble (avg of 15 models) on the most recent bar and return
    predictions per symbol. LGBM natively handles NaN features; only the
    `require_features` list (defaults to A + sym_id) must be non-null.
    PEAD (B_*) features are allowed to be NaN — happens when yfinance lacks
    surprise data for a name's most recent earnings event.
    """
    if require_features is None:
        require_features = [c for c in feats if not c.startswith("B_")]
    latest_ts = panel["ts"].max()
    latest = panel[panel["ts"] == latest_ts].copy()
    if latest.empty:
        raise RuntimeError(f"no rows for latest ts {latest_ts}")
    latest = latest.dropna(subset=require_features)
    if latest.empty:
        raise RuntimeError(f"all rows at {latest_ts} dropped by require_features")
    X = latest[feats].to_numpy(dtype=np.float32)
    horizon_seed_preds = []
    for (h, seed), m in models.items():
        horizon_seed_preds.append(m.predict(X))
    latest["pred"] = np.mean(horizon_seed_preds, axis=0)
    n_nan = latest[feats].isna().sum().sum()
    if n_nan > 0:
        log.info("  predict ran with %d NaN feature cells (handled natively by LGBM)", n_nan)
    return latest_ts, latest[["symbol", "pred"]]


# ---- dispersion gate (PIT) -------------------------------------------------

def gate_open_at(regime: pd.DataFrame, ts: pd.Timestamp,
                   pctile: float = 0.6, window_days: int = 252) -> tuple[bool, dict]:
    """Returns (is_open, info_dict). PIT: trailing 252d quantile, shifted by 1."""
    rg = regime.sort_values("ts").reset_index(drop=True).copy()
    rg["thresh"] = (rg["disp_22d"]
                     .rolling(window=window_days, min_periods=60)
                     .quantile(pctile)
                     .shift(1))
    row = rg[rg["ts"] == ts]
    if row.empty:
        return False, {"reason": f"ts {ts} not in regime"}
    r = row.iloc[0]
    disp = r["disp_22d"]; thresh = r["thresh"]
    if pd.isna(disp) or pd.isna(thresh):
        return False, {"reason": "thresh or disp NaN", "disp": disp, "thresh": thresh}
    return disp >= thresh, {"disp": float(disp), "thresh": float(thresh)}


# ---- portfolio: Tier A K=3 M=1 hysteresis -----------------------------------

TIER_A = ["AAPL", "GOOGL", "META", "MSFT", "MU", "NVDA", "PLTR", "TSLA"]
TIER_AB = TIER_A + ["AMZN", "ORCL", "NFLX"]
FULL15 = TIER_AB + ["AMD", "INTC", "COST", "LLY"]

UNIVERSE_PRESETS = {
    # name -> (universe list, K, M)  // K/M scale ~ proportionally to N names
    "tier_a":  (TIER_A,  3, 1),
    "tier_ab": (TIER_AB, 4, 1),
    "full15":  (FULL15,  5, 2),
}
DEFAULT_PRESET = "tier_ab"


def select_with_hysteresis(preds: pd.DataFrame, prev_long: set, prev_short: set,
                             allowed: set, top_k: int,
                             exit_buffer: int) -> tuple[set, set]:
    """Return (new_long, new_short). Same logic as
    daily_portfolio_hysteresis but stateful across calls.

    Raises if there are too few non-NaN predictions to satisfy K + M
    hysteresis depth — this is a fail-loud signal, not a "go flat" trigger.
    The legitimate flat path is gate-closed, handled separately in run_cycle.
    """
    bar = preds[preds["symbol"].isin(allowed)].dropna(subset=["pred"]).copy()
    if len(bar) < 2 * top_k + exit_buffer:
        raise RuntimeError(
            f"insufficient predictions: {len(bar)} valid names < "
            f"{2 * top_k + exit_buffer} required for K={top_k} M={exit_buffer}. "
            f"Aborting before state mutation. Check feature pipeline / NaN "
            f"distribution / universe membership."
        )
    bar = bar.sort_values("pred").reset_index(drop=True)
    n = len(bar)
    bar["rank_top"] = n - 1 - bar.index
    bar["rank_bot"] = bar.index

    new_long = set(prev_long)
    for s in list(new_long):
        r = bar[bar["symbol"] == s]
        if r.empty or r["rank_top"].iloc[0] > top_k + exit_buffer - 1:
            new_long.discard(s)
    cands = bar[bar["rank_top"] < top_k]["symbol"].tolist()
    for s in cands:
        if len(new_long) >= top_k: break
        new_long.add(s)
    if len(new_long) > top_k:
        ranked = bar[bar["symbol"].isin(new_long)].sort_values("rank_top")
        new_long = set(ranked.head(top_k)["symbol"])

    new_short = set(prev_short)
    for s in list(new_short):
        r = bar[bar["symbol"] == s]
        if r.empty or r["rank_bot"].iloc[0] > top_k + exit_buffer - 1:
            new_short.discard(s)
    cands_s = bar[bar["rank_bot"] < top_k]["symbol"].tolist()
    for s in cands_s:
        if len(new_short) >= top_k: break
        new_short.add(s)
    if len(new_short) > top_k:
        ranked = bar[bar["symbol"].isin(new_short)].sort_values("rank_bot")
        new_short = set(ranked.head(top_k)["symbol"])

    return new_long, new_short


# ---- xyz mid price fetch ---------------------------------------------------

def fetch_xyz_l2_book(symbol: str) -> dict | None:
    """Fetch L2 orderbook for one xyz perp. Returns
    {'bids': [(px,sz),...], 'asks': [(px,sz),...], 'mid': float, 'time_ms': int}
    or None on failure."""
    try:
        r = requests.post(HL_INFO_URL,
                            json={"type": "l2Book", "coin": f"xyz:{symbol}"},
                            timeout=10)
        r.raise_for_status()
        j = r.json()
        levels = j.get("levels", [[], []])
        bids = [(float(l["px"]), float(l["sz"])) for l in levels[0]]
        asks = [(float(l["px"]), float(l["sz"])) for l in levels[1]]
        if not bids or not asks:
            return None
        return {"bids": bids, "asks": asks,
                "mid": 0.5 * (bids[0][0] + asks[0][0]),
                "time_ms": int(j.get("time", 0))}
    except Exception as e:
        log.warning("  l2 fetch %s failed: %s", symbol, e)
        return None


def simulate_taker_fill(book: dict, side: str, notional_usd: float) -> dict:
    """Walk one side of the book to fill `notional_usd`.
    side: 'buy' walks asks (long entry / short exit);
          'sell' walks bids (short entry / long exit).
    Returns vwap, mid, slippage_bps (signed +ve = adverse), qty,
    levels_consumed, fully_filled.
    """
    levels = book["asks"] if side == "buy" else book["bids"]
    mid = book["mid"]
    consumed_qty = 0.0
    consumed_notional = 0.0
    remaining = notional_usd
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
                "qty": 0.0, "levels_consumed": 0, "fully_filled": False,
                "filled_notional": 0.0}
    vwap = consumed_notional / consumed_qty
    sign = 1.0 if side == "buy" else -1.0
    slippage_bps = sign * (vwap - mid) / mid * 1e4
    return {"vwap": vwap, "mid": mid, "slippage_bps": slippage_bps,
            "qty": consumed_qty, "levels_consumed": levels_consumed,
            "fully_filled": fully_filled,
            "filled_notional": consumed_notional}


def fetch_xyz_mids(symbols: list[str]) -> dict[str, float]:
    """Return {symbol: mid_price} for active xyz perps. Pulls full universe
    in one call and filters."""
    payload = {"type": "metaAndAssetCtxs", "dex": "xyz"}
    r = requests.post(HL_INFO_URL, json=payload, timeout=15)
    r.raise_for_status()
    j = r.json()
    if not isinstance(j, list) or len(j) < 2:
        raise RuntimeError(f"unexpected xyz metaAndAssetCtxs response: {type(j)}")
    meta = j[0]
    ctxs = j[1]
    universe = meta.get("universe", [])
    out = {}
    for u, ctx in zip(universe, ctxs):
        name = u.get("name", "")
        if not name.startswith("xyz:"):
            continue
        sym = name[4:]
        if sym not in symbols:
            continue
        mid_str = ctx.get("midPx")
        if mid_str is None:
            continue
        out[sym] = float(mid_str)
    return out


# ---- shadow P&L of prior cycle ---------------------------------------------

def close_prior_cycle(prev_state: dict, exit_books: dict[str, dict | None],
                        new_long: set, new_short: set,
                        notional_usd: float, taker_fee_bps: float) -> dict:
    """Compute realized 1d P&L of prev_state against current xyz L2 books.

    Uses taker-fill simulation to get realistic exit prices including
    spread + walk-the-book slippage. Costs include actual fee + slippage
    on both entry (recorded at decision time) and exit (simulated now).

    Raises before writing accounting rows if a required quote, mark reference,
    or simulated fill is unavailable.
    """
    long_set = set(prev_state.get("long", []))
    short_set = set(prev_state.get("short", []))
    entry_fills = prev_state.get("entry_fills", {})
    entry_mids = prev_state.get("entry_mids", {})
    K = prev_state.get("top_k", 3)
    decision_ts = prev_state.get("decision_ts")
    per_name_notional = notional_usd / max(K, 1)

    # Per-leg P&L:
    #   - HELD names (in prev set AND new set): use mid-to-mid (no transaction, no slippage)
    #   - ROTATED-OUT names (in prev set, NOT in new set): close at exit_vwap (slippage paid)
    #   - NEW names (NOT in prev, in new): no P&L this cycle — they enter at curr decision
    #     time (entry slippage attributed to NEXT cycle when they rotate or are first marked)
    long_rets, short_rets, missing = [], [], []
    entry_slip_total = 0.0
    exit_slip_total = 0.0
    n_rotated_fills = 0
    n_held = 0

    def _per_name_pnl(s, side: str, opp_side: str, target_set: set):
        """side: 'long' or 'short'. opp_side: 'sell' (long exit) or 'buy' (short exit).
        Uses entry_mids[s] as the rolling mark-reference: on entry it's set to
        entry_vwap (captures entry slippage on first close); on subsequent held
        cycles it gets updated to that cycle's mid (clean mid-to-mid carry).
        """
        nonlocal entry_slip_total, exit_slip_total, n_rotated_fills, n_held
        prev_ref = entry_mids.get(s)
        bk = exit_books.get(s)
        if prev_ref is None:
            raise RuntimeError(f"missing mark reference for open position {s}")
        if bk is None:
            raise RuntimeError(f"missing exit book for open position {s}")
        cur_mid = bk["mid"]
        if s in target_set:
            n_held += 1
            return np.log(cur_mid / prev_ref)
        else:
            # ROTATED OUT: close at exit_vwap; mark vs prev ref (which absorbed
            # entry slippage exactly once at first close)
            ex = simulate_taker_fill(bk, opp_side, per_name_notional)
            ex = _require_full_fill(ex, s, opp_side, per_name_notional)
            ent = entry_fills.get(s) or {}
            entry_slip_total += ent.get("slippage_bps", 0.0) or 0.0
            exit_slip_total += ex["slippage_bps"]
            n_rotated_fills += 1
            return np.log(ex["vwap"] / prev_ref)

    # For both held and rotated, use entry_mids[s] as the prev-cycle mark
    # reference (it was set to entry_vwap on the cycle the name first entered,
    # then updated to that-cycle's mid on subsequent held cycles — see state-
    # save logic). This way the entry slippage is captured once at first mark.
    for s in long_set:
        r = _per_name_pnl(s, "long", "sell", new_long)
        if r is not None: long_rets.append(r)
    for s in short_set:
        r = _per_name_pnl(s, "short", "buy", new_short)
        if r is not None: short_rets.append(r)

    long_alpha = float(np.mean(long_rets)) if long_rets else 0.0
    short_alpha = float(np.mean(short_rets)) if short_rets else 0.0
    spread_bps_gross = (long_alpha - short_alpha) * 1e4

    avg_entry_slip = entry_slip_total / max(n_rotated_fills, 1)
    avg_exit_slip = exit_slip_total / max(n_rotated_fills, 1)

    # Turnover-aware fee cost: rotation count × 2 × taker_fee_bps,
    # normalized to leg notional via /K
    long_chg = len(long_set.symmetric_difference(new_long))
    short_chg = len(short_set.symmetric_difference(new_short))
    turnover = (long_chg + short_chg) / max(2 * K, 1)
    fee_bps = (long_chg + short_chg) / max(K, 1) * taker_fee_bps

    return {
        "decision_ts": decision_ts,
        "close_ts": datetime.now(timezone.utc).isoformat(),
        "K": K,
        "n_long": len(long_set),
        "n_short": len(short_set),
        "n_missing_mid": len(missing),
        "long_alpha_bps": long_alpha * 1e4,
        "short_alpha_bps": short_alpha * 1e4,
        "spread_bps": spread_bps_gross,
        "avg_entry_slip_bps": avg_entry_slip,
        "avg_exit_slip_bps": avg_exit_slip,
        "long_chg": long_chg,
        "short_chg": short_chg,
        "turnover": turnover,
        "fee_bps": fee_bps,
        "net_bps": spread_bps_gross - fee_bps,
        "missing": ",".join(sorted(missing)) if missing else "",
        "prev_long_set": ",".join(sorted(long_set)),
        "prev_short_set": ",".join(sorted(short_set)),
    }


# ---- main cycle ------------------------------------------------------------

def run_cycle(refresh: bool = True, preset: str = DEFAULT_PRESET,
               notional_usd: float = DEFAULT_NOTIONAL_USD,
               taker_fee_bps: float = DEFAULT_TAKER_FEE_BPS) -> dict:
    if preset not in UNIVERSE_PRESETS:
        raise ValueError(f"unknown preset {preset!r}; valid: {list(UNIVERSE_PRESETS)}")
    universe, top_k, exit_buffer = UNIVERSE_PRESETS[preset]
    log.info("=== xyz paper bot cycle start (preset=%s, N=%d, K=%d, M=%d, "
             "notional=$%.0f/leg, fee=%.2fbps/side) ===",
             preset, len(universe), top_k, exit_buffer, notional_usd, taker_fee_bps)
    models, meta = load_artifact()
    prev_state = load_state()
    if prev_state:
        # Flush any pending cycle row from prior run that died between
        # save_state and append_cycle_row (transactional recovery).
        prev_state = _flush_pending_cycle_row(prev_state)
        prev_state = _flush_pending_prediction_rows(prev_state)
        log.info("loaded prev state: decision_ts=%s long=%d short=%d",
                 prev_state.get("decision_ts"),
                 len(prev_state.get("long", [])),
                 len(prev_state.get("short", [])))
    else:
        log.info("no prev state — first cycle")

    if refresh:
        from data_collectors.sp100_loader import SP100
        log.info("refreshing yfinance cache (last 10 days)...")
        n = refresh_yf_cache(SP100, days_back=10)
        log.info("  refreshed %d/%d names", n, len(SP100))

    log.info("building panel + features...")
    panel, regime, _anchors, feats = build_panel()
    panel = map_sym_id(panel, meta)

    # Defensive: training and inference feature order MUST match. Without this
    # guard, a future refactor that reorders features in add_features_A/B/F
    # would silently corrupt predictions (LGBM fit was on column index, not
    # column name). Abort loudly if mismatch.
    if feats != meta["feat_cols"]:
        diff_idx = [i for i, (a, b) in enumerate(zip(feats, meta["feat_cols"]))
                      if a != b]
        raise RuntimeError(
            f"feat_cols mismatch between inference ({len(feats)}) and "
            f"trained meta ({len(meta['feat_cols'])}). Diffs at indices "
            f"{diff_idx[:5]}. Inference: {feats}. Trained: {meta['feat_cols']}. "
            f"Retrain the artifact, or fix the feature pipeline to match."
        )
    log.info("  panel: %d rows, %d symbols, latest ts=%s",
             len(panel), panel["symbol"].nunique(), panel["ts"].max())

    log.info("predicting for latest bar...")
    decision_ts, preds = predict_for_latest_bar(panel, models, feats)
    log.info("  decision_ts=%s, %d predictions", decision_ts, len(preds))
    # NOTE: predictions log appended only on first run for this decision_ts;
    # see same-day guard below.

    is_open, gate_info = gate_open_at(regime, decision_ts)
    log.info("  dispersion gate: %s  disp=%.4f thresh=%.4f",
             "OPEN" if is_open else "CLOSED",
             gate_info.get("disp", float("nan")),
             gate_info.get("thresh", float("nan")))

    # Compute new target weights first (so cost calc can be turnover-aware)
    prev_long = set(prev_state.get("long", [])) if prev_state else set()
    prev_short = set(prev_state.get("short", [])) if prev_state else set()
    ranked = preds[preds["symbol"].isin(universe)].sort_values("pred", ascending=False)
    ranking_pairs: list[tuple[str, float]] = [
        (r.symbol, float(r.pred)) for _, r in ranked.iterrows()
    ]
    same_day = (prev_state is not None
                  and str(prev_state.get("decision_ts", "")) == str(decision_ts))
    if same_day:
        log.info("  same decision_ts as prev (%s) — state/P&L left unchanged",
                 decision_ts)
        # Idempotent append fills a missing predictions log if a prior run
        # committed state but died before writing predictions.
        append_predictions(decision_ts, preds)
        _send_rebalance_telegram(
            decision_ts=decision_ts, preset=preset, top_k=top_k,
            exit_buffer=exit_buffer, gate_open=bool(is_open),
            gate_disp=gate_info.get("disp", float("nan")),
            gate_thresh=gate_info.get("thresh", float("nan")),
            new_long=prev_long, new_short=prev_short,
            ranking_pairs=ranking_pairs, cycle_row=None,
            entry_fills=prev_state.get("entry_fills", {}) if prev_state else {},
            prev_long=prev_long, prev_short=prev_short,
            notional_usd=notional_usd, same_day=True,
        )
        return prev_state

    if not is_open:
        log.info("  gate closed — flat next cycle (no positions)")
        new_long, new_short = set(), set()
    else:
        new_long, new_short = select_with_hysteresis(
            preds, prev_long, prev_short, set(universe),
            top_k=top_k, exit_buffer=exit_buffer)
        log.info("  new long  = %s", sorted(new_long))
        log.info("  new short = %s", sorted(new_short))
        log.info("  ranking (%s): %s", preset,
                 ", ".join(f"{s}:{p:+.4f}" for s, p in ranking_pairs))

    # Fetch L2 books for: (a) prior-cycle exit fills, (b) new-cycle entry fills
    # Cover both prev positions (need exit fills) and new positions (need entry fills).
    book_symbols = sorted(set(universe) | set(prev_state.get("long", []) if prev_state else [])
                            | set(prev_state.get("short", []) if prev_state else [])
                            | new_long | new_short)
    books: dict[str, dict | None] = {}
    for s in book_symbols:
        books[s] = fetch_xyz_l2_book(s)
        time.sleep(0.05)
    n_book_ok = sum(1 for b in books.values() if b is not None)
    log.info("  fetched L2 books: %d/%d names", n_book_ok, len(book_symbols))
    current_mids = {s: b["mid"] for s, b in books.items() if b}

    required_books = sorted(prev_long | prev_short | new_long | new_short)
    missing_books = [s for s in required_books if books.get(s) is None]
    if missing_books:
        raise RuntimeError(
            "missing xyz L2 books for required position symbols; refusing to "
            f"mutate state: {missing_books}"
        )

    # Close prior cycle (if any) — but skip if same decision_ts as prev
    cycle_row = None
    if prev_state and (prev_state.get("long") or prev_state.get("short")):
        cycle_row = close_prior_cycle(prev_state, books, new_long, new_short,
                                        notional_usd=notional_usd,
                                        taker_fee_bps=taker_fee_bps)
        cycle_row["gate_open_now"] = bool(is_open)
        cycle_row["disp_now"] = gate_info.get("disp")
        cycle_row["thresh_now"] = gate_info.get("thresh")
        cycle_row["notional_usd"] = notional_usd
        cycle_row["taker_fee_bps"] = taker_fee_bps
        log.info("  prior cycle closed: spread=%+.2fbps slip(in/out)=%+.2f/%+.2fbps "
                 "turnover=%.2f fee=%+.2fbps net=%+.2fbps",
                 cycle_row["spread_bps"],
                 cycle_row["avg_entry_slip_bps"], cycle_row["avg_exit_slip_bps"],
                 cycle_row["turnover"], cycle_row["fee_bps"], cycle_row["net_bps"])

    # Simulate entry fills for newly-entered names only. Preserve prev entry
    # fills for held names so their original slippage diagnostic isn't lost.
    per_name_notional = notional_usd / max(top_k, 1)
    prev_fills = prev_state.get("entry_fills", {}) if prev_state else {}
    entry_fills: dict[str, dict] = {}
    n_new_sims = 0
    for s in new_long:
        if s in prev_long and s in prev_fills:
            entry_fills[s] = prev_fills[s]  # held: preserve original
            continue
        bk = books.get(s)
        if bk is None: continue
        f = simulate_taker_fill(bk, "buy", per_name_notional)
        f = _require_full_fill(f, s, "buy", per_name_notional)
        entry_fills[s] = {"vwap": f["vwap"], "mid": f["mid"],
                            "slippage_bps": f["slippage_bps"],
                            "qty": f["qty"], "fully_filled": f["fully_filled"]}
        n_new_sims += 1
    for s in new_short:
        if s in prev_short and s in prev_fills:
            entry_fills[s] = prev_fills[s]
            continue
        bk = books.get(s)
        if bk is None: continue
        f = simulate_taker_fill(bk, "sell", per_name_notional)
        f = _require_full_fill(f, s, "sell", per_name_notional)
        entry_fills[s] = {"vwap": f["vwap"], "mid": f["mid"],
                            "slippage_bps": f["slippage_bps"],
                            "qty": f["qty"], "fully_filled": f["fully_filled"]}
        n_new_sims += 1
    if n_new_sims:
        new_slips = [entry_fills[s]["slippage_bps"] for s in (new_long | new_short)
                       if s not in (prev_long | prev_short) and s in entry_fills]
        avg_slip = float(np.mean(new_slips)) if new_slips else 0.0
        log.info("  entry fills: %d new sims (avg slip=%+.2fbps), %d preserved",
                 n_new_sims, avg_slip, len(entry_fills) - n_new_sims)
    else:
        log.info("  no rotations — all %d positions held with preserved fills",
                 len(entry_fills))

    # Mark-reference rule for next cycle's close:
    #   - newly entered names: ref = entry_vwap (first close picks up entry slippage)
    #   - held names: ref = today's mid (clean mid-to-mid carry, no slippage re-paid)
    next_mark_refs = {}
    for s in new_long:
        if s in prev_long:
            if s in current_mids:
                next_mark_refs[s] = current_mids[s]
        else:
            f = entry_fills.get(s)
            if f and np.isfinite(f["vwap"]):
                next_mark_refs[s] = f["vwap"]
    for s in new_short:
        if s in prev_short:
            if s in current_mids:
                next_mark_refs[s] = current_mids[s]
        else:
            f = entry_fills.get(s)
            if f and np.isfinite(f["vwap"]):
                next_mark_refs[s] = f["vwap"]

    state = {
        "decision_ts": str(decision_ts),
        "decision_at_utc": datetime.now(timezone.utc).isoformat(),
        "long": sorted(new_long),
        "short": sorted(new_short),
        "entry_fills": entry_fills,
        "entry_mids": next_mark_refs,
        "top_k": top_k,
        "exit_buffer": exit_buffer,
        "preset": preset,
        "universe": universe,
        "notional_usd": notional_usd,
        "taker_fee_bps": taker_fee_bps,
        "gate_open": is_open,
        "gate_disp": gate_info.get("disp"),
        "gate_thresh": gate_info.get("thresh"),
        "missing_mids": [s for s in (new_long | new_short) if s not in current_mids],
    }
    # Transactional commit: stage cycle_row inside state, save atomically.
    # The next run's load_state() will flush pending_cycle_row to cycles.csv
    # before processing. Guarantees we never write cycles.csv without an
    # accompanying durable state update — and never lose a realized row to
    # a half-completed save.
    if cycle_row is not None:
        state["pending_cycle_row"] = cycle_row
    state["pending_prediction_rows"] = _prediction_payload(decision_ts, preds)
    save_state(state)
    if cycle_row is not None:
        # Try to flush immediately; if this dies, next load will retry. Either
        # way the row will reach cycles.csv exactly once.
        state = _flush_pending_cycle_row(state)
    state = _flush_pending_prediction_rows(state)
    log.info("=== cycle done. state saved to %s ===", POSITIONS_PATH)

    _send_rebalance_telegram(
        decision_ts=decision_ts, preset=preset, top_k=top_k, exit_buffer=exit_buffer,
        gate_open=bool(is_open),
        gate_disp=gate_info.get("disp", float("nan")),
        gate_thresh=gate_info.get("thresh", float("nan")),
        new_long=new_long, new_short=new_short,
        ranking_pairs=ranking_pairs,
        cycle_row=cycle_row,
        entry_fills=entry_fills,
        prev_long=prev_long, prev_short=prev_short,
        notional_usd=notional_usd,
        same_day=False,
    )
    return state


def _send_rebalance_telegram(*, decision_ts, preset, top_k, exit_buffer,
                                gate_open, gate_disp, gate_thresh,
                                new_long, new_short, ranking_pairs,
                                cycle_row, entry_fills, prev_long, prev_short,
                                notional_usd, same_day: bool = False):
    """Daily rebalance summary for Telegram."""
    def _bps(x):
        if x is None or not np.isfinite(x): return "n/a"
        return f"{x:+.2f}"

    rows = [
        f"📊 <b>v7 xyz rebalance</b>  ({datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC)",
        f"decision_ts: {pd.Timestamp(decision_ts).strftime('%Y-%m-%d')}  •  "
        f"preset: {preset}  •  K={top_k} M={exit_buffer}  •  notional: ${notional_usd:,.0f}/leg",
        f"",
    ]
    rows.append(f"Gate: <b>{'OPEN' if gate_open else 'CLOSED'}</b>  "
                f"(disp={gate_disp:.4f} vs thresh {gate_thresh:.4f})")

    # Closing prior cycle
    if same_day:
        rows += [
            "",
            "<i>Same-day re-run; state and realized P&L left unchanged.</i>",
        ]
    elif cycle_row is not None:
        rows += [
            f"",
            f"<b>Prior cycle realized</b>:",
            f"  spread: {_bps(cycle_row['spread_bps'])} bps  "
            f"(long {_bps(cycle_row['long_alpha_bps'])} − short {_bps(cycle_row['short_alpha_bps'])})",
            f"  rotations: long Δ{cycle_row['long_chg']}  short Δ{cycle_row['short_chg']}  "
            f"turnover {cycle_row['turnover']:.2f}",
            f"  slippage: in {_bps(cycle_row['avg_entry_slip_bps'])} • out {_bps(cycle_row['avg_exit_slip_bps'])} bps",
            f"  fee: {_bps(cycle_row['fee_bps'])} bps",
            f"  <b>net: {_bps(cycle_row['net_bps'])} bps</b>",
        ]
    else:
        rows.append(f"<i>First cycle — no prior to close.</i>")

    # Cumulative shadow stats from cycles.csv
    if CYCLES_PATH.exists():
        try:
            d = pd.read_csv(CYCLES_PATH)
            n = len(d)
            cum = float(d["net_bps"].sum())
            mean = float(d["net_bps"].mean()) if n else 0.0
            hit = float((d["net_bps"] > 0).mean()) if n else 0.0
            sh = (d["net_bps"].mean() / d["net_bps"].std() * np.sqrt(252)
                    if n > 2 and d["net_bps"].std() > 0 else 0.0)
            rows += [
                f"",
                f"<b>Cumulative</b> (N={n}): {cum:+.1f} bps  •  "
                f"mean/cyc {mean:+.2f} bps  •  hit {hit:.0%}  •  Sh~{sh:+.2f}",
            ]
        except Exception:
            pass

    # Ranking
    if ranking_pairs:
        rows += ["", f"<b>Ranking ({preset})</b>:"]
        for sym, pred in ranking_pairs:
            mark = ""
            if sym in new_long: mark = " 🟢"
            elif sym in new_short: mark = " 🔴"
            rows.append(f"  {sym:<6} {pred:+.4f}{mark}")

    # New target weights
    if same_day:
        rows += [
            "",
            f"<b>Open Long</b>: {', '.join(sorted(new_long)) or '(none)'}",
            f"<b>Open Short</b>: {', '.join(sorted(new_short)) or '(none)'}",
        ]
    elif gate_open:
        long_changes = sorted(set(new_long) - set(prev_long))
        long_drops = sorted(set(prev_long) - set(new_long))
        short_changes = sorted(set(new_short) - set(prev_short))
        short_drops = sorted(set(prev_short) - set(new_short))

        rows += ["", f"<b>Long</b>: {', '.join(sorted(new_long))}"]
        if long_changes: rows.append(f"  +new: {', '.join(long_changes)}")
        if long_drops:   rows.append(f"  -drop: {', '.join(long_drops)}")
        rows.append(f"<b>Short</b>: {', '.join(sorted(new_short))}")
        if short_changes: rows.append(f"  +new: {', '.join(short_changes)}")
        if short_drops:   rows.append(f"  -drop: {', '.join(short_drops)}")

        # Entry fill quality for new positions
        new_names = (set(new_long) | set(new_short)) - (set(prev_long) | set(prev_short))
        if new_names and entry_fills:
            slips = [entry_fills[s].get("slippage_bps", 0)
                     for s in new_names if s in entry_fills]
            if slips:
                rows.append(f"<i>New-entry avg slip: {np.mean(slips):+.2f} bps</i>")
    else:
        rows += ["", f"<b>Flat next cycle</b> (gate closed)"]

    text = "\n".join(rows)
    sent = notify_telegram(text)
    log.info("telegram rebalance: %s (%d chars)",
             "sent" if sent else "skipped", len(text))
    return sent


def cli_check_state():
    state = load_state()
    if state is None:
        print("(no state)"); return
    print(json.dumps(state, indent=2, default=str))
    if CYCLES_PATH.exists():
        df = pd.read_csv(CYCLES_PATH)
        print(f"\ncycles log: {len(df)} rows")
        if not df.empty:
            print(df.tail(5).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true",
                     help="skip yfinance refresh (use cached data only)")
    ap.add_argument("--check-state", action="store_true",
                     help="print current state and exit")
    ap.add_argument("--universe", default=DEFAULT_PRESET,
                     choices=list(UNIVERSE_PRESETS),
                     help=f"universe preset (default {DEFAULT_PRESET})")
    ap.add_argument("--notional-usd", type=float, default=DEFAULT_NOTIONAL_USD,
                     help=f"per-leg shadow notional (default ${DEFAULT_NOTIONAL_USD:.0f})")
    ap.add_argument("--taker-fee-bps", type=float, default=DEFAULT_TAKER_FEE_BPS,
                     help=f"per-side taker fee bps (default {DEFAULT_TAKER_FEE_BPS})")
    args = ap.parse_args()

    if args.check_state:
        cli_check_state(); return

    run_cycle(refresh=not args.no_refresh, preset=args.universe,
                notional_usd=args.notional_usd,
                taker_fee_bps=args.taker_fee_bps)


if __name__ == "__main__":
    main()
