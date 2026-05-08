"""v7 xyz alpha-residual paper-trade bot — daily cycle, shadow mode.

Each invocation = one decision cycle. Designed to run as a daily cron after
US RTH close (~21:30 UTC). State persists in live/state/xyz/.

Cycle steps:
  1. Load v7 model artifact (15 models) and previous state.
  2. Refresh yfinance daily data for full S&P 100 (incremental, last ~10 days).
  3. If a previous open portfolio exists: fetch current xyz mid prices,
     mark-to-market against prior entry mids → realized 1d P&L per leg →
     append to cycles.csv (closing the prior cycle).
  4. Build full panel (returns, basket, residual, A+B+sector features).
  5. Predict for today's bar using all 15 models, average.
  6. Filter to Tier A 8 names, rank, apply hysteresis (K=3, M=1) using prior
     state's long/short sets.
  7. Apply dispersion gate (60-pctile of trailing 252d).
  8. Fetch xyz mids for new long/short → these become next cycle's entry mids.
  9. Save new state.

Run modes:
    python -m live.xyz_paper_bot                   # one cycle now
    python -m live.xyz_paper_bot --no-refresh      # skip yfinance refresh
    python -m live.xyz_paper_bot --check-state     # dump current state

Cost model (memory): 1.5 bps/side patient maker on xyz growth-mode.
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
COST_BPS_PER_SIDE = 1.5


# ---- model artifact loading -------------------------------------------------

def load_artifact():
    with ARTIFACT_PATH.open("rb") as fh:
        models = pickle.load(fh)
    with META_PATH.open() as fh:
        meta = json.load(fh)
    log.info("loaded artifact: %d models, %d features",
             len(models), len(meta["feat_cols"]))
    log.info("  trained_at=%s", meta.get("trained_at_utc", "?"))
    log.info("  execution_universe=%s", meta["execution_universe"])
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
    with POSITIONS_PATH.open("w") as fh:
        json.dump(state, fh, indent=2, default=_json_default)


def append_cycle_row(row: dict):
    df = pd.DataFrame([row])
    if CYCLES_PATH.exists():
        df.to_csv(CYCLES_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(CYCLES_PATH, index=False)


def append_predictions(ts: pd.Timestamp, preds: pd.DataFrame):
    out = preds[["symbol", "pred"]].copy()
    out["ts"] = pd.Timestamp(ts).isoformat()
    if PREDICTIONS_PATH.exists():
        out.to_csv(PREDICTIONS_PATH, mode="a", header=False, index=False)
    else:
        out.to_csv(PREDICTIONS_PATH, index=False)


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
DEFAULT_PRESET = "tier_a"


def select_with_hysteresis(preds: pd.DataFrame, prev_long: set, prev_short: set,
                             allowed: set, top_k: int,
                             exit_buffer: int) -> tuple[set, set]:
    """Return (new_long, new_short). Same logic as
    daily_portfolio_hysteresis but stateful across calls."""
    bar = preds[preds["symbol"].isin(allowed)].dropna(subset=["pred"]).copy()
    if len(bar) < 2 * top_k + exit_buffer:
        log.warning("  bar size %d < %d required for K=%d M=%d — skip cycle",
                    len(bar), 2 * top_k + exit_buffer, top_k, exit_buffer)
        return set(), set()
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

def close_prior_cycle(prev_state: dict, current_mids: dict[str, float],
                        new_long: set, new_short: set) -> dict:
    """Compute realized 1d P&L of prev_state against current xyz mids.
    Cost is turnover-aware: fraction of leg notional that rotated to new
    target × 2 × cost_per_side (matches backtest's daily_portfolio_basic).
    Returns a row to be appended to cycles.csv."""
    long_set = set(prev_state.get("long", []))
    short_set = set(prev_state.get("short", []))
    entry_mids = prev_state.get("entry_mids", {})
    K = prev_state.get("top_k", 3)
    decision_ts = prev_state.get("decision_ts")

    long_rets, short_rets, missing = [], [], []
    for s in long_set:
        if s in current_mids and s in entry_mids and entry_mids[s] > 0:
            long_rets.append(np.log(current_mids[s] / entry_mids[s]))
        else:
            missing.append(s)
    for s in short_set:
        if s in current_mids and s in entry_mids and entry_mids[s] > 0:
            short_rets.append(np.log(current_mids[s] / entry_mids[s]))
        else:
            missing.append(s)
    long_alpha = float(np.mean(long_rets)) if long_rets else 0.0
    short_alpha = float(np.mean(short_rets)) if short_rets else 0.0
    spread = long_alpha - short_alpha

    # Turnover-aware cost (backtest formula)
    long_chg = len(long_set.symmetric_difference(new_long))
    short_chg = len(short_set.symmetric_difference(new_short))
    turnover = (long_chg + short_chg) / max(2 * K, 1)
    cost_bps = turnover * 2 * COST_BPS_PER_SIDE

    return {
        "decision_ts": decision_ts,
        "close_ts": datetime.now(timezone.utc).isoformat(),
        "K": K,
        "n_long": len(long_set),
        "n_short": len(short_set),
        "n_missing_mid": len(missing),
        "long_alpha_bps": long_alpha * 1e4,
        "short_alpha_bps": short_alpha * 1e4,
        "spread_bps": spread * 1e4,
        "long_chg": long_chg,
        "short_chg": short_chg,
        "turnover": turnover,
        "cost_bps": cost_bps,
        "net_bps": spread * 1e4 - cost_bps,
        "missing": ",".join(sorted(missing)) if missing else "",
        "prev_long_set": ",".join(sorted(long_set)),
        "prev_short_set": ",".join(sorted(short_set)),
    }


# ---- main cycle ------------------------------------------------------------

def run_cycle(refresh: bool = True) -> dict:
    log.info("=== xyz paper bot cycle start ===")
    models, meta = load_artifact()
    prev_state = load_state()
    if prev_state:
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
    log.info("  panel: %d rows, %d symbols, latest ts=%s",
             len(panel), panel["symbol"].nunique(), panel["ts"].max())

    log.info("predicting for latest bar...")
    decision_ts, preds = predict_for_latest_bar(panel, models, feats)
    log.info("  decision_ts=%s, %d predictions", decision_ts, len(preds))
    append_predictions(decision_ts, preds)

    is_open, gate_info = gate_open_at(regime, decision_ts)
    log.info("  dispersion gate: %s  disp=%.4f thresh=%.4f",
             "OPEN" if is_open else "CLOSED",
             gate_info.get("disp", float("nan")),
             gate_info.get("thresh", float("nan")))

    # Compute new target weights first (so cost calc can be turnover-aware)
    if not is_open:
        log.info("  gate closed — flat next cycle (no positions)")
        new_long, new_short = set(), set()
    else:
        prev_long = set(prev_state.get("long", [])) if prev_state else set()
        prev_short = set(prev_state.get("short", [])) if prev_state else set()
        new_long, new_short = select_with_hysteresis(
            preds, prev_long, prev_short, set(TIER_A))
        log.info("  new long  = %s", sorted(new_long))
        log.info("  new short = %s", sorted(new_short))
        ranked = preds[preds["symbol"].isin(TIER_A)].sort_values("pred", ascending=False)
        log.info("  ranking (Tier A): %s",
                 ", ".join(f"{r.symbol}:{r.pred:+.4f}" for _, r in ranked.iterrows()))

    # Fetch current xyz mids (used for both prior-cycle close and new-cycle entries)
    current_mids = {}
    try:
        current_mids = fetch_xyz_mids(TIER_A)
        log.info("  fetched xyz mids: %d/%d names", len(current_mids), len(TIER_A))
    except Exception as e:
        log.warning("  xyz mid fetch failed: %s", e)

    # Close prior cycle (if any) — but skip if same decision_ts as prev
    # (re-run on same day; we'd be marking against unchanged entry mids)
    cycle_row = None
    same_day = (prev_state is not None
                  and str(prev_state.get("decision_ts", "")) == str(decision_ts))
    if same_day:
        log.info("  same decision_ts as prev (%s) — re-run on same day, skip cycle close",
                 decision_ts)
    elif prev_state and (prev_state.get("long") or prev_state.get("short")):
        cycle_row = close_prior_cycle(prev_state, current_mids, new_long, new_short)
        cycle_row["gate_open_now"] = bool(is_open)
        cycle_row["disp_now"] = gate_info.get("disp")
        cycle_row["thresh_now"] = gate_info.get("thresh")
        log.info("  prior cycle closed: spread=%+.2fbps turnover=%.2f cost=%+.2fbps net=%+.2fbps",
                 cycle_row["spread_bps"], cycle_row["turnover"],
                 cycle_row["cost_bps"], cycle_row["net_bps"])
        append_cycle_row(cycle_row)

    # New entry mids (only for selected names)
    new_entry_mids = {s: current_mids[s] for s in (new_long | new_short)
                        if s in current_mids}
    state = {
        "decision_ts": str(decision_ts),
        "decision_at_utc": datetime.now(timezone.utc).isoformat(),
        "long": sorted(new_long),
        "short": sorted(new_short),
        "entry_mids": new_entry_mids,
        "top_k": TOP_K,
        "exit_buffer": EXIT_BUFFER,
        "tier": "A",
        "gate_open": is_open,
        "gate_disp": gate_info.get("disp"),
        "gate_thresh": gate_info.get("thresh"),
        "missing_mids": [s for s in (new_long | new_short) if s not in current_mids],
    }
    save_state(state)
    log.info("=== cycle done. state saved to %s ===", POSITIONS_PATH)
    return state


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
    args = ap.parse_args()

    if args.check_state:
        cli_check_state(); return

    run_cycle(refresh=not args.no_refresh)


if __name__ == "__main__":
    main()
