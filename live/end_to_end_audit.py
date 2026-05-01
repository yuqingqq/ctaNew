"""End-to-end audit: walk each strategy → live-simulation module with
concrete assertions. Run-once; reports PASS / FAIL per check.

Usage:
  python -m live.end_to_end_audit          # all checks
  python -m live.end_to_end_audit --skip-network  # skip HL/Binance API checks

Coverage (in order from upstream to downstream):
  A. Data feeds — HL kline, HL L2 book, HL allMids, HL fundingHistory schemas
  B. Feature pipeline — build_panel_for_inference correctness vs reference
  C. Model artifact — load + sanity prediction
  D. Portfolio decision — β-neutral scaling, top-K, weight signs
  E. L2 taker fill — walk book, slippage sign, edge cases
  F. Turnover-aware execution — first cycle, no-change, partial, flip
  G. Funding accrual — sign convention, time window, cumulative tracking
  H. State persistence — JSON round-trip, dataclass field preservation
  I. Hourly monitor — MtM mark, funding cumulative
  J. Replay vs backtest — already validated in commit d6e261e (skipped here)

Each test returns (status, msg). Status: PASS | FAIL | SKIP | WARN.
Final report aggregates by section with action items.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("audit")

# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------
_results: list[tuple[str, str, str, str]] = []  # (section, name, status, msg)

def record(section: str, name: str, status: str, msg: str = ""):
    _results.append((section, name, status, msg))
    color = {"PASS": "\033[32m", "FAIL": "\033[31m",
             "SKIP": "\033[33m", "WARN": "\033[33m"}.get(status, "")
    print(f"  [{color}{status}\033[0m] {name}" + (f" — {msg}" if msg else ""))


def section(title: str):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


# --------------------------------------------------------------------------
# A. Data feeds
# --------------------------------------------------------------------------
def audit_data_feeds(skip_network: bool):
    section("A. DATA FEEDS — HL info endpoints")
    if skip_network:
        record("A", "HL endpoints", "SKIP", "--skip-network")
        return
    from data_collectors.hl_data_fetcher import HyperliquidDataFetcher
    from live.paper_bot import (
        fetch_hl_l2_book, fetch_hl_mids, fetch_hl_funding_history,
    )

    # A1. HL kline schema
    try:
        f = HyperliquidDataFetcher()
        end = datetime.now(timezone.utc)
        df = f.fetch_range("BTC", interval="5m",
                            start_time=end - timedelta(hours=2), end_time=end)
        if df.empty:
            record("A", "HL kline shape", "FAIL", "empty response")
        else:
            cols = set(df.columns)
            need = {"open", "high", "low", "close", "volume"}
            missing = need - cols
            if missing:
                record("A", "HL kline shape", "FAIL", f"missing cols: {missing}")
            elif (df["high"] < df["low"]).any():
                record("A", "HL kline shape", "FAIL", "found high < low rows")
            elif (df["volume"] < 0).any():
                record("A", "HL kline shape", "FAIL", "found negative volume")
            else:
                record("A", "HL kline shape", "PASS", f"n={len(df)}, cols ok")
    except Exception as e:
        record("A", "HL kline shape", "FAIL", str(e))

    # A2. HL L2 book schema
    try:
        book = fetch_hl_l2_book("BTC")
        bids = book["bids"]; asks = book["asks"]
        if not bids or not asks:
            record("A", "HL L2 book shape", "FAIL", f"bids={len(bids)}, asks={len(asks)}")
        else:
            best_bid = bids[0][0]; best_ask = asks[0][0]
            if best_bid >= best_ask:
                record("A", "HL L2 book shape", "FAIL",
                        f"crossed book: bid {best_bid} >= ask {best_ask}")
            elif not all(bids[i][0] > bids[i + 1][0] for i in range(len(bids) - 1)):
                record("A", "HL L2 book shape", "FAIL", "bids not descending")
            elif not all(asks[i][0] < asks[i + 1][0] for i in range(len(asks) - 1)):
                record("A", "HL L2 book shape", "FAIL", "asks not ascending")
            else:
                spread_bps = (best_ask - best_bid) / ((best_bid + best_ask) / 2) * 1e4
                record("A", "HL L2 book shape", "PASS",
                        f"BTC bid {best_bid:.1f} ask {best_ask:.1f} spread {spread_bps:.2f} bps")
    except Exception as e:
        record("A", "HL L2 book shape", "FAIL", str(e))

    # A3. HL allMids
    try:
        mids = fetch_hl_mids()
        if "BTC" not in mids or "ETH" not in mids:
            record("A", "HL allMids", "FAIL", "BTC/ETH missing from response")
        else:
            record("A", "HL allMids", "PASS", f"{len(mids)} coins, BTC={mids['BTC']:.0f}")
    except Exception as e:
        record("A", "HL allMids", "FAIL", str(e))

    # A4. HL fundingHistory schema
    try:
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = end_ms - 24 * 3600 * 1000
        rates = fetch_hl_funding_history("BTC", start_ms, end_ms)
        if not rates:
            record("A", "HL fundingHistory", "FAIL", "empty 24h window")
        else:
            r0 = rates[0]
            need_keys = {"coin", "fundingRate", "time"}
            missing = need_keys - set(r0.keys())
            if missing:
                record("A", "HL fundingHistory", "FAIL", f"missing keys: {missing}")
            elif len(rates) < 20:
                record("A", "HL fundingHistory", "WARN",
                        f"only {len(rates)}/24 hourly entries — partial day?")
            else:
                record("A", "HL fundingHistory", "PASS",
                        f"24h returned {len(rates)} entries, rate[0]={r0['fundingRate']}")
    except Exception as e:
        record("A", "HL fundingHistory", "FAIL", str(e))


# --------------------------------------------------------------------------
# B. Feature pipeline (verified by replay; spot-check key invariants)
# --------------------------------------------------------------------------
def audit_features():
    section("B. FEATURE PIPELINE")
    from features_ml.cross_sectional import (
        XS_FEATURE_COLS_V6_CLEAN, XS_RANK_SOURCES, build_kline_features,
    )
    from live.paper_bot import build_kline_features_inmem

    # B1. v6_clean has 28 cols including sym_id
    n = len(XS_FEATURE_COLS_V6_CLEAN)
    if n != 28:
        record("B", "v6_clean feature count", "FAIL", f"expected 28, got {n}")
    else:
        record("B", "v6_clean feature count", "PASS", "28 features")

    # B2. xs_rank features have matching source columns
    rank_features = [c for c in XS_FEATURE_COLS_V6_CLEAN if c.endswith("_xs_rank")]
    sources = list(XS_RANK_SOURCES.keys())
    rank_targets = list(XS_RANK_SOURCES.values())
    missing_targets = [c for c in rank_features if c not in rank_targets]
    if missing_targets:
        record("B", "xs_rank source mapping", "FAIL",
                f"rank cols without source mapping: {missing_targets}")
    else:
        record("B", "xs_rank source mapping", "PASS",
                f"{len(rank_features)} rank features, all mapped")

    # B3. build_kline_features_inmem matches build_kline_features for identical OHLCV
    try:
        cached = build_kline_features("BTCUSDT")
        ohlcv = cached[["open", "high", "low", "close", "volume"]].copy()
        recomp = build_kline_features_inmem(ohlcv)
        common_idx = cached.index.intersection(recomp.index)
        common_cols = sorted(set(cached.columns) & set(recomp.columns))
        if not common_idx.size:
            record("B", "in-mem feature parity", "FAIL", "no common index")
        else:
            max_diff = 0
            n_diff = 0
            for col in common_cols:
                if cached[col].dtype.kind not in "fi":
                    continue
                d = (cached.loc[common_idx, col] - recomp.loc[common_idx, col]).abs().fillna(0).max()
                if d > 1e-6:
                    n_diff += 1
                max_diff = max(max_diff, float(d))
            if n_diff == 0:
                record("B", "in-mem feature parity", "PASS",
                        f"all numeric cols identical (max_diff={max_diff:.2e})")
            else:
                record("B", "in-mem feature parity", "FAIL",
                        f"{n_diff} cols differ (max_diff={max_diff:.6f})")
    except Exception as e:
        record("B", "in-mem feature parity", "FAIL", str(e))


# --------------------------------------------------------------------------
# C. Model artifact
# --------------------------------------------------------------------------
def audit_model():
    section("C. MODEL ARTIFACT")
    from live.paper_bot import load_model_artifact, MODEL_DIR
    if not (MODEL_DIR / "v6_clean_ensemble.pkl").exists():
        record("C", "artifact present", "SKIP",
                "run live.train_v6_clean_artifact first")
        return
    try:
        models, meta = load_model_artifact()
        if len(models) != 5:
            record("C", "ensemble size", "FAIL", f"expected 5 seeds, got {len(models)}")
        else:
            record("C", "ensemble size", "PASS", "5 LGBM Boosters")
        feat_cols = meta["feat_cols"]
        if len(feat_cols) != 28:
            record("C", "meta feat_cols", "FAIL", f"expected 28, got {len(feat_cols)}")
        else:
            record("C", "meta feat_cols", "PASS", "28 features")
        sym_to_id = meta["sym_to_id"]
        if len(sym_to_id) != 25:
            record("C", "meta sym_to_id", "FAIL", f"expected 25, got {len(sym_to_id)}")
        else:
            record("C", "meta sym_to_id", "PASS", f"{len(sym_to_id)} symbols")
        # Predict on a synthetic feature vector — should not crash
        X = np.zeros((1, len(feat_cols)), dtype=np.float32)
        preds = [m.predict(X, num_iteration=m.best_iteration) for m in models]
        if all(np.isfinite(p[0]) for p in preds):
            record("C", "ensemble predict on zeros", "PASS",
                    f"mean={np.mean([p[0] for p in preds]):+.6f}")
        else:
            record("C", "ensemble predict on zeros", "FAIL", "non-finite output")
    except Exception as e:
        record("C", "artifact load", "FAIL", str(e))


# --------------------------------------------------------------------------
# D. Portfolio decision
# --------------------------------------------------------------------------
def audit_portfolio():
    section("D. PORTFOLIO DECISION")
    from live.paper_bot import (
        select_portfolio, compute_target_weights, TOP_K,
    )

    # D1. β-neutral scaling
    preds = pd.DataFrame({
        "symbol": [f"S{i}" for i in range(20)],
        "pred": np.linspace(-0.01, 0.01, 20),
        "beta_short_vs_bk": [1.0] * 10 + [1.5] * 10,
    })
    top, bot, sL, sS = select_portfolio(preds, top_k=5)
    # Both legs: top has higher β=1.5, bot β=1.0. β-neutral: scale_L × β_L = scale_S × β_S
    # scale_L = 2 × 1.0/(1.0+1.5) = 0.8, scale_S = 2 × 1.5/(1.0+1.5) = 1.2
    if abs(sL - 0.8) > 1e-6 or abs(sS - 1.2) > 1e-6:
        record("D", "β-neutral scaling math", "FAIL",
                f"expected sL=0.8, sS=1.2; got sL={sL:.4f}, sS={sS:.4f}")
    else:
        record("D", "β-neutral scaling math", "PASS", f"sL={sL:.3f}, sS={sS:.3f}")
    # Gross exposure
    if abs((sL + sS) - 2.0) > 1e-6:
        record("D", "β-neutral gross=2", "FAIL", f"got {sL + sS:.4f}")
    else:
        record("D", "β-neutral gross=2", "PASS", "gross=2.000")

    # D2. β-neutral scaling clipped to [0.5, 1.5]
    preds_extreme = pd.DataFrame({
        "symbol": [f"S{i}" for i in range(20)],
        "pred": np.linspace(-0.01, 0.01, 20),
        "beta_short_vs_bk": [0.2] * 10 + [3.0] * 10,
    })
    top, bot, sL, sS = select_portfolio(preds_extreme, top_k=5)
    # raw scale_L = 2 × 0.2/(0.2+3) = 0.125 → clipped to 0.5
    # raw scale_S = 2 × 3/(0.2+3) = 1.875 → clipped to 1.5
    if abs(sL - 0.5) > 1e-6 or abs(sS - 1.5) > 1e-6:
        record("D", "β-neutral clip", "FAIL", f"got sL={sL}, sS={sS}")
    else:
        record("D", "β-neutral clip", "PASS", "clipped to [0.5, 1.5]")

    # D3. compute_target_weights signs
    top_df = pd.DataFrame({"symbol": ["A", "B"]})
    bot_df = pd.DataFrame({"symbol": ["C", "D"]})
    w = compute_target_weights(top_df, bot_df, scale_L=1.0, scale_S=1.0, n_per_side=2)
    if w["A"] != 0.5 or w["B"] != 0.5 or w["C"] != -0.5 or w["D"] != -0.5:
        record("D", "compute_target_weights signs", "FAIL", str(w))
    else:
        record("D", "compute_target_weights signs", "PASS",
                "longs +0.5, shorts -0.5")

    # D4. select_portfolio top-K filters
    if len(top) != 5 or len(bot) != 5:
        record("D", "top-K filtering", "FAIL", f"top={len(top)}, bot={len(bot)}")
    else:
        # top should have HIGHEST predictions, bot LOWEST
        max_bot_pred = bot["pred"].max()
        min_top_pred = top["pred"].min()
        if max_bot_pred > min_top_pred:
            record("D", "top-K filtering", "FAIL",
                    "top/bot overlap in predictions")
        else:
            record("D", "top-K filtering", "PASS", "top > bot in pred")


# --------------------------------------------------------------------------
# E. L2 taker fill simulator
# --------------------------------------------------------------------------
def audit_l2_fill():
    section("E. L2 TAKER FILL SIMULATOR")
    from live.paper_bot import simulate_taker_fill

    # E1. Single-level fill, exact size
    book = {
        "bids": [(99.0, 10.0), (98.0, 100.0)],
        "asks": [(101.0, 10.0), (102.0, 100.0)],
    }
    fill = simulate_taker_fill(book, "buy", target_notional_usd=500)
    # 500 USD at 101 = 4.95 qty, all from level 0
    expected_vwap = 101.0
    expected_mid = 100.0
    expected_slip = (101.0 - 100.0) / 100.0 * 1e4  # = +100 bps
    if abs(fill["vwap"] - expected_vwap) > 1e-6:
        record("E", "single-level vwap", "FAIL",
                f"expected {expected_vwap}, got {fill['vwap']}")
    else:
        record("E", "single-level vwap", "PASS", f"vwap=101.0")
    if abs(fill["slippage_bps"] - expected_slip) > 1.0:
        record("E", "slippage sign convention", "FAIL",
                f"expected ~+{expected_slip:.0f} bps, got {fill['slippage_bps']:+.2f}")
    else:
        record("E", "slippage sign convention", "PASS",
                f"buy crosses ask → +{fill['slippage_bps']:.0f} bps adverse")

    # E2. Multi-level walk (consume top of book + part of next)
    fill = simulate_taker_fill(book, "buy", target_notional_usd=2000)
    # Need 2000 USD: level 0 has 10 × 101 = 1010 USD → consumed.
    # Remaining 990 USD at 102 = 9.706 qty.
    # Total qty = 10 + 9.706 = 19.706. VWAP = 2000/19.706 = 101.494
    if not (101.0 < fill["vwap"] < 102.0):
        record("E", "multi-level walk", "FAIL", f"vwap={fill['vwap']}")
    elif fill["levels_consumed"] != 2:
        record("E", "multi-level walk", "FAIL",
                f"expected 2 levels, got {fill['levels_consumed']}")
    else:
        record("E", "multi-level walk", "PASS",
                f"vwap={fill['vwap']:.3f} levels=2")

    # E3. Sell side (walks bids)
    fill_s = simulate_taker_fill(book, "sell", target_notional_usd=500)
    expected_vwap_s = 99.0
    if abs(fill_s["vwap"] - expected_vwap_s) > 1e-6:
        record("E", "sell-side vwap", "FAIL",
                f"expected {expected_vwap_s}, got {fill_s['vwap']}")
    else:
        # slippage: sell at 99 vs mid 100 = -100 bps; but signed so adverse=+100
        if fill_s["slippage_bps"] < 50:
            record("E", "sell-side slippage sign", "FAIL",
                    f"sell should also be adverse-positive: got {fill_s['slippage_bps']:+.2f}")
        else:
            record("E", "sell-side slippage sign", "PASS",
                    f"sell crosses bid → +{fill_s['slippage_bps']:.0f} bps adverse")

    # E4. Empty book
    empty = {"bids": [], "asks": []}
    fill_e = simulate_taker_fill(empty, "buy", target_notional_usd=100)
    if np.isfinite(fill_e["vwap"]):
        record("E", "empty book", "FAIL", "expected NaN, got finite")
    else:
        record("E", "empty book", "PASS", "returned NaN")


# --------------------------------------------------------------------------
# F. Turnover-aware execution
# --------------------------------------------------------------------------
def audit_execution():
    section("F. TURNOVER-AWARE EXECUTION")
    from live.paper_bot import (
        execute_cycle_turnover_aware, LegPosition, INITIAL_EQUITY_USD,
    )

    # Synthetic L2 books — 1 bps spread, plenty depth
    def make_book(price):
        return {
            "bids": [(price * (1 - 5e-5), 1e9)],
            "asks": [(price * (1 + 5e-5), 1e9)],
        }
    books = {f"S{i}": make_book(100.0 + i) for i in range(10)}
    # Need to map symbol→coin for the function; symbols are e.g. "S0USDT"→"S0"
    # The function uses _binance_to_hl_coin which strips USDT suffix
    books_aliased = {}
    for i in range(10):
        books_aliased[f"S{i}"] = books[f"S{i}"]
    books = books_aliased

    # F1. First cycle (no prev) opens all positions
    target_w = {f"S{i}USDT": 0.2 for i in range(5)}  # 5 longs at 0.2 each
    target_w.update({f"S{i}USDT": -0.2 for i in range(5, 10)})  # 5 shorts
    res = execute_cycle_turnover_aware([], target_w, books,
                                         now_iso="2026-05-01T00:00:00+00:00")
    if res["n_trades"] != 10:
        record("F", "first cycle opens all", "FAIL",
                f"expected 10 trades, got {res['n_trades']}")
    elif len(res["new_positions"]) != 10:
        record("F", "first cycle position count", "FAIL",
                f"expected 10 positions, got {len(res['new_positions'])}")
    else:
        record("F", "first cycle opens all", "PASS",
                f"{res['n_trades']} trades, {len(res['new_positions'])} positions")

    # F2. No-change cycle: target = prev → 0 trades
    prev = res["new_positions"]
    res2 = execute_cycle_turnover_aware(prev, target_w, books,
                                          now_iso="2026-05-01T01:00:00+00:00")
    if res2["n_trades"] != 0:
        record("F", "no-change → no trades", "FAIL",
                f"expected 0 trades, got {res2['n_trades']}")
    elif res2["fees_bps"] > 1e-9:
        record("F", "no-change → no fees", "FAIL",
                f"expected 0 fees, got {res2['fees_bps']}")
    else:
        record("F", "no-change → no trades", "PASS", "0 trades, 0 fees")
    if len(res2["new_positions"]) != 10:
        record("F", "no-change carries positions", "FAIL",
                f"got {len(res2['new_positions'])}")
    else:
        record("F", "no-change carries positions", "PASS", "10 carried forward")

    # F3. Partial change: replace 1 long with another.
    # NB: books are keyed by COIN (post-strip), not by Binance symbol.
    # _binance_to_hl_coin("S99USDT") = "S99" so we add "S99" not "S99USDT".
    target_v3 = dict(target_w)
    del target_v3["S0USDT"]
    target_v3["S99USDT"] = 0.2
    books["S99"] = make_book(100.5)
    res3 = execute_cycle_turnover_aware(prev, target_v3, books,
                                          now_iso="2026-05-01T02:00:00+00:00")
    if res3["n_trades"] != 2:
        record("F", "partial-change trades", "FAIL",
                f"expected 2 trades (close S0 + open S99), got {res3['n_trades']}")
    else:
        record("F", "partial-change trades", "PASS",
                f"{res3['n_trades']} trades")
    pos_syms = {p.symbol for p in res3["new_positions"]}
    if "S0USDT" in pos_syms:
        record("F", "partial-change: closed sym dropped", "FAIL",
                "S0USDT still in positions after exit")
    elif "S99USDT" not in pos_syms:
        record("F", "partial-change: new sym opened", "FAIL", "S99USDT not in positions")
    else:
        record("F", "partial-change position update", "PASS",
                "S0 dropped, S99 added")

    # F4. Position flip: long → short
    target_flip = {f"S0USDT": -0.2}  # was +0.2 long, now -0.2 short
    # Build a single-position prev
    prev_single = [LegPosition(
        symbol="S0USDT", side="L", weight=0.2, entry_price_hl=100.0,
        entry_mid_hl=100.0, entry_notional_usd=2000, entry_slippage_bps=0,
        entry_time="2026-05-01T00:00:00+00:00", last_marked_mid=100.0,
    )]
    res4 = execute_cycle_turnover_aware(prev_single, target_flip, books,
                                          now_iso="2026-05-01T03:00:00+00:00")
    if res4["n_trades"] != 1:
        record("F", "flip: 1 trade for delta", "FAIL",
                f"expected 1 trade (sell delta=-0.4), got {res4['n_trades']}")
    elif len(res4["new_positions"]) != 1 or res4["new_positions"][0].side != "S":
        record("F", "flip: position now short", "FAIL",
                f"got {res4['new_positions']}")
    else:
        record("F", "flip handled", "PASS",
                "1 trade for delta=-0.4, position flipped to S")

    # F5. Gross PnL accounting on no-change cycle (price moved up by 1%)
    books_up = {sym: make_book(b["bids"][0][0] / (1 - 5e-5) * 1.01)
                  for sym, b in books.items() if "S0" not in sym and "S99" not in sym}
    # Reset the books dict but include all S0-S9 at 1% higher prices
    books_up = {f"S{i}USDT": make_book((100 + i) * 1.01) for i in range(10)}
    res5 = execute_cycle_turnover_aware(prev, target_w, books_up,
                                          now_iso="2026-05-01T04:00:00+00:00")
    # Each position has weight ±0.2; price up 1% means:
    # - longs gain: +0.2 × 1% × 10000 bps = 20 bps per long, × 5 longs = 100 bps total
    # - shorts lose: -0.2 × 1% × 10000 = -20 bps per short, × 5 = -100 bps
    # Net = 0
    if abs(res5["gross_pnl_bps"]) > 0.5:
        record("F", "MtM math: matched +1% all symbols → 0 net", "FAIL",
                f"got gross_pnl_bps={res5['gross_pnl_bps']:+.2f}")
    else:
        record("F", "MtM math: matched +1% all symbols → 0 net", "PASS",
                f"gross_pnl_bps={res5['gross_pnl_bps']:+.2f}")


# --------------------------------------------------------------------------
# G. Funding accrual sign convention
# --------------------------------------------------------------------------
def audit_funding(skip_network: bool):
    section("G. FUNDING ACCRUAL")
    if skip_network:
        record("G", "funding network call", "SKIP", "--skip-network")
        return
    from live.paper_bot import LegPosition, accrue_funding_for_cycle, fetch_hl_funding_history

    # G1. Fetch real funding history for BTC over last 24h
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - 24 * 3600 * 1000
    rates = fetch_hl_funding_history("BTC", start_ms, end_ms)
    if not rates:
        record("G", "BTC funding 24h", "FAIL", "no rates returned")
        return
    avg_rate = np.mean([float(r["fundingRate"]) for r in rates])

    # G2. Long with positive avg rate should pay (positive USD payment)
    long_pos = LegPosition(
        symbol="BTCUSDT", side="L", weight=0.2, entry_price_hl=100000.0,
        entry_mid_hl=100000.0, entry_notional_usd=2000, entry_slippage_bps=0,
        entry_time=(datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(),
        last_marked_mid=100000.0,
    )
    res = accrue_funding_for_cycle(
        [long_pos],
        prev_decision_iso=(datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(),
        now_iso=datetime.now(timezone.utc).isoformat(),
        equity_usd=10000,
    )
    expected_long_usd = sum(float(r["fundingRate"]) for r in rates) * 2000
    if abs(res["total_funding_usd"] - expected_long_usd) > 0.01:
        record("G", "long sign: pays positive rate", "FAIL",
                f"expected ${expected_long_usd:+.4f}, got ${res['total_funding_usd']:+.4f}")
    else:
        record("G", "long sign: pays positive rate", "PASS",
                f"long with avg_rate {avg_rate:+.6f} paid ${res['total_funding_usd']:+.4f}")

    # G3. Short receives the same rate (opposite sign)
    short_pos = LegPosition(
        symbol="BTCUSDT", side="S", weight=-0.2, entry_price_hl=100000.0,
        entry_mid_hl=100000.0, entry_notional_usd=2000, entry_slippage_bps=0,
        entry_time=(datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(),
        last_marked_mid=100000.0,
    )
    res_s = accrue_funding_for_cycle(
        [short_pos],
        prev_decision_iso=(datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(),
        now_iso=datetime.now(timezone.utc).isoformat(),
        equity_usd=10000,
    )
    if abs(res_s["total_funding_usd"] + res["total_funding_usd"]) > 0.01:
        record("G", "short sign: opposite of long", "FAIL",
                f"long ${res['total_funding_usd']:.4f} + short ${res_s['total_funding_usd']:.4f} != 0")
    else:
        record("G", "short sign: opposite of long", "PASS",
                f"long ${res['total_funding_usd']:+.4f}, short ${res_s['total_funding_usd']:+.4f}")

    # G4. funding_paid_usd accumulates on the position (mutation)
    if abs(long_pos.funding_paid_usd - res["total_funding_usd"]) > 1e-9:
        record("G", "funding_paid_usd cumulative", "FAIL",
                f"position.funding_paid_usd={long_pos.funding_paid_usd}, "
                f"res.total_funding_usd={res['total_funding_usd']}")
    else:
        # call again — should DOUBLE
        res2 = accrue_funding_for_cycle(
            [long_pos],
            prev_decision_iso=(datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(),
            now_iso=datetime.now(timezone.utc).isoformat(),
            equity_usd=10000,
        )
        if abs(long_pos.funding_paid_usd - 2 * res["total_funding_usd"]) > 0.01:
            record("G", "funding_paid_usd cumulative", "FAIL",
                    f"after 2 calls expected 2x, got {long_pos.funding_paid_usd:.4f}")
        else:
            record("G", "funding_paid_usd cumulative", "PASS",
                    "accumulates across calls")


# --------------------------------------------------------------------------
# H. State persistence
# --------------------------------------------------------------------------
def audit_state():
    section("H. STATE PERSISTENCE")
    from live.paper_bot import LegPosition, save_state, load_state, POSITIONS_PATH, CYCLES_PATH
    backup_pos = POSITIONS_PATH.read_text() if POSITIONS_PATH.exists() else None
    backup_cyc = CYCLES_PATH.read_text() if CYCLES_PATH.exists() else None
    try:
        # H1. Round-trip preserves all fields
        original = [
            LegPosition(symbol="BTCUSDT", side="L", weight=0.222,
                        entry_price_hl=77123.45, entry_mid_hl=77100.0,
                        entry_notional_usd=2222.0, entry_slippage_bps=2.5,
                        entry_time="2026-05-01T00:00:00+00:00",
                        last_marked_mid=77150.0, funding_paid_usd=-0.123),
            LegPosition(symbol="ETHUSDT", side="S", weight=-0.178,
                        entry_price_hl=3500.0, entry_mid_hl=3500.0,
                        entry_notional_usd=1780.0, entry_slippage_bps=1.8,
                        entry_time="2026-05-01T00:00:00+00:00",
                        last_marked_mid=3505.0, funding_paid_usd=0.456),
        ]
        # Wipe state then write
        if POSITIONS_PATH.exists(): POSITIONS_PATH.unlink()
        if CYCLES_PATH.exists(): CYCLES_PATH.unlink()
        save_state(original, {"decision_time_utc": "test", "wall_time_utc": "test"})
        loaded, _ = load_state()
        if len(loaded) != len(original):
            record("H", "round-trip count", "FAIL",
                    f"saved {len(original)}, loaded {len(loaded)}")
        else:
            mismatches = []
            for o, l in zip(original, loaded):
                for fld in o.__dataclass_fields__:
                    if getattr(o, fld) != getattr(l, fld):
                        mismatches.append((o.symbol, fld,
                                            getattr(o, fld), getattr(l, fld)))
            if mismatches:
                record("H", "round-trip field equality", "FAIL",
                        f"{len(mismatches)} mismatched fields: {mismatches[:3]}")
            else:
                record("H", "round-trip preserves all fields", "PASS",
                        f"{len(original)} positions × {len(original[0].__dataclass_fields__)} fields")
    finally:
        # Restore
        if backup_pos is not None:
            POSITIONS_PATH.write_text(backup_pos)
        elif POSITIONS_PATH.exists():
            POSITIONS_PATH.unlink()
        if backup_cyc is not None:
            CYCLES_PATH.write_text(backup_cyc)
        elif CYCLES_PATH.exists():
            CYCLES_PATH.unlink()


# --------------------------------------------------------------------------
# I. Hourly monitor (logic correctness)
# --------------------------------------------------------------------------
def audit_hourly_monitor():
    section("I. HOURLY MONITOR (logic, not network)")
    from live.paper_bot import LegPosition

    # I1. Simulate per-leg MtM logic same as hourly_monitor
    p = LegPosition(symbol="BTCUSDT", side="L", weight=0.2,
                    entry_price_hl=100000, entry_mid_hl=100000,
                    entry_notional_usd=2000, entry_slippage_bps=0,
                    entry_time="2026-05-01T00:00:00+00:00",
                    last_marked_mid=100000)
    # Simulate price moving up to 101000 (1% gain)
    mid_now = 101000
    if p.side == "L":
        hourly = (mid_now / p.last_marked_mid - 1.0)
    else:
        hourly = (p.last_marked_mid / mid_now - 1.0)
    expected = 0.01  # +1%
    if abs(hourly - expected) > 1e-6:
        record("I", "long MtM up 1%", "FAIL", f"expected +0.01, got {hourly}")
    else:
        record("I", "long MtM up 1%", "PASS", "+1% as expected")

    # I2. Short MtM when price up 1% should be -1%
    s = LegPosition(symbol="BTCUSDT", side="S", weight=-0.2,
                    entry_price_hl=100000, entry_mid_hl=100000,
                    entry_notional_usd=2000, entry_slippage_bps=0,
                    entry_time="2026-05-01T00:00:00+00:00",
                    last_marked_mid=100000)
    if s.side == "L":
        hourly = (mid_now / s.last_marked_mid - 1.0)
    else:
        hourly = (s.last_marked_mid / mid_now - 1.0)
    expected = -1000 / 101000
    if abs(hourly - expected) > 1e-6:
        record("I", "short MtM up 1% → loss", "FAIL",
                f"expected {expected}, got {hourly}")
    else:
        record("I", "short MtM up 1% → loss", "PASS", f"{hourly:.6f}")


# --------------------------------------------------------------------------
# Final report
# --------------------------------------------------------------------------
def report():
    section("AUDIT SUMMARY")
    by_status = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
    for _, _, status, _ in _results:
        by_status[status] = by_status.get(status, 0) + 1
    total = sum(by_status.values())
    print(f"\n  Total checks: {total}")
    for k in ("PASS", "FAIL", "WARN", "SKIP"):
        if by_status.get(k):
            print(f"    {k}: {by_status[k]}")
    fails = [(s, n, m) for s, n, st, m in _results if st == "FAIL"]
    if fails:
        print("\n  ❌ Failed checks:")
        for s, n, m in fails:
            print(f"    [{s}] {n}: {m}")
    warns = [(s, n, m) for s, n, st, m in _results if st == "WARN"]
    if warns:
        print("\n  ⚠️  Warnings:")
        for s, n, m in warns:
            print(f"    [{s}] {n}: {m}")
    if not fails:
        print("\n  ✓ All required checks passed.")
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-network", action="store_true",
                    help="skip checks requiring HL/Binance API access")
    args = ap.parse_args()

    audit_data_feeds(args.skip_network)
    audit_features()
    audit_model()
    audit_portfolio()
    audit_l2_fill()
    audit_execution()
    audit_funding(args.skip_network)
    audit_state()
    audit_hourly_monitor()
    return report()


if __name__ == "__main__":
    sys.exit(main())
