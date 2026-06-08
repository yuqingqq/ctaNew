"""Real-fill round-trip PnL ledger for the convexity v1 forward test (Hyperliquid execution).

Books PnL from REAL Hyperliquid fills, not Binance close-to-close. Each cycle consumes the legs the bot
decided (decision.json `turnover` = net per-symbol trade) priced at the real HL fill captured at the bar
boundary (convexity_slippage --decide), maintains a per-symbol FIFO lot book, realizes PnL when a trade
reduces a position (round-trip on HL prices), and marks open lots at the current HL mid.

Because entry AND exit are both on HL, the Binance<->HL price-LEVEL basis cancels in the round-trip; what
remains is the true tradeable HL path net of: book-walk slippage (fill vs mid), latency drift (exec mid vs
the Binance bar-close the signal saw), and taker fee. The first two are logged as a per-cycle cost
decomposition. Forward-only: the ledger starts when switched on (no historical HL book to replay).

State (realfill/ledger.json): equity0, realized_cum, lots{sym:[[signed_units, entry_px],...]},
last_open_time, cycles[]. PnL is in dollars off equity0 (default $10k).

Usage:
  python3 live/convexity_realfill.py --state live/state/convexity_v1   # update from decision.json + slip csv
  python3 live/convexity_realfill.py --selftest                        # synthetic round-trip assertions
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

REPO = Path("/home/yuqing/ctaNew")
EQUITY0 = 10000.0
TAKER_FEE_BPS = 4.5


def _new_ledger(equity0: float = EQUITY0) -> dict:
    return {"equity0": equity0, "realized_cum": 0.0, "lots": {}, "last_open_time": None, "cycles": []}


def load_ledger(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else _new_ledger()


def save_ledger(path: Path, led: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(led, indent=2))


def _apply_trade(lots: list, signed_units: float, fill_px: float) -> float:
    """Apply a signed trade (long +, short -) to a symbol's FIFO lot list at fill_px. Returns realized $.
    Consumes opposite-sign lots FIFO (round-trip realization), then opens a lot for any remainder (incl.
    a flip). Mutates `lots`. Realized for a closed portion = (fill_px - entry_px) * closed_signed_units,
    which is correct for both longs (sold higher = +) and shorts (bought back lower = +)."""
    realized = 0.0
    rem = signed_units
    while abs(rem) > 1e-12 and lots and (lots[0][0] > 0) != (rem > 0):   # opposite sign -> close
        u, px = lots[0]
        amt = min(abs(rem), abs(u))
        u_portion = (1.0 if u > 0 else -1.0) * amt
        realized += (fill_px - px) * u_portion
        u_new = u - u_portion
        rem += u_portion
        if abs(u_new) < 1e-12:
            lots.pop(0)
        else:
            lots[0][0] = u_new
    if abs(rem) > 1e-12:                                                 # same-sign remainder -> open lot
        lots.append([rem, fill_px])
    return realized


def mark_to_market(lots_by_sym: dict, mids: dict) -> tuple[float, float, int]:
    """Unrealized $ + gross exposure $ + #open-symbols, marking each open lot at the current HL mid."""
    unreal = gross = 0.0
    n = 0
    for sym, lots in lots_by_sym.items():
        if not lots:
            continue
        n += 1
        mid = mids.get(sym)
        for u, px in lots:
            mk = mid if mid is not None and np.isfinite(mid) else px     # fall back to entry if no mid
            unreal += (mk - px) * u
            gross += abs(u * mk)
    return unreal, gross, n


def update_cycle(led: dict, decision: dict, fills: dict, mids: dict, close_ref: dict,
                 decision_mids: dict | None = None, fee_bps: float = TAKER_FEE_BPS) -> dict:
    """Book one cycle: apply the decided turnover at real HL fills, realize round-trips, mark open lots.
    fills:         {sym: {"fill_px","mid","slippage_bps","fully_filled"}}  real HL fill per traded leg
    mids:          {sym: hl_mid}   current HL mid for ALL open symbols (for MtM); falls back to fills' mid
    close_ref:     {sym: binance_close_px}   Binance bar-close the signal saw (for BASIS reporting only)
    decision_mids: {sym: hl_mid_at_bar_close}  HL mid at the decision instant → TRUE HL→HL latency ref.
                   If absent, latency falls back to the Binance cref (basis-contaminated — flagged)."""
    ot = decision["open_time"]
    # size off the real-fill book's OWN equity (independent $10k track), not the modeled strategy equity —
    # the bot's weights are equity-independent fractions, so only the $ scale differs (bps stay comparable).
    equity = float(led["cycles"][-1]["equity"]) if led.get("cycles") else float(led["equity0"])
    net_after = {s: float(w) for s, w in decision.get("net_after", {}).items()}
    turnover = {s for s, w in decision.get("turnover", {}).items() if abs(float(w)) > 1e-9}
    lots = led["lots"]
    realized = fees = 0.0
    book_cost = lat_cost = basis_cost = traded_notional = 0.0
    n_filled = n_unfilled = 0
    decision_mids = decision_mids or {}
    for sym in turnover:                                                # rebalance each symbol the bot traded
        f = fills.get(sym)
        fill_px = f.get("fill_px") if f else None
        if fill_px is None or not np.isfinite(fill_px) or fill_px <= 0:
            n_unfilled += 1
            continue                                                    # probe failed -> skip leg (logged)
        sym_lots = lots.setdefault(sym, [])
        held = sum(u for u, _ in sym_lots)
        target_units = net_after.get(sym, 0.0) * equity / fill_px       # hold net notional weight at fill px
        trade_units = target_units - held                               # → on a full exit (target 0) closes ALL held
        if abs(trade_units) < 1e-9:
            if not sym_lots:
                lots.pop(sym, None)
            continue
        n_filled += 1
        mid = f.get("mid")
        mid = mid if (mid is not None and np.isfinite(mid)) else fill_px
        notional = abs(trade_units * fill_px)
        realized += _apply_trade(sym_lots, trade_units, fill_px)
        if not sym_lots:
            lots.pop(sym, None)
        fees += notional * fee_bps / 1e4
        traded_notional += notional
        side = 1.0 if trade_units > 0 else -1.0
        book_cost += notional * side * (fill_px - mid) / mid            # adverse fill vs HL mid (>0 = cost)
        # LATENCY = HL exec mid vs HL DECISION mid — both HL, so basis-free (the real drift in the latency
        # window). BASIS (HL decision mid vs Binance cref) is reported separately; it CANCELS on the round
        # trip (entry & exit both on HL) so it is NOT charged in exec_cost. Fall back to cref only if no
        # decision mid was captured (then latency is basis-contaminated — a degraded measurement).
        dmid = decision_mids.get(sym)
        cref = close_ref.get(sym)
        if dmid is not None and np.isfinite(dmid) and dmid > 0:
            lat_cost += notional * side * (mid - dmid) / dmid
            if cref and np.isfinite(cref) and cref > 0:
                basis_cost += notional * side * (dmid - cref) / cref
        elif cref and np.isfinite(cref) and cref > 0:
            lat_cost += notional * side * (mid - cref) / cref           # fallback: basis-contaminated
    realized -= fees
    led["realized_cum"] = float(led.get("realized_cum", 0.0)) + realized
    # carry-forward mids for open syms not traded this cycle so MtM still works (use last-known via fills mid)
    for s, f in fills.items():
        if s not in mids and f and np.isfinite(f.get("mid", np.nan)):
            mids[s] = f["mid"]
    unreal, gross, n_open = mark_to_market(lots, mids)
    eq = led["equity0"] + led["realized_cum"] + unreal
    rec = {
        "open_time": ot, "regime": decision.get("regime"),
        "n_trades": n_filled, "n_unfilled": n_unfilled,
        "realized_pnl": round(realized, 4), "realized_cum": round(led["realized_cum"], 4),
        "unrealized_pnl": round(unreal, 4), "equity": round(eq, 2),
        "gross_exposure": round(gross, 2), "n_open_syms": n_open,
        "fee_bps": round(fees / traded_notional * 1e4, 2) if traded_notional else 0.0,
        "book_slip_bps": round(book_cost / traded_notional * 1e4, 2) if traded_notional else 0.0,
        "latency_drift_bps": round(lat_cost / traded_notional * 1e4, 2) if traded_notional else 0.0,
        "basis_bps": round(basis_cost / traded_notional * 1e4, 2) if traded_notional else 0.0,  # cancels round-trip
        "exec_cost_bps": round((fees + book_cost + lat_cost) / traded_notional * 1e4, 2) if traded_notional else 0.0,
        "traded_notional": round(traded_notional, 1),
    }
    led["last_open_time"] = ot
    led.setdefault("cycles", []).append(rec)
    return rec


# ---- live wiring: read decision.json + the decide-slip csv for the same bar ----------------------------

def _read_decide_slip(csv_path: Path, open_time: str) -> dict:
    """Parse convexity_slippage --decide rows for `open_time` -> {sym: {fill_px, mid, slippage_bps,...}}."""
    out = {}
    if not csv_path.exists():
        return out
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if str(row.get("bar_open_time")) != str(open_time):
                continue
            def _f(k):
                try:
                    return float(row[k])
                except (KeyError, ValueError, TypeError):
                    return float("nan")
            out[row["symbol"]] = {"fill_px": _f("fill_px"), "mid": _f("mid_px"),
                                  "slippage_bps": _f("slippage_bps"),
                                  "fully_filled": str(row.get("fully_filled")) == "True"}
    return out


def selftest():
    """Synthetic 2-cycle round-trip: enter long+short at real fills, exit next cycle; assert realized PnL."""
    led = _new_ledger(10000.0)
    # cycle 1: open +0.10 ABC (fill 100, mid 99.9, close 99.8) and -0.10 XYZ (fill 50, mid 50.05, close 50.1)
    d1 = {"open_time": "T1", "regime": "side", "equity": 10000.0,
          "net_after": {"ABC": 0.10, "XYZ": -0.10}, "turnover": {"ABC": 0.10, "XYZ": -0.10}}
    f1 = {"ABC": {"fill_px": 100.0, "mid": 99.9}, "XYZ": {"fill_px": 50.0, "mid": 50.05}}
    r1 = update_cycle(led, d1, f1, mids={"ABC": 99.9, "XYZ": 50.05}, close_ref={"ABC": 99.8, "XYZ": 50.1})
    # units: ABC +1000/100=+10 ; XYZ -1000/50=-20. realized=0 (opening). fees=2*1000*4.5/1e4=0.9
    assert abs(r1["realized_pnl"] - (-0.9)) < 1e-6, r1
    assert led["lots"]["ABC"] == [[10.0, 100.0]] and led["lots"]["XYZ"] == [[-20.0, 50.0]], led["lots"]
    # cycle 2: sleeve ages out -> target weights go to 0 (both exit fully, regardless of exit price)
    d2 = {"open_time": "T2", "regime": "side", "equity": 10000.0,
          "net_after": {}, "turnover": {"ABC": -0.10, "XYZ": 0.10}}
    f2 = {"ABC": {"fill_px": 102.0, "mid": 102.1}, "XYZ": {"fill_px": 49.0, "mid": 48.95}}
    r2 = update_cycle(led, d2, f2, mids={}, close_ref={})
    # ABC: close +10 @102 from 100 -> (102-100)*10=+20 ; XYZ: close -20 @49 from 50 -> (49-50)*-20=+20
    # fees cycle2 = (1000+1000)*4.5/1e4=0.9 ; realized2 = 40 - 0.9 = 39.1
    assert abs(r2["realized_pnl"] - 39.1) < 1e-6, r2
    assert not led["lots"], ("flat after round-trip", led["lots"])
    assert abs(r2["equity"] - (10000.0 - 0.9 + 39.1)) < 1e-6, r2
    # flip test: from flat, buy then oversell to flip short
    lots3 = []
    assert abs(_apply_trade(lots3, +10, 100.0)) < 1e-12                  # open long, realized 0
    rr = _apply_trade(lots3, -15, 110.0)                                 # close 10 @110 (+100), open -5
    assert abs(rr - 100.0) < 1e-9 and lots3 == [[-5.0, 110.0]], (rr, lots3)
    print("selftest OK: round-trip realized = +39.10 on $10k (ABC +20 / XYZ +20 − 1.8 fees), flip handled")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=str, help="convexity_v1 state dir (reads state/decision.json + decide_slip.csv)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.state:
        st = Path(a.state)
        dec = json.loads((st / "state" / "decision.json").read_text())
        led_path = st / "realfill" / "ledger.json"
        led = load_ledger(led_path)
        if led.get("last_open_time") == dec["open_time"]:
            print(f"[realfill] already booked {dec['open_time']} — skip"); raise SystemExit(0)
        fills = _read_decide_slip(st / "realfill" / "decide_slip.csv", dec["open_time"])
        cref_path = st / "decide" / "close_ref.json"
        close_ref = json.loads(cref_path.read_text()) if cref_path.exists() else {}
        # MtM of ALL open positions at the REAL HL ORDERBOOK mid — not just this cycle's traded legs. With 6
        # overlapping sleeves most open symbols are CARRIED (not in turnover); marking them at entry price left
        # their unrealized PnL at 0 and the equity wrong. Probe the whole open book (current lots ∪ this cycle's
        # target net_after) off HL L2; supplement with the fresh traded-leg mids for anything the snapshot missed.
        from live.convexity_slippage import snapshot_exec_marks
        net_after = {s: float(w) for s, w in dec.get("net_after", {}).items() if abs(float(w)) > 1e-9}
        lot_net = {s: sum(u for u, _ in led.get("lots", {}).get(s, [])) for s in led.get("lots", {})}
        net_dir = {**lot_net, **net_after}                          # every open position w/ direction (target wins)
        net_dir = {s: w for s, w in net_dir.items() if abs(w) > 1e-12}
        try:   # mark the whole open book at the REALIZABLE close price (long@bid, short@ask), not entry, not mid
            snapshot_exec_marks(net_dir, st / "decide" / "open_mids.json")
            mids = json.loads((st / "decide" / "open_mids.json").read_text()).get("mids", {})
        except Exception as e:
            print(f"[realfill] open-book HL exec-mark snapshot failed ({type(e).__name__}); using traded-leg mids only")
            mids = {}
        for s, f in fills.items():
            if s not in mids and np.isfinite(f.get("mid", np.nan)):
                mids[s] = f["mid"]
        # HL DECISION mids (captured at cycle start) → true HL→HL latency; absent → basis-contaminated fallback
        dmids_path = st / "decide" / "decision_mids.json"
        try:   # snapshot-mids is backgrounded + writes non-atomically; a mid-write read must not crash booking
            decision_mids = json.loads(dmids_path.read_text()).get("mids", {}) if dmids_path.exists() else {}
        except Exception:
            decision_mids = {}   # falls back to basis-contaminated latency (flagged), not a crash
        rec = update_cycle(led, dec, fills, mids=mids, close_ref=close_ref, decision_mids=decision_mids)
        save_ledger(led_path, led)
        print(f"[realfill] {rec['open_time']} [{rec['regime']}]: {rec['n_trades']} fills "
              f"({rec['n_unfilled']} unfilled) | realized {rec['realized_pnl']:+.2f} "
              f"unreal {rec['unrealized_pnl']:+.2f} eq ${rec['equity']:,.0f} | "
              f"exec cost {rec['exec_cost_bps']:.1f}bps (slip {rec['book_slip_bps']:.1f} + "
              f"lat {rec['latency_drift_bps']:.1f} + fee {rec['fee_bps']:.1f}) | basis {rec['basis_bps']:+.1f} (cancels)")
    else:
        ap.error("need --state or --selftest")
