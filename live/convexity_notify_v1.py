"""Per-cycle Telegram snapshot for the Convexity v1 LIVE forward test (single low-vol book + resid_rev).

Each 4h cycle reports: regime, the new legs entered (L/S), this cycle's PnL (modeled + net-of-real-slip),
the CURRENT portfolio (net position aggregated across the 6 overlapping sleeves), forward equity off a
$10k base, forward Sharpe, and realized HL slippage. Forward cycles only (after launch_marker.txt) so the
track record starts clean from go-live. Wired into run_convexity_v1_live.sh after the advance step.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import os
for envf in (REPO/".env", REPO/"live/.env"):                 # load TELEGRAM_BOT_TOKEN / CHAT_ID
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
from live.telegram import notify_telegram

OUT = REPO/"live/state"/os.environ.get("CONVEXITY_BOOK", "convexity_v1")
ST = OUT/"state"
BASE = 10000.0          # forward-test starting equity
MODELED = 4.5           # bps/leg modeled cost (for net-of-real-slip scaling)
ANN = np.sqrt(6 * 365)  # 6 cycles/day annualization
TRAIN_CUTOFF = "2026-05-29"   # frozen-model training cutoff → out-of-sample modeled benchmark line


def _sh(x):
    x = np.asarray(x, float) / 1e4
    return float(x.mean() / x.std() * ANN) if len(x) > 2 and x.std() > 0 else float("nan")


def _fwd():
    """Forward cycles only — those after launch_marker.txt (go-live anchor)."""
    c = pd.read_csv(ST / "cycles.csv"); c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    mk = OUT / "launch_marker.txt"
    if mk.exists():
        c = c[c["open_time"] > pd.to_datetime(mk.read_text().strip(), utc=True)]   # marker → UTC-aware
    return c.sort_values("open_time")


def _portfolio(top=8, hold=6):
    """Net position aggregated over all active sleeves (each weight / HOLD) → compact L/S holdings."""
    pj = ST / "positions.json"
    if not pj.exists():
        return ""
    d = json.load(open(pj)); net = {}
    for sl in d.get("active_sleeves", []):
        for s, w in sl.items():
            net[s] = net.get(s, 0.0) + float(w) / hold
    net = {s.replace("USDT", "").replace("__BTC_HEDGE__", "BTC.h"): w
           for s, w in net.items() if abs(w) > 1e-4}
    if not net:
        return "  FLAT (no open exposure)"
    longs = sorted([(s, w) for s, w in net.items() if w > 0], key=lambda x: -x[1])[:top]
    shorts = sorted([(s, w) for s, w in net.items() if w < 0], key=lambda x: x[1])[:top]
    fmt = lambda xs: " ".join(f"{s}{w*100:+.0f}" for s, w in xs)   # name + net % of equity
    gross = sum(abs(w) for w in net.values())
    return (f"  L: {fmt(longs)}\n  S: {fmt(shorts)}\n"
            f"  gross {gross*100:.0f}% • {len(net)} names")


def _realfill_summary():
    """CUMULATIVE real-fill P&L since launch + this-cycle exec cost. Equity = base + Σrealized + open MtM,
    split so the headline isn't confused by open unrealized swings. None if the ledger isn't running yet."""
    lp = OUT / "realfill" / "ledger.json"
    if not lp.exists():
        return None
    led = json.loads(lp.read_text()); cyc = led.get("cycles", [])
    if not cyc:
        return {"armed": True}
    e0 = float(led.get("equity0", BASE))
    cum_real = sum(float(c.get("realized_pnl", 0) or 0) for c in cyc)   # locked P&L over ALL cycles
    # perfect-fill = the SAME book/marks but zero execution cost: add back every cycle's exec cost ($) =
    # exec_cost_bps × traded_notional. This is the clean apples-to-apples "what if fills were perfect".
    cum_cost = sum((float(c.get("exec_cost_bps", 0) or 0) / 1e4) * float(c.get("traded_notional", 0) or 0) for c in cyc)
    r = cyc[-1]; eq = float(r["equity"]); open_un = float(r.get("unrealized_pnl", 0) or 0)
    n_real = sum(1 for c in cyc if "RESET" not in str(c.get("note", "")) and "RESTART" not in str(c.get("note", "")))
    return {"armed": False, "n": len(cyc), "n_real": n_real, "start": str(cyc[0].get("open_time"))[:16], "eq": eq, "ret": (eq / e0 - 1) * 100,
            "locked": cum_real / e0 * 100, "open": open_un / e0 * 100, "n_open": int(r.get("n_open_syms", 0)),
            "perfect_ret": ((eq + cum_cost) / e0 - 1) * 100, "drag": cum_cost / e0 * 100,
            "exec": r.get("exec_cost_bps"), "slip": r.get("book_slip_bps"),
            "lat": r.get("latency_drift_bps"), "fee": r.get("fee_bps"),
            "n_trades": int(r.get("n_trades", 0)), "n_unfilled": int(r.get("n_unfilled", 0))}


def _modeled_ret():
    """Cumulative modeled forward return % since launch (settled reference cycles). (None, 0) if none yet."""
    c = _fwd()
    if len(c) == 0:
        return None, 0
    return (float((1 + c["pnl_bps"].values / 1e4).prod()) - 1) * 100, len(c)


def _modeled_since(start: str):
    """Cumulative modeled (perfect-fill) return % from `start` to now, off cycles.csv — the out-of-sample
    benchmark for the frozen models since the training cutoff. (None, 0) if cycles.csv missing / empty."""
    p = ST / "cycles.csv"
    if not p.exists():
        return None, 0
    c = pd.read_csv(p); c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    c = c[c["open_time"] >= pd.to_datetime(start, utc=True)].sort_values("open_time")
    if len(c) == 0:
        return None, 0
    return (float((1 + c["pnl_bps"].values / 1e4).prod()) - 1) * 100, len(c)


def _exec_latency():
    """Real decision→fill wall-clock for the latest cycle: HL decision-mids snapshot (cycle start) → the HL
    fill probe. This IS the window the latency slippage is measured over."""
    try:
        dm = json.loads((OUT / "decide" / "decision_mids.json").read_text())
        t0 = pd.to_datetime(dm["captured_at"], utc=True)
        sl = pd.read_csv(OUT / "realfill" / "decide_slip.csv")
        last = sl["bar_open_time"].astype(str).iloc[-1]
        t1 = pd.to_datetime(sl[sl["bar_open_time"].astype(str) == last]["captured_at"], utc=True).max()
        return (t1 - t0).total_seconds()
    except Exception:
        return None


def _new_legs():
    """This cycle's new sleeve picks (longs/shorts) + bar/regime. Uses the decision's longs/shorts lists,
    NOT the turnover signs (turnover mixes new entries with closing/rolling old sleeves)."""
    dj = ST / "decision.json"
    if not dj.exists():
        return "—", "—", "FLAT", "FLAT"
    d = json.loads(dj.read_text())
    L = [s.replace("USDT", "") for s in d.get("longs", [])]
    S = [s.replace("USDT", "") for s in d.get("shorts", [])]
    return d.get("open_time", "—"), d.get("regime", "—"), ", ".join(L) or "FLAT", ", ".join(S) or "FLAT"


def build_message() -> str:
    """Simple snapshot: ONE cumulative P&L since launch (real-fill, marked now), split into locked vs open,
    plus the modeled reference as a single line, this cycle's new legs, and the execution cost."""
    rf = _realfill_summary()
    ot, regime, L, S = _new_legs()
    if rf is None or rf.get("armed"):
        return (f"⏳ <b>Convexity v2</b> (paper) — {ot} • {regime}\n"
                "Real-fill armed — no fills booked yet.")
    arrow = "🟢" if rf["ret"] >= 0 else "🔴"
    out = [f"{arrow} <b>Convexity v2</b> (paper, LIVE) — {ot} • {regime}",
           f"💰 <b>Real PnL since $10k restart ({rf['start'][:10]}): {rf['ret']:+.2f}%</b>  (equity ${rf['eq']:,.0f} / ${BASE/1e3:.0f}k · {rf['n_real']} real cycles)",
           f"   {rf['locked']:+.2f}% locked + {rf['open']:+.2f}% open ({rf['n_open']} positions still held)",
           f"✨ Perfect-fill (same trades, zero fee/slip/latency): <b>{rf['perfect_ret']:+.2f}%</b>  → execution cost {-rf['drag']:+.2f}%",
           f"New L: {L}  ·  S: {S}"]
    m529, n529 = _modeled_since(TRAIN_CUTOFF)
    if m529 is not None:
        out.append(f"📊 Modeled since 5.29 training (perfect-fill, OOS): <b>{m529:+.2f}%</b> · {n529} cycles")
    if rf["n_trades"]:
        unf = f", {rf['n_unfilled']} unfilled" if rf["n_unfilled"] else ""
        lat_s = _exec_latency()
        tstr = f" · {lat_s:.0f}s decision→fill" if lat_s is not None else ""
        out.append(f"⚙️ <b>execution</b>: {rf['n_trades']} filled{unf}{tstr}")
        out.append(f"   {rf['exec']:+.1f}bps = book-slip {rf['slip']:+.1f} + latency {rf['lat']:+.1f} + fee {rf['fee']:+.1f}")
    return "\n".join(out)


if __name__ == "__main__":
    msg = build_message()
    ok = notify_telegram(msg)
    print(("SENT" if ok else "FAILED") + ":\n" + msg)
    sys.exit(0 if ok else 1)   # non-zero on failure so the supervisor logs WARN, not a false "sent"
