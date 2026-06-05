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


def _sh(x):
    x = np.asarray(x, float) / 1e4
    return float(x.mean() / x.std() * ANN) if len(x) > 2 and x.std() > 0 else float("nan")


def _fwd():
    """Forward cycles only — those after launch_marker.txt (go-live anchor)."""
    c = pd.read_csv(ST / "cycles.csv"); c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    mk = OUT / "launch_marker.txt"
    if mk.exists():
        c = c[c["open_time"] > pd.Timestamp(mk.read_text().strip())]
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


def _realfill_block():
    """Real-fill (HL round-trip) summary from the ledger: equity, last-cycle realized/unrealized, and the
    execution-cost decomposition (book slip + latency drift + fee). None if the ledger isn't running yet."""
    lp = OUT / "realfill" / "ledger.json"
    if not lp.exists():
        return None
    led = json.loads(lp.read_text())
    cyc = led.get("cycles", [])
    e0 = float(led.get("equity0", BASE))
    if not cyc:
        return "💵 <b>Real-fill</b> (HL execution): armed — no fills yet"
    r = cyc[-1]
    eq = float(r["equity"])
    out = [f"💵 <b>Real-fill</b> (HL round-trip) — eq ${eq:,.0f} ({(eq/e0-1)*100:+.1f}%) • open {r['n_open_syms']} syms",
           f"  last cycle: realized {r['realized_pnl']:+.2f} • unreal {r['unrealized_pnl']:+.2f}"]
    if r.get("n_trades", 0):
        out.append(f"  exec cost {r['exec_cost_bps']:.1f}bps = slip {r['book_slip_bps']:.1f} "
                   f"+ lat {r['latency_drift_bps']:.1f} + fee {r['fee_bps']:.1f} ({r['n_trades']} legs"
                   + (f", {r['n_unfilled']} unfilled" if r.get('n_unfilled') else "") + ")")
    return "\n".join(out)


def build_message() -> str:
    c = _fwd()
    if len(c) == 0:
        mk = OUT / "launch_marker.txt"
        settle = (pd.Timestamp(mk.read_text().strip()) + pd.Timedelta(hours=8)) if mk.exists() else "next settle"
        return ("⏳ <b>Convexity v1</b> (low-vol + resid_rev) — WARM-UP (not live yet)\n"
                "$10k base • K=3 long/short • 6 overlapping sleeves • regime-gated\n"
                f"First LIVE snapshot ~{settle} UTC once the 4h return settles.")
    eq = BASE * (1 + c["pnl_bps"].values / 1e4).cumprod()
    r = c.iloc[-1]
    # net-of-real-slip: scale this cycle's modeled cost by measured/modeled HL slippage
    slip = "n/a"; net_pnl = float(r["pnl_bps"]); sf = OUT / "slippage.csv"
    if sf.exists() and os.path.getsize(sf) > 0:
        s = pd.read_csv(sf)
        if len(s):
            s = s[s["open_time"] == s["open_time"].iloc[-1]]
            tc = pd.to_numeric(s["total_cost_bps"], errors="coerce").dropna()
            nf = (s["fully_filled"].astype(str) != "True").sum()
            if len(tc):
                slip = f"med {tc.median():.0f}bps p90 {np.percentile(tc,90):.0f} • unfilled {nf}/{len(s)}"
                g, cost = float(r.get("gross_pnl_bps", r["pnl_bps"])), float(r.get("cost_bps", 0))
                net_pnl = g - cost * (tc.median() / MODELED)
    sh_all = _sh(c["pnl_bps"].values)        # realtime: since-launch forward Sharpe (all fwd cycles)
    sh_30 = _sh(c["pnl_bps"].values[-30:])   # recent form: trailing ≤30 cycles (noisier)
    arrow = "🟢" if float(r["pnl_bps"]) >= 0 else "🔴"
    L_str, S_str = str(r.get("top_k_long", "")), str(r.get("bot_k_short", ""))
    flat = L_str in ("nan", "") and S_str in ("nan", "")
    out = [f"{arrow} <b>Convexity v1</b> (paper, LIVE) — {r['open_time']} • {r['regime']}"]
    rfb = _realfill_block()
    if rfb:
        out += [rfb, "— modeled reference —"]
    out += [
        f"Equity ${eq[-1]:,.0f} • fwd cycle {len(c)}",
        f"Sharpe: realtime {sh_all:+.2f} (n={len(c)}) • trailing-30 {sh_30:+.2f}",
        f"Cycle PnL {float(r['pnl_bps']):+.0f}bps modeled / {net_pnl:+.0f} net-of-slip",
        f"  gross {float(r.get('gross_pnl_bps',0)):+.0f} − cost {float(r.get('cost_bps',0)):.0f}"
        f" • turnover {float(r.get('turnover',0))*100:.0f}%",
    ]
    if flat:
        out.append("New entries: FLAT (regime gate — no positions)")
    else:
        out += [f"New L: {L_str[:60]}", f"New S: {S_str[:60]}"]
    out += ["<b>Portfolio</b> (net across 6 sleeves):", _portfolio(), f"Slip: {slip}"]
    return "\n".join([x for x in out if x])


if __name__ == "__main__":
    msg = build_message()
    ok = notify_telegram(msg)
    print(("SENT" if ok else "FAILED") + ":\n" + msg)
    sys.exit(0 if ok else 1)   # non-zero on failure so the supervisor logs WARN, not a false "sent"
