"""4h Telegram snapshot for the convexity TWO-BOOK champion live forward test.

Combines BookA(flow)+BookB(price) 50/50 at the PnL level, reports combined equity ($100k base),
per-book latest cycle + positions, realized HL slippage, forward Sharpe. Warm-up aware via each
book's launch_marker.txt (forward cycles only). Wired into run_convexity_twobook_live.sh.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import os
for envf in (REPO/".env", REPO/"live/.env"):
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
from live.telegram import notify_telegram

STA = REPO/"live/state/convexity_bookA"; STB = REPO/"live/state/convexity_bookB"
OUT = REPO/"live/state/convexity_twobook"
ANN = np.sqrt(6 * 365)


def _sh(x):
    x = np.asarray(x, float)/1e4
    return float(x.mean()/x.std()*ANN) if len(x) > 2 and x.std() > 0 else float("nan")


def _fwd(st):
    c = pd.read_csv(st/"cycles.csv"); c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    mk = st/"launch_marker.txt"
    if mk.exists():
        c = c[c["open_time"] > pd.Timestamp(mk.read_text().strip())]
    return c


def _portfolio(st, top=6, hold=6):
    """Aggregate net position (all active sleeves, weight/HOLD) → compact long/short holdings string."""
    pj = st/"positions.json"
    if not pj.exists():
        return ""
    d = json.load(open(pj)); net = {}
    for sl in d.get("active_sleeves", []):
        for s, w in sl.items():
            net[s] = net.get(s, 0.0) + float(w)/hold
    net = {s.replace("USDT", "").replace("__BTC_HEDGE__", "BTC.h"): w
           for s, w in net.items() if abs(w) > 1e-4}
    longs = sorted([(s, w) for s, w in net.items() if w > 0], key=lambda x: -x[1])[:top]
    shorts = sorted([(s, w) for s, w in net.items() if w < 0], key=lambda x: x[1])[:top]
    fmt = lambda xs: " ".join(f"{s}{w*100:+.0f}" for s, w in xs)   # name+net% (×100 of equity weight)
    return f"  hold L: {fmt(longs)}\n  hold S: {fmt(shorts)}"


def build_message() -> str:
    A, B = _fwd(STA), _fwd(STB)
    if len(A) == 0 and len(B) == 0:
        mk = STA/"launch_marker.txt"
        settle = (pd.Timestamp(mk.read_text().strip()) + pd.Timedelta(hours=8)) if mk.exists() else "next settle"
        return ("⏳ <b>Convexity TWO-BOOK</b> — WARM-UP (NOT live yet)\n"
                "$100k base • BookA(flow,80) + BookB(price,94), 50/50 • 6 sleeves each\n"
                f"First LIVE cycle reports ~{settle} UTC once the 4h return settles.")
    # 50/50 combined forward equity off $100k
    m = A[["open_time", "pnl_bps"]].merge(B[["open_time", "pnl_bps"]], on="open_time",
                                          suffixes=("_a", "_b"), how="outer").sort_values("open_time")
    comb = 0.5*m["pnl_bps_a"].fillna(0) + 0.5*m["pnl_bps_b"].fillna(0)
    eq = 100000.0 * (1 + comb.values/1e4).cumprod()
    ra, rb = (A.iloc[-1] if len(A) else None), (B.iloc[-1] if len(B) else None)
    # realized slip per book → scale each book's modeled cost (4.5/leg) by measured/modeled, net off gross,
    # then 50/50-combine for the honest net-of-real-slip combined cycle PnL.
    MODELED = 4.5; slip = "n/a"; comb_net = comb.iloc[-1]; sf = OUT/"slippage.csv"
    if sf.exists():
        s = pd.read_csv(sf); s = s[s["open_time"] == s["open_time"].iloc[-1]]
        tc = pd.to_numeric(s["total_cost_bps"], errors="coerce").dropna()
        nf = (s["fully_filled"].astype(str) != "True").sum()
        if len(tc): slip = f"med {tc.median():.0f}bps p90 {np.percentile(tc,90):.0f} • unfilled {nf}/{len(s)}"
        def _net(row, book):
            if row is None: return 0.0
            sb = pd.to_numeric(s[s["book"] == book]["total_cost_bps"], errors="coerce").dropna()
            med = sb.median() if len(sb) else (tc.median() if len(tc) else MODELED)
            g, c = float(row.get("gross_pnl_bps", row.get("pnl_bps", 0))), float(row.get("cost_bps", 0))
            return g - c * (med / MODELED)
        comb_net = 0.5*_net(ra, "A") + 0.5*_net(rb, "B")
    shA, shB = _sh(A["pnl_bps"].values[-30:]), _sh(B["pnl_bps"].values[-30:])
    L = [f"📊 <b>Convexity TWO-BOOK</b> (paper, LIVE) — {m['open_time'].iloc[-1]}",
         f"Equity ${eq[-1]:,.0f} • fwd cycles {len(comb)}",
         f"Combined PnL {comb.iloc[-1]:+.0f}bps modeled / {comb_net:+.0f} net-of-real-slip",
         f"Fwd Sharpe (≤30): flow {shA:+.2f} / price {shB:+.2f}",
         f"Realized slip: {slip}"]
    if ra is not None:
        L.append(f"<b>BookA(flow)</b> {ra['pnl_bps']:+.0f}bps • new L:{str(ra.get('top_k_long',''))[:55]} S:{str(ra.get('bot_k_short',''))[:55]}")
        L.append(_portfolio(STA))
    if rb is not None:
        L.append(f"<b>BookB(price)</b> {rb['pnl_bps']:+.0f}bps • new L:{str(rb.get('top_k_long',''))[:55]} S:{str(rb.get('bot_k_short',''))[:55]}")
        L.append(_portfolio(STB))
    return "\n".join([x for x in L if x])


if __name__ == "__main__":
    msg = build_message()
    ok = notify_telegram(msg)
    print(("SENT" if ok else "FAILED") + ":\n" + msg)
