"""4h Telegram status/alert heartbeat for the convexity two-book forward test.

Reads the live state (twobook_summary.json, per-book cycles.csv, slippage.csv), runs the kill-switch
checks, and sends a compact status message to Telegram. Substantive new-cycle PnL lands daily (the
data advance is a daily batch — X132 panel rebuild ~2hr); this heartbeat gives 4h visibility + alerts.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))

# load .env so TELEGRAM_BOT_TOKEN/CHAT_ID are available
envf = REPO / ".env"
if envf.exists():
    import os
    for line in envf.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
from live.telegram import notify_telegram

OUT = REPO / "live/state/convexity_twobook"
ANN = np.sqrt(6 * 365)


def _sh(x):
    x = np.asarray(x, float) / 1e4
    return float(x.mean() / x.std() * ANN) if len(x) > 2 and x.std() > 0 else float("nan")


def build_message() -> str:
    L = []
    summ = json.loads((OUT / "twobook_summary.json").read_text()) if (OUT / "twobook_summary.json").exists() else {}
    A = pd.read_csv(REPO / "live/state/convexity_bookA/cycles.csv")
    B = pd.read_csv(REPO / "live/state/convexity_bookB/cycles.csv")
    last_ot = A["open_time"].iloc[-1]
    # combined equity curve (50/50 both-active)
    m = (A[["open_time", "pnl_bps"]].merge(B[["open_time", "pnl_bps"]], on="open_time",
         suffixes=("_a", "_b"), how="inner").sort_values("open_time"))
    comb = 0.5 * m["pnl_bps_a"] + 0.5 * m["pnl_bps_b"]
    eq = 10000.0 * (1 + comb.values / 1e4).cumprod()
    tail30 = comb.values[-30:]
    # latest positions
    row = A.iloc[-1]
    longs = str(row.get("top_k_long", "") or ""); shorts = str(row.get("bot_k_short", "") or "")
    regime = row.get("regime", "?")
    # slippage (latest cycle)
    slip_line = "n/a"
    sf = OUT / "slippage.csv"
    if sf.exists():
        s = pd.read_csv(sf)
        s = s[s["open_time"] == s["open_time"].iloc[-1]]
        tc = pd.to_numeric(s["total_cost_bps"], errors="coerce").dropna()
        nf = (s["fully_filled"].astype(str) != "True").sum()
        if len(tc):
            slip_line = f"med {tc.median():.0f}bps, p90 {np.percentile(tc,90):.0f}, not-filled {nf}/{len(s)}"
    # kill-switch checks
    alerts = []
    if _sh(tail30) < 0: alerts.append("trailing-30 Sharpe below 0")
    fwd = comb.values[-60:]  # forward window only (not bootstrap history)
    cum = pd.Series(fwd).cumsum(); dd = (cum - cum.cummax()).min()
    if dd < -2500: alerts.append(f"maxDD {dd:.0f} below -2500")
    if summ.get("book_pnl_corr", 0) > 0.5: alerts.append(f"book corr {summ['book_pnl_corr']:.2f} over 0.5")
    flag = "🟢" if not alerts else "🔴"
    L.append(f"{flag} <b>Convexity 2-book</b> (paper) — {last_ot}")
    L.append(f"Equity: ${eq[-1]:,.0f}  •  cycles {len(A)}")
    L.append(f"Sharpe: full {summ.get('sharpe_both_active', float('nan')):+.2f} | trail-30 {_sh(tail30):+.2f}")
    L.append(f"Books: flow {summ.get('sharpe_bookA', float('nan')):+.2f} / price {summ.get('sharpe_bookB', float('nan')):+.2f} (corr {summ.get('book_pnl_corr', float('nan')):.2f})")
    L.append(f"Regime: {regime}  •  realized slip: {slip_line}")
    L.append(f"L: {longs[:80]}\nS: {shorts[:80]}")
    if alerts: L.append("⚠️ KILL-SWITCH: " + "; ".join(alerts))
    return "\n".join(L)


if __name__ == "__main__":
    msg = build_message()
    ok = notify_telegram(msg)
    print(("SENT" if ok else "FAILED (check creds)") + ":\n" + msg)
