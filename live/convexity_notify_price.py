"""4h Telegram snapshot for the convexity PRICE-book live forward test (single book).

Reads live/state/convexity_bookB/{cycles.csv, positions.json, slippage.csv} and sends a compact
snapshot to Telegram: latest cycle, equity (10k base), regime, positions, realized HL slippage,
trailing-30 forward Sharpe + simple kill-switch flags. Wired into run_convexity_live.sh per cycle.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))

# load creds from .env / live/.env so TELEGRAM_BOT_TOKEN/CHAT_ID are available
import os
for envf in (REPO/".env", REPO/"live/.env"):
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
from live.telegram import notify_telegram

ST = REPO/"live/state/convexity_bookB"
ANN = np.sqrt(6 * 365)   # 6 cycles/day, annualize


def _sh(x):
    x = np.asarray(x, float) / 1e4
    return float(x.mean()/x.std()*ANN) if len(x) > 2 and x.std() > 0 else float("nan")


def build_message() -> str:
    c = pd.read_csv(ST/"cycles.csv")
    c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    eq = json.load(open(ST/"positions.json")).get("equity", float("nan")) if (ST/"positions.json").exists() else float("nan")
    # split warm-up (bootstrap) from LIVE forward cycles via the launch marker
    mk = ST/"launch_marker.txt"
    marker = pd.Timestamp(mk.read_text().strip()) if mk.exists() else None
    fwd = c[c["open_time"] > marker] if marker is not None else c
    if len(fwd) == 0:   # nothing live yet — warm-up only
        # first forward DECISION bar = marker+4h; it SETTLES (scored) 4h later = marker+8h
        bar = (marker + pd.Timedelta(hours=4)) if marker is not None else None
        settle = (marker + pd.Timedelta(hours=8)) if marker is not None else "next settle"
        return ("⏳ <b>Convexity price-book</b> — WARM-UP (NOT live yet)\n"
                f"Equity ${eq:,.0f} base • 6 sleeves seeded from OOS bootstrap\n"
                f"First LIVE cycle = the {bar} bar, reports ~{settle} UTC once its 4h return settles.\n"
                "PnL/slippage report on live cycles only.")
    row = fwd.iloc[-1]
    longs = str(row.get("top_k_long", "") or ""); shorts = str(row.get("bot_k_short", "") or "")
    regime = row.get("regime", "?"); pnl = float(row.get("pnl_bps", float("nan")))
    cost = float(row.get("cost_bps", float("nan"))); nuniv = row.get("n_universe", "?")
    gross = float(row.get("gross_pnl_bps", float("nan")))
    stop_on = str(row.get("stop_engaged", "")).strip() in ("True", "1", "1.0")
    tail = fwd["pnl_bps"].values[-30:]; n_fwd = len(fwd)

    # realized HL-L2 slippage for the latest cycle → scale the bot's modeled cost (4.5bps/leg) by the
    # measured/modeled ratio to get the realized per-cycle drag, then net it off the gross PnL.
    MODELED_LEG = 4.5; slip = "n/a"; pnl_real = pnl
    sf = ST/"slippage.csv"
    if sf.exists():
        s = pd.read_csv(sf); s = s[s["open_time"] == s["open_time"].iloc[-1]]
        tc = pd.to_numeric(s["total_cost_bps"], errors="coerce").dropna()
        nf = (s["fully_filled"].astype(str) != "True").sum()
        if len(tc):
            slip = f"med {tc.median():.0f}bps p90 {np.percentile(tc,90):.0f} • unfilled {nf}/{len(s)}"
            cost_real = cost * (tc.median() / MODELED_LEG)        # realized per-cycle cost (bps)
            if np.isfinite(gross): pnl_real = gross - cost_real    # gross net of REALIZED slippage

    alerts = []
    if _sh(tail) < 0 and len(tail) >= 10: alerts.append("trail-30 Sharpe < 0")
    if stop_on: alerts.append("vol-stop engaged")
    flag = "🟢" if not alerts else "🔴"

    L = [f"{flag} <b>Convexity price-book</b> (paper, LIVE) — {row['open_time']}",
         f"Equity ${eq:,.0f} • fwd cycle {n_fwd} • univ {nuniv}",
         f"Cycle PnL {pnl:+.0f} bps modeled / {pnl_real:+.0f} net-of-real-slip • {regime}",
         f"Fwd Sharpe (≤30) {_sh(tail):+.2f}",
         f"Realized slip: {slip}",
         f"L: {longs[:90]}", f"S: {shorts[:90]}"]
    if alerts: L.append("⚠️ " + "; ".join(alerts))
    return "\n".join(L)


if __name__ == "__main__":
    msg = build_message()
    ok = notify_telegram(msg)
    print(("SENT" if ok else "FAILED (check TG creds)") + ":\n" + msg)
