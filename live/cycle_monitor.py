"""Per-cycle processing monitor for the convexity v1 real-time flow.

Reviews every completed cycle and flags problems automatically: STALE/wrong decide-bar (decided ≠ boundary),
missing/failed steps, funding fallback, HL-probe fill rate, trigger source + latency. Sends a concise review
to Telegram each cycle and a detailed line to monitor.log. Also alerts if a 4h boundary passes UNprocessed
(collector + fallback both missed). Read-only — never touches the live state.

Usage:
  python3 live/cycle_monitor.py            # review the latest completed cycle once (print)
  python3 live/cycle_monitor.py --watch    # loop: review each new cycle + miss alerts → Telegram
"""
from __future__ import annotations
import os, sys, time, json, re
from pathlib import Path
import pandas as pd
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
for envf in (REPO/".env", REPO/"live/.env"):
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
from live.telegram import notify_telegram

OUT = REPO/"live/state"/os.environ.get("CONVEXITY_BOOK", "convexity_v1")
RTLOG = OUT/"realtime.log"
MONLOG = OUT/"monitor.log"


def _completed_blocks(text: str):
    """Split realtime.log into per-cycle blocks; return only those that reached 'done ==='."""
    blocks, cur = [], None
    for ln in text.splitlines():
        if "cycle start ===" in ln:
            cur = [ln]
        elif cur is not None:
            cur.append(ln)
            if "done ===" in ln:
                blocks.append(cur); cur = None
    return blocks


def review_block(block: list) -> dict:
    text = "\n".join(block)
    def f(pat, grp=1, d=None):
        m = re.search(pat, text); return m.group(grp) if m else d
    src   = f(r"\[cycle/(\w+)\]")
    B     = f(r"boundary (\S+ \S+): cycle start")
    start = f(r"\[([\d\- :]+)\] \[cycle/\w+\] === boundary \S+ \S+: cycle start")
    dec   = f(r"decided: (\S+ \S+) \w+ - \d+ legs")
    reg   = f(r"decided: \S+ \S+ (\w+) - \d+ legs")
    legs  = f(r"decided: \S+ \S+ \w+ - (\d+) legs")
    fresh = (B is not None and dec is not None and B == dec)
    lat = None
    if B and start:
        try: lat = (pd.Timestamp(start, tz="UTC") - pd.Timestamp(B)).total_seconds()
        except Exception: pass
    issues = []
    if dec and not fresh: issues.append(f"STALE: decided {dec} ≠ boundary {B}")
    if "funding STALE" in text: issues.append("funding stale→FAPI")
    # data-health: residual gaps FAPI couldn't fill + degraded-feed decision aborts
    bf = re.search(r"backfilled, \d+ gapless, (\d+) still-gappy, (\d+) errors", text)
    if bf and (int(bf.group(1)) > 0 or int(bf.group(2)) > 0):
        issues.append(f"feed gaps: {bf.group(1)} still-gappy / {bf.group(2)} FAPI-err after backfill")
    if "DEGRADED FEED" in text: issues.append("⛔ DEGRADED feed — decision ABORTED (no trade)")
    cm = re.search(r"xs-rank cohort (\d+) < 174", text)
    if cm: issues.append(f"xs-rank cohort {cm.group(1)}/174 — peer klines stale, rank drifting")
    if "on ffill-patched bars" in text: issues.append("some picks on ffill-patched bars")
    if "FAIL" in text: issues.append("step FAILed")
    if "settle WARN" in text: issues.append("settle warn")
    if "ledger WARN" in text: issues.append("ledger warn")
    if re.search(r"tg (snapshot )?WARN", text): issues.append("snapshot warn")
    if "HL probe WARN" in text: issues.append("HL probe warn")
    # backfill perf — should be a fast no-op now the WS streams the full universe (≈4s/0 fetched). If it
    # creeps back up, the collector is dropping klines again (the 350-stream overload regression) and the
    # decision→fill latency balloons.
    bfp = re.search(r"backfill_klines_gaps\].*?(\d+) gap-filled, (\d+) stale-refreshed.*?\[(\d+)s\]", text)
    bf_s = int(bfp.group(3)) if bfp else None
    bf_fetched = (int(bfp.group(1)) + int(bfp.group(2))) if bfp else None
    if bf_s is not None and (bf_s > 25 or (bf_fetched or 0) > 40):
        issues.append(f"⏱ backfill {bf_s}s / {bf_fetched} fetched — WS dropping klines? (healthy ≈ 4s/0)")
    # WS disk-readiness: "N gapless" = symbols whose boundary bar the collector WROTE without needing FAPI.
    # Low gapless = the collector orphan-race (dirty_kl.clear dropping bars added mid-flush-write) regressed.
    # Healthy ≈ 170-175 since the difference_update fix; <168 means boundary klines are going un-written again.
    gp = re.search(r"(\d+) gapless", text)
    gapless = int(gp.group(1)) if gp else None
    if gapless is not None and gapless < 168:
        issues.append(f"⚠️ WS disk-readiness {gapless} gapless (<168) — collector orphan-race regression?")
    return {"src": src, "B": B, "decided": dec, "regime": reg, "legs": legs, "gapless": gapless,
            "fresh": fresh, "lat": lat, "bf_s": bf_s, "bf_fetched": bf_fetched, "issues": issues}


def _hl_fills(boundary) -> tuple:
    p = OUT/"realfill"/"decide_slip.csv"
    if not p.exists() or boundary is None: return (0, 0)
    try:
        s = pd.read_csv(p); s = s[s["bar_open_time"].astype(str) == str(boundary)]
        filled = int(s["fill_px"].apply(lambda x: str(x) not in ("", "nan")).sum())
        return (filled, len(s))
    except Exception:
        return (0, 0)


def _exec_time(boundary):
    """decision-mids snapshot (cycle start) → HL fill probe wall-clock for this boundary — the real
    decision→fill latency window. None if not available for this cycle."""
    try:
        dm = json.loads((OUT/"decide"/"decision_mids.json").read_text())
        t0 = pd.to_datetime(dm["captured_at"], utc=True)
        s = pd.read_csv(OUT/"realfill"/"decide_slip.csv")
        s = s[s["bar_open_time"].astype(str) == str(boundary)]
        if not len(s): return None
        return (pd.to_datetime(s["captured_at"], utc=True).max() - t0).total_seconds()
    except Exception:
        return None


def build_message(r: dict) -> str:
    ok = r["fresh"] and not r["issues"]
    latstr = f"+{int(r['lat']//60)}m{int(r['lat']%60):02d}s" if r["lat"] is not None else "?"
    out = [f"{'✅' if ok else '⚠️'} <b>Cycle review</b> — boundary {r['B']}",
           f"trigger {r['src']} • fired {latstr} after boundary",
           f"decide-bar {r['decided']} {'(=boundary, fresh ✓)' if r['fresh'] else '(STALE ✗)'}",
           f"regime {r['regime']} • {r['legs']} legs"]
    f, n = _hl_fills(r["B"])
    if n: out.append(f"HL fills {f}/{n}")
    if r.get("bf_s") is not None:
        out.append(f"⏱ backfill {r['bf_s']}s ({r['bf_fetched']} fetched) {'✓' if r['bf_s'] <= 25 else '⚠️ slow'}")
    et = _exec_time(r["B"])
    if et is not None:
        out.append(f"⚙️ execution {et:.0f}s decision→fill")
    out.append("issues: " + "; ".join(r["issues"]) if r["issues"] else "all steps OK ✓")
    return "\n".join(out)


def latest_review():
    if not RTLOG.exists(): return None
    blocks = _completed_blocks(RTLOG.read_text())
    if not blocks: return None
    r = review_block(blocks[-1])
    return r, build_message(r)


def _booked_boundary():
    p = OUT/"realfill"/"ledger.json"
    if not p.exists(): return None
    try: return json.loads(p.read_text()).get("last_open_time")
    except Exception: return None


def watch(poll=60):
    res = latest_review()                                   # anchor to the cycle already done at launch so
    last_B = res[0]["B"] if res else None                  # the monitor reviews only NEW cycles from here on
    missed = set()
    notify_telegram(f"🔭 cycle monitor started — reviewing convexity v1 each cycle (last done: {last_B})")
    while True:
        try:
            res = latest_review()
            if res:
                r, msg = res
                if r["B"] and r["B"] != last_B:
                    notify_telegram(msg)
                    MONLOG.parent.mkdir(parents=True, exist_ok=True)
                    with open(MONLOG, "a") as fh: fh.write(f"[{pd.Timestamp.utcnow()}]\n{msg}\n\n")
                    last_B = r["B"]
            # miss alert: a boundary that should be done by now (boundary+10min) but isn't booked
            exp = (pd.Timestamp.utcnow() - pd.Timedelta(minutes=10)).floor("4h")
            booked = _booked_boundary()
            bk = pd.Timestamp(booked) if booked else None
            if (bk is None or exp > bk) and exp not in missed:
                notify_telegram(f"⚠️ <b>Cycle monitor</b>: boundary {exp} not processed "
                                f"(last booked {booked}) — check collector trigger / fallback")
                missed.add(exp)
        except Exception as e:
            print("[monitor] err", e, flush=True)
        time.sleep(poll)


if __name__ == "__main__":
    if "--watch" in sys.argv:
        watch()
    else:
        res = latest_review()
        print(res[1] if res else "no completed cycle yet")
