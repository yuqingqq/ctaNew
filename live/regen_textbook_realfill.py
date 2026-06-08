"""Reset the real-fill ledger to mirror the CLEAN textbook (modeled) book.

Modes:
  (default) CONTINUITY — equity0=$10k, the modeled cumulative pnl carried as locked realized_cum (keeps the
            launch record, e.g. -3.58%).
  --fresh   $10k RESTART — equity0=$10k, realized_cum=0; discards prior pnl for a clean $10k forward test.

State-driven (no hardcoded dates): seeds the modeled net book (positions.prev_agg) priced at the modeled book's
own boundary (positions.last_open_time), with the new ledger.last_open_time = the real-fill ledger's CURRENT
boundary, so the NEXT 4h cycle is the first one booked on the reset book. Entry=mark -> unrealized 0 at the seed.

Run: .venv/bin/python live/regen_textbook_realfill.py [--fresh] [--apply]   (dry-run without --apply)
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
from live.incremental_xs_feats import CACHE

OUT = REPO / "live/state/convexity_v2"
BASE = 10000.0


def modeled_equity() -> float:
    c = pd.read_csv(OUT / "state/cycles.csv"); c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    mk = pd.to_datetime((OUT / "launch_marker.txt").read_text().strip(), utc=True)
    fwd = c[c["open_time"] > mk]
    return float(BASE * (1 + fwd["pnl_bps"].values / 1e4).prod())


def mark_at(sym, when) -> float:
    p = CACHE / f"xs_feats_{sym}.parquet"
    if not p.exists():
        return float("nan")
    cc = pd.read_parquet(p, columns=["close"]); cc.index = pd.to_datetime(cc.index, utc=True)
    return float(cc["close"].asof(when))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="$10k clean restart (realized_cum=0); else continuity")
    a = ap.parse_args()

    pos = json.load(open(OUT / "state/positions.json"))
    led_cur = json.load(open(OUT / "realfill/ledger.json"))
    bk_boundary = pd.Timestamp(pos["last_open_time"])           # the modeled book's "as of" boundary
    switch = str(led_cur.get("last_open_time"))                 # next cycle after this = first booked on reset
    net = {s: float(w) for s, w in pos["prev_agg"].items() if abs(float(w)) > 1e-9}
    mod_eq = modeled_equity()
    eq = BASE if a.fresh else mod_eq                            # sizing + headline equity
    realized_cum = 0.0 if a.fresh else round(mod_eq - BASE, 4)
    try:
        regime = json.load(open(OUT / "state/decision.json")).get("regime", "—")
    except Exception:
        regime = "—"

    lots, gross, n_open, missing = {}, 0.0, 0, []
    for s, w in net.items():
        px = mark_at(s, bk_boundary)
        if not np.isfinite(px) or px <= 0:
            missing.append(s); continue
        units = w * eq / px                                    # net notional weight of equity, at the book mark
        lots[s] = [[units, px]]                                # entry = mark -> unrealized 0 at the seed
        gross += abs(units * px); n_open += 1
    if missing:
        print(f"!! missing marks @ {bk_boundary} for {missing} — ABORT (would drop names)"); sys.exit(1)

    seed = {
        "open_time": switch, "regime": regime, "n_trades": 0, "n_unfilled": 0,
        "realized_pnl": realized_cum, "realized_cum": realized_cum, "unrealized_pnl": 0.0,
        "equity": round(eq, 2), "gross_exposure": round(gross, 2), "n_open_syms": n_open,
        "fee_bps": 0.0, "book_slip_bps": 0.0, "latency_drift_bps": 0.0, "basis_bps": 0.0,
        "exec_cost_bps": 0.0, "traded_notional": 0.0,
        "note": f"{'FRESH $10k' if a.fresh else 'CONTINUITY'} RESET — mirrors modeled book @ {bk_boundary}; "
                f"next cycle after {switch} is the first booked on the reset book",
    }
    led = {"equity0": BASE, "realized_cum": realized_cum, "lots": lots, "last_open_time": switch, "cycles": [seed]}

    mode = "FRESH $10k" if a.fresh else "CONTINUITY"
    print(f"=== {mode} RESET (dry-run{'' if a.apply else ' — pass --apply to write'}) ===")
    print(f"  book: {n_open} names, gross ${gross:,.0f} ({gross/eq*100:.0f}%) priced @ {bk_boundary}")
    print(f"  equity ${eq:,.2f} | equity0 ${BASE:,.0f} | realized_cum {realized_cum} | new last_open_time {switch}")
    print(f"  modeled book boundary: {bk_boundary}   (must be current — else stale positions)")
    print(f"  BEFORE: {len(led_cur['cycles'])} cycles, last {led_cur.get('last_open_time')}, "
          f"eq ${led_cur['cycles'][-1].get('equity',0):,.0f}")
    if a.apply:
        bak = OUT / "realfill" / f"ledger.pre_{'fresh' if a.fresh else 'continuity'}_reset.json"
        shutil.copy(OUT / "realfill/ledger.json", bak)
        (OUT / "realfill/ledger.json").write_text(json.dumps(led, indent=2))
        print(f"  APPLIED. backup -> {bak}")
    else:
        print("  (no write — dry-run)")


if __name__ == "__main__":
    main()
