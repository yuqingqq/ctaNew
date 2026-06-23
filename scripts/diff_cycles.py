"""Cross-reference a reproduced cycles.csv against the live convexity_v2 baseline.

Run on the backtest server AFTER reproducing the modeled track, to confirm the
environment matches the live run before trusting any experiment delta. Reports
per-field parity vs the protocol tolerances + the first divergent cycle, and the
cumulative-OOS match. Exit 0 = PASS (within tolerance), 1 = DIVERGENT.

Usage:
  python scripts/diff_cycles.py --candidate path/to/your/cycles.csv
      [--baseline data/live_export/convexity_v2_cycles_through_2026-06-21.csv]
      [--oos-start 2026-05-29]
"""
from __future__ import annotations
import argparse, sys
import numpy as np, pandas as pd

EXACT_CAT = ["regime", "stop_engaged"]
PICK_COLS = ["top_k_long", "bot_k_short"]
ALPHA_COLS = ["long_ret_bps", "short_ret_bps", "long_alpha_bps", "short_alpha_bps"]
NUM_TOL = {  # field -> abs tolerance (see diff_protocol.md §4)
    "pnl_bps": 0.5, "gross_pnl_bps": 0.5, "cost_bps": 0.1, "turnover": 1e-4,
    "long_ret_bps": 1.0, "short_ret_bps": 1.0, "long_alpha_bps": 1.0, "short_alpha_bps": 1.0,
    "gross_after_stop": 1e-4, "net_target": 1e-4, "pred_disp": 1e-3, "btc_ret_30d": 1e-6,
}


def _load(p):
    d = pd.read_csv(p); d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    return d.drop_duplicates("open_time").sort_values("open_time")


def _pset(s):
    return frozenset(x for x in str(s).split(",") if x and x != "nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", default="data/live_export/convexity_v2_cycles_through_2026-06-21.csv")
    ap.add_argument("--oos-start", default="2026-05-29")
    a = ap.parse_args()
    oos = pd.Timestamp(a.oos_start, tz="UTC")
    b = _load(a.baseline); b = b[b.open_time >= oos]
    c = _load(a.candidate)
    m = b.merge(c, on="open_time", suffixes=("_b", "_c"))
    print(f"baseline OOS {len(b)} cyc | matched {len(m)} | {str(b.open_time.min())[:16]} -> {str(b.open_time.max())[:16]}")
    if len(m) < len(b):
        miss = sorted(set(b.open_time) - set(c.open_time))
        print(f"  WARN candidate missing {len(miss)} cycles: {[str(t)[5:16] for t in miss[:8]]}")
    fails = 0
    for col in EXACT_CAT:
        if col + "_b" in m:
            bad = m[m[col + "_b"].astype(str) != m[col + "_c"].astype(str)]
            print(f"  {col:18} {'OK' if bad.empty else f'FAIL {len(bad)} (first {str(bad.open_time.iloc[0])[:16]})'}")
            fails += not bad.empty
    for col in PICK_COLS:
        if col + "_b" in m:
            bad = [t for t, x, y in zip(m.open_time, m[col + "_b"], m[col + "_c"]) if _pset(x) != _pset(y)]
            print(f"  {col:18} {'OK' if not bad else f'FAIL {len(bad)} (first {str(bad[0])[:16]})'}")
            fails += bool(bad)
    for col, tol in NUM_TOL.items():
        if col + "_b" not in m:
            continue
        bb = m[col + "_b"].astype(float); cc = m[col + "_c"].astype(float)
        mask = bb.notna() & cc.notna()        # alpha cols: baseline NaN window (06-05..06-09) auto-skipped here
        d = (bb - cc).abs()[mask]
        nbad = int((d > tol).sum()); worst = float(d.max()) if len(d) else 0.0
        first = str(m.open_time[mask][d > tol].iloc[0])[:16] if nbad else "-"
        print(f"  {col:18} max|Δ| {worst:.4g} (tol {tol}) {'OK' if nbad == 0 else f'FAIL {nbad} (first {first})'}")
        fails += nbad > 0
    # cumulative over the MATCHED window only (apples-to-apples; candidate may extend past the baseline)
    cb = (np.prod(1 + m["pnl_bps_b"].dropna().values / 1e4) - 1) * 100
    cc = (np.prod(1 + m["pnl_bps_c"].dropna().values / 1e4) - 1) * 100
    print(f"  cumulative over matched window: baseline {cb:+.3f}% | candidate {cc:+.3f}% | Δ {cc - cb:+.3f}%")
    ok = fails == 0 and abs(cc - cb) < 0.1
    print("VERDICT:", "PASS" if ok else "DIVERGENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
