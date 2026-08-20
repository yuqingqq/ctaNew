"""EXP-M6 — settlement truth (the foundation gate).

Does the recorded Chainlink TWAP stream plus the stated rule reproduce the
winners Polymarket actually paid? Nothing downstream is valid until it does:
p̂'s target is X_T vs X_0, and if that target is wrong every calibration number
is measuring the wrong thing.

Convention grid (pre-registered, EXPERIMENT_PLAN §1 / M6 review):
    X_T ∈ {S60(T), S30(T), mean S60 over [t0,T]}
    X_0 ∈ {S60(t0), S30(t0)}
    boundary reader: last sample at or before the boundary
    tie: X_T ≥ X_0 → Up
Scored by exact winner reproduction, notional-blind (each window one vote).

KNOWLEDGE TIME. Payload timestamps are ~1.7 s ahead of when we could know them
(PM publish delay + transport). Settlement reconstruction is a POST-HOC audit of
what the venue did, so it legitimately reads by event time — but that is exactly
the read a live model may NOT make. Both are computed here and reported side by
side, because the gap between them is the size of the look-ahead a careless
backtest would bank.

Run: python3 -u -m live.pm_research.exp_m6_settlement
"""
from __future__ import annotations

import glob
import gzip
import json
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PM = REPO / "data/pm_5min"
COINS = {"btc": "btc/usd", "eth": "eth/usd", "sol": "sol/usd", "xrp": "xrp/usd",
         "doge": "doge/usd", "bnb": "bnb/usd", "hype": "hype/usd"}


def load_streams():
    """symbol -> (t_event_ms[], t_known_ms[], value[]) sorted by event time."""
    rows = defaultdict(list)
    for topic, w in (("crypto_prices_twap_sixty", 60), ("crypto_prices_twap_thirty", 30)):
        for f in sorted(glob.glob(str(PM / "prices" / topic / "*.csv*"))):
            op = gzip.open if f.endswith(".gz") else open
            with op(f, "rt") as fh:
                for ln in fh:
                    p = ln.split("\t", 1)
                    if len(p) < 2:
                        continue
                    try:
                        m = json.loads(p[1])
                        pl = m.get("payload") or {}
                        sym, ts = pl.get("symbol"), pl.get("timestamp")
                        if not sym or not ts:
                            continue
                        rows[(sym, w)].append((int(ts), int(p[0]) // 10**6,
                                               float(pl.get("full_accuracy_value", pl.get("value", 0)))))
                    except Exception:
                        pass
    out = {}
    for k, v in rows.items():
        v.sort()
        # dedupe by event time, keep earliest observation (M11: t_known discipline)
        te, tk, val = [], [], []
        for a, b, c in v:
            if te and te[-1] == a:
                continue
            te.append(a); tk.append(b); val.append(c)
        out[k] = (te, tk, val)
    return out


def read_at(series, boundary_ms, by_known=False):
    """Last sample at or before the boundary. by_known=True uses OBSERVATION
    time (what a live model could have used); False uses event time."""
    te, tk, val = series
    axis = tk if by_known else te
    i = bisect_right(axis, boundary_ms) - 1
    return (val[i], axis[i]) if i >= 0 else (None, None)


def mean_over(series, a_ms, b_ms, by_known=False):
    te, tk, val = series
    axis = tk if by_known else te
    lo, hi = bisect_right(axis, a_ms) - 1, bisect_right(axis, b_ms)
    lo = max(lo, 0)
    seg = val[lo:hi]
    return (sum(seg) / len(seg)) if seg else None


def main():
    streams = load_streams()
    print(f"[m6] TWAP series loaded: {len(streams)} (symbol, window) pairs")

    markets = {}
    for ln in open(PM / "markets.jsonl"):
        try:
            m = json.loads(ln)
            markets[m["slug"]] = m
        except Exception:
            pass
    res = {}
    for ln in open(PM / "resolutions.jsonl"):
        try:
            r = json.loads(ln)
            if r.get("closed") is True and r.get("winners"):
                res[r["slug"]] = r["winners"]
        except Exception:
            pass
    print(f"[m6] markets={len(markets)}  resolved={len(res)}")

    conventions = [
        ("S60(T) vs S60(t0)", 60, "point", 60),
        ("S30(T) vs S30(t0)", 30, "point", 30),
        ("S60(T) vs S30(t0)", 60, "point", 30),
        ("meanS60[t0,T] vs S60(t0)", 60, "mean", 60),
    ]
    tally = {c[0]: {"n": 0, "hit": 0, "margins": []} for c in conventions}
    tally_known = {c[0]: {"n": 0, "hit": 0} for c in conventions}
    skipped = 0

    for slug, winners in sorted(res.items()):
        m = markets.get(slug)
        if not m:
            continue
        sym = COINS.get(m["coin"])
        t0, T = m["window_start"] * 1000, m["window_end"] * 1000
        up_won = bool(winners.get("Up"))
        for name, wT, mode, w0 in conventions:
            sT, s0 = streams.get((sym, wT)), streams.get((sym, w0))
            if not sT or not s0:
                continue
            for by_known, store in ((False, tally), (True, tally_known)):
                xT = (mean_over(sT, t0, T, by_known)[0] if mode == "mean" and False
                      else (mean_over(sT, t0, T, by_known) if mode == "mean"
                            else read_at(sT, T, by_known)[0]))
                x0 = read_at(s0, t0, by_known)[0]
                if xT is None or x0 is None:
                    if not by_known:
                        skipped += 1
                    continue
                pred_up = xT >= x0
                store[name]["n"] += 1
                store[name]["hit"] += int(pred_up == up_won)
                if not by_known:
                    store[name]["margins"].append(
                        (abs(xT - x0) / x0 * 1e4, pred_up == up_won))

    print(f"\n=== E-M6 convention grid (event-time read = post-hoc audit) ===")
    print(f"{'convention':<28} {'n':>5} {'agree':>8} {'agree|>0.5bp':>13}")
    best = None
    for name, _, _, _ in conventions:
        t = tally[name]
        if not t["n"]:
            continue
        acc = t["hit"] / t["n"]
        big = [ok for mg, ok in t["margins"] if mg > 0.5]
        accb = sum(big) / len(big) if big else float("nan")
        print(f"{name:<28} {t['n']:>5} {acc:>7.1%} {accb:>12.1%}")
        if best is None or acc > best[1]:
            best = (name, acc, t["n"])

    print(f"\n=== same grid read at KNOWLEDGE time (what a live model could use) ===")
    for name, _, _, _ in conventions:
        t = tally_known[name]
        if t["n"]:
            print(f"{name:<28} {t['n']:>5} {t['hit']/t['n']:>7.1%}")

    if best:
        name, acc, n = best
        print(f"\nBEST: {name}  {acc:.2%} on {n} windows")
        print("GATE: CONFIRM needs ≥99.0% pooled and ≥99.5% at |margin|>0.5bp.")
        print("VERDICT:", "CONFIRM" if acc >= 0.99 else
              "REFUTE — the endgame model must be re-derived" if n >= 400 else
              "UNDERPOWERED (<400 windows with full coverage)")
    print(f"[m6] window-convention pairs skipped for missing coverage: {skipped}")


if __name__ == "__main__":
    main()
